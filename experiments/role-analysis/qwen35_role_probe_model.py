"""Native Qwen3.5-397B-A17B (GPTQ Int4) backbone for role-probe extraction on four H100s.

The official GPTQ checkpoint quantizes only the routed MoE experts (per-expert ``gate_proj`` /
``up_proj`` / ``down_proj`` in GPTQ v1 layout: ``qweight`` int32 with eight 4-bit values per word along
the input axis, one fp16 ``scales`` row per 128 inputs, symmetric zero point 8).  Transformers stores
experts as fused dense 3D parameters, which would need ~800 GB, so this module keeps the packed words on
the GPUs and expands only the experts hit by a forward pass (``PackedExperts``).  Everything else
(attention, Gated DeltaNet, shared experts, router, norms) is the unmodified Transformers model in BF16.

Probe features: the gated attention output of the 15 full-attention layers right before ``o_proj``
(32 heads x 256 dims).  ``LAYERS`` therefore counts full-attention layers; scanned layer ``i`` is decoder
layer ``FULL_LAYERS[i]``.  Linear-attention (Gated DeltaNet) layers have no per-head softmax output.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time

import numpy as np

DECODER_LAYERS = 60
FULL_LAYERS = tuple(range(3, DECODER_LAYERS, 4))
LAYERS = len(FULL_LAYERS)  # scanned (full-attention) layers
HEADS = 32
HEAD_DIM = 256
HIDDEN_SIZE = 4096
EXPERTS = 512
INTERMEDIATE = 1024
GROUP = 128
FEATURE_POSITION = "full-attention-gated-head-output-before-o_proj"
DEFAULT_MODEL = "/scratch/wz55191/Hugginface/Qwen3.5-397B-A17B-GPTQ-Int4"
EXPERT_KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(qweight|qzeros|scales|g_idx)$")
EXPERT_CHUNK = 64  # experts expanded at once (transient ~1.6 GB BF16)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def device_of_layer(layer, device_count=4):
    return layer * device_count // DECODER_LAYERS


def memory_report(device_count=4, synchronize=True):
    import torch
    result = []
    for i in range(device_count):
        if synchronize:
            torch.cuda.synchronize(i)
        free, total = torch.cuda.mem_get_info(i)
        result.append({"device": f"cuda:{i}", "name": torch.cuda.get_device_name(i),
                       "allocated_bytes": torch.cuda.memory_allocated(i),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
                       "device_used_bytes": total - free, "device_total_bytes": total})
    return result


def unpack_nibbles(words):
    """int32 [..., R, O] with eight 4-bit values per word -> uint8 [..., R * 8, O] (input index r * 8 + j)."""
    import torch
    shape = words.shape
    octets = words.contiguous().view(torch.uint8).view(*shape, 4)            # little endian bytes of each word
    pairs = torch.stack((octets & 15, octets >> 4), dim=-1)                  # [..., R, O, 4, 2]: j = 2 * byte + nibble
    lead = len(shape) - 2
    pairs = pairs.permute(*range(lead), lead, lead + 2, lead + 3, lead + 1)  # [..., R, 4, 2, O]
    return pairs.reshape(*shape[:-2], shape[-2] * 8, shape[-1])


def expand(words, scales):
    """GPTQ v1 symmetric 4-bit -> BF16 weight laid out [..., in, out] (so y = x @ W)."""
    import torch
    levels = unpack_nibbles(words).to(torch.int8) - 8
    return levels.to(scales.dtype) * scales.repeat_interleave(GROUP, dim=-2)


def build_experts_class():
    import torch
    from torch import nn
    from transformers.activations import ACT2FN

    class PackedExperts(nn.Module):
        """Routed experts kept as packed GPTQ words; only experts hit by the batch are expanded."""

        def __init__(self, config):
            super().__init__()
            self.num_experts, self.hidden_dim, self.intermediate_dim = config.num_experts, config.hidden_size, config.moe_intermediate_size
            self.act_fn = ACT2FN[config.hidden_act]
            # gate and up share the input; index 0 = gate_proj, 1 = up_proj.  Storage is allocated on the GPU by load_model.
            for name in ("gate_up_words", "gate_up_scales", "down_words", "down_scales"):
                self.register_buffer(name, None, persistent=False)

        def shapes(self):
            import torch
            hidden, inter, n = self.hidden_dim, self.intermediate_dim, self.num_experts
            return {"gate_up_words": ((n, 2, hidden // 8, inter), torch.int32), "gate_up_scales": ((n, 2, hidden // GROUP, inter), torch.bfloat16),
                    "down_words": ((n, inter // 8, hidden), torch.int32), "down_scales": ((n, inter // GROUP, hidden), torch.bfloat16)}

        def forward(self, hidden_states, top_k_index, top_k_weights):
            tokens, k = top_k_index.shape
            expert = top_k_index.reshape(-1)
            order = torch.argsort(expert, stable=True)
            token = torch.arange(tokens, device=expert.device).repeat_interleave(k)[order]
            weight = top_k_weights.reshape(-1)[order].to(hidden_states.dtype)
            counts = torch.bincount(expert, minlength=self.num_experts)
            ends = torch.cumsum(counts, 0)
            hit = torch.nonzero(counts).flatten()
            counts, ends = counts.tolist(), ends.tolist()
            output = torch.zeros_like(hidden_states)
            for start in range(0, len(hit), EXPERT_CHUNK):
                chosen = hit[start:start + EXPERT_CHUNK]
                gate_up = expand(self.gate_up_words[chosen], self.gate_up_scales[chosen])  # [C, 2, hidden, inter]
                down = expand(self.down_words[chosen], self.down_scales[chosen])           # [C, inter, hidden]
                for local, index in enumerate(chosen.tolist()):
                    rows = token[ends[index] - counts[index]:ends[index]]
                    state = hidden_states[rows]
                    value = self.act_fn(state @ gate_up[local, 0]) * (state @ gate_up[local, 1])
                    value = (value @ down[local]) * weight[ends[index] - counts[index]:ends[index], None]
                    output.index_add_(0, rows, value)
                del gate_up, down
            return output

    return PackedExperts


def shard_header(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    return header


def load_model(model_dir=DEFAULT_MODEL, run_dir=None, devices="0,1,2,3", lm_head=True):
    """Load the text backbone across four H100s, frozen; experts stay packed."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import transformers
    from accelerate import dispatch_model, init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling

    require(torch.cuda.device_count() == 4, "CUDA visibility must expose exactly four GPUs")
    root = Path(model_dir).resolve()
    source = json.loads((root / "config.json").read_text())
    quantization = source.get("quantization_config", {})
    require(source.get("model_type") == "qwen3_5_moe" and quantization.get("quant_method") == "gptq" and quantization.get("bits") == 4
            and quantization.get("group_size") == GROUP and quantization.get("sym") is True and quantization.get("desc_act") is False,
            "Expected the official Qwen3.5 GPTQ Int4 checkpoint (4 bit, group 128, symmetric, no desc_act)")
    config = AutoConfig.from_pretrained(str(root), local_files_only=True).text_config
    require((config.num_hidden_layers, config.num_attention_heads, config.head_dim, config.hidden_size, config.num_experts,
             config.moe_intermediate_size) == (DECODER_LAYERS, HEADS, HEAD_DIM, HIDDEN_SIZE, EXPERTS, INTERMEDIATE), "Unexpected geometry")
    require(tuple(i for i, kind in enumerate(config.layer_types) if kind == "full_attention") == FULL_LAYERS, "Unexpected layer types")
    started = time.monotonic()
    print(json.dumps({"stage": "loading_qwen35", "model_dir": str(root)}), flush=True)
    with init_empty_weights(include_buffers=False):  # meta parameters; small buffers (rotary frequencies) keep real values
        model = modeling.Qwen3_5MoeForCausalLM(config)
    packed = build_experts_class()
    for layer in model.model.layers:  # the dense 3D expert parameters (meta) are replaced by packed storage
        layer.mlp.experts = packed(config)
    model.eval().requires_grad_(False)
    device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 3, "lm_head": 3}
    device_map.update({f"model.layers.{i}": device_of_layer(i) for i in range(DECODER_LAYERS)})
    # real storage for the packed experts
    for i, layer in enumerate(model.model.layers):
        experts, device = layer.mlp.experts, f"cuda:{device_of_layer(i)}"
        for name, (shape, dtype) in experts.shapes().items():
            experts._buffers[name] = torch.empty(shape, dtype=dtype, device=device)
    expected = {name for name, _ in model.named_parameters()}
    if not lm_head:
        expected.discard("lm_head.weight")
    loaded, expert_tensors, skipped = set(), Counter(), Counter()
    shards = sorted(root.glob("model.safetensors-*.safetensors"))
    require(shards, "No checkpoint shards")
    zero_word, group_index = None, None
    for number, shard in enumerate(shards):
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                match = EXPERT_KEY.match(key)
                if match:
                    layer, expert, projection, kind = int(match[1]), int(match[2]), match[3], match[4]
                    experts = model.model.layers[layer].mlp.experts
                    tensor = reader.get_tensor(key)
                    if kind == "qzeros":  # symmetric: every zero point is the same nibble
                        zero_word = int(tensor.flatten()[0]) if zero_word is None else zero_word
                        require(bool((tensor == zero_word).all()), f"Non-constant zero point in {key}")
                    elif kind == "g_idx":
                        if group_index is None or len(group_index) != len(tensor):
                            group_index = torch.arange(len(tensor), dtype=tensor.dtype) // GROUP
                        require(bool((tensor == group_index).all()), f"Unexpected group index in {key}")
                    else:
                        words = kind == "qweight"
                        if projection == "down_proj":
                            target = experts.down_words if words else experts.down_scales
                            target[expert].copy_(tensor)
                        else:
                            target = experts.gate_up_words if words else experts.gate_up_scales
                            target[expert, 0 if projection == "gate_proj" else 1].copy_(tensor)
                    expert_tensors[kind] += 1
                    continue
                if key.startswith("model.language_model."):
                    name = "model." + key[len("model.language_model."):]
                elif key == "lm_head.weight":
                    name = key
                else:
                    skipped[key.split(".")[0] + "." + key.split(".")[1]] += 1
                    continue
                if name not in expected:
                    skipped["unexpected:" + name.rsplit(".", 2)[-2]] += 1
                    continue
                module = next((prefix for prefix in sorted(device_map, key=len, reverse=True) if name.startswith(prefix + ".")), None)
                require(module is not None, f"No device for {name}")
                set_module_tensor_to_device(model, name, f"cuda:{device_map[module]}", value=reader.get_tensor(key), dtype=torch.bfloat16)
                loaded.add(name)
        print(json.dumps({"stage": "loaded_shard", "index": number + 1, "of": len(shards), "seconds": round(time.monotonic() - started, 1)}), flush=True)
    if not lm_head:
        model.lm_head = torch.nn.Identity()
        device_map.pop("lm_head")
    missing = sorted(expected - loaded)
    require(not missing, f"Missing weights: {missing[:8]} ({len(missing)})")
    per_kind = DECODER_LAYERS * EXPERTS * 3
    require(all(expert_tensors[kind] == per_kind for kind in ("qweight", "qzeros", "scales", "g_idx")), f"Incomplete experts: {dict(expert_tensors)}")
    # GPTQ v1 stores zero - 1 (0x77777777); v2 stores zero (0x88888888). Either way the symmetric zero point is 8.
    require(zero_word in (0x77777777, -0x77777778), f"Unexpected zero-point word {zero_word}")
    model.config._attn_implementation = "sdpa"
    model.model.config._attn_implementation = "sdpa"
    model = dispatch_model(model, device_map=device_map)
    model.config.use_cache = False
    audit = audit_model(model)
    backend = {**audit, "model_path": str(root), "device_map": device_map, "config_sha256": _sha(root / "config.json"),
               "checkpoint_quantization_config": quantization, "zero_point_word": zero_word, "skipped_checkpoint_keys": dict(skipped),
               "torch": torch.__version__, "transformers": transformers.__version__, "load_seconds": time.monotonic() - started,
               "capture": FEATURE_POSITION, "scanned_layers": list(FULL_LAYERS), "heads_per_layer": HEADS, "head_dim": HEAD_DIM,
               "experts": "packed GPTQ int4 words on GPU, expanded per forward for routed experts only",
               "backend_source_sha256": _sha(__file__), "gpu_memory": memory_report()}
    model._qwen35_role_probe_backend = backend
    if run_dir is not None:
        _atomic_json(Path(run_dir) / "model_backend.json", backend)
    print(json.dumps({"stage": "qwen35_loaded", "load_seconds": backend["load_seconds"], "gpu_memory": backend["gpu_memory"]}), flush=True)
    return model


def audit_model(model, expected_devices=4):
    import torch
    require(not model.training, "The backbone must be in eval mode")
    bad = [(n, str(p.device)) for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.device.type != "cuda"]
    require(not bad, f"CPU/meta weights are forbidden: {bad[:8]}")
    require(not any(p.requires_grad for p in model.parameters()), "The backbone is not frozen")
    dtype_bytes = Counter()
    for _, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        dtype_bytes[str(tensor.dtype)] += tensor.numel() * tensor.element_size()
    for i, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        require(type(experts).__name__ == "PackedExperts" and experts.gate_up_words.dtype == torch.int32, f"Layer {i} experts are not packed")
        require(hasattr(layer, "self_attn") == (i in FULL_LAYERS), f"Layer {i} has an unexpected mixer")
    return {"parameters_frozen": True, "all_on_cuda": True, "dtype_bytes": dict(dtype_bytes), "bytes_total": sum(dtype_bytes.values()),
            "attention_implementation": model.config._attn_implementation}


def _forward(model, input_ids, attention_mask=None):
    """Backbone hidden states only (no vocabulary logits)."""
    import torch
    device = model.model.embed_tokens.weight.device
    tensor = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    mask = torch.ones_like(tensor) if attention_mask is None else torch.as_tensor(attention_mask, dtype=torch.long, device=device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    require(mask.shape == tensor.shape, "Attention mask and input IDs differ in shape")
    with torch.inference_mode():
        return model.model(input_ids=tensor, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state


class HeadCapture:
    """Sample positions of the pre-``o_proj`` head outputs of the full-attention layers.

    ``positions``: token indices for a one-item batch, or ``[batch, token]`` coordinates.  With
    ``offset`` the hooked forward holds only tokens ``offset..`` (cached-prefix continuation).
    ``receive(layer, x)`` gets a CUDA tensor ``[positions, heads, 256]`` for scanned layer ``layer``.
    """
    def __init__(self, positions, selected=None, receive=None, *, offset=0):
        self.positions = np.asarray(positions, dtype=np.int64)
        require(self.positions.ndim == 1 or (self.positions.ndim == 2 and self.positions.shape[1] == 2), "Bad positions")
        require(len(self.positions) > 0 and self.positions.min() >= 0, "Positions must be nonempty and nonnegative")
        if selected is not None:
            selected = np.asarray(selected, dtype=np.int64)
            require(selected.ndim == 1 and len(selected) > 0 and selected.min() >= 0 and selected.max() < LAYERS * HEADS
                    and len(np.unique(selected)) == len(selected), "Invalid selected heads")
        self.selected, self.receive, self.offset = selected, receive, int(offset)
        self.layers = list(range(LAYERS)) if selected is None else sorted(set(map(int, selected // HEADS)))
        require(selected is not None or receive is not None, "All-head capture requires a streaming callback")
        self.blocks, self.seen, self._active = {}, set(), False

    def hook(self, layer):
        def capture(module, args):
            import torch
            require(layer not in self.seen, f"Head hook invoked twice at layer {layer}")
            self.seen.add(layer)
            value = args[0].detach()
            require(value.ndim == 3 and value.shape[-1] == HEADS * HEAD_DIM, f"Unexpected pre-o_proj geometry: {tuple(value.shape)}")
            batch, length = value.shape[:2]
            value = value.reshape(batch, length, HEADS, HEAD_DIM)
            if self.positions.ndim == 1:
                require(batch == 1 and self.positions.max() - self.offset < length, "Position outside the input")
                x = value[0].index_select(0, torch.as_tensor(self.positions - self.offset, device=value.device))
            else:
                coordinates = torch.as_tensor(self.positions, device=value.device)
                x = value[coordinates[:, 0], coordinates[:, 1] - self.offset]
            if self.selected is not None:
                heads = self.selected[np.flatnonzero(self.selected // HEADS == layer)] % HEADS
                x = x.index_select(1, torch.as_tensor(heads, device=value.device))
            require(bool(torch.isfinite(x).all()), f"Nonfinite head activation at layer {layer}")
            if self.receive is not None:
                self.receive(layer, x)
            else:
                self.blocks[layer] = x.to(device="cpu", dtype=torch.float16).numpy()
            return None
        return capture

    @contextmanager
    def install(self, model):
        require(not self._active, "HeadCapture is already installed")
        self.blocks.clear(); self.seen.clear(); self._active = True
        handles = []
        try:
            for layer in self.layers:
                site = model.model.layers[FULL_LAYERS[layer]].self_attn.o_proj
                handles.append(site.register_forward_pre_hook(self.hook(layer)))
            yield self
            require(self.seen == set(self.layers), "A required pre-o_proj hook did not execute")
        finally:
            for handle in handles:
                handle.remove()
            self._active = False

    def matrix(self):
        require(self.selected is not None and self.receive is None and set(self.blocks) == set(self.layers), "Capture is incomplete")
        result = np.empty((len(self.positions), len(self.selected), HEAD_DIM), dtype=np.float16)
        for layer, block in self.blocks.items():
            result[:, np.flatnonzero(self.selected // HEADS == layer)] = block
        return result.reshape(len(self.positions), -1)


def extract_selected(model, input_ids, positions, selected, attention_mask=None):
    capture = HeadCapture(positions, selected=selected)
    with capture.install(model):
        _forward(model, input_ids, attention_mask)
    return capture.matrix()


def scan_heads(model, input_ids, positions, receive, attention_mask=None):
    capture = HeadCapture(positions, receive=receive)
    with capture.install(model):
        _forward(model, input_ids, attention_mask)
    return {"layers": len(capture.seen), "positions": len(capture.positions), "heads_per_layer": HEADS, "head_dim": HEAD_DIM}


def self_test():
    """CPU check of the nibble unpacking against a direct loop."""
    import torch
    generator = torch.Generator().manual_seed(0)
    levels = torch.randint(0, 16, (3, 16, 5), generator=generator)
    words = torch.zeros((3, 2, 5), dtype=torch.int64)
    for j in range(8):
        words |= levels[:, j::8, :] << (4 * j)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    assert torch.equal(unpack_nibbles(words).long(), levels)
    scales = torch.rand((3, 1, 5), generator=generator)
    global GROUP
    saved, GROUP = GROUP, 16
    try:
        assert torch.allclose(expand(words, scales), (levels - 8).float() * scales)
    finally:
        GROUP = saved
    print("self_test passed")


if __name__ == "__main__":
    self_test()
