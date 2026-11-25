"""Native Llama-4-Scout-17B-16E-Instruct text backbone for role-probe extraction on four H100s.

The checkpoint is dense BF16 (about 217 GB with the vision tower).  Only the text decoder
(``language_model.*``) is loaded, as the unmodified Transformers ``Llama4ForCausalLM``, split over four
GPUs.  Probe features: the attention output of every decoder layer right before ``o_proj``
(48 layers x 40 heads x 128 dims), the same capture point as the DeepSeek / Qwen probes.

Same interface as qwen35_role_probe_model (LAYERS, HEADS, HEAD_DIM, FEATURE_POSITION, load_model,
audit_model, memory_report, _forward, HeadCapture, extract_selected, scan_heads).
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

DECODER_LAYERS = 48
FULL_LAYERS = tuple(range(DECODER_LAYERS))  # every layer has softmax attention heads
LAYERS = DECODER_LAYERS
HEADS = 40
HEAD_DIM = 128
HIDDEN_SIZE = 5120
EXPERTS = 16
FEATURE_POSITION = "attention-head-output-before-o_proj"
DEFAULT_MODEL = "/scratch/wz55191/Hugginface/Llama-4-Scout-17B-16E-Instruct"
TEXT_PREFIX = "language_model."


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


def sparse_moe_forward(self, hidden_states):
    """Llama4TextMoe.forward for top-1 routing without running the 15 unselected experts on zero inputs.

    The Transformers forward multiplies every token by its router score per expert (0 for unselected
    experts) and runs all 16 experts on all tokens; unselected experts then contribute exactly zero (no
    biases).  Only the selected expert is evaluated here: same function, 16x less work.
    """
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_scores, router_logits = self.router(hidden_states)
    out = self.shared_expert(hidden_states)
    experts, index = self.experts, router_scores.argmax(dim=1)
    for expert in index.unique().tolist():
        rows = (index == expert).nonzero().flatten()
        state = hidden_states[rows] * router_scores[rows, expert].unsqueeze(1)
        gate, up = (state @ experts.gate_up_proj[expert]).chunk(2, dim=-1)
        out.index_add_(0, rows, (up * experts.act_fn(gate)) @ experts.down_proj[expert])
    return out, router_logits


def set_moe_mode(model, sparse):
    """Switch every MoE block between the sparse top-1 forward and the stock Transformers forward."""
    import types
    for layer in model.model.layers:
        block = layer.feed_forward
        if hasattr(block, "experts"):
            require(block.top_k == 1, "The sparse forward assumes top-1 routing")
            if sparse:
                block.forward = types.MethodType(sparse_moe_forward, block)
            else:
                block.__dict__.pop("forward", None)
    model._llama4_sparse_moe = bool(sparse)


def load_model(model_dir=DEFAULT_MODEL, run_dir=None, devices="0,1,2,3", lm_head=True):
    """Load the text decoder across four H100s, frozen, BF16, no offload."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import transformers
    from accelerate import dispatch_model, init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig
    import transformers.models.llama4.modeling_llama4 as modeling

    require(torch.cuda.device_count() == 4, "CUDA visibility must expose exactly four GPUs")
    root = Path(model_dir).resolve()
    config = AutoConfig.from_pretrained(str(root), local_files_only=True).text_config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    require((config.num_hidden_layers, config.num_attention_heads, head_dim, config.hidden_size, config.num_local_experts)
            == (DECODER_LAYERS, HEADS, HEAD_DIM, HIDDEN_SIZE, EXPERTS), "Unexpected Llama 4 Scout geometry")
    started = time.monotonic()
    print(json.dumps({"stage": "loading_llama4", "model_dir": str(root)}), flush=True)
    with init_empty_weights(include_buffers=False):
        model = modeling.Llama4ForCausalLM(config)
    model.eval().requires_grad_(False)
    device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 3, "lm_head": 3}
    device_map.update({f"model.layers.{i}": device_of_layer(i) for i in range(DECODER_LAYERS)})
    expected = {name for name, _ in model.named_parameters()}
    if not lm_head:
        expected.discard("lm_head.weight")
    loaded, skipped = set(), Counter()
    shards = sorted(root.glob("model-*-of-*.safetensors"))
    require(shards, "No checkpoint shards")
    for number, shard in enumerate(shards):
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if not key.startswith(TEXT_PREFIX):
                    skipped[key.split(".")[0]] += 1
                    continue
                name = key[len(TEXT_PREFIX):]
                if name not in expected:
                    skipped["unexpected:" + name.split(".")[0]] += 1
                    continue
                module = next((prefix for prefix in sorted(device_map, key=len, reverse=True) if name.startswith(prefix + ".")), None)
                require(module is not None, f"No device for {name}")
                tensor = reader.get_tensor(key)
                require(tensor.dtype == torch.bfloat16, f"{key} is {tensor.dtype}, expected BF16")
                set_module_tensor_to_device(model, name, f"cuda:{device_map[module]}", value=tensor, dtype=torch.bfloat16)
                loaded.add(name)
        print(json.dumps({"stage": "loaded_shard", "index": number + 1, "of": len(shards), "seconds": round(time.monotonic() - started, 1)}), flush=True)
    if not lm_head:
        model.lm_head = torch.nn.Identity()
        device_map.pop("lm_head")
    missing = sorted(expected - loaded)
    require(not missing, f"Missing weights: {missing[:8]} ({len(missing)})")
    model.config._attn_implementation = "sdpa"
    model.model.config._attn_implementation = "sdpa"
    model = dispatch_model(model, device_map=device_map)
    model.config.use_cache = False
    set_moe_mode(model, os.environ.get("LLAMA4_SPARSE_MOE", "1") != "0")
    audit = audit_model(model)
    backend = {**audit, "model_path": str(root), "device_map": device_map, "config_sha256": _sha(root / "config.json"),
               "skipped_checkpoint_keys": dict(skipped), "torch": torch.__version__, "transformers": transformers.__version__,
               "load_seconds": time.monotonic() - started, "capture": FEATURE_POSITION, "scanned_layers": list(FULL_LAYERS),
               "heads_per_layer": HEADS, "head_dim": HEAD_DIM, "layer_types": list(getattr(config, "layer_types", []) or []),
               "no_rope_layers": list(getattr(config, "no_rope_layers", []) or []),
               "attention_chunk_size": getattr(config, "attention_chunk_size", None),
               "moe_forward": "sparse top-1 (sparse_moe_forward)" if model._llama4_sparse_moe else "transformers dense",
               "backend_source_sha256": _sha(__file__), "gpu_memory": memory_report()}
    model._llama4_role_probe_backend = backend
    if run_dir is not None:
        _atomic_json(Path(run_dir) / "model_backend.json", backend)
    print(json.dumps({"stage": "llama4_loaded", "load_seconds": backend["load_seconds"], "gpu_memory": backend["gpu_memory"]}), flush=True)
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
    require(len(model.model.layers) == DECODER_LAYERS and all(hasattr(layer, "self_attn") for layer in model.model.layers), "Unexpected decoder")
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
    """Sample positions of the pre-``o_proj`` head outputs of the decoder layers.

    ``positions``: token indices for a one-item batch, or ``[batch, token]`` coordinates.  With
    ``offset`` the hooked forward holds only tokens ``offset..`` (cached-prefix continuation).
    ``receive(layer, x)`` gets a CUDA tensor ``[positions, heads, 128]`` for layer ``layer``.
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
