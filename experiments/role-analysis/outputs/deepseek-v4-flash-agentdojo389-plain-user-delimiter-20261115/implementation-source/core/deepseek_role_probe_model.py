"""Frozen, all-CUDA DeepSeek V4 Flash loader and pre-output head capture.

The official checkpoint keeps routed experts in packed FP4 and dense matrices in
FP8. This module does not quantize weights, dequantize the model, or offload model
weights. A recent Transformers release with ``deepseek_v4`` and packed-FP4
``FineGrainedFP8Config`` support is required.

V4's output projection has two stages. The input of ``o_a_proj`` is the head
output *after inverse RoPE*, before its first grouped output projection. The
input of ``o_b_proj`` already mixes heads and is not a head-feature capture site.
Global head IDs are ``layer * 64 + head``; selected-head matrices preserve the
caller's order and have 512 columns per head (8192 for sixteen heads).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

LAYERS = 43
HEADS = 64
HEAD_DIM = 512
HIDDEN_SIZE = 4096
FEATURE_POSITION = "attention-head-output-before-o_a_proj"
ATTENTION_IMPLEMENTATION = "deepseek_role_probe_tiled_eager"
QUERY_TILE_SIZE = 256
EXPERTS_IMPLEMENTATION = "batched_mm"
DEFAULT_MODEL = "/scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def flash_device_map(device_count=4, num_layers=LAYERS):
    """Contiguous decoder blocks, with every model module assigned to CUDA."""
    require(device_count == 4, "This extraction protocol requires four H100s")
    result = {"model.embed_tokens": 0, "model.rotary_emb": 0,
              "model.norm": device_count - 1, "model.hc_head": device_count - 1,
              "lm_head": device_count - 1}
    result.update({f"model.layers.{i}": i * device_count // num_layers for i in range(num_layers)})
    return result


def memory_report(device_count=4, synchronize=True):
    """Return per-device allocator peaks and device-wide free/total bytes."""
    import torch
    result = []
    for i in range(device_count):
        if synchronize:
            torch.cuda.synchronize(i)
        free, total = torch.cuda.mem_get_info(i)
        result.append({"device": f"cuda:{i}", "name": torch.cuda.get_device_name(i),
                       "allocated_bytes": torch.cuda.memory_allocated(i),
                       "reserved_bytes": torch.cuda.memory_reserved(i),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(i),
                       "device_used_bytes": total - free, "device_total_bytes": total})
    return result


def kernel_report():
    """Record the exact Hub revisions of kernels loaded by this process."""
    from kernels import get_loaded_kernels
    result = []
    for loaded in get_loaded_kernels():
        origin = loaded.repo_info
        result.append({"name": str(loaded.metadata.name), "id": str(loaded.metadata.id),
                       "version": str(loaded.metadata.version),
                       "repository": None if origin is None else origin.repo_id,
                       "revision": None if origin is None else str(origin.revision),
                       "module_path": getattr(loaded.module, "__file__", None)})
    return result


def _quantization_dict(config):
    value = getattr(config, "quantization_config", None)
    return value.to_dict() if hasattr(value, "to_dict") else dict(value or {})


def tiled_eager_attention_forward(module, query, key, value, attention_mask,
                                  scaling, dropout=0.0, **kwargs):
    """Run HF's unchanged V4 eager formula on bounded groups of query rows.

    Each row still attends to every permitted key, including compressed keys
    and the learned sink. In particular, HF's softmax dtype (which can be FP32
    because the sinks are FP32) is preserved by calling its original function.
    Attention weights are not retained: this extractor observes head vectors.
    """
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Attention, eager_attention_forward)
    require(isinstance(module, DeepseekV4Attention), "This attention implementation is restricted to DeepSeek V4")
    require(not module.training and float(dropout) == 0.0, "Tiled attention is for frozen eval extraction only")
    require(not kwargs.get("output_attentions", False), "Tiled extraction does not retain attention matrices")
    tile_size = int(getattr(module, "_deepseek_role_probe_query_tile_size", QUERY_TILE_SIZE))
    require(tile_size > 0, "Query tile size must be positive")
    query_length = query.shape[-2]
    require(query_length > 0, "Attention received no query rows")
    if attention_mask is not None:
        require(attention_mask.shape[-2] in (1, query_length), "Attention mask has an unexpected query axis")
    output = None
    for start in range(0, query_length, tile_size):
        end = min(start + tile_size, query_length)
        mask = attention_mask
        if mask is not None and mask.shape[-2] != 1:
            mask = mask[..., start:end, :]
        piece, weights = eager_attention_forward(
            module, query[..., start:end, :], key, value, mask,
            scaling=scaling, dropout=dropout, **kwargs)
        # The eager output layout is [B, query_rows, H, D]. Release each block's
        # weights before computing the next one, keeping the temporary O(tile*K).
        del weights
        if output is None:
            output = piece.new_empty((piece.shape[0], query_length, *piece.shape[2:]))
        output[:, start:end] = piece
        del piece
    return output, None


def install_tiled_attention(model, query_tile_size=QUERY_TILE_SIZE):
    """Register a dedicated V4 attention interface with HF's eager mask builder."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    require(int(query_tile_size) > 0, "Query tile size must be positive")
    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, tiled_eager_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    for layer in model.model.layers:
        layer.self_attn._deepseek_role_probe_query_tile_size = int(query_tile_size)
        layer.self_attn.config._attn_implementation = ATTENTION_IMPLEMENTATION
    model.config._attn_implementation = ATTENTION_IMPLEMENTATION
    model.model.config._attn_implementation = ATTENTION_IMPLEMENTATION
    return model


def audit_model(model, expected_devices=4, check_geometry=True):
    """Fail closed if CPU weights, trainable weights, or expanded FP4 appear."""
    import torch
    from transformers.integrations.finegrained_fp8 import FP8Experts

    if check_geometry:
        geometry = (model.config.num_hidden_layers, model.config.num_attention_heads,
                    model.config.head_dim, model.config.hidden_size)
        require(geometry == (LAYERS, HEADS, HEAD_DIM, HIDDEN_SIZE), f"Unexpected V4 Flash geometry: {geometry}")
    require(not model.training, "The backbone must be in eval mode")
    bad_devices = [(n, str(p.device)) for n, p in model.named_parameters() if p.device.type != "cuda"]
    require(not bad_devices, f"CPU/disk/meta model weight offload is forbidden: {bad_devices[:8]}")
    bad_buffers = [(n, str(b.device)) for n, b in model.named_buffers() if b.device.type != "cuda"]
    require(not bad_buffers, f"CPU/meta model buffers are forbidden: {bad_buffers[:8]}")
    require(not any(p.requires_grad for p in model.parameters()), "The backbone is not frozen")
    used = {p.device.index for p in model.parameters()}
    require(used == set(range(expected_devices)), f"Unexpected CUDA placement: {sorted(used)}")
    for name, device in getattr(model, "hf_device_map", {}).items():
        require(str(device) not in ("cpu", "disk", "meta"), f"Forbidden placement for {name}: {device}")
    quantizer = getattr(model, "hf_quantizer", None)
    require(quantizer is not None, "Missing native FP8/FP4 quantizer")
    q = _quantization_dict(model.config)
    require(q.get("quant_method") == "fp8", f"Unexpected quantization method: {q}")
    require(not getattr(quantizer.quantization_config, "dequantize", True), "Full-model dequantization is forbidden")
    expert_dtype = q.get("expert_dtype", getattr(model.config, "expert_dtype", None))
    require(expert_dtype == "fp4" and q.get("scale_fmt") == "ue8m0",
            f"Native packed FP4 and UE8M0 scales were not preserved: {q}")
    require(tuple(q.get("weight_block_size", ())) == (128, 128), "Native FP8 block size changed")

    experts = []
    for i, layer in enumerate(model.model.layers):
        expert = layer.mlp.experts
        require(isinstance(expert, FP8Experts), f"Layer {i} is not the native mixed-precision experts backend")
        for name in ("gate_up_proj", "down_proj"):
            weight = getattr(expert, name)
            scale = getattr(expert, name + "_scale_inv")
            require(weight.dtype == torch.int8, f"Layer {i} {name} no longer stores packed FP4: {weight.dtype}")
            require(scale.dtype == torch.float8_e8m0fnu, f"Layer {i} {name} has non-native scale dtype: {scale.dtype}")
            require(weight.ndim == 3 and scale.ndim == 3, "Unexpected expert weight/scale rank")
            require(scale.shape[:-1] == weight.shape[:-1] and scale.shape[-1] * 16 == weight.shape[-1],
                    f"Layer {i} {name} is not two FP4 values per byte with one scale per 32 values")
        experts.append(type(expert).__name__)
        require(hasattr(layer.self_attn, "o_a_proj"), f"Layer {i} lacks the unprojected-head capture site")

    dtype_counts, dtype_bytes, by_device = Counter(), Counter(), defaultdict(Counter)
    for _, parameter in model.named_parameters():
        dtype, device = str(parameter.dtype), str(parameter.device)
        size = parameter.numel() * parameter.element_size()
        dtype_counts[dtype] += parameter.numel()
        dtype_bytes[dtype] += size
        by_device[device][dtype] += size
    require(dtype_bytes["torch.int8"] > 0 and dtype_bytes["torch.float8_e4m3fn"] > 0,
            "Expected both packed FP4 expert weights and FP8 dense weights")
    return {"parameters_frozen": True, "all_parameters_on_cuda": True,
            "native_packed_fp4_fp8": True, "quantization_config": q,
            "expert_backends": sorted(set(experts)),
            "experts_implementation": model.config._experts_implementation,
            "attention_implementation": model.config._attn_implementation,
            "dtype_numel": dict(dtype_counts),
            "dtype_bytes": dict(dtype_bytes), "parameter_bytes_by_device_dtype": dict(by_device),
            "parameter_bytes_total": sum(dtype_bytes.values())}


def load_model(model_dir=DEFAULT_MODEL, run_dir=None, devices="0,1,2,3", cache_dir=None,
               query_tile_size=QUERY_TILE_SIZE):
    """Load the official native mixed checkpoint across four H100s, frozen."""
    if isinstance(devices, (list, tuple)):
        devices = ",".join(map(str, devices))
    requested = str(devices).split(",")
    require(len(requested) == 4 and len(set(requested)) == 4, "Exactly four distinct H100 devices are required")
    # Set visibility before the lazy Torch import. Callers must do likewise if
    # they initialize CUDA before calling this function.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if cache_dir:
        os.environ["HF_HOME"] = str(Path(cache_dir) / "huggingface")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, FineGrainedFP8Config
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    require(torch.cuda.device_count() == 4, "CUDA visibility must expose exactly four GPUs")
    require(all("H100" in torch.cuda.get_device_name(i) for i in range(4)), "This protocol requires H100 GPUs")
    require(all(torch.cuda.get_device_capability(i) == (9, 0) for i in range(4)), "Expected SM90 H100 hardware")
    from transformers.integrations.finegrained_fp8 import FP8Experts
    require("fp4" in inspect.getsource(inspect.unwrap(FP8Experts.__init__)),
            "Install a recent Transformers build with native packed-FP4 experts")

    root = Path(model_dir).resolve()
    source = json.loads((root / "config.json").read_text())
    source_q = dict(source.get("quantization_config", {}))
    require(source.get("model_type") == "deepseek_v4" and source.get("expert_dtype") == "fp4",
            "Expected the official DeepSeek V4 Flash native FP4 checkpoint")
    require(source_q.get("quant_method") == "fp8" and source_q.get("scale_fmt") == "ue8m0",
            "The checkpoint does not have the expected native FP8 quantization configuration")
    source_q["expert_dtype"] = "fp4"
    source_q["dequantize"] = False
    config = AutoConfig.from_pretrained(str(root), local_files_only=True)
    config.quantization_config = FineGrainedFP8Config(**source_q).to_dict()
    require((config.num_hidden_layers, config.num_attention_heads, config.head_dim, config.hidden_size)
            == (LAYERS, HEADS, HEAD_DIM, HIDDEN_SIZE), "Unexpected V4 Flash architecture")

    original_validate = FineGrainedFP8HfQuantizer.validate_environment

    def strict_validate(quantizer, *args, **kwargs):
        original_validate(quantizer, *args, **kwargs)
        require(not quantizer.quantization_config.dequantize, "Refusing silent full-model BF16 dequantization")
        expert_dtype = getattr(quantizer.quantization_config, "expert_dtype", getattr(config, "expert_dtype", None))
        require(expert_dtype == "fp4", "Refusing expanded FP8/BF16 experts")

    started = time.monotonic()
    print(json.dumps({"stage": "loading_deepseek_v4", "model_dir": str(root),
                      "device_map": flash_device_map()}), flush=True)
    FineGrainedFP8HfQuantizer.validate_environment = strict_validate
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(root), config=config, local_files_only=True, dtype=torch.bfloat16,
            device_map=flash_device_map(), attn_implementation="eager", use_kernels=False,
            experts_implementation=EXPERTS_IMPLEMENTATION,
        ).eval().requires_grad_(False)
    finally:
        FineGrainedFP8HfQuantizer.validate_environment = original_validate
    # Main's quantizer does this too; keep the guard explicit because cached
    # DeepGEMM CUDA-function handles are unsafe across GPUs in one process.
    from transformers.integrations.finegrained_fp8 import _disable_deepgemm_on_multi_device
    _disable_deepgemm_on_multi_device(model)
    install_tiled_attention(model, query_tile_size)
    # Resolve the official packed-FP4/FP8 fallback before writing provenance.
    # Importing it does not run the model or expand its packed weights.
    from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
    load_finegrained_fp8_kernel()
    model.config.use_cache = False
    model.model.config.use_cache = False
    audit = audit_model(model)
    package = importlib.metadata.distribution("transformers")
    backend = {**audit, "model_path": str(root), "device_map": model.hf_device_map,
               "official_checkpoint_quantization_config": source.get("quantization_config"),
               "official_checkpoint_expert_dtype": source.get("expert_dtype"),
               "config_sha256": _sha(root / "config.json"),
               "index_sha256": _sha(root / "model.safetensors.index.json"),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "triton": importlib.metadata.version("triton"),
               "kernels": importlib.metadata.version("kernels"),
               "loaded_kernels": kernel_report(),
               "transformers": transformers.__version__,
               "transformers_install_source": package.read_text("direct_url.json"),
               "load_seconds": time.monotonic() - started,
               "capture": FEATURE_POSITION, "layers": LAYERS, "heads_per_layer": HEADS,
               "capture_after_inverse_rope": True,
               "attention_query_tile_size": int(query_tile_size),
               "attention_formula": "unmodified HF V4 eager function per query tile; original mask, sinks, scaling and softmax dtype",
               "backend_source_sha256": _sha(__file__),
               "head_dim": HEAD_DIM, "use_cache": False,
               "forward_target": "model.model (no language-model head)",
               "mtp": "Transformers ignores the checkpoint's optional speculative-decoding module",
               "gpu_memory": memory_report()}
    model._deepseek_role_probe_backend = backend
    if run_dir is not None:
        _atomic_json(Path(run_dir) / "model_backend.json", backend)
    print(json.dumps({"stage": "deepseek_v4_loaded", "load_seconds": backend["load_seconds"],
                      "parameter_bytes": backend["parameter_bytes_total"],
                      "gpu_memory": backend["gpu_memory"]}), flush=True)
    return model


def _forward(model, input_ids, attention_mask=None):
    """Return backbone hidden states only; never compute vocabulary logits."""
    import torch
    device = model.model.embed_tokens.weight.device
    tensor = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    require(tensor.ndim == 2, "Expected input IDs with shape [tokens] or [batch, tokens]")
    mask = torch.ones_like(tensor) if attention_mask is None else torch.as_tensor(
        attention_mask, dtype=torch.long, device=device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    require(mask.shape == tensor.shape, "Attention mask and input IDs differ in shape")
    with torch.inference_mode():
        return model.model(input_ids=tensor, attention_mask=mask, use_cache=False,
                           output_attentions=False, output_hidden_states=False,
                           return_dict=True).last_hidden_state


class HeadCapture:
    """Sample body positions, optionally selecting ordered global head IDs.

    Positions are token indices for a one-item batch, or explicit ``(batch,
    token)`` coordinates. ``receive(layer, x)`` receives a detached CUDA tensor
    ``[sampled_positions, selected_layer_heads_or_64, 512]`` while its layer is
    live. The scanner must consume it immediately without retaining it. All-head
    scans require a callback, avoiding a complete all-layer activation bank.
    """
    def __init__(self, positions, selected=None, receive=None, *, num_layers=LAYERS,
                 num_heads=HEADS, head_dim=HEAD_DIM):
        self.positions = np.asarray(positions, dtype=np.int64)
        require(self.positions.ndim == 1 or (self.positions.ndim == 2 and self.positions.shape[1] == 2),
                "Positions must be token indices or [batch, token] coordinates")
        require(len(self.positions) > 0 and self.positions.min() >= 0, "Positions must be nonempty and nonnegative")
        self.num_layers, self.num_heads, self.head_dim = num_layers, num_heads, head_dim
        if selected is not None:
            selected = np.asarray(selected, dtype=np.int64)
            if selected.ndim == 2 and selected.shape[1] == 2:
                require((selected[:, 1] >= 0).all() and (selected[:, 1] < num_heads).all(), "Invalid local head index")
                selected = selected[:, 0] * num_heads + selected[:, 1]
            require(selected.ndim == 1 and len(selected) > 0, "Selected heads must be a nonempty vector")
            require(selected.min() >= 0 and selected.max() < num_layers * num_heads, "Selected global head is out of range")
            require(len(np.unique(selected)) == len(selected), "Selected heads must be unique")
        self.selected = selected
        self.layers = list(range(num_layers)) if selected is None else sorted(set(map(int, selected // num_heads)))
        require(selected is not None or receive is not None, "All-head capture requires a streaming callback")
        self.receive = receive
        self.blocks = {}
        self.seen = set()
        self._active = False

    def hook(self, layer):
        def capture(module, args):
            import torch
            require(layer not in self.seen, f"Head hook invoked twice at layer {layer}")
            self.seen.add(layer)
            value = args[0].detach()
            require(value.ndim == 4 and value.shape[-2] * value.shape[-1] == self.num_heads * self.head_dim,
                    f"Unexpected pre-o_a_proj geometry at layer {layer}: {tuple(value.shape)}")
            batch, length = value.shape[:2]
            value = value.reshape(batch, length, self.num_heads, self.head_dim)
            if self.positions.ndim == 1:
                require(batch == 1, "Batched extraction needs explicit [batch, token] coordinates")
                require(self.positions.max() < length, "Body position is outside the input")
                x = value[0].index_select(0, torch.as_tensor(self.positions, device=value.device))
            else:
                require(self.positions[:, 0].max() < batch and self.positions[:, 1].max() < length,
                        "Body coordinate is outside the input batch")
                coordinates = torch.as_tensor(self.positions, device=value.device)
                x = value[coordinates[:, 0], coordinates[:, 1]]
            if self.selected is not None:
                indices = np.flatnonzero(self.selected // self.num_heads == layer)
                heads = self.selected[indices] % self.num_heads
                x = x.index_select(1, torch.as_tensor(heads, device=value.device))
            require(bool(torch.isfinite(x).all()), f"Nonfinite attention head activation at layer {layer}")
            if self.receive is not None:
                self.receive(layer, x)
            else:
                require(bool((x.abs() <= np.finfo(np.float16).max).all()), "Activation overflows the FP16 feature cache")
                self.blocks[layer] = x.to(device="cpu", dtype=torch.float16).numpy()
            # None: observational pre-hooks must never replace model activations.
            return None
        return capture

    @contextmanager
    def install(self, model):
        require(not self._active, "HeadCapture is already installed")
        self.blocks.clear()
        self.seen.clear()
        self._active = True
        handles = []
        try:
            for layer in self.layers:
                site = model.model.layers[layer].self_attn.o_a_proj
                handles.append(site.register_forward_pre_hook(self.hook(layer)))
            yield self
            require(self.seen == set(self.layers), "A required pre-o_a_proj hook did not execute")
        finally:
            for handle in handles:
                handle.remove()
            self._active = False

    def matrix(self):
        require(self.selected is not None and self.receive is None,
                "A selected-head matrix requires capture without a callback")
        require(self.seen == set(self.layers) and set(self.blocks) == set(self.layers), "Selected-head capture is incomplete")
        result = np.empty((len(self.positions), len(self.selected), self.head_dim), dtype=np.float16)
        for layer, block in self.blocks.items():
            result[:, np.flatnonzero(self.selected // self.num_heads == layer)] = block
        return result.reshape(len(self.positions), -1)


def extract_selected(model, input_ids, positions, selected, attention_mask=None):
    """Convenience wrapper returning an ordered FP16 selected-head matrix."""
    capture = HeadCapture(positions, selected=selected)
    with capture.install(model):
        _forward(model, input_ids, attention_mask)
    return capture.matrix()


def scan_heads(model, input_ids, positions, receive, attention_mask=None):
    """Invoke ``receive(layer, head_vectors)`` once per layer, on its GPU."""
    capture = HeadCapture(positions, receive=receive)
    with capture.install(model):
        _forward(model, input_ids, attention_mask)
    return {"layers": len(capture.seen), "positions": len(capture.positions),
            "heads_per_layer": HEADS, "head_dim": HEAD_DIM}


def self_test():
    """Small CPU-only hook contract test; never loads production weights."""
    import torch
    from types import SimpleNamespace

    class Projection(torch.nn.Module):
        def forward(self, value):
            return value.square()

    sites = [Projection() for _ in range(3)]
    fake = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(o_a_proj=site)) for site in sites]))
    values = [torch.arange(2 * 5 * 2 * 6, dtype=torch.float32).reshape(2, 5, 2, 6) + i for i in range(3)]
    # Non-layer-sorted head ordering, and multiple sampled batch rows.
    chosen = [9, 2, 4]
    positions = [(1, 3), (0, 1), (1, 0)]
    capture = HeadCapture(positions, selected=chosen, num_layers=3, num_heads=4, head_dim=3)
    before = [site(value) for site, value in zip(sites, values)]
    with capture.install(fake):
        after = [site(value) for site, value in zip(sites, values)]
    require(all(torch.equal(a, b) for a, b in zip(before, after)), "Observational hooks changed outputs")
    expected = np.stack([np.stack([values[global_head // 4].reshape(2, 5, 4, 3)[b, t, global_head % 4].numpy()
                                   for global_head in chosen]) for b, t in positions]).reshape(3, 9).astype(np.float16)
    require(np.array_equal(capture.matrix(), expected), "Head order or sampled-position order changed")
    require(all(not site._forward_pre_hooks for site in sites), "Hooks leaked after extraction")
    observed = {}
    scan = HeadCapture([1, 3], receive=lambda layer, x: observed.update({layer: x.clone()}),
                       num_layers=3, num_heads=4, head_dim=3)
    with scan.install(fake):
        for site, value in zip(sites, values):
            site(value[:1])
    require(all(x.shape == (2, 4, 3) for x in observed.values()) and len(observed) == 3,
            "All-head scanner callback contract changed")
    # Incomplete forwards must fail and still remove every installed hook.
    incomplete = HeadCapture([0], selected=[0, 8], num_layers=3, num_heads=4, head_dim=3)
    try:
        with incomplete.install(fake):
            sites[0](values[0][:1])
    except ValueError as error:
        require("did not execute" in str(error), "Unexpected incomplete-capture error")
    else:
        raise AssertionError("Incomplete capture was accepted")
    require(all(not site._forward_pre_hooks for site in sites), "Hooks leaked after a failed extraction")
    return {"status": "passed", "cpu_only": True, "selected_order": True,
            "batch_coordinates": True, "all_head_callback": True,
            "hidden_states_bitexact": True, "failed_forward_hook_cleanup": True}


def tiling_reference_test(model, device="cpu"):
    """Compare every query tile to HF eager with identical sinks and masks."""
    import torch
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import eager_attention_forward
    attention = model.model.layers[0].self_attn
    previous = getattr(attention, "_deepseek_role_probe_query_tile_size", None)
    attention._deepseek_role_probe_query_tile_size = 7
    results = []
    try:
        for dtype in (torch.float32, torch.bfloat16):
            query = torch.randn(2, model.config.num_attention_heads, 37, model.config.head_dim,
                                device=device, dtype=dtype) * .1
            kv = torch.randn(2, 1, 43, model.config.head_dim, device=device, dtype=dtype) * .1
            rows = torch.arange(37, device=device)[:, None]
            columns = torch.arange(43, device=device)[None, :]
            mask = torch.zeros(37, 43, device=device, dtype=dtype)
            mask.masked_fill_(columns > rows + 2, torch.finfo(dtype).min)
            for kind, current_mask in (("full-query-mask", mask[None, None]),
                                        ("broadcast-query-mask", mask[:1][None, None]),
                                        ("no-mask", None)):
                expected, _ = eager_attention_forward(
                    attention, query, kv, kv, current_mask, scaling=attention.scaling)
                actual, weights = tiled_eager_attention_forward(
                    attention, query, kv, kv, current_mask, scaling=attention.scaling)
                require(weights is None and actual.dtype == expected.dtype, "Tiling changed attention output dtype")
                tolerance = {"rtol": 1e-2, "atol": 1e-4} if dtype == torch.bfloat16 else {"rtol": 1e-5, "atol": 1e-6}
                torch.testing.assert_close(actual, expected, **tolerance)
                results.append({"dtype": str(dtype), "mask": kind, "query_rows": 37,
                                "query_tile_size": 7, "bitexact": bool(torch.equal(actual, expected)),
                                "maximum_absolute_difference": float((actual - expected).abs().max())})
    finally:
        if previous is None:
            del attention._deepseek_role_probe_query_tile_size
        else:
            attention._deepseek_role_probe_query_tile_size = previous
    return results


def tiny_model_test(device="cpu", native_roundtrip=False):
    """Exercise real HF V4 attention on a tiny random three-layer model.

    The optional round trip runs only on one CUDA GPU. It constructs random,
    already-packed test weights; it never quantizes the official checkpoint.
    """
    import tempfile
    import torch
    from transformers import (AutoModelForCausalLM, DeepseekV4Config,
                              DeepseekV4ForCausalLM, FineGrainedFP8Config)

    require(not native_roundtrip or str(device).startswith("cuda"), "Native kernel round trip requires one CUDA GPU")
    torch.manual_seed(17)
    config = DeepseekV4Config(
        vocab_size=256, hidden_size=256, moe_intermediate_size=256,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=1,
        head_dim=128, q_lora_rank=256, o_groups=2, o_lora_rank=256,
        n_routed_experts=1, n_shared_experts=1, num_experts_per_tok=1,
        layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
        mlp_layer_types=["hash_moe", "moe", "moe"],
        index_n_heads=4, index_head_dim=32, index_topk=2,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
        sliding_window=8, max_position_embeddings=32, qk_rope_head_dim=8,
        hc_mult=2, hc_sinkhorn_iters=4, use_cache=False,
    )
    config._attn_implementation = "eager"
    if native_roundtrip:
        config._experts_implementation = "batched_mm"
    model = DeepseekV4ForCausalLM(config).eval().requires_grad_(False).to(device)
    native_check = {}
    if native_roundtrip:
        from transformers.integrations.finegrained_fp8 import FP8Experts, replace_with_fp8_linear
        from transformers.quantizers.auto import AutoHfQuantizer
        from transformers.utils.quantization_config import QuantizationMethod

        config.expert_dtype = "fp4"  # Older stable releases read this legacy location.
        q = FineGrainedFP8Config(expert_dtype="fp4", scale_fmt="ue8m0", dequantize=False)
        # Match the quantizer's production exclusion list, including norm and
        # compressor matrices that ship without scales in the real checkpoint.
        quantizer = AutoHfQuantizer.from_config(q, pre_quantized=True)
        quantizer._process_model_before_weight_loading(model)
        for name, parameter in list(model.named_parameters()):
            if parameter.device.type != "meta":
                parameter.data = parameter.data.to(device)
                continue
            shape = tuple(parameter.shape)
            if parameter.dtype == torch.int8:
                value = torch.randint(-128, 128, shape, dtype=torch.int8, device=device)
            elif parameter.dtype == torch.float8_e8m0fnu:
                value = torch.full(shape, 2.0 ** -8, dtype=torch.float32, device=device).to(parameter.dtype)
            elif parameter.dtype == torch.float8_e4m3fn:
                value = (torch.randn(shape, device=device) * 4).to(parameter.dtype)
            else:
                value = torch.zeros(shape, dtype=parameter.dtype, device=device)
            parent_name, _, leaf = name.rpartition(".")
            model.get_submodule(parent_name)._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        model.config.quantization_config = q
        model.hf_quantizer = quantizer
        model.is_quantized = True
        model.quantization_method = QuantizationMethod.FP8
        # Batched routing avoids one Python-dispatched GEMM per active expert.
        # The weights and scales remain in the same native packed format.
        for layer in model.model.layers:
            require(isinstance(layer.mlp.experts, FP8Experts), "Tiny model experts were not replaced")
            require(layer.mlp.experts.gate_up_proj.dtype == torch.int8, "Tiny model FP4 unpacked at construction")
        audit_model(model, expected_devices=1, check_geometry=False)
    ids = list(range(2, 14))
    chosen, positions = [9, 2, 4], [0, 5, 11]
    original = _forward(model, ids).detach().cpu()
    tiling_checks = tiling_reference_test(model, device)
    install_tiled_attention(model, query_tile_size=5)
    baseline = _forward(model, ids).detach().cpu()
    torch.testing.assert_close(baseline, original, rtol=1e-5, atol=1e-5)
    capture = HeadCapture(positions, selected=chosen, num_layers=3, num_heads=4, head_dim=config.head_dim)
    with capture.install(model):
        observed = _forward(model, ids).detach().cpu()
    require(torch.equal(baseline, observed), "Real V4 head hooks changed hidden states")
    require(bool(torch.isfinite(observed).all()), "Tiny V4 produced nonfinite hidden states")
    require(capture.matrix().shape == (3, 3 * config.head_dim), "Tiny V4 selected-head output shape changed")
    if native_roundtrip:
        with tempfile.TemporaryDirectory(prefix="deepseek-v4-native-test-") as directory:
            model.save_pretrained(directory)
            restored = AutoModelForCausalLM.from_pretrained(
                directory, device_map={"": device}, local_files_only=True,
                dtype=torch.float32, attn_implementation="eager", use_kernels=False,
                experts_implementation="batched_mm",
            ).eval().requires_grad_(False)
            for layer in restored.model.layers:
                require(layer.mlp.experts.gate_up_proj.dtype == torch.int8, "FP4 unpacked during native reload")
                require(layer.mlp.experts.gate_up_proj_scale_inv.dtype == torch.float8_e8m0fnu,
                        "UE8M0 scales changed during reload")
            audit_model(restored, expected_devices=1, check_geometry=False)
            install_tiled_attention(restored, query_tile_size=5)
            require(torch.equal(baseline, _forward(restored, ids).cpu()), "Native packed checkpoint reload changed hidden states")
            native_check = {"packed_fp4_roundtrip_bitexact": True, "fp8_dense_preserved": True,
                            "experts_implementation": "batched_mm", "loaded_kernels": kernel_report()}
            del restored
    return {"status": "passed", "device": str(device), "layers": 3,
            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "attention_types": config.layer_types, "hidden_states_bitexact": True,
            "attention_tiling_reference_checks": tiling_checks,
            "whole_model_tiled_eager_maximum_absolute_difference": float((baseline - original).abs().max()),
            "selected_feature_shape": list(capture.matrix().shape), **native_check}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tiny-test", action="store_true")
    parser.add_argument("--native-roundtrip", action="store_true")
    parser.add_argument("--test-device", default="cpu")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.tiny_test:
        print(json.dumps(tiny_model_test(args.test_device, args.native_roundtrip), indent=2))
    else:
        require(args.output is not None, "A full-model load requires --output for its audit report")
        load_model(args.model, args.output)


if __name__ == "__main__":
    main()
