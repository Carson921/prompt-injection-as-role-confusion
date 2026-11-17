"""Read-only, resident-model diagnostics of M=8 versus M=1 native operations.

The caller supplies an already loaded adapter. No model is loaded, no parameter
or precision option is changed, and these results never authorize benchmark ASR.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state(model):
    """Identity, storage and version counters, not a claim to hash all weights."""
    return {kind + name: [id(value), value.data_ptr(), value._version,
                         str(value.dtype), str(value.device), list(value.shape), value.requires_grad]
            for kind, iterator in (("parameter/", model.named_parameters()), ("buffer/", model.named_buffers()))
            for name, value in iterator}


def _precision():
    import torch
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}


def _bits(tensor):
    import torch
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy()


def _metrics(a, b):
    import torch
    _require(a.shape == b.shape and a.dtype == b.dtype, "Compared output shape/dtype differ")
    av, bv = a.detach().double().cpu(), b.detach().double().cpu()
    delta = av - bv
    finite = bool(torch.isfinite(av).all() and torch.isfinite(bv).all())
    return {"shape": list(a.shape), "dtype": str(a.dtype), "finite": finite,
            "bitexact": bool((_bits(a) == _bits(b)).all()),
            "different_elements": int(torch.count_nonzero(delta)), "elements": a.numel(),
            "max_abs": float(delta.abs().max()), "rms": float(delta.square().mean().sqrt()),
            "reference_rms": float(av.square().mean().sqrt())}


def _leaves(value, prefix="output"):
    import torch
    if torch.is_tensor(value):
        return {prefix: value}
    _require(isinstance(value, (tuple, list)), "Unexpected operation output tree")
    result = {}
    for index, item in enumerate(value):
        result.update(_leaves(item, prefix + f"_{index}"))
    return result


def _measure(name, value, function, *, output_axis=1):
    """All comparisons use exactly the final original input row, bit for bit."""
    import torch
    single = value[:, -1:].contiguous()
    row_bits = _bits(single)
    _require(bool((row_bits == _bits(value[:, -1:])).all()), "Single input row changed")
    many = _leaves(function(value))
    one = _leaves(function(single))
    many_repeat = _leaves(function(value))
    one_repeat = _leaves(function(single))
    _require(many.keys() == one.keys() == many_repeat.keys() == one_repeat.keys(), "Output tree differs")
    outputs = {}
    for key in many:
        def last(x):
            index = [slice(None)] * x.ndim
            index[output_axis] = slice(-1, None)
            return x[tuple(index)]
        outputs[key] = {"m8_vs_m1": _metrics(last(many[key]), one[key]),
                        "m8_repeat": _metrics(many[key], many_repeat[key]),
                        "m1_repeat": _metrics(one[key], one_repeat[key])}
    return {"name": name, "input_shape": list(value.shape), "single_input_shape": list(single.shape),
            "input_dtype": str(value.dtype), "input_device": str(value.device),
            "single_input_bitexact": True, "single_input_native_bytes_sha256": hashlib.sha256(row_bits.tobytes()).hexdigest(),
            "outputs": outputs}


def _capture_inputs(adapter, token_ids):
    """Execute only the actual first decoder layer with the model's native setup."""
    import torch
    from transformers.masking_utils import create_sliding_window_causal_mask
    backbone = adapter.model.model
    layer = backbone.layers[0]
    ids = torch.tensor([token_ids], dtype=torch.long, device=backbone.embed_tokens.weight.device)
    observed, handles = {}, []
    targets = {"attn_hc": layer.attn_hc, "ffn_hc": layer.ffn_hc, "mlp": layer.mlp}
    targets.update({"attention/" + name: getattr(layer.self_attn, name)
                    for name in ("q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj")})
    for label, module in targets.items():
        def before(module, args, label=label):
            _require(label not in observed, "First-layer capture ran more than once")
            observed[label] = args[0].detach().clone()
        handles.append(module.register_forward_pre_hook(before))
    try:
        adapter.prepare_query_length(len(token_ids))
        embedding = backbone.embed_tokens(ids)
        positions = torch.arange(len(token_ids), device=embedding.device)[None]
        mask = create_sliding_window_causal_mask(config=adapter.model.config, inputs_embeds=embedding,
                    attention_mask=torch.ones_like(ids), past_key_values=None, position_ids=positions)
        rotary = {kind: backbone.rotary_emb(embedding, position_ids=positions, layer_type=kind)
                  for kind in ("main", "compress")}
        streams = embedding.unsqueeze(2).expand(-1, -1, adapter.model.config.hc_mult, -1).contiguous()
        layer(streams, input_ids=ids, position_embeddings=rotary, position_ids=positions,
              attention_mask=mask, past_key_values=None)
    finally:
        for handle in handles:
            handle.remove()
    _require(observed.keys() == targets.keys(), "First-layer native input capture is incomplete")
    return ids, observed


def run_gpu_diagnostics(adapter, output, *, _cpu_test=False):
    """Resident-job entry point. ``_cpu_test`` is restricted to tiny CPU fixtures."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = adapter.model
    device = model.model.embed_tokens.weight.device
    _require(device.type == ("cpu" if _cpu_test else "cuda"), "Expected an already loaded native CUDA model")
    _require(not model.training and not any(p.requires_grad for p in model.parameters()), "Model must be frozen/eval")
    before, precision_before = _state(model), _precision()
    words = adapter.tokenizer.encode("The quick brown fox checks the complete cache prefix. " * 2,
                                     add_special_tokens=False)
    ids_list = ([0, 128803] + words[:4] + [128804, 128821]) if not _cpu_test else list(range(8))
    _require(len(ids_list) == 8, "Diagnostic requires eight actual input tokens")
    started, operations = time.time(), []
    with torch.inference_mode():
        ids, inputs = _capture_inputs(adapter, ids_list)
        adapter.prepare_query_length(1)
        layer = model.model.layers[0]
        for name in ("attn_hc", "ffn_hc"):
            hc = getattr(layer, name)
            value = hc.input_norm(inputs[name].flatten(2).float())
            operations.append(_measure(name + "/native_fp32_linear", value,
                                       lambda x, hc=hc: F.linear(x, hc.fn.float())))
            operations.append(_measure(name + "/complete_native_hc", inputs[name], hc))
        for name in ("q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj"):
            module = getattr(layer.self_attn, name)
            item = _measure("attention/" + name, inputs["attention/" + name], module)
            item.update(weight_dtype=str(module.weight.dtype), weight_shape=list(module.weight.shape),
                        block_size=getattr(module, "block_size", None), module_class=type(module).__name__)
            operations.append(item)

        mlp_input = inputs["mlp"]
        def route(x):
            return layer.mlp.gate(x, ids if x.shape[1] == 8 else ids[:, -1:]) if layer.mlp.is_hash else layer.mlp.gate(x)
        operations.append(_measure("mlp/router_native_linear", mlp_input,
                                   lambda x: F.linear(x.reshape(-1, x.shape[-1]), layer.mlp.gate.weight), output_axis=0))
        operations.append(_measure("mlp/router", mlp_input, route, output_axis=0))
        _, weights8, experts8 = route(mlp_input)
        _, weights1, experts1 = route(mlp_input[:, -1:].contiguous())
        routing = {"is_hash": bool(layer.mlp.is_hash), "gate_weight_dtype": str(layer.mlp.gate.weight.dtype),
                   "ids_m8": experts8.cpu().tolist(), "weights_m8": weights8.float().cpu().tolist(),
                   "ids_m1": experts1.cpu().tolist(), "weights_m1": weights1.float().cpu().tolist(),
                   "last_row_ids_order_equal": bool(torch.equal(experts8[-1:], experts1)),
                   "last_row_ids_set_equal": bool(torch.equal(experts8[-1:].sort().values, experts1.sort().values))}
        def fixed_routing_experts(x):
            many = x.shape[1] == 8
            return layer.mlp.experts(x.reshape(-1, x.shape[-1]), experts8 if many else experts8[-1:].contiguous(),
                                    weights8 if many else weights8[-1:].contiguous()).reshape(1, x.shape[1], -1)
        operations.append(_measure("mlp/fp4_experts_fixed_routing_and_weights", mlp_input, fixed_routing_experts))
        operations.append(_measure("mlp/shared_experts", mlp_input, layer.mlp.shared_experts))
        operations.append(_measure("mlp/complete_native_mlp", mlp_input,
                    lambda x: layer.mlp(x, input_ids=ids if x.shape[1] == 8 else ids[:, -1:])))

        expert = layer.mlp.experts
        packed_fp4 = expert.gate_up_proj.dtype == torch.int8
        if packed_fp4:
            from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
            bundle = load_finegrained_fp8_kernel()
            top_k = experts8.shape[-1]
            def projection(x, *, weight, scale):
                many = x.shape[1] == 8
                return bundle.batched_matmul(x.reshape(-1, x.shape[-1]), weight, scale,
                    block_size=expert.block_size,
                    expert_ids=(experts8 if many else experts8[-1:].contiguous()).reshape(-1)).reshape(1, x.shape[1], top_k, -1)
            repeated_input = mlp_input.unsqueeze(2).expand(-1, -1, top_k, -1).contiguous()
            operations.append(_measure("mlp/fp4_gate_up_projection_fixed_experts", repeated_input,
                    lambda x: projection(x, weight=expert.gate_up_proj, scale=expert.gate_up_proj_scale_inv)))
            gate_up8 = projection(repeated_input, weight=expert.gate_up_proj, scale=expert.gate_up_proj_scale_inv)
            down_input = expert._apply_gate(gate_up8).contiguous()
            operations.append(_measure("mlp/fp4_down_projection_identical_activated_input_fixed_experts", down_input,
                    lambda x: projection(x, weight=expert.down_proj, scale=expert.down_proj_scale_inv)))

    after, precision_after = _state(model), _precision()
    _require(before == after, "Model storage/version/parameter or buffer metadata changed")
    _require(precision_before == precision_after, "Precision configuration changed")
    arrays = {name.replace("/", "__"): _bits(value) for name, value in inputs.items()}
    arrays["input_ids"] = np.asarray(ids_list, dtype=np.int64)
    archive = output / "captured-input-native-bytes.npz"
    np.savez(archive, **arrays)
    source_paths = {"diagnostic": Path(__file__), "official_modeling": Path(inspect.getfile(type(layer))),
                    "official_experts": Path(inspect.getfile(type(layer.mlp.experts)))}
    all_finite = all(item[metric]["finite"] for op in operations for item in op["outputs"].values()
                     for metric in ("m8_vs_m1", "m8_repeat", "m1_repeat"))
    result = {"status": "diagnostic_only", "complete": True, "engineering_test_only": bool(_cpu_test),
              "started_unix": started, "finished_unix": time.time(), "input_tokens": ids_list,
              "layer": 0, "comparison_row": 7, "all_outputs_finite": all_finite,
              "native_packed_fp4": packed_fp4, "model_storage_and_versions_unchanged": True,
              "precision_before": precision_before, "precision_after": precision_after,
              "capture": "Actual first decoder layer, native embeddings/mHC/mask/RoPE, no cache",
              "source": {name: {"path": str(path.resolve()), "sha256": _sha(path)} for name, path in source_paths.items()},
              "captured_inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                  "archive_array": name.replace("/", "__"), "encoding": "exact contiguous native bytes as uint8"}
                  for name, value in inputs.items()},
              "input_archive": {"path": str(archive.resolve()), "sha256": _sha(archive)},
              "routing": routing, "operations": operations,
              "interpretation": "Observed shape dependence only; no tolerance relaxed, cache changed or ASR result produced"}
    path = output / "kernel-row-diagnostics.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
    return {"status": "diagnostic_only", "results_path": str(path.resolve()), "all_outputs_finite": all_finite,
            "shape_dependent_operations": [op["name"] for op in operations
                if any(not item["m8_vs_m1"]["bitexact"] for item in op["outputs"].values())]}
