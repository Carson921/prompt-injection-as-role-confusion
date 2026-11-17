"""Independent cached-forward and observational-hook checks before rollout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


def check_cached_forward(adapter, token_ids, *, selected, tolerance=0.25, rms_tolerance=0.05):
    """Compare a complete prefix with prefill plus an own-token cached step.

    BF16/FP8 kernels can round differently at different query lengths; the
    production check records those differences and also requires identical
    greedy next-token choice. The hook/no-hook comparison is exactly equal.
    """
    import numpy as np
    import torch
    from transformers import DynamicCache
    from deepseek_role_probe_model import HeadCapture
    model = adapter.model
    ids = torch.as_tensor([token_ids], dtype=torch.long, device=next(model.parameters()).device)
    length = ids.shape[1]
    if length < 2:
        raise ValueError("Cached-forward preflight needs at least two tokens")
    geometry = {"num_layers": model.config.num_hidden_layers,
                "num_heads": model.config.num_attention_heads, "head_dim": model.config.head_dim}
    prepare = getattr(adapter, "prepare_query_length", lambda n: None)

    def forward(values, *, mask_length, cache=None, use_cache=False):
        prepare(values.shape[1])
        return model(input_ids=values,
                     attention_mask=torch.ones((1, mask_length), dtype=torch.long, device=ids.device),
                     past_key_values=cache, use_cache=use_cache, logits_to_keep=1,
                     return_dict=True)

    with torch.inference_mode():
        full = forward(ids, mask_length=length)
        full_logits = full.logits.detach().float().cpu()
        del full
        capture = HeadCapture([length - 1], selected=selected, **geometry)
        with capture.install(model):
            observed = forward(ids, mask_length=length)
        observed_logits = observed.logits.detach().float().cpu()
        del observed
        if not torch.equal(full_logits, observed_logits):
            raise ValueError("Observational hooks changed native logits")
        full_features = capture.matrix()
        cache = DynamicCache(config=model.config)
        prefix = forward(ids[:, :-1], mask_length=length - 1, cache=cache, use_cache=True)
        if prefix.past_key_values is not cache:
            cache = prefix.past_key_values
        del prefix
        if cache is None or cache.get_seq_length() != length - 1:
            raise ValueError("Native prefill cache did not consume the complete prefix")
        cached_capture = HeadCapture([0], selected=selected, **geometry)
        with cached_capture.install(model):
            output = forward(ids[:, -1:], mask_length=length, cache=cache, use_cache=True)
        cached_logits = output.logits.detach().float().cpu()
        if output.past_key_values.get_seq_length() != length:
            raise ValueError("Own-token cached forward has an incorrect sequence position")
        cached_features = cached_capture.matrix()
        difference = (full_logits - cached_logits).double()
        maximum = float(difference.abs().max())
        rms = float(difference.square().mean().sqrt())
        same_next = bool(torch.equal(full_logits.argmax(-1), cached_logits.argmax(-1)))
        finite = bool(torch.isfinite(cached_logits).all()) and bool(np.isfinite(cached_features).all())
        if not finite or not same_next or maximum > tolerance or rms > rms_tolerance:
            raise ValueError(f"Cached-forward mismatch at {length}: finite={finite}, same_argmax={same_next}, max={maximum}, rms={rms}")
        result = {"tokens": length, "cache_consumed_tokens": length,
                  "finite": finite, "hook_nohook_bitexact": True,
                  "cached_full_next_token_equal": same_next,
                  "cached_full_logits_max_abs": maximum, "cached_full_logits_rms": rms,
                  "logits_max_abs_tolerance": tolerance, "logits_rms_tolerance": rms_tolerance,
                  "selected_features_shape": list(cached_features.shape),
                  "selected_features_max_abs_difference": float(np.max(np.abs(
                      cached_features.astype(np.float32) - full_features.astype(np.float32))))}
        del output, cache, capture, cached_capture, full_logits, observed_logits, cached_logits
        return result


def strict_same_shape_and_observe_split(adapter, token_ids, *, selected, reference_features):
    """Gate identical query shapes; report different-shape arithmetic separately.

    Native FP32 mHC matrix products and packed low precision arithmetic need
    not be bitwise invariant to matrix shape. This function does not apply a
    permissive numerical tolerance to that distinct computation.
    """
    import numpy as np
    import torch
    from transformers import DynamicCache
    from deepseek_role_probe_model import HeadCapture
    model = adapter.model
    ids = torch.as_tensor([token_ids], dtype=torch.long, device=model.model.embed_tokens.weight.device)
    n = len(token_ids)
    geometry = {"num_layers": model.config.num_hidden_layers,
                "num_heads": model.config.num_attention_heads, "head_dim": model.config.head_dim}
    prepare = getattr(adapter, "prepare_query_length", lambda n: None)
    def forward(values, cache=None, start=0, use_cache=False):
        prepare(values.shape[1])
        return model(input_ids=values, attention_mask=torch.ones((1, start+values.shape[1]), dtype=torch.long,
                     device=ids.device), position_ids=torch.arange(start, start+values.shape[1], device=ids.device)[None],
                     past_key_values=cache, use_cache=use_cache, logits_to_keep=1, return_dict=True)
    with torch.inference_mode():
        baseline = forward(ids).logits.detach().float().cpu()
        capture = HeadCapture(list(range(n)), selected=selected, **geometry)
        with capture.install(model):
            observed = forward(ids)
        if not torch.equal(baseline, observed.logits.detach().float().cpu()):
            raise ValueError("Frozen training head hooks changed same-shape native logits")
        legacy_features = capture.matrix()
        del observed, capture
        cache_on = forward(ids, DynamicCache(config=model.config), use_cache=True)
        if not torch.equal(baseline, cache_on.logits.detach().float().cpu()):
            raise ValueError("Same-shape native cache toggle changed logits")
        if cache_on.past_key_values.get_seq_length() != n:
            raise ValueError("Same-shape native prefill cache length differs")
        if not np.array_equal(legacy_features, reference_features):
            raise ValueError("Frozen training head capture differs from official-generation prefill features")
        del cache_on
        prefix = forward(ids[:, :-1], DynamicCache(config=model.config), use_cache=True)
        cache = prefix.past_key_values
        if cache.get_seq_length() != n-1:
            raise ValueError("Split native prefill cache length differs")
        del prefix
        last = HeadCapture([0], selected=selected, **geometry)
        with last.install(model):
            split = forward(ids[:, -1:], cache, n-1, True)
        split_logits = split.logits.detach().float().cpu()
        split_features = last.matrix()
        if split.past_key_values.get_seq_length() != n or not bool(torch.isfinite(split_logits).all()) or not np.isfinite(split_features).all():
            raise ValueError("Split native forward has invalid state or sequence position")
        delta = (baseline - split_logits).double()
        result = {"tokens": n, "cache_consumed_tokens": n,
                  "hook_nohook_bitexact": True, "same_shape_cache_toggle_bitexact": True,
                  "frozen_training_vs_generation_head_capture_bitexact": True,
                  "selected_features_shape": list(legacy_features.shape),
                  "full_vs_split_observation": {"is_acceptance_gate": False,
                      "same_next_token": bool(torch.equal(baseline.argmax(-1), split_logits.argmax(-1))),
                      "logits_max_abs": float(delta.abs().max()),
                      "logits_rms": float(delta.square().mean().sqrt()),
                      "selected_features_max_abs": float(np.abs(legacy_features[-1:].astype(np.float32)
                                                          - split_features.astype(np.float32)).max()),
                      "meaning": "Different native matrix and reduction shapes; no numerical equivalence is asserted"}}
        del split, cache, last
        return result


def check_all_head_writer(adapter, token_ids, output_dir):
    """Prove full native BF16 readout and selected features are observational."""
    import numpy as np
    import torch
    from transformers import DynamicCache
    from deepseek_agentdojo_model import ForwardCapture
    from deepseek_role_probe_model import HeadCapture
    model = adapter.model
    positions = [1, 2]
    geometry = {"num_layers": model.config.num_hidden_layers,
                "num_heads": model.config.num_attention_heads, "head_dim": model.config.head_dim}
    ids = torch.as_tensor([token_ids], dtype=torch.long, device=model.model.embed_tokens.weight.device)
    prepare = getattr(adapter, "prepare_query_length", lambda n: None)
    prepare(len(token_ids))
    def forward():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids),
                     past_key_values=DynamicCache(config=model.config), use_cache=True,
                     logits_to_keep=1, return_dict=True)
    blocks = {}
    with torch.inference_mode():
        baseline = forward().logits.detach().float().cpu()
        capture = ForwardCapture(model, adapter.selected_heads, positions=positions,
                    all_head_positions=positions, all_head_writer=lambda layer, bits: blocks.__setitem__(layer, bits.copy()),
                    **geometry)
        with capture:
            observed = forward()
        if not torch.equal(baseline, observed.logits.detach().float().cpu()):
            raise ValueError("Full native head writer changed logits")
        if set(blocks) != set(range(geometry["num_layers"])):
            raise ValueError("Full native head writer omitted layers")
        all_heads = np.stack([blocks[i] for i in range(geometry["num_layers"])], axis=1)
        if all_heads.dtype != np.uint16:
            raise ValueError("Full native BF16 head storage changed dtype")
        pairs = [divmod(h, geometry["num_heads"]) for h in adapter.selected_heads]
        selected_bits = np.stack([all_heads[:, layer, head] for layer, head in pairs], axis=1)
        restored = (selected_bits.astype(np.uint32) << 16).view(np.float32).astype(np.float16).reshape(len(positions), -1)
        if not np.array_equal(restored, capture.features):
            raise ValueError("Native BF16 head bits do not reconstruct selected FP16 features")
        old = HeadCapture(positions, selected=adapter.selected_heads, **geometry)
        with old.install(model):
            legacy = forward()
        if not torch.equal(baseline, legacy.logits.detach().float().cpu()) or not np.array_equal(old.matrix(), restored):
            raise ValueError("Full native writer differs from the frozen training capture")
        root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
        path = root / "full-head-observation.npz"
        probabilities = adapter.probe.predict_proba(restored)
        np.savez(path, bf16_uint16=all_heads, selected_fp16=restored, probabilities=probabilities,
                 positions=np.asarray(positions, np.int64), input_ids=np.asarray(token_ids, np.int64),
                 selected_heads=np.asarray(adapter.selected_heads, np.int64), roles=np.asarray(adapter.probe.roles))
        return {"status": "passed", "hook_nohook_bitexact": True, "native_bf16_dtype": True,
                "all_layers_observed": True, "selected_bits_reconstruction_bitexact": True,
                "frozen_training_capture_bitexact": True, "shape": list(all_heads.shape),
                "artifact": str(path), "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_preflight(adapter, output_dir):
    import numpy as np
    import torch
    from deepseek_role_probe_model import audit_model, memory_report
    from deepseek_role_probe_train import verify_freeze
    from deepseek_agentdojo_generate_diagnostics import run_gpu_diagnostics as compare_official_generation
    from deepseek_agentdojo_cache_audit import run_gpu_diagnostics as inspect_cache
    root = Path(output_dir)
    started = time.time()
    checks, cache_checks = [], []
    try:
        before = audit_model(adapter.model)
        frozen = verify_freeze(adapter.probe_dir)
        metadata = frozen["metadata"]
        if metadata["feature_dim"] != 8192 or len(metadata["heads"]) != 16:
            raise ValueError("The native probe geometry differs")
        tokenizer = adapter.tokenizer
        from deepseek_agentdojo_protocol import USER_PREFIX
        native_ids = {s: tokenizer.encode(s, add_special_tokens=False)
                      for s in (USER_PREFIX, "<｜Assistant｜>", "<think>")}
        if native_ids != {USER_PREFIX: [128803], "<｜Assistant｜>": [128804], "<think>": [128821]}:
            raise ValueError("Delimiter characters are not the verified native single tokens")
        long_ids = tokenizer.encode("The quick brown fox checks the complete cache prefix. " * 600,
                                    add_special_tokens=False)
        evidence_path = root / "native-cache-numerics-explanation.json"
        evidence = json.loads(evidence_path.read_text())
        if (evidence.get("status") != "root_cause_verified"
                or evidence.get("adapter_identity_sha256") != adapter.identity_sha256):
            raise ValueError("The source-bound native shape-dependence explanation is missing or belongs to another model")
        precision = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                     "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                     "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                     "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}
        if evidence["precision"] != precision or evidence["probe_checkpoint_sha256"] != frozen["probe_sha256"]:
            raise ValueError("Native arithmetic diagnosis precision or frozen probe differs")
        for absolute, expected in evidence["official_source_sha256"].items():
            if hashlib.sha256(Path(absolute).read_bytes()).hexdigest() != expected:
                raise ValueError("Native arithmetic source changed since diagnosis")
        for relative, expected in evidence["artifacts_sha256"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Native arithmetic diagnosis artifact changed")
        for length in (8, 127, 257, 1029):
            ids = [0, 128803] + long_ids[:length - 4] + [128804, 128821]
            if len(ids) != length:
                raise ValueError("Preflight token length differs")
            folder = root / "preflight" / f"official-generate-{length}"
            reference = compare_official_generation(adapter, folder, token_ids=ids, steps=32)
            exact_keys = ("generated_ids_equal", "sampling_logits_bitexact", "common_observed_own_features_bitexact",
                          "common_observed_own_probabilities_bitexact")
            if reference["status"] != "passed" or not all(reference[k] for k in exact_keys):
                raise ValueError("Manual cached generation differs from explicit official greedy generate")
            if reference["terminal_reference"]["status"] not in ("passed", "eos_unforwarded"):
                raise ValueError("Terminal own-state reference differs")
            with np.load(folder / "generation-diagnostic.npz", allow_pickle=False) as arrays:
                check = strict_same_shape_and_observe_split(adapter, ids, selected=metadata["heads"],
                                                            reference_features=arrays["hf_input_features"])
            check["official_generation"] = {k: reference[k] for k in
                ("status", "hf_generated_tokens", "adapter_generated_tokens", *exact_keys, "input_features", "input_probabilities", "terminal_reference")}
            for key in ("input_features", "input_probabilities"):
                if not reference[key]["bitexact"]:
                    raise ValueError("Official generation input readouts differ")
            check["artifacts"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(folder.glob("generation-diagnostic*")) if path.is_file()}
            checks.append(check)
            print(json.dumps({"stage": "native_decode_preflight", **check}), flush=True)
        for length in (8, 128, 129):
            ids = [0, 128803] + long_ids[:length - 4] + [128804, 128821]
            folder = root / "preflight" / f"cache-state-{length}"
            result = inspect_cache(adapter, folder, token_ids=ids)
            if result["status"] != "diagnostic_completed" or not result["structural_invariants_passed"]:
                raise ValueError("Native cache state violates a strict structural invariant")
            cache_checks.append({"tokens": length, "structural_invariants_passed": True,
                                 "checks": sum(len(s["checks"]) for s in result["snapshots"]),
                                 "artifacts": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in sorted(folder.glob("cache-state-*")) if path.is_file()}})
        full_head_check = check_all_head_writer(adapter,
                    [0, 128803] + long_ids[:4] + [128804, 128821], root / "preflight")
        after = audit_model(adapter.model)
        if before != after:
            raise ValueError("Preflight changed the frozen native backbone audit")
        result = {"format_version": "deepseek-native-generation-preflight-v2", "status": "passed",
                  "reference": "Strict bitexact official HF greedy cached generation with identical query shapes",
                  "different_shape_policy": "Full versus split values remain diagnostic; no old tolerance was widened",
                  "root_cause_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                  "started_unix": started, "finished_unix": time.time(),
                  "checkpoint_sha256": frozen["probe_sha256"], "checks": checks,
                  "cache_structure_checks": cache_checks,
                  "full_head_writer_check": full_head_check,
                  "native_delimiter_token_ids": native_ids, "backbone_audit": after,
                  "gpu_memory": memory_report(), "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    except BaseException as error:
        result = {"status": "failed", "started_unix": started, "finished_unix": time.time(),
                  "checks": checks, "cache_structure_checks": cache_checks,
                  "error_type": type(error).__name__, "error": str(error)}
        path = root / "native-generation-preflight.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        raise
    path = root / "native-generation-preflight.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.cuda.empty_cache()
    return result
