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


def run_preflight(adapter, output_dir):
    import torch
    from deepseek_role_probe_model import audit_model, memory_report
    from deepseek_role_probe_train import verify_freeze
    root = Path(output_dir)
    started = time.time()
    checks = []
    try:
        before = audit_model(adapter.model)
        frozen = verify_freeze(adapter.probe_dir)
        metadata = frozen["metadata"]
        if metadata["feature_dim"] != 8192 or len(metadata["heads"]) != 16:
            raise ValueError("The native probe geometry differs")
        tokenizer = adapter.tokenizer
        from deepseek_agentdojo_protocol import USER_PREFIX, USER_SUFFIX, build_payload
        native_ids = {s: tokenizer.encode(s, add_special_tokens=False)
                      for s in (USER_PREFIX, "<｜Assistant｜>", "<think>")}
        if native_ids != {USER_PREFIX: [128803], "<｜Assistant｜>": [128804], "<think>": [128821]}:
            raise ValueError("Delimiter characters are not the verified native single tokens")
        long_ids = tokenizer.encode("The quick brown fox checks the complete cache prefix. " * 600,
                                    add_special_tokens=False)
        for length in (8, 257, 1029):
            ids = [0, 128803] + long_ids[:length - 4] + [128804, 128821]
            if len(ids) != length:
                raise ValueError("Preflight token length differs")
            check = check_cached_forward(adapter, ids, selected=metadata["heads"])
            checks.append(check)
            print(json.dumps({"stage": "native_decode_preflight", **check}), flush=True)
        after = audit_model(adapter.model)
        if before != after:
            raise ValueError("Preflight changed the frozen native backbone audit")
        result = {"status": "passed", "started_unix": started, "finished_unix": time.time(),
                  "checkpoint_sha256": frozen["probe_sha256"], "checks": checks,
                  "native_delimiter_token_ids": native_ids, "backbone_audit": after,
                  "gpu_memory": memory_report(), "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    except BaseException as error:
        result = {"status": "failed", "started_unix": started, "finished_unix": time.time(),
                  "checks": checks, "error_type": type(error).__name__, "error": str(error)}
        path = root / "native-generation-preflight.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        raise
    path = root / "native-generation-preflight.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    torch.cuda.empty_cache()
    return result
