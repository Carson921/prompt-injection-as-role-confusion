"""Compare official HF generation with the unchanged native adapter loop.

This is a diagnostic job for an already resident model. It never loads weights,
changes their representation, runs AgentDojo, emits benchmark scores, or changes
adapter budgets. All hooks are observational and removed even after a failure.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape, "Compared shapes differ")
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {"bitexact": bool(np.array_equal(a, b)), "max_abs": float(np.abs(delta).max(initial=0)),
            "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0., "shape": list(a.shape)}


@contextmanager
def observe_official_forwards(adapter, records):
    """Install a fresh ForwardCapture for *each* real CausalLM invocation."""
    from deepseek_agentdojo_model import ForwardCapture
    active = {"capture": None, "record": None}

    def before(module, args, kwargs):
        import torch
        require(active["capture"] is None, "Unexpected recursive CausalLM forward")
        values = kwargs.get("input_ids", args[0] if args else None)
        require(torch.is_tensor(values) and values.ndim == 2 and values.shape[0] == 1,
                "Official reference requires real B=1 token IDs")
        require(kwargs.get("use_cache") is True, "Official reference silently disabled its cache")
        cache = kwargs.get("past_key_values")
        start = 0 if cache is None else int(cache.get_seq_length())
        positions = kwargs.get("position_ids")
        if positions is not None:
            require(positions.tolist() == [list(range(start, start + values.shape[1]))],
                    "Official reference position IDs are not consecutive absolute positions")
        record = {"input_ids": values.detach().cpu().numpy()[0].copy(), "absolute_start": start,
                  "query_tokens": int(values.shape[1]), "position_ids_supplied": positions is not None}
        mask = kwargs.get("attention_mask")
        if mask is not None:
            require(mask.ndim == 2 and tuple(mask.shape) == (1, start + values.shape[1])
                    and bool(mask.bool().all()), "Official reference did not use the complete unpadded input mask")
        adapter.prepare_query_length(values.shape[1])
        capture = ForwardCapture(adapter.model, adapter.selected_heads, **adapter.geometry)
        active["capture"], active["record"] = capture, record
        capture.__enter__()

    def after(module, args, kwargs, result):
        capture, record = active["capture"], active["record"]
        if capture is None:
            return
        active["capture"], active["record"] = None, None
        if result is None:
            capture.__exit__(RuntimeError, RuntimeError("Official forward failed"), None)
            return
        capture.__exit__(None, None, None)
        record["features"] = capture.features.copy()
        record["probabilities"] = adapter._probabilities(record["features"])
        record["logits"] = result.logits[:, -1, :].detach().float().cpu().numpy()[0].copy()
        record["cache_consumed_tokens"] = int(result.past_key_values.get_seq_length())
        require(record["cache_consumed_tokens"] == record["absolute_start"] + record["query_tokens"],
                "Official cache consumed the wrong number of tokens")
        records.append(record)

    pre = adapter.model.register_forward_pre_hook(before, with_kwargs=True)
    post = adapter.model.register_forward_hook(after, with_kwargs=True, always_call=True)
    try:
        yield
    finally:
        pre.remove()
        post.remove()
        capture = active["capture"]
        if capture is not None:
            capture.__exit__(RuntimeError, RuntimeError("Generation ended during a forward"), None)


def run_gpu_diagnostics(adapter, output, token_ids=None, steps=32):
    """Run on the caller's already loaded model; CPU tiny adapters also work."""
    import torch
    from transformers import DynamicCache

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    steps = int(steps)
    require(steps > 0, "Generation diagnostic needs a positive step budget")
    if token_ids is None:
        words = adapter.tokenizer.encode("The quick brown fox checks the complete cache prefix. " * 600,
                                         add_special_tokens=False)
        token_ids = [0, 128803] + words[:4] + [128804, 128821]
    token_ids = list(map(int, token_ids))
    require(token_ids and len(token_ids) + steps <= adapter.max_model_len, "Diagnostic context budget exceeded")
    eos = int(adapter.codec.eos_token_id)
    model = adapter.model
    device = model.model.embed_tokens.weight.device
    initial_generation_config = copy.deepcopy(model.generation_config.to_dict())
    initial_adapter_generation = copy.deepcopy(adapter.backend["generation"])
    generation_config = copy.deepcopy(model.generation_config)
    generation_config._from_model_config = False
    settings = {"do_sample": False, "use_cache": True, "max_new_tokens": steps, "num_beams": 1,
                "num_return_sequences": 1, "return_dict_in_generate": True, "output_logits": True,
                "output_scores": False, "output_hidden_states": False, "output_attentions": False,
                "eos_token_id": eos, "pad_token_id": eos,
                "min_new_tokens": 0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
                "forced_bos_token_id": None, "forced_eos_token_id": None,
                "suppress_tokens": None, "begin_suppress_tokens": None, "renormalize_logits": False}
    for name, value in settings.items():
        setattr(generation_config, name, value)
    metadata = {"format_version": "deepseek-agentdojo-official-generate-diagnostic-v1",
                "diagnostic_only": True, "asr_or_case_scores_produced": False,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "adapter_identity_sha256": adapter.identity_sha256,
                "engineering_test_only": bool(getattr(adapter, "cpu_test", False)),
                "input_tokens": len(token_ids), "steps_budget": steps, "hf_generate_settings": settings,
                "sampling_alignment": "step j compares the logits that sample generated token j",
                "own_state_alignment": "features row j consumes generated token j; it does not predict that token",
                "terminal_policy": "Manual adapter forwards a final normal token for its own state; HF generate stops before consuming it. EOS is unforwarded in both."}
    write(output / "generation-diagnostic-input.json", {**metadata, "input_ids": token_ids})
    records = []
    try:
        with adapter._lock, torch.inference_mode():
            adapter.prepare_query_length(len(token_ids))
            adapter.prepare_query_length(1)
            ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            with observe_official_forwards(adapter, records):
                official = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    generation_config=generation_config, past_key_values=DynamicCache(config=model.config))
            official_ids = official.sequences[0, len(token_ids):].detach().cpu().numpy().astype(np.int64)
            official_logits = np.stack([value[0].detach().float().cpu().numpy() for value in official.logits])
            require(len(records) == len(official_ids) == len(official_logits), "HF generate forward/output lengths differ")
            require(np.array_equal(records[0]["input_ids"], token_ids), "Official prefill changed input IDs")
            require(np.array_equal(np.stack([r["logits"] for r in records]), official_logits),
                    "HF raw generation logits differ from its actual CausalLM outputs")
            official_input_features = records[0]["features"]
            official_input_probabilities = records[0]["probabilities"]
            feature_dim = adapter.probe.feature_dim
            hf_own_features = np.full((len(official_ids), feature_dim), np.nan, np.float16)
            hf_own_probabilities = np.full((len(official_ids), 5), np.nan, np.float32)
            hf_own_observed = np.zeros(len(official_ids), bool)
            for j, record in enumerate(records[1:]):
                require(record["input_ids"].tolist() == [int(official_ids[j])]
                        and record["absolute_start"] == len(token_ids) + j, "HF cached own-token alignment differs")
                hf_own_features[j] = record["features"][0]
                hf_own_probabilities[j] = record["probabilities"][0]
                hf_own_observed[j] = True
            # Release HF's cache before constructing the independent manual cache.
            del official
            manual_ids, manual_logits, own_features, own_probabilities, own_observed = [], [], [], [], []
            logits, cache, input_features = adapter._forward(token_ids, None, 0)
            input_probabilities = adapter._probabilities(input_features)
            for j in range(steps):
                manual_logits.append(logits[0].detach().float().cpu().numpy().copy())
                token = int(torch.argmax(logits[0]).item())
                manual_ids.append(token)
                own_features.append(np.full(feature_dim, np.nan, np.float16))
                own_probabilities.append(np.full(5, np.nan, np.float32))
                own_observed.append(False)
                if token == eos:
                    break
                logits, cache, features = adapter._forward([token], cache, len(token_ids) + j)
                own_features[-1] = features[0].copy()
                own_probabilities[-1] = adapter._probabilities(features)[0]
                own_observed[-1] = True
            del cache, logits
        manual_ids = np.asarray(manual_ids, np.int64)
        manual_logits = np.asarray(manual_logits, np.float32)
        own_features = np.asarray(own_features, np.float16)
        own_probabilities = np.asarray(own_probabilities, np.float32)
        own_observed = np.asarray(own_observed, bool)
        require(np.isfinite(official_logits).all() and np.isfinite(manual_logits).all(), "Nonfinite generation logits")
        count = min(len(official_ids), len(manual_ids))
        prefix_same, rows = True, []
        for j in range(count):
            equal_token = bool(official_ids[j] == manual_ids[j])
            common_own = bool(hf_own_observed[j] and own_observed[j])
            rows.append({"step": j, "absolute_generated_position": len(token_ids) + j,
                         "sampling_history_equal_before_step": prefix_same,
                         "hf_token_id": int(official_ids[j]), "adapter_token_id": int(manual_ids[j]),
                         "token_equal": equal_token, "sampling_logits": difference(official_logits[j], manual_logits[j]),
                         "own_feature_shape": [feature_dim], "hf_own_state_observed": bool(hf_own_observed[j]),
                         "adapter_own_state_observed": bool(own_observed[j]),
                         "own_features": difference(hf_own_features[j], own_features[j]) if common_own else None,
                         "own_probabilities": difference(hf_own_probabilities[j], own_probabilities[j]) if common_own else None})
            prefix_same &= equal_token
        same_ids = bool(np.array_equal(official_ids, manual_ids))
        exact_logits = same_ids and all(row["sampling_logits"]["bitexact"] for row in rows)
        same_own = same_ids and all(row["own_features"] is None or row["own_features"]["bitexact"] for row in rows)
        same_probs = same_ids and all(row["own_probabilities"] is None or row["own_probabilities"]["bitexact"] for row in rows)
        require(model.generation_config.to_dict() == initial_generation_config,
                "Diagnostic changed the model's generation configuration")
        require(adapter.backend["generation"] == initial_adapter_generation,
                "Diagnostic changed the adapter's fixed generation protocol")
        np.savez(output / "generation-diagnostic.npz", input_ids=np.asarray(token_ids, np.int64),
                 hf_generated_ids=official_ids, adapter_generated_ids=manual_ids,
                 hf_sampling_logits=official_logits, adapter_sampling_logits=manual_logits,
                 hf_input_features=official_input_features, adapter_input_features=input_features,
                 hf_input_probabilities=official_input_probabilities, adapter_input_probabilities=input_probabilities,
                 hf_own_features=hf_own_features, adapter_own_features=own_features,
                 hf_own_probabilities=hf_own_probabilities, adapter_own_probabilities=own_probabilities,
                 hf_own_observed=hf_own_observed, adapter_own_observed=own_observed,
                 roles=np.asarray(adapter.probe.roles))
        result = {**metadata, "status": "passed" if exact_logits and same_own and same_probs else "mismatch",
                  "hf_generated_tokens": len(official_ids), "adapter_generated_tokens": len(manual_ids),
                  "generated_ids_equal": same_ids, "sampling_logits_bitexact": exact_logits,
                  "common_observed_own_features_bitexact": same_own,
                  "common_observed_own_probabilities_bitexact": same_probs,
                  "input_features": difference(official_input_features, input_features),
                  "input_probabilities": difference(official_input_probabilities, input_probabilities),
                  "hf_forward_observations": [{key: value for key, value in record.items()
                        if key not in ("features", "probabilities", "logits", "input_ids")} for record in records],
                  "steps": rows, "seconds": time.monotonic() - started,
                  "files": {"generation-diagnostic.npz": hashlib.sha256((output / "generation-diagnostic.npz").read_bytes()).hexdigest()}}
        write(output / "generation-diagnostic.json", result)
        return result
    except Exception as error:
        write(output / "generation-diagnostic.json", {**metadata, "status": "failed",
              "error": f"{type(error).__name__}: {error}", "hf_completed_forwards": len(records),
              "seconds": time.monotonic() - started})
        raise
