"""Role-probe pipeline for Qwen3.5-397B-A17B-GPTQ-Int4: the DeepSeek runner with the Qwen backbone.

Stages are those of run_deepseek_role_probe.py (preflight, FIT-only head scan, FIT/DEV extraction,
candidate training with DEV selection, freeze, TEST extraction, evaluation).  Differences:
  * backbone = qwen35_role_probe_model (packed GPTQ experts; 15 full-attention layers x 32 heads x 256);
  * extraction batches consecutive records (right padding, causal model) because expanding the packed
    experts is a per-forward cost; a padded-vs-single check is part of the preflight.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np

import deepseek_role_probe_train as trainer
import qwen35_role_probe_model as backend
import run_deepseek_role_probe as R

R.backend = backend
R.CAPTURE = trainer.CAPTURE = backend.FEATURE_POSITION
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs/qwen35-397b-gptq-int4-role-probe-20261120"
BATCH_TOKENS, BATCH_RECORDS = 24000, 32
require, read, write, sha = R.require, R.read, R.write, R.sha


def model_binding(model_dir):
    root = Path(model_dir)
    return {"model_id": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4", "path": str(root.resolve()), "config_sha256": sha(root / "config.json"),
            "tokenizer_sha256": sha(root / "tokenizer.json"), "chat_template_sha256": sha(root / "chat_template.jinja"),
            "experts": "packed GPTQ int4", "dense": "BF16", "offload": False}


def batches(iterator):
    """Group consecutive (index, record) pairs under a padded-token budget."""
    group, longest = [], 0
    for item in iterator:
        length = len(item[1]["input_ids"])
        if group and (max(longest, length) * (len(group) + 1) > BATCH_TOKENS or len(group) >= BATCH_RECORDS):
            yield group
            group, longest = [], 0
        group.append(item)
        longest = max(longest, length)
    if group:
        yield group


def extract_batch(model, group, heads, pad_id):
    """-> list of float16 matrices, one per record (None when a record has no targets)."""
    width = max(len(record["input_ids"]) for _, record in group)
    ids = np.full((len(group), width), pad_id, np.int64)
    mask = np.zeros((len(group), width), np.int64)
    coordinates, counts = [], []
    for row, (_, record) in enumerate(group):
        n = len(record["input_ids"])
        ids[row, :n], mask[row, :n] = record["input_ids"], 1
        coordinates += [(row, t["token_ix"]) for t in record["targets"]]
        counts.append(len(record["targets"]))
    if not coordinates:
        return [None] * len(group)
    capture = backend.HeadCapture(np.asarray(coordinates), selected=heads)
    with capture.install(model):
        backend._forward(model, ids, mask)
    matrix, result, cursor = capture.matrix(), [], 0
    for count in counts:
        result.append(matrix[cursor:cursor + count] if count else None)
        cursor += count
    return result


def generation_check(model, tokenizer, steps=16):
    """Greedy continuation through the cache path: garbage here means the packed experts are read wrongly."""
    import torch
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    device = model.model.embed_tokens.weight.device
    produced, cache, step = [], None, torch.as_tensor([ids], device=device)
    started = time.monotonic()
    with torch.inference_mode():
        for _ in range(steps):
            out = model.model(input_ids=step, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            token = int(model.lm_head(out.last_hidden_state[:, -1]).float().argmax(-1))
            produced.append(token)
            if token == tokenizer.eos_token_id:
                break
            step = torch.as_tensor([[token]], device=device)
    text = tokenizer.decode(produced)
    print(json.dumps({"stage": "generation_check", "text": text, "seconds": time.monotonic() - started}), flush=True)
    require("Paris" in text, f"Greedy generation is not sensible: {text!r}")
    return {"prompt": prompt, "completion": text, "tokens": produced}


def preflight(model, data_dir, output, config, pad_id, tokenizer):
    import torch
    generation = generation_check(model, tokenizer)
    records = [r for _, r in zip(range(48), R.records(data_dir / "inputs-fit.jsonl"))]
    long_record = max((r for s in ("fit", "dev") for r in R.records(data_dir / f"inputs-{s}.jsonl")), key=lambda r: len(r["input_ids"]))
    checks = []
    for record in (records[0], long_record):
        ids = record["input_ids"]
        started = time.monotonic()
        before = backend._forward(model, ids).float().cpu()
        positions = sorted(set((0, len(ids) // 2, len(ids) - 1)))
        captured = {}
        capture = backend.HeadCapture(positions, receive=lambda layer, x: captured.update({layer: tuple(x.shape)}))
        with capture.install(model):
            after = backend._forward(model, ids).float().cpu()
        require(bool(torch.isfinite(before).all()), "Nonfinite preflight output")
        checks.append({"tokens": len(ids), "layers_captured": len(captured), "hook_output_bitexact": bool(torch.equal(before, after)),
                       "seconds": time.monotonic() - started})
        print(json.dumps({"stage": "preflight_forward", **checks[-1]}), flush=True)
    # padded batch vs single forwards (selected = one head per scanned layer)
    heads = [layer * backend.HEADS + (7 * layer) % backend.HEADS for layer in range(backend.LAYERS)]
    group = [(i, r) for i, r in enumerate(records[:6])]
    started = time.monotonic()
    batched = extract_batch(model, group, heads, pad_id)
    batch_seconds = time.monotonic() - started
    # Batch composition changes BF16 rounding, and top-k expert routing turns rounding into discrete changes, so
    # batched and single forwards agree only approximately in late layers (also true for identical unpadded rows).
    worst, cosines, relative = 0.0, [], []
    for (_, record), block in zip(group, batched):
        single = backend.extract_selected(model, record["input_ids"], [t["token_ix"] for t in record["targets"]], heads).astype(np.float32)
        block = block.astype(np.float32)
        worst = max(worst, float(np.abs(single - block).max()))
        cosines.append(float((single * block).sum() / np.linalg.norm(single) / np.linalg.norm(block)))
        relative.append(float(np.linalg.norm(single - block) / np.linalg.norm(single)))
    result = {"status": "passed", "generation_check": generation, "checks": checks,
              "padded_batch_vs_single": {"records": len(group), "max_abs_difference": worst, "cosine": cosines,
                                         "relative_l2_difference": relative, "batch_seconds": batch_seconds,
                                         "note": "difference is batch-composition numerics (BF16 + discrete expert routing), not padding: two identical unpadded rows differ from the single forward by the same amount"},
              "backbone_audit": backend.audit_model(model), "gpu_memory": backend.memory_report(),
              "maximum_input_tokens": len(long_record["input_ids"])}
    print(json.dumps({"stage": "preflight_batch", **result["padded_batch_vs_single"]}), flush=True)
    require(min(cosines) >= 0.9, "Batched and single forwards disagree beyond numerical noise")
    write(output / "preflight.json", result)
    return result


def extract(model, data_dir, output, config, roles, binding, selection, splits, pad_id):
    """R.extract with batched forwards (same shard format, resume rules and TEST seal)."""
    root = output / "features"
    root.mkdir(exist_ok=True)
    manifest_path = root / "manifest.json"
    split_sources = {s: sha(data_dir / f"inputs-{s}.jsonl") for s in R.SPLITS}
    signature = {"model_binding": binding, "heads": selection["heads"], "capture": R.CAPTURE,
                 "selection_sha256": sha(output / "head-selection.json"), "pilot_fit_sha256": split_sources["fit"],
                 "config_sha256": R.object_sha(config), "runner_implementation_sha256": sha(__file__),
                 "capture_implementation_sha256": sha(backend.__file__)}
    manifest = {"format_version": "deepseek-role-features-v1", "roles": roles, "feature_dim": 16 * backend.HEAD_DIM,
                "head_dim": backend.HEAD_DIM, "heads": selection["heads"], "feature_position": R.CAPTURE, "model_binding": binding,
                "selection_sha256": signature["selection_sha256"], "pilot_fit_sha256": split_sources["fit"],
                "data_hash": R.object_sha(split_sources), "config_hash": R.object_sha(config), "signature": signature,
                "split_sources": split_sources, "splits": {}, "progress": {}}
    if manifest_path.exists():
        existing = read(manifest_path)
        require(existing["signature"] == signature and existing["split_sources"] == split_sources, "Feature cache belongs to a different run")
        manifest = existing
    write(manifest_path, manifest)
    for split in splits:
        if split == "test":
            require((output / "freeze.json").is_file(), "TEST extraction is sealed until classifier freeze")
            trainer.verify_freeze(output)
        split_signature = {**signature, "split": split, "input_sha256": split_sources[split]}
        entries = manifest["splits"].setdefault(split, [])
        cursor = 0
        for entry in entries:
            meta = R.verify_feature_shard(root, entry, split_signature)
            require(meta["record_start"] == cursor, "Feature cache has a record gap")
            cursor = meta["record_end"]
        if manifest["progress"].get(split, {}).get("complete"):
            continue
        buffers = {key: [] for key in ("x", "y", "domain", "source", "family", "groups")}
        buffered, start, count, tokens, started = 0, cursor, cursor, 0, time.monotonic()
        pending = ((ix, record) for ix, record in enumerate(R.records(data_dir / f"inputs-{split}.jsonl")) if ix >= cursor)
        for group in batches(pending):
            for _, record in group:
                R.validate_record(record, split, roles)
            for (ix, record), x in zip(group, extract_batch(model, group, selection["heads"], pad_id)):
                count = ix + 1
                tokens += len(record["input_ids"])
                if x is not None:
                    for key, values in R.feature_arrays(record, x, roles).items():
                        buffers[key].append(values)
                    buffered += len(x)
            if buffered >= int(config["extraction_shard_rows"]):  # shards end on batch boundaries
                entries.append(R.save_feature_shard(root, split, len(entries), buffers, start, count, split_signature))
                manifest["progress"][split] = {"complete": False, "records": count, "rows": sum(e["rows"] for e in entries)}
                write(manifest_path, manifest)
                buffers = {key: [] for key in buffers}; buffered = 0; start = count
            print(json.dumps({"stage": "extract", "split": split, "records": count, "input_tokens": tokens, "batch": len(group),
                              "committed_rows": sum(e["rows"] for e in entries), "seconds": round(time.monotonic() - started, 1)}), flush=True)
        if buffered:
            entries.append(R.save_feature_shard(root, split, len(entries), buffers, start, count, split_signature))
        manifest["progress"][split] = {"complete": True, "records": count, "rows": sum(e["rows"] for e in entries)}
        write(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(backend.DEFAULT_MODEL))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("preflight", "scan", "extract", "all"), default="all")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    config = {"model": str(args.model.resolve()), "seed": 20261115, "top_k": 16,
              "scan_statistics_sources_per_stratum": int(os.environ.get("QWEN_SCAN_STAT", 16)), "scan_inner_sources_per_stratum": int(os.environ.get("QWEN_SCAN_INNER", 8)),
              "scan_rows_per_domain_role": 2048, "scan_checkpoint_records": 64, "extraction_shard_rows": 8192,
              "extraction_batch_tokens": BATCH_TOKENS, "extraction_batch_records": BATCH_RECORDS,
              "native_roles": "system,user,cot,assistant,tool",
              "classifier": {"device": "cuda:0", "batch_size": 4096, "learning_rate": .001, "weight_decay": .001, "dropout": .1,
                             "gradient_clip": 5., "max_epochs": 45, "min_epochs": 10, "patience": 8, "checkpoint_steps": 100,
                             "bootstrap_replicates": 2000, "cache_features_on_gpu": True,
                             "profile_weights": {"c4_dolma3": .5, "agentdojo": .5}}}
    config_path = output / "config.json"
    if config_path.exists():
        require(read(config_path) == config, "Run config changed; use a fresh output directory")
    else:
        write(config_path, config)
    (output / ".gitignore").write_text("data/\nfeatures/\nhead-scan/*.npz\n*.log\n")
    roles = read(data_dir / "manifest.json")["roles"]
    binding = model_binding(args.model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = backend.load_model(args.model, output, lm_head=True)
    import torch
    try:
        if not (output / "preflight.json").exists():
            preflight(model, data_dir, output, config, pad_id, tokenizer)
        if args.stage == "preflight":
            return
        selection = R.select_heads(model, data_dir, output, config, roles, binding)
        if args.stage == "scan":
            return
        extract(model, data_dir, output, config, roles, binding, selection, ("fit", "dev"), pad_id)
        if args.stage == "extract":
            return
        if (output / "freeze.json").exists():
            trainer.verify_freeze(output)
        else:
            trainer.train_candidates(output / "features", output, config, resume=True)
            trainer.freeze(output)
        gc.collect(); torch.cuda.empty_cache()
        frozen_sha = sha(output / "checkpoint.npz")
        release_path = output / "test-release.json"
        if release_path.exists():
            require(read(release_path)["checkpoint_sha256"] == frozen_sha, "TEST release belongs to another probe")
        else:
            write(release_path, {"checkpoint_sha256": frozen_sha, "released_unix": time.time(), "selection_uses_dev_only": True})
        extract(model, data_dir, output, config, roles, binding, selection, ("test",), pad_id)
        require(sha(output / "checkpoint.npz") == frozen_sha, "Checkpoint changed after TEST release")
        trainer.evaluate(output / "features", output, "test")
        write(output / "runtime-final.json", {"backbone_audit": backend.audit_model(model), "gpu_memory": backend.memory_report(),
                                               "checkpoint_sha256": frozen_sha, "finished_unix": time.time()})
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
