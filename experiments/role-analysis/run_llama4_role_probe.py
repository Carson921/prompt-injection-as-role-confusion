"""Role-probe pipeline for Llama-4-Scout-17B-16E-Instruct: the DeepSeek / Qwen runner with the Llama 4 backbone.

Stages are those of run_qwen35_role_probe.py (preflight, FIT-only head scan, batched FIT/DEV extraction,
candidate training with DEV selection, freeze, TEST extraction, evaluation).  Differences:
  * backbone = llama4_role_probe_model (dense BF16 text decoder, 48 layers x 40 heads x 128 dims);
  * four roles (system, user, assistant, tool): the model has no reasoning channel;
  * the preflight also compares the sparse top-1 MoE forward with the stock Transformers forward.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np

import llama4_role_probe_model as backend
import run_qwen35_role_probe as Q   # generic stage code; its backend is swapped below

R, trainer = Q.R, Q.trainer
Q.backend = R.backend = backend
R.CAPTURE = trainer.CAPTURE = backend.FEATURE_POSITION
Q.BATCH_TOKENS, Q.BATCH_RECORDS = int(os.environ.get("LLAMA4_BATCH_TOKENS", 12000)), 32
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs/llama4-scout-17b-16e-role-probe-20261125"
require, read, write, sha = R.require, R.read, R.write, R.sha


def model_binding(model_dir):
    root = Path(model_dir)
    return {"model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "path": str(root.resolve()), "config_sha256": sha(root / "config.json"),
            "tokenizer_sha256": sha(root / "tokenizer.json"), "chat_template_sha256": sha(root / "chat_template.jinja"),
            "weights": "BF16 dense, text decoder only", "offload": False}


def moe_check(model, data_dir):
    """Selected-layer features with the sparse top-1 MoE forward vs the stock Transformers forward."""
    record = next(R.records(data_dir / "inputs-fit.jsonl"))
    heads = [layer * backend.HEADS + (7 * layer) % backend.HEADS for layer in range(backend.LAYERS)]
    positions = [t["token_ix"] for t in record["targets"]]
    values = {}
    for sparse in (True, False):
        backend.set_moe_mode(model, sparse)
        started = time.monotonic()
        values[sparse] = backend.extract_selected(model, record["input_ids"], positions, heads).astype(np.float32)
        values[f"seconds_{sparse}"] = time.monotonic() - started
    backend.set_moe_mode(model, os.environ.get("LLAMA4_SPARSE_MOE", "1") != "0")
    a, b = values[True], values[False]
    result = {"tokens": len(record["input_ids"]), "cosine": float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b)),
              "relative_l2_difference": float(np.linalg.norm(a - b) / np.linalg.norm(b)), "max_abs_difference": float(np.abs(a - b).max()),
              "sparse_seconds": values["seconds_True"], "dense_seconds": values["seconds_False"], "sparse_in_use": model._llama4_sparse_moe}
    print(json.dumps({"stage": "moe_check", **result}), flush=True)
    require(result["cosine"] >= 0.999, "Sparse and dense MoE forwards disagree")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(backend.DEFAULT_MODEL))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("preflight", "scan", "extract", "all"), default="all")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    config = {"model": str(args.model.resolve()), "seed": 20261115, "top_k": 16,
              "scan_statistics_sources_per_stratum": int(os.environ.get("LLAMA4_SCAN_STAT", 48)),
              "scan_inner_sources_per_stratum": int(os.environ.get("LLAMA4_SCAN_INNER", 24)),
              "scan_rows_per_domain_role": 2048, "scan_checkpoint_records": 64, "extraction_shard_rows": 8192,
              "extraction_batch_tokens": Q.BATCH_TOKENS, "extraction_batch_records": Q.BATCH_RECORDS,
              "native_roles": "system,user,assistant,tool",
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
            check = moe_check(model, data_dir)
            Q.preflight(model, data_dir, output, config, pad_id, tokenizer)
            write(output / "preflight.json", {**read(output / "preflight.json"), "sparse_vs_dense_moe": check})
        if args.stage == "preflight":
            return
        selection = R.select_heads(model, data_dir, output, config, roles, binding)
        if args.stage == "scan":
            return
        Q.extract(model, data_dir, output, config, roles, binding, selection, ("fit", "dev"), pad_id)
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
        Q.extract(model, data_dir, output, config, roles, binding, selection, ("test",), pad_id)
        require(sha(output / "checkpoint.npz") == frozen_sha, "Checkpoint changed after TEST release")
        trainer.evaluate(output / "features", output, "test")
        write(output / "runtime-final.json", {"backbone_audit": backend.audit_model(model), "gpu_memory": backend.memory_report(),
                                               "checkpoint_sha256": frozen_sha, "finished_unix": time.time()})
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
