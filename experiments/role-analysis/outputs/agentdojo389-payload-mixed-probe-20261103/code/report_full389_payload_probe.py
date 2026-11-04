"""Aggregate and independently audit new first-payload probability exports.

All payload tokens are retained, with goal ownership at overlapping boundaries.
No ordinary tool/input/generated hidden states exist in the source packet.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import time

import numpy as np

import synthetic_agent_classifier as classifier

ROLES = classifier.ROLES
LATEST_SHA = "fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33"
GPU_UUID = "GPU-94adac15-2f77-5460-4ffa-0a7eee70d46a"
PARTS = ("template", "goal", "full_payload")
sha = classifier._sha
write = classifier._atomic_json


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    require(sha(args.probe) == LATEST_SHA, "Not the latest mixed-source probe")
    probe = classifier.load_probe(args.probe)
    freeze_root = args.probe.parent / "run"
    freeze = read(freeze_root / "freeze.json")
    require(freeze["probe_sha256"] == LATEST_SHA and freeze["selection_split"] == "dev", "Latest probe freeze differs")
    for field, name in (("selection_sha256", "selection.json"), ("training_sha256", "training.json")):
        require(sha(freeze_root / name) == freeze[field], "Frozen decision changed")
    manifest = read(root / "rescore_manifest.json")
    require(manifest["complete"] and manifest["probe_sha256"] == LATEST_SHA, "Rescore incomplete or wrong probe")
    source = Path(manifest["source_run_root"])
    sealed = read(source / "result_manifest.json")
    require(sealed["status"] == "complete_with_initial_errors", "Unexpected original packet state")
    inventory = {row["path"]: row for row in sealed["files"]}
    first_path = source / "analysis/first_occurrences.json"
    require(sha(first_path) == inventory["analysis/first_occurrences.json"]["sha256"], "Original first-occurrence inventory changed")
    first = {row["case_id"]: row for row in read(first_path)}
    require(len(first) == len(manifest["records"]) == 755, "Expected all 755 first-payload episodes")
    schedule = read(source / "schedule.json")
    require(sha(source / "schedule.json") == inventory["schedule.json"]["sha256"], "Schedule changed")
    require(len(schedule) == 778, "Expected 389 episodes per condition")
    original_rows = {}
    with (root / "per_case_probabilities.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            original_rows[(row["case_id"], row["part"])] = row
    # Probe implementations are independent: production export is NumPy/SciPy;
    # audit reconstructs raw BF16 bits and uses this repository's PyTorch net.
    import torch
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID, "Pin the local 2080 Ti")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and "2080 Ti" in torch.cuda.get_device_name(0), "2080 Ti unavailable")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = probe.torch("cuda:0")
    mean, scale = (torch.as_tensor(value, device="cuda:0") for value in (probe.mean, probe.scale))
    strata = defaultdict(list)
    for item in first.values():
        strata[(item["condition"], item["suite"])].append(item)
    audit_ids = set()
    for items in strata.values():
        items = sorted(items, key=lambda r: r["case_id"])
        for item in (items[0], items[-1], max(items, key=lambda r: len(r["input_token_indices"]))):
            audit_ids.add(item["case_id"])
    sums, counts, case_means = defaultdict(lambda: np.zeros(6)), Counter(), defaultdict(list)
    per_case, verified, gpu_audits, seen = [], {}, [], set()
    token_fields = ["condition", "suite", "case_id", "call_id", "part", "input_token_index", "token_id", "token_piece",
                    "char_start", "char_end", *("p_" + role for role in ROLES)]
    with gzip.open(root / "token_probabilities.csv.gz", "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=token_fields)
        writer.writeheader()
        for number, record in enumerate(manifest["records"], 1):
            case_id = record["case_id"]
            require(case_id in first and case_id not in seen, "Unexpected or repeated payload episode")
            seen.add(case_id)
            item = first[case_id]
            condition, suite = item["condition"], item["suite"]
            archive = root / "probabilities" / case_id / "payload_probabilities.npz"
            require(sha(archive) == record["probabilities_artifact"]["sha256"], "New prediction bytes changed")
            meta = read(archive.with_suffix(".json"))
            require(meta["probe_sha256"] == LATEST_SHA and meta["roles"] == list(ROLES), "New prediction identity differs")
            request_key = f"calls/{item['call_id']}/request.json"
            input_key = f"calls/{item['call_id']}/input.npz"
            require(meta["source_request"]["sha256"] == inventory[request_key]["sha256"], "Unsealed source request")
            require(meta["source_input_npz_sha256"] == inventory[input_key]["sha256"], "Unsealed full-input identity")
            verified[request_key] = inventory[request_key]["sha256"]
            verified[input_key] = inventory[input_key]["sha256"]
            for reference in record["source_payload_artifacts"]:
                path = Path(reference["path"])
                if path.is_absolute():
                    # Original remote prefixes are retained in native requests.
                    suffix = Path("calls") / item["call_id"] / path.name
                    key = suffix.as_posix()
                else:
                    key = path.as_posix()
                require(inventory[key]["sha256"] == reference["sha256"], "Cache is not bound to the sealed result")
                verified[key] = reference["sha256"]
            with np.load(archive, allow_pickle=False) as z:
                p, indices, ids, offsets = (z[k] for k in ("probabilities", "input_token_indices", "input_ids", "token_char_offsets"))
                require(z["roles"].tolist() == list(ROLES) and z["head_ids"].tolist() == probe.metadata["heads"], "Prediction layout differs")
            require(p.shape == (len(indices), 6) and p.dtype == np.float32 and np.isfinite(p).all(), "Invalid probability matrix")
            require(np.max(abs(p.sum(1) - 1)) < 3e-5 and (p >= 0).all(), "Invalid probabilities")
            require(indices.tolist() == item["input_token_indices"] == meta["input_token_indices"], "First payload positions changed")
            a, b = meta["goal_char_span"]
            goal = (offsets[:, 1] > a) & (offsets[:, 0] < b)
            require(indices[goal].tolist() == item["parts"]["goal"] == meta["goal_token_indices"], "Goal boundary differs")
            require(indices[~goal].tolist() == item["parts"]["payload_without_goal"] == meta["stem_token_indices"], "Template boundary differs")
            with np.load(source / input_key, allow_pickle=False) as z:
                pieces = z["token_pieces"][indices]
            for part, mask in (("template", ~goal), ("goal", goal), ("full_payload", np.ones(len(goal), bool))):
                require(mask.any(), "Empty payload segment")
                values, n = p[mask].astype(np.float64), int(mask.sum())
                average = values.mean(0)
                producer_part = "payload_without_goal" if part == "template" else part
                original = original_rows[(case_id, producer_part)]
                require(n == int(original["tokens"]) and np.max(abs(average - [float(original["p_" + r]) for r in ROLES])) < 1e-12,
                        "Independent per-episode aggregation differs")
                per_case.append({"condition": condition, "suite": suite, "case_id": case_id, "part": part, "tokens": n,
                                 **dict(zip(("p_" + r for r in ROLES), map(float, average)))})
                for group in (condition, "combined"):
                    key = (group, part)
                    sums[key] += values.sum(0)
                    counts[key] += n
                    case_means[key].append(average)
            for i in range(len(indices)):
                writer.writerow({"condition": condition, "suite": suite, "case_id": case_id, "call_id": item["call_id"],
                                 "part": "goal" if goal[i] else "template", "input_token_index": int(indices[i]),
                                 "token_id": int(ids[i]), "token_piece": str(pieces[i]),
                                 "char_start": int(offsets[i, 0]), "char_end": int(offsets[i, 1]),
                                 **dict(zip(("p_" + r for r in ROLES), map(float, p[i])))})
            if case_id in audit_ids:
                raw_path = source / "calls" / item["call_id"] / "payload_all_heads.npz"
                with np.load(raw_path, allow_pickle=False) as z:
                    bits = z["native_bf16_bits"]
                selected_bits = bits.reshape(len(bits), 2304, 64)[:, probe.metadata["heads"], :]
                x = (selected_bits.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float16).reshape(len(bits), 1024)
                with torch.inference_mode():
                    gpu = net((torch.as_tensor(x.astype(np.float32), device="cuda:0") - mean) / scale).softmax(1).cpu().numpy()
                error = float(np.max(abs(gpu - p)))
                disagreements = int(np.count_nonzero(gpu.argmax(1) != p.argmax(1)))
                require(error < 5e-4 and disagreements == 0, "Independent GPU readout disagrees")
                gpu_audits.append({"case_id": case_id, "tokens": len(p), "max_absolute_error": error, "argmax_disagreements": disagreements})
            if number % 100 == 0:
                print(json.dumps({"stage": "audit_aggregate", "episodes": number, "total": 755}), flush=True)
    require(seen == set(first) and sum(counts[(c, "full_payload")] for c in ("plain", "prerequisite")) == 39176, "Changed payload coverage")
    aggregate, summary = [], {}
    for condition in ("plain", "prerequisite", "combined"):
        summary[condition] = {}
        for part in PARTS:
            key = (condition, part)
            pooled, equal = sums[key] / counts[key], np.mean(case_means[key], axis=0)
            summary[condition][part] = {"episodes": len(case_means[key]), "tokens": counts[key],
                                        "token_pooled_mean": dict(zip(ROLES, map(float, pooled))),
                                        "episode_equal_mean": dict(zip(ROLES, map(float, equal)))}
            for aggregation, values in (("token_pooled_mean", pooled), ("episode_equal_mean", equal)):
                aggregate.append({"condition": condition, "part": part, "aggregation": aggregation,
                                  "episodes": len(case_means[key]), "tokens": counts[key],
                                  **dict(zip(("p_" + r for r in ROLES), map(float, values)))})
    csv_write(root / "mean_probabilities.csv", aggregate)
    csv_write(root / "per_episode_segment_probabilities.csv", per_case)
    coverage = {}
    for condition in ("plain", "prerequisite"):
        scheduled = {r["case_id"] for r in schedule if r["condition"] == condition}
        observed = {key for key in seen if first[key]["condition"] == condition}
        require(len(scheduled) == 389 and observed <= scheduled, "Condition coverage differs")
        coverage[condition] = {"scheduled": len(scheduled), "with_saved_first_payload": len(observed),
                               "without_payload": len(scheduled - observed), "missing_case_ids": sorted(scheduled - observed)}
    validation = {"status": "passed", "source_result_manifest_sha256": sha(source / "result_manifest.json"),
                  "sealed_source_bindings": verified, "all_new_probability_files_verified": True,
                  "all_episode_means_independently_recomputed": True, "all_boundaries_verified_from_original_offsets": True,
                  "gpu": torch.cuda.get_device_name(0), "gpu_uuid": GPU_UUID,
                  "independent_gpu_episodes": len(gpu_audits), "independent_gpu_tokens": sum(r["tokens"] for r in gpu_audits),
                  "max_numpy_gpu_probability_error": max(r["max_absolute_error"] for r in gpu_audits),
                  "argmax_disagreements": sum(r["argmax_disagreements"] for r in gpu_audits), "gpu_checks": gpu_audits}
    write(root / "validation.json", validation)
    result = {"status": "complete_for_existing_payload_features", "created_unix": time.time(),
              "probe_sha256": LATEST_SHA, "probe": str(args.probe.resolve()), "candidate": probe.metadata["candidate_id"],
              "checkpoint_epoch": probe.metadata["best_epoch"], "roles": list(ROLES), "coverage": coverage, "summary": summary,
              "primary_aggregation": "token_pooled_mean", "first_payload_only": True,
              "token_policy": "Keep all originally captured payload tokens; tokens overlapping the template/goal boundary belong to goal",
              "nonpayload_tool_response": {"status": "not_computed", "reason": "No cached hidden states; deferred at user request", "probabilities": None},
              "backbone_forward_performed": False, "probe_retrained": False, "whole_trajectory_scored": False}
    write(root / "summary.json", result)
    lines = ["# 最新混合训练 probe：AgentDojo 双 payload 已存特征预测", "",
             f"使用 `{probe.metadata['candidate_id']}`，第 {probe.metadata['best_epoch']} 轮 checkpoint，SHA256 `{LATEST_SHA}`。", "",
             "本次按用户后续要求先完成已有 payload 特征。所有已存首次完整 payload token 都重新预测；没有重新运行 120B 或修改原始轨迹。", "",
             "| 条件 | 原 episode 数 | 有首次 payload 特征 | 未暴露 payload |", "|---|---:|---:|---:|"]
    for condition, counts_ in coverage.items():
        lines.append(f"| {condition} | {counts_['scheduled']} | {counts_['with_saved_first_payload']} | {counts_['without_payload']} |")
    lines += ["", "主结果为 token 合并平均：各部分六类概率逐 token 求和后除以该部分总 token 数。表内概率采用 0–1 刻度，较小值以科学计数法显示。另在 CSV/JSON 中保留先在 episode 内平均、再对 episode 等权平均的结果。", "",
              "| 条件 | 部分 | token 数 | system | developer | user | cot | assistant | tool |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in ("plain", "prerequisite", "combined"):
        for part in ("template", "goal"):
            value = summary[condition][part]
            values = [value["token_pooled_mean"][r] for r in ROLES]
            lines.append(f"| {condition} | {part} | {value['tokens']:,} | " + " | ".join(f"{p:.8f}" if p >= 1e-4 else f"{p:.3e}" for p in values) + " |")
    lines += ["", "template 指 payload 中除 goal 外的全部 token；与边界相交的 token 归入 goal。没有排除词或重新对片段分词。未暴露 payload 的 23 个 episode 不填零，原生异常但特征有效的 episode 保留。", "",
              "payload 所在 tool response 的其他 token 没有保存 hidden states，当前没有最新 probe 的对应概率。本报告不混入旧 probe 的历史概率。", "",
              f"独立验证通过：全部 755 个 episode 的分段和均值逐项复算；另在本地 2080 Ti 上从 BF16 位模式独立重建 {len(gpu_audits)} 个分层样本的特征并用 PyTorch 前向，最大概率差 {validation['max_numpy_gpu_probability_error']:.3g}，argmax 差异为 0。", "",
              "- `summary.json` / `mean_probabilities.csv`：汇总及两种平均口径。",
              "- `token_probabilities.csv.gz`：39,176 个 payload token 的六类概率、原始位置、文本及所属部分。",
              "- `probabilities/`：逐 episode 的全部 payload token 预测 NPZ 和来源元数据。",
              "- `per_episode_segment_probabilities.csv`：每个 episode 的分段平均。",
              "- `validation.json`：封存来源核验、分段和独立 GPU 验证。"]
    (root / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n")
    code = root / "code"
    code.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(classifier.__file__)):
        shutil.copy2(path, code / path.name)
    external = Path("/home/wentian/Github/SPAR/delimeter-token")
    for relative in ("experiments/ChatInject/role_probe_full_payload/rescore.py", "experiments/ChatInject/role_probe_full_payload/capture.py",
                     "experiments/ChatInject/role_probe/probe.py"):
        destination = code / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(external / relative, destination)
    reproduce = ["#!/usr/bin/env bash", "set -euo pipefail", 'task_output="${1:?Provide a NEW output directory}"',
                 f"cd {shlex.quote(str(external))}", "export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8",
                 "/tmp/agentdojo-payload-test-venv/bin/python -m experiments.ChatInject.role_probe_full_payload.rescore "
                 f"--run-root {shlex.quote(str(source))} --probe {shlex.quote(str(args.probe.resolve()))} --expected-probe-sha256 {LATEST_SHA} --output \"$task_output\"",
                 f"export CUDA_VISIBLE_DEVICES={GPU_UUID}", f"{shlex.quote(os.sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} --probe {shlex.quote(str(args.probe.resolve()))} --output \"$task_output\""]
    (root / "reproduce.sh").write_text("\n".join(reproduce) + "\n")
    (root / "reproduce.sh").chmod(0o755)
    print(json.dumps({"status": result["status"], "coverage": coverage, "summary": summary,
                      "gpu_max_error": validation["max_numpy_gpu_probability_error"]}), flush=True)


if __name__ == "__main__":
    main()
