"""Retrain a six-role probe using only v7 single-message cached features.

No backbone is loaded. Source backups are read only. FIT, DEV and TEST are
filtered independently before training; AgentDojo is read only after freezing.
The ordered 16 heads are inherited from v7, whose head discovery used mixed
pilot FIT contexts. This is a single-message readout ablation, not a new
single-message-only head-discovery experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np

import synthetic_agent_classifier as classifier
import synthetic_role_warmstart_eval as cached

ROLES = classifier.ROLES
ARRAYS = ("x", "y", "structure", "source", "family", "weight", "groups")
GPU_UUID = "GPU-94adac15-2f77-5460-4ffa-0a7eee70d46a"
HEAD_LIMITATION = (
    "Reuse the original ordered 16 v7 heads, discovered using mixed pilot FIT. "
    "Only the probe weights and FIT standardizer are retrained using single "
    "messages. No historical probe weights or standardizer are inherited."
)
sha = classifier._sha
write = classifier._atomic_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest_object(value):
    return hashlib.sha256(classifier._json(value).encode()).hexdigest()


class TaskClusterRows(SimpleNamespace):
    def __len__(self):
        return len(self.y)


def input_selection(path, split):
    """Independently reconstruct eligible feature rows from actual inputs."""
    selected, expected = [], {key: [] for key in ("y", "structure", "source", "family", "weight")}
    records = sources = 0
    offset = 0
    corpus = Counter()
    clusters, source_ids = set(), set()
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            require(record["split"] == split, "Input split mismatch")
            targets = record["targets"]
            if record["profile"] == "single":
                spans = record["message_spans"]
                require(len(spans) == 1 and record["text"].count("<|start|>") == 1,
                        "Single profile contains more than one native message")
                require(spans[0]["supervised"] and not spans[0]["tool_definition"],
                        "Single message has no article-body supervision")
                require(not record["round_segments"], "Single message contains round segments")
                require(record["source"] in ("c4", "dolma3"), "Unexpected corpus")
                require(bool(targets), "Empty single-message supervision")
                records += 1
                clusters.add(record["cluster_id"])
                source_ids.add(record["source_id"])
                corpus[record["source"]] += len(targets)
                selected.extend(range(offset, offset + len(targets)))
                for target in targets:
                    require(target["structure"].startswith("single:"), "Non-single target")
                    require(target["role"] == spans[0]["role"] == ROLES[target["label"]],
                            "Single target disagrees with native enclosing message role")
                    require(target["message_id"] == spans[0]["message_id"], "Target message mismatch")
                    expected["y"].append(target["label"])
                    expected["structure"].append(target["structure"])
                    expected["source"].append(record["cluster_id"])
                    expected["family"].append(target["family_id"])
                    expected["weight"].append(target.get("weight", 1))
            offset += len(targets)
    return np.asarray(selected, np.int64), expected, {
        "single_message_records": records, "sources": len(source_ids),
        "source_clusters": len(clusters), "rows_by_corpus": dict(corpus),
        "all_original_rows": offset, "input_sha256": sha(path),
    }


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / "protocol.json").exists(), "Preparation already exists; use another output directory")
    (root / ".gitignore").write_text("features/\n*.log\nagentdojo/probabilities/\n")
    source = args.features.resolve()
    original = read(source / "manifest.json")
    source_sha = sha(source / "manifest.json")
    protocol = {
        "format_version": "single-message-readout-ablation-v1", "created_unix": time.time(),
        "source_feature_root": str(source), "source_feature_manifest_sha256": source_sha,
        "source_data_root": str(args.data.resolve()), "selection_rule": "structure starts with single: AND input profile == single",
        "split_policy": "Preserve original v7 FIT/DEV/TEST; no resplitting or source reassignment",
        "standardizer": "Recompute family-weighted mean/std on single-message FIT only",
        "training": "Randomly initialize all seven original linear/MLP candidates; select only on single-message DEV",
        "agentdojo": "Post-freeze readout on sealed no-injection 57-task first-position normal-body masks",
        "head_discovery_limitation": HEAD_LIMITATION, "heads": original["heads"],
        "roles": list(ROLES), "feature_dim": 1024,
        "device_uuid": GPU_UUID, "implementation_sha256": sha(Path(__file__)),
        "training_implementation_sha256": sha(Path(classifier.__file__)),
    }
    features = root / "features"
    features.mkdir(exist_ok=True)
    derived = copy.deepcopy(original)
    derived["splits"], derived["progress"] = {}, {}
    derived["derivation"] = {"rule": protocol["selection_rule"], "source_manifest_sha256": source_sha,
                             "source_root": str(source), "new_backbone_forward": False}
    derived["signature"]["derivation"] = derived["derivation"]
    derived["data_hash"] = digest_object({"source_data_hash": original["data_hash"], **derived["derivation"]})
    derived["feature_root"] = str(features)
    audit, all_sources = {}, {}
    for split in ("fit", "dev", "test"):
        input_path = args.data / f"inputs-{split}.jsonl"
        eligible, expected, summary = input_selection(input_path, split)
        require(summary["input_sha256"] == original["split_sources"][split], "Input does not match extracted cache")
        signature = {**derived["signature"], "split": split, "input_sha256": original["split_sources"][split]}
        entries, actual_indices = [], []
        global_start = accepted = 0
        role_counts, structures, source_files = Counter(), set(), {}
        for entry in original["splits"][split]:
            folder = source / entry["path"]
            require(sha(folder / "complete.json") == entry["sha256"], "Original shard commit changed")
            commit = read(folder / "complete.json")
            source_files[entry["path"] + "/complete.json"] = entry["sha256"]
            arrays = {}
            for key in ARRAYS:
                path = folder / (key + ".npy")
                expected_sha = commit["files"][path.name]
                if isinstance(expected_sha, dict):
                    expected_sha = expected_sha["sha256"]
                require(sha(path) == expected_sha, "Original array changed: " + str(path))
                arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
            n = entry["rows"]
            require(arrays["x"].shape == (n, 1024) and arrays["x"].dtype == np.float16, "Bad original geometry")
            mask = np.char.startswith(arrays["structure"], "single:")
            local = np.flatnonzero(mask)
            actual_indices.extend((global_start + local).tolist())
            global_start += n
            if not len(local):
                continue
            take = {key: np.array(value[local]) for key, value in arrays.items()}
            end = accepted + len(local)
            for key in expected:
                require(np.array_equal(take[key], np.asarray(expected[key][accepted:end], dtype=take[key].dtype)),
                        f"Actual input target/cache alignment differs: {split}/{entry['path']}/{key}")
            require(np.isfinite(take["x"]).all(), "Nonfinite single features")
            for group in np.unique(take["groups"]):
                meta = json.loads(str(group))
                require(meta["profile"] == "single" and meta["rounds"] == 0, "Mixed context entered subset")
            name = f"{split}-{len(entries):06d}"
            destination = features / name
            destination.mkdir()
            for key, value in take.items():
                np.save(destination / (key + ".npy"), value, allow_pickle=False)
            np.save(destination / "original_row_indices.npy", global_start - n + local, allow_pickle=False)
            files = {p.name: sha(p) for p in destination.glob("*.npy")}
            write(destination / "complete.json", {
                "format_version": "synthetic-agent-feature-shard-v1", "rows": len(local),
                "signature": signature, "files": files, "derived_not_reextracted": True,
                "original_shard": str(folder), "original_complete_sha256": entry["sha256"],
            })
            entries.append({"path": name, "rows": len(local), "sha256": sha(destination / "complete.json")})
            accepted = end
            role_counts.update(map(int, take["y"]))
            structures.update(map(str, take["structure"]))
        require(global_start == summary["all_original_rows"], "Original inputs/cache total differs")
        require(np.array_equal(np.asarray(actual_indices, np.int64), eligible), "Input/cache single-row selections differ")
        require(accepted == len(eligible) > 0 and set(role_counts) == set(range(6)), "Missing eligible rows/roles")
        all_sources[split] = set(expected["source"])
        derived["splits"][split] = entries
        derived["progress"][split] = {"complete": True, "records": summary["single_message_records"], "rows": accepted}
        audit[split] = {**summary, "rows": accepted, "rows_by_role": {ROLES[k]: role_counts[k] for k in range(6)},
                        "structures": sorted(structures), "original_shard_commits": source_files,
                        "selected_original_row_indices_sha256": hashlib.sha256(eligible.tobytes()).hexdigest(),
                        "input_cache_alignment": "all selected labels/structures/sources/families/weights match actual input targets"}
        print(json.dumps({"stage": "prepared", "split": split, "rows": accepted, "sources": summary["sources"]}), flush=True)
    overlaps = {a + "/" + b: len(all_sources[a] & all_sources[b])
                for a, b in (("fit", "dev"), ("fit", "test"), ("dev", "test"))}
    require(not any(overlaps.values()), "Single-message splits overlap by source cluster")
    write(features / "manifest.json", derived)
    protocol["derived_feature_manifest_sha256"] = sha(features / "manifest.json")
    write(root / "protocol.json", protocol)
    write(root / "data-audit.json", {"status": "passed", "splits": audit, "cross_split_source_overlap": overlaps,
                                    "only_single_message": True, "head_discovery_limitation": HEAD_LIMITATION})
    config = {"phase": "single-message", "classifier": {
        "device": "cuda:0", "batch_size": 4096, "learning_rate": .001, "weight_decay": .001,
        "dropout": .1, "gradient_clip": 5., "max_epochs": 45, "min_epochs": 10, "patience": 8,
        "checkpoint_steps": 100, "bootstrap_replicates": 2000,
        "profile_weights": {"single": 1., "one_round": 0., "multi_round": 0.}}}
    write(root / "config.json", config)
    code = root / "code"
    code.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(classifier.__file__), Path(cached.__file__)):
        shutil.copy2(path, code / path.name)


def cuda_ready(output):
    import torch
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID, "Pin CUDA_VISIBLE_DEVICES to the requested 2080 Ti UUID")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one visible CUDA GPU")
    require("2080 Ti" in torch.cuda.get_device_name(0), "Training must use the requested 2080 Ti")
    torch.set_num_threads(8)
    return {"name": torch.cuda.get_device_name(0), "uuid": GPU_UUID, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
            "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)}


def train(args):
    import torch
    root = args.output.resolve()
    protocol = read(root / "protocol.json")
    require(sha(root / "features/manifest.json") == protocol["derived_feature_manifest_sha256"], "Derived data changed")
    resource = cuda_ready(root)
    resource["started_unix"] = time.time()
    write(root / "gpu-start.json", resource)
    run = root / "run"
    classifier.train_candidates(root / "features", run, read(root / "config.json"), resume=args.resume)
    classifier.freeze(run)
    resource.update(finished_unix=time.time(), max_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    max_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    write(root / "gpu-training.json", resource)
    for split in ("fit", "dev", "test"):
        result = classifier.evaluate(root / "features", run, split, device="cuda:0", bootstrap_replicates=2000)
        print(json.dumps({"stage": "single_evaluation", "split": split, **result["overall"]}), flush=True)


def evaluate_agentdojo(args):
    import torch
    root = args.output.resolve()
    cuda_ready(root)
    output = root / "agentdojo"
    require(not (output / "metrics.json").exists(), "AgentDojo evaluation already exists")
    output.mkdir(exist_ok=True)
    freeze = read(root / "run/freeze.json")
    require(freeze["selection_split"] == "dev" and freeze["probe_sha256"] == sha(root / "run/best.npz"), "Freeze mismatch")
    require(freeze["selection_sha256"] == sha(root / "run/selection.json"), "Frozen DEV decision changed")
    probe = classifier.load_probe(root / "run/best.npz")
    parent = classifier.load_probe(args.parent)
    for key in cached.FEATURE_KEYS:
        require(probe.metadata[key] == parent.metadata[key], "AgentDojo feature binding differs: " + key)
    net = probe.torch("cuda:0")
    mean = torch.as_tensor(probe.mean, device="cuda:0")
    scale = torch.as_tensor(probe.scale, device="cuda:0")
    source_root = args.agentdojo.resolve()
    source = cached.SourceFiles(source_root)
    provenance = read(source.verify(source_root / "probe_provenance.json"))
    require(provenance["sha256"] == sha(args.parent), "Wrong archival probe")
    rows, masks = cached.load_masks(source.verify(source_root / "analysis/call_body_probabilities.csv"))
    original_metrics = read(source.verify(source_root / "analysis/metrics.json"))
    calls = sorted((source_root / "calls").glob("*/*/*/call_*/request.json"))
    require(len(calls) == original_metrics["calls"], "Call inventory changed")
    y_blocks, new_blocks, baseline_blocks = [], [], []
    metadata = {key: [] for key in ("task", "scope", "tag", "suite")}
    counters = Counter()
    selected_counts = Counter()
    templates = {"system": set(), "developer": set()}
    seen, per_body = set(), []
    for call_number, request_path in enumerate(calls, 1):
        request = read(source.verify(request_path))
        call_id = request_path.parent.relative_to(source_root / "calls").as_posix()
        require(request["status"] == "complete" and request["probe_sha256"] == provenance["sha256"], "Invalid source request")
        for scope, filename in zip(cached.SCOPES, ("input.npz", "generation_probe.npz")):
            artifact = source.verify(request_path.parent / filename)
            with np.load(artifact, allow_pickle=False) as z:
                arrays = {key: z[key] for key in z.files}
            observed = cached.validate_archive(arrays, request, scope, parent)
            counters[scope + "/observed"] += int(observed.sum())
            counters[scope + "/stored"] += len(observed)
            for row in masks.get((call_id, scope), []):
                seen.add((call_id, scope))
                indices = row["indices"]
                require(observed[indices].all(), "Scoring unobserved stop tokens")
                x = arrays["head_features"][indices]
                with torch.inference_mode():
                    pieces = [net((torch.as_tensor(x[i:i+8192].astype(np.float32), device="cuda:0") - mean) / scale).softmax(1).cpu().numpy()
                              for i in range(0, len(x), 8192)]
                p = np.concatenate(pieces) if pieces else np.empty((0, 6), np.float32)
                old = arrays["probabilities"][indices]
                reference_sums = np.array([float(row["sum_" + role]) for role in ROLES])
                require(np.max(abs(old.astype(np.float64).sum(0) - reference_sums), initial=0) <= 1e-9,
                        "Original first-position body mask changed")
                y = np.full(len(indices), row["label"], np.int64)
                y_blocks.append(y)
                new_blocks.append(p)
                baseline_blocks.append(old)
                selected_counts[scope + "/" + row["tag"]] += len(indices)
                for key, value in (("task", row["task_key"]), ("scope", scope), ("tag", row["tag"]),
                                   ("suite", row["task_key"].split("/")[0])):
                    metadata[key].extend([value] * len(y))
                if scope == "actual_input" and row["tag"] in templates:
                    a, b = int(row["body_char_start"]), int(row["body_char_end"])
                    templates[row["tag"]].add(hashlib.sha256(request["rendered_prompt"][a:b].encode()).hexdigest())
                per_body.append({"task": row["task_key"], "call_id": call_id, "scope": scope, "role": row["tag"],
                                 "tokens": len(y), "accuracy": float((p.argmax(1) == y).mean()) if len(y) else None})
        if call_number % 50 == 0:
            print(json.dumps({"stage": "agentdojo", "calls": call_number, "total": len(calls)}), flush=True)
    require(set(masks) == seen, "Missing cached first-position masks")
    require(dict(selected_counts) == original_metrics["retained_normal_positions"], "Changed body denominators")
    y, p, baseline = np.concatenate(y_blocks), np.concatenate(new_blocks), np.concatenate(baseline_blocks)
    metadata = {key: np.asarray(value) for key, value in metadata.items()}
    require(len(np.unique(metadata["task"])) == 57 and len(y) == 155103, "Unexpected 57-task scoring inventory")
    np.savez_compressed(output / "first-position-probabilities.npz", labels=y, probabilities=p, archived_baseline=baseline, **metadata)
    report = {"overall": cached.metric_report(y, p), "archived_mixed_probe_baseline": cached.metric_report(y, baseline),
              "by_scope": {}, "by_suite": {}, "task_count": 57, "calls": len(calls),
              "probe_sha256": freeze["probe_sha256"], "selection_split": "single-message DEV only",
              "metric_policy": "Sealed first-position normal article/message body masks; no repeated context bodies or tool-call JSON",
              "agent_rerun": False, "backbone_forward": False, "gpu_readout": True,
              "head_discovery_limitation": HEAD_LIMITATION,
              "unique_sd_templates": {key: len(values) for key, values in templates.items()}}
    for key in ("scope", "suite"):
        for value in np.unique(metadata[key]):
            m = metadata[key] == value
            report["by_" + key][str(value)] = cached.metric_report(y[m], p[m])
    report["confidence_intervals"] = classifier.bootstrap_clusters(TaskClusterRows(source=metadata["task"], y=y), p, replicates=2000)
    report["confidence_intervals"]["unit"] = "unique_agentdojo_task"
    cached.write_csv(output / "per-body.csv", per_body)
    per_task, summary = cached.probability_tables(y, p, metadata["scope"], metadata["tag"], metadata["task"])
    cached.write_csv(output / "per-task-probabilities.csv", per_task)
    cached.write_csv(output / "mean-probabilities.csv", summary)
    write(output / "cache-audit.json", {"status": "passed", "consumed_source_files": source.consumed,
                                       "source_manifest_sha256": source.manifest_sha256,
                                       "counters": dict(counters), "retained_normal_positions": dict(selected_counts),
                                       "retained_tokens": len(y), "selection_not_changed_after_agentdojo": True})
    write(output / "metrics.json", report)
    print(json.dumps({"stage": "agentdojo_complete", **report["overall"]}), flush=True)


def plot_confusion(axis, metric, title):
    cm = np.asarray(metric["confusion"])
    fractions = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    axis.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
    axis.set(xticks=range(6), yticks=range(6), xticklabels=ROLES, yticklabels=ROLES,
             xlabel="Predicted role", ylabel="Actual role", title=title)
    axis.tick_params(axis="x", rotation=40)
    for i in range(6):
        for j in range(6):
            axis.text(j, i, f"{100*fractions[i,j]:.1f}", ha="center", va="center", fontsize=8,
                      color="white" if fractions[i,j] > .5 else "black")


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = args.output.resolve()
    audit = read(root / "data-audit.json")
    selection = read(root / "run/selection.json")
    winner = next(item for item in selection["candidate_results"] if item["candidate_id"] == selection["candidate_id"])
    gpu = read(root / "gpu-training.json")
    evaluations = {split: read(root / f"run/evaluations/{split}.json") for split in ("fit", "dev", "test")}
    agent = read(root / "agentdojo/metrics.json")
    figures = root / "plots"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    plot_confusion(axes[0], evaluations["test"]["overall"], "Single-message article TEST")
    plot_confusion(axes[1], agent["overall"], "AgentDojo no-injection 57 tasks")
    fig.suptitle("C4 + Dolma3 single-message role probe (row percentages)")
    for ext in ("png", "pdf"):
        fig.savefig(figures / ("test-and-agentdojo-confusion." + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for candidate in selection["candidate_results"]:
        history = read(root / "run/candidates" / candidate["candidate_id"] / "history.json")
        epochs = [h.get("epoch", i+1) for i, h in enumerate(history)]
        axes[0].plot(epochs, [h["train_loss_for_processed_batches"] for h in history], label=candidate["candidate_id"])
        axes[1].plot(epochs, [h["dev"]["overall"]["macro_f1"] for h in history], label=candidate["candidate_id"])
    axes[0].set(xlabel="Epoch", ylabel="Sampled training cross entropy")
    axes[1].set(xlabel="Epoch", ylabel="Single-message DEV macro-F1")
    axes[1].legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(figures / ("training-curves." + ext), dpi=180)
    plt.close(fig)
    text = ["# C4＋Dolma3 单消息六角色 probe：2080 Ti 本地训练", "",
            f"选中候选：`{selection['candidate_id']}`，最佳 epoch {winner['best_epoch']}，实际训练 {winner['epochs']} 轮。只按单消息 DEV 选择，之后冻结，再评分 TEST 和 AgentDojo。", "",
            "原 v7 的单轮、多轮样本不进入本次 FIT、DEV、TEST；标准化器只由单消息 FIT 拟合，分类器随机初始化。", "",
            "| 划分 | token 位置 | 来源 | accuracy | macro-F1 |", "|---|---:|---:|---:|---:|"]
    for split, value in evaluations.items():
        metric = value["overall"]
        text.append(f"| {split.upper()} 单消息 | {metric['tokens']:,} | {audit['splits'][split]['sources']:,} | {metric['accuracy']:.4%} | {metric['macro_f1']:.4%} |")
    text.extend(["", "| AgentDojo 57 任务 | accuracy | macro-F1 |", "|---|---:|---:|",
                 f"| 本次单消息 probe | {agent['overall']['accuracy']:.4%} | {agent['overall']['macro_f1']:.4%} |",
                 f"| 原混合训练 probe（缓存基线） | {agent['archived_mixed_probe_baseline']['accuracy']:.4%} | {agent['archived_mixed_probe_baseline']['macro_f1']:.4%} |", "",
                 "| AgentDojo 分组 | token 位置 | accuracy | macro-F1 |", "|---|---:|---:|---:|"])
    for scope in ("actual_input", "actual_generated"):
        metric = agent["by_scope"][scope]
        text.append(f"| {scope} | {metric['tokens']:,} | {metric['accuracy']:.4%} | {metric['macro_f1']:.4%} |")
    text.extend(["", "| 角色 | AgentDojo 支持 token | precision | recall | F1 |", "|---|---:|---:|---:|---:|"])
    for role in ROLES:
        m = agent["overall"]["roles"][role]
        values = ["N/A" if m[key] is None else f"{m[key]:.4%}" for key in ("precision", "recall", "f1")]
        text.append(f"| {role} | {m['support']:,} | " + " | ".join(values) + " |")
    text.extend(["", "![测试与 AgentDojo 混淆矩阵](plots/test-and-agentdojo-confusion.png)", "",
                 "![训练与单消息 DEV 曲线](plots/training-curves.png)", "",
                 "使用原始缓存中 359 次调用、155,103 个首次普通正文位置；沿用原去重复上下文 mask。没有重跑 agent，不能据此推断任务成功率变化。",
                 "57 任务只有 1 种 system 和 3 种 developer 提示模板，属于已观察的历史回归集。", "",
                 "**特征限制：** 本次沿用 v7 的固定 16 头／1024 维缓存；这些头最初由混合上下文 pilot FIT 选出。本次的 probe 参数、标准化器和选模只使用单消息。", "",
                 f"本次在 {gpu['name']} 上依次训练七个候选，训练与选模耗时 {(gpu['finished_unix']-gpu['started_unix'])/60:.2f} 分钟；PyTorch 峰值 allocated 为 {gpu['max_allocated_mib']:.2f} MiB（不含驱动与其他进程）。", "",
                 "独立验证见 `independent-validation.json`：重新计算单消息 FIT 标准化器、核验原始特征向量、独立 NumPy/SciPy forward 与 GPU 概率，以及由保存的预测重算全部混淆矩阵和每角色指标。", "",
                 "数据审计见 `data-audit.json`，设备与显存见 `gpu-training.json`，冻结权重见 `run/best.npz`，逐角色原始指标见 `run/evaluations/test.json` 和 `agentdojo/metrics.json`。运行环境见 `requirements.txt`，复现入口见 `reproduce.sh`。"])
    (root / "REPORT.zh-CN.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "agentdojo", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--agentdojo", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "agentdojo": evaluate_agentdojo, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
