"""Train a fresh six-role readout on article singles plus native AgentDojo bodies.

Split articles by source cluster (stratified by corpus/role), and AgentDojo by
whole task (stratified by suite), before training. Cached probabilities are
integrity references only, never targets or inputs. No backbone is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np

import synthetic_agent_classifier as classifier
import synthetic_role_warmstart_eval as cached
from run_single_message_probe import cuda_ready, input_selection

ROLES = classifier.ROLES
DOMAINS = ("c4_dolma3", "agentdojo")
SPLITS = ("fit", "dev", "test")
ARRAYS = ("x", "y", "structure", "source", "family", "weight", "groups")
BASE = Path(__file__).resolve().parent
BACKUP = Path("/home/wentian/Github/SPAR/runpod-backups/role-retraining-20261026-20261027/workspace-task")
AGENTDOJO = Path("/home/wentian/Github/SPAR/delimeter-token/results/ChatInject/agentdojo_role_probe_gptoss120b/full57_no_payload_first_occurrence_probe_v7_20261027")
PARENT = BASE / "outputs/synthetic-agent-120b-20261026/final-v7/formal/best.npz"
GPU_UUID = "GPU-94adac15-2f77-5460-4ffa-0a7eee70d46a"
sha = classifier._sha
write = classifier._atomic_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def object_sha(value):
    return hashlib.sha256(classifier._json(value).encode()).hexdigest()


def assign_groups(strata, seed):
    """Deterministic ~70/15/15 allocation in each source-only stratum."""
    rng = np.random.default_rng(seed)
    assignment, counts = {}, {}
    for stratum in sorted(strata):
        identifiers = np.asarray(sorted(strata[stratum]))
        identifiers = identifiers[rng.permutation(len(identifiers))]
        n_fit, n_dev = round(.70 * len(identifiers)), round(.15 * len(identifiers))
        require(min(n_fit, n_dev, len(identifiers) - n_fit - n_dev) > 0, "Stratum too small")
        partitions = np.split(identifiers, [n_fit, n_fit + n_dev])
        counts[stratum] = {}
        for split, values in zip(SPLITS, partitions):
            counts[stratum][split] = len(values)
            for value in values:
                require(str(value) not in assignment, "Cluster occurs in more than one stratum")
                assignment[str(value)] = split
    return assignment, counts


def article_pool(args):
    """Verify original shard bytes and independently reconstruct raw input targets."""
    manifest = read(args.features / "manifest.json")
    blocks = defaultdict(list)
    strata = defaultdict(set)
    cluster_stratum = {}
    provenance = {}
    for old_split in SPLITS:
        bank = classifier.FeatureBank(args.features, old_split)
        indices, expected, summary = input_selection(args.data / f"inputs-{old_split}.jsonl", old_split, ("single",))
        require(summary["input_sha256"] == manifest["split_sources"][old_split], "Raw inputs do not match cache")
        actual = np.flatnonzero(np.char.startswith(bank.structure, "single:"))
        require(np.array_equal(indices, actual), "Raw input single-message masks disagree")
        for key, values in expected.items():
            require(np.array_equal(getattr(bank, key)[indices], np.asarray(values)), "Article target mismatch: " + key)
        groups = np.unique(bank.groups[indices])
        require(all(json.loads(str(g))["profile"] == "single" and json.loads(str(g))["rounds"] == 0 for g in groups),
                "Non-single article context entered pool")
        # Each source contributes one single-role, 97-position article message.
        corpora = np.asarray([str(f).split("-", 1)[0] for f in bank.family[indices]])
        require(set(corpora) == {"c4", "dolma3"}, "Unexpected article corpus")
        sources = np.asarray(["article:" + str(s) for s in bank.source[indices]])
        y = bank.y[indices]
        for corpus in ("c4", "dolma3"):
            for role_id, role in enumerate(ROLES):
                key = corpus + "/" + role
                for source in np.unique(sources[(corpora == corpus) & (y == role_id)]):
                    source = str(source)
                    require(source not in cluster_stratum or cluster_stratum[source] == key, "Article stratum inconsistent")
                    cluster_stratum[source] = key
                    strata[key].add(source)
        blocks["x"].append(bank.read(indices).astype(np.float16))
        blocks["y"].append(y)
        blocks["source"].append(sources)
        blocks["family"].append(np.asarray(["article:" + str(v) for v in bank.family[indices]]))
        blocks["corpus"].append(corpora)
        blocks["origin_partition"].append(np.full(len(indices), old_split))
        blocks["origin_row"].append(indices)
        provenance[old_split] = {**summary, "original_split_identity_sha256": bank.split_identity_hash}
        print(json.dumps({"stage": "article_verified", "original_split": old_split, "rows": len(indices)}), flush=True)
    pool = {key: np.concatenate(parts) for key, parts in blocks.items()}
    require(len(pool["y"]) == 546304 and len(np.unique(pool["source"])) == 5632, "Unexpected article inventory")
    require(len(np.unique(pool["family"])) == len(pool["y"]), "Single article position repeated")
    pool["structure"] = np.full(len(pool["y"]), "c4_dolma3:single")
    pool["weight"] = np.ones(len(pool["y"]), np.float32)
    pool["groups"] = np.asarray([classifier._json({"domain": "c4_dolma3", "corpus": str(c), "scope": "single", "suite": "N/A"})
                                  for c in pool["corpus"]])
    assignment, counts = assign_groups(strata, args.seed)
    return pool, assignment, counts, provenance


def agentdojo_pool(args, binding):
    root = args.agentdojo
    source = cached.SourceFiles(root)
    provenance = read(source.verify(root / "probe_provenance.json"))
    parent = classifier.load_probe(args.parent)
    require(provenance["sha256"] == sha(args.parent), "Wrong archival probe identity")
    for key in cached.FEATURE_KEYS:
        require(parent.metadata[key] == binding[key], "Feature binding differs: " + key)
    rows, masks = cached.load_masks(source.verify(root / "analysis/call_body_probabilities.csv"))
    original = read(source.verify(root / "analysis/metrics.json"))
    calls = sorted((root / "calls").glob("*/*/*/call_*/request.json"))
    require(len(calls) == original["calls"] == 359, "Call inventory changed")
    blocks, selected_counts, templates = defaultdict(list), Counter(), defaultdict(set)
    seen = set()
    coordinate_records = []
    for number, request_path in enumerate(calls, 1):
        request = read(source.verify(request_path))
        call_id = request_path.parent.relative_to(root / "calls").as_posix()
        require(request["status"] == "complete" and request["probe_sha256"] == provenance["sha256"], "Incomplete source request")
        for scope, filename in zip(cached.SCOPES, ("input.npz", "generation_probe.npz")):
            with np.load(source.verify(request_path.parent / filename), allow_pickle=False) as z:
                arrays = {key: z[key] for key in z.files}
            observed = cached.validate_archive(arrays, request, scope, parent)
            for row in masks.get((call_id, scope), []):
                seen.add((call_id, scope))
                indices = row["indices"]
                require(observed[indices].all(), "Unforwarded positions in retained mask")
                reference = np.asarray([float(row["sum_" + role]) for role in ROLES])
                require(np.max(abs(arrays["probabilities"][indices].astype(float).sum(0) - reference), initial=0) < 1e-9,
                        "Archived first-position mask changed")
                n = len(indices)
                selected_counts[scope + "/" + row["tag"]] += n
                if not n:
                    continue
                task = row["task_key"]
                # Labels come from native message/channel roles, never old probe predictions.
                require(cached.TAG_ROLE[row["tag"]] == row["label"], "Unknown native role")
                blocks["x"].append(arrays["head_features"][indices])
                blocks["y"].append(np.full(n, row["label"], np.int64))
                blocks["source"].append(np.full(n, "agentdojo:" + task))
                blocks["family"].append(np.asarray([f"agentdojo:{call_id}/{scope}/{int(i)}" for i in indices]))
                blocks["call_id"].append(np.full(n, call_id))
                blocks["scope"].append(np.full(n, scope))
                blocks["token_index"].append(indices)
                blocks["tag"].append(np.full(n, row["tag"]))
                blocks["suite"].append(np.full(n, task.split("/")[0]))
                blocks["groups"].append(np.full(n, classifier._json({"domain": "agentdojo", "corpus": "agentdojo",
                                                                  "scope": scope, "suite": task.split("/")[0]})))
                coordinate_records.append({k: v for k, v in row.items() if k != "indices"})
                if scope == "actual_input" and row["tag"] in ("system", "developer"):
                    a, b = int(row["body_char_start"]), int(row["body_char_end"])
                    templates[row["tag"]].add(hashlib.sha256(request["rendered_prompt"][a:b].encode()).hexdigest())
        if number % 50 == 0:
            print(json.dumps({"stage": "agentdojo_verified", "calls": number, "total": len(calls)}), flush=True)
    require(set(masks) == seen and dict(selected_counts) == original["retained_normal_positions"], "Missing native mask positions")
    pool = {key: np.concatenate(parts) for key, parts in blocks.items()}
    require(len(pool["y"]) == 155103 and len(np.unique(pool["source"])) == 57, "Unexpected AgentDojo inventory")
    require(len(np.unique(pool["family"])) == len(pool["y"]), "Repeated native positions")
    pool["structure"] = np.full(len(pool["y"]), "agentdojo:native")
    pool["weight"] = np.ones(len(pool["y"]), np.float32)
    strata = defaultdict(set)
    for task in np.unique(pool["source"]):
        strata[str(task).split(":", 1)[1].split("/")[0]].add(str(task))
    assignment, counts = assign_groups(strata, args.seed + 1)
    audit = {"status": "passed", "result_manifest_sha256": source.manifest_sha256,
             "consumed_source_files": source.consumed, "retained_normal_positions": dict(selected_counts),
             "calls": len(calls), "tasks": len(assignment), "unique_prompt_templates": {k: len(v) for k, v in templates.items()},
             "labels": "Native message/channel roles; archived probabilities checked for integrity only"}
    return pool, assignment, counts, audit, coordinate_records


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / "protocol.json").exists(), "Preparation already exists; use a fresh output")
    (root / ".gitignore").write_text("features/\n*.log\n")
    original = read(args.features / "manifest.json")
    article, article_assignment, article_counts, article_audit = article_pool(args)
    native, native_assignment, native_counts, native_audit, coordinates = agentdojo_pool(args, original)
    assignments = {"c4_dolma3": article_assignment, "agentdojo": native_assignment}
    write(root / "split-assignment.json", {"seed": args.seed, "ratios": [.7, .15, .15], "domains": assignments,
                                            "strata_counts": {"c4_dolma3": article_counts, "agentdojo": native_counts}})
    write(root / "article-source-audit.json", article_audit)
    write(root / "agentdojo-source-audit.json", native_audit)
    cached.write_csv(root / "agentdojo-retained-bodies.csv", coordinates)
    features = root / "features"
    features.mkdir(exist_ok=True)
    manifest = copy.deepcopy(original)
    derivation = {"experiment": "c4-dolma3-single-plus-agentdojo-native", "article_profile": "single",
                  "article_manifest_sha256": sha(args.features / "manifest.json"),
                  "agentdojo_result_manifest_sha256": native_audit["result_manifest_sha256"],
                  "split_assignment_sha256": sha(root / "split-assignment.json"), "new_backbone_forward": False}
    manifest["derivation"] = derivation
    manifest["signature"]["derivation"] = derivation
    manifest["data_hash"] = object_sha(derivation)
    manifest["feature_root"] = str(features)
    manifest["splits"], manifest["progress"], manifest["split_sources"] = {}, {}, {}
    audit = {"status": "passed", "splits": {}, "totals": {"c4_dolma3": len(article["y"]), "agentdojo": len(native["y"])}}
    source_sets = {}
    for split in SPLITS:
        manifest["split_sources"][split] = object_sha({d: sorted(k for k, v in a.items() if v == split) for d, a in assignments.items()})
        signature = {**manifest["signature"], "split": split, "input_sha256": manifest["split_sources"][split]}
        entries, split_audit, sources = [], {}, set()
        for domain, pool in (("c4_dolma3", article), ("agentdojo", native)):
            selected = np.asarray([assignments[domain][str(s)] == split for s in pool["source"]])
            indices = np.flatnonzero(selected)
            folder = features / f"{split}-{domain}"
            folder.mkdir(exist_ok=True)
            for key, values in pool.items():
                np.save(folder / f"{key}.npy", values[indices], allow_pickle=False)
            files = {p.name: sha(p) for p in folder.glob("*.npy")}
            write(folder / "complete.json", {"format_version": "synthetic-agent-feature-shard-v1", "rows": len(indices),
                                              "signature": signature, "files": files, "derived_not_reextracted": True})
            entries.append({"path": folder.name, "rows": len(indices), "sha256": sha(folder / "complete.json")})
            clusters = set(map(str, pool["source"][indices]))
            sources.update(clusters)
            split_audit[domain] = {"rows": len(indices), "groups": len(clusters),
                                   "rows_by_role": dict(zip(ROLES, map(int, np.bincount(pool["y"][indices], minlength=6))))}
            require(set(pool["y"][indices]) == set(range(6)), "A split/domain is missing a role")
        source_sets[split] = sources
        manifest["splits"][split] = entries
        manifest["progress"][split] = {"complete": True, "rows": sum(e["rows"] for e in entries), "records": len(sources)}
        audit["splits"][split] = split_audit
    audit["cross_split_source_overlap"] = {a + "/" + b: len(source_sets[a] & source_sets[b])
                                            for a, b in (("fit", "dev"), ("fit", "test"), ("dev", "test"))}
    require(not any(audit["cross_split_source_overlap"].values()), "Cross-split cluster leakage")
    write(features / "manifest.json", manifest)
    write(root / "data-audit.json", audit)
    protocol = {"format_version": "mixed-single-agentdojo-probe-v1", "created_unix": time.time(), "seed": args.seed,
                "article_feature_root": str(args.features.resolve()), "article_data_root": str(args.data.resolve()),
                "agentdojo_root": str(args.agentdojo.resolve()), "archival_probe": str(args.parent.resolve()),
                "derived_feature_manifest_sha256": sha(features / "manifest.json"),
                "split_assignment_sha256": sha(root / "split-assignment.json"), "split_ratios": [.7, .15, .15],
                "article_split_unit": "source cluster; stratified by corpus and native role",
                "agentdojo_split_unit": "whole task; stratified by banking/slack/travel suite",
                "sampling": "Uniform domain, then uniform present role, then uniform unique position; replacement sampling",
                "standardizer": "Unweighted FIT-position population mean/std; each unique position has family mass one",
                "selection": "Seven original random-init linear/MLP candidates, DEV-only selection and early stopping",
                "test_policy": "Freeze checkpoint before TEST inference; no changes or retuning after TEST",
                "runtime_profile_schema": list(DOMAINS),
                "heads": original["heads"], "roles": list(ROLES), "feature_dim": 1024, "feature_position": "pre-o_proj",
                "head_discovery_limitation": "Fixed v7 heads originally selected on mixed-context pilot FIT; this is a new probe-level split, not new head discovery",
                "agentdojo_limitations": "Task-disjoint split shares system/developer templates and suite/tool environment; historical 57-task cache, no new agent rollout",
                "device_uuid": GPU_UUID, "implementation_sha256": sha(Path(__file__)),
                "training_implementation_sha256": sha(Path(classifier.__file__))}
    write(root / "protocol.json", protocol)
    config = {"phase": "c4-single-plus-agentdojo-native", "classifier": {
        "device": "cuda:0", "batch_size": 4096, "learning_rate": .001, "weight_decay": .001,
        "dropout": .1, "gradient_clip": 5., "max_epochs": 45, "min_epochs": 10, "patience": 8,
        "checkpoint_steps": 100, "bootstrap_replicates": 2000,
        "profile_weights": {d: .5 for d in DOMAINS}, "cache_features_on_gpu": True}}
    write(root / "config.json", config)
    code = root / "code"
    code.mkdir(exist_ok=True)
    for path in (Path(__file__), Path(classifier.__file__), Path(cached.__file__), BASE / "run_single_message_probe.py",
                 BASE / "validate_mixed_single_agentdojo_probe.py"):
        if path.exists():
            shutil.copy2(path, code / path.name)
    print(json.dumps({"stage": "prepared", **audit}), flush=True)


def train(args):
    import torch
    root = args.output.resolve()
    protocol = read(root / "protocol.json")
    require(sha(root / "features/manifest.json") == protocol["derived_feature_manifest_sha256"], "Derived manifest changed")
    require(sha(root / "split-assignment.json") == protocol["split_assignment_sha256"], "Group split changed")
    require(sha(Path(__file__)) == protocol["implementation_sha256"], "Runner changed after preparation")
    resources = cuda_ready(root)
    resources["started_unix"] = time.time()
    write(root / "gpu-start.json", resources)
    # The existing hierarchical sampler is reused with domains as its profile
    # axis. This process-local schema leaves the shared classifier file intact.
    classifier.PROFILE_ORDER = DOMAINS
    classifier.DEFAULT_PROFILE_WEIGHTS = {d: .5 for d in DOMAINS}
    classifier.train_candidates(root / "features", root / "run", read(root / "config.json"), resume=args.resume)
    freeze = classifier.freeze(root / "run")
    resources.update(finished_unix=time.time(), max_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                     max_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    write(root / "gpu-training.json", resources)
    shutil.copy2(root / "run/best.npz", root / "checkpoint.npz")
    require(sha(root / "checkpoint.npz") == freeze["probe_sha256"], "Checkpoint copy changed")
    for split in SPLITS:
        result = classifier.evaluate(root / "features", root / "run", split, device="cuda:0", bootstrap_replicates=2000)
        print(json.dumps({"stage": "evaluation", "split": split, **result["overall"]}), flush=True)


class ClusterRows(SimpleNamespace):
    def __len__(self):
        return len(self.y)


def report(args):
    root = args.output.resolve()
    protocol, audit = read(root / "protocol.json"), read(root / "data-audit.json")
    selection = read(root / "run/selection.json")
    freeze = read(root / "run/freeze.json")
    selected = next(r for r in selection["candidate_results"] if r["candidate_id"] == selection["candidate_id"])
    result = {s: read(root / f"run/evaluations/{s}.json") for s in SPLITS}
    bank = classifier.FeatureBank(root / "features", "test")
    p = np.load(root / "run/evaluations/test-probabilities.npy", mmap_mode="r")
    domains = np.asarray([json.loads(str(g))["domain"] for g in bank.groups])
    for domain in DOMAINS:
        mask = domains == domain
        result["test"]["by_attribute"]["domain"][domain]["confidence_intervals"] = classifier.bootstrap_clusters(
            ClusterRows(source=bank.source[mask], y=bank.y[mask]), p[mask], replicates=2000, seed=args.seed)
    summary = {"selected_candidate": selected["candidate_id"], "best_epoch": selected["best_epoch"],
               "checkpoint_sha256": freeze["probe_sha256"], "results": result}
    write(root / "evaluation-summary.json", summary)
    lines = ["# C4＋Dolma3 单消息与 AgentDojo 无注入数据混合训练", "",
             f"新训练 probe：`{selected['candidate_id']}`，最佳 epoch {selected['best_epoch']}。随机初始化；只用 DEV 选模并冻结后评估 TEST。", "",
             "按约 70%／15%／15% 重新划分。文章按来源聚类并按语料、角色分层；AgentDojo 按完整任务划分并按 suite 分层。不同集合的文章来源、任务均不重叠。", "",
             "| 集合 | 单消息文章数 | 单消息 token | AgentDojo 任务数 | AgentDojo token | 合计 token |",
             "|---|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        a, b = (audit["splits"][split][d] for d in DOMAINS)
        lines.append(f"| {split.upper()} | {a['groups']:,} | {a['rows']:,} | {b['groups']} | {b['rows']:,} | {a['rows'] + b['rows']:,} |")
    lines += ["", "| 最终 TEST | token 数 | Accuracy | Macro-F1 |", "|---|---:|---:|---:|"]
    for label, metric in (("合并", result["test"]["overall"]), *result["test"]["by_attribute"]["domain"].items()):
        lines.append(f"| {label} | {sum(r['support'] for r in metric['roles'].values()):,} | {metric['accuracy']:.4%} | {metric['macro_f1']:.4%} |")
    for label, metric in (("合并 TEST", result["test"]["overall"]), *result["test"]["by_attribute"]["domain"].items()):
        lines += ["", f"## {label} 的各角色结果", "", "| 角色 | token 数 | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
        for role, values in metric["roles"].items():
            fields = [f"{values[k]:.4%}" if values[k] is not None else "N/A" for k in ("precision", "recall", "f1")]
            lines.append(f"| {role} | {values['support']:,} | " + " | ".join(fields) + " |")
    resources = read(root / "gpu-training.json")
    lines += ["", "训练时两个来源等概率采样，再均衡六角色；每轮抽样量等于 FIT 行数，有放回抽样。标准化器只由 FIT 的原始 token 特征拟合。",
              "AgentDojo 只保留原有首次出现普通正文 mask，包含实际输入和实际生成；标签由原生消息角色／channel 确定。排除重复上下文、结构性工具调用 JSON 与未 forward 的停止符。", "",
              f"NVIDIA RTX 2080 Ti，训练和选模耗时 {(resources['finished_unix']-resources['started_unix'])/60:.2f} 分钟，峰值 allocated {resources['max_allocated_mib']:.1f} MiB。", "",
              "这次 AgentDojo 已参与训练；其 TEST 是从 57 个历史任务中留出的任务，不能当作完整 57 任务的外部测试。任务间仍共享 system／developer 模板及工具环境，结果不代表未见模板或新 benchmark 的泛化。", "",
              "沿用原 v7 的 16 个 pre-o_proj 注意力头（1024 维）；原选头使用混合上下文 pilot FIT，本次只重新划分 probe 数据并训练 probe，未重新选头或运行 120B。", "",
              "权重：`checkpoint.npz`；具体任务／文章划分：`split-assignment.json`；原始指标与置信区间：`evaluation-summary.json`；独立复核：`independent-validation.json`。"]
    (root / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n")
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (title, metric) in zip(axes, (("Combined TEST", result["test"]["overall"]),
                                        ("Article single TEST", result["test"]["by_attribute"]["domain"]["c4_dolma3"]),
                                        ("AgentDojo held-out tasks", result["test"]["by_attribute"]["domain"]["agentdojo"]))):
        cm = np.asarray(metric["confusion"])
        fractions = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        ax.set(xticks=range(6), yticks=range(6), xticklabels=ROLES, yticklabels=ROLES,
               xlabel="Predicted", ylabel="Actual", title=title)
        ax.tick_params(axis="x", rotation=45)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{fractions[i,j]:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if fractions[i,j] > .6 else "black")
    fig.savefig(plots / "test-confusion.png", dpi=180)
    fig.savefig(plots / "test-confusion.pdf")
    plt.close(fig)
    (root / "requirements.txt").write_text(subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True))
    command = f"{shlex.quote(os.sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} --output \"$task_output\""
    script = ["#!/usr/bin/env bash", "set -euo pipefail", 'task_output="${1:?Provide a NEW output directory}"',
              f"export CUDA_VISIBLE_DEVICES={GPU_UUID}", "export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8",
              "export MPLCONFIGDIR=/tmp/mixed-role-probe-matplotlib", f"{command} --stage prepare", f"{command} --stage train",
              f"{command} --stage report", f"{shlex.quote(os.sys.executable)} {shlex.quote(str(BASE / 'validate_mixed_single_agentdojo_probe.py'))} --output \"$task_output\""]
    (root / "reproduce.sh").write_text("\n".join(script) + "\n")
    (root / "reproduce.sh").chmod(0o755)
    print(json.dumps({"stage": "reported", "candidate": selected["candidate_id"], "test": result["test"]["overall"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=BACKUP / "runs-v7/formal/features/formal")
    parser.add_argument("--data", type=Path, default=BACKUP / "data-v7/formal")
    parser.add_argument("--agentdojo", type=Path, default=AGENTDOJO)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--stage", choices=("prepare", "train", "report"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
