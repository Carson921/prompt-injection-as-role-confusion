"""Refit on the union of v7 single-message FIT, DEV and TEST on local 2080 Ti.

Architecture, seed and final epoch are fixed from the previous DEV-selected
experiment before training. Every row is visited once per shuffled epoch.
No article holdout remains. AgentDojo is evaluated only after final freezing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

import synthetic_agent_classifier as classifier
import run_single_message_probe as single

ROLES = classifier.ROLES
read, write, sha, require = single.read, single.write, single.sha, single.require


class AllSingleBank:
    def __init__(self, source):
        self.banks = [classifier.FeatureBank(source / "features", s) for s in ("fit", "dev", "test")]
        self.offsets = np.cumsum([0] + [len(b) for b in self.banks])
        self.split = "fit"  # The union is now the complete training population.
        for key in ("y", "source", "family", "structure", "weight", "base_weight"):
            setattr(self, key, np.concatenate([getattr(b, key) for b in self.banks]))
        require(np.char.startswith(self.structure, "single:").all(), "Non-single training row")
        require(len(np.unique(self.family)) == len(self.family) and np.all(self.weight == 1), "Unexpected family weights")
        for b in self.banks:
            for value in np.unique(b.groups):
                group = json.loads(str(value))
                require(group["profile"] == "single" and group["rounds"] == 0, "Non-single input group")
        require(len(self) == 546304 and len(np.unique(self.source)) == 5632, "Incomplete all-single population")

    def __len__(self):
        return int(self.offsets[-1])

    def read(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        require(np.all((indices >= 0) & (indices < len(self))), "Row index outside union")
        result = np.empty((len(indices), 1024), np.float32)
        for i, bank in enumerate(self.banks):
            positions = np.flatnonzero((indices >= self.offsets[i]) & (indices < self.offsets[i+1]))
            if len(positions):
                result[positions] = bank.read(indices[positions] - self.offsets[i])
        return result


def train(args):
    import torch
    root, source = args.output.resolve(), args.source.resolve()
    require(not root.exists(), "Use a fresh output directory")
    resource = single.cuda_ready(root)
    previous = read(source / "run/selection.json")
    chosen = next(c for c in previous["candidate_results"] if c["candidate_id"] == previous["candidate_id"])
    require(previous["candidate_id"] == "mlp256-seed29" and chosen["best_epoch"] == 20, "Prior settings changed")
    root.mkdir(parents=True)
    (root / ".gitignore").write_text("*.log\n")
    run = root / "run"
    run.mkdir()
    options = {"hidden": 256, "seed": 29, "epochs": 20, "batch_size": 4096,
               "learning_rate": .001, "weight_decay": .001, "dropout": .1, "gradient_clip": 5.,
               "sampling": "shuffle every epoch; each of all 546304 rows exactly once; no replacement"}
    protocol = {
        "experiment": "all-single-fixed-epoch", "created_unix": time.time(),
        "source_previous_run": str(source), "feature_root": str(source / "features"),
        "feature_manifest_sha256": sha(source / "features/manifest.json"),
        "source_data_audit_sha256": sha(source / "data-audit.json"),
        "source_original_selection_sha256": sha(source / "run/selection.json"),
        "source_splits": ["fit", "dev", "test"], "classifier": options,
        "checkpoint_rule": "Fixed final epoch 20 from the previous DEV-selected configuration, declared before all-data training",
        "initialization": "Random seed29; no previous checkpoint weights or standardizer loaded",
        "validation": "No article validation or test holdout remains; former DEV and TEST are training data",
        "agentdojo": "Post-freeze evaluation only; same previously observed sealed no-injection 57-task cache",
        "head_discovery_limitation": single.HEAD_LIMITATION,
        "implementation_sha256": sha(Path(__file__)), "classifier_sha256": sha(Path(classifier.__file__)),
    }
    write(root / "protocol.json", protocol)
    resource["started_unix"] = time.time()
    write(root / "gpu-start.json", resource)
    bank = AllSingleBank(source)
    audit = {"status": "passed", "only_single_message": True, "rows": len(bank),
             "source_clusters": len(np.unique(bank.source)), "source_splits": {},
             "rows_by_role": {r: int((bank.y == i).sum()) for i, r in enumerate(ROLES)},
             "former_dev_and_test_are_training_data": True, "no_article_holdout": True,
             "feature_manifest_sha256": protocol["feature_manifest_sha256"]}
    for split, b in zip(protocol["source_splits"], bank.banks):
        audit["source_splits"][split] = {"rows": len(b), "source_clusters": len(np.unique(b.source)),
                                         "split_identity_hash": b.split_identity_hash}
    write(root / "data-audit.json", audit)
    print(json.dumps({"stage": "all_single_union_verified", **{k:audit[k] for k in ("rows", "source_clusters", "rows_by_role")}}), flush=True)
    mean, scale = classifier.fit_standardizer(bank)
    np.savez(run / "standardizer.npz", mean=mean, scale=scale)
    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    torch.cuda.manual_seed_all(options["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = classifier.make_net(options["hidden"], dropout=options["dropout"]).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    initial_weights = {f"weight::{k}": v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}
    np.savez_compressed(run / "initialization.npz", **initial_weights)
    x = torch.empty((len(bank), 1024), dtype=torch.float32, device="cuda:0")
    for start in range(0, len(bank), 8192):
        x[start:start+8192] = torch.from_numpy(bank.read(np.arange(start, min(start+8192, len(bank))))).cuda()
    mean_gpu, scale_gpu = torch.as_tensor(mean, device="cuda:0"), torch.as_tensor(scale, device="cuda:0")
    x.sub_(mean_gpu).div_(scale_gpu)
    y = torch.from_numpy(bank.y).cuda()
    binding = bank.banks[0].binding()
    binding["data_hash"] = single.digest_object({"union": audit["source_splits"], "source_manifest": protocol["feature_manifest_sha256"]})
    binding["config_hash"] = single.digest_object(options)
    training = {"format_version": "all-single-refit-v1", "protocol_sha256": sha(root / "protocol.json"),
                "options": options, "rows": len(bank), "binding": binding,
                "source_splits": audit["source_splits"], "standardizer_sha256": sha(run / "standardizer.npz"),
                "initialization_sha256": sha(run / "initialization.npz"), "initialization": "random",
                "device": resource["name"], "device_uuid": resource["uuid"]}
    write(run / "training.json", training)
    training_hash = sha(run / "training.json")
    history, rng = [], np.random.default_rng(options["seed"])
    for epoch in range(1, options["epochs"]+1):
        order = rng.permutation(len(bank))
        require(np.array_equal(np.sort(order), np.arange(len(bank))), "An epoch did not cover every row once")
        order_gpu = torch.from_numpy(order).cuda()
        net.train()
        loss_sum, seen, correct = 0., 0, 0
        for start in range(0, len(bank), options["batch_size"]):
            indices = order_gpu[start:start+options["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = net(x[indices])
            labels = y[indices]
            loss = torch.nn.functional.cross_entropy(logits, labels)
            require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), options["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(indices)
        require(seen == len(bank), "Incomplete training epoch")
        item = {"epoch": epoch, "rows": seen, "unique_rows": len(order), "all_rows_once": True,
                "shuffle_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
                "training_cross_entropy": loss_sum/seen, "online_training_accuracy": correct/seen}
        history.append(item)
        write(run / "history.json", history)
        print(json.dumps(item), flush=True)
    net.eval()
    metadata = {**binding, "format_version": classifier.PROBE_FORMAT, "candidate_id": "mlp256-seed29-all-single-epoch20",
                "hidden": 256, "seed": 29, "dropout": .1, "activation": "gelu_exact",
                "standardizer": "ALL_single_training_population_std", "training_hash": training_hash,
                "epoch": 20, "checkpoint_rule": "fixed_final_epoch", "training_device": "cuda:0",
                "engineering_smoke_only": False}
    classifier._export(run / "best.npz", net, mean, scale, metadata)
    shutil.copy2(run / "best.npz", root / "checkpoint.npz")
    selection = {"candidate_id": metadata["candidate_id"], "selection_split": "fixed_epochs_all_single",
                 "checkpoint_epoch": options["epochs"], "criterion": protocol["checkpoint_rule"],
                 "validation_used_in_this_run": False, "agentdojo_used_for_selection": False}
    write(run / "selection.json", selection)
    freeze = {"format_version": "all-single-fixed-epoch-freeze-v1", "frozen_at_unix": time.time(),
              "selection_split": "fixed_epochs_all_single", "epoch": options["epochs"],
              "probe_sha256": sha(run / "best.npz"), "selection_sha256": sha(run / "selection.json"),
              "training_sha256": training_hash, "protocol_sha256": sha(root / "protocol.json")}
    write(run / "freeze.json", freeze)
    resource.update(finished_unix=time.time(), max_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    max_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    write(root / "gpu-training.json", resource)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(bank), 8192):
            predictions.append(net(x[start:start+8192]).softmax(1).cpu().numpy())
    p = np.concatenate(predictions)
    np.save(run / "train-probabilities.npy", p, allow_pickle=False)
    write(run / "train-metrics.json", {"overall": classifier.metrics(p, bank.y), "evaluation_type": "training_population_only_not_held_out"})
    code = root / "code"
    code.mkdir()
    for file in (Path(__file__), Path(single.__file__), Path(classifier.__file__), Path(single.cached.__file__)):
        shutil.copy2(file, code / file.name)
    shutil.copy2(source / "requirements.txt", root / "requirements.txt")
    print(json.dumps({"stage": "frozen", "probe_sha256": freeze["probe_sha256"], "epoch": options["epochs"]}), flush=True)


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root, source = args.output.resolve(), args.source.resolve()
    audit, agent = read(root / "data-audit.json"), read(root / "agentdojo/metrics.json")
    train_metrics, previous = read(root / "run/train-metrics.json")["overall"], read(source / "agentdojo/metrics.json")["overall"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    single.plot_confusion(axes[0], train_metrics, "All single-message TRAIN (no article holdout)")
    single.plot_confusion(axes[1], agent["overall"], "AgentDojo no-injection 57 tasks")
    fig.suptitle("All C4 + Dolma3 single-message rows: fixed epoch20 refit")
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(plots / ("all-train-and-agentdojo-confusion."+ext), dpi=180)
    plt.close(fig)
    history = read(root / "run/history.json")
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    axes[0].plot([h["epoch"] for h in history],[h["training_cross_entropy"] for h in history])
    axes[0].set(xlabel="Epoch",ylabel="Online training cross entropy")
    axes[1].plot([h["epoch"] for h in history],[h["online_training_accuracy"] for h in history])
    axes[1].set(xlabel="Epoch",ylabel="Online training accuracy")
    fig.savefig(plots / "training-curves.png",dpi=180)
    plt.close(fig)
    lines = ["# 全部单消息训练 → AgentDojo 57 任务测试", "",
             "将原单消息 FIT、DEV、TEST 合并为训练集；仅 C4＋Dolma3，546,304 个 token 位置、5,632 篇来源。没有独立的文章验证集或测试集。", "",
             "沿用上一轮已定的 MLP256、seed29 和 20 个 epochs，重新随机初始化，重新计算全部训练数据的 mean/std。每轮无放回打乱并遍历全部样本。取第 20 轮最终 checkpoint，没有按 AgentDojo 成绩选模型。", "",
             "| 数据 | Accuracy | Macro-F1 |", "|---|---:|---:|",
             f"| 全部单消息训练集（仅训练集成绩） | {train_metrics['accuracy']:.4%} | {train_metrics['macro_f1']:.4%} |",
             f"| AgentDojo 57 任务：全部单消息训练 | {agent['overall']['accuracy']:.4%} | {agent['overall']['macro_f1']:.4%} |",
             f"| AgentDojo 57 任务：上一轮仅 FIT 训练 | {previous['accuracy']:.4%} | {previous['macro_f1']:.4%} |", "",
             "| AgentDojo 分组 | token | Accuracy | Macro-F1 |", "|---|---:|---:|---:|"]
    for scope,m in agent["by_scope"].items():
        lines.append(f"| {scope} | {m['tokens']:,} | {m['accuracy']:.4%} | {m['macro_f1']:.4%} |")
    lines.extend(["", "| 角色 | token | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"])
    for role,m in agent["overall"]["roles"].items():
        values=["N/A" if m[k] is None else f"{m[k]:.4%}" for k in ("precision","recall","f1")]
        lines.append(f"| {role} | {m['support']:,} | " + " | ".join(values) + " |")
    lines.extend(["", "![训练集及 AgentDojo 混淆矩阵](plots/all-train-and-agentdojo-confusion.png)", "",
                  "AgentDojo 沿用原始 359 次调用、155,103 个首次普通正文位置的激活与 mask；没有重跑 agent。该 57 任务集已被观察，是历史回归集。", "",
                  "固定 16 头／1024 维仍沿用 v7，历史选头曾使用混合上下文。此次分类器和标准化器只由全部单消息训练。", "",
                  "checkpoint：`checkpoint.npz`；训练方案：`protocol.json`；数据审计：`data-audit.json`；设备记录：`gpu-training.json`；独立复核：`independent-validation.json`。"])
    (root / "REPORT.zh-CN.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "report"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"train": train, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
