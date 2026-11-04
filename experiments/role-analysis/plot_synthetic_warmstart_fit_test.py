"""Read-only CPU FIT/held-out article TEST diagnosis of frozen role probes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from synthetic_agent_classifier import Probe, ROLES, _metrics_from_confusion

PARENT_SHA = "22cb7251cd96c81d8d1225311b1e3048499850f738b56f4b57040ad7bedf4125"
NEW_SHA = "45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df"
BINDING = ("roles", "heads", "feature_dim", "feature_position", "model_binding")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def safe(root, relative):
    p = (root / relative).resolve()
    require(p.is_relative_to(root.resolve()), "Cache path escapes root")
    return p


def evaluate(root, split, probes, evidence, indices=None, reference=None):
    manifest = read(root / "manifest.json")
    evidence[str(root / "manifest.json")] = sha(root / "manifest.json")
    for key in BINDING:
        require(manifest[key] == probes["new"].metadata[key], "Feature binding differs: " + key)
    if "progress" in manifest:
        require(manifest["progress"][split]["complete"], "Incomplete feature split")
    matrices = {name: np.zeros((6, 6), np.int64) for name in probes}
    losses = {name: 0. for name in probes}
    sources, all_rows, scored, spot = set(), 0, 0, []
    published = np.load(reference, mmap_mode="r", allow_pickle=False) if reference else None
    max_p_error, argmax_disagreements = 0., 0
    for entry in manifest["splits"][split]:
        folder = safe(root, entry["path"])
        commit_path = folder / "complete.json"
        require(sha(commit_path) == entry["sha256"], "Shard commit SHA differs")
        evidence[str(commit_path)] = entry["sha256"]
        commit = read(commit_path)
        n = int(entry["rows"])
        require(commit["rows"] == n, "Shard row count differs")
        rows = None if indices is None else indices[(indices >= all_rows) & (indices < all_rows + n)] - all_rows
        all_rows += n
        if rows is not None and not len(rows):
            continue
        arrays = {}
        for key in ("x", "y", "source"):
            p = folder / (key + ".npy")
            expected = commit["files"][p.name]
            expected = expected["sha256"] if isinstance(expected, dict) else expected
            require(sha(p) == expected, "Feature array SHA differs: " + str(p))
            evidence[str(p)] = expected
            arrays[key] = np.load(p, mmap_mode="r", allow_pickle=False)
        require(arrays["x"].shape == (n, 1024) and arrays["x"].dtype == np.float16, "Invalid feature shape/dtype")
        require(arrays["y"].shape == arrays["source"].shape == (n,), "Label/source shape differs")
        if rows is None:
            rows = np.arange(n)
        for start in range(0, len(rows), 8192):
            selection = rows[start:start + 8192]
            x, y = arrays["x"][selection], arrays["y"][selection].astype(np.int64)
            require(np.isfinite(x).all() and ((y >= 0) & (y < 6)).all(), "Invalid feature/label")
            sources.update(map(str, arrays["source"][selection]))
            for name, probe in probes.items():
                p = probe.predict_proba(x)
                pred = p.argmax(1)
                matrices[name] += np.bincount(y * 6 + pred, minlength=36).reshape(6, 6)
                losses[name] += float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-30, 1)).sum(dtype=np.float64))
                if published is not None and name == "new":
                    ref = published[scored:scored + len(y)]
                    max_p_error = max(max_p_error, float(np.max(abs(p-ref))))
                    argmax_disagreements += int(np.sum(pred != ref.argmax(1)))
            if len(spot) < 2:
                spot.append((str(folder), [int(i) for i in selection[:64]]))
            scored += len(y)
    require(scored > 0 and (indices is None or scored == len(indices)), "Missing scored FIT/TEST rows")
    if indices is not None:
        require(indices.min() >= 0 and indices.max() < all_rows and len(np.unique(indices)) == len(indices), "Invalid replay indices")
    if published is not None:
        require(len(published) == scored and max_p_error < 5e-4 and argmax_disagreements == 0, "Published TEST predictions differ")
    result = {"rows": scored, "sources": len(sources), "metric_distribution": "Each retained position once; not the replacement-sampled training-loss distribution.",
        "models": {name: {**_metrics_from_confusion(matrix, losses[name]/scored), "errors": scored-int(np.trace(matrix)),
                           "perfect_separation": bool(np.trace(matrix) == scored)} for name, matrix in matrices.items()},
        "torch_spot_check_locations": spot}
    if published is not None:
        result["published_TEST_comparison"] = {"maximum_probability_error": max_p_error, "argmax_disagreements": argmax_disagreements}
    print(json.dumps({"stage": "scored", "split": split, "replay": indices is not None, "rows": scored,
                      "new_accuracy": result["models"]["new"]["accuracy"], "new_errors": result["models"]["new"]["errors"]}), flush=True)
    return result


def plot(panels, title, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(7.3*len(panels), 6.6), layout="constrained")
    for ax, (name, metric) in zip(np.atleast_1d(axes), panels):
        cm = np.asarray(metric["confusion"])
        support = cm.sum(1)
        normalized = cm / np.maximum(1, support[:, None])
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(6), ROLES, rotation=35, ha="right")
        ax.set_yticks(range(6), [f"{r}\n(n={int(n):,})" for r,n in zip(ROLES,support)])
        ax.set_xlabel("Predicted role"); ax.set_ylabel("True role")
        ax.set_title(f"{name}\nAccuracy {100*metric['accuracy']:.2f}% | macro-F1 {100*metric['macro_f1']:.2f}%")
        for i in range(6):
            for j in range(6):
                ax.text(j,i,f"{100*normalized[i,j]:.2f}%",ha="center",va="center",fontsize=9,
                        color="white" if normalized[i,j]>.5 else "black")
    fig.suptitle(title, fontsize=14)
    fig.colorbar(im,ax=list(np.atleast_1d(axes)),shrink=.75,label="Fraction within each true role")
    fig.supxlabel("Article-derived synthetic role contexts; article-body positions only. No AgentDojo/gardening/real-dialogue dataset.\n"
                  "Row-normalized; rounded percentages. Current scratch epoch11 is an INELIGIBLE research candidate.",fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix("."+extension),dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root",type=Path,required=True)
    parser.add_argument("--old-features",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    out, task = args.output_dir.resolve(), args.task_root.resolve()
    require(not out.exists(), "Use a new output directory")
    require(not out.is_relative_to(task) and not task.is_relative_to(out), "Output overlaps immutable mirror")
    out.mkdir(parents=True)
    started=time.time(); evidence={}
    parent_path, new_path = task/"parent/best.npz",task/"run/best.npz"
    require(sha(parent_path)==PARENT_SHA and sha(new_path)==NEW_SHA,"Checkpoint identity differs")
    probes={"parent":Probe(parent_path),"new":Probe(new_path)}
    require(np.array_equal(probes["parent"].mean,probes["new"].mean) and np.array_equal(probes["parent"].scale,probes["new"].scale),"Standardizer changed")
    training=read(task/"run/training.json");freeze=read(task/"run/freeze.json")
    require(freeze["probe_sha256"]==NEW_SHA and freeze["training_sha256"]==sha(task/"run/training.json"),"Freeze changed")
    replay_path=task/"run/replay-indices.npy"
    require(sha(replay_path)==training["replay_sha256"],"Replay identity differs")
    replay=np.load(replay_path,allow_pickle=False)
    require(replay.shape==(250000,) and replay.dtype==np.int64,"Replay shape differs")
    data=read(task/"data/data-manifest.json")
    source_counts=Counter()
    with (task/"data/sources.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);source_counts[(row["split"],row["source"])]+=1
    require(source_counts[("test","c4")]==source_counts[("test","dolma3")]==64,"Unexpected article TEST provenance")
    evidence[str(task/"data/sources.jsonl")]=sha(task/"data/sources.jsonl")
    require(evidence[str(task/"data/sources.jsonl")]==data["sha256"]["sources.jsonl"],"Article source file changed")
    datasets={}
    datasets["article_TEST"]=evaluate(task/"features","test",probes,evidence,reference=task/"run/evaluations/test-probabilities.npy")
    sealed=read(task/"run/evaluations/test.json")["overall"]
    require(datasets["article_TEST"]["models"]["new"]["confusion"]==sealed["confusion"],"Sealed TEST confusion differs")
    datasets["new_article_FIT"]=evaluate(task/"features","fit",probes,evidence)
    datasets["old_FIT_replay"]=evaluate(args.old_features.resolve(),"fit",probes,evidence,indices=replay)
    for name,expected in [("article_TEST",133631),("new_article_FIT",535328),("old_FIT_replay",250000)]:
        require(datasets[name]["rows"]==expected,"Dataset denominator differs")
    result={"status":"complete","generated_unix":time.time(),"checkpoint_sha256":{"parent":PARENT_SHA,"new":NEW_SHA},
        "roles":list(ROLES),"new_candidate":"scratch epoch11; DEV ineligible; no default replacement",
        "data_scope":"C4/Dolma3 article-derived synthetic role contexts, with no real-dialogue dataset added.",
        "provenance_limit":"Source-type and construction check, not a new full human semantic audit that every webpage has no quoted dialogue.",
        "FIT_metric_scope":"All retained new FIT rows and the exact fixed old FIT replay subset, each once with dropout off. Not the replacement-sampled optimizer event distribution.",
        "source_counts":{s:{k:v for (split,k),v in source_counts.items() if split==s} for s in ("fit","dev","test")},
        "datasets":datasets,"evidence":evidence,"implementation_sha256":sha(__file__),"elapsed_seconds":time.time()-started}
    write(out/"metrics.json",result)
    plot([("Previous final MLP128",datasets["article_TEST"]["models"]["parent"]),("Current scratch epoch11",datasets["article_TEST"]["models"]["new"])],
         "Held-out ARTICLE TEST | 128 sources, 133,631 positions",out/"article-test-parent-vs-new")
    plot([("New article FIT: 512 sources",datasets["new_article_FIT"]["models"]["new"]),("Old FIT replay: fixed 250,000 rows",datasets["old_FIT_replay"]["models"]["new"])],
         "TRAINING data readout | current checkpoint, deterministic inference",out/"article-fit-current")
    write(out/"manifest.json",{"status":"complete","script_sha256":sha(__file__),"files":[{"path":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file() and p.name!="manifest.json"]})
    print(json.dumps({"status":"complete","output":str(out),"seconds":time.time()-started}),flush=True)


if __name__=="__main__":
    main()
