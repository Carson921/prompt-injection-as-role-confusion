"""Verify the all-single refit from raw arrays, saved predictions and weights.

The training runner and its feature/probe consumers are not imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import erf

from validate_single_message_probe import compare_metric, measures, require, sha, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    protocol, audit = read(root / "protocol.json"), read(root / "data-audit.json")
    source = Path(protocol["source_previous_run"])
    manifest = read(source / "features/manifest.json")
    freeze, training = read(root / "run/freeze.json"), read(root / "run/training.json")
    require(protocol["source_splits"] == ["fit", "dev", "test"], "Incomplete training union")
    require(protocol["classifier"]["epochs"] == 20 and protocol["classifier"]["seed"] == 29, "Changed fixed settings")
    require(sha(source / "features/manifest.json") == protocol["feature_manifest_sha256"], "Source manifest changed")
    require(sha(source / "data-audit.json") == protocol["source_data_audit_sha256"], "Source single-message audit changed")
    require(sha(root / "run/training.json") == freeze["training_sha256"], "Training config changed")
    require(sha(root / "run/selection.json") == freeze["selection_sha256"], "Final epoch decision changed")
    require(sha(root / "protocol.json") == freeze["protocol_sha256"] == training["protocol_sha256"], "Protocol changed")
    require(sha(root / "checkpoint.npz") == sha(root / "run/best.npz") == freeze["probe_sha256"], "Frozen checkpoint changed")
    require(freeze["selection_split"] == "fixed_epochs_all_single", "Unexpected validation/model selection")
    require(audit["no_article_holdout"] and audit["former_dev_and_test_are_training_data"], "Incorrect holdout claim")
    gpu = read(root / "gpu-training.json")
    require(gpu["name"] == "NVIDIA GeForce RTX 2080 Ti", "Wrong training GPU")
    with np.load(root / "checkpoint.npz", allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    metadata = json.loads(str(arrays["metadata"]))
    require(metadata["epoch"] == 20 and metadata["hidden"] == 256 and not metadata["engineering_smoke_only"], "Wrong model")
    require(metadata["heads"] == manifest["heads"], "Changed ordered heads")
    mean, scale = arrays["mean"], arrays["scale"]
    def forward(x):
        h = (x.astype(np.float32) - mean) / scale
        h = h @ arrays["weight::0.weight"].T + arrays["weight::0.bias"]
        h = .5 * h * (1 + erf(h / np.float32(np.sqrt(2))))
        logits = h @ arrays["weight::3.weight"].T + arrays["weight::3.bias"]
        p = np.exp(logits - logits.max(1,keepdims=True))
        return p / p.sum(1,keepdims=True)
    published = np.load(root / "run/train-probabilities.npy", mmap_mode="r")
    labels, sources, families = [], [], []
    sums, squares = np.zeros(1024), np.zeros(1024)
    cursor, max_error, disagreements = 0, 0., 0
    per_origin = {}
    for split in protocol["source_splits"]:
        origin_rows = 0
        for entry in manifest["splits"][split]:
            folder = source / "features" / entry["path"]
            commit = read(folder / "complete.json")
            require(sha(folder / "complete.json") == entry["sha256"], "Source subset commit changed")
            values = {}
            for key in ("x", "y", "source", "family", "structure", "groups", "weight"):
                path = folder / (key+".npy")
                require(sha(path) == commit["files"][path.name], "Source subset array changed")
                values[key] = np.load(path,mmap_mode="r",allow_pickle=False)
            require(values["x"].dtype == np.float16 and values["x"].shape == (entry["rows"],1024), "Feature geometry changed")
            require(np.char.startswith(values["structure"],"single:").all(), "Non-single training row")
            require(np.all(values["weight"] == 1), "Unexpected family weights")
            for g in np.unique(values["groups"]):
                g=json.loads(str(g));require(g["profile"] == "single" and g["rounds"] == 0, "Non-single input")
            labels.append(np.array(values["y"]))
            sources.extend(map(str,values["source"]))
            families.extend(map(str,values["family"]))
            for start in range(0,entry["rows"],8192):
                x=values["x"][start:start+8192]
                actual=forward(x)
                reference=published[cursor:cursor+len(x)]
                max_error=max(max_error,float(np.max(abs(actual-reference))))
                disagreements+=int((actual.argmax(1)!=reference.argmax(1)).sum())
                xd=x.astype(np.float64);sums+=xd.sum(0);squares+=(xd*xd).sum(0)
                cursor+=len(x);origin_rows+=len(x)
        per_origin[split]=origin_rows
    require(per_origin == {"fit":397312,"dev":49664,"test":99328}, "Not all previous splits were consumed")
    require(cursor == len(published) == 546304 and len(set(sources)) == 5632, "Incomplete training population")
    require(len(set(families)) == len(families), "Unexpected duplicated family")
    expected_mean=(sums/cursor).astype(np.float32)
    expected_scale=np.sqrt(np.maximum(squares/cursor-(sums/cursor)**2,0)).astype(np.float32)
    expected_scale[expected_scale<1e-6]=1
    require(np.max(abs(expected_mean-mean))<1e-6 and np.max(abs(expected_scale-scale))<1e-6, "All-data standardizer differs")
    require(max_error<5e-4 and disagreements==0, "Independent CPU predictions disagree with GPU")
    labels=np.concatenate(labels)
    compare_metric(measures(labels,published),read(root/"run/train-metrics.json")["overall"])
    history=read(root/"run/history.json")
    require(len(history)==20, "Wrong epoch count")
    rng=np.random.default_rng(29)
    for epoch,item in enumerate(history,1):
        order=rng.permutation(cursor)
        require(item["epoch"]==epoch and item["rows"]==item["unique_rows"]==cursor and item["all_rows_once"], "Incomplete epoch coverage")
        require(item["shuffle_sha256"]==hashlib.sha256(order.tobytes()).hexdigest(), "Epoch shuffle differs")
    with np.load(root/"agentdojo/first-position-probabilities.npz",allow_pickle=False) as z:
        y,p,tasks=z["labels"],z["probabilities"],z["task"]
        require(len(y)==155103 and len(np.unique(tasks))==57, "Wrong AgentDojo test population")
        agent=read(root/"agentdojo/metrics.json")
        require(agent["selection_split"]=="fixed_epochs_all_single" and agent["probe_sha256"]==freeze["probe_sha256"], "Wrong evaluated checkpoint")
        compare_metric(measures(y,p),agent["overall"])
        for scope,m in agent["by_scope"].items():
            mask=z["scope"]==scope;compare_metric(measures(y[mask],p[mask]),m)
    result={"status":"passed","probe_sha256":freeze["probe_sha256"],"all_training_rows":cursor,
            "source_clusters":5632,"rows_by_original_split":per_origin,"no_article_holdout":True,
            "all_20_epochs_visit_every_row_once":True,"all_data_standardizer_independently_recomputed":True,
            "maximum_cpu_gpu_probability_difference":max_error,"argmax_disagreements":disagreements,
            "agentdojo_tokens":155103,"agentdojo_tasks":57,"metrics_recomputed_from_saved_predictions":True}
    (root/"independent-validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
