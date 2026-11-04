"""Post-freeze diagnostic of byte-identical feature vectors, without retuning."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
root = parser.parse_args().output.resolve()
manifest = json.loads((root / "features/manifest.json").read_text())
roles = manifest["roles"]
fit_hashes = set()
for entry in manifest["splits"]["fit"]:
    x = np.load(root / "features" / entry["path"] / "x.npy", mmap_mode="r")
    fit_hashes.update(hashlib.sha256(row.tobytes()).digest() for row in x)
p = np.load(root / "run/evaluations/test-probabilities.npy", mmap_mode="r")
offset, report, task_rows = 0, {}, []
for entry in manifest["splits"]["test"]:
    folder = root / "features" / entry["path"]
    x, y, source, groups = [np.load(folder / (key + ".npy"), mmap_mode="r") for key in ("x", "y", "source", "groups")]
    domain = json.loads(str(groups[0]))["domain"]
    duplicate = np.asarray([hashlib.sha256(row.tobytes()).digest() in fit_hashes for row in x])
    correct = p[offset:offset+len(x)].argmax(1) == y
    report[domain] = {"test_rows": len(x), "exact_fit_feature_matches": int(duplicate.sum()),
                      "by_role": {role: {"test_rows": int((y == i).sum()),
                                         "exact_fit_feature_matches": int((duplicate & (y == i)).sum())}
                                  for i, role in enumerate(roles)},
                      "nonmatching_test_rows": int((~duplicate).sum()),
                      "nonmatching_test_accuracy": float(correct[~duplicate].mean())}
    if domain == "agentdojo":
        for task in np.unique(source):
            mask = source == task
            task_rows.append({"task": str(task).split(":", 1)[1], "tokens": int(mask.sum()),
                              "correct": int(correct[mask].sum()), "accuracy": float(correct[mask].mean()),
                              "exact_fit_feature_matches": int(duplicate[mask].sum())})
    offset += len(x)
report["note"] = ("Whole tasks and articles are disjoint. Byte-identical states can still occur with shared causal prompt prefixes. "
                  "This is an interpretation diagnostic; no rows, training, checkpoint selection, or main TEST metrics were changed.")
(root / "exact-feature-overlap.json").write_text(json.dumps(report, indent=2) + "\n")
with (root / "agentdojo-test-per-task.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(task_rows[0]))
    writer.writeheader()
    writer.writerows(task_rows)
print(json.dumps(report), flush=True)
