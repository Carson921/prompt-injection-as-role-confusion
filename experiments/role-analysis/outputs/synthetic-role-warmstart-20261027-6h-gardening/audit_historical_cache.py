"""Read-only assessment of whether archived gardening artifacts retain raw heads.

This writes a scoped assessment, never a new-checkpoint classification result.
Run with the local CPU tokenizer environment and Python -B.
"""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


REPO = Path(__file__).resolve().parents[4]
BACKUP = REPO.parent / "runpod-backups/role-retraining-20261026-20261027"
ARCHIVE = BACKUP / "workspace-task/diagnostics/gardening-pilot-vs-first-formal"
TOMATO = REPO / "experiments/role-analysis/outputs/local-gptoss120b/tomato"
FEATURE_SEARCH = TOMATO.parent / "feature-search/tomato"
PARENT = REPO / "experiments/role-analysis/outputs/synthetic-agent-120b-20261026/final-v7/formal/best.npz"
NEW = REPO.parent / "runpod-backups/role-warmstart-20261027-6h-final/task/run/best.npz"
HEADS = [1166, 1069, 2296, 1161, 2093, 2088, 2295, 2045,
         2179, 2288, 1274, 897, 1305, 2091, 2094, 1930]
ROLES = ["system", "developer", "user", "cot", "assistant", "tool"]
EXPECTED = dict(system=57, developer=0, user=97, cot=179, assistant=583, tool=0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"].item()))


def main():
    evidence = {}

    def checked(path, expected=None):
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        evidence[str(path)] = dict(sha256=digest, bytes=path.stat().st_size)
        return digest

    parent_sha = checked(PARENT, "22cb7251cd96c81d8d1225311b1e3048499850f738b56f4b57040ad7bedf4125")
    new_sha = checked(NEW, "45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df")
    parent, new = metadata(PARENT), metadata(NEW)
    delivery = read(ARCHIVE / "delivery-manifest.json")
    checked(ARCHIVE / "delivery-manifest.json")
    for relative, entry in delivery["files"].items():
        path = ARCHIVE / relative
        assert path.stat().st_size == entry["bytes"]
        checked(path, entry["sha256"])
    result = read(ARCHIVE / "results.json")
    prepared = read(ARCHIVE / "prepared-inputs.json")
    checked(ARCHIVE / "prepared-inputs.json", result["prepared_inputs_sha256"])
    checked(ARCHIVE / "prepare-manifest.json", result["prepare_manifest_sha256"])
    for meta in [parent, new] + [entry["metadata"] for entry in result["models"].values()]:
        assert meta["heads"] == HEADS and meta["feature_dim"] == 1024
        assert meta["feature_position"] == "pre-o_proj"
        assert meta["model_binding"] == parent["model_binding"]

    tokenizer = AutoTokenizer.from_pretrained(BACKUP / "model-metadata", local_files_only=True)
    samples = {}
    for sample in prepared:
        ids = tokenizer.encode(sample["text"], add_special_tokens=False)
        assert ids == sample["input_ids"] and len(sample["content"]) == 916
        assert sha256_text(sample["text"]) == sample["prompt_sha256"]
        for target in sample["content"]:
            assert target["token_id"] == ids[target["token_position"]]
            assert target["label"] == ROLES.index(target["semantic_role"])
        support = Counter(t["semantic_role"] for t in sample["content"])
        assert {r: support[r] for r in ROLES} == EXPECTED
        samples[sample["prompt_key"]] = dict(input_tokens=len(ids), complete_body_tokens=916,
            semantic_role_support=EXPECTED,
            saved_formal_roles=dict(Counter(str(t["formal_role"]) for t in sample["content"])),
            input_ids_sha256=hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
            token_ids_reproduced_with_official_tokenizer=True)
    assert set(samples) == {"proper_tags", "everything_in_user_tags", "basic_no_format"}

    probabilities = {}
    for path in sorted(ARCHIVE.glob("*.npy")):
        array = np.load(path, allow_pickle=False)
        assert array.shape == (916, 6) and np.isfinite(array).all()
        probabilities[path.name] = dict(shape=list(array.shape), dtype=str(array.dtype),
            content="Already classified six-role probabilities, not 1024 raw head features")
    assert len(probabilities) == 6

    schemas = {}
    for name in ("tokens.parquet", "labels.parquet", "inputs.parquet"):
        path = TOMATO / name
        checked(path)
        table = pq.read_table(path)
        schemas[name] = dict(rows=len(table), columns={f.name: str(f.type) for f in table.schema})
        assert all(str(field.type) in {"int64", "double", "string"} for field in table.schema)
        assert not any(field.name in {"x", "features", "activations"} for field in table.schema)
    projection_columns = {}
    for path in (TOMATO / "selected_projections.csv", FEATURE_SEARCH / "projections.csv.gz"):
        checked(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="") as stream:
            projection_columns[str(path)] = next(csv.reader(stream))
        assert "prob" in projection_columns[str(path)]
    for path in (TOMATO / "manifest.json", TOMATO / "extraction.json", FEATURE_SEARCH / "extraction.json"):
        checked(path)
    assert not (TOMATO / "activations.pt").exists()
    assert not (FEATURE_SEARCH / "activations.pt").exists()
    inference_source = BACKUP / "root-task/code-v7/evaluate_synthetic_gardening.py"
    checked(inference_source)
    report = dict(format_version="warmstart-gardening-historical-cache-assessment-v1",
        scope="Historical local artifact reuse only, before any supplementary gardening forward",
        status="raw_features_not_found_in_examined_archives",
        parent_probe_sha256=parent_sha, new_probe_sha256=new_sha,
        heads=HEADS, feature_position="pre-o_proj", feature_dim=1024,
        same_model_binding_and_feature_contract=True,
        archival_gardening_directory=str(ARCHIVE), archived_samples=samples,
        archived_probability_arrays=probabilities,
        archived_models={k: dict(candidate_id=v["candidate_id"], checkpoint_sha256=v["checkpoint_sha256"])
                         for k, v in result["models"].items()},
        parquet_schemas=schemas, projection_columns=projection_columns,
        incompatible_other_capture_points={"tomato": "post_attention_layernorm, 9 layers of 2880 residual dimensions",
            "feature_search": "norm-20 and mid-16/mid-20/mid-24 residual features"},
        raw_916_by_1024_cache_present=False, cpu_reclassification_currently_possible=False,
        new_probe_gardening_metrics=None, parent_final_probe_gardening_metrics=None,
        limitation="N/A is a scoped historical-cache assessment, not a permanent claim that supplementary measurement is impossible.",
        no_gpu_or_remote_activity=True, historical_artifacts_modified=False,
        evidence=evidence)
    output = Path(__file__).with_name("historical-cache-assessment.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], output=str(output), sha256=sha(output),
                          verified_files=len(evidence), archived_array_shapes=[916, 6],
                          parent_new_metrics="N/A pending raw feature measurement")))


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    main()
