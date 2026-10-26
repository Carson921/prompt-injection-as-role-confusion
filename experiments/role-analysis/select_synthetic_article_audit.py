"""Export deterministic FIT/DEV article audit samples without exporting TEST.

Only sample.jsonl, proof.json, and a body-free data-manifest.json are written.
The source corpus and its source-provenance.json are never copied. No model,
tokenizer, activation cache, classifier, or test prediction is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

VERSION = "synthetic-article-audit-selection-v1"
BODY_KEYS = {"text", "body", "content", "article_text", "sample_text", "excerpt", "audit_examples"}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def assert_body_free(value):
    """Fail closed instead of copying an unexpected embedded article preview."""
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in BODY_KEYS:
                raise ValueError("Data manifest contains a possible article-body field; do not export it")
            assert_body_free(item)
    elif isinstance(value, list):
        for item in value:
            assert_body_free(item)


def source_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def select_rows(rows, seed, per_stratum, excluded_ids=()):
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    excluded_ids = set(excluded_ids)
    eligible, excluded, total = Counter(), Counter(), Counter()
    group_eligible, group_total, group_excluded = Counter(), Counter(), Counter()
    primary, group_best, seen_ids = defaultdict(list), {}, set()
    for row in rows:
        source_id = row["source_id"]
        if source_id in seen_ids:
            raise ValueError("Duplicate source_id in the prepared source corpus")
        seen_ids.add(source_id)
        # TEST text is never retained in a candidate pool, sample, or proof.
        if row["split"] not in {"fit", "dev"}:
            continue
        key = (row["source"], row["split"])
        total[key] += 1
        group = None
        if row["source"] == "dolma3":
            if not row.get("source_group"):
                raise ValueError("Dolma3 article has no source_group provenance")
            group = (str(row["source_group"]), row["split"])
            group_total[group] += 1
        if source_id in excluded_ids:
            excluded[key] += 1
            if group:
                group_excluded[group] += 1
            continue
        eligible[key] += 1
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("An eligible FIT/DEV article has no body text")
        rank = hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()
        candidate = (rank, source_id, row)
        primary[key].append(candidate)
        primary[key].sort(key=lambda item: (item[0], item[1]))
        del primary[key][per_stratum:]
        if row["source"] == "dolma3":
            group_eligible[group] += 1
            if group not in group_best or (rank, source_id) < group_best[group][:2]:
                group_best[group] = candidate
    if not eligible:
        raise ValueError("No eligible FIT/DEV articles remain")
    corpora = {corpus for corpus, _ in total}
    for corpus in corpora:
        for split in ["fit", "dev"]:
            if eligible[(corpus, split)] < per_stratum:
                raise ValueError(f"Insufficient eligible articles for {corpus}/{split}; do not silently shrink the audit quota")
    chosen, reasons = {}, defaultdict(list)
    for key in sorted(primary):
        for rank, source_id, row in primary[key]:
            chosen[source_id] = (rank, row)
            reasons[source_id].append("primary_corpus_split_quota")
    covered_groups = {(str(row["source_group"]), row["split"]) for _, row in chosen.values()
                      if row["source"] == "dolma3"}
    supplements = []
    for group in sorted(group_best):
        if group not in covered_groups:
            rank, source_id, row = group_best[group]
            chosen[source_id] = (rank, row)
            reasons[source_id].append("missing_dolma3_source_group_split_minimum_hash")
            covered_groups.add(group)
            supplements.append(source_id)
    selected = []
    for source_id, (rank, row) in chosen.items():
        item = dict(row)
        if "_audit_selection" in item:
            raise ValueError("Unexpected existing audit-selection metadata")
        item["_audit_selection"] = {"rank_sha256": rank, "reasons": reasons[source_id]}
        selected.append(item)
    selected.sort(key=lambda row: (row["source"], row["split"], row["_audit_selection"]["rank_sha256"], row["source_id"]))
    if any(row["split"] not in {"fit", "dev"} or row["source_id"] in excluded_ids for row in selected):
        raise AssertionError("TEST or an excluded pilot article escaped the audit sample mask")
    statistics = {
        "strata": {f"{corpus}/{split}": {"total": total[(corpus, split)],
                     "excluded_pilot": excluded[(corpus, split)], "eligible": eligible[(corpus, split)],
                     "primary_selected": len(primary[(corpus, split)]),
                     "selected_after_group_supplements": sum(row["source"] == corpus and row["split"] == split for row in selected)}
                    for corpus in sorted(corpora) for split in ["fit", "dev"]},
        "dolma3_group_split": {f"{group}/{split}": {"total": count,
                     "excluded_pilot": group_excluded[(group, split)], "eligible": group_eligible[(group, split)],
                     "selected": sum(row["source"] == "dolma3" and str(row["source_group"]) == group and row["split"] == split for row in selected)}
                    for (group, split), count in sorted(group_total.items())},
        "supplement_source_ids": supplements,
    }
    return selected, statistics


def select_audit(data_dir, phase, output_dir, seed, per_stratum=12, exclude_pilot=False):
    if phase != "formal":
        raise ValueError("This tool is restricted to formal FIT/DEV audit exports")
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    folder = data_dir / phase
    if output_dir == folder:
        raise ValueError("Use a separate audit-sample directory; do not overwrite canonical data artifacts")
    manifest_path, sources_path = folder / "data-manifest.json", folder / "sources.jsonl"
    manifest = read(manifest_path)
    if manifest.get("phase") != phase:
        raise ValueError("Prepared data manifest phase does not match")
    assert_body_free(manifest)
    manifest_sha, sources_sha = sha(manifest_path), sha(sources_path)
    if manifest["sha256"]["sources.jsonl"] != sources_sha:
        raise ValueError("Source corpus differs from the completed formal data manifest")
    excluded = set()
    pilot_key_sha = None
    if exclude_pilot:
        pilot_keys = data_dir / "pilot" / "source-keys.json"
        pilot_manifest = read(data_dir / "pilot" / "data-manifest.json")
        pilot_key_sha = sha(pilot_keys)
        if pilot_manifest["sha256"]["source-keys.json"] != pilot_key_sha:
            raise ValueError("Pilot source-key metadata changed")
        excluded = {row["source_id"] for row in read(pilot_keys)}
    binding = {"phase": phase, "seed": seed, "per_stratum": per_stratum, "exclude_pilot": exclude_pilot,
               "pilot_source_keys_sha256": pilot_key_sha, "data_manifest_sha256": manifest_sha,
               "sources_sha256": sources_sha, "implementation_sha256": sha(__file__)}
    output_dir.mkdir(parents=True, exist_ok=True)
    proof_path = output_dir / "proof.json"
    if proof_path.exists():
        existing = read(proof_path)
        if existing["binding"] != binding:
            raise ValueError("Audit sample already exists with different inputs/seed/code; use a new output directory")
        if sha(output_dir / "sample.jsonl") != existing["sample_sha256"] or sha(output_dir / "data-manifest.json") != manifest_sha:
            raise ValueError("Audit output bytes changed")
        saved_rows = list(source_rows(output_dir / "sample.jsonl"))
        if len(saved_rows) != existing["selected_rows"] or any(row["split"] not in {"fit", "dev"} or row["source_id"] in excluded for row in saved_rows):
            raise ValueError("Existing audit sample violates the FIT/DEV-only non-pilot scope")
        return existing
    selected, stats = select_rows(source_rows(sources_path), seed, per_stratum, excluded)
    if sha(manifest_path) != manifest_sha or sha(sources_path) != sources_sha:
        raise ValueError("Formal inputs changed during article sampling")
    output = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode() for row in selected)
    atomic_bytes(output_dir / "sample.jsonl", output)
    atomic_bytes(output_dir / "data-manifest.json", manifest_path.read_bytes())
    proof = {"format_version": VERSION, "binding": binding,
             "data_manifest_sha256": manifest_sha, "sources_sha256": sources_sha,
             "sample_sha256": sha(output_dir / "sample.jsonl"), "selected_rows": len(selected),
             "rule": {"primary": "For each corpus × FIT/DEV, take the first per_stratum articles ordered by SHA256(seed + ':' + source_id), then source_id",
                      "exclude_pilot": "Exclude every pilot source_id using the checksum-verified metadata-only source-keys.json",
                      "supplement": "For each eligible Dolma3 source_group × FIT/DEV absent from the primary sample, add its minimum SHA256 rank article",
                      "test": "TEST bodies never enter candidate pools or exported samples; no TEST predictions are opened"},
             **stats, "selected": [{"source_id": row["source_id"], "source": row["source"], "split": row["split"],
                        "source_group": row.get("source_group"), "rank_sha256": row["_audit_selection"]["rank_sha256"],
                        "reasons": row["_audit_selection"]["reasons"],
                        "sample_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()} for row in selected],
             "exported_files": ["sample.jsonl", "proof.json", "data-manifest.json"],
             "body_export_scope": "Only the selected, non-pilot FIT/DEV full articles; source corpus and source-provenance.json remain remote"}
    atomic_bytes(proof_path, json_bytes(proof))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=["formal"], default="formal")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default="audit-formal-v5-1")
    parser.add_argument("--per-stratum", type=int, default=12)
    parser.add_argument("--exclude-pilot", action="store_true")
    args = parser.parse_args()
    result = select_audit(args.data_dir, args.phase, args.output_dir, args.seed, args.per_stratum, args.exclude_pilot)
    print(json.dumps({"status": "complete", "selected_rows": result["selected_rows"],
                      "data_manifest_sha256": result["data_manifest_sha256"], "sources_sha256": result["sources_sha256"],
                      "proof": str(args.output_dir / "proof.json")}), flush=True)


if __name__ == "__main__":
    main()
