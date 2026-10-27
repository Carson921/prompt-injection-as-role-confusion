"""Structure-only repository export scan; never interprets model score values."""
from pathlib import Path
import argparse
import ast
import collections
import csv
import hashlib
import json
import re
import time
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    dangerous = {"text", "full_text", "article_text", "body", "content", "prompt", "rendered_prompt",
                 "messages", "normalized_messages", "raw_content", "input_ids", "generated_ids",
                 "token_pieces", "head_features", "features", "x"}
    commentary_fields = ("reason", "rationale", "limitation", "limit", "scope", "criteria", "rule", "protocol",
                         "note", "purpose", "selection", "description", "inference", "exclusion", "statement",
                         "sampling", "reviewer", "notice", "ordering", "supervision", "containment", "claim",
                         "formula", "standardization", "sampling_frame", "window", "body_denominator", "wording",
                         "token_rule", "output_rule", "archive", "roles", "require", "role_confusion", "explanation",
                         "no_", "quality", "acceptance", "restoration", "transfer_template", "current_action")
    problems, exceptions, inventory = [], [], []
    counts = collections.Counter()

    def flag(file, location, kind, size):
        problems.append({"file": file, "location": location, "kind": kind, "size": size})

    def walk(value, file, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                location = f"{path}.{key}"
                if key in dangerous:
                    if isinstance(child, (dict, list)) and child:
                        flag(file, location, "structured_sensitive_payload", len(child))
                    elif isinstance(child, str) and len(child) > 120 and "coverage_notes" not in location:
                        flag(file, location, "sensitive_text_payload", len(child))
                walk(child, file, location)
        elif isinstance(value, list):
            if len(value) > 1000:
                flag(file, path, "large_embedded_array_requires_review", len(value))
            for i, child in enumerate(value):
                if isinstance(child, (dict, list, str)):
                    walk(child, file, f"{path}[{i}]")
        elif isinstance(value, str):
            word_count = len(re.findall(r"[A-Za-z]+", value))
            if len(value) >= 240 and word_count >= 35 and not any(k in path.lower() for k in commentary_fields):
                if path == ".unit_tests.log" and " tests in " in value and value.strip().endswith("OK"):
                    exceptions.append({"file": file, "location": path, "reason": "Unittest names/output only, validated footer; no article text"})
                else:
                    flag(file, path, "unclassified_natural_language", len(value))

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        name = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        counts[path.suffix] += 1
        inventory.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        if path.suffix == ".json":
            walk(json.loads(raw), name)
        elif path.suffix == ".jsonl":
            rows = raw.decode().splitlines()
            for i, line in enumerate(rows):
                walk(json.loads(line), name, f"line[{i}]")
            if len(rows) > 1000:
                flag(name, "", "large_jsonl_cache_requires_review", len(rows))
        elif path.suffix in (".csv", ".tsv"):
            rows = csv.DictReader(raw.decode("utf-8-sig").splitlines(), delimiter="\t" if path.suffix == ".tsv" else ",")
            for i, row in enumerate(rows):
                walk(row, name, f"row[{i}]")
        elif path.suffix in (".log", ".md", ".txt"):
            text = raw.decode()
            for i, line in enumerate(text.splitlines(), 1):
                if path.suffix != ".md" and len(line) > 300 and len(re.findall(r"[A-Za-z]+", line)) >= 45:
                    flag(name, f"line[{i}]", "long_log_text_requires_review", len(line))
                if re.search(r"(?:input_ids|generated_ids|head_features)\s*[:=]\s*(?:tensor\()?\[", line) and len(re.findall(r"\b\d+\b", line)) > 32:
                    flag(name, f"line[{i}]", "raw_tensor_in_plaintext", len(line))
        elif path.suffix == ".py":
            for node in ast.walk(ast.parse(raw)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 500:
                    flag(name, f"line[{node.lineno}]", "long_source_literal_requires_review", len(node.value))
        elif path.suffix == ".npz":
            # Named small classifier weights are permitted; no tensor values read.
            names = zipfile.ZipFile(path).namelist()
            if len(raw) > 5 * 1024 ** 2 or any(n.removesuffix(".npy") not in ("mean", "scale", "metadata") and not n.startswith("weight::") for n in names):
                flag(name, "", "nonclassifier_npz_or_large_archive", len(raw))
        elif path.suffix in (".npy", ".pt", ".safetensors", ".parquet", ".arrow", ".zst"):
            flag(name, "", "raw_cache_or_resume_payload_requires_private_backup", len(raw))

    audit_path = root / "data-preparation/v5/article-audit.json"
    review = json.loads(audit_path.read_text())
    quality_path = root / "data-preparation/v5/quality-excluded-clusters.json"
    quality = json.loads(quality_path.read_text())
    origin = json.loads((root / "data-preparation/v5/quality-origin-summary.json").read_text())
    proof_failures = []
    for row in review["reviews"]:
        proof = root / row["proof_file"]
        if not proof.is_file() or sha(proof) != row["proof_file_sha256"]:
            proof_failures.append(row["sample_index"])
    assert sha(root / "data-preparation/v5/data-manifest.json") == review["data_manifest_sha256"]
    assert len(review["reviews"]) == 50 and all(r["decision"] == "pass" for r in review["reviews"])
    assert sum(r["inherited_by_sha"] for r in review["reviews"]) == 49
    assert review["current_sample_sources_in_quality_exclusions"] == 0
    assert len(set(quality["excluded_clusters"])) == origin["global_excluded"] == 14
    assert origin["retained_v7_excluded"] == 12 and origin["supplementary_excluded"] == 2
    assert origin["retained_v7_by_split"] == {"fit": 8, "test": 4}
    assert not proof_failures

    result = {"format_version": "warmstart-repository-payload-audit-v1", "status": "passed_snapshot" if not problems else "requires_review",
              "audited_at_unix": time.time(), "root": str(root), "auditor_sha256": sha(__file__),
              "files_scanned": len(inventory), "suffix_counts": dict(counts), "problems": problems,
              "reviewed_exceptions": exceptions, "inventory": inventory,
              "score_values_interpreted": False, "frozen_core_modified": False,
              "allowed_metadata": ["Review rationale summaries", "Public URLs", "Hashes and source/cluster IDs", "Ordered16 head IDs", "Shapes/dtypes and timing/resource counts"],
              "scope_limitations": ["Snapshot only: rerun after new report/artifact additions", "PNG/PDF visual contents were not OCR-audited", "Structured score values are not interpreted or used for model comparison"],
              "final_source_quality": {"global_excluded": 14, "retained_v7": 12, "supplementary": 2, "retained_v7_by_split": origin["retained_v7_by_split"],
                                       "quality_ledger_sha256": sha(quality_path)},
              "final_article_review": {"status": review["status"], "reviewed_articles": 50, "inherited_by_body_sha": 49, "new_full_review": 1,
                                       "proof_hash_failures": proof_failures, "sampling_frame": review["sampling_frame"],
                                       "article_audit_sha256": sha(audit_path)},
              "historical_gate_interpretation": {"v1": "Failed article semantics; three original source rejections; immutable evidence retained",
                                                  "v3": "Failed follow-up property/MLS source semantics; not a final passed dataset",
                                                  "v4": "Failed follow-up rental-inventory source semantics; not a final passed dataset",
                                                  "v5": "Only final source-semantic pass, bound by current data manifest and actual50 review proof",
                                                  "mask_audit_passes": "Historical v1/v3/v4 input-mask audits can pass while source-semantic gate fails; scope is explicit in their file names and the preparation README"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "files_scanned": len(inventory), "problems": problems, "report_sha256": sha(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit(arguments.root, arguments.output)
