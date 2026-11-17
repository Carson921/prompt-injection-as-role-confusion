"""Seal the complete, independently audited IPI no-payload experiment.

Hidden payload hashes are taken only from records covered by the full streamed
independent validator. All other delivery files are hashed here. No inference,
generation, scenario tools, or external requests run during sealing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import seal_agentdojo_trace_interventions as common

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT = HERE / "outputs/ipi-benign-trace-interventions-20261117"
SHA = common.PROBE_SHA
read, write, digest, require = common.read, common.write, common.digest, common.require


def check(root, relative, expected):
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()) and path.is_file(), "Invalid delivery path: " + relative)
    require(digest(path) == expected, "Checksum changed: " + relative)


def gate(root, name):
    stage = root / name
    run, protocol = read(stage / "run-manifest.json"), read(stage / "protocol.json")
    validation, export = read(stage / "independent-validation.json"), read(stage / "export-manifest.json")
    export_audit = read(stage / "export-independent-validation.json")
    stats = read(stage / "analysis/independent-statistics-validation.json")
    n_traces = 1 if name == "pilot" else 41
    n_inputs, n_tokens = protocol["inputs"], protocol["expected_token_rows"]
    require(run["status"] == "complete" and run["stage"] == name and run["traces"] == n_traces
            and run["inputs"] == n_inputs and run["summary_rows"] == protocol["expected_summary_rows"],
            "Incomplete inference stage: " + name)
    require(run["probe_sha256"] == protocol["probe_sha256"] == protocol["expected_probe_sha256"] == SHA,
            "Checkpoint identity differs")
    for key, filename in (("inputs_sha256", "inputs.jsonl"), ("protocol_sha256", "protocol.json"),
                          ("summary_sha256", "input-summary.csv")):
        check(root, name + "/" + filename, run[key])
    for key in ("source_records", "wrapper_catalog"):
        check(root, protocol[key + "_path"], protocol[key + "_sha256"])
    require(validation["status"] == "passed" and validation["checkpoint_sha256"] == SHA
            and validation["protocol_sha256"] == run["protocol_sha256"]
            and validation["inputs_sha256"] == run["inputs_sha256"]
            and validation["inputs"]["traces"] == n_traces and validation["inputs"]["inputs"] == n_inputs,
            "Independent input validation differs")
    require(validation["validator_sha256"] == digest(HERE / "validate_ipi_trace_interventions.py"),
            "Independent validator changed")
    for flag in ("all_probe_rows_independently_recomputed", "hiddenstate_full_payload_sha256_all_verified",
                 "all36_residual_layers_plus_embedding_final_norm", "native_bf16_to_checkpoint_fp16_bit_exact",
                 "same_body_embedding_rows_bit_exact_across_all_interventions"):
        require(validation.get(flag) is True, "Full independent validation is missing: " + flag)
    totals = validation["totals"]
    require(validation["independent_probe_row_cap"] == 0
            and totals["saved_body_token_rows"] == totals["independent_probe_token_rows"] == n_tokens
            and totals["hiddenstate_tensors"] == 43 * n_inputs
            and totals["residual_layer_tensors"] == 36 * n_inputs
            and totals["hiddenstate_bytes"] == run["hiddenstate_bytes"]
            and validation["summary_rows"] == protocol["expected_summary_rows"],
            "Independent full-payload audit has different denominators")
    require(validation["collection"]["status"] == "passed"
            and validation["collection"]["traces"] == n_traces
            and validation["collection"]["exact_original_ids_reconstructed"] is True,
            "Native collection conservation audit is incomplete")
    for flag in ("opaque_generated_call_ids_verified", "current_call_and_history_tool_response_references_verified",
                 "worldsim_administrative_ids_exclude_case_identifiers"):
        require(validation["collection"].get(flag) is True, "Clean simulator identifier audit is incomplete: " + flag)
    origins = read(stage / "first-body-provenance.json")
    require(origins["status"] == "passed" and origins["traces"] == n_traces
            and origins["source_traces_sha256"] == protocol["source_records_sha256"]
            and origins["role_support"] == validation["inputs"]["role_support"]
            and origins["auditor_sha256"] == digest(HERE / "audit_ipi_trace_body_provenance.py"),
            "First-body source provenance is incomplete or stale")
    require(export["status"] == "complete" and export["inputs"] == export["hiddenstate_files"]
            == export["readout_files"] == n_inputs and export["token_rows"] == n_tokens,
            "Exports are incomplete")
    require(export["exporter_sha256"] == digest(HERE / "export_ipi_trace_readouts.py"), "Exporter changed")
    for artifact in ("token_readouts", "hiddenstates_index"):
        check(root, export[artifact]["path"], export[artifact]["sha256"])
    require(export_audit["status"] == "passed" and export_audit["inputs"] == n_inputs
            and export_audit["token_rows"] == n_tokens
            and export_audit["differing_float32_bit_patterns"] == 0
            and export_audit["run_manifest_sha256"] == digest(stage / "run-manifest.json")
            and export_audit["source_traces_sha256"] == protocol["source_records_sha256"]
            and export_audit["validator_sha256"] == digest(HERE / "audit_ipi_trace_readout_export.py")
            and export_audit["export_manifest_sha256"] == digest(stage / "export-manifest.json"),
            "Independent export audit is incomplete")
    for flag in ("all_saved_token_rows_compared", "coordinates_and_token_ids_bit_exact", "saved_logits_bit_exact",
                 "saved_probabilities_bit_exact", "baseline_and_variant_role_masks_exact",
                 "body_token_offsets_exact", "hiddenstate_index_matches_committed_records", "exporter_not_imported"):
        require(export_audit.get(flag) is True, "Export audit did not prove: " + flag)
    require(stats["status"] == "passed" and stats["input_summary_sha256"] == run["summary_sha256"]
            and stats["analyzer_not_imported"] is True
            and stats["auditor_sha256"] == digest(HERE / "audit_ipi_trace_intervention_statistics.py"),
            "Statistics audit is incomplete")
    for relative, expected in export["source_hashes"].items():
        check(root, relative, expected)
    audit_time = (stage / "independent-validation.json").stat().st_mtime_ns
    for input_id, commitment in run["records"].items():
        relative = "records/" + input_id + ".json"
        record = read(root / relative)
        require(commitment == {key: record[key] for key in ("readout_sha256", "hiddenstates_sha256")},
                "Run and per-input record differ")
        require(export["record_hashes"][relative] == export_audit["record_hashes"][relative]
                == digest(root / relative), "Input record changed after independent export audit")
        hidden = root / record["hiddenstates"]
        require(hidden.is_file() and hidden.stat().st_size == record["hiddenstate_bytes"]
                and hidden.stat().st_mtime_ns <= audit_time
                and (root / relative).stat().st_mtime_ns <= audit_time,
                "Hidden-state payload or record changed after full audit")
        require(export["readout_hashes"][record["readout"]] == export_audit["readout_hashes"][record["readout"]]
                == record["readout_sha256"], "Readout identity changed after audit")
        check(root, record["readout"], record["readout_sha256"])
    require({r["cohort"]: r["traces"] for r in stats["cohorts"]} == {name: n_traces},
            "Statistics cohort differs")
    for cohort in stats["cohorts"]:
        require(cohort["status"] == "passed" and cohort["scalar_checks"] > 0, "Statistics audit failed")
        if n_traces > 1:
            require(cohort["bootstrap_resamples"] == 2000
                    and cohort["paired_pointwise_95_percent_intervals_all_recomputed"] is True,
                    "Bootstrap intervals were not independently recomputed")
        for filename, expected in cohort["files_sha256"].items():
            check(root, f"{name}/analysis/{name}/{filename}", expected)
    folder = stage / "analysis" / name
    for stem in ("role-interventions-main", "wrapper-destination-attraction", "full-six-role-probabilities", "role-token-curves"):
        for extension in ("png", "pdf", "svg"):
            require((folder / f"{stem}.{extension}").is_file(), "Missing English figure")
    curves = read(folder / "token-curve-validation.json")
    require(curves["inputs_sha256"] == run["inputs_sha256"]
            and curves["input_summary_sha256"] == run["summary_sha256"]
            and curves["all_curve_means_match_primary_summary"] is True
            and curves["token_ids_and_original_positions_identical_across_conditions"] is True,
            "Token curves do not agree with primary readouts")
    return protocol, run


def seal(root):
    root = root.resolve()
    require(not (root / "delivery-manifest.json").exists(), "Delivery is already sealed")
    stages = {name: gate(root, name) for name in ("pilot", "all")}
    collection = read(root / "collection/protocol.json")
    require(read(root / "prepared/manifest.json")["cases"] == 41
            and read(root / "sources/all-collection-manifest.json")["traces"] == 41,
            "All 41 benchmark tasks must be collected")
    feedback = read(root / "pilot/feedback.json")
    require(feedback["feedback_delivered"] is True
            and feedback["pilot_run_manifest_sha256"] == digest(root / "pilot/run-manifest.json"),
            "Actual single-trace feedback has not been delivered")
    require(feedback["pilot_validation_sha256"] == digest(root / "pilot/independent-validation-at-feedback.json"),
            "Feedback is not bound to the actual passed pilot validation")
    at_feedback = read(root / "pilot/independent-validation-at-feedback.json")
    require(at_feedback["status"] == "passed" and at_feedback["protocol_sha256"] == stages["pilot"][1]["protocol_sha256"]
            and at_feedback["inputs_sha256"] == stages["pilot"][1]["inputs_sha256"], "Feedback used a different pilot")
    require((root / "REPORT.zh-CN.md").is_file(), "Final report is missing")
    for protocol, _ in stages.values():
        for name, expected in protocol["runtime_code_sha256"].items():
            require(digest(REPO / name) == expected, "Frozen runtime code changed: " + name)
        require(digest(HERE / "run_ipi_trace_interventions.py") == protocol["runner_sha256"]
                and digest(HERE / "ipi_trace_interventions_data.py") == protocol["parser_sha256"],
                "Frozen inference implementation changed")
    for name, expected in collection["code_sha256"].items():
        require(digest(REPO / name) == expected, "Frozen collection implementation changed: " + name)
    reproduction = root / "reproduction"
    files = set(HERE.glob("*ipi*trace*.py")) | {HERE / "ipi_benign_source_data.py"}
    files.update(HERE / name for name in (
        "run_role_signal_factorial.py", "synthetic_agent_classifier.py", "run_agentdojo_trace_interventions.py",
        "analyze_agentdojo_trace_interventions.py", "plot_agentdojo_trace_token_curves.py", "export_agentdojo_trace_readouts.py",
        "seal_agentdojo_trace_interventions.py"))
    files.update(REPO / name for name in collection["code_sha256"])
    files.update(REPO / name for name in stages["all"][0]["runtime_code_sha256"])
    code = {}
    for source in sorted(files):
        relative = source.relative_to(REPO)
        target = reproduction / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(digest(source) == digest(target), "Code changed while snapshotting")
        code[str(relative)] = digest(target)
    probe = Path(stages["all"][0]["probe"])
    require(digest(probe) == SHA, "Checkpoint changed")
    (reproduction / "checkpoint").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(probe, reproduction / "checkpoint/checkpoint.npz")
    model = Path(stages["all"][0]["model_path"])
    shutil.copyfile(model / "staging-proof.json", reproduction / "model-staging-proof.json")
    import numpy as np
    with np.load(probe, allow_pickle=False) as archive:
        binding = json.loads(str(archive["metadata"]))["model_binding"]
    proof = read(model / "staging-proof.json")
    require(proof["all_shards_verified"] is True
            and {item["name"]: item["sha256"] for item in proof["shards"]} == binding["weight_shards_sha256"],
            "Native model and checkpoint identities differ")
    for name, expected in binding["assets_sha256"].items():
        require(digest(model / name) == expected, "Model/tokenizer asset changed")
        common.snapshot_file(model / name, reproduction / "model-assets" / name)
    for name in ("model.safetensors.index.json", "generation_config.json", "chat_template.jinja"):
        if (model / name).is_file():
            common.snapshot_file(model / name, reproduction / "model-assets" / name)
    packages = {name: importlib.metadata.version(name) for name in common.DEPENDENCIES}
    for name in ("pilot", "all"):
        for package, version in read(root / name / "runtime.json")["packages"].items():
            require(importlib.metadata.version(package) == version, "Inference dependency version changed: " + package)
    libraries = {}
    for relative in common.CRITICAL_LIBRARY_FILES:
        package, rest = relative.split("/", 1)
        spec = importlib.util.find_spec(package)
        source = Path(spec.origin).parent / rest
        if source.is_file():
            libraries[relative] = digest(source)
            common.snapshot_file(source, reproduction / "dependency-source" / relative)
    revision = "9fbc1b48e710ac8abf18ea2650ee690d52dc3859"
    kernel = REPO / "experiments/single-message-pre-mlp/outputs/huggingface/hub/models--kernels-community--gpt-oss-triton-kernels/snapshots" / revision
    require(kernel.is_dir(), "Pinned native kernel implementation is missing")
    kernel_files = {}
    for source in sorted(kernel.rglob("*")):
        if source.is_file() and source.suffix in (".py", ".json", ".toml", ".so") and "__pycache__" not in source.parts:
            relative = source.relative_to(kernel)
            common.snapshot_file(source, reproduction / "kernel-source" / relative)
            kernel_files[str(relative)] = digest(source)
    require(kernel_files, "Pinned kernel snapshot is empty")
    write(reproduction / "environment.json", dict(python=sys.version,executable=sys.executable,
          packages=packages, critical_library_sha256=libraries,kernel_revision=revision,kernel_files_sha256=kernel_files,
          code_sha256=code, native_weights="External; identities recorded in model-staging-proof.json",
          probe_sha256=SHA))
    write(reproduction / "external-materials.json", dict(model_path=str(model),checkpoint_original_path=str(probe),
          model_shards=proof["shards"],model_hash_policy="All shards verified during staging and bound to checkpoint; not rehashed by seal.",
          limitation="Critical Python/kernel source and versions are saved; binary distributions and model weights remain external prerequisites."))
    source_doc = HERE / "ipi_trace_interventions_reproduction.txt"
    require(source_doc.is_file(), "Reproduction instructions are missing")
    shutil.copyfile(source_doc, root / "REPRODUCE.txt")
    payloads = {}
    audited_ids = set().union(*(run["records"] for _,run in stages.values()))
    require({p.stem for p in (root / "records").glob("*.json")} == audited_ids,
            "Orphan inference records are not covered by the full independent audits")
    for input_id in sorted(audited_ids):
        require(Path(input_id).name == input_id, "Unsafe input identity")
        path = root / "records" / (input_id + ".json")
        record = read(path)
        require(record["status"] == "complete", "Uncommitted input record")
        payloads[record["hiddenstates"]] = record["hiddenstates_sha256"]
    require({str(p.relative_to(root)) for p in (root / "hiddenstates").iterdir() if p.is_file()} == set(payloads),
            "Orphan hidden payloads are not covered by the full independent audits")
    inventory = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if relative in ("gpu.lock", "delivery-manifest.json", "SHA256SUMS") or path.suffix in (".log", ".pyc"):
            continue
        require(".tmp" not in path.name, "Uncommitted temporary artifact: " + relative)
        inventory[relative] = dict(bytes=path.stat().st_size,
                                   sha256=payloads[relative] if relative in payloads else digest(path))
    write(root / "delivery-manifest.json", dict(status="sealed", dataset="IPI Arena no-payload continuations",
          sealed_at_utc=datetime.now(timezone.utc).isoformat(), probe_sha256=SHA,
          stages={k:dict(traces=v[1]["traces"],inputs=v[1]["inputs"],token_rows=v[0]["expected_token_rows"],
                        hiddenstate_bytes=v[1]["hiddenstate_bytes"]) for k,v in stages.items()},
          hidden_payload_hash_policy="Reuses full-stream SHA256 checks from independent cap0 validator; all other files hashed at seal.",
          files=inventory))
    checksum_lines = [f"{v['sha256']}  {k}" for k,v in sorted(inventory.items())]
    checksum_lines.append(f"{digest(root / 'delivery-manifest.json')}  delivery-manifest.json")
    (root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    print(json.dumps(dict(status="sealed",files=len(inventory),root=str(root))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    seal(parser.parse_args().output)
