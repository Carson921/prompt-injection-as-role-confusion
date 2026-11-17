"""Seal completed, exported and independently audited trace interventions.

This tool neither runs inference nor edits a frozen implementation. It hashes
small artifacts and readouts, snapshots the implementation and dependency
identities, and reuses hidden-state hashes that the independent full-payload
validator has already checked. No hidden-state payload is read by this tool.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import shlex
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROBE_SHA = "fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33"
EXPECTED = {"pilot": (1, 43, 6032), "all": (57, 2451, 719832)}
FINAL_SUFFIXES = {".json", ".jsonl", ".csv", ".npz", ".parquet", ".png", ".pdf", ".svg", ".txt", ".md"}
SOURCE_FILES = ("manifest.json", "traces.jsonl", "source-manifest.json", "source-verification.json",
                "pilot-selection.json", "reconstruction-audit.json")
STAGE_FILES = ("inputs.jsonl", "protocol.json", "runtime.json", "run-manifest.json", "input-summary.csv",
               "independent-validation.json", "export-manifest.json", "token-readouts.parquet",
               "export-independent-validation.json", "hiddenstates-index.parquet", "hiddenstates-loading-example.txt",
               "report-manifest.json", "figure-caption.en.txt")
DEPENDENCIES = ("torch", "transformers", "accelerate", "kernels", "triton", "safetensors", "numpy",
                "pandas", "pyarrow", "matplotlib", "tokenizers", "huggingface-hub")
CRITICAL_LIBRARY_FILES = (
    "torch/version.py", "transformers/models/gpt_oss/modeling_gpt_oss.py",
    "transformers/models/gpt_oss/configuration_gpt_oss.py", "transformers/modeling_utils.py",
    "transformers/cache_utils.py", "transformers/integrations/hub_kernels.py",
    "transformers/integrations/mxfp4.py", "transformers/quantizers/quantizer_mxfp4.py",
    "transformers/tokenization_utils_fast.py", "transformers/tokenization_utils_tokenizers.py",
    "transformers/tokenization_utils_base.py", "transformers/masking_utils.py", "transformers/modeling_layers.py",
    "transformers/modeling_rope_utils.py", "transformers/processing_utils.py", "transformers/utils/hub.py",
    "accelerate/hooks.py", "accelerate/big_modeling.py", "accelerate/utils/operations.py",
    "safetensors/__init__.py", "safetensors/torch.py", "kernels/__init__.py",
    "numpy/version.py", "pandas/__init__.py", "pyarrow/__init__.py", "matplotlib/__init__.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    require(Path(path).is_file(), "Required completed artifact is missing: " + str(path))
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def child(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "Invalid relative artifact path")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Artifact escapes the output root: " + relative)
    require(path.is_file(), "Required artifact is missing: " + relative)
    return path


def check_hash(root, relative, expected):
    require(digest(child(root, relative)) == expected, "Artifact checksum mismatch: " + relative)


def stage_gate(root, stage_name, complete_report):
    stage = root / stage_name
    n_traces, n_inputs, n_tokens = EXPECTED[stage_name]
    run, protocol = read(stage / "run-manifest.json"), read(stage / "protocol.json")
    validation, exported = read(stage / "independent-validation.json"), read(stage / "export-manifest.json")
    statistics = read(stage / "analysis/independent-statistics-validation.json")
    require(run.get("status") == "complete" and run.get("stage") == stage_name and
            run.get("inputs") == n_inputs and run.get("traces") == n_traces, "Stage inference is not committed complete: " + stage_name)
    require(run["probe_sha256"] == protocol["probe_sha256"] == protocol["expected_probe_sha256"] == PROBE_SHA,
            "Frozen probe identity differs")
    for field, filename in (("inputs_sha256", "inputs.jsonl"), ("protocol_sha256", "protocol.json"),
                            ("summary_sha256", "input-summary.csv")):
        check_hash(root, f"{stage_name}/{filename}", run[field])
    require(validation.get("status") == "passed" and validation.get("stage") == stage_name and
            validation["checkpoint_sha256"] == PROBE_SHA and
            validation["protocol_sha256"] == run["protocol_sha256"] and
            validation["inputs_sha256"] == run["inputs_sha256"], "Independent validation is missing or bound to different inputs")
    require(validation["inputs"]["status"] == "passed" and validation["inputs"]["inputs"] == n_inputs and
            validation["inputs"]["traces"] == n_traces, "Independent input validation does not cover the whole stage")
    for flag in ("all_probe_rows_independently_recomputed", "hiddenstate_full_payload_sha256_all_verified",
                 "all36_residual_layers_plus_embedding_final_norm", "native_bf16_to_checkpoint_fp16_bit_exact",
                 "same_body_embedding_rows_bit_exact_across_all_interventions"):
        require(validation.get(flag) is True, "Required full independent audit did not pass: " + flag)
    totals = validation["totals"]
    require(validation["independent_probe_row_cap"] == 0 and
            totals["saved_body_token_rows"] == totals["independent_probe_token_rows"] == n_tokens and
            totals["hiddenstate_tensors"] == 43 * n_inputs and totals["residual_layer_tensors"] == 36 * n_inputs and
            totals["hiddenstate_bytes"] == run["hiddenstate_bytes"] and validation["summary_rows"] == 48 * n_traces,
            "Independent audit is sampled, incomplete, or has different denominators")
    if stage_name == "all":
        require(validation["validator_sha256"] == digest(HERE / "validate_agentdojo_trace_interventions.py"),
                "Full-stage independent validator source differs from the audited version")
    require(exported.get("status") == "complete" and exported.get("stage") == stage_name and
            exported["inputs"] == exported["hiddenstate_files"] == exported["readout_files"] == n_inputs and
            exported["token_rows"] == n_tokens and exported["hiddenstate_bytes"] == run["hiddenstate_bytes"],
            "Readout/hidden-state exports are not complete")
    if stage_name == "all":
        require(exported["exporter_sha256"] == digest(HERE / "export_agentdojo_trace_readouts.py"),
                "Final exporter differs from the recorded export implementation")
    for relative, expected in exported["source_hashes"].items():
        check_hash(root, relative, expected)
    for artifact in ("token_readouts", "hiddenstates_index"):
        check_hash(root, exported[artifact]["path"], exported[artifact]["sha256"])
    export_audit = read(stage / "export-independent-validation.json")
    require(export_audit.get("status") == "passed" and export_audit.get("stage") == stage_name and
            export_audit["inputs"] == export_audit["original_npz_files_compared"] == n_inputs and
            export_audit["token_rows"] == n_tokens and export_audit["differing_float32_bit_patterns"] == 0,
            "Independent Parquet/NPZ comparison is missing or incomplete")
    for flag in ("all_saved_token_rows_compared", "coordinates_and_token_ids_bit_exact", "saved_logits_bit_exact",
                 "saved_probabilities_bit_exact", "floats_compared_as_uint32_bits_including_signed_zero",
                 "baseline_and_variant_role_masks_exact", "body_token_offsets_exact", "hiddenstate_index_matches_committed_records",
                 "exporter_not_imported"):
        require(export_audit.get(flag) is True, "Independent export comparison did not prove: " + flag)
    require(export_audit["run_manifest_sha256"] == digest(stage / "run-manifest.json") and
            export_audit["source_traces_sha256"] == protocol["source_records_sha256"] and
            export_audit["export_manifest_sha256"] == digest(stage / "export-manifest.json") and
            export_audit["hiddenstate_index_rows_checked"] == n_inputs and
            export_audit["hiddenstate_index_bytes"] == run["hiddenstate_bytes"],
            "Independent export comparison refers to different committed artifacts")
    if stage_name == "all":
        require(export_audit["validator_sha256"] == digest(HERE / "validate_agentdojo_trace_readout_export.py"),
                "Independent export checker differs from its audited implementation")
    for relative, expected in export_audit["source_hashes"].items():
        check_hash(root, relative, expected)
    require(statistics.get("status") == "passed" and statistics.get("stage") == stage_name and
            statistics["input_summary_sha256"] == run["summary_sha256"] and statistics["analyzer_not_imported"] is True,
            "Independent statistics audit is incomplete or bound to a different summary")
    expected_cohorts = {"pilot": 1} if stage_name == "pilot" else {"fit": 40, "all": 57}
    require({item["cohort"]: item["traces"] for item in statistics["cohorts"]} == expected_cohorts,
            "Independent statistics cohort support differs")
    if stage_name == "all":
        require(statistics["auditor_sha256"] == digest(HERE / "audit_agentdojo_trace_intervention_statistics.py"),
                "Statistics auditor differs from the audited source")
    for cohort in statistics["cohorts"]:
        require(cohort["status"] == "passed" and cohort["scalar_checks"] > 0, "A statistics cohort audit failed")
        if cohort["traces"] > 1:
            require(cohort["bootstrap_resamples"] == 2000 and
                    cohort["paired_pointwise_95_percent_intervals_all_recomputed"] is True,
                    "Bootstrap intervals were not fully audited")
        for filename, expected in cohort["files_sha256"].items():
            check_hash(root, f"{stage_name}/analysis/{cohort['cohort']}/{filename}", expected)
    report = read(stage / "report-manifest.json")
    require(report["stage"] == stage_name and report["traces"] == n_traces and
            report["source_sha256"] == protocol["source_records_sha256"] and
            report["input_summary_sha256"] == run["summary_sha256"], "Report support/source binding differs")
    check_hash(root, report["report"], report["report_sha256"])
    child(root, report["caption"])
    for key, value in report.items():
        if isinstance(value, str) and key + "_sha256" in report:
            check_hash(root, value, report[key + "_sha256"])
    if complete_report:
        # The report schema has no status field. Completion is established by
        # its actual files and its bindings to all passed audit manifests.
        for field, filename in (("independent_validation_sha256", "independent-validation.json"),
                                ("independent_statistics_validation_sha256", "analysis/independent-statistics-validation.json"),
                                ("export_manifest_sha256", "export-manifest.json")):
            require(report.get(field) == digest(stage / filename), "Final report lacks a current passed-audit binding: " + field)
        for cohort in expected_cohorts:
            folder = stage / "analysis" / cohort
            for stem in ("role-interventions-main", "wrapper-destination-attraction", "full-six-role-probabilities", "role-token-curves"):
                for extension in ("png", "pdf", "svg"):
                    require((folder / f"{stem}.{extension}").is_file(), "Missing final English figure: " + str(folder / stem))
            curves = read(folder / "token-curve-validation.json")
            require(curves["inputs_sha256"] == run["inputs_sha256"] and curves["input_summary_sha256"] == run["summary_sha256"] and
                    curves["all_curve_means_match_primary_summary"] is True and
                    curves["token_ids_and_original_positions_identical_across_conditions"] is True,
                    "Supplemental token curves are not validated against the primary readouts")
    return run, protocol, validation, exported


def gates(root):
    # Fail before snapshotting or writing anything if the all-stage run has not
    # reached its committed final state.
    run = read(root / "all/run-manifest.json")
    require(run.get("status") == "complete" and run.get("inputs") == 2451, "All 2451 inputs must finish before sealing")
    result = {name: stage_gate(root, name, complete_report=(name == "all")) for name in EXPECTED}
    source = read(root / "sources/manifest.json")
    require(source["status"] == "complete" and source["splits"] == {"fit": 40, "dev": 8, "test": 9} and
            source["traces"] == 57 and source["calls"] == 359 and source["native_passes"] == 58,
            "Frozen native source inventory differs")
    require(source["source_verification"]["status"] == "passed" and source["first_role_training_lineage_verified"] == 342,
            "Frozen source lineage was not verified")
    require(source["script_sha256"] == digest(HERE / "prepare_agentdojo_trace_sources.py"),
            "Frozen source reconstruction implementation changed")
    for relative, expected in source["frozen_source_files"].items():
        check_hash(root, "sources/" + relative, expected)
    feedback = read(root / "pilot/feedback.json")
    require(feedback["feedback_delivered"] is True and
            feedback["pilot_run_manifest_sha256"] == digest(root / "pilot/run-manifest.json"),
            "Single-trace feedback is not attested against the actual completed pilot")
    check_hash(root, "pilot/independent-validation-at-feedback.json", feedback["pilot_validation_sha256"])
    return result, source


def snapshot_file(source, destination):
    require(source.is_file(), "Missing source/dependency file: " + str(source))
    expected = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    require(digest(destination) == expected and digest(source) == expected, "Source changed during snapshot: " + str(source))
    return dict(original_path=str(source.resolve()), snapshot_path=str(destination), bytes=source.stat().st_size, sha256=expected)


def freeze_environment(root, protocols, source):
    destination = root / "reproduction"
    code = destination / "code"
    files = set(HERE.glob("*agentdojo*trace*.py"))
    files.update(HERE / name for name in ("run_role_signal_factorial.py", "synthetic_agent_classifier.py"))
    files.update(REPO / name for name in ("experiments/agentdojo-first-tool/collect.py",
                                        "experiments/agentdojo-first-tool/attention.py", "experiments/agentdojo-first-tool/spans.py",
                                        "experiments/autodan-role-probe/scorer.py"))
    for protocol in protocols:
        require(digest(HERE / "agentdojo_trace_interventions_data.py") == protocol["parser_sha256"] and
                digest(HERE / "run_agentdojo_trace_interventions.py") == protocol["runner_sha256"], "Frozen implementation changed")
        for name, expected in protocol["runtime_code_sha256"].items():
            require(digest(REPO / name) == expected, "Bound runtime dependency changed: " + name)
            files.add(REPO / name)
    snapshots = {}
    for path in sorted(files):
        relative = path.relative_to(REPO)
        item = snapshot_file(path, code / relative)
        item["snapshot_path"] = str((code / relative).relative_to(root))
        snapshots[str(relative)] = item
    doc = HERE / "agentdojo_trace_interventions_reproduction.txt"
    snapshot_file(doc, root / "REPRODUCE.txt")
    model, probe = Path(protocols[-1]["model_path"]), Path(protocols[-1]["probe"])
    require(digest(probe) == PROBE_SHA, "Actual checkpoint changed before sealing")
    for name in ("checkpoint.npz", "split-assignment.json", "agentdojo-source-audit.json", "protocol.json"):
        snapshot_file(probe.parent / name, destination / "checkpoint" / name)
    proof = read(model / "staging-proof.json")
    require(proof["all_shards_verified"] is True, "Model staging lacks prior complete shard verification")
    for stage_name in EXPECTED:
        require(read(root / stage_name / "runtime.json")["staging_proof_sha256"] == digest(model / "staging-proof.json"),
                "Model staging proof differs from the actual inference runtime")
    import numpy as np
    with np.load(probe, allow_pickle=False) as archive:
        binding = json.loads(str(archive["metadata"]))["model_binding"]
    require({item["name"]: item["sha256"] for item in proof["shards"]} == binding["weight_shards_sha256"],
            "Checkpoint and prior model-weight verification refer to different shards")
    external_model = []
    for item in proof["shards"]:
        path = model / item["name"]
        require(path.is_file() and path.stat().st_size == item["bytes"], "Staged model shard size changed")
        external_model.append(dict(path=str(path.resolve()), bytes=item["bytes"], sha256=item["sha256"],
                                   sha256_source="prior staging proof matched to frozen checkpoint binding; payload not rehashed by seal"))
    for name, expected in binding["assets_sha256"].items():
        require(digest(model / name) == expected, "Small model/tokenizer asset changed: " + name)
        snapshot_file(model / name, destination / "model-assets" / name)
    for name in ("staging-proof.json", "model.safetensors.index.json", "generation_config.json", "chat_template.jinja"):
        if (model / name).is_file():
            snapshot_file(model / name, destination / "model-assets" / name)
    original_source = Path(source["source_verification"]["source_root"])
    require(digest(original_source / "result_manifest.json") == source["source_verification"]["result_manifest_sha256"],
            "Original native result manifest changed")
    snapshot_file(original_source / "result_manifest.json", destination / "native-source-result-manifest.json")
    archive_path = Path(source["source_verification"]["archive"])
    require(archive_path.is_file(), "Locked user-uploaded native archive is unavailable")
    distributions = {}
    for name in DEPENDENCIES:
        distribution = metadata.distribution(name)
        package_info = dict(version=distribution.version, installation_root=str(distribution.locate_file("").resolve()),
                            distribution_metadata={})
        for relative in distribution.files or ():
            if relative.name in ("METADATA", "WHEEL", "RECORD") and any(part.endswith(".dist-info") for part in relative.parts):
                path = distribution.locate_file(relative)
                item = snapshot_file(path, destination / "distribution-metadata" / name / relative.name)
                item["snapshot_path"] = str((destination / "distribution-metadata" / name / relative.name).relative_to(root))
                package_info["distribution_metadata"][relative.name] = item
        distributions[name] = package_info
    for stage_name in EXPECTED:
        for name, version in read(root / stage_name / "runtime.json")["packages"].items():
            require(metadata.version(name) == version, "Current environment differs from saved inference runtime: " + name)
    library_files = {}
    base = Path(metadata.distribution("transformers").locate_file(""))
    for relative in CRITICAL_LIBRARY_FILES:
        path = base / relative
        if path.is_file():
            item = snapshot_file(path, destination / "dependency-source" / relative)
            item["snapshot_path"] = str((destination / "dependency-source" / relative).relative_to(root))
            library_files[relative] = item
    kernel_revision = "9fbc1b48e710ac8abf18ea2650ee690d52dc3859"
    kernel = REPO / "experiments/single-message-pre-mlp/outputs/huggingface/hub/models--kernels-community--gpt-oss-triton-kernels/snapshots" / kernel_revision
    require(kernel.is_dir(), "Pinned native MXFP4 kernel source cache is unavailable")
    kernel_files = {}
    for path in sorted(kernel.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".json", ".toml", ".so") and "__pycache__" not in path.parts:
            relative = path.relative_to(kernel)
            item = snapshot_file(path, destination / "kernel-source" / relative)
            item["snapshot_path"] = str((destination / "kernel-source" / relative).relative_to(root))
            kernel_files[str(relative)] = item
    require(kernel_files, "No pinned kernel implementation files were snapshotted")
    all_packages = sorted({(item.metadata.get("Name", "unknown"), item.version) for item in metadata.distributions()})
    (destination / "requirements.freeze.txt").write_text("".join(f"{name}=={version}\n" for name, version in all_packages))
    environment = dict(python=sys.version, executable=sys.executable, executable_resolved=str(Path(sys.executable).resolve()),
                       prefix=sys.prefix, platform=platform.platform(), machine=platform.machine(),
                       runtime_by_stage={name: read(root / name / "runtime.json") for name in EXPECTED},
                       distributions=distributions, kernel_revision=kernel_revision,
                       declared_native_requirement="Exactly two visible H100 GPUs; native MXFP4, Torch 2.9.1+cu128 / CUDA 12.8, eager attention tiled by 256 queries.",
                       environment_variables_recorded="Only experiment-declared cache paths; no ambient environment or credentials captured.",
                       declared_cache_paths=dict(HF_HOME=str(REPO / "experiments/single-message-pre-mlp/outputs/huggingface"),
                                                 TRITON_CACHE_DIR="/tmp/wz55191-agentdojo-trace-interventions-triton"))
    write(destination / "environment.json", environment)
    write(destination / "source-snapshots.json", snapshots)
    write(destination / "dependency-snapshots.json", dict(library_files=library_files,
                                                         declared_library_files_not_present=[name for name in CRITICAL_LIBRARY_FILES if name not in library_files],
                                                         kernel_source_path=str(kernel.resolve()), kernel_revision=kernel_revision,
                                                         kernel_files=kernel_files,
                                                         limitation="Critical Python/kernel implementation files and package versions are frozen. Full installed binary packages and model weights remain external prerequisites; this is not a container image."))
    write(destination / "external-materials.json", dict(
        model_path=str(model.resolve()), model_shards=external_model,
        uploaded_archive=dict(path=str(archive_path.resolve()), bytes=archive_path.stat().st_size,
                              sha256=source["source_verification"]["archive_sha256"],
                              sha256_source="locked uploaded archive verified during source reconstruction; seal does not rehash its payload"),
        native_source_root=str(original_source.resolve()), checkpoint_original_path=str(probe.resolve())))
    return environment


def reproduction_commands(root, protocol):
    # Resolving a venv Python symlink would bypass that environment's packages.
    python, scripts = str(Path(sys.executable).absolute()), str(HERE)
    commands = {}
    for stage in EXPECTED:
        common = ["--output", str(root), "--stage", stage]
        commands[stage] = [[python, str(HERE / name), *common] for name in
                           ("validate_agentdojo_trace_interventions.py", "export_agentdojo_trace_readouts.py",
                            "validate_agentdojo_trace_readout_export.py",
                            "analyze_agentdojo_trace_interventions.py", "audit_agentdojo_trace_intervention_statistics.py",
                            "plot_agentdojo_trace_token_curves.py", "report_agentdojo_trace_interventions.py")]
    commands["seal"] = [python, str(Path(__file__).resolve()), "--output", str(root)]
    commands["full_recomputation"] = dict(
        scripts_path=scripts, model_path=protocol["model_path"], checkpoint_path=protocol["probe"],
        source_script="prepare_agentdojo_trace_sources.py", inference_script="run_agentdojo_trace_interventions.py",
        order="Reconstruct into a fresh output, prepare/run/validate/analyze/report the pilot, deliver the pilot feedback, then prepare/run/postprocess all. The runner requires an honest pilot feedback attestation before all-stage preparation.",
        numerical_limit="Same native token sequences and fixed inference specification are reproducible; cross-backend bitwise probability identity is not claimed.")
    write(root / "reproduction/commands.json", commands)
    text = ["#!/usr/bin/env bash", "set -euo pipefail",
            "# Rebuild CPU analyses from existing caches; full validation rereads every hidden-state payload.",
            "# This rewrites derived artifacts, so rerun the seal command after completing all commands."]
    for stage in EXPECTED:
        text.extend(shlex.join(command) for command in commands[stage])
    text.append(shlex.join(commands["seal"]))
    (root / "reproduction/reproduce-postprocessing.sh").write_text("\n".join(text) + "\n")


def inventory(root, stage_info):
    files = {root / name for name in ("README.zh-CN.md", "PILOT.zh-CN.md", "REPRODUCE.txt")}
    files.update(root / "sources" / name for name in SOURCE_FILES)
    for stage_name in EXPECTED:
        files.update(root / stage_name / name for name in STAGE_FILES)
        report = read(root / stage_name / "report-manifest.json")
        for key, value in report.items():
            if isinstance(value, str) and (key + "_sha256" in report or key == "caption"):
                files.add(child(root, value))
        files.update(path for path in (root / stage_name / "analysis").rglob("*") if path.is_file() and
                     path.suffix in FINAL_SUFFIXES and not any(part.startswith(".") for part in path.relative_to(root).parts))
        for optional in ("feedback.json", "independent-input-validation.json", "independent-validation-at-feedback.json", "passive-hook-validation.json"):
            path = root / stage_name / optional
            if path.is_file():
                files.add(path)
    files.update(path for path in (root / "reproduction").rglob("*") if path.is_file() and path.suffix not in (".pyc", ".log", ".tmp"))
    small, hidden = {}, {}
    run, _, validation, exported = stage_info["all"]
    export_audit = read(root / "all/export-independent-validation.json")
    audit_path = root / "all/independent-validation.json"
    audit_sha, audit_mtime = digest(audit_path), audit_path.stat().st_mtime_ns
    for key, commitment in run["records"].items():
        require(Path(key).name == key, "Unsafe readout key")
        record_path = root / "records" / (key + ".json")
        record = read(record_path)
        require(record["status"] == "complete" and record["input_id"] == key and
                commitment == {field: record[field] for field in ("readout_sha256", "hiddenstates_sha256")},
                "Run manifest and audited record hashes differ")
        relative_record = str(record_path.relative_to(root))
        require(exported["record_hashes"][relative_record] == digest(record_path), "Record changed since export")
        require(export_audit["record_hashes"][relative_record] == digest(record_path),
                "Record changed after the independent Parquet comparison")
        require(record_path.stat().st_mtime_ns <= audit_mtime,
                "An inference record was modified after the passed full audit")
        readout = child(root, record["readout"])
        require(exported["readout_hashes"][record["readout"]] == record["readout_sha256"], "Export and record readout hashes differ")
        require(export_audit["readout_hashes"][record["readout"]] == record["readout_sha256"],
                "Independent Parquet comparison and readout hashes differ")
        hidden_path = child(root, record["hiddenstates"])
        stat = hidden_path.stat()
        require(stat.st_size == record["hiddenstate_bytes"] and stat.st_mtime_ns <= audit_mtime,
                "Hidden-state file length changed or its modification postdates the passed full audit")
        hidden[record["hiddenstates"]] = dict(bytes=stat.st_size, sha256=record["hiddenstates_sha256"],
                                              sha256_method="independent_full_payload_audit_then_run_record_export_binding",
                                              audit_evidence="all/independent-validation.json", audit_evidence_sha256=audit_sha,
                                              payload_rehashed_by_seal=False)
        files.update((record_path, readout))
    require(len(hidden) == 2451 and sum(item["bytes"] for item in hidden.values()) == run["hiddenstate_bytes"] == validation["totals"]["hiddenstate_bytes"],
            "Audited hidden-state inventory is incomplete")
    for path in sorted(files):
        require(path.is_file(), "Missing final deliverable: " + str(path))
        relative = str(path.relative_to(root))
        require("source-recovery" not in path.relative_to(root).parts and path.suffix not in (".log", ".tmp"),
                "Temporary recovery/log artifact cannot be sealed")
        value = digest(path)
        if relative.startswith("readouts/"):
            require(value == run["records"][path.stem]["readout_sha256"], "Small readout checksum mismatch")
        small[relative] = dict(bytes=path.stat().st_size, sha256=value, sha256_method="actual_file_hash_at_seal")
    return small, hidden


def seal(root, check_only=False):
    root = root.resolve()
    stage_info, source = gates(root)
    if check_only:
        print(json.dumps(dict(status="passed", action="completion_gates_only", output=str(root),
                              hidden_payloads_read=False, files_written=False)))
        return
    freeze_environment(root, [stage_info[name][1] for name in EXPECTED], source)
    reproduction_commands(root, stage_info["all"][1])
    small, hidden = inventory(root, stage_info)
    # Recheck all source gates after snapshotting to avoid sealing a changed
    # report, run commitment or audit result.
    gates(root)
    entries = {**small, **hidden}
    sums = "".join(f"{item['sha256']}  {relative}\n" for relative, item in sorted(entries.items()))
    sums_path = root / "SHA256SUMS"
    sums_temporary = sums_path.with_name(sums_path.name + ".tmp")
    sums_temporary.write_text(sums)
    sums_temporary.replace(sums_path)
    manifest = dict(format_version="agentdojo-trace-interventions-delivery-v1", status="complete",
                    sealed_at_utc=datetime.now(timezone.utc).isoformat(), output_root=str(root),
                    traces=57, source_splits={"fit": 40, "dev": 8, "test": 9},
                    inputs=2451, token_readout_rows=719832, hiddenstate_files=2451,
                    checkpoint_sha256=PROBE_SHA, hiddenstate_bytes=sum(item["bytes"] for item in hidden.values()),
                    source_traces_sha256=source["traces_sha256"],
                    gates={name: dict(run_manifest_sha256=digest(root / name / "run-manifest.json"),
                                     independent_validation_sha256=digest(root / name / "independent-validation.json"),
                                     export_manifest_sha256=digest(root / name / "export-manifest.json"),
                                     export_independent_validation_sha256=digest(root / name / "export-independent-validation.json"),
                                     statistics_validation_sha256=digest(root / name / "analysis/independent-statistics-validation.json"),
                                     report_manifest_sha256=digest(root / name / "report-manifest.json")) for name in EXPECTED},
                    files=entries, file_count=len(entries),
                    independent_export_validation=dict(path="all/export-independent-validation.json",
                                                       sha256=digest(root / "all/export-independent-validation.json"),
                                                       inputs=2451, token_rows=719832,
                                                       logits_and_probabilities_float32_bit_exact=True,
                                                       coordinates_and_token_ids_exact=True,
                                                       hiddenstate_payloads_reread=False),
                    sha256sums=dict(path="SHA256SUMS", sha256=digest(sums_path)),
                    hidden_hash_policy="Hashes were fully verified by the passed independent validator, then matched across committed run, records and export. Seal checks current sizes and absence of modification after that audit, without reading hidden-state payloads again.",
                    model_weight_hash_policy="Prior staging-proof hashes bound to the supplied checkpoint; payload sizes checked and large weights not rehashed during sealing.",
                    excluded=["source-recovery/", "sources/*.ready.*", "run.log", "status.json", "*.tmp", "__pycache__/"],
                    self_excluded="delivery-manifest.json cannot include its own SHA256; SHA256SUMS lists all other sealed payloads.",
                    report_completion_policy="Report schema has no status flag; final report completion is proved by current report/caption files and hashes binding it to the passed whole-stage inference/export/independent/statistics audits.")
    write(root / "delivery-manifest.json", manifest)
    print(json.dumps(dict(status="complete", files=len(entries), hiddenstate_files=len(hidden),
                          hiddenstate_bytes=manifest["hiddenstate_bytes"], hidden_payloads_read=False,
                          manifest=str(root / "delivery-manifest.json"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Check final completion gates without snapshotting or writing")
    args = parser.parse_args()
    try:
        seal(args.output, args.check_only)
    except ValueError as error:
        print(json.dumps(dict(status="not_sealed", reason=str(error)), ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
