#!/usr/bin/env python3
"""Export clean IPI intervention NPZ readouts and a BF16 hidden-state index.

Adapts the frozen AgentDojo export implementation to dynamic source-role support,
one external IPI cohort, and the 1+7m input design. Saved float32 probabilities and
logits are copied bit for bit; hidden residual payloads are referenced, not copied
or recomputed. Headers, byte sizes, identities and alignment vectors are checked.
Independent validation must separately verify full hidden-state payload hashes.

Usage:
    python export_ipi_trace_readouts.py --output ROOT --stage pilot
    python export_ipi_trace_readouts.py --output ROOT --stage all

No model or tool execution occurs. The original exporter is never edited.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np

import analyze_ipi_trace_interventions as analysis
import export_agentdojo_trace_readouts as native


ROLES = native.ROLES


def role_masks(row):
    """Validate actual native target masks, allowing absent source-role classes."""
    name = row["input_id"]
    count = row["target_token_count"]
    positions = np.asarray(row["target_positions"], dtype=np.int64)
    original = np.asarray(row["target_original_positions"], dtype=np.int64)
    native.require(positions.shape == original.shape == (count,) and count > 0,
                   "Invalid target coordinates: " + name)
    native.require(np.all(np.diff(positions) > 0) and positions[0] >= 0 and positions[-1] < len(row["input_ids"]),
                   "Target coordinates must be unique, ordered input positions")
    ownership = {}
    for role, mask in row["role_target_positions"].items():
        native.require(role in ROLES and len(mask) > 0, "Invalid or empty native source role")
        originals = row["original_role_target_positions"][role]
        native.require(len(mask) == len(originals), "Original body mask length differs")
        native.require(mask == list(range(mask[0], mask[0] + len(mask))), "Target body is not contiguous")
        native.require(originals == list(range(originals[0], originals[0] + len(originals))), "Original body is not contiguous")
        for offset, (position, orig) in enumerate(zip(mask, originals)):
            native.require(position not in ownership and orig >= 0, "Body masks overlap or lack original token ownership")
            ownership[position] = (role, offset, orig)
    native.require(set(ownership) == set(positions.tolist()), "Role masks differ from selected body positions")
    native.require([ownership[int(position)][2] for position in positions] == original.tolist(), "Original body coordinates differ")
    target_roles = row["target_roles"]
    native.require(isinstance(target_roles, list) and len(target_roles) == len(set(target_roles)) and
                   set(target_roles) == set(row["role_target_positions"]) and bool(target_roles), "Target-role inventory differs")
    if row["operation"] == "baseline":
        native.require(row["source_role"] is None and set(target_roles).issubset(ROLES), "Invalid baseline source-role union")
    else:
        native.require(row["source_role"] in ROLES and target_roles == [row["source_role"]], "Variant must target one existing source body")
    native.require(row["operation"] in ("baseline", "strip", "swap", "move"), "Unknown intervention")
    native.require(row["destination_role"] in ROLES and row["destination_role"] != row["source_role"] if row["operation"] == "swap"
                   else row["destination_role"] is None, "Invalid destination role")
    native.require(row["source_split"] == "external", "IPI input must belong to the external cohort")
    return positions, original, ownership


def schema_metadata(schema, *, token: bool):
    metadata = dict(schema.metadata or {})
    metadata[b"format"] = b"ipi-trace-body-token-readouts-v1" if token else b"ipi-trace-hiddenstates-index-v1"
    metadata[b"dataset"] = b"IPI Arena clean continuation"
    metadata[b"source_role_policy"] = b"Only actually present nonempty source bodies; absent roles are missing, not fabricated"
    if token:
        metadata[b"baseline"] = b"Union of actually present first-role bodies; source_role is assigned separately to each token"
    else:
        metadata[b"scope"] = b"Target body positions only; baseline stores the union of actually present first-role bodies"
    return schema.with_metadata(metadata)


@contextmanager
def ipi_export_boundary(expected_inputs, expected_tokens, stage_name):
    old_expected, old_masks = native.EXPECTED, native.role_masks
    old_token, old_hidden, old_example = native.token_schema, native.hidden_schema, native.write_example

    def example(stage, root):
        old_example(stage, root)
        path = stage / "hiddenstates-loading-example.txt"
        path.write_text(path.read_text().replace("union of the six first-role bodies", "union of actually present first-role bodies"))

    native.EXPECTED = {stage_name: (expected_inputs, expected_tokens)}
    native.role_masks = role_masks
    native.token_schema = lambda: schema_metadata(old_token(), token=True)
    native.hidden_schema = lambda: schema_metadata(old_hidden(), token=False)
    native.write_example = example
    try:
        yield
    finally:
        native.EXPECTED, native.role_masks = old_expected, old_masks
        native.token_schema, native.hidden_schema, native.write_example = old_token, old_hidden, old_example


def export(root: Path, stage_name: str):
    root = root.resolve()
    stage = root / stage_name
    frame, validation = analysis.read_and_validate(stage / "input-summary.csv", stage_name)
    rows = list(native.lines(stage / "inputs.jsonl"))
    native.require(len(rows) == validation["unique_inputs"], "Input count differs from the complete dynamic condition design")
    expected_by_trace = {trace: set(group.loc[group.operation == "baseline", "source_role"])
                         for trace, group in frame.groupby("trace_id")}
    baseline_seen = set()
    for row in rows:
        native.require(row["trace_id"] in expected_by_trace, "Unknown input trace")
        role_masks(row)
        if row["operation"] == "baseline":
            native.require(row["trace_id"] not in baseline_seen, "Repeated baseline input")
            baseline_seen.add(row["trace_id"])
            native.require(set(row["target_roles"]) == expected_by_trace[row["trace_id"]], "Baseline union differs from actually present source roles")
        else:
            native.require(row["source_role"] in expected_by_trace[row["trace_id"]], "Intervention fabricates an absent source body")
    native.require(baseline_seen == set(expected_by_trace), "Missing baseline input")
    expected_tokens = sum(row["target_token_count"] for row in rows)
    with ipi_export_boundary(len(rows), expected_tokens, stage_name):
        manifest = export_tables(root, stage_name, rows, expected_tokens, validation)
    return manifest


def export_tables(root, stage_name, rows, expected_tokens, validation):
    """Dynamic IPI driver; native per-input validators and serializers are reused."""
    started = time.monotonic()
    stage = root / stage_name
    run_path, input_path, protocol_path = [stage / name for name in ("run-manifest.json", "inputs.jsonl", "protocol.json")]
    run, protocol = native.read(run_path), native.read(protocol_path)
    expected_inputs = len(rows)
    native.require(run.get("status") == "complete" and run.get("stage") == stage_name and run["inputs"] == expected_inputs,
                   "Inference must be committed complete before export")
    hashes = {str(path.relative_to(root)): native.digest(path) for path in (run_path, input_path, protocol_path)}
    native.require(run["inputs_sha256"] == protocol["inputs_sha256"] == hashes[str(input_path.relative_to(root))] and
                   run["protocol_sha256"] == hashes[str(protocol_path.relative_to(root))], "Frozen input/protocol identity differs")
    native.require(run["probe_sha256"] == protocol["probe_sha256"], "Frozen probe identity differs")
    native.require(run["summary_sha256"] == native.digest(stage / "input-summary.csv"), "Completed summary changed")
    hashes[str((stage / "input-summary.csv").relative_to(root))] = run["summary_sha256"]
    source_relative = protocol.get("source_records_path", "sources/traces.jsonl")
    source_path = native.contained_file(root, source_relative)
    hashes[source_relative] = native.digest(source_path)
    native.require(hashes[source_relative] == protocol["source_records_sha256"], "Frozen source trace file changed")
    if "wrapper_catalog_path" in protocol:
        relative = protocol["wrapper_catalog_path"]
        hashes[relative] = native.digest(native.contained_file(root, relative))
        native.require(hashes[relative] == protocol["wrapper_catalog_sha256"], "Frozen wrapper catalog changed")
    ids = [row["input_id"] for row in rows]
    native.require(len(set(ids)) == expected_inputs and set(ids) == set(run["records"]), "Completed input inventory differs")
    token_format, hidden_format = native.token_schema(), native.hidden_schema()
    token_tmp, hidden_tmp, manifest_tmp = [native.temporary(stage, suffix) for suffix in (".parquet", ".parquet", ".json")]
    token_path, hidden_path = stage / "token-readouts.parquet", stage / "hiddenstates-index.parquet"
    count, hidden_bytes = 0, 0
    readout_hashes, record_hashes = {}, {}
    split_inputs, split_tokens, operations = Counter(), Counter(), Counter()
    try:
        with native.pq.ParquetWriter(token_tmp, token_format, compression="zstd", use_dictionary=list(native.IDENTITIES)) as token_writer, \
                native.pq.ParquetWriter(hidden_tmp, hidden_format, compression="zstd", use_dictionary=list(native.IDENTITIES)) as hidden_writer:
            for row in rows:
                identity = native.read(root / "records" / (row["input_id"] + ".json"))["signature"]
                for key in ("probe_sha256", "parser_sha256", "runner_sha256", "runtime_code_sha256"):
                    native.require(identity[key] == protocol[key], "Input record differs from frozen protocol: " + key)
                table, index_table, record, record_sha = native.input_tables(root, row, run["records"][row["input_id"]], token_format, hidden_format)
                token_writer.write_table(table, row_group_size=table.num_rows)
                hidden_writer.write_table(index_table, row_group_size=1)
                count += table.num_rows
                hidden_bytes += record["hiddenstate_bytes"]
                readout_hashes[record["readout"]] = record["readout_sha256"]
                record_hashes["records/" + row["input_id"] + ".json"] = record_sha
                split_inputs[row["source_split"]] += 1
                split_tokens[row["source_split"]] += table.num_rows
                operations[row["operation"]] += 1
        native.require(count == expected_tokens and hidden_bytes == run["hiddenstate_bytes"], "Export totals differ from completed run")
        token_meta, hidden_meta = native.pq.read_metadata(token_tmp), native.pq.read_metadata(hidden_tmp)
        native.require(token_meta.num_rows == expected_tokens and hidden_meta.num_rows == expected_inputs and
                       token_meta.num_row_groups == hidden_meta.num_row_groups == expected_inputs, "Parquet counts differ")
        native.require(native.pq.read_schema(token_tmp).equals(token_format, check_metadata=True) and
                       native.pq.read_schema(hidden_tmp).equals(hidden_format, check_metadata=True), "Parquet schema changed")
        for relative, expected_sha in {**hashes, **record_hashes}.items():
            native.require(native.digest(root / relative) == expected_sha, "Source changed during export: " + relative)
        manifest = dict(
            format_version="ipi-trace-readouts-export-v1", dataset="IPI Arena", cohort="external", status="complete", stage=stage_name,
            inputs=expected_inputs, token_rows=count, hiddenstate_files=expected_inputs, readout_files=len(readout_hashes),
            input_counts_by_split=dict(split_inputs), token_rows_by_split=dict(split_tokens), input_counts_by_operation=dict(operations),
            hiddenstate_bytes=hidden_bytes, hiddenstate_gib=hidden_bytes / (1 << 30),
            source_hashes=hashes, readout_hashes=readout_hashes, record_hashes=record_hashes,
            source_records_path=source_relative, exporter_sha256=native.digest(Path(__file__)),
            reused_exporter_sha256=native.digest(Path(native.__file__)), dynamic_design_validation=validation,
            token_readouts=dict(path=str(token_path.relative_to(root)), sha256=native.digest(token_tmp),
                                row_groups=token_meta.num_row_groups, columns={field.name: str(field.type) for field in token_format}),
            hiddenstates_index=dict(path=str(hidden_path.relative_to(root)), sha256=native.digest(hidden_tmp),
                                    row_groups=hidden_meta.num_row_groups, columns={field.name: str(field.type) for field in hidden_format}),
            roles=list(ROLES), coordinates="Zero-based input and original token coordinates; body_token_offset restarts per source body",
            baseline_policy="Baseline stores the union of actually present first-role bodies; source_role is assigned per token. Missing source bodies are not fabricated.",
            destination_role_policy="Original inputs.jsonl value: destination role for swaps, null otherwise",
            value_policy="Saved float32 logits and probabilities copied directly without loading or recomputing the model/probe",
            hiddenstate_verification="File lengths, 43 tensor headers, signatures and small coordinate vectors checked; payload hashes copied from completed records",
            hidden_payload_sha256_recomputed=False, hiddenstate_payloads_duplicated=False,
            elapsed_seconds=time.monotonic() - started, software=dict(numpy=np.__version__, pyarrow=native.pa.__version__),
        )
        manifest_tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        os.replace(token_tmp, token_path)
        os.replace(hidden_tmp, hidden_path)
        os.replace(manifest_tmp, stage / "export-manifest.json")
        native.write_example(stage, root)
        print(json.dumps({key: manifest[key] for key in ("status", "stage", "inputs", "token_rows", "hiddenstate_files", "hiddenstate_gib")}))
        return manifest
    finally:
        for path in (token_tmp, hidden_tmp, manifest_tmp):
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("pilot", "all"))
    args = parser.parse_args()
    export(args.output, args.stage)


if __name__ == "__main__":
    main()
