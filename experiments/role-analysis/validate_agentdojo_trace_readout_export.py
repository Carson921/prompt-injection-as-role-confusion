"""Independently compare every exported token row against original NPZ arrays.

No exporter, runner, analysis code, probe, Torch or safetensors reader is
imported. Float32 values are compared as uint32 bit patterns, including signed
zero. Parquet row groups are streamed one input at a time. Hidden-state paths,
sizes and recorded hashes are checked through the index and completed run;
their large payloads are neither opened nor hashed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
EXPECTED = {"pilot": (43, 6032), "all": (2451, 719832)}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            result.update(block)
    return result.hexdigest()


def read(path):
    require(Path(path).is_file(), "Missing required artifact: " + str(path))
    return json.loads(Path(path).read_text())


def jsonlines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def relative_file(root, value):
    require(isinstance(value, str) and value and not Path(value).is_absolute(), "Invalid artifact reference")
    path = root / value
    require(path.resolve().is_relative_to(root) and path.is_file(), "Missing or escaping artifact reference: " + value)
    return path


def numeric_column(table, name, dtype):
    column = table.column(name)
    require(column.null_count == 0, "Unexpected null exported scalar: " + name)
    values = column.combine_chunks().to_numpy(zero_copy_only=False)
    require(values.dtype == np.dtype(dtype), "Parquet scalar dtype differs: " + name)
    return values


def validate(root, stage_name):
    root = root.resolve()
    stage = root / stage_name
    expected_inputs, expected_tokens = EXPECTED[stage_name]
    started = time.monotonic()
    run = read(stage / "run-manifest.json")
    require(run.get("status") == "complete" and run.get("stage") == stage_name and run["inputs"] == expected_inputs,
            "Stage inference must be committed complete before export validation")
    exported = read(stage / "export-manifest.json")
    require(exported.get("status") == "complete" and exported.get("stage") == stage_name and
            exported["inputs"] == exported["hiddenstate_files"] == expected_inputs and
            exported["token_rows"] == expected_tokens, "Stage export must be committed complete")
    protocol = read(stage / "protocol.json")
    paths = [stage / name for name in ("run-manifest.json", "protocol.json", "inputs.jsonl", "input-summary.csv", "export-manifest.json")]
    paths += [root / "sources/traces.jsonl", stage / "token-readouts.parquet", stage / "hiddenstates-index.parquet"]
    bindings = {str(path.relative_to(root)): sha(path) for path in paths}
    require(bindings[f"{stage_name}/inputs.jsonl"] == run["inputs_sha256"] == protocol["inputs_sha256"] and
            bindings[f"{stage_name}/protocol.json"] == run["protocol_sha256"] and
            bindings[f"{stage_name}/input-summary.csv"] == run["summary_sha256"] and
            bindings["sources/traces.jsonl"] == protocol["source_records_sha256"], "Completed source/run identity differs")
    for field in ("token_readouts", "hiddenstates_index"):
        item = exported[field]
        require(bindings[item["path"]] == item["sha256"], "Committed exported Parquet checksum differs")
    for relative, expected in exported["source_hashes"].items():
        require(sha(relative_file(root, relative)) == expected, "Export source binding changed: " + relative)

    token_file = pq.ParquetFile(stage / "token-readouts.parquet")
    hidden_file = pq.ParquetFile(stage / "hiddenstates-index.parquet")
    require(token_file.metadata.num_rows == expected_tokens and hidden_file.metadata.num_rows == expected_inputs and
            token_file.num_row_groups == hidden_file.num_row_groups == expected_inputs,
            "Exported Parquet stage-wide row counts differ")
    token_columns = {"input_id", "trace_id", "source_split", "operation", "source_role", "destination_role",
                     "token_position", "original_token_position", "token_id", "body_token_offset"}
    token_columns.update(prefix + role for prefix in ("z_", "p_") for role in ROLES)
    require(set(token_file.schema_arrow.names) == token_columns, "Unexpected token-table schema")
    require(all(token_file.schema_arrow.field(prefix + role).type == pa.float32()
                for prefix in ("z_", "p_") for role in ROLES), "Exported model values are not float32")
    identities, readout_hashes, record_hashes = set(), {}, {}
    split_inputs, split_tokens = Counter(), Counter()
    row_total, hidden_bytes, role_assignments = 0, 0, Counter()
    for group_index, row in enumerate(jsonlines(stage / "inputs.jsonl")):
        name = row["input_id"]
        require(isinstance(name, str) and name and Path(name).name == name and name not in identities,
                "Repeated or unsafe source input identifier")
        identities.add(name)
        table = token_file.read_row_group(group_index)
        expected_positions = np.asarray(row["target_positions"], dtype=np.int64)
        count = len(expected_positions)
        require(table.num_rows == count == row["target_token_count"], "Target row-group size differs: " + name)
        for field in ("input_id", "trace_id", "source_split", "operation", "destination_role"):
            require(table.column(field).to_pylist() == [row[field]] * count,
                    "Exported categorical input identity differs: " + name + "/" + field)
        source_roles, offsets, originals = {}, {}, {}
        for role, positions in row["role_target_positions"].items():
            require(role in ROLES, "Unknown role in source mask")
            old_positions = row["original_role_target_positions"][role]
            require(len(positions) == len(old_positions) > 0, "Invalid original body mask")
            for offset, (position, old_position) in enumerate(zip(positions, old_positions)):
                require(position not in source_roles, "Overlapping original role bodies")
                source_roles[position], offsets[position], originals[position] = role, offset, old_position
        require(set(source_roles) == set(expected_positions.tolist()), "Role masks do not cover every exported token")
        require(table.column("source_role").to_pylist() == [source_roles[position] for position in row["target_positions"]],
                "Baseline union or variant source-role assignment differs: " + name)
        actual_offset = numeric_column(table, "body_token_offset", np.int32)
        require(np.array_equal(actual_offset, [offsets[position] for position in row["target_positions"]]),
                "Body-relative token offsets differ: " + name)
        expected_originals = np.asarray([originals[position] for position in row["target_positions"]], dtype=np.int64)
        require(np.array_equal(expected_originals, row["target_original_positions"]), "Input original-position mask differs")
        record_relative = "records/" + name + ".json"
        record_path = relative_file(root, record_relative)
        record = read(record_path)
        require(record["status"] == "complete" and record["input_id"] == name and
                record["trace_id"] == row["trace_id"] and record["body_tokens"] == count,
                "Saved readout record identity differs")
        require(run["records"][name] == {key: record[key] for key in ("readout_sha256", "hiddenstates_sha256")},
                "Readout record differs from the committed run")
        record_hashes[record_relative] = sha(record_path)
        require(exported["record_hashes"][record_relative] == record_hashes[record_relative],
                "Readout record changed after the export")
        npz_path = relative_file(root, record["readout"])
        readout_hashes[record["readout"]] = sha(npz_path)
        require(readout_hashes[record["readout"]] == record["readout_sha256"] == exported["readout_hashes"][record["readout"]],
                "Current original NPZ differs from the committed source")
        with np.load(npz_path, allow_pickle=False) as saved:
            require(tuple(saved["roles"].tolist()) == ROLES, "Original readout role order differs")
            for column, array_name, expected in (("token_position", "positions", expected_positions),
                                                  ("original_token_position", "original_positions", expected_originals),
                                                  ("token_id", "token_ids", np.asarray(row["input_ids"], dtype=np.int64)[expected_positions])):
                original_values = saved[array_name]
                require(original_values.dtype == np.int64 and original_values.shape == (count,) and
                        np.array_equal(original_values, expected), "Original saved coordinate/token identity differs")
                require(np.array_equal(numeric_column(table, column, np.int64), original_values),
                        "Parquet/NPZ coordinate or token IDs differ: " + name + "/" + column)
            for prefix, array_name in (("z_", "logits"), ("p_", "probabilities")):
                original_values = saved[array_name]
                require(original_values.dtype == np.float32 and original_values.shape == (count, 6) and
                        np.isfinite(original_values).all(), "Invalid original saved float32 values")
                for role_index, role in enumerate(ROLES):
                    actual = numeric_column(table, prefix + role, np.float32)
                    require(np.array_equal(actual.view(np.uint32), original_values[:, role_index].view(np.uint32)),
                            "Float32 bit pattern differs: " + name + "/" + prefix + role)
        index_table = hidden_file.read_row_group(group_index)
        require(index_table.num_rows == 1, "Hidden-state index row group is not one input")
        index_row = index_table.to_pylist()[0]
        for field in ("input_id", "trace_id", "source_split", "operation", "source_role", "destination_role", "target_roles"):
            require(index_row[field] == row[field], "Hidden-state index source identity differs: " + field)
        require(index_row["path"] == record["hiddenstates"] and index_row["sha256"] == record["hiddenstates_sha256"] and
                index_row["bytes"] == record["hiddenstate_bytes"] and index_row["body_tokens"] == count and
                index_row["layers"] == 36 and index_row["hidden_size"] == 2880 and index_row["native_dtype"] == "BF16" and
                index_row["tensor_count"] == 43 and index_row["embedding"] is True and index_row["final_norm"] is True,
                "Hidden-state index geometry/hash differs from its committed record")
        require(relative_file(root, index_row["path"]).stat().st_size == index_row["bytes"], "Indexed hidden-state file size changed")
        row_total += count
        hidden_bytes += index_row["bytes"]
        split_inputs[row["source_split"]] += 1
        split_tokens[row["source_split"]] += count
        role_assignments.update(table.column("source_role").to_pylist())
    require(len(identities) == expected_inputs and row_total == expected_tokens and
            identities == set(run["records"]) and hidden_bytes == run["hiddenstate_bytes"],
            "Independent export audit did not cover the entire committed stage")
    for relative, expected in bindings.items():
        require(sha(root / relative) == expected, "Bound source or Parquet changed during independent comparison: " + relative)
    result = dict(format_version="agentdojo-trace-export-independent-validation-v1", status="passed", stage=stage_name,
                  inputs=expected_inputs, token_rows=row_total, original_npz_files_compared=len(readout_hashes),
                  source_hashes=bindings, readout_hashes=readout_hashes, record_hashes=record_hashes,
                  run_manifest_sha256=bindings[f"{stage_name}/run-manifest.json"],
                  source_traces_sha256=bindings["sources/traces.jsonl"],
                  export_manifest_sha256=bindings[f"{stage_name}/export-manifest.json"],
                  validator_sha256=sha(Path(__file__)), exporter_not_imported=True,
                  all_saved_token_rows_compared=True, coordinates_and_token_ids_bit_exact=True,
                  saved_logits_bit_exact=True, saved_probabilities_bit_exact=True,
                  floats_compared_as_uint32_bits_including_signed_zero=True, differing_float32_bit_patterns=0,
                  baseline_and_variant_role_masks_exact=True, body_token_offsets_exact=True,
                  hiddenstate_index_rows_checked=expected_inputs, hiddenstate_index_bytes=hidden_bytes,
                  hiddenstate_index_matches_committed_records=True, hiddenstate_payloads_read=False,
                  hiddenstate_payload_hashes_recomputed=False,
                  input_counts_by_split=dict(split_inputs), token_rows_by_split=dict(split_tokens),
                  token_rows_by_source_role=dict(role_assignments), seconds=time.monotonic() - started,
                  software=dict(numpy=np.__version__, pyarrow=pa.__version__))
    destination = stage / "export-independent-validation.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps({key: result[key] for key in ("status", "stage", "inputs", "token_rows", "original_npz_files_compared", "seconds")}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(EXPECTED), required=True)
    args = parser.parse_args()
    validate(args.output, args.stage)


if __name__ == "__main__":
    main()
