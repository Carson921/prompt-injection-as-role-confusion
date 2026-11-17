#!/usr/bin/env python3
"""Independently audit every clean IPI Parquet row against its source NPZ.

This checker imports no exporter, runner, analyzer, probe, Torch or safetensors
reader. It streams one input row group at a time. All 12 saved float32 logit and
probability columns are compared as raw uint32 bits. Body coordinates, IDs and
offsets are checked against inputs.jsonl. It also reads each indexed safetensors
header, three small alignment vectors and its two selected-head feature tensors:
FP16 probe features must match the NPZ bit for bit and equal the BF16 head values
converted to FP16. The 38 large residual payloads are not read or rehashed here.

Usage:
    python audit_ipi_trace_readout_export.py --output ROOT --stage pilot
    python audit_ipi_trace_readout_export.py --output ROOT --stage all
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
IDENTITIES = ("input_id", "trace_id", "source_split", "operation", "source_role", "destination_role")
EXPECTED_TRACES = {"pilot": 1, "all": 41}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            value.update(block)
    return value.hexdigest()


def object_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read(path):
    require(Path(path).is_file(), "Missing required artifact: " + str(path))
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def relative_file(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "Invalid artifact reference")
    path = root / relative
    require(path.resolve().is_relative_to(root) and path.is_file(), "Missing or escaping artifact reference: " + relative)
    return path


def expected_schemas():
    token = [pa.field(name, pa.string(), nullable=name == "destination_role") for name in IDENTITIES]
    token += [pa.field(name, kind, nullable=False) for name, kind in (
        ("token_position", pa.int64()), ("original_token_position", pa.int64()),
        ("token_id", pa.int64()), ("body_token_offset", pa.int32()))]
    token += [pa.field(prefix + role, pa.float32(), nullable=False) for prefix in ("z_", "p_") for role in ROLES]
    hidden = [pa.field(name, pa.string(), nullable=name in ("source_role", "destination_role")) for name in IDENTITIES]
    hidden += [pa.field(name, kind, nullable=False) for name, kind in (
        ("path", pa.string()), ("body_tokens", pa.int32()), ("layers", pa.int16()),
        ("hidden_size", pa.int16()), ("native_dtype", pa.string()), ("bytes", pa.int64()),
        ("sha256", pa.string()), ("target_roles", pa.list_(pa.field("element", pa.string()))),
        ("tensor_count", pa.int16()), ("embedding", pa.bool_()), ("final_norm", pa.bool_()))]
    return pa.schema(token), pa.schema(hidden)


def column(table, name, dtype):
    value = table.column(name)
    require(value.null_count == 0, "Unexpected null exported scalar: " + name)
    array = value.combine_chunks().to_numpy(zero_copy_only=False)
    require(array.dtype == np.dtype(dtype), "Exported dtype differs: " + name)
    return array


def input_inventory(rows, stage_name):
    traces, ids = {}, set()
    for row in rows:
        name = row["input_id"]
        require(isinstance(name, str) and name and Path(name).name == name and name not in ids, "Unsafe or duplicate input ID")
        ids.add(name)
        require(row["source_split"] == "external", "IPI source split is not external")
        require(row["operation"] in ("baseline", "strip", "swap", "move"), "Unknown operation")
        traces.setdefault(row["trace_id"], []).append(row)
    require(len(traces) == EXPECTED_TRACES[stage_name], "Wrong external task cohort size")
    for trace, items in traces.items():
        baseline = [row for row in items if row["operation"] == "baseline"]
        require(len(baseline) == 1, "A task must have exactly one baseline input: " + trace)
        present = set(baseline[0]["role_target_positions"])
        require(present and present.issubset(ROLES) and set(baseline[0]["target_roles"]) == present,
                "Invalid baseline actual-source inventory")
        require(baseline[0]["source_role"] is None and baseline[0]["destination_role"] is None, "Baseline falsely labels one source body")
        require(len(items) == 1 + 7 * len(present), "Incorrect 1+7m design count")
        variants = [row for row in items if row["operation"] != "baseline"]
        require({row["source_role"] for row in variants} == present, "Variant source-body inventory differs")
        for role in present:
            selected = [row for row in variants if row["source_role"] == role]
            require(Counter(row["operation"] for row in selected) == {"strip": 1, "move": 1, "swap": 5},
                    "Incomplete interventions for actual source body")
            swaps = [row for row in selected if row["operation"] == "swap"]
            require({row["destination_role"] for row in swaps} == set(ROLES) - {role}, "Five alternative wrapper destinations not covered")
            for row in selected:
                require(row["target_roles"] == [role] and set(row["role_target_positions"]) == {role}, "Variant targets extra or absent source bodies")
                require(row["destination_role"] is None if row["operation"] != "swap" else True, "Non-swap has destination role")
    return ids, traces


def hidden_alignment_and_heads(path, row, record, positions, original, token_ids, npz_heads):
    """Independent safetensors header parser; no large residual payload reads."""
    size = path.stat().st_size
    require(size == record["hiddenstate_bytes"], "Indexed hidden-state size differs from record")
    n = len(positions)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Incomplete safetensors header")
        header_bytes = struct.unpack("<Q", prefix)[0]
        require(0 < header_bytes < min(size - 8, 1 << 20), "Invalid safetensors header size")
        header = json.loads(stream.read(header_bytes))
        metadata = header.get("__metadata__", {})
        require(metadata.get("input_id") == row["input_id"] and metadata.get("trace_id") == row["trace_id"] and
                metadata.get("native_dtype") == "BF16" and json.loads(metadata.get("signature", "null")) == record["signature"],
                "Indexed hidden-state identity metadata differs")
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        residuals = ("embedding", "final_norm", *(f"residual_layer_{i:02d}" for i in range(36)))
        expected = set(residuals) | {"pre_o_proj_heads_bf16", "probe_features_fp16", "positions", "original_positions", "token_ids"}
        require(set(tensors) == expected and len(tensors) == record["hiddenstate_tensors"] == 43, "Hidden-state tensor inventory differs")
        for name in residuals:
            require(tensors[name]["dtype"] == "BF16" and tensors[name]["shape"] == [n, 2880], "Residual tensor dtype/shape differs: " + name)
        require(tensors["pre_o_proj_heads_bf16"]["dtype"] == "BF16" and tensors["pre_o_proj_heads_bf16"]["shape"] == [n, 16, 64], "Native head-feature geometry differs")
        require(tensors["probe_features_fp16"]["dtype"] == "F16" and tensors["probe_features_fp16"]["shape"] == [n, 1024], "FP16 head-feature geometry differs")
        spans, widths = [], {"BF16": 2, "F16": 2, "I64": 8}
        for name, tensor in tensors.items():
            start, end = tensor["data_offsets"]
            require(tensor["dtype"] in widths and 0 <= start <= end and
                    end - start == int(np.prod(tensor["shape"])) * widths[tensor["dtype"]], "Invalid hidden tensor byte span: " + name)
            spans.append((start, end))
        spans.sort()
        require(spans[0][0] == 0 and all(a[1] == b[0] for a, b in zip(spans, spans[1:])) and
                8 + header_bytes + spans[-1][1] == size, "Safetensors byte spans do not cover the file")

        def payload(name, dtype):
            tensor = tensors[name]
            start, end = tensor["data_offsets"]
            stream.seek(8 + header_bytes + start)
            data = stream.read(end - start)
            require(len(data) == end - start, "Truncated alignment/head-feature payload")
            return np.frombuffer(data, dtype=dtype).reshape(tensor["shape"])

        for name, expected_values in (("positions", positions), ("original_positions", original), ("token_ids", token_ids)):
            require(tensors[name]["dtype"] == "I64" and tensors[name]["shape"] == [n], "Invalid coordinate tensor")
            require(np.array_equal(payload(name, "<i8"), expected_values), "Hidden-state coordinate/body-ID vector differs: " + name)
        fp16_bits = payload("probe_features_fp16", "<u2")
        require(np.array_equal(fp16_bits, npz_heads.view(np.uint16)), "Hidden-state FP16 head features differ from original NPZ bits")
        bf16_bits = payload("pre_o_proj_heads_bf16", "<u2")
        float32_values = (bf16_bits.astype(np.uint32) << 16).view(np.float32).reshape(n, 1024)
        require(np.isfinite(float32_values).all(), "Nonfinite native BF16 head features")
        with np.errstate(over="ignore", invalid="ignore"):
            expected_fp16 = float32_values.astype(np.float16)
        require(np.array_equal(expected_fp16.view(np.uint16), fp16_bits), "Native BF16 to FP16 probe feature conversion differs")


def validate(root, stage_name):
    root = Path(root).resolve()
    stage = root / stage_name
    started = time.monotonic()
    run = read(stage / "run-manifest.json")
    require(run.get("status") == "complete" and run.get("stage") == stage_name, "Inference must be committed complete before export audit")
    rows = list(lines(stage / "inputs.jsonl"))
    identities, traces = input_inventory(rows, stage_name)
    expected_inputs, expected_tokens = len(rows), sum(row["target_token_count"] for row in rows)
    require(run["inputs"] == expected_inputs and set(run["records"]) == identities, "Completed run input inventory differs")
    exported, protocol = read(stage / "export-manifest.json"), read(stage / "protocol.json")
    require(exported.get("format_version") == "ipi-trace-readouts-export-v1" and exported.get("status") == "complete" and
            exported.get("stage") == stage_name and exported["inputs"] == exported["hiddenstate_files"] == expected_inputs and
            exported["token_rows"] == expected_tokens, "IPI export is incomplete or has incorrect dynamic counts")
    paths = [stage / name for name in ("run-manifest.json", "protocol.json", "inputs.jsonl", "input-summary.csv", "export-manifest.json")]
    source_relative = protocol.get("source_records_path", "sources/traces.jsonl")
    source_path = relative_file(root, source_relative)
    paths += [source_path, stage / "token-readouts.parquet", stage / "hiddenstates-index.parquet"]
    bindings = {str(path.relative_to(root)): sha(path) for path in paths}
    require(bindings[f"{stage_name}/inputs.jsonl"] == run["inputs_sha256"] == protocol["inputs_sha256"] and
            bindings[f"{stage_name}/protocol.json"] == run["protocol_sha256"] and
            bindings[f"{stage_name}/input-summary.csv"] == run["summary_sha256"] and
            bindings[source_relative] == protocol["source_records_sha256"] and
            run["probe_sha256"] == protocol["probe_sha256"], "Source/protocol/run identity differs")
    if "selected_traces" in protocol:
        require(set(protocol["selected_traces"]) == set(traces), "Task cohort differs from frozen protocol")
    for field in ("token_readouts", "hiddenstates_index"):
        item = exported[field]
        require(bindings[item["path"]] == item["sha256"], "Committed Parquet checksum differs")
    for relative, expected in exported["source_hashes"].items():
        require(sha(relative_file(root, relative)) == expected, "Export source binding changed: " + relative)
    token_file, hidden_file = pq.ParquetFile(stage / "token-readouts.parquet"), pq.ParquetFile(stage / "hiddenstates-index.parquet")
    require(token_file.metadata.num_rows == expected_tokens and hidden_file.metadata.num_rows == expected_inputs and
            token_file.num_row_groups == hidden_file.num_row_groups == expected_inputs, "Parquet row counts/groups differ")
    token_schema, hidden_schema = expected_schemas()
    require(token_file.schema_arrow.equals(token_schema, check_metadata=False), "Full token schema/order/nullability differs")
    require(hidden_file.schema_arrow.equals(hidden_schema, check_metadata=False), "Full hidden-state index schema/order/nullability differs")
    token_meta, hidden_meta = token_file.schema_arrow.metadata or {}, hidden_file.schema_arrow.metadata or {}
    require(token_meta.get(b"format") == b"ipi-trace-body-token-readouts-v1" and
            hidden_meta.get(b"format") == b"ipi-trace-hiddenstates-index-v1", "Parquet dataset format metadata differs")
    require(json.loads(token_meta.get(b"roles", b"null")) == list(ROLES) and
            token_meta.get(b"dataset") == hidden_meta.get(b"dataset") == b"IPI Arena clean continuation", "Parquet role/dataset metadata differs")
    readout_hashes, record_hashes = {}, {}
    split_inputs, split_tokens, operations, role_tokens = Counter(), Counter(), Counter(), Counter()
    hidden_bytes, row_total = 0, 0
    for group_index, row in enumerate(rows):
        name = row["input_id"]
        table = token_file.read_row_group(group_index)
        positions = np.asarray(row["target_positions"], dtype=np.int64)
        count = len(positions)
        require(count > 0 and count == row["target_token_count"] == table.num_rows and
                np.all(np.diff(positions) > 0) and positions[0] >= 0 and positions[-1] < len(row["input_ids"]), "Invalid target row-group size/coordinates: " + name)
        for field in ("input_id", "trace_id", "source_split", "operation", "destination_role"):
            require(table.column(field).to_pylist() == [row[field]] * count, "Token-table categorical identity differs: " + name + "/" + field)
        ownership = {}
        for role, mask in row["role_target_positions"].items():
            old_mask = row["original_role_target_positions"][role]
            require(role in ROLES and len(mask) == len(old_mask) > 0 and
                    mask == list(range(mask[0], mask[0] + len(mask))) and
                    old_mask == list(range(old_mask[0], old_mask[0] + len(old_mask))), "Invalid source body mask")
            for offset, (position, old_position) in enumerate(zip(mask, old_mask)):
                require(position not in ownership and old_position >= 0, "Overlapping or unowned body token")
                ownership[position] = (role, offset, old_position)
        require(set(ownership) == set(positions.tolist()), "Body masks do not cover exported tokens")
        require(table.column("source_role").to_pylist() == [ownership[int(p)][0] for p in positions], "Baseline/variant token source roles differ")
        require(np.array_equal(column(table, "body_token_offset", np.int32), [ownership[int(p)][1] for p in positions]), "Body token offsets differ")
        original = np.asarray([ownership[int(p)][2] for p in positions], dtype=np.int64)
        require(np.array_equal(original, row["target_original_positions"]) and
                [row["token_source_positions"][int(p)] for p in positions] == original.tolist(), "Original body-token ownership differs")
        record_relative = "records/" + name + ".json"
        record_path = relative_file(root, record_relative)
        record = read(record_path)
        require(record["status"] == "complete" and record["input_id"] == name and record["trace_id"] == row["trace_id"] and
                record["body_tokens"] == count and record["input_tokens"] == len(row["input_ids"]), "Input record identity/count differs")
        require(run["records"][name] == {key: record[key] for key in ("readout_sha256", "hiddenstates_sha256")}, "Record hashes differ from committed run")
        for field in ("probe_sha256", "parser_sha256", "runner_sha256", "runtime_code_sha256"):
            require(record["signature"][field] == protocol[field], "Input signature differs from protocol: " + field)
        for field, expected in (("input_ids_sha256", row["input_ids"]), ("positions_sha256", row["target_positions"]),
                                ("original_positions_sha256", row["target_original_positions"])):
            require(record["signature"][field] == object_sha(expected), "Input token/coordinate signature differs")
        require(record["signature"]["trace_sha256"] == row["trace_sha256"], "Trace signature differs")
        record_hashes[record_relative] = sha(record_path)
        require(exported["record_hashes"][record_relative] == record_hashes[record_relative], "Input record changed after export")
        npz_path = relative_file(root, record["readout"])
        readout_hashes[record["readout"]] = sha(npz_path)
        require(readout_hashes[record["readout"]] == record["readout_sha256"] == exported["readout_hashes"][record["readout"]], "Original NPZ checksum differs")
        with np.load(npz_path, allow_pickle=False) as saved:
            require(set(saved.files) == {"positions", "original_positions", "token_ids", "head_features", "logits", "probabilities", "roles"} and
                    tuple(saved["roles"].tolist()) == ROLES, "Original NPZ field inventory/class order differs")
            token_ids = np.asarray(row["input_ids"], dtype=np.int64)[positions]
            for field, key, expected in (("token_position", "positions", positions), ("original_token_position", "original_positions", original), ("token_id", "token_ids", token_ids)):
                values = saved[key]
                require(values.dtype == np.int64 and values.shape == (count,) and np.array_equal(values, expected) and
                        np.array_equal(column(table, field, np.int64), values), "Parquet/NPZ/native coordinate or body ID differs: " + field)
            heads = saved["head_features"]
            require(heads.dtype == np.float16 and heads.shape == (count, 1024) and np.isfinite(heads).all(), "Invalid saved FP16 head features")
            for prefix, key in (("z_", "logits"), ("p_", "probabilities")):
                values = saved[key]
                require(values.dtype == np.float32 and values.shape == (count, 6) and np.isfinite(values).all(), "Invalid original six-class float32 values")
                for ri, role in enumerate(ROLES):
                    require(np.array_equal(column(table, prefix + role, np.float32).view(np.uint32), values[:, ri].view(np.uint32)),
                            "Saved float32 bit pattern differs: " + name + "/" + prefix + role)
            require((saved["probabilities"] >= 0).all() and (saved["probabilities"] <= 1).all() and
                    np.allclose(saved["probabilities"].sum(1), 1, atol=1e-6, rtol=0), "Invalid source probability simplex")
            index_table = hidden_file.read_row_group(group_index)
            require(index_table.num_rows == 1, "Hidden index row group does not represent one input")
            entry = index_table.to_pylist()[0]
            for field in (*IDENTITIES, "target_roles"):
                require(entry[field] == row[field], "Hidden index categorical identity differs: " + field)
            require(entry["path"] == record["hiddenstates"] and entry["sha256"] == record["hiddenstates_sha256"] and
                    entry["bytes"] == record["hiddenstate_bytes"] and entry["body_tokens"] == count and entry["layers"] == 36 and
                    entry["hidden_size"] == 2880 and entry["native_dtype"] == "BF16" and entry["tensor_count"] == 43 and
                    entry["embedding"] is True and entry["final_norm"] is True, "Hidden index geometry/recorded hash differs")
            hidden_path = relative_file(root, entry["path"])
            hidden_alignment_and_heads(hidden_path, row, record, positions, original, token_ids, heads)
        require(sha(npz_path) == record["readout_sha256"], "NPZ changed during independent comparison")
        row_total += count
        hidden_bytes += entry["bytes"]
        split_inputs[row["source_split"]] += 1
        split_tokens[row["source_split"]] += count
        operations[row["operation"]] += 1
        role_tokens.update(table.column("source_role").to_pylist())
    require(row_total == expected_tokens and hidden_bytes == run["hiddenstate_bytes"] == exported["hiddenstate_bytes"], "Independent export audit totals differ")
    require(exported["input_counts_by_split"] == dict(split_inputs) and exported["token_rows_by_split"] == dict(split_tokens) and
            exported["input_counts_by_operation"] == dict(operations), "Export manifest dynamic subtotals differ")
    require(set(exported["readout_hashes"]) == set(readout_hashes) and set(exported["record_hashes"]) == set(record_hashes), "Export source inventory differs")
    for relative, expected in bindings.items():
        require(sha(root / relative) == expected, "Bound artifact changed during audit: " + relative)
    result = dict(
        format_version="ipi-trace-export-independent-validation-v1", status="passed", stage=stage_name,
        traces=len(traces), inputs=expected_inputs, token_rows=row_total, original_npz_files_compared=len(readout_hashes),
        source_hashes=bindings, readout_hashes=readout_hashes, record_hashes=record_hashes,
        run_manifest_sha256=bindings[f"{stage_name}/run-manifest.json"], source_traces_sha256=bindings[source_relative],
        source_records_path=source_relative,
        export_manifest_sha256=bindings[f"{stage_name}/export-manifest.json"], validator_sha256=sha(Path(__file__)),
        exporter_not_imported=True, runner_and_analyzer_not_imported=True, no_torch_or_gpu=True,
        dynamic_present_role_design_verified=True, full_parquet_schemas_verified=True,
        all_saved_token_rows_compared=True, coordinates_and_token_ids_bit_exact=True,
        saved_logits_bit_exact=True, saved_probabilities_bit_exact=True, floats_compared_as_uint32_bits_including_signed_zero=True,
        differing_float32_bit_patterns=0, baseline_and_variant_role_masks_exact=True, body_token_offsets_exact=True,
        npz_fp16_heads_and_indexed_hiddenstate_fp16_heads_bit_exact=True, bf16_to_fp16_head_conversion_bit_exact=True,
        head_feature_scalar_values_compared=row_total * 1024, head_features_are_parquet_columns=False,
        head_feature_policy="Parquet has six logits/probabilities; FP16 head values are verified between each NPZ and its indexed source safetensors file",
        hiddenstate_index_rows_checked=expected_inputs, hiddenstate_index_bytes=hidden_bytes,
        hiddenstate_index_matches_committed_records=True, hiddenstate_headers_and_alignment_vectors_checked=True,
        hiddenstate_selected_head_payloads_read=True, hiddenstate_residual_payloads_read=False,
        hiddenstate_full_payload_hashes_recomputed=False,
        input_counts_by_split=dict(split_inputs), token_rows_by_split=dict(split_tokens), token_rows_by_source_role=dict(role_tokens),
        seconds=time.monotonic() - started, software=dict(numpy=np.__version__, pyarrow=pa.__version__),
    )
    destination = stage / "export-independent-validation.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps({key: result[key] for key in ("status", "stage", "traces", "inputs", "token_rows", "seconds")}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(EXPECTED_TRACES), required=True)
    args = parser.parse_args()
    validate(args.output, args.stage)


if __name__ == "__main__":
    main()
