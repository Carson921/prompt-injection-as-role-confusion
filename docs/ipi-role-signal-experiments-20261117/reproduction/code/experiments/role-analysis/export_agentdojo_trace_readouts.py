"""Export completed trace interventions without recomputing any model value.

The token table copies saved float32 logits/probabilities. The hidden-state
index references the original BF16 safetensors files; it does not duplicate
their large residual arrays. The exporter checks their headers, byte lengths,
identity metadata and small alignment vectors, and carries forward SHA256
digests recorded by the completed runner. It does not rehash the residual
payloads, which may total approximately 150 GiB.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
EXPECTED = {"pilot": (43, 6032), "all": (2451, 719832)}
IDENTITIES = ("input_id", "trace_id", "source_split", "operation", "source_role", "destination_role")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            value.update(chunk)
    return value.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: malformed JSON") from error


def contained_file(root, relative):
    require(isinstance(relative, str) and relative, "Missing artifact path")
    path = root / relative
    require(not Path(relative).is_absolute() and path.resolve().is_relative_to(root.resolve()),
            "Artifact path escapes the experiment root: " + relative)
    require(path.is_file(), "Missing artifact: " + str(path))
    return path


def token_schema():
    fields = [pa.field(name, pa.string(), nullable=name == "destination_role") for name in IDENTITIES]
    fields += [pa.field("token_position", pa.int64(), nullable=False),
               pa.field("original_token_position", pa.int64(), nullable=False),
               pa.field("token_id", pa.int64(), nullable=False),
               pa.field("body_token_offset", pa.int32(), nullable=False)]
    fields += [pa.field(prefix + role, pa.float32(), nullable=False)
               for prefix in ("z_", "p_") for role in ROLES]
    return pa.schema(fields, metadata={
        b"format": b"agentdojo-trace-body-token-readouts-v1",
        b"roles": json.dumps(ROLES).encode(),
        b"coordinates": b"zero-based absolute token positions; offset restarts for each role body",
        b"value_policy": b"saved float32 logits and probabilities copied directly; never recomputed",
        b"destination_role": b"copied from inputs.jsonl; null outside wrapper swaps",
        b"baseline": b"union of six first-role bodies; source_role assigned separately to each token",
    })


def hidden_schema():
    fields = [pa.field(name, pa.string(), nullable=name in ("source_role", "destination_role"))
              for name in IDENTITIES]
    fields += [pa.field("path", pa.string(), nullable=False),
               pa.field("body_tokens", pa.int32(), nullable=False),
               pa.field("layers", pa.int16(), nullable=False),
               pa.field("hidden_size", pa.int16(), nullable=False),
               pa.field("native_dtype", pa.string(), nullable=False),
               pa.field("bytes", pa.int64(), nullable=False),
               pa.field("sha256", pa.string(), nullable=False),
               pa.field("target_roles", pa.list_(pa.field("element", pa.string())), nullable=False),
               pa.field("tensor_count", pa.int16(), nullable=False),
               pa.field("embedding", pa.bool_(), nullable=False),
               pa.field("final_norm", pa.bool_(), nullable=False)]
    return pa.schema(fields, metadata={
        b"format": b"agentdojo-trace-hiddenstates-index-v1",
        b"path_base": b"experiment output root, not the stage subdirectory",
        b"scope": b"target body positions only; baseline stores the union of first six roles",
        b"row_alignment": b"safetensors row i matches positions[i] and the same input_id/token_position in token-readouts.parquet",
        b"sha256_policy": b"runner-recorded payload hashes cross-checked against completed run manifest; exporter checks header and size without rehashing large residual payloads",
    })


def role_masks(row):
    count = row["target_token_count"]
    positions = np.asarray(row["target_positions"], dtype=np.int64)
    original = np.asarray(row["target_original_positions"], dtype=np.int64)
    require(positions.shape == original.shape == (count,) and count > 0,
            "Invalid manifested target coordinates: " + row["input_id"])
    require(np.all(np.diff(positions) > 0) and positions[0] >= 0 and
            positions[-1] < len(row["input_ids"]), "Target positions are not unique ordered input coordinates")
    ownership = {}
    for role, mask in row["role_target_positions"].items():
        require(role in ROLES and len(mask) > 0, "Invalid target role mask")
        originals = row["original_role_target_positions"][role]
        require(len(mask) == len(originals), "Original role mask length differs")
        require(mask == list(range(mask[0], mask[0] + len(mask))), "Target body mask is not contiguous")
        for offset, (position, orig) in enumerate(zip(mask, originals)):
            require(position not in ownership, "Role body masks overlap")
            ownership[position] = (role, offset, orig)
    require(set(ownership) == set(positions.tolist()), "Role masks differ from selected target positions")
    require([ownership[int(position)][2] for position in positions] == original.tolist(),
            "Original positions differ from role masks")
    expected_roles = set(ROLES) if row["operation"] == "baseline" else {row["source_role"]}
    require(set(row["target_roles"]) == set(row["role_target_positions"]) == expected_roles,
            "First-role target inventory differs")
    if row["operation"] != "baseline":
        require(row["source_role"] in ROLES, "Unknown variant source role")
    require(row["operation"] in ("baseline", "strip", "swap", "move"), "Unknown intervention")
    require(row["destination_role"] in ROLES if row["operation"] == "swap" else
            row["destination_role"] is None, "Unexpected destination role")
    return positions, original, ownership


def hidden_header(path, row, record, positions, originals, token_ids):
    """Inspect the header and three small I64 vectors, not the BF16 payloads."""
    size = path.stat().st_size
    require(size == record["hiddenstate_bytes"], "Hidden-state file length differs")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Incomplete safetensors header")
        header_bytes = struct.unpack("<Q", prefix)[0]
        require(0 < header_bytes < min(size - 8, 1 << 20), "Invalid safetensors header length")
        header = json.loads(stream.read(header_bytes))
        metadata = header.get("__metadata__", {})
        require(metadata.get("input_id") == row["input_id"] and metadata.get("trace_id") == row["trace_id"],
                "Hidden-state metadata identity differs")
        require(metadata.get("native_dtype") == "BF16" and
                json.loads(metadata.get("signature", "null")) == record["signature"],
                "Hidden-state identity signature differs")
        expected = {f"residual_layer_{layer:02}" for layer in range(36)}
        expected.update(("embedding", "final_norm", "pre_o_proj_heads_bf16", "probe_features_fp16",
                         "positions", "original_positions", "token_ids"))
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        require(set(tensors) == expected and len(tensors) == record["hiddenstate_tensors"] == 43,
                "Hidden-state tensor inventory differs")
        n = len(positions)
        for name in ["embedding", "final_norm"] + [f"residual_layer_{layer:02}" for layer in range(36)]:
            require(tensors[name]["dtype"] == "BF16" and tensors[name]["shape"] == [n, 2880],
                    "Native residual dtype/shape differs: " + name)
        require(tensors["pre_o_proj_heads_bf16"]["dtype"] == "BF16" and
                tensors["pre_o_proj_heads_bf16"]["shape"] == [n, 16, 64], "Native head features differ")
        require(tensors["probe_features_fp16"]["dtype"] == "F16" and
                tensors["probe_features_fp16"]["shape"] == [n, 1024], "Probe feature header differs")
        offsets = []
        widths = {"BF16": 2, "F16": 2, "I64": 8}
        for name, tensor in tensors.items():
            a, b = tensor["data_offsets"]
            require(0 <= a <= b and tensor["dtype"] in widths and
                    b - a == int(np.prod(tensor["shape"])) * widths[tensor["dtype"]],
                    "Malformed safetensors byte span: " + name)
            offsets.append((a, b))
        offsets.sort()
        require(offsets[0][0] == 0 and all(left[1] == right[0] for left, right in zip(offsets, offsets[1:]))
                and 8 + header_bytes + offsets[-1][1] == size, "Safetensors data spans do not cover the file")
        for name, expected_values in (("positions", positions), ("original_positions", originals),
                                      ("token_ids", token_ids)):
            tensor = tensors[name]
            require(tensor["dtype"] == "I64" and tensor["shape"] == [n], "Invalid alignment vector header")
            a, b = tensor["data_offsets"]
            stream.seek(8 + header_bytes + a)
            observed = np.frombuffer(stream.read(b - a), dtype="<i8")
            require(np.array_equal(observed, expected_values), "Hidden-state/readout row alignment differs: " + name)
    return len(tensors)


def input_tables(root, row, run_record, token_format, hidden_format):
    name = row["input_id"]
    require(isinstance(name, str) and Path(name).name == name and name,
            "Unsafe input identifier")
    record_path = root / "records" / (name + ".json")
    record = read(record_path)
    require(record.get("status") == "complete" and record["input_id"] == name and
            record["trace_id"] == row["trace_id"], "Input record is incomplete or belongs to another trace")
    require(record["readout_sha256"] == run_record["readout_sha256"] and
            record["hiddenstates_sha256"] == run_record["hiddenstates_sha256"],
            "Input record hashes differ from completed run")
    positions, originals, ownership = role_masks(row)
    n = len(positions)
    require(record["body_tokens"] == n and record["hiddenstate_layers"] == 36 and
            record["input_tokens"] == len(row["input_ids"]), "Input record counts differ")
    for key, value in (("input_ids_sha256", row["input_ids"]), ("positions_sha256", row["target_positions"]),
                       ("original_positions_sha256", row["target_original_positions"])):
        require(record["signature"][key] == object_sha(value), "Input identity signature differs: " + key)
    require(record["signature"]["trace_sha256"] == row["trace_sha256"], "Trace identity differs")
    source = contained_file(root, record["readout"])
    require(digest(source) == record["readout_sha256"], "Readout checksum mismatch: " + name)
    with np.load(source, allow_pickle=False) as arrays:
        require(set(arrays.files) == {"positions", "original_positions", "token_ids", "head_features", "logits",
                                      "probabilities", "roles"}, "Unexpected readout array inventory")
        require(tuple(arrays["roles"].tolist()) == ROLES, "Readout role order differs")
        for field, expected_values in (("positions", positions), ("original_positions", originals)):
            values = arrays[field]
            require(values.dtype == np.int64 and values.shape == (n,) and np.array_equal(values, expected_values),
                    "Saved coordinate vector differs: " + field)
        token_ids = arrays["token_ids"]
        require(token_ids.dtype == np.int64 and token_ids.shape == (n,) and
                np.array_equal(token_ids, np.asarray(row["input_ids"], dtype=np.int64)[positions]),
                "Saved body token IDs differ")
        require([row["token_source_positions"][int(position)] for position in positions] == originals.tolist(),
                "Body original-token ownership differs")
        features = arrays["head_features"]
        require(features.dtype == np.float16 and features.shape == (n, 1024) and np.isfinite(features).all(),
                "Saved probe feature dtype/shape or values differ")
        logits, probabilities = arrays["logits"], arrays["probabilities"]
        for label, values in (("logits", logits), ("probabilities", probabilities)):
            require(values.dtype == np.float32 and values.shape == (n, 6) and np.isfinite(values).all(),
                    "Invalid saved " + label)
        require(((probabilities >= 0) & (probabilities <= 1)).all() and
                np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0), "Invalid saved probability simplex")
        columns = {field: [row[field]] * n for field in IDENTITIES if field != "source_role"}
        columns.update(source_role=[ownership[int(position)][0] for position in positions],
                       token_position=positions, original_token_position=originals, token_id=token_ids,
                       body_token_offset=np.asarray([ownership[int(position)][1] for position in positions], dtype=np.int32))
        for prefix, values in (("z_", logits), ("p_", probabilities)):
            for column, role in enumerate(ROLES):
                columns[prefix + role] = values[:, column]
        token_table = pa.Table.from_pydict(columns, schema=token_format)
        hidden_path = contained_file(root, record["hiddenstates"])
        tensor_count = hidden_header(hidden_path, row, record, positions, originals, token_ids)
    require(digest(source) == record["readout_sha256"], "Readout changed during export")
    hidden_row = {field: row[field] for field in IDENTITIES}
    hidden_row.update(path=record["hiddenstates"], body_tokens=n, layers=36, hidden_size=2880,
                      native_dtype="BF16", bytes=record["hiddenstate_bytes"], sha256=record["hiddenstates_sha256"],
                      target_roles=row["target_roles"], tensor_count=tensor_count, embedding=True, final_norm=True)
    return token_table, pa.Table.from_pylist([hidden_row], schema=hidden_format), record, digest(record_path)


def temporary(directory, suffix):
    handle = tempfile.NamedTemporaryFile(prefix=".trace-export-", suffix=suffix, dir=directory, delete=False)
    handle.close()
    return Path(handle.name)


def export(root, stage_name):
    started = time.monotonic()
    root = root.resolve()
    stage = root / stage_name
    run_path, input_path, protocol_path = (stage / name for name in ("run-manifest.json", "inputs.jsonl", "protocol.json"))
    require(run_path.is_file(), "Run must finish before export: missing " + str(run_path))
    run, protocol = read(run_path), read(protocol_path)
    expected_inputs, expected_tokens = EXPECTED[stage_name]
    require(run.get("status") == "complete" and run.get("stage") == stage_name and run.get("inputs") == expected_inputs,
            "Run is not a complete expected stage")
    hashes = {str(path.relative_to(root)): digest(path) for path in (run_path, input_path, protocol_path)}
    require(run["inputs_sha256"] == protocol["inputs_sha256"] == hashes[str(input_path.relative_to(root))] and
            run["protocol_sha256"] == hashes[str(protocol_path.relative_to(root))], "Completed stage input/protocol identity differs")
    require(run["probe_sha256"] == protocol["probe_sha256"], "Completed stage probe identity differs")
    require(run["summary_sha256"] == digest(stage / "input-summary.csv"), "Completed summary changed")
    hashes[str((stage / "input-summary.csv").relative_to(root))] = run["summary_sha256"]
    for relative in ("sources/traces.jsonl", "sources/manifest.json"):
        path = root / relative
        if path.exists():
            hashes[relative] = digest(path)
    require(hashes.get("sources/traces.jsonl") == protocol["source_records_sha256"], "Frozen source traces changed")
    rows = list(lines(input_path))
    ids = [row["input_id"] for row in rows]
    require(len(rows) == len(set(ids)) == expected_inputs and set(ids) == set(run["records"]),
            "Completed stage readout inventory differs from manifested inputs")
    require(sum(row["target_token_count"] for row in rows) == expected_tokens, "Unexpected stage body-token total")
    token_format, hidden_format = token_schema(), hidden_schema()
    token_tmp, hidden_tmp, manifest_tmp = [temporary(stage, suffix) for suffix in (".parquet", ".parquet", ".json")]
    parquet_path, hidden_index_path = stage / "token-readouts.parquet", stage / "hiddenstates-index.parquet"
    count, hidden_bytes = 0, 0
    readout_hashes, record_hashes = {}, {}
    split_inputs, split_tokens, operations = Counter(), Counter(), Counter()
    try:
        with pq.ParquetWriter(token_tmp, token_format, compression="zstd", use_dictionary=list(IDENTITIES)) as token_writer, \
                pq.ParquetWriter(hidden_tmp, hidden_format, compression="zstd", use_dictionary=list(IDENTITIES)) as hidden_writer:
            for row in rows:
                record_identity = read(root / "records" / (row["input_id"] + ".json"))["signature"]
                for key in ("probe_sha256", "parser_sha256", "runner_sha256", "runtime_code_sha256"):
                    require(record_identity[key] == protocol[key], "Input record differs from the frozen stage protocol: " + key)
                table, index_table, record, record_sha = input_tables(root, row, run["records"][row["input_id"]],
                                                                    token_format, hidden_format)
                token_writer.write_table(table, row_group_size=table.num_rows)
                hidden_writer.write_table(index_table, row_group_size=1)
                count += table.num_rows
                hidden_bytes += record["hiddenstate_bytes"]
                readout_hashes[record["readout"]] = record["readout_sha256"]
                record_hashes["records/" + row["input_id"] + ".json"] = record_sha
                split_inputs[row["source_split"]] += 1
                split_tokens[row["source_split"]] += table.num_rows
                operations[row["operation"]] += 1
        require(count == expected_tokens and hidden_bytes == run["hiddenstate_bytes"], "Export totals differ from completed run")
        token_meta, hidden_meta = pq.read_metadata(token_tmp), pq.read_metadata(hidden_tmp)
        require(token_meta.num_rows == expected_tokens and hidden_meta.num_rows == expected_inputs and
                token_meta.num_row_groups == hidden_meta.num_row_groups == expected_inputs, "Parquet counts differ")
        require(pq.read_schema(token_tmp).equals(token_format, check_metadata=True) and
                pq.read_schema(hidden_tmp).equals(hidden_format, check_metadata=True), "Parquet schema changed")
        for relative, expected_sha in hashes.items():
            require(digest(root / relative) == expected_sha, "Source artifact changed during export: " + relative)
        for relative, expected_sha in record_hashes.items():
            require(digest(root / relative) == expected_sha, "Input record changed during export: " + relative)
        manifest = dict(format_version="agentdojo-trace-readouts-export-v1", status="complete", stage=stage_name,
                        inputs=expected_inputs, token_rows=count, hiddenstate_files=expected_inputs,
                        input_counts_by_split=dict(split_inputs), token_rows_by_split=dict(split_tokens),
                        input_counts_by_operation=dict(operations), readout_files=len(readout_hashes),
                        hiddenstate_bytes=hidden_bytes, hiddenstate_gib=hidden_bytes / (1 << 30),
                        source_hashes=hashes, readout_hashes=readout_hashes, record_hashes=record_hashes,
                        exporter_sha256=digest(Path(__file__)),
                        token_readouts=dict(path=str(parquet_path.relative_to(root)), sha256=digest(token_tmp),
                                            row_groups=token_meta.num_row_groups, columns={f.name: str(f.type) for f in token_format}),
                        hiddenstates_index=dict(path=str(hidden_index_path.relative_to(root)), sha256=digest(hidden_tmp),
                                                row_groups=hidden_meta.num_row_groups, columns={f.name: str(f.type) for f in hidden_format}),
                        roles=list(ROLES), coordinates="Zero-based absolute token_position and original_token_position; body_token_offset restarts per role body.",
                        baseline_policy="Baseline readout contains the union of six first-role bodies. Token source_role and body_token_offset are assigned using role_target_positions in the stage inputs.jsonl; the index retains source_role=null and all six target_roles.",
                        destination_role_policy="Original inputs.jsonl value: destination role for swaps, null otherwise.",
                        value_policy="Saved float32 logits and probabilities copied directly, without model/probe loading or recomputation.",
                        hiddenstate_verification="File lengths, 43 tensor headers, native BF16 36 layers plus embedding/final norm, record signatures and small alignment vectors checked. Payload SHA256 copied from runner records and cross-checked against completed run manifest; large residual payloads not rehashed by exporter.",
                        hidden_payload_sha256_recomputed=False, hiddenstate_payloads_duplicated=False,
                        elapsed_seconds=time.monotonic() - started,
                        software=dict(numpy=np.__version__, pyarrow=pa.__version__))
        manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(token_tmp, parquet_path)
        os.replace(hidden_tmp, hidden_index_path)
        os.replace(manifest_tmp, stage / "export-manifest.json")
        write_example(stage, root)
        print(json.dumps({key: manifest[key] for key in ("status", "stage", "inputs", "token_rows", "hiddenstate_files", "hiddenstate_gib")}))
    finally:
        for path in (token_tmp, hidden_tmp, manifest_tmp):
            path.unlink(missing_ok=True)


def write_example(stage, root):
    text = '''Hidden-state loading and row alignment

Paths in hiddenstates-index.parquet are relative to the experiment output root.
Each safetensors file stores only scored body positions. Baseline files store
the union of the six first-role bodies; use the token table's source_role or
role_target_positions in this stage's inputs.jsonl to choose a baseline role.
The same token_position identifies the table row and the saved positions vector.
body_token_offset starts at zero separately for each role body.

Example (CPU; requires pandas, pyarrow, torch, safetensors):

from pathlib import Path
import pandas as pd
from safetensors import safe_open

ROOT = Path(ROOT_PLACEHOLDER)
STAGE = STAGE_PLACEHOLDER
index = pd.read_parquet(ROOT / STAGE / "hiddenstates-index.parquet")
entry = index.iloc[0]
tokens = pd.read_parquet(ROOT / STAGE / "token-readouts.parquet",
                         filters=[("input_id", "==", entry.input_id)])
with safe_open(str(ROOT / entry.path), framework="pt", device="cpu") as saved:
    positions = saved.get_tensor("positions")
    original_positions = saved.get_tensor("original_positions")
    token_ids = saved.get_tensor("token_ids")
    # Load only the first two selected body rows from layer 18; dtype is BF16.
    hidden = saved.get_slice("residual_layer_18")[:2]
    assert tokens.token_position.tolist() == positions.tolist()
    assert tokens.original_token_position.tolist() == original_positions.tolist()
    assert tokens.token_id.tolist() == token_ids.tolist()

Available tensors: embedding, residual_layer_00 ... residual_layer_35,
final_norm (all BF16, N x 2880), pre_o_proj_heads_bf16 (N x 16 x 64),
probe_features_fp16 (N x 1024), positions, original_positions, token_ids.
N equals body_tokens in the index. Logits and probabilities are available in
the original NPZ and token-readouts.parquet and were copied without recomputation.

The index carries hashes recorded during inference. Export checks file sizes,
headers and alignment vectors; it does not read and rehash all residual payloads.
'''
    text = text.replace("ROOT_PLACEHOLDER", repr(str(root))).replace("STAGE_PLACEHOLDER", repr(stage.name))
    (stage / "hiddenstates-loading-example.txt").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(EXPECTED), required=True)
    args = parser.parse_args()
    export(args.output, args.stage)


if __name__ == "__main__":
    main()
