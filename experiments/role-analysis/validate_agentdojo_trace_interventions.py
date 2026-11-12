"""Independent CPU audit of native AgentDojo trace interventions and caches.

No runner, intervention builder, classifier, Torch, or safetensors library is
imported. An independent token-frame oracle checks all 43 interventions, and
the exported MLP is evaluated with NumPy float64 and scalar math.erf. Every
hiddenstate payload is streamed once for both SHA256 and full finite checks;
its header independently proves embedding / 36 residual layers / final norm
coverage, native BF16 storage, shapes, and selected-token ownership.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
PROBE_SHA = "fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33"
START, MESSAGE, END, RETURN, CALL = 200006, 200008, 200007, 200002, 200012
CLOSERS = (END, RETURN, CALL)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def obj_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


class HeaderDecoder:
    """Decode header bytes independently from the frozen tokenizer JSON vocab."""
    def __init__(self, path):
        source = read(path)
        self.tokens = {value: key for key, value in source["model"]["vocab"].items()}
        self.tokens.update({item["id"]: item["content"] for item in source["added_tokens"]})
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        originals, rendered, extra = visible.copy(), visible.copy(), 0
        for byte in range(256):
            if byte not in visible:
                originals.append(byte)
                rendered.append(256 + extra)
                extra += 1
        self.bytes = {chr(character): byte for character, byte in zip(rendered, originals)}

    def __call__(self, ids):
        text = "".join(self.tokens[value] for value in ids)
        return bytes(self.bytes[character] for character in text).decode("utf-8")


def original_frames(ids, decoder, allow_terminal_body):
    """Independent sequential frame reader, with a single explicitly opted-in EOF."""
    frames, cursor = [], 0
    while cursor < len(ids):
        start = cursor
        require(ids[cursor] == START, f"Original trace has unparsed token {cursor}")
        cursor += 1
        while cursor < len(ids) and ids[cursor] != MESSAGE:
            require(ids[cursor] not in (*CLOSERS, START), "Malformed original native header")
            cursor += 1
        head = decoder(ids[start + 1:cursor])
        # Recipient declarations take precedence over analysis/final channels.
        structural = head.startswith("assistant") and bool(re.search(r"(?:^|\s)to=", head))
        if head.startswith("functions."):
            role = "tool"
        elif structural:
            role = None
        elif head.startswith("assistant"):
            role = "cot" if "<|channel|>analysis" in head else "assistant"
        elif head in ("system", "developer", "user"):
            role = head
        else:
            raise AssertionError(f"Unknown original native header: {head!r}")
        if cursor == len(ids):
            require(head == "assistant", "Only an open assistant generation header may trail a trace")
            frames.append(dict(start=start, body_start=cursor, body_end=cursor, end=cursor,
                               role=role, structural=False, complete=False, terminated=False))
            break
        body_start = cursor + 1
        cursor = body_start
        while cursor < len(ids) and ids[cursor] not in CLOSERS:
            require(ids[cursor] not in (START, MESSAGE), "Ambiguous structural delimiter inside original body")
            cursor += 1
        terminated = cursor < len(ids)
        require(terminated or (allow_terminal_body and role == "assistant" and not structural),
                "Unexplained original terminal body truncation")
        if terminated:
            require((ids[cursor] == CALL) == structural, "Original tool-call ending differs")
            require(ids[cursor] != RETURN or role == "assistant", "Original return marker belongs to wrong role")
        frames.append(dict(start=start, body_start=body_start, body_end=cursor,
                           end=cursor + int(terminated), role=role, structural=structural,
                           complete=True, terminated=terminated))
        cursor += int(terminated)
    first = {}
    for index, frame in enumerate(frames):
        frame["block_index"] = index
        if frame["complete"] and frame["role"] in ROLES and frame["role"] not in first:
            require(frame["body_end"] > frame["body_start"], "Empty first normal role body")
            first[frame["role"]] = frame
    require(set(first) == set(ROLES), "Original trace lacks one of six ordinary roles")
    return frames, first


def audit_inputs(output, stage_name, protocol, decoder):
    sources_path = output / "sources/traces.jsonl"
    require(sha(sources_path) == protocol["source_records_sha256"], "Original source-record hash changed")
    sources = jsonl(sources_path)
    by_trace = {source["trace_id"]: source for source in sources}
    require(len(sources) == len(by_trace) == 57, "Expected exactly 57 distinct source traces")
    require(Counter(source["source_split"] for source in sources) == {"fit": 40, "dev": 8, "test": 9},
            "Original source task split inventory differs")
    rows_path = output / stage_name / "inputs.jsonl"
    require(sha(rows_path) == protocol["inputs_sha256"], "Prepared input hash changed")
    rows = jsonl(rows_path)
    by_input = {row["input_id"]: row for row in rows}
    require(len(by_input) == len(rows), "Repeated input ID")
    grouped = defaultdict(list)
    for row in rows:
        require(row["input_id"] == Path(row["input_id"]).name, "Unsafe input archive filename")
        grouped[row["trace_id"]].append(row)
    require(set(grouped) == set(protocol["selected_traces"]), "Selected/prepared trace inventory differs")
    require(len(rows) == protocol["inputs"] == 43 * len(grouped), "Wrong number of intervention inputs")
    if stage_name == "pilot":
        require(len(grouped) == 1 and set(grouped) == {protocol["pilot_trace_id"]}, "Pilot must have exactly its selected trace")
        require(by_trace[protocol["pilot_trace_id"]]["source_split"] == "fit", "Pilot is outside training FIT")
    else:
        require(set(grouped) == set(by_trace), "All-stage does not cover all 57 source traces")
    terminal = []
    for trace_id, variants in grouped.items():
        source = by_trace[trace_id]
        ids = source["input_ids"]
        frames, first = original_frames(ids, decoder, source.get("allow_terminal_body", False))
        observed_terminal = [frame for frame in frames if frame["complete"] and not frame["terminated"]]
        require(bool(observed_terminal) == source.get("allow_terminal_body", False), "Terminal-censoring flag differs from original tokens")
        if observed_terminal:
            provenance = source["provenance"]["terminal_censoring"]
            require(provenance["finish_reason"] == "length" and provenance["artificial_ending_tokens_added"] == 0,
                    "Terminal observed frame lacks no-artificial-token length-limit proof")
            terminal.append(trace_id)
        require(Counter(row["operation"] for row in variants) == {"baseline": 1, "strip": 6, "swap": 30, "move": 6},
                f"Incomplete intervention grid for {trace_id}")
        expected_cells = {("baseline", None, None)} | {(operation, role, None) for operation in ("strip", "move") for role in ROLES}
        expected_cells |= {("swap", source_role, destination) for source_role in ROLES for destination in ROLES if source_role != destination}
        actual_cells = [(row["operation"], row["source_role"], row["destination_role"]) for row in variants]
        require(len(set(actual_cells)) == 43 and set(actual_cells) == expected_cells, "Repeated or omitted directed interventions")
        positions_by_role = {role: list(range(first[role]["body_start"], first[role]["body_end"])) for role in ROLES}
        for row in variants:
            key, operation, role = row["input_id"], row["operation"], row["source_role"]
            require(row["source_split"] == source["source_split"] and row["trace_sha256"] == obj_sha(ids), "Source provenance differs: " + key)
            if operation == "baseline":
                expected_origins = list(range(len(ids)))
                expected_positions = sorted(value for positions in positions_by_role.values() for value in positions)
                original_positions = expected_positions
                role_positions = positions_by_role
                require(row["block_index"] is None, "Baseline names a manipulated block")
            else:
                frame = first[role]
                a, b, c, d = (frame[name] for name in ("start", "body_start", "body_end", "end"))
                original_positions = positions_by_role[role]
                require(row["block_index"] == frame["block_index"], "Intervention did not select first normal role frame")
                if operation == "strip":
                    expected_origins = list(range(a)) + list(range(b, c)) + list(range(d, len(ids)))
                    new_start = a
                elif operation == "swap":
                    wrapper = first[row["destination_role"]]
                    h0, h1, t0, t1 = (wrapper[name] for name in ("start", "body_start", "body_end", "end"))
                    expected_origins = list(range(a)) + list(range(h0, h1)) + list(range(b, c)) + list(range(t0, t1)) + list(range(d, len(ids)))
                    new_start = a + h1 - h0
                    require(row["applied_header_ids"] == ids[h0:h1] and row["applied_ending_ids"] == ids[t0:t1],
                            "Swap did not use exact observed destination wrapper")
                else:
                    remaining = list(range(a)) + list(range(d, len(ids)))
                    # Enumerate every native boundary in the remaining complete
                    # history, independently of the builder's two-endpoint rule.
                    boundaries, cursor = [0], 0
                    for other in frames:
                        if other is not frame and other["complete"]:
                            cursor += other["end"] - other["start"]
                            boundaries.append(cursor)
                    candidates = [(boundary + b - a, boundary) for boundary in boundaries]
                    new_start, insertion = min(candidates, key=lambda pair: (-abs(pair[0] - b), pair[0]))
                    expected_origins = remaining[:insertion] + list(range(a, d)) + remaining[insertion:]
                    require(sorted(expected_origins) == list(range(len(ids))), "Move did not preserve every source token exactly once")
                    require(row["move_distance_tokens"] == abs(new_start - b), "Recorded movement distance differs")
                expected_positions = list(range(new_start, new_start + c - b))
                role_positions = {role: expected_positions}
                require(row["original_body_start"] == b and row["new_body_start"] == new_start,
                        "Body-start metadata differs: " + key)
            require(row["token_source_positions"] == expected_origins, "Full-input token ownership differs: " + key)
            require(row["input_ids"] == [ids[position] for position in expected_origins], "Intervention changed or invented tokens: " + key)
            require(row["target_positions"] == expected_positions and row["target_original_positions"] == original_positions,
                    "Selected target body ownership differs: " + key)
            require(row["role_target_positions"] == role_positions, "Role capture masks differ: " + key)
            require(row["original_role_target_positions"] == {r: positions_by_role[r] for r in role_positions},
                    "Original role body masks differ: " + key)
            require(len(row["input_ids"]) <= 131072 and row["input_tokens"] == len(row["input_ids"]), "Invalid/truncated input length")
    return rows, dict(status="passed", traces=len(grouped), inputs=len(rows),
                      trace_split_counts=dict(Counter(by_trace[trace_id]["source_split"] for trace_id in grouped)),
                      terminal_censored_traces=terminal, all43_token_native_interventions_independently_rebuilt=True)


def softmax(values):
    centered = values.astype(np.float64) - values.max(axis=1, keepdims=True)
    exponential = np.exp(centered)
    return exponential / exponential.sum(axis=1, keepdims=True)


def independent_logits(features, parameters, metadata):
    hidden = (features.astype(np.float64) - parameters["mean"]) / parameters["scale"]
    hidden = hidden @ parameters["weight::0.weight"].astype(np.float64).T + parameters["weight::0.bias"]
    if metadata["hidden"]:
        error_function = np.frompyfunc(math.erf, 1, 1)
        hidden = hidden * 0.5 * (1.0 + error_function(hidden / math.sqrt(2.0)).astype(np.float64))
        hidden = hidden @ parameters["weight::3.weight"].astype(np.float64).T + parameters["weight::3.bias"]
    return hidden


def safetensors_header(path, count):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Missing safetensors length prefix")
        length = struct.unpack("<Q", prefix)[0]
        require(0 < length < 10 << 20, "Invalid safetensors header length")
        raw = stream.read(length)
        require(len(raw) == length, "Truncated safetensors header")
    header = json.loads(raw)
    expected = {"embedding": ("BF16", [count, 2880]), "final_norm": ("BF16", [count, 2880]),
                **{f"residual_layer_{layer:02d}": ("BF16", [count, 2880]) for layer in range(36)},
                "pre_o_proj_heads_bf16": ("BF16", [count, 16, 64]),
                "probe_features_fp16": ("F16", [count, 1024]),
                **{key: ("I64", [count]) for key in ("positions", "original_positions", "token_ids")}}
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    require(set(tensors) == set(expected), "Saved hiddenstate keys omit/duplicate a residual layer or body metadata")
    sizes = {"BF16": 2, "F16": 2, "I64": 8}
    for name, tensor in tensors.items():
        dtype, shape = expected[name]
        require(tensor["dtype"] == dtype and tensor["shape"] == shape, "Hiddenstate geometry/dtype differs: " + name)
        a, b = tensor["data_offsets"]
        require(b - a == math.prod(shape) * sizes[dtype], "Hiddenstate payload length differs: " + name)
    ordered = sorted(tensors.items(), key=lambda item: item[1]["data_offsets"][0])
    cursor = 0
    for _, tensor in ordered:
        a, b = tensor["data_offsets"]
        require(a == cursor and b >= a, "Overlapping/noncontiguous hiddenstate payload")
        cursor = b
    require(8 + length + cursor == path.stat().st_size, "Hiddenstate file is truncated or has unowned trailing bytes")
    return header, prefix + raw, 8 + length, ordered


def audit_hidden(path, row, record, signature, readout):
    count = len(row["target_positions"])
    header, prefix, data_start, ordered = safetensors_header(path, count)
    metadata = header["__metadata__"]
    require(metadata["input_id"] == row["input_id"] and metadata["trace_id"] == row["trace_id"]
            and metadata["native_dtype"] == "BF16" and json.loads(metadata["signature"]) == signature,
            "Hiddenstate metadata/signature differs")
    require(record["hiddenstate_tensors"] == 43 and record["hiddenstate_layers"] == 36,
            "Commit record does not prove all 36 residual layers")
    digest = hashlib.sha256(prefix)
    # Hash and check every floating value during the same sequential file read.
    with path.open("rb") as stream:
        stream.seek(data_start)
        for name, tensor in ordered:
            remaining = tensor["data_offsets"][1] - tensor["data_offsets"][0]
            while remaining:
                block = stream.read(min(8 << 20, remaining))
                require(bool(block), "Hiddenstate payload ended before its committed extent")
                digest.update(block)
                remaining -= len(block)
                if tensor["dtype"] == "BF16":
                    bits = np.frombuffer(block, dtype="<u2")
                    require(bool(((bits & 0x7f80) != 0x7f80).all()), "Nonfinite native BF16 hiddenstate: " + name)
                elif tensor["dtype"] == "F16":
                    require(bool(np.isfinite(np.frombuffer(block, dtype="<f2")).all()), "Nonfinite FP16 probe features")
    require(digest.hexdigest() == record["hiddenstates_sha256"], "Hiddenstate full-payload SHA256 differs")

    def load_small(name):
        tensor = header[name]
        a, b = tensor["data_offsets"]
        with path.open("rb") as stream:
            stream.seek(data_start + a)
            raw = stream.read(b - a)
        dtype = {"BF16": "<u2", "F16": "<f2", "I64": "<i8"}[tensor["dtype"]]
        return np.frombuffer(raw, dtype=dtype).reshape(tensor["shape"])

    for name, readout_name in (("positions", "positions"), ("original_positions", "original_positions"), ("token_ids", "token_ids")):
        require(np.array_equal(load_small(name), readout[readout_name]), "Hiddenstate selected-token ownership differs: " + name)
    features = load_small("probe_features_fp16")
    require(np.array_equal(features, readout["head_features"]), "Hiddenstate/readout FP16 features differ")
    bits = load_small("pre_o_proj_heads_bf16")
    native = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    require(np.array_equal(native.astype(np.float16).reshape(count, 1024), features),
            "Native BF16 head capture does not reproduce checkpoint FP16 features")
    embedding = load_small("embedding")
    positions = np.asarray(row["target_positions"])
    embedding_hashes = {}
    for role, role_positions in row["role_target_positions"].items():
        offsets = np.searchsorted(positions, role_positions)
        require(np.array_equal(positions[offsets], role_positions), "Embedding role rows are not captured")
        embedding_hashes[role] = hashlib.sha256(np.ascontiguousarray(embedding[offsets]).tobytes()).hexdigest()
    return path.stat().st_size, embedding_hashes


def validate(args):
    output, stage_name = args.output.resolve(), args.stage
    stage = output / stage_name
    protocol = read(stage / "protocol.json")
    probe_path = args.probe.resolve() if args.probe else Path(protocol["probe"])
    require(sha(probe_path) == protocol["probe_sha256"] == protocol["expected_probe_sha256"] == PROBE_SHA,
            "Requested frozen checkpoint differs")
    with np.load(probe_path, allow_pickle=False) as archive:
        parameters = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(parameters["metadata"]))
    require(tuple(metadata["roles"]) == ROLES and metadata["feature_position"] == "pre-o_proj"
            and metadata["feature_dim"] == 1024 and metadata["activation"] == "gelu_exact", "Checkpoint interface differs")
    require(parameters["mean"].shape == parameters["scale"].shape == (1024,) and (parameters["scale"] > 0).all(),
            "Invalid checkpoint standardizer")
    require(sha(HERE / "agentdojo_trace_interventions_data.py") == protocol["parser_sha256"], "Prepared parser source changed")
    require(sha(HERE / "run_agentdojo_trace_interventions.py") == protocol["runner_sha256"], "Prepared runner source changed")
    for name, expected in protocol["runtime_code_sha256"].items():
        require(sha(ROOT / name) == expected, "Bound native runtime source changed: " + name)
    model = args.model.resolve() if args.model else Path(protocol["model_path"])
    for name, expected in metadata["model_binding"]["assets_sha256"].items():
        require(sha(model / name) == expected, "Bound tokenizer/backbone asset changed: " + name)
    rows, input_result = audit_inputs(output, stage_name, protocol, HeaderDecoder(model / "tokenizer.json"))
    result = dict(status="passed", stage=stage_name, checkpoint_sha256=PROBE_SHA,
                  protocol_sha256=sha(stage / "protocol.json"), inputs_sha256=sha(stage / "inputs.jsonl"),
                  validator_sha256=sha(__file__), inputs=input_result, validation_scope="inputs only")
    if args.inputs_only:
        atomic_json(stage / "independent-input-validation.json", result)
        print(json.dumps(result), flush=True)
        return result
    manifest = read(stage / "run-manifest.json")
    require(manifest["status"] == "complete" and manifest["inputs"] == len(rows) and manifest["probe_sha256"] == PROBE_SHA,
            "Run is not committed complete")
    require(manifest["protocol_sha256"] == sha(stage / "protocol.json") and manifest["inputs_sha256"] == sha(stage / "inputs.jsonl")
            and manifest["summary_sha256"] == sha(stage / "input-summary.csv"), "Run manifest file hashes differ")
    require(set(manifest["records"]) == {row["input_id"] for row in rows}, "Committed readout inventory differs")
    with (stage / "input-summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    indexed_summary = {(row["input_id"], row["source_role"]): row for row in summaries}
    expected_summary = {(row["input_id"], role) for row in rows for role in row["role_target_positions"]}
    require(len(summaries) == len(indexed_summary) == 48 * len(protocol["selected_traces"])
            and set(indexed_summary) == expected_summary, "Summary omits or repeats role bodies")
    if stage_name == "pilot":
        passive = read(stage / "passive-hook-validation.json")
        require(passive["status"] == "passed" and passive["selected_head_values_bit_exact"] is True
                and passive["hiddenstate_tensors"] == 38, "Passive residual hooks were not verified on the pilot")
    totals = dict(hiddenstate_bytes=0, saved_body_token_rows=0, independent_probe_token_rows=0,
                  hiddenstate_tensors=0, residual_layer_tensors=0)
    errors = dict(softmax_max_abs=0.0, probe_logits_max_abs=0.0, probe_probability_max_abs=0.0, summary_max_abs=0.0)
    embedding_references = {}
    started = time.monotonic()
    for number, row in enumerate(rows, 1):
        key = row["input_id"]
        record = read(output / "records" / (key + ".json"))
        signature = dict(input_ids_sha256=obj_sha(row["input_ids"]), positions_sha256=obj_sha(row["target_positions"]),
                         original_positions_sha256=obj_sha(row["target_original_positions"]), trace_sha256=row["trace_sha256"],
                         runtime_code_sha256=protocol["runtime_code_sha256"], probe_sha256=PROBE_SHA,
                         parser_sha256=protocol["parser_sha256"], runner_sha256=protocol["runner_sha256"])
        require(record["status"] == "complete" and record["signature"] == signature
                and record["input_id"] == key and record["trace_id"] == row["trace_id"], "Readout commit signature differs")
        require(manifest["records"][key] == {name: record[name] for name in ("readout_sha256", "hiddenstates_sha256")},
                "Run/record artifact hash bindings differ")
        readout_path = output / "readouts" / (key + ".npz")
        hidden_path = output / "hiddenstates" / (key + ".safetensors")
        require(record["readout"] == str(readout_path.relative_to(output))
                and record["hiddenstates"] == str(hidden_path.relative_to(output)), "Committed cache paths differ")
        require(sha(readout_path) == record["readout_sha256"] and hidden_path.stat().st_size == record["hiddenstate_bytes"],
                "Readout hash or hiddenstate length changed")
        with np.load(readout_path, allow_pickle=False) as archive:
            readout = {name: archive[name] for name in archive.files}
        positions = np.asarray(row["target_positions"], dtype=np.int64)
        count = len(positions)
        require(readout["positions"].dtype == np.int64 and np.array_equal(readout["positions"], positions)
                and np.array_equal(readout["original_positions"], row["target_original_positions"]), "Readout body coordinates differ")
        require(np.array_equal(readout["token_ids"], np.asarray(row["input_ids"])[positions])
                and tuple(readout["roles"].tolist()) == ROLES, "Readout token IDs or role ordering differs")
        features, logits, probabilities = (readout[name] for name in ("head_features", "logits", "probabilities"))
        require(features.dtype == np.float16 and features.shape == (count, 1024) and np.isfinite(features).all(), "Invalid FP16 checkpoint inputs")
        require(logits.dtype == probabilities.dtype == np.float32 and logits.shape == probabilities.shape == (count, 6)
                and np.isfinite(logits).all() and np.isfinite(probabilities).all(), "Invalid saved probe output")
        require((probabilities >= 0).all() and (probabilities <= 1).all(), "Probe probability outside [0,1]")
        difference = float(np.max(np.abs(softmax(logits) - probabilities)))
        errors["softmax_max_abs"] = max(errors["softmax_max_abs"], difference)
        require(difference <= 2e-6, "Saved probabilities are not the saved logits' softmax")
        selected = (np.arange(count) if not args.probe_row_cap or count <= args.probe_row_cap
                    else np.unique(np.linspace(0, count - 1, args.probe_row_cap, dtype=int)))
        oracle_logits = independent_logits(features[selected], parameters, metadata)
        logit_error = float(np.max(np.abs(oracle_logits - logits[selected])))
        probability_error = float(np.max(np.abs(softmax(oracle_logits) - probabilities[selected])))
        errors["probe_logits_max_abs"] = max(errors["probe_logits_max_abs"], logit_error)
        errors["probe_probability_max_abs"] = max(errors["probe_probability_max_abs"], probability_error)
        require(np.allclose(oracle_logits, logits[selected], rtol=2e-5, atol=5e-4)
                and probability_error <= 5e-5, "Independent math.erf checkpoint evaluation differs")
        totals["independent_probe_token_rows"] += len(selected)
        hidden_bytes, embedding_hashes = audit_hidden(hidden_path, row, record, signature, readout)
        totals["hiddenstate_bytes"] += hidden_bytes
        for role, embedding_sha in embedding_hashes.items():
            reference_key = row["trace_id"], role
            if reference_key in embedding_references:
                require(embedding_references[reference_key] == embedding_sha,
                        "Identical body token IDs have different saved embedding rows: " + key + "/" + role)
            else:
                embedding_references[reference_key] = embedding_sha
        totals["saved_body_token_rows"] += count
        totals["hiddenstate_tensors"] += 43
        totals["residual_layer_tensors"] += 36
        for role, role_positions in row["role_target_positions"].items():
            locations = np.searchsorted(positions, role_positions)
            require(np.array_equal(positions[locations], role_positions), "Summary mask references an uncaptured body row")
            summary = indexed_summary[key, role]
            destination = (row["destination_role"] if row["operation"] == "swap"
                           else role if row["operation"] in ("baseline", "move") else "")
            require(all(summary[name] == expected for name, expected in
                        (("trace_id", row["trace_id"]), ("source_split", row["source_split"]),
                         ("operation", row["operation"]), ("destination_role", destination))), "Summary condition metadata differs")
            p, z = probabilities[locations].astype(np.float64), logits[locations].astype(np.float64)
            expected = {**{"p_" + label: p[:, index].mean() for index, label in enumerate(ROLES)},
                        **{"z_" + label: z[:, index].mean() for index, label in enumerate(ROLES)},
                        "mean_original_role_probability": p[:, ROLES.index(role)].mean(),
                        "argmax_original_role_fraction": (p.argmax(1) == ROLES.index(role)).mean(),
                        "body_tokens": len(locations), "total_input_tokens": len(row["input_ids"]),
                        "original_body_start": row["original_role_target_positions"][role][0],
                        "new_body_start": role_positions[0]}
            for name, value in expected.items():
                error = abs(float(summary[name]) - float(value))
                errors["summary_max_abs"] = max(errors["summary_max_abs"], error)
                require(error <= 1e-10 + 1e-12 * abs(value), "Summary token mean/metadata differs: " + key + "/" + name)
        if number % 43 == 0 or number == len(rows):
            print(json.dumps(dict(stage="independent_validation", selection=stage_name, inputs=number,
                                  total=len(rows), hiddenstate_bytes=totals["hiddenstate_bytes"], seconds=time.monotonic() - started)), flush=True)
    require(totals["hiddenstate_bytes"] == manifest["hiddenstate_bytes"], "Manifest hiddenstate byte total differs")
    result.update(validation_scope="all input/readout/summary/hiddenstate files; all hidden floating values streamed for finiteness",
                  totals=totals, maximum_errors=errors, seconds=time.monotonic() - started,
                  independent_probe_row_cap=args.probe_row_cap,
                  all_probe_rows_independently_recomputed=(args.probe_row_cap == 0),
                  hiddenstate_full_payload_sha256_all_verified=True, all36_residual_layers_plus_embedding_final_norm=True,
                  native_bf16_to_checkpoint_fp16_bit_exact=True,
                  same_body_embedding_rows_bit_exact_across_all_interventions=True, summary_rows=len(summaries))
    atomic_json(stage / "independent-validation.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "all"), required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--probe-row-cap", type=int, default=0,
                        help="0 independently recomputes every token; positive values evenly sample up to N rows/file")
    args = parser.parse_args()
    require(args.probe_row_cap >= 0, "Probe row cap must be nonnegative")
    validate(args)


if __name__ == "__main__":
    main()
