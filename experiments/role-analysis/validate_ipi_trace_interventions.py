"""Independent CPU audit of native IPI trace interventions and caches.

No runner, intervention builder, classifier, Torch, or safetensors library is
imported. An independent token-frame oracle rebuilds all 1+7m interventions
using actual source frames and the frozen observed-pilot wrapper catalog, and
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
        require(terminated or (allow_terminal_body and (role in ("cot", "assistant") or structural)),
                "Unexplained original terminal body truncation")
        if terminated:
            require((ids[cursor] == CALL) == structural, "Original tool-call ending differs")
            require(ids[cursor] != RETURN or role == "assistant", "Original return marker belongs to wrong role")
        frames.append(dict(start=start, body_start=body_start, body_end=cursor,
                           end=cursor + int(terminated), role=role, structural=structural,
                           complete=True, terminated=terminated))
        cursor += int(terminated)
    first, first_normal = {}, {}
    for index, frame in enumerate(frames):
        frame["block_index"] = index
        if frame["complete"] and not frame["structural"] and frame["role"] in ROLES:
            first_normal.setdefault(frame["role"], frame)
    first = {role: frame for role, frame in first_normal.items() if frame["body_end"] > frame["body_start"]}
    require(bool(first), "Original trace has no nonempty normal role targets")
    return frames, first, first_normal


def rooted_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_catalog(output, protocol):
    """The frozen catalog is evidence to verify against raw native pilot IDs."""
    if "wrapper_catalog" in protocol:
        catalog = protocol["wrapper_catalog"]
        if "wrapper_catalog_path" in protocol:
            source = output / protocol["wrapper_catalog_path"]
            require(read(source) == catalog, "Inline/file wrapper catalogs differ")
    else:
        source = output / protocol["wrapper_catalog_path"]
        catalog = read(source)
    require(isinstance(catalog, dict) and set(catalog) == set(ROLES),
            "Frozen pilot catalog must provide all six destination wrappers")
    if "wrapper_catalog_sha256" in protocol:
        require(sha(source) == protocol["wrapper_catalog_sha256"], "Frozen wrapper catalog file hash differs")
    return catalog


def audit_opaque_call_references(record, fixture, collection, case_names):
    """Independently follow generated opaque IDs through live/simulator history."""
    history = list(fixture["messages"])

    def administrative_ids(messages):
        for message in messages:
            for call in message.get("tool_calls", []):
                yield call["id"]
            if "tool_call_id" in message:
                yield message["tool_call_id"]

    def clean_ids(values):
        for value in values:
            require(isinstance(value, str) and value and not any(case in value for case in case_names),
                    "WorldSim administrative call ID reveals a benchmark case identifier")

    fixture_ids = set(administrative_ids(history))
    clean_ids(fixture_ids)
    used, generated_count = set(fixture_ids), 0
    names = {tool["function"]["name"] for tool in fixture["tools"]}
    for index, step in enumerate(record["steps"], 1):
        require(step["step"] == index and step["messages"] == history,
                "Saved live history has inconsistent generated call/tool-response references")
        parsed = step["parsed_response"]
        calls = parsed["tool_calls"]
        terminal = (step["finish_reason"] == "length" or not calls
                    or calls[0]["name"] not in names)
        if terminal:
            require(index == len(record["steps"]) and not step["worldsim"],
                    "A terminal generation unexpectedly submitted a simulator request")
            continue
        require(len(calls) == 1, "A generated native tool call was omitted or multiplied")
        call, expected_id = calls[0], f"call_{index:04d}"
        require(call.get("id") == expected_id and expected_id not in used,
                "Generated call ID is not its opaque step ID or collides with fixture/history")
        clean_ids([call["id"]])
        used.add(expected_id)
        generated_count += 1
        assistant = dict(role="assistant", content="", tool_calls=[dict(id=expected_id, type="function",
                         function=dict(name=call["name"], arguments=call["arguments"]))])
        thinking = "\n".join(value for value in (parsed.get("reasoning"), parsed.get("content")) if value)
        if thinking:
            assistant["thinking"] = thinking
        history.append(assistant)
        if index == collection["max_steps"]:
            require(index == len(record["steps"]) and not step["worldsim"],
                    "A step-censored final tool call has an unexpected response")
            continue
        require(step["worldsim"] and step["worldsim"][-1]["status"] == "complete",
                "A continuing generated call lacks its actual successful tool response")
        visible = [{key: value for key, value in message.items()
                    if key in ("role", "content", "tool_calls", "tool_call_id", "name")} for message in history]
        for response in step["worldsim"]:
            body = json.loads(response["request"]["messages"][1]["content"])
            require(body["CURRENT_CALL"] == dict(name=call["name"], arguments=call["arguments"], id=expected_id),
                    "WorldSim current-call identity differs from its generated opaque step ID")
            require(body["CLEAN_CONVERSATION"] == visible,
                    "WorldSim history differs from the actual visible call/tool-response history")
            clean_ids([body["CURRENT_CALL"]["id"], *administrative_ids(body["CLEAN_CONVERSATION"])])
        history.append(dict(role="tool", name=call["name"], tool_call_id=expected_id,
                            content=step["worldsim"][-1]["tool_response"]))
    require(record["messages"] == history, "Final tool-response references differ from generated opaque IDs")
    return generated_count


def audit_collection(output, stage_name, protocol):
    """Rebuild chronological sources directly from committed raw generation IDs."""
    manifest_path = output / "sources" / (stage_name + "-collection-manifest.json")
    require(sha(manifest_path) == protocol["source_collection_manifest_sha256"],
            "Frozen raw collection manifest changed")
    manifest = read(manifest_path)
    require(manifest["status"] == "complete" and manifest["stage"] == stage_name,
            "Raw collection was not committed complete")
    collection_path = output / "collection/protocol.json"
    require(sha(collection_path) == manifest["collection_protocol_sha256"], "Raw collection protocol changed")
    collection = read(collection_path)
    for name, expected in collection["code_sha256"].items():
        require(sha(ROOT / name) == expected, "Bound raw collection source changed: " + name)
    prepared_path = output / "prepared/manifest.json"
    require(sha(prepared_path) == collection["prepared_manifest_sha256"], "Frozen 41-task prepared manifest changed")
    prepared = read(prepared_path)
    require(prepared["status"] == "complete" and prepared["cases"] == len(prepared["rows"]) == 41,
            "Prepared fixture inventory does not cover all 41 benchmark tasks")
    require(Counter(item["category"] for item in prepared["rows"]) == {"tool": 18, "coding": 15, "browser": 8},
            "Prepared benchmark task category counts differ")
    prepared_cases = {item["case_id"]: item for item in prepared["rows"]}
    require(len(prepared_cases) == 41, "Repeated prepared benchmark task ID")
    sources_path = output / protocol["source_records_path"]
    require(manifest["source_records_path"] == protocol["source_records_path"]
            and manifest["source_records_sha256"] == sha(sources_path), "Raw collection source-record binding differs")
    sources = jsonl(sources_path)
    require(manifest["traces"] == len(sources) == len(manifest["cases"]), "Raw collection trace inventory differs")
    require("tool_call_id_policy" in collection, "Frozen collection omits its opaque call-ID policy")
    case_ids, raw_tokens, steps_total, worldsim_calls, generated_calls = [], 0, 0, 0, 0
    for source in sources:
        case_id = source["provenance"]["collection_case"]
        require(case_id in prepared_cases and case_id in manifest["cases"], "Source has no frozen benchmark fixture")
        case_ids.append(case_id)
        item, record_path = prepared_cases[case_id], output / "collection/cases" / (case_id + ".json")
        require(sha(record_path) == manifest["cases"][case_id]["sha256"], "Raw native collection record changed")
        record = read(record_path)
        require(record["status"] == "complete" and record["case_id"] == case_id
                and record["trace_id"] == source["trace_id"] == item["trace_id"] and record["trace"] == source,
                "Raw record and frozen chronological source differ")
        require(record["signature"] == source["provenance"]["collection_signature"]
                == obj_sha(dict(config=collection, prepared_sha256=record["prepared_sha256"])),
                "Raw collection signature changed")
        prepared_record_path = output / item["path"]
        require(sha(prepared_record_path) == item["sha256"] == record["prepared_sha256"],
                "The raw collection did not start from the frozen prepared fixture")
        fixture = read(prepared_record_path)
        require(fixture["attack_text"] == "" and record["steps"][0]["input_ids"] == fixture["initial_input_ids"],
                "Native initial request differs from the frozen no-payload input")
        generated_calls += audit_opaque_call_references(record, fixture, collection, prepared_cases.keys())
        reconstructed, segments = [], []
        for index, step in enumerate(record["steps"], 1):
            require(step["step"] == index and step["input_ids_sha256"] == obj_sha(step["input_ids"]),
                    "Raw native step numbering/input-ID hash differs")
            require(len(step["generated_ids"]) <= collection["max_new_tokens"], "Raw generation exceeds the frozen token limit")
            if index == 1:
                reconstructed.extend(step["input_ids"])
                segments.append(dict(kind="initial_request", step=index, start=0, end=len(reconstructed),
                                     request_start=0, request_end=len(step["input_ids"])))
            elif step.get("new_request_tail"):
                start = len(reconstructed)
                require(step["new_request_tail"] == step["input_ids"][step["new_request_tail_start"]:],
                        "New visible tool tail was retokenized or altered")
                reconstructed.extend(step["new_request_tail"])
                segments.append(dict(kind="new_tool_and_open_header", step=index, start=start, end=len(reconstructed),
                                     request_start=step["new_request_tail_start"], request_end=len(step["input_ids"])))
            else:
                raise AssertionError("A later raw generation has no new native tool/request tail")
            start = len(reconstructed)
            reconstructed.extend(step["generated_ids"])
            segments.append(dict(kind="raw_generated", step=index, start=start, end=len(reconstructed),
                                 generated_start=0, generated_end=len(step["generated_ids"])))
            raw_tokens += len(step["generated_ids"])
            steps_total += 1
            for response in step.get("worldsim", []):
                request = response["request"]
                body = json.loads(request["messages"][1]["content"])
                require(set(body) == {"TOOL_SCHEMAS", "CLEAN_CONVERSATION", "CURRENT_CALL"},
                        "WorldSim request contains non-allowlisted behavior/attack metadata")
                require(all(not any(key in message for key in ("thinking", "reasoning", "reasoning_content"))
                            for message in body["CLEAN_CONVERSATION"]), "WorldSim received private analysis")
                worldsim_calls += 1
        require(source["input_ids"] == reconstructed and source["provenance"]["segments"] == segments,
                "Chronological native trace does not preserve every raw generated token exactly once")
        require(source["eof_finish_reason"] == record["steps"][-1]["finish_reason"],
                "Observed EOF provenance differs from the raw generation record")
        require(source["provenance"]["raw_generated_tokens"] == sum(len(step["generated_ids"]) for step in record["steps"])
                and source["provenance"]["raw_generated_tokens_preserved_exactly_once"] is True
                and source["provenance"]["real_tool_execution"] is False,
                "Raw native conservation/no-real-tool provenance differs")
    require(len(set(case_ids)) == len(case_ids), "Repeated collected benchmark fixture")
    if stage_name == "all":
        require(set(case_ids) == set(prepared_cases), "Collected traces omit a benchmark task")
    return dict(status="passed", traces=len(sources), generation_steps=steps_total,
                raw_generated_tokens=raw_tokens, worldsim_requests=worldsim_calls,
                exact_original_ids_reconstructed=True, no_payload_all41_prepared=True,
                generated_tool_calls=generated_calls, opaque_generated_call_ids_verified=True,
                current_call_and_history_tool_response_references_verified=True,
                worldsim_administrative_ids_exclude_case_identifiers=True)


def audit_inputs(output, stage_name, protocol, decoder):
    sources_path = output / protocol["source_records_path"]
    require(sha(sources_path) == protocol["source_records_sha256"], "Original source-record hash changed")
    sources = jsonl(sources_path)
    by_trace = {source["trace_id"]: source for source in sources}
    require(len(sources) == len(by_trace), "Repeated original source trace")
    require(all(source["source_split"] == "external" for source in sources),
            "IPI traces must be external to the frozen probe training split")
    require(protocol["source_trace_count"] == 41, "IPI benchmark inventory is not the frozen 41-task set")
    expected_sources = 1 if stage_name == "pilot" else 41
    require(len(sources) == expected_sources, "Wrong number of source traces")
    if stage_name == "all":
        require(len(sources) == 41, "All-stage must include all 41 benchmark tasks")
    catalog = load_catalog(output, protocol)
    parsed, terminal = {}, []
    for source in sources:
        trace_id, ids = source["trace_id"], source["input_ids"]
        require(ids and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in ids),
                "Original native IDs must be nonnegative integers")
        allow = source.get("allow_terminal_body", False)
        require(isinstance(allow, bool), "Terminal-body permission is not boolean")
        provenance = source.get("provenance", {})
        eof = source.get("eof_finish_reason", provenance.get("eof_finish_reason"))
        require(not allow or eof == "length", "Observed EOF body has no raw length-limited generation proof")
        frames, first, first_normal = original_frames(ids, decoder, allow)
        observed = [frame for frame in frames if frame["complete"] and not frame["terminated"]]
        require(bool(observed) == allow, "Terminal-body flag differs from the actual native source extent")
        if observed:
            require(len(observed) == 1 and observed[0] is frames[-1], "Only the observed last frame may lack its ending")
            terminal.append(trace_id)
        parsed[trace_id] = frames, first, first_normal
    for role, entry in catalog.items():
        trace_id = entry["source_trace_id"]
        require(trace_id in by_trace, "Catalog source is absent from the authoritative raw trace records")
        frames, _, _ = parsed[trace_id]
        index = entry.get("source_block_index", entry.get("block_index"))
        require(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(frames),
                "Catalog source block index is invalid")
        frame, ids = frames[index], by_trace[trace_id]["input_ids"]
        require(frame["complete"] and not frame["structural"] and frame["role"] == role,
                "Catalog entry is not an observed normal destination-role frame")
        require(entry["header_ids"] == ids[frame["start"]:frame["body_start"]]
                and entry["ending_ids"] == ids[frame["body_end"]:frame["end"]],
                "Catalog wrapper differs from its exact observed native pilot tokens")
        if not frame["terminated"]:
            require(entry.get("eof_finish_reason") == "length", "Catalog EOF ending has no observed length proof")
        if "pilot_trace_id" in protocol:
            require(trace_id == protocol["pilot_trace_id"], "Catalog source differs from the frozen pilot")
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
    expected_inputs = sum(1 + 7 * len(parsed[trace_id][1]) for trace_id in grouped)
    require(len(rows) == protocol["inputs"] == expected_inputs, "Wrong dynamic 1+7m intervention input count")
    if stage_name == "pilot":
        require(len(grouped) == 1 and set(grouped) == {protocol["pilot_trace_id"]},
                "Pilot must contain exactly its frozen selected trace")
    else:
        require(set(grouped) == set(by_trace), "All-stage omits benchmark traces")
    supports, availability_by_trace, wrapper_counts = Counter(), {}, Counter()
    for trace_id, variants in grouped.items():
        source = by_trace[trace_id]
        ids = source["input_ids"]
        frames, first, first_normal = parsed[trace_id]
        supports.update(first.keys())
        availability = {role: ("absent" if role not in first_normal else
                               "present" if role in first else "empty_first_body") for role in ROLES}
        availability_by_trace[trace_id] = availability
        m = len(first)
        require(Counter(row["operation"] for row in variants) == {"baseline": 1, "strip": m, "swap": 5*m, "move": m},
                "Incomplete intervention grid for " + trace_id)
        expected_cells = {("baseline", None, None)} | {(operation, role, None) for operation in ("strip", "move") for role in first}
        expected_cells |= {("swap", role, destination) for role in first for destination in ROLES if role != destination}
        actual_cells = [(row["operation"], row["source_role"], row["destination_role"]) for row in variants]
        require(len(set(actual_cells)) == 1 + 7*m and set(actual_cells) == expected_cells,
                "Repeated or omitted directed interventions")
        positions_by_role = {role: list(range(frame["body_start"], frame["body_end"])) for role, frame in first.items()}
        for row in variants:
            key, operation, role = row["input_id"], row["operation"], row["source_role"]
            require(row["source_split"] == "external" and row["trace_sha256"] == obj_sha(ids),
                    "Source provenance differs: " + key)
            require(row["role_availability"] == availability, "Missing/empty role status differs: " + key)
            if operation == "baseline":
                expected_origins, expected_ids = list(range(len(ids))), ids
                expected_positions = sorted(value for positions in positions_by_role.values() for value in positions)
                original_positions, role_positions = expected_positions, positions_by_role
                require(row["block_index"] is None and row["target_roles"] == [r for r in ROLES if r in first],
                        "Baseline target selection differs")
            else:
                frame = first[role]
                a, b, c, d = (frame[name] for name in ("start", "body_start", "body_end", "end"))
                original_positions = positions_by_role[role]
                require(row["block_index"] == frame["block_index"], "Intervention did not select first normal role frame")
                require(row["body_token_ids"] == ids[b:c], "Source body IDs differ")
                require(row["source_block_terminated"] == frame["terminated"]
                        and row["observed_terminal_body"] == (not frame["terminated"]), "Observed source termination metadata differs")
                if operation == "strip":
                    expected_origins = list(range(a)) + list(range(b, c)) + list(range(d, len(ids)))
                    expected_ids, new_start = [ids[position] for position in expected_origins], a
                    require(row["removed_header_ids"] == ids[a:b] and row["removed_ending_ids"] == ids[c:d],
                            "Strip removal metadata differs")
                elif operation == "swap":
                    destination, provenance = row["destination_role"], row["wrapper_provenance"]
                    if destination in first_normal:
                        wrapper = first_normal[destination]
                        h0, h1, t0, t1 = (wrapper[name] for name in ("start", "body_start", "body_end", "end"))
                        header, ending = ids[h0:h1], ids[t0:t1]
                        header_origins, ending_origins = list(range(h0, h1)), list(range(t0, t1))
                        expected_provenance = dict(kind="same_trace", role=destination, source_trace_id=trace_id,
                            source_block_index=wrapper["block_index"], header_ids=header, ending_ids=ending)
                        require(row["wrapper_block_index"] == wrapper["block_index"], "Wrong same-trace wrapper frame")
                    else:
                        entry = catalog[destination]
                        header, ending = entry["header_ids"], entry["ending_ids"]
                        header_origins, ending_origins = [-1]*len(header), [-1]*len(ending)
                        expected_provenance = dict(kind="pilot_catalog", role=destination,
                            source_trace_id=entry["source_trace_id"],
                            source_block_index=entry.get("source_block_index", entry.get("block_index")),
                            catalog_entry_sha256=obj_sha(entry), header_ids=header, ending_ids=ending)
                        require(row["wrapper_block_index"] is None, "Catalog wrapper attributed to a same-trace block")
                    expected_provenance["wrapper_tokens_sha256"] = obj_sha(header + ending)
                    require(provenance == expected_provenance, "Swap wrapper provenance differs: " + key)
                    wrapper_counts[provenance["kind"]] += 1
                    require(row["wrapper_original_positions"] == header_origins + ending_origins,
                            "Wrapper token-source map differs")
                    expected_origins = list(range(a)) + header_origins + list(range(b, c)) + ending_origins + list(range(d, len(ids)))
                    expected_ids = ids[:a] + header + ids[b:c] + ending + ids[d:]
                    new_start = a + len(header)
                    require(row["applied_header_ids"] == header and row["applied_ending_ids"] == ending
                            and row["applied_wrapper_terminated"] == bool(ending), "Applied swap wrapper IDs/termination differ")
                else:
                    remaining = list(range(a)) + list(range(d, len(ids)))
                    boundaries, cursor = [0], 0
                    for other in frames:
                        if other is not frame and other["complete"]:
                            cursor += other["end"] - other["start"]
                            boundaries.append(cursor)
                    candidates = [(boundary + b - a, boundary) for boundary in boundaries]
                    new_start, insertion = min(candidates, key=lambda pair: (-abs(pair[0] - b), pair[0]))
                    expected_origins = remaining[:insertion] + list(range(a, d)) + remaining[insertion:]
                    expected_ids = [ids[position] for position in expected_origins]
                    require(sorted(expected_origins) == list(range(len(ids))), "Move does not preserve every original token exactly once")
                    require(row["move_distance_tokens"] == abs(new_start - b), "Recorded movement distance differs")
                    require(row["applied_header_ids"] == ids[a:b] and row["applied_ending_ids"] == ids[c:d],
                            "Move changed the source's observed wrapper")
                expected_positions = list(range(new_start, new_start + c - b))
                role_positions = {role: expected_positions}
                require(row["original_body_start"] == b and row["new_body_start"] == new_start,
                        "Body-start metadata differs: " + key)
            require(row["token_source_positions"] == expected_origins, "Full-input token ownership differs: " + key)
            require(row["input_ids"] == expected_ids, "Intervention changed or invented unattributed tokens: " + key)
            require(row["target_positions"] == expected_positions and row["target_original_positions"] == original_positions,
                    "Selected target body ownership differs: " + key)
            require(row["role_target_positions"] == role_positions, "Role capture masks differ: " + key)
            require(row["original_role_target_positions"] == {r: positions_by_role[r] for r in role_positions},
                    "Original role body masks differ: " + key)
            require(len(row["input_ids"]) <= 131072 and row["input_tokens"] == len(row["input_ids"])
                    and row["target_token_count"] == len(expected_positions), "Invalid/truncated input or capture length")
    require(sum(len(row["target_positions"]) for row in rows) == protocol["expected_token_rows"],
            "Frozen selected body token count differs")
    require(sum(len(row["role_target_positions"]) for row in rows) == protocol["expected_summary_rows"],
            "Frozen role-body summary count differs")
    return rows, dict(status="passed", traces=len(grouped), inputs=len(rows),
                      trace_split_counts={"external": len(grouped)}, role_support={role: supports[role] for role in ROLES},
                      role_availability=availability_by_trace, wrapper_provenance_counts=dict(wrapper_counts),
                      terminal_censored_traces=terminal, all_token_native_interventions_independently_rebuilt=True,
                      catalog_wrappers_match_authoritative_native_pilot=True)


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
    require(sha(HERE / "ipi_trace_interventions_data.py") == protocol["parser_sha256"], "Prepared parser source changed")
    require(sha(HERE / "run_ipi_trace_interventions.py") == protocol["runner_sha256"], "Prepared runner source changed")
    for name, expected in protocol["runtime_code_sha256"].items():
        require(sha(ROOT / name) == expected, "Bound native runtime source changed: " + name)
    model = args.model.resolve() if args.model else Path(protocol["model_path"])
    for name, expected in metadata["model_binding"]["assets_sha256"].items():
        require(sha(model / name) == expected, "Bound tokenizer/backbone asset changed: " + name)
    collection_result = audit_collection(output, stage_name, protocol)
    rows, input_result = audit_inputs(output, stage_name, protocol, HeaderDecoder(model / "tokenizer.json"))
    result = dict(status="passed", stage=stage_name, checkpoint_sha256=PROBE_SHA,
                  protocol_sha256=sha(stage / "protocol.json"), inputs_sha256=sha(stage / "inputs.jsonl"),
                  validator_sha256=sha(__file__), inputs=input_result, collection=collection_result,
                  validation_scope="inputs only")
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
    require(len(summaries) == len(indexed_summary) == protocol["expected_summary_rows"]
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
                         parser_sha256=protocol["parser_sha256"], runner_sha256=protocol["runner_sha256"],
                         wrapper_catalog_sha256=protocol["wrapper_catalog_sha256"])
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
