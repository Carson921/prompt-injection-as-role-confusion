"""CPU-only inventory of actual first-role body origins in collected IPI traces.

The inventory labels the initial request as template_or_fixture_prefix, not as
verbatim fixture text. It separately intersects body/header/ending spans with
the collector's chronological segments. No model, inference runner, WorldSim,
or task tool is loaded or called. A stage must have completed collection first.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import ipi_trace_interventions_data as native


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINS = {"initial_request": "template_or_fixture_prefix",
           "raw_generated": "target_generation",
           "new_tool_and_open_header": "simulated_tool_return"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class HeaderTokenizer:
    """Decode only native headers with the frozen JSON vocabulary; no HF/Torch."""
    def __init__(self, path):
        data = read(path)
        self.vocab = {value: key for key, value in data["model"]["vocab"].items()}
        self.vocab.update({row["id"]: row["content"] for row in data["added_tokens"]})
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        original, rendered, extra = visible.copy(), visible.copy(), 0
        for byte in range(256):
            if byte not in visible:
                original.append(byte)
                rendered.append(256 + extra)
                extra += 1
        self.byte_values = {chr(character): byte for character, byte in zip(rendered, original)}

    def decode(self, ids, *, skip_special_tokens=False):
        require(not skip_special_tokens, "Header decoder must preserve special tokens")
        text = "".join(self.vocab[value] for value in ids)
        return bytes(self.byte_values[character] for character in text).decode("utf-8")


def overlap(start, end, segments):
    counts, spans = Counter(), []
    for segment in segments:
        left, right = max(start, segment["start"]), min(end, segment["end"])
        if left < right:
            origin = ORIGINS[segment["kind"]]
            counts[origin] += right - left
            spans.append(dict(origin=origin, kind=segment["kind"], step=segment["step"],
                              start=left, end=right, tokens=right-left))
    require(sum(counts.values()) == end-start, "Source spans do not cover the audited interval")
    return dict(counts), spans


def finer_origin(role, body_counts, prepared):
    if not body_counts:
        return "empty_body"
    if len(body_counts) != 1:
        return "mixed_chronological_segments"
    origin = next(iter(body_counts))
    if origin == "target_generation":
        return "newly_sampled_target_" + role
    if origin == "simulated_tool_return":
        return "template_serialized_simulated_tool_return"
    if role == "system":
        return "native_default_system_identity_date_reasoning"
    if role == "developer":
        return ("template_rendered_fixture_instructions_and_tool_schemas"
                if prepared["provenance"]["fixture_system_prompt_present"]
                else "template_rendered_tool_schemas_without_fixture_system_prompt")
    return {"user": "template_rendered_fixture_user_body",
            "cot": "template_assigned_fixture_assistant_analysis",
            "assistant": "template_rendered_fixture_assistant_final",
            "tool": "template_serialized_fixture_tool_return"}[role]


def reconstruct(case):
    ids, segments = [], []
    for index, step in enumerate(case["steps"], 1):
        require(step["step"] == index, "Nonchronological target-generation steps")
        if index == 1:
            ids.extend(step["input_ids"])
            segments.append(dict(kind="initial_request", step=index, start=0, end=len(ids),
                                 request_start=0, request_end=len(step["input_ids"])))
        else:
            start_in_request = step["new_request_tail_start"]
            tail = step["input_ids"][start_in_request:]
            require(tail == step["new_request_tail"], "New native request tail differs")
            if tail:
                start = len(ids)
                ids.extend(tail)
                segments.append(dict(kind="new_tool_and_open_header", step=index, start=start,
                                     end=len(ids), request_start=start_in_request,
                                     request_end=len(step["input_ids"])))
        start = len(ids)
        ids.extend(step["generated_ids"])
        segments.append(dict(kind="raw_generated", step=index, start=start, end=len(ids),
                             generated_start=0, generated_end=len(step["generated_ids"])))
    return ids, segments


def audit(output: Path, stage: str, model: Path | None = None, *, write_outputs: bool = True):
    output = output.resolve()
    source_path = output / "sources" / ("pilot-traces.jsonl" if stage == "pilot" else "traces.jsonl")
    manifest_path = output / "sources" / (stage + "-collection-manifest.json")
    collection_path, prepared_path = output / "collection/protocol.json", output / "prepared/manifest.json"
    manifest, collection, prepared_manifest = read(manifest_path), read(collection_path), read(prepared_path)
    require(manifest["status"] == "complete" and manifest["traces"] == (1 if stage == "pilot" else 41),
            "Collection stage must be complete before provenance auditing")
    require(manifest["source_records_sha256"] == sha(source_path)
            and manifest["collection_protocol_sha256"] == sha(collection_path)
            and collection["prepared_manifest_sha256"] == sha(prepared_path), "Source binding differs")
    require(collection["code_sha256"]["experiments/role-analysis/ipi_trace_interventions_data.py"]
            == sha(HERE / "ipi_trace_interventions_data.py"), "Frozen parser changed")
    model = Path(model) if model is not None else Path(prepared_manifest["model_path"])
    require(sha(model / "tokenizer.json") == prepared_manifest["tokenizer_sha256"]["tokenizer.json"],
            "Frozen tokenizer differs")
    tokenizer = HeaderTokenizer(model / "tokenizer.json")
    sources = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    require(len(sources) == manifest["traces"] and len({row["trace_id"] for row in sources}) == len(sources),
            "Source trace inventory differs")
    prepared_rows = {row["case_id"]: row for row in prepared_manifest["rows"]}
    source_cases = [row["provenance"]["collection_case"] for row in sources]
    expected_cases = ({collection["pilot_case_id"]} if stage == "pilot" else set(prepared_rows))
    require(len(set(source_cases)) == len(source_cases) and set(source_cases) == expected_cases,
            "Source case names differ from the fixed complete stage inventory")
    body_rows, trace_rows = [], []
    role_support, missing = Counter(), defaultdict(Counter)
    origin_traces, origin_tokens, header_tokens = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    body_hashes, origin_body_hashes = defaultdict(set), defaultdict(lambda: defaultdict(set))
    endings, eof_reasons = Counter(), Counter()
    for source in sources:
        case_id = source["provenance"]["collection_case"]
        case_path = output / "collection/cases" / (case_id + ".json")
        case, spec = read(case_path), prepared_rows[case_id]
        prepared = read(output / spec["path"])
        require(source["trace_id"] == spec["trace_id"]
                and source["category"] == spec["category"]
                and source["source_split"] == "external",
                "Source identity/category/external cohort differs from prepared inventory")
        require(sha(case_path) == manifest["cases"][case_id]["sha256"]
                and sha(output / spec["path"]) == spec["sha256"], "Case/prepared binding differs")
        require(case["status"] == "complete" and case["trace"] == source
                and case["steps"][0]["input_ids"] == prepared["initial_input_ids"],
                "Collection source differs from the frozen prepared input")
        ids, segments = reconstruct(case)
        require(ids == source["input_ids"] and segments == source["provenance"]["segments"],
                "Chronological source reconstruction differs")
        require(all(len(step["generated_ids"]) <= collection["max_new_tokens"] for step in case["steps"])
                and len(case["steps"]) <= collection["max_steps"], "Generation budget exceeded")
        opaque_start = source.get("opaque_terminal_start")
        blocks = native.parse_native_blocks(ids, tokenizer,
                    allow_terminal_body=source.get("allow_terminal_body", False),
                    opaque_terminal_start=opaque_start)
        if opaque_start is not None:
            require(blocks[-1].get("opaque") is True
                    and blocks[-1]["start"] == opaque_start
                    and blocks[-1]["opaque_ids"] == ids[opaque_start:],
                    "Opaque terminal suffix was not preserved exactly")
        first = {}
        for block in blocks:
            if block["complete"] and not block["structural_tool_call"] and block["role"] in native.ROLES:
                first.setdefault(block["role"], block)
        present = []
        for role in native.ROLES:
            block = first.get(role)
            availability = "absent" if block is None else "present" if block["body_ids"] else "empty_first_body"
            row = dict(trace_id=source["trace_id"], category=source["category"], source_role=role,
                       availability=availability, scored=availability == "present")
            if block is not None:
                require(opaque_start is None or block["end"] <= opaque_start,
                        "A first-role source overlaps uninterpreted terminal context")
                bc, bs = overlap(block["body_start"], block["body_end"], segments)
                hc, hs = overlap(block["start"], block["body_start"], segments)
                ec, es = overlap(block["body_end"], block["end"], segments)
                row.update(block_index=block["block_index"], body_start=block["body_start"],
                           body_end=block["body_end"], body_tokens=len(block["body_ids"]),
                           body_ids_sha256=object_sha(block["body_ids"]),
                           body_origin_token_counts=bc, body_origin_spans=bs,
                           body_origin_detail=finer_origin(role, bc, prepared),
                           header_origin_token_counts=hc, header_origin_spans=hs,
                           ending_origin_token_counts=ec, ending_origin_spans=es,
                           terminated=block["terminated"], observed_terminal_body=block["observed_terminal_body"])
                header_tokens[role].update(hc)
                if availability == "present":
                    present.append(role)
                    role_support[role] += 1
                    origin_traces[role].update(bc.keys())
                    origin_tokens[role].update(bc)
                    body_hashes[role].add(row["body_ids_sha256"])
                    for origin in bc:
                        origin_body_hashes[role][origin].add(row["body_ids_sha256"])
            if availability != "present":
                missing[role][availability] += 1
            body_rows.append(row)
        attempts = [attempt for step in case["steps"] for attempt in step["worldsim"]]
        endings[case["terminal_reason"]] += 1
        eof_reasons[source["eof_finish_reason"]] += 1
        last = blocks[-1]
        trace_rows.append(dict(trace_id=source["trace_id"], case_id=case_id, category=source["category"],
            trace_tokens=len(ids), initial_request_tokens=len(prepared["initial_input_ids"]),
            generations=len(case["steps"]), raw_generated_tokens=sum(len(x["generated_ids"]) for x in case["steps"]),
            terminal_reason=case["terminal_reason"], eof_finish_reason=source["eof_finish_reason"],
            opaque_terminal_start=opaque_start,
            opaque_terminal_tokens=len(ids)-opaque_start if opaque_start is not None else 0,
            observed_terminal_body=last["observed_terminal_body"], last_frame_role=last["role"],
            last_frame_structural_tool_call=last["structural_tool_call"],
            last_generation_tokens=len(case["steps"][-1]["generated_ids"]),
            worldsim_attempts=len(attempts), worldsim_complete=sum(x["status"] == "complete" for x in attempts),
            worldsim_error_attempts=sum(x["status"] != "complete" for x in attempts),
            roles_present=present, inputs_for_present_roles=1+7*len(present)))
    result = dict(status="passed", stage=stage, traces=len(sources),
        auditor_sha256=sha(Path(__file__)), parser_sha256=sha(HERE / "ipi_trace_interventions_data.py"),
        source_traces_sha256=sha(source_path), collection_manifest_sha256=sha(manifest_path),
        collection_protocol_sha256=sha(collection_path), prepared_manifest_sha256=sha(prepared_path),
        tokenizer_sha256=sha(model / "tokenizer.json"),
        source_origin_definitions=ORIGINS,
        fixed_stage_case_inventory_verified=True,
        categories=dict(Counter(row["category"] for row in sources)),
        source_splits=dict(Counter(row["source_split"] for row in sources)),
        origin_caveat="Prefix means native template OR fixture. Generated CoT/final can be context even when all selected first bodies are from the prefix. Header and ending origins are counted separately; a generated frame can have an initial-request assistant header prefix.",
        exact_chronological_token_reconstruction=True, first_ordinary_frame_only=True,
        role_support={role: role_support[role] for role in native.ROLES},
        missing_reasons={role: dict(missing[role]) for role in native.ROLES},
        body_origin_trace_counts={role: dict(origin_traces[role]) for role in native.ROLES},
        body_origin_token_counts={role: dict(origin_tokens[role]) for role in native.ROLES},
        unique_first_body_id_sequences={role: len(body_hashes[role]) for role in native.ROLES},
        unique_first_body_id_sequences_by_origin={role: {origin: len(values) for origin, values in origin_body_hashes[role].items()}
                                                 for role in native.ROLES},
        first_frame_header_origin_token_counts={role: dict(header_tokens[role]) for role in native.ROLES},
        terminal_reasons=dict(endings), eof_finish_reasons=dict(eof_reasons),
        total_raw_generated_tokens=sum(row["raw_generated_tokens"] for row in trace_rows),
        opaque_terminal_traces=sum(row["opaque_terminal_start"] is not None for row in trace_rows),
        opaque_terminal_tokens=sum(row["opaque_terminal_tokens"] for row in trace_rows),
        opaque_terminal_policy="The entire unparseable final generation and its actual open-header prefix are preserved as uninterpreted context, excluded from source/wrapper selection. Move-end is the final verified native boundary before that suffix.",
        max_new_tokens=collection["max_new_tokens"], max_steps=collection["max_steps"],
        trace_inventory=trace_rows, first_body_inventory=body_rows)
    if not write_outputs:
        return result
    folder = output / stage
    folder.mkdir(exist_ok=True)
    json_path, csv_path = folder / "first-body-provenance.json", folder / "first-body-provenance.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    keys = ["trace_id", "category", "source_role", "availability", "scored", "block_index",
            "body_start", "body_end", "body_tokens", "body_ids_sha256", "body_origin_detail",
            "body_origin_token_counts", "header_origin_token_counts", "ending_origin_token_counts",
            "terminated", "observed_terminal_body"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in body_rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                             for key, value in row.items() if key in keys})
    print(json.dumps({"status": "passed", "stage": stage, "traces": len(sources),
                      "role_support": result["role_support"], "body_origin_trace_counts": result["body_origin_trace_counts"],
                      "terminal_reasons": result["terminal_reasons"], "json_sha256": sha(json_path)}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "all"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Read and validate without writing artifacts")
    args = parser.parse_args()
    result = audit(args.output, args.stage, args.model, write_outputs=not args.check_only)
    if args.check_only:
        print(json.dumps({"status": result["status"], "traces": result["traces"],
                          "role_support": result["role_support"],
                          "body_origin_trace_counts": result["body_origin_trace_counts"],
                          "total_raw_generated_tokens": result["total_raw_generated_tokens"]}))
