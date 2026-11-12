"""Reconstruct immutable chronological benign traces from sealed native calls.

The archived native API removes analysis before the next request and renders
tool-call JSON again. A final request therefore is not a full running trace.
This script keeps each exact original generation, plus the exact input tokens
of newly introduced tool results, once. It never tokenizes or normalizes a
body. Each native reset starts a new recorded pass, explicitly marked in the
provenance of the retrospective task transcript.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from transformers import AutoTokenizer

from agentdojo_trace_interventions_data import (
    ROLES, first_role_targets, parse_native_blocks, require,
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def safe_extract(archive: Path, destination: Path):
    """Reject links, special files, absolute paths and parent traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            rel = Path(member.name)
            require(not rel.is_absolute() and ".." not in rel.parts,
                    "Unsafe archive path: " + member.name)
            require(not member.issym() and not member.islnk(),
                    "Archive links are unsupported: " + member.name)
            target = destination / rel
            require(target.resolve().is_relative_to(destination.resolve()),
                    "Archive member escapes extraction root")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                require(member.isfile(), "Unsupported archive special file")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream, target.open("wb") as output:
                    while chunk := stream.read(8 * 1024 * 1024):
                        output.write(chunk)


def validate_source(root: Path, checkpoint_root: Path, archive: Path | None):
    audit = read(checkpoint_root / "agentdojo-source-audit.json")
    sealed = read(root / "result_manifest.json")
    require(sha(root / "result_manifest.json") == audit["result_manifest_sha256"],
            "Result manifest differs from the actual checkpoint's training source")
    expected = {item["path"]: item for item in sealed["files"]}
    checked, total_bytes = {}, 0
    for relative, item in expected.items():
        path = root / relative
        require(path.is_file() and path.stat().st_size == item["bytes"],
                "Missing or changed sealed source: " + relative)
        digest = sha(path)
        require(digest == item["sha256"], "Sealed source hash mismatch: " + relative)
        checked[relative] = digest
        total_bytes += item["bytes"]
    for relative, item in audit["consumed_source_files"].items():
        require(checked.get(relative) == item["sha256"],
                "Checkpoint-consumed source hash mismatch: " + relative)
    return dict(status="passed", source_root=str(root.resolve()),
                archive=str(archive.resolve()) if archive else None,
                archive_sha256=sha(archive) if archive else None,
                checkpoint_sha256=sha(checkpoint_root / "checkpoint.npz"),
                checkpoint_source_audit_sha256=sha(checkpoint_root / "agentdojo-source-audit.json"),
                result_manifest_sha256=sha(root / "result_manifest.json"),
                sealed_files_verified=len(checked), sealed_bytes_verified=total_bytes,
                checkpoint_consumed_files_verified=len(audit["consumed_source_files"]))


def reconstruct_task(root, task, calls, tokenizer, retained, source_split):
    ids, segments, call_records, passes = [], [], [], []
    terminal_censoring = None
    previous_input_blocks = None
    previous_generated_blocks = None
    previous_pass = None

    def append(values, call_id, scope, source_start, pass_index, reason):
        if not values:
            return
        segments.append(dict(trace_start=len(ids), trace_end=len(ids) + len(values),
                             call_id=call_id, scope=scope,
                             source_token_start=source_start,
                             source_token_end=source_start + len(values),
                             native_pass_index=pass_index, reason=reason))
        ids.extend(values)

    for entry in calls:
        call_id, pass_index = entry["call_id"], entry["native_pass_index"]
        folder = root / "calls" / call_id
        request = read(folder / "request.json")
        require(request["status"] == "complete", "Incomplete archived request")
        with np.load(folder / "input.npz", allow_pickle=False) as source:
            input_ids = source["input_ids"].tolist()
        generated_ids = request["generated_ids"]
        with np.load(folder / "generation_probe.npz", allow_pickle=False) as source:
            require(source["generated_ids"].tolist() == generated_ids,
                    "Generation request and original NPZ token IDs differ")
        require(len(input_ids) == request["input_tokens"], "Input length mismatch")
        require(tokenizer.decode(input_ids, skip_special_tokens=False) == request["rendered_prompt"],
                "Original input IDs do not decode to the archived native prompt")
        require(tokenizer.decode(generated_ids, skip_special_tokens=False) == request["generated_text"],
                "Original generated IDs do not decode to the archived generation")
        blocks = parse_native_blocks(input_ids, tokenizer)
        require(not blocks[-1]["complete"] and blocks[-1]["header_text"] == "assistant",
                "Archived request has no native open assistant header")
        complete, open_header = blocks[:-1], blocks[-1]
        finish_length = request["finish_reason"] == "length"
        if finish_length:
            require(entry is calls[-1] and request["parsing_diagnostics"]["length_limited"],
                    "Only the authoritative final length-limited generation may contain a censored final body")
            terminal_censoring = dict(call_id=call_id, finish_reason="length",
                                      observed_generated_tokens=len(generated_ids),
                                      original_last_token_id=generated_ids[-1],
                                      artificial_ending_tokens_added=0,
                                      final_body_bounded_by="original_generation_EOF",
                                      interpretation="Observed final assistant body is length-censored; its true ending wrapper was never generated.")
        generated_blocks = parse_native_blocks(input_ids[open_header["start"]:] + generated_ids,
                                               tokenizer, allow_open_header=False,
                                               allow_terminal_body=finish_length)
        require(all(block["complete"] for block in generated_blocks),
                "Incomplete raw generation cannot be repaired or synthesized")
        reset = pass_index != previous_pass
        if reset:
            require(previous_pass is None or pass_index == previous_pass + 1,
                    "Native pass ordering differs")
            passes.append(dict(native_pass_index=pass_index, trace_start=len(ids),
                               first_call_id=call_id, reset_prefix_input_tokens=open_header["start"],
                               previous_pass_last_generated_ending_ids=(
                                   previous_generated_blocks[-1]["ending_ids"] if previous_generated_blocks else None)))
            append(input_ids, call_id, "actual_input", 0, pass_index,
                   "initial_native_input" if previous_pass is None else "exact_native_reset_input")
            new_tool_blocks = []
        else:
            # Native history is append-only within a pass. It contains a
            # canonical re-rendering of the prior raw tool call and its result.
            old = previous_input_blocks
            require(len(complete) >= len(old), "Native input history shrank without a documented reset")
            for before, after in zip(old, complete):
                require(before["header_ids"] + before["body_ids"] + before["ending_ids"] ==
                        after["header_ids"] + after["body_ids"] + after["ending_ids"],
                        "Earlier native history changed within a pass")
            new_blocks = complete[len(old):]
            native_calls = [block for block in new_blocks if block["structural_tool_call"]]
            raw_calls = [block for block in previous_generated_blocks if block["structural_tool_call"]]
            require(len(native_calls) == len(raw_calls), "Native tool-call multiplicity differs")
            for native, raw in zip(native_calls, raw_calls):
                require(native["recipient"].split("<|", 1)[0] == raw["recipient"].split("<|", 1)[0],
                        f"Native tool-call recipient changed in {call_id}: {native['recipient']!r} vs {raw['recipient']!r}")
                native_body = tokenizer.decode(native["body_ids"], skip_special_tokens=False)
                raw_body = tokenizer.decode(raw["body_ids"], skip_special_tokens=False)
                require(json.loads(native_body) == json.loads(raw_body),
                        "Native tool-call JSON differs semantically from the exact raw generation")
            new_tool_blocks = [block for block in new_blocks if block["role"] == "tool"]
            require(len(new_blocks) == len(native_calls) + len(new_tool_blocks),
                    "New native frame is neither a prior tool call nor its result")
            require(len(new_tool_blocks) == len(native_calls), "Native tool-result multiplicity differs")
            for block in new_tool_blocks:
                append(input_ids[block["start"]:block["end"]], call_id, "actual_input",
                       block["start"], pass_index, "new_native_tool_result")
            append(input_ids[open_header["start"]:], call_id, "actual_input", open_header["start"],
                   pass_index, "native_open_assistant_header")
        append(generated_ids, call_id, "actual_generated", 0, pass_index, "exact_raw_generation")
        call_records.append(dict(call_id=call_id, native_pass_index=pass_index,
                                 input_tokens=len(input_ids), generated_tokens=len(generated_ids),
                                 newly_appended_tool_results=len(new_tool_blocks),
                                 finish_reason=request["finish_reason"],
                                 parsing_diagnostics=request["parsing_diagnostics"],
                                 generated_last_token=generated_ids[-1],
                                 source_files={filename: sha(folder / filename) for filename in
                                               ("request.json", "input.npz", "generation_probe.npz")}))
        previous_input_blocks, previous_generated_blocks, previous_pass = complete, generated_blocks, pass_index
    for index, item in enumerate(passes):
        item["trace_end"] = passes[index + 1]["trace_start"] if index + 1 < len(passes) else len(ids)
        item["tokens"] = item["trace_end"] - item["trace_start"]

    blocks = parse_native_blocks(ids, tokenizer, allow_open_header=False,
                                 allow_terminal_body=terminal_censoring is not None)
    targets = first_role_targets(blocks)
    ownership = {}
    for segment in segments:
        for trace_pos, source_pos in zip(range(segment["trace_start"], segment["trace_end"]),
                                         range(segment["source_token_start"], segment["source_token_end"])):
            ownership[trace_pos] = (segment["call_id"], segment["scope"], source_pos)
    lineage = {}
    for role, block in targets.items():
        mapping = [ownership[pos] for pos in range(block["body_start"], block["body_end"])]
        require(len({(call, scope) for call, scope, _ in mapping}) == 1,
                "Target body spans multiple source calls")
        call_id, scope = mapping[0][:2]
        indices = [item[2] for item in mapping]
        candidates = [row for row in retained if row["call_id"] == call_id and row["scope"] == scope
                      and row["probe_role"] == role and json.loads(row["token_indices"]) == indices]
        require(len(candidates) == 1, f"First {role} body lacks exact checkpoint-retained token-mask lineage: {task}")
        row = candidates[0]
        lineage[role] = dict(block_index=block["block_index"], trace_body_start=block["body_start"],
                             trace_body_end=block["body_end"], body_tokens=len(block["body_ids"]),
                             call_id=call_id, scope=scope, source_token_indices=indices,
                             stable_frame_key=row["stable_frame_key"],
                             native_pass_index=int(row["native_pass_index"]),
                             source_training_mask_exact_match=True,
                             body_token_ids_sha256=hashlib.sha256(
                                 np.asarray(block["body_ids"], dtype="<i8").tobytes()).hexdigest())
    return dict(trace_id=task, source_split=source_split, input_ids=ids,
                allow_terminal_body=terminal_censoring is not None,
                input_tokens=len(ids), source_suite=task.split("/")[0],
                provenance=dict(format_version="chronological-native-benign-trace-v1",
                                definition="Exact original initial input + exact raw generated IDs + newly introduced native tool frames; native resets retain their exact input prefix.",
                                context_kind="retrospective_chronological_transcript",
                                all_raw_cot_preserved=True, body_retokenization=False,
                                normalized_tool_call_bodies_substituted=False,
                                native_passes=passes, calls=call_records, segments=segments,
                                terminal_censoring=terminal_censoring,
                                first_role_training_lineage=lineage,
                                no_payload=True, complete_native_blocks=len(blocks),
                                source_token_ids_sha256=hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    verification = validate_source(args.source_root, args.checkpoint_root, args.archive)
    write(args.output / "source-verification.json", verification)
    print(json.dumps({"stage": "source_verified", **verification}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    assignment = read(args.checkpoint_root / "split-assignment.json")["domains"]["agentdojo"]
    evidence = read(args.source_root / "analysis/body_mapping_evidence.json")
    require(evidence["valid"], "Invalid authoritative body mapping")
    grouped = defaultdict(list)
    for entry in evidence["calls"]:
        grouped[entry["task_key"]].append(entry)
    with (args.source_root / "analysis/call_body_probabilities.csv").open(encoding="utf-8-sig") as stream:
        retained = list(csv.DictReader(stream))
    tag_roles = dict(system="system", developer="developer", user="user", tool="tool",
                     assistant_analysis="cot", assistant_final="assistant", assistant_commentary="assistant")
    for row in retained:
        row["probe_role"] = tag_roles[row["tag"]]
    records = []
    for task in sorted(grouped):
        calls = sorted(grouped[task], key=lambda x: int(x["call_id"].rsplit("_", 1)[1]))
        record = reconstruct_task(args.source_root, task, calls, tokenizer,
                                  [row for row in retained if row["task_key"] == task and row["content_filter"] == "normal_content"],
                                  assignment["agentdojo:" + task])
        records.append(record)
        print(json.dumps({"stage": "trace_reconstructed", "trace_id": task,
                          "source_split": record["source_split"], "input_tokens": record["input_tokens"],
                          "native_passes": len(record["provenance"]["native_passes"])}), flush=True)
    require(len(records) == 57 and Counter(record["source_split"] for record in records) ==
            dict(fit=40, dev=8, test=9), "Task-disjoint source inventory changed")
    with (args.output / "traces.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    fit = sorted((record for record in records if record["source_split"] == "fit"),
                 key=lambda record: (record["input_tokens"], record["trace_id"]))
    pilot = fit[0]
    write(args.output / "pilot-selection.json", dict(
        selection_rule="Shortest complete FIT training task transcript with all six first-role targets, selected before any new inference scores; lexicographic tie break.",
        trace_id=pilot["trace_id"], input_tokens=pilot["input_tokens"], source_split="fit",
        first_role_training_lineage=pilot["provenance"]["first_role_training_lineage"],
        candidate_lengths=[dict(trace_id=record["trace_id"], input_tokens=record["input_tokens"]) for record in fit]))
    write(args.output / "source-manifest.json", dict(
        status="complete", traces=57, splits=dict(Counter(record["source_split"] for record in records)),
        calls=sum(len(record["provenance"]["calls"]) for record in records),
        native_passes=sum(len(record["provenance"]["native_passes"]) for record in records),
        trace_tokens=sum(record["input_tokens"] for record in records),
        traces_sha256=sha(args.output / "traces.jsonl"), tokenizer_files={
            name: sha(args.tokenizer / name) for name in ("tokenizer.json", "tokenizer_config.json")},
        pilot_trace_id=pilot["trace_id"], pilot_tokens=pilot["input_tokens"],
        source_verification=verification, script_sha256=sha(Path(__file__)),
        reset_tasks=[record["trace_id"] for record in records if len(record["provenance"]["native_passes"]) > 1],
        terminal_censored_tasks=[record["trace_id"] for record in records if record["allow_terminal_body"]],
        context_limitation="Native API drops analysis from later histories. Keeping all original analysis produces a retrospective transcript, so newly measured probabilities are teacher-forced transcript readouts, not the archived live-call probabilities."))
    print(json.dumps({"stage": "complete", "pilot_trace_id": pilot["trace_id"],
                      "pilot_tokens": pilot["input_tokens"], "traces": len(records)}), flush=True)


if __name__ == "__main__":
    main()
