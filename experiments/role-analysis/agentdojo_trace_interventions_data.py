"""Token-native role interventions on an immutable benign running trace.

This module does not render, normalize, re-tokenize, or run a language model.
The source extractor owns construction of a complete chronological trace.
Only a header is decoded, to classify the six probe roles. Content remains an
opaque token sequence, including native JSON escaping in tool responses.

All position intervals use Python's half-open convention. ``header_end`` is
the exclusive end of the complete opening wrapper (including <|message|>),
and is therefore equal to ``body_start``. A trailing open assistant header is
preserved as a separate incomplete block and is never an intervention target.
"""
from __future__ import annotations

import hashlib
import numbers
import re
from typing import Any, Mapping, Sequence


ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
START, CHANNEL, MESSAGE = 200006, 200005, 200008
END, RETURN, CALL = 200007, 200002, 200012
ENDINGS = frozenset((END, RETURN, CALL))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _token_list(values: Sequence[int]) -> list[int]:
    result = list(values)
    require(all(isinstance(value, numbers.Integral) and not isinstance(value, bool)
                and value >= 0 for value in result), "Token IDs must be nonnegative integers")
    return [int(value) for value in result]


def _header_role(header_ids: Sequence[int], tokenizer: Any) -> dict[str, Any]:
    """Classify a complete/open header, giving recipients precedence over CoT."""
    text = tokenizer.decode(list(header_ids), skip_special_tokens=False)
    parts = text.split("<|channel|>")
    require(len(parts) <= 2, f"Multiple native channels in a header: {text!r}")
    base = parts[0].strip()
    channel = parts[1].strip().split()[0] if len(parts) == 2 and parts[1].strip() else None
    recipients = re.findall(r"(?:^|\s)to=([^\s<]+)", text.replace("<|channel|>", " "))
    require(len(recipients) <= 1, f"Multiple recipients in a native header: {text!r}")
    recipient = recipients[0] if recipients else None
    structural = False
    if base in ("system", "developer", "user"):
        require(channel is None and recipient is None, f"Unsupported ordinary role header: {text!r}")
        role = base
    elif base.startswith("functions."):
        require(base.split()[0] != "functions.", "Native tool header has an empty tool name")
        role = "tool"
    elif base and base.split(maxsplit=1)[0] == "assistant":
        if recipient is not None:
            # This is an actual tool-call frame, not the probe's assistant class.
            structural, role = True, None
        elif channel == "analysis":
            role = "cot"
        elif channel in (None, "final", "commentary"):
            role = "assistant"
        else:
            raise ValueError(f"Unsupported assistant channel in native header: {text!r}")
    else:
        raise ValueError(f"Unknown native role header: {text!r}")
    return dict(role=role, header_text=text, channel=channel,
                recipient=recipient, structural_tool_call=structural)


def parse_native_blocks(input_ids: Sequence[int], tokenizer: Any, *,
                        allow_open_header: bool = True,
                        allow_terminal_body: bool = False) -> list[dict[str, Any]]:
    """Parse strict, contiguous Harmony blocks without changing any token ID.

    Only an optional trailing assistant header may be incomplete. An unfinished
    body, nested structural start/message marker, stray token, unknown header,
    or a non-call frame ending in <|call|> is rejected rather than repaired.
    This token-only reader cannot disambiguate literal structural delimiters
    embedded in body text; such sources require authoritative source spans.
    Explicit ``allow_terminal_body=True`` permits an observed normal assistant
    body cut off at EOF. It has ``complete=True`` as a bounded observed frame,
    ``terminated=False``, and no invented closing token. The source extractor
    must separately establish its audited length-limited generation provenance.
    """
    ids = _token_list(input_ids)
    require(bool(ids), "A native trace must contain at least one token")
    blocks, cursor = [], 0
    while cursor < len(ids):
        start = cursor
        require(ids[cursor] == START, f"Unparsed token outside a native block at {cursor}")
        cursor += 1
        while cursor < len(ids) and ids[cursor] != MESSAGE:
            require(ids[cursor] not in ENDINGS and ids[cursor] != START,
                    f"Malformed native header at token {cursor}")
            cursor += 1
        header_ids = ids[start:cursor + 1] if cursor < len(ids) else ids[start:cursor]
        classified = _header_role(ids[start + 1:cursor], tokenizer)
        if cursor == len(ids):
            require(allow_open_header and classified["header_text"] == "assistant",
                    "Only a trailing open '<|start|>assistant' header is allowed")
            blocks.append(dict(block_index=len(blocks), start=start, header_end=cursor,
                               body_start=cursor, body_end=cursor, end=cursor,
                               header_ids=header_ids, body_ids=[], ending_ids=[],
                               complete=False, terminated=False,
                               observed_terminal_body=False, **classified))
            break
        body_start = cursor + 1
        cursor = body_start
        while cursor < len(ids) and ids[cursor] not in ENDINGS:
            require(ids[cursor] not in (START, MESSAGE),
                    f"Nested structural marker in native body at token {cursor}")
            cursor += 1
        terminated = cursor < len(ids)
        require(terminated or (allow_terminal_body and classified["role"] == "assistant"
                               and not classified["structural_tool_call"]),
                f"Unterminated native body starting at token {body_start}")
        if terminated:
            require(ids[cursor] != CALL or classified["structural_tool_call"],
                    "Only a structural assistant tool call may end in <|call|>")
            require(not classified["structural_tool_call"] or ids[cursor] == CALL,
                    "A structural assistant tool call must end in <|call|>")
            require(ids[cursor] != RETURN or classified["role"] == "assistant",
                    "Only a normal assistant block may end in <|return|>")
        body_end, end = cursor, cursor + int(terminated)
        blocks.append(dict(block_index=len(blocks), start=start, header_end=body_start,
                           body_start=body_start, body_end=body_end, end=end,
                           header_ids=header_ids, body_ids=ids[body_start:body_end],
                           ending_ids=ids[body_end:end], complete=True,
                           terminated=terminated, observed_terminal_body=not terminated, **classified))
        cursor = end
    require([token for block in blocks
             for token in block["header_ids"] + block["body_ids"] + block["ending_ids"]] == ids,
            "Native parser failed token conservation")
    return blocks


def first_role_targets(blocks: Sequence[Mapping[str, Any]], *,
                       require_all_roles: bool = True) -> dict[str, Mapping[str, Any]]:
    """Select exactly the first complete non-tool-call block for each role."""
    first = {}
    for block in blocks:
        role = block["role"]
        if block["complete"] and not block["structural_tool_call"] and role in ROLES and role not in first:
            require(bool(block["body_ids"]), f"The first {role} body is empty; P(role) is undefined")
            first[role] = block
    if require_all_roles:
        require(set(first) == set(ROLES), f"Native trace lacks role targets: {sorted(set(ROLES) - set(first))}")
    require(bool(first), "Native trace has no normal role targets")
    return {role: first[role] for role in ROLES if role in first}


def _positions(block: Mapping[str, Any]) -> list[int]:
    return list(range(block["body_start"], block["body_end"]))


def _input_prefix(trace_id: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "__", trace_id).strip("_.-") or "trace"
    return label + "-" + hashlib.sha256(trace_id.encode()).hexdigest()[:10]


def build_interventions(trace: Mapping[str, Any], tokenizer: Any, *,
                        require_all_roles: bool = True) -> list[dict[str, Any]]:
    """Build baseline, independent strip/swap/move variants from one trace.

    With all six role targets this returns 43 rows: one original baseline,
    six removals, 30 directed wrapper swaps, and six farthest-end moves.
    Every intervention starts from the original trace, never another variant.
    A swap copies the exact first destination-role opening and closing wrapper
    from this same trace. A move transports the original full native block to
    the beginning or end of complete blocks, before an optional open header;
    it maximizes absolute body-start displacement, breaking ties to the start.

    ``target_positions`` selects rows to capture. Baseline selects the sorted
    union of first-role bodies, with ``role_target_positions`` giving each mask.
    Other variants select only the manipulated source body. The aligned list
    ``target_original_positions`` identifies those same original token rows.
    ``token_source_positions`` proves token provenance for the entire variant;
    swapped wrapper tokens map to the destination block's original positions.
    """
    require(isinstance(trace.get("trace_id"), str) and bool(trace["trace_id"]), "Trace needs a nonempty trace_id")
    trace_id = trace["trace_id"]
    ids = _token_list(trace["input_ids"])
    blocks = parse_native_blocks(ids, tokenizer, allow_terminal_body=trace.get("allow_terminal_body", False))
    targets = first_role_targets(blocks, require_all_roles=require_all_roles)
    complete_end = max(block["end"] for block in blocks if block["complete"])
    origins = list(range(len(ids)))
    original_by_role = {role: _positions(block) for role, block in targets.items()}
    union = sorted(position for values in original_by_role.values() for position in values)
    prefix = _input_prefix(trace_id)
    rows = [dict(input_id=prefix + "--baseline", trace_id=trace_id, operation="baseline",
                 source_role=None, destination_role=None, block_index=None,
                 input_ids=ids.copy(), target_positions=union.copy(),
                 target_original_positions=union.copy(), token_source_positions=origins.copy(),
                 original_body_start=None, new_body_start=None,
                 role_target_positions=original_by_role,
                 original_role_target_positions=original_by_role,
                 target_token_count=len(union), original_input_tokens=len(ids), input_tokens=len(ids),
                 target_roles=list(targets), complete_blocks_end=complete_end,
                 observed_terminal_body=any(block["observed_terminal_body"] for block in blocks),
                 trace_has_observed_terminal_body=any(block["observed_terminal_body"] for block in blocks),
                 trailing_open_header_ids=ids[complete_end:])]

    def add(operation: str, role: str, new_ids: list[int], token_origins: list[int],
            body_start: int, *, destination_role: str | None = None, **extra: Any) -> None:
        block = targets[role]
        body_ids = block["body_ids"]
        positions = list(range(body_start, body_start + len(body_ids)))
        original = original_by_role[role]
        require([new_ids[position] for position in positions] == body_ids, "Intervention changed target body IDs")
        require([token_origins[position] for position in positions] == original,
                "Intervention changed target body token ownership")
        require(len(new_ids) == len(token_origins) and
                new_ids == [ids[position] for position in token_origins], "Variant token provenance differs")
        suffix = operation + "-" + role + ("-to-" + destination_role if destination_role else "")
        rows.append(dict(input_id=prefix + "--" + suffix, trace_id=trace_id, operation=operation,
                         source_role=role, destination_role=destination_role,
                         block_index=block["block_index"], input_ids=new_ids,
                         token_source_positions=token_origins, target_positions=positions,
                         target_original_positions=original, role_target_positions={role: positions},
                         original_role_target_positions={role: original},
                         original_body_start=block["body_start"], new_body_start=body_start,
                         original_body_end=block["body_end"], new_body_end=body_start + len(body_ids),
                         body_token_ids=body_ids.copy(), target_token_count=len(body_ids),
                         original_input_tokens=len(ids), input_tokens=len(new_ids),
                         observed_terminal_body=block["observed_terminal_body"],
                         source_block_terminated=block["terminated"],
                         trace_has_observed_terminal_body=any(item["observed_terminal_body"] for item in blocks),
                         target_roles=[role], **extra))

    for role, block in targets.items():
        start, end = block["start"], block["end"]
        body_start, body_end = block["body_start"], block["body_end"]
        kept = origins[:start] + origins[body_start:body_end] + origins[end:]
        add("strip", role, [ids[position] for position in kept], kept, start,
            removed_header_ids=block["header_ids"].copy(), removed_ending_ids=block["ending_ids"].copy(),
            removed_tokens=(body_start - start) + (end - body_end))

    for role, block in targets.items():
        for destination, wrapper in targets.items():
            if destination == role:
                continue
            wrapper_positions = (origins[wrapper["start"]:wrapper["body_start"]]
                                 + origins[wrapper["body_end"]:wrapper["end"]])
            new_origins = (origins[:block["start"]]
                           + origins[wrapper["start"]:wrapper["body_start"]]
                           + original_by_role[role]
                           + origins[wrapper["body_end"]:wrapper["end"]]
                           + origins[block["end"]:])
            add("swap", role, [ids[position] for position in new_origins], new_origins,
                block["start"] + len(wrapper["header_ids"]), destination_role=destination,
                wrapper_block_index=wrapper["block_index"], wrapper_original_positions=wrapper_positions,
                applied_header_ids=wrapper["header_ids"].copy(), applied_ending_ids=wrapper["ending_ids"].copy(),
                applied_wrapper_terminated=wrapper["terminated"],
                header_length_difference=len(wrapper["header_ids"]) - len(block["header_ids"]))

    for role, block in targets.items():
        start, end = block["start"], block["end"]
        length, header_length = end - start, len(block["header_ids"])
        remaining = origins[:start] + origins[end:]
        insertions = {"start": 0, "end": complete_end - length}
        candidate_starts = {name: position + header_length for name, position in insertions.items()}
        distances = {name: abs(position - block["body_start"]) for name, position in candidate_starts.items()}
        destination = "start" if distances["start"] >= distances["end"] else "end"
        insertion = insertions[destination]
        new_origins = remaining[:insertion] + origins[start:end] + remaining[insertion:]
        require(sorted(new_origins) == origins, "Moving a block must permute every original token once")
        require([ids[position] for position in new_origins[complete_end:]] == ids[complete_end:],
                "Moving a block changed the trailing open header")
        add("move", role, [ids[position] for position in new_origins], new_origins,
            candidate_starts[destination], move_destination=destination,
            move_insertion_position=insertion,
            move_original_insertion_boundary=0 if destination == "start" else complete_end,
            move_candidate_body_starts=candidate_starts, move_candidate_distances=distances,
            move_displacement_tokens=candidate_starts[destination] - block["body_start"],
            move_distance_tokens=distances[destination], moved_block_tokens=length,
            applied_header_ids=block["header_ids"].copy(), applied_ending_ids=block["ending_ids"].copy())
    require(len(rows) == 1 + len(targets) * (len(targets) + 1), "Intervention grid is incomplete")
    require(len({row["input_id"] for row in rows}) == len(rows), "Duplicate intervention input IDs")
    return rows


# Explicit aliases for callers that use the shorter role-analysis terminology.
parse_blocks = parse_native_blocks
build_variants = build_interventions
