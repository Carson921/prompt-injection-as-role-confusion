"""Full-context tomato figures for a frozen six-role article-only probe.

Cached figures show *only* sampled article-body probabilities. Unscored article
text and structural tokens remain visible and have separate backgrounds. The
optional dense diagnostic helper performs new forwards outside formal metrics.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from synthetic_agent_classifier import FeatureBank, load_probe
from synthetic_agent_gpu import ROLES, atomic_json, digest, require

COLORS = ["#9467bd", "#8c564b", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
STRUCTURE = "#d9d9d9"
UNSCORED_ARTICLE = "#fff2d5"


def _offsets(record, tokenizer):
    if "token_offsets" in record:
        offsets = record["token_offsets"]
    else:
        require(tokenizer is not None, "The bound tokenizer is required to reconstruct exact character offsets")
        encoded = tokenizer(record["text"], add_special_tokens=False, return_offsets_mapping=True)
        require(encoded["input_ids"] == record["input_ids"], "Plot tokenizer does not reproduce the frozen input IDs")
        offsets = encoded["offset_mapping"]
    require(len(offsets) == len(record["input_ids"]), "Plot token-offset length mismatch")
    return [tuple(map(int, pair)) for pair in offsets]


def _styles(record, offsets):
    # -2: structural context; -1: article body outside the sampled scoring mask.
    token_style = np.full(len(offsets), -2, dtype=np.int8)
    char_style = np.full(len(record["text"]), -2, dtype=np.int8)
    for span in record["message_spans"]:
        if span["supervised"]:
            char_style[span["body_start"]:span["body_end"]] = -1
            for ix, (left, right) in enumerate(offsets):
                if span["body_start"] <= left < right <= span["body_end"]:
                    token_style[ix] = -1
    for target in record["targets"]:
        ix, label = int(target["token_ix"]), int(target["label"])
        require(token_style[ix] == -1, "Figure scoring mask touches structural context")
        token_style[ix] = label
        left, right = offsets[ix]
        char_style[left:right] = label
    return token_style, char_style


def _wrapped_lines(text, width=200):
    lines, current = [], []
    for ix, char in enumerate(text):
        if char == "\r":
            continue
        if char == "\n":
            lines.append(current)
            current = []
            continue
        if len(current) >= width:
            lines.append(current)
            current = []
        current.append((ix, " " if char == "\t" else char))
    if current or not lines:
        lines.append(current)
    return lines


def _runs(values):
    if len(values):
        left = 0
        for right in range(1, len(values) + 1):
            if right == len(values) or values[right] != values[left]:
                yield left, right, int(values[left])
                left = right


def _draw(record, probabilities, offsets, destination, *, title, semantic_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    probability = np.asarray(probabilities)
    require(probability.shape == (len(record["targets"]), 6) and np.isfinite(probability).all(),
            "Figure probabilities do not match the article scoring mask")
    require(np.allclose(probability.sum(1), 1, atol=2e-5), "Invalid six-role probability normalization")
    offsets = list(offsets)
    token_style, char_style = _styles(record, offsets)
    lines = _wrapped_lines(record["text"])
    text_height = max(1.2, len(lines) * .092)
    fig = plt.figure(figsize=(12, text_height + 5.4), layout="constrained")
    grid = fig.add_gridspec(8, 1, height_ratios=[text_height, .38, .65, .65, .65, .65, .65, .65])
    context = fig.add_subplot(grid[0])
    context.set_axis_off()
    context.set_title(title, loc="left", fontsize=9, pad=12)
    for row, line in enumerate(lines):
        column = 0
        while column < len(line):
            style = int(char_style[line[column][0]])
            stop = column + 1
            while stop < len(line) and int(char_style[line[stop][0]]) == style:
                stop += 1
            text = "".join(char for _, char in line[column:stop])
            color = COLORS[style] if style >= 0 else ("#707070" if style == -1 else "#a0a0a0")
            context.annotate(text, (0, 1 - (row + .5) / len(lines)),
                             xycoords="axes fraction", xytext=(column * 5.5 * .602, 0),
                             textcoords="offset points", fontsize=5.5, fontfamily="DejaVu Sans Mono",
                             color=color, va="center", ha="left", annotation_clip=False, parse_math=False)
            column = stop
    legend_ax = fig.add_subplot(grid[1])
    legend_ax.set_axis_off()
    handles = [Patch(facecolor=COLORS[i], label=role) for i, role in enumerate(ROLES)]
    handles += [Patch(facecolor=STRUCTURE, label="Unscored structure / Q / tools"),
                Patch(facecolor=UNSCORED_ARTICLE, label="Unscored article text")]
    legend_ax.legend(handles=handles, loc="center left", ncol=4, fontsize=6, frameon=False)
    positions = np.asarray([target["token_ix"] for target in record["targets"]], dtype=np.int64)
    labels = np.asarray([target["label"] for target in record["targets"]], dtype=np.int64)
    point_colors = [COLORS[label] for label in labels]
    previous = None
    for role in range(6):
        ax = fig.add_subplot(grid[role + 2], sharex=previous)
        previous = ax
        for left, right, style in _runs(token_style):
            if style < 0:
                ax.axvspan(left - .5, right - .5, color=STRUCTURE if style == -2 else UNSCORED_ARTICLE,
                           alpha=.75, lw=0, zorder=0)
        # No interpolation across unsampled body positions: vertical strokes are
        # exactly the cached scoring mask, not an invented dense probability trace.
        ax.vlines(positions, 0, probability[:, role], colors=point_colors, linewidth=.65, zorder=2)
        ax.set_ylim(0, 1)
        ax.set_xlim(-.5, len(record["input_ids"]) - .5)
        ax.set_yticks([0, 1])
        ax.set_ylabel(ROLES[role], rotation=0, ha="right", va="center", fontsize=7, labelpad=12)
        ax.tick_params(labelsize=6)
        if role != 5:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Token position in the complete actual input", fontsize=7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.supxlabel(semantic_note, fontsize=6, x=.06, ha="left")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def _coverage(records):
    """Select examples from structure and IDs alone, before reading predictions."""
    selected, chains = {}, defaultdict(dict)

    def keep(category, item):
        record = item[0]
        key = (str(record.get("cluster_id", record["source_id"])), record["id"])
        if category not in selected or key < selected[category][0]:
            selected[category] = (key, item)

    for record, start in records:
        if not record["targets"]:
            continue
        item = (record, start)
        segments = record.get("round_segments", [])
        stage, policy = record.get("snapshot_stage"), record.get("history_policy")
        if record["profile"] == "single":
            keep("single_" + record["template_id"], item)
        if record["profile"] == "one_round" and stage == "output":
            keep("one_round_" + record["template_id"], item)
        if record["profile"] == "multi_round" and stage == "output":
            if len(set(segments)) < len(segments):
                keep("multi_repeated_segments", item)
            if segments[-1:] == ["UCQ"]:
                keep("multi_terminal_UCQ", item)
            if len(segments) in (5, 6):
                keep(f"multi_K{len(segments)}", item)
        if policy in ("retain", "drop_completed_round_cot"):
            chains[record["call_chain_id"]][policy] = item
        if stage in ("pre_tool_call", "tool_reply") or policy == "drop_prior_call_cot":
            chains[record["call_chain_id"]][stage] = item
    paired = [(chain, versions) for chain, versions in chains.items()
              if {"retain", "drop_completed_round_cot"} <= set(versions)]
    if paired:
        _, versions = min(paired)
        for key in ["retain", "drop_completed_round_cot"]:
            keep("history_cot_" + key, versions[key])
    staged = [(chain, versions) for chain, versions in chains.items()
              if {"pre_tool_call", "tool_reply", "output"} <= set(versions)]
    if staged:
        _, versions = min(staged)
        for key in ["pre_tool_call", "tool_reply", "output"]:
            keep("call_chain_" + key, versions[key])
    return {category: item for category, (_, item) in sorted(selected.items())}


def plot_run(run_dir, data_dir, phase="formal", split="test", feature_dir=None):
    """Render cached mask-only results using the already frozen new classifier."""
    from transformers import AutoTokenizer
    from synthetic_agent_data import iter_records
    root = Path(run_dir)
    frozen = json.loads((root / "freeze.json").read_text())
    require(frozen["probe_sha256"] == digest(root / "best.npz"), "Frozen probe changed before plotting")
    require(frozen["selection_sha256"] == digest(root / "selection.json"), "Frozen DEV selection changed before plotting")
    probe = load_probe(root / "best.npz")
    bank = FeatureBank(feature_dir or root / "features" / phase, split, expected_binding=probe.metadata)
    input_path = Path(data_dir) / phase / f"inputs-{split}.jsonl"
    require(bank.manifest["split_sources"][split] == digest(input_path), "Plot inputs do not match feature rows")
    cursor = 0
    records = []
    for record in iter_records(data_dir, phase, split):
        rows = len(record["targets"])
        require(np.array_equal(bank.y[cursor:cursor + rows], [row["label"] for row in record["targets"]]),
                "Feature labels are not aligned with plot body positions")
        require(np.all(bank.source[cursor:cursor + rows].astype(str) == str(record.get("cluster_id", record["source_id"]))),
                "Feature source-cluster order does not match plot inputs")
        records.append((record, cursor))
        cursor += rows
    require(cursor == len(bank), "Feature/input row count mismatch")
    examples = _coverage(records)
    require(examples, "No structurally selected examples are available")
    backend = json.loads((root / "model_backend.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(backend["load_path"], local_files_only=True)
    destination = root / "plots" / split
    destination.mkdir(parents=True, exist_ok=True)
    index = {"phase": phase, "split": split, "probe_sha256": frozen["probe_sha256"],
             "scoring": "sampled article-body mask only; no interpolation across unsampled tokens",
             "selection": "fixed structural categories; lexicographically first cluster/input ID, independent of predictions",
             "examples": {}}
    for category, (record, start) in examples.items():
        n = len(record["targets"])
        p = probe.predict_proba(bank.read(np.arange(start, start + n)))
        offsets = _offsets(record, tokenizer)
        stem = category + "-" + hashlib.sha256(record["id"].encode()).hexdigest()[:10]
        note = "Native Harmony; colored strokes = scored article tokens. Gray = structure; pale yellow = unscored article text."
        title = (f"{category} | {record['template_id']} | {record['snapshot_stage']} | "
                 f"{record['history_policy']} | {len(record['input_ids'])} input tokens, {n} scored")
        _draw(record, p, offsets, destination / stem, title=title, semantic_note=note)
        # Preserve a compact, portable mapping after the remote tokenizer disappears.
        atomic_json(destination / (stem + ".json"), {"record_id": record["id"], "category": category,
                    "input_tokens": len(record["input_ids"]), "scored_positions": [t["token_ix"] for t in record["targets"]],
                    "original_roles": [t["role"] for t in record["targets"]], "probabilities": p.tolist(),
                    "token_offsets": offsets, "probe_sha256": frozen["probe_sha256"]})
        index["examples"][category] = {"record_id": record["id"], "png": stem + ".png", "pdf": stem + ".pdf",
                                       "scored_tokens": n, "input_tokens": len(record["input_ids"])}
    expected = ["single_" + symbol for symbol in "SDUCAT"] + [
        "one_round_SD" + segment for segment in ["UA", "UCA", "UCQ", "UQTCA", "UQTA"]] + [
        "multi_repeated_segments", "multi_terminal_UCQ", "history_cot_retain",
        "history_cot_drop_completed_round_cot", "call_chain_pre_tool_call", "call_chain_tool_reply", "call_chain_output"]
    if phase == "formal" and split == "test":
        expected += ["multi_K5", "multi_K6"]
    index["missing_coverage"] = [category for category in expected if category not in examples]
    index["call_chain_note"] = "Stages contain different target bodies; their overall scores do not estimate the causal effect of deleting CoT."
    atomic_json(destination / "index.json", index)
    return index


def diagnostic_record(record, tokenizer, variant="native"):
    """Build a dense, unscored diagnostic with original semantic-role labels."""
    require(variant in ["native", "all_user", "no_tags"], "Unknown diagnostic variant")
    parts, spans, cursor = [], [], 0
    for original in record["message_spans"]:
        fragment = record["text"][original["body_start"]:original["message_end"]]
        match = re.search(r"<\|(?:end|call|return)\|>$", fragment)
        require(match is not None, "Diagnostic message has no recognized ending")
        body = fragment[:match.start()]
        if variant == "native":
            prefix = record["text"][original["header_start"]:original["body_start"]]
            suffix = match.group()
        elif variant == "all_user":
            prefix, suffix = "<|start|>user<|message|>", "<|end|>"
        else:
            prefix, suffix = "", "\n"
        span = dict(original, header_start=cursor, body_start=cursor + len(prefix),
                    body_end=cursor + len(prefix) + original["body_end"] - original["body_start"],
                    message_end=cursor + len(prefix) + len(body) + len(suffix))
        spans.append(span)
        piece = prefix + body + suffix
        parts.append(piece)
        cursor += len(piece)
    text = "".join(parts)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if variant == "native":
        require(text == record["text"] and encoded["input_ids"] == record["input_ids"],
                "Native dense diagnostic does not reproduce the exact frozen input")
    require(0 < len(encoded["input_ids"]) <= 4096, "Diagnostic exceeds the tested input limit")
    targets = []
    for ix, (left, right) in enumerate(encoded["offset_mapping"]):
        if right <= left:
            continue
        for span in spans:
            if span["supervised"] and span["body_start"] <= left < right <= span["body_end"]:
                targets.append({"token_ix": ix, "label": ROLES.index(span["role"]), "role": span["role"]})
                break
    return {**record, "id": record["id"] + "/diagnostic/" + variant, "text": text,
            "input_ids": encoded["input_ids"], "token_offsets": encoded["offset_mapping"],
            "message_spans": spans, "targets": targets, "diagnostic_variant": variant,
            "formal_scoring": False}


def dense_diagnostic(record, model, probe, tokenizer, output_dir, variant="native"):
    """Recompute dense selected-head features; never append to a scored cache."""
    import torch
    from synthetic_agent_gpu import HeadCapture, _forward
    if isinstance(probe, (str, Path)):
        probe = load_probe(probe)
    require(getattr(model, "_synthetic_binding", None) == probe.metadata["model_binding"],
            "Diagnostic model differs from the frozen probe backbone")
    dense = diagnostic_record(record, tokenizer, variant)
    capture = HeadCapture([target["token_ix"] for target in dense["targets"]], selected=probe.metadata["heads"])
    with torch.inference_mode(), capture.install(model):
        _forward(model, dense["input_ids"])
    probabilities = probe.predict_proba(capture.matrix())
    output = Path(output_dir) / (variant + "-" + hashlib.sha256(record["id"].encode()).hexdigest()[:10])
    note = ("Dense deployment diagnostic only. Scores refer to original semantic roles; "
            "changed-format scores are not formal role accuracy.")
    _draw(dense, probabilities, dense["token_offsets"], output,
          title=f"Dense diagnostic: {variant} | {record['template_id']} | frozen new six-role probe", semantic_note=note)
    metadata = {"record_id": record["id"], "variant": variant, "formal_scoring": False,
                "note": note, "input_tokens": len(dense["input_ids"]), "body_tokens": len(dense["targets"]),
                "probabilities": probabilities.tolist(), "positions": [target["token_ix"] for target in dense["targets"]]}
    atomic_json(output.with_suffix(".json"), metadata)
    return metadata
