"""CPU figures for the one authorized public gardening diagnostic.

Consumes already prepared inputs and already saved six-role probabilities; this
module never loads a backbone/checkpoint or changes any classifier selection.
Explicit body token positions, including the token-109 emoji continuation, are
authoritative even when character offsets have zero width or overlap.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
FORMATS = ("proper_tags", "everything_in_user_tags", "basic_no_format")
MODELS = ("pilot", "first_formal")
SUPPORT = {"system": 57, "developer": 0, "user": 97, "cot": 179, "assistant": 583, "tool": 0}
COLORS = ("#9467bd", "#8c564b", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
FORMAT_TITLES = {"proper_tags": "Native Harmony: formal body-role accuracy",
                 "everything_in_user_tags": "All in user: original-semantic alignment only",
                 "basic_no_format": "No tags: original-semantic alignment only"}


def require(value, message):
    if not value:
        raise ValueError(message)


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def _json_value(value):
    if isinstance(value, (str, Path)):
        path = Path(value)
        return json.loads(path.read_text()), path.resolve().parent
    return value, None


def _prepared(value):
    value, _ = _json_value(value)
    if isinstance(value, dict):
        value = value.get("samples", list(value.values()))
    require(isinstance(value, list), "Prepared gardening inputs must be a list or samples mapping")
    samples = {sample["prompt_key"]: sample for sample in value}
    require(len(value) == len(samples) == 3 and set(samples) == set(FORMATS), "Exactly the three archived gardening formats are required")
    reference = None
    for key in FORMATS:
        sample = samples[key]
        text, ids, offsets, rows = sample["text"], sample["input_ids"], sample["token_offsets"], sample["content"]
        require(isinstance(text, str) and 0 < len(ids) <= 4096 and len(offsets) == len(ids), "Invalid complete input or offsets")
        if "prompt_sha256" in sample:
            require(sample["prompt_sha256"] == _sha_bytes(text.encode()), "Complete original prompt SHA256 changed")
        require(len(rows) == 916, "The complete 916 body tokens, including final emoji continuation, are required")
        require(all(isinstance(pair, (list, tuple)) and len(pair) == 2 and
                    all(isinstance(x, int) and not isinstance(x, bool) for x in pair) and
                    0 <= pair[0] <= pair[1] <= len(text) for pair in offsets), "Invalid character offsets")
        positions = [row["token_position"] for row in rows]
        require(all(type(pos) is int and 0 <= pos < len(ids) for pos in positions) and
                positions == sorted(set(positions)), "Body positions are duplicated, unordered, or outside the input")
        counts = {role: 0 for role in ROLES}
        identity = []
        for row in rows:
            role = row["semantic_role"]
            require(role in ROLES and row["label"] == ROLES.index(role) and ids[row["token_position"]] == row["token_id"],
                    "Body token ID or original six-role label differs from the prepared input")
            expected_formal = role if key == "proper_tags" else ("user" if key == "everything_in_user_tags" else None)
            require(row["formal_role"] == expected_formal, "Formal and original semantic role labels were conflated")
            counts[role] += 1
            identity.append((row["message_ix"], role))
        require(counts == SUPPORT, "Gardening body support must be S57/D0/U97/C179/A583/T0")
        require(rows[-1]["token_id"] == 109 and rows[-1]["semantic_role"] == "assistant" and
                rows[-1].get("in_old_915_comparison") is False, "Final emoji continuation token 109 was not restored explicitly")
        require(reference is None or reference == identity, "Original message/role identity differs across formats")
        reference = identity
    return samples


def _metric_counts(sample, probabilities):
    labels = np.array([row["label"] for row in sample["content"]], dtype=np.int64)
    predicted = probabilities.argmax(1)
    matrix = np.bincount(labels * 6 + predicted, minlength=36).reshape(6, 6)
    result = {}
    for i, role in enumerate(ROLES):
        support, detected, correct = int(matrix[i].sum()), int(matrix[:, i].sum()), int(matrix[i, i])
        result[role] = {"support": support, "predicted": detected, "true_positive": correct,
                        "false_positive": detected - correct, "false_negative": support - correct,
                        "precision": correct / detected if detected else None,
                        "recall": correct / support if support else None,
                        "f1": 2 * correct / (support + detected) if support else None}
    return result, matrix, float(np.mean(labels == predicted))


def _loaded_results(value, samples, out_dir):
    results, source_dir = _json_value(value)
    require(isinstance(results, dict) and tuple(results["roles"]) == ROLES, "Only native six-role gardening results are supported")
    require(set(results["models"]) == set(MODELS) and set(results["formats"]) == set(FORMATS), "Both fixed checkpoints and all three formats are required")
    base = Path(results.get("artifact_dir", source_dir or out_dir)).resolve()
    loaded = {}
    for model in MODELS:
        meta = results["models"][model]
        heads = meta["heads"]
        require(len(heads) == len(set(heads)) == 16 and all(type(h) is int and 0 <= h < 2304 for h in heads),
                "Both new classifiers require their own ordered 16 heads; old ten-head/five-role readouts are incompatible")
        require(isinstance(meta["checkpoint_sha256"], str) and len(meta["checkpoint_sha256"]) == 64 and
                all(char in "0123456789abcdef" for char in meta["checkpoint_sha256"]), "Checkpoint SHA256 is required")
        if model == "first_formal":
            require(meta["candidate_id"] == "linear-seed17", "The first formal comparison is the predeclared fixed linear-seed17 candidate")
    require(results["models"]["pilot"]["heads"] == results["models"]["first_formal"]["heads"],
            "The shared-forward comparison requires identical ordered feature heads")
    for key in FORMATS:
        section = results["formats"][key]
        native = key == "proper_tags"
        require(section["metric_kind"] == ("formal_role" if native else "original_semantic_alignment"),
                "Changed-format alignment must not be labelled formal role accuracy")
        loaded[key] = {}
        for model in MODELS:
            entry = section["models"][model]
            if "probabilities" in entry:
                require(results.get("engineering_only") is True, "Inline probabilities are reserved for explicitly marked CPU engineering fixtures")
                probabilities = np.asarray(entry["probabilities"], dtype=np.float32)
                provenance = {"inline_fixture": True}
            else:
                path = Path(entry["probabilities_file"])
                path = path if path.is_absolute() else base / path
                require(path.name == f"{key}-{model}-probabilities.npy", "Probability filename does not identify the fixed format/model")
                probabilities = np.load(path, mmap_mode="r", allow_pickle=False)
                provenance = {"probabilities_file": str(path.resolve()), "sha256": _sha_file(path)}
                if "probabilities_sha256" in entry:
                    require(entry["probabilities_sha256"] == provenance["sha256"], "Saved gardening probability SHA256 changed")
            require(probabilities.shape == (916, 6) and np.isfinite(probabilities).all() and
                    np.all(probabilities >= 0) and np.all(probabilities <= 1) and
                    np.allclose(probabilities.sum(1), 1, atol=3e-5), "Invalid or incomplete six-role probability rows")
            expected, confusion, accuracy = _metric_counts(samples[key], probabilities)
            metrics = entry["metrics"]
            require(metrics.get("metric_kind") == section["metric_kind"], "Per-model metric kind differs from its declared format")
            require(metrics["rows"] == 916 and metrics["confusion_matrix"] == confusion.tolist(), "Reported gardening metrics disagree with the saved probabilities")
            for role, actual in expected.items():
                for field, target in actual.items():
                    observed = metrics["per_role"][role][field]
                    require(observed is None if target is None else observed is not None and np.isclose(observed, target, atol=3e-6),
                            f"Reported {role} {field} disagrees with the body-only mask")
            score_key = "accuracy" if native else "semantic_alignment_accuracy"
            require(np.isclose(metrics[score_key], accuracy, atol=3e-6), "Overall score disagrees with its declared metric kind")
            require(np.isclose(metrics["macro_f1_present"], np.mean([row["f1"] for row in expected.values() if row["support"]]), atol=3e-6),
                    "Present-class macro-F1 disagrees with the original body labels")
            if not native:
                require(metrics.get("accuracy") is None, "Altered-format original-role alignment is not formal accuracy")
            loaded[key][model] = {"probabilities": probabilities, "metrics": metrics, "provenance": provenance}
    return results, loaded


def _styles(sample):
    token_role = np.full(len(sample["input_ids"]), -1, dtype=np.int8)
    char_role = np.full(len(sample["text"]), -1, dtype=np.int8)
    for row in sample["content"]:
        token_ix, label = row["token_position"], row["label"]
        token_role[token_ix] = label  # Deliberately independent of character width.
        left, right = sample["token_offsets"][token_ix]
        if right > left:
            prior = char_role[left:right]
            require(np.all((prior == -1) | (prior == label)), "Overlapping character offsets assign conflicting original roles")
            char_role[left:right] = label
    return token_role, char_role


def _lines(text, width=180):
    lines, line = [], []
    for i, char in enumerate(text):
        if char == "\n":
            lines.append(line)
            line = []
        else:
            if len(line) >= width:
                lines.append(line)
                line = []
            line.append((i, " " if char == "\t" else char))
    lines.append(line)
    return lines


def _runs(values):
    if len(values):
        left = 0
        for right in range(1, len(values) + 1):
            if right == len(values) or values[right] != values[left]:
                yield left, right, int(values[left])
                left = right


def _figure(sample, entries, model_metadata, *, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    order = [model for model in MODELS if model in entries]
    token_role, char_role = _styles(sample)
    lines = _lines(sample["text"])
    context_height = max(1.3, len(lines) * .105)
    fig = plt.figure(figsize=(12.5 * len(order), context_height + 6.3), layout="constrained")
    grid = fig.add_gridspec(8, len(order), height_ratios=[context_height, .42, .78, .78, .78, .78, .78, .78])
    positions = np.array([row["token_position"] for row in sample["content"]])
    labels = np.array([row["label"] for row in sample["content"]])
    colors = [COLORS[label] for label in labels]
    boundaries = [sample["content"][i]["token_position"] for i in range(1, 916)
                  if sample["content"][i]["message_ix"] != sample["content"][i - 1]["message_ix"]]
    axes = {}
    for column, model in enumerate(order):
        entry = entries[model]
        native = sample["prompt_key"] == "proper_tags"
        score = entry["metrics"]["accuracy" if native else "semantic_alignment_accuracy"]
        score_name = "body accuracy" if native else "semantic alignment"
        context = fig.add_subplot(grid[0, column])
        context.set_axis_off()
        context.set_title(f"{model_metadata[model]['label']} | {score_name}={100 * score:.2f}%", loc="left", fontsize=10, pad=12)
        for line_ix, line in enumerate(lines):
            start = 0
            while start < len(line):
                role = int(char_role[line[start][0]])
                stop = start + 1
                while stop < len(line) and int(char_role[line[stop][0]]) == role:
                    stop += 1
                context.annotate("".join(char for _, char in line[start:stop]), (0, 1 - (line_ix + .5) / len(lines)),
                    xycoords="axes fraction", xytext=(start * 6.6 * .602, 0), textcoords="offset points",
                    fontsize=6.6, fontfamily="DejaVu Sans Mono", color=COLORS[role] if role >= 0 else "#888888",
                    ha="left", va="center", annotation_clip=False, parse_math=False)
                start = stop
        legend = fig.add_subplot(grid[1, column])
        legend.set_axis_off()
        handles = [Patch(facecolor=COLORS[i], label=role) for i, role in enumerate(ROLES)]
        handles.append(Patch(facecolor="#dedede", label="Unscored structure"))
        legend.legend(handles=handles, ncol=4, loc="center left", fontsize=6.5, frameon=False)
        previous = None
        axes[model] = []
        for role_ix, role in enumerate(ROLES):
            axis = fig.add_subplot(grid[role_ix + 2, column], sharex=previous)
            previous = axis
            axes[model].append(axis)
            for left, right, style in _runs(token_role):
                if style < 0:
                    axis.axvspan(left - .5, right - .5, color="#dedede", alpha=.7, lw=0)
            axis.vlines(positions, 0, entry["probabilities"][:, role_ix], colors=colors, linewidth=.68, rasterized=True)
            for boundary in boundaries:
                axis.axvline(boundary - .5, color="#a0a0a0", linewidth=.5, alpha=.7)
            row_metrics = entry["metrics"]["per_role"][role]
            recall = "N/A" if row_metrics["recall"] is None else f"{100 * row_metrics['recall']:.1f}%"
            axis.set_ylabel(f"{role}\nR {recall} | FP {row_metrics['false_positive']}", rotation=0, ha="right", va="center", fontsize=7, labelpad=7)
            axis.set_ylim(0, 1)
            axis.set_xlim(-.5, len(sample["input_ids"]) - .5)
            axis.set_yticks([0, .5, 1])
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.tick_params(labelsize=6, labelbottom=role_ix == 5)
            if role_ix == 5:
                axis.set_xlabel("Token position in the complete actual input; all 916 body tokens", fontsize=7)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
    fig.suptitle(title, fontsize=12)
    note = ("Public paper example; diagnosis only, no selection. Gray = unscored structure; colors = original message roles. "
            "D/T have no true body: recall N/A; false positives shown. Seedling U+1F331: all 916 body tokens, including continuation 109. "
            "No thresholds or smoothing. Each wrapper has its own actual token coordinates; six punctuation/LF fusions occur in altered formats, "
            "so IDs differ across formats. Only same-format model pairs share identical input IDs.")
    if sample["prompt_key"] != "proper_tags":
        note += " Changed-format scores measure original-semantic alignment, not formal role accuracy."
    fig.supxlabel(textwrap.fill(note, width=150 * len(order)), fontsize=7)
    return fig, axes


def plot_gardening(prepared, results, out_dir):
    """Produce one native two-model comparison and six individual-format pairs."""
    samples = _prepared(prepared)
    result, values = _loaded_results(results, samples, out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_metadata = result["models"]
    index = {"format_version": "synthetic-gardening-plots-v1", "engineering_only": bool(result.get("engineering_only", False)),
             "roles": list(ROLES), "models": {model: {key: model_metadata[model][key]
                        for key in ("label", "candidate_id", "heads", "checkpoint_sha256")} for model in MODELS},
             "scope": "One previously used public paper example. Pilot DEV-selected probe vs the first formal fixed linear-seed17 candidate; the latter is not the final formal DEV selection.",
             "cross_format_alignment": "Original message/semantic-role order and 916 body-row counts agree; actual token IDs and positions are wrapper-specific, including six punctuation/LF fusions in altered formats. Identical input IDs apply only within the same-format model comparison.",
             "selection_permitted": False, "figures": {}, "formats": {}}
    for key in FORMATS:
        sample = samples[key]
        last = sample["content"][-1]
        index["formats"][key] = {"input_tokens": len(sample["input_ids"]), "body_tokens": 916,
            "text_sha256": _sha_bytes(sample["text"].encode()), "body_positions": [row["token_position"] for row in sample["content"]],
            "restored_emoji_continuation": {"token_id": 109, "token_position": last["token_position"],
                                             "character_offset": sample["token_offsets"][last["token_position"]],
                                             "unicode_scalar": "U+1F331", "retained": True},
            "metric_kind": "formal_role" if key == "proper_tags" else "original_semantic_alignment",
            "models": {model: values[key][model]["provenance"] for model in MODELS}}
    jobs = [("gardening-native-comparison", "proper_tags", MODELS)] + [
        (f"gardening-{model}-{key}", key, (model,)) for model in MODELS for key in FORMATS]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for stem, key, models in jobs:
        title = "Gardening | " + FORMAT_TITLES[key]
        if result.get("engineering_only"):
            title = "ENGINEERING FIXTURE | " + title
        if len(models) == 2:
            title += " | pilot vs first formal fixed candidate"
        figure, _ = _figure(samples[key], {model: values[key][model] for model in models}, model_metadata, title=title)
        try:
            files = {}
            for suffix in ("png", "pdf"):
                path = out_dir / f"{stem}.{suffix}"
                figure.savefig(path, dpi=180)
                files[suffix] = {"path": path.name, "sha256": _sha_file(path)}
            index["figures"][stem] = {"format": key, "models": list(models), "complete_context": True,
                                       "body_tokens_per_model": 916, "probability_rows": 6, "files": files}
        finally:
            plt.close(figure)
    temporary = out_dir / "index.json.tmp"
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(out_dir / "index.json")
    return index
