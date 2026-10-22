"""Render the archived five-class tomato predictions from single-message training.

This script reads predictions already saved by ``tomato_feature_probes.py``.
It never loads a model or checkpoint and never performs inference. All 915
content tokens in each format are shown, including the system message.

Run from any working directory::

    python experiments/role-analysis/plot_single_message_tomato.py

Only matplotlib and the Python standard library are required. The two altered
formats retain colors for the ORIGINAL message functions. Their agreement with
those functions is descriptive semantic alignment, not formal role accuracy.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "outputs/local-gptoss120b/feature-search"
DEFAULT_PROJECTIONS = ARCHIVE / "tomato/projections.csv.gz"
DEFAULT_OUT = ARCHIVE / "tomato/plots/single-message-only"
FEATURE = "mid-16+20+24"
MODEL_PATH = ARCHIVE / f"models/final-document-sucat-{FEATURE}-0.005.pkl"
MODEL_SHA256 = "ddba6ed427f4381a86e4848e6856516abf5aca56ff5cd5bc73adfd281904b51e"
ROLES = ("system", "user", "cot", "assistant", "tool")
FORMATS = ("proper_tags", "everything_in_user_tags", "basic_no_format")
ROLE_LABELS = ("Systemness", "Userness", "CoTness", "Assistantness", "Toolness")
COLORS = {
    "system": "#90a1b9",
    "user": "#00a6f4",
    "cot": "#fd9a00",
    "assistant": "#00b980",
    "tool": "#9b5de5",
}
EXPECTED_SUPPORT = {"system": 57, "user": 97, "cot": 179, "assistant": 582, "tool": 0}
EXPECTED_NATIVE_CORRECT = {"system": 19, "user": 89, "cot": 0, "assistant": 582, "tool": 0}
STEM = "five-class-single-message-only"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Keep archive assertions active even when Python is run with -O."""
    if not condition:
        raise AssertionError(message)


def read_predictions(path: Path) -> dict[str, list[dict]]:
    """Validate the exact selected document-split SUCAT archive before plotting."""
    tokens: dict[tuple[str, int], dict] = {}
    selected_rows = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {
            "sample_ix", "target_role", "prob", "prompt_ix", "prompt_key",
            "base_message_ix", "base_message_type", "token_in_prompt_ix",
            "token", "role_space", "model_variant", "feature", "protocol",
        }
        require(needed.issubset(reader.fieldnames or ()), "Missing required projection columns")
        for row in reader:
            if (row["model_variant"], row["role_space"], row["protocol"]) != (
                "selected", "sucat", "document"
            ):
                continue
            require(row["prompt_key"] in FORMATS, "Unexpected format in selected SUCAT predictions")
            require(row["feature"] == FEATURE, "Projection feature does not match the archived selected model")
            role = row["target_role"]
            require(role in ROLES, f"Unexpected output role: {role}")
            original_role = row["base_message_type"]
            require(original_role in ROLES, f"Unexpected original function: {original_role}")
            probability = float(row["prob"])
            require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid probability")
            key = (row["prompt_key"], int(row["sample_ix"]))
            metadata = {
                "prompt_ix": int(row["prompt_ix"]),
                "base_message_ix": int(float(row["base_message_ix"])),
                "original_function": original_role,
                "token_in_prompt_ix": int(row["token_in_prompt_ix"]),
                "token": row["token"],
            }
            if key not in tokens:
                tokens[key] = {**metadata, "sample_ix": key[1], "probabilities": {}}
            record = tokens[key]
            require(all(record[field] == value for field, value in metadata.items()), "Inconsistent token metadata")
            require(role not in record["probabilities"], f"Duplicate token/output-role row: {key}, {role}")
            record["probabilities"][role] = probability
            selected_rows += 1

    require(selected_rows == 3 * 915 * 5, f"Expected 13,725 selected rows; found {selected_rows}")
    formats: dict[str, list[dict]] = {}
    for prompt_key in FORMATS:
        records = [record for (name, _), record in tokens.items() if name == prompt_key]
        records.sort(key=lambda record: record["token_in_prompt_ix"])
        require(len(records) == 915, f"{prompt_key}: expected all 915 content tokens")
        positions = [record["token_in_prompt_ix"] for record in records]
        require(len(set(positions)) == len(records), f"{prompt_key}: repeated prompt position")
        require(len({record["prompt_ix"] for record in records}) == 1, "Multiple prompts in one format")
        support = {role: 0 for role in ROLES}
        for record in records:
            probabilities = record["probabilities"]
            require(set(probabilities) == set(ROLES), "A token does not have all five output probabilities")
            require(math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=2e-6), "Probabilities do not sum to one")
            record["prediction"] = max(ROLES, key=probabilities.__getitem__)
            support[record["original_function"]] += 1
        require(support == EXPECTED_SUPPORT, f"{prompt_key}: unexpected original-function token counts")
        formats[prompt_key] = records

    validate_message_identity(formats)
    native_correct = count_matches(formats["proper_tags"])
    require(native_correct == EXPECTED_NATIVE_CORRECT, f"Archived native counts changed: {native_correct}")
    return formats


def validate_message_identity(formats: dict[str, list[dict]]) -> dict:
    """Allow only the six observed join-LFs fused into message-final tokens.

    Removing native delimiters makes the tokenizer combine each separating LF
    with the preceding punctuation token. We retain those raw archived tokens
    in the figure and scores. Identity checks permit this exact boundary
    change, rather than stripping whitespace throughout the message text.
    """
    reference = formats["proper_tags"]
    expected_changed_indices = [56, 101, 184, 514, 566, 662]
    details = {}
    for name in FORMATS[1:]:
        changed = []
        for index, (native, altered) in enumerate(zip(reference, formats[name])):
            require((native["base_message_ix"], native["original_function"]) ==
                    (altered["base_message_ix"], altered["original_function"]),
                    f"{name}: original message identity or function changed")
            if native["token"] == altered["token"]:
                continue
            require(index + 1 < len(reference) and
                    reference[index + 1]["base_message_ix"] != native["base_message_ix"],
                    f"{name}: token text changed inside a message")
            require(altered["token"] == native["token"] + "\n",
                    f"{name}: unexpected boundary token change")
            changed.append(index)
        require(changed == expected_changed_indices,
                f"{name}: expected exactly six known boundary LF token changes; got {changed}")
        details[name] = {
            "boundary_tokens_with_added_join_LF": len(changed),
            "zero_based_content_indices": changed,
            "all_other_token_strings_identical": True,
            "original_message_ids_and_functions_identical": True,
        }
    return {
        "strict_raw_token_string_equality": False,
        "original_message_bodies_match_after_exact_boundary_join_LF_removal": True,
        "allowed_difference": "Only a single added LF on each non-final message's last token; no general whitespace normalization",
        "non_native_formats": details,
    }


def count_matches(records: list[dict]) -> dict[str, int]:
    counts = {role: 0 for role in ROLES}
    for record in records:
        if record["prediction"] == record["original_function"]:
            counts[record["original_function"]] += 1
    return counts


def metrics(records: list[dict], native: bool) -> dict:
    correct = count_matches(records)
    if native:
        return {
            "metric": "native_role_accuracy",
            "correct": sum(correct.values()),
            "support": len(records),
            "accuracy": sum(correct.values()) / len(records),
            "per_role": {
                role: {"correct": correct[role], "support": EXPECTED_SUPPORT[role],
                       "recall": correct[role] / EXPECTED_SUPPORT[role] if EXPECTED_SUPPORT[role] else None}
                for role in ROLES
            },
        }
    return {
        "metric": "semantic_alignment_to_original_function",
        "formal_multiclass_accuracy_reported": False,
        "matched_original_function": sum(correct.values()),
        "support": len(records),
        "alignment_fraction": sum(correct.values()) / len(records),
        "per_original_function": {
            role: {"matched_original_function": correct[role], "support": EXPECTED_SUPPORT[role],
                   "alignment_fraction": correct[role] / EXPECTED_SUPPORT[role] if EXPECTED_SUPPORT[role] else None}
            for role in ROLES
        },
    }


def plot(formats: dict[str, list[dict]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    titles = (
        "Native delimiters (Correct tags)\nTrue roles: S / U / C / A; probe outputs all 5 roles",
        "Everything in user tags\nFormal role: all user; colors show ORIGINAL functions",
        "No message delimiters (No tags)\nColors show ORIGINAL functions; no formal role labels",
    )
    fig, axes = plt.subplots(5, 3, figsize=(18, 10), sharey=True)
    for column, prompt_key in enumerate(FORMATS):
        records = formats[prompt_key]
        positions = [record["token_in_prompt_ix"] for record in records]
        boundaries = [records[i]["token_in_prompt_ix"] for i in range(1, len(records))
                      if records[i]["base_message_ix"] != records[i - 1]["base_message_ix"]]
        for row, role in enumerate(ROLES):
            axis = axes[row, column]
            for original_role in ROLES:
                group = [record for record in records if record["original_function"] == original_role]
                if group:
                    axis.scatter([record["token_in_prompt_ix"] for record in group],
                                 [record["probabilities"][role] for record in group],
                                 s=4, color=COLORS[original_role], alpha=0.85,
                                 linewidths=0, rasterized=True)
            for boundary in boundaries:
                axis.axvline(boundary, color="#d7dce2", linewidth=0.7, zorder=0)
            axis.set_xlim(min(positions) - 5, max(positions) + 5)
            axis.set_ylim(-0.035, 1.035)
            axis.set_yticks([0, 0.5, 1])
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axis.grid(axis="y", color="#edf0f3", linewidth=0.6)
            axis.tick_params(labelsize=8)
            if column == 0:
                axis.set_ylabel(ROLE_LABELS[row], fontsize=11)
            if row == 0:
                axis.set_title(titles[column], fontsize=10, pad=12)
            if row != 4:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("Original token position in serialized input (delimiter gaps retained)", fontsize=8)
        total_matches = sum(count_matches(records).values())
        qualifier = "Native accuracy" if column == 0 else "Original-function alignment only"
        axes[-1, column].text(0.5, -0.41,
            f"{qualifier}: {total_matches}/915 = {total_matches / 915:.2%}"
            + ("" if column == 0 else "\nNo formal multiclass accuracy reported"),
            transform=axes[-1, column].transAxes, ha="center", va="top", fontsize=9)

    fig.suptitle("GPT-OSS-120B | Probe trained only on C4 + Dolma3 single-message inputs\n"
                 "Selected SUCAT linear probe: mid-16+20+24, C=0.005 | Frozen MXFP4 model",
                 fontsize=14, y=0.985)
    handles = [Line2D([], [], linestyle="", marker="o", markersize=5, color=COLORS[role],
                      label=label) for role, label in zip(ROLES, ("System", "User", "CoT", "Assistant", "Tool (no tool body in this case)"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5,
               title="Color = ORIGINAL message function (including in the two altered formats)",
               fontsize=9, title_fontsize=9, frameon=False)
    fig.text(0.5, 0.012,
             "All 915 content tokens shown per format, including all 57 system tokens. "
             "Raw per-token probabilities: no smoothing or truncation.\n"
             "Toolness is a real fifth output; this case has no tool body. Altered formats retain 6 join-LFs fused into message-final tokens.",
             ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.067, right=0.99, top=0.825, bottom=0.145, hspace=0.22, wspace=0.11)
    fig.savefig(out_dir / f"{STEM}.png", dpi=180, facecolor="white")
    fig.savefig(out_dir / f"{STEM}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projections", "--projection-path", dest="projections", type=Path, default=DEFAULT_PROJECTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    projection_path = args.projections.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    before_sha = sha256(projection_path)
    formats = read_predictions(projection_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(formats, out_dir)
    require(sha256(projection_path) == before_sha, "Projection input changed while rendering")
    summary = {
        "source_projection_path": str(projection_path),
        "source_projection_sha256": before_sha,
        "model_path": str(MODEL_PATH),
        "model_sha256": MODEL_SHA256,
        "model_loaded": False,
        "inference_performed": False,
        "model_variant": "selected",
        "protocol": "document",
        "role_space": "sucat",
        "feature": FEATURE,
        "C": 0.005,
        "model_identity_note": "Archived selection identity and checkpoint fingerprint; only the projection CSV was read, not the checkpoint. The model hash is recorded, not recomputed by this script.",
        "training_scope": "C4 + Dolma3 single-message wrappers only; no constructed or natural multi-turn training",
        "probability_role_order": list(ROLES),
        "input_case_original_structure": ["system", "user", "cot", "assistant", "user", "cot", "assistant"],
        "plotting": {"system_body_retained": True, "smoothing": False, "truncation": False,
                     "x_axis": "archived token_in_prompt_ix; serialized-input delimiter gaps retained",
                     "colors": "original message functions, including non-native formats"},
        "validation": {"rows": 13725, "tokens_per_format": 915,
                       "all_five_probabilities_per_token": True, "duplicate_rows": False,
                       "finite_probability_range_checked": True, "probability_sum_absolute_tolerance": 2e-6,
                       "cross_format_content_identity": validate_message_identity(formats),
                       "native_expected_correct_counts_checked": EXPECTED_NATIVE_CORRECT},
        "formats": {},
        "outputs": [f"{STEM}.png", f"{STEM}.pdf"],
    }
    for name in FORMATS:
        entry = metrics(formats[name], native=name == "proper_tags")
        entry["formal_role_labels"] = (
            "native message roles/channels" if name == "proper_tags" else
            "all user" if name == "everything_in_user_tags" else
            "no original message-level delimiter labels"
        )
        entry["position_min"] = min(r["token_in_prompt_ix"] for r in formats[name])
        entry["position_max"] = max(r["token_in_prompt_ix"] for r in formats[name])
        summary["formats"][name] = entry
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "source_sha256": before_sha,
                      "native_accuracy": summary["formats"]["proper_tags"]["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
