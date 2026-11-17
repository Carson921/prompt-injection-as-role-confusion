#!/usr/bin/env python3
"""External clean IPI Arena first-role interventions: statistics and English plots.

This adapter reuses the frozen AgentDojo numerical/bootstrap and figure routines
without editing them. IPI has one external cohort (41 tasks), not FIT/DEV/TEST.
Only genuinely present, nonempty source bodies are analyzed. Each present source
has strip, move and all five alternative wrappers; a frozen pilot wrapper catalog
supplies a destination wrapper when it is absent from the same trace. Absent
source roles remain NA with support zero. Every contrast is paired within trace.

Usage:
    python analyze_ipi_trace_interventions.py --output ROOT --stage pilot
    python analyze_ipi_trace_interventions.py --output ROOT --stage all

No model, generation backend or tools are loaded. Token curves are supplemental:
pilot shows every token; all uses 20 relative-body bins with equal trace weights.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analyze_agentdojo_trace_interventions as core
import plot_agentdojo_trace_token_curves as profiles


ROLES = core.ROLES
P_COLUMNS = core.P_COLUMNS
EXPECTED_TRACES = {"pilot": 1, "all": 41}


def validate_opaque_target_mask(row: dict, positions=None) -> bool:
    """Check authoritative suffix coordinates without decoding or parsing it."""
    boundary = row.get("opaque_terminal_start")
    declared = row.get("trace_has_opaque_terminal", False)
    original_boundary = row.get("original_opaque_terminal_start")
    suffix = row.get("trailing_opaque_ids", [])
    if boundary is None:
        if declared or original_boundary is not None or suffix:
            raise ValueError("Opaque suffix metadata lacks its authoritative boundary")
        return False
    ids, origins = row["input_ids"], row["token_source_positions"]
    if (declared is not True or not isinstance(boundary, int) or isinstance(boundary, bool)
            or not isinstance(original_boundary, int) or isinstance(original_boundary, bool)
            or not 0 <= boundary < len(ids) or original_boundary < 0
            or len(origins) != len(ids) or not suffix or ids[boundary:] != suffix
            or origins[boundary:] != list(range(original_boundary, original_boundary + len(suffix)))):
        raise ValueError("Opaque suffix coordinates, tokens or original ownership differ")
    selected = row["target_positions"] if positions is None else positions
    if (any(int(position) >= boundary for position in selected)
            or any(int(position) >= original_boundary for position in row["target_original_positions"])):
        raise ValueError("An unlabelled opaque suffix token was selected as a source body")
    return True


def read_and_validate(path: Path, stage: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check the external cohort and all eight readouts per present source body."""
    frame = pd.read_csv(path, dtype={"input_id": str, "trace_id": str}, keep_default_na=False)
    required = {
        "input_id", "trace_id", "source_split", "operation", "source_role", "destination_role",
        "body_tokens", "original_body_start", "new_body_start", "total_input_tokens", *P_COLUMNS,
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing input-summary columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input summary is empty")
    for name in ("input_id", "trace_id", "source_role", "operation"):
        if (frame[name].astype(str).str.len() == 0).any():
            raise ValueError(f"Empty {name}")
    if set(frame.source_split) != {"external"}:
        raise ValueError("IPI source_split must be external; no fabricated training split")
    if not set(frame.source_role).issubset(ROLES) or not set(frame.operation).issubset(core.OPERATIONS):
        raise ValueError("Unknown source role or intervention")
    counts = ("body_tokens", "original_body_start", "new_body_start", "total_input_tokens")
    for name in (*counts, *P_COLUMNS):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name]).all():
            raise ValueError(f"Nonfinite {name}")
    for name in counts:
        if (frame[name] % 1 != 0).any():
            raise ValueError(f"{name} must contain integer token counts/indices")
    if (frame.body_tokens <= 0).any() or (frame.total_input_tokens <= 0).any():
        raise ValueError("Body and input token counts must be positive")
    if (frame[["original_body_start", "new_body_start"]] < 0).any().any():
        raise ValueError("Body start indices must be nonnegative")
    if (frame.new_body_start + frame.body_tokens > frame.total_input_tokens).any():
        raise ValueError("Body exceeds its input length")
    probabilities = frame[list(P_COLUMNS)].to_numpy(dtype=float)
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Role probabilities must lie in [0, 1]")
    sum_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1)))
    if sum_error > 2e-5:
        raise ValueError("Six-class probabilities do not sum to one")
    trace_ids = sorted(frame.trace_id.unique())
    if len(trace_ids) != EXPECTED_TRACES[stage]:
        raise ValueError(f"Expected {EXPECTED_TRACES[stage]} IPI traces; found {len(trace_ids)}")
    coverage = []
    for trace_id, rows in frame.groupby("trace_id", sort=True):
        baseline = rows[rows.operation == "baseline"]
        if baseline.empty or baseline.source_role.duplicated().any():
            raise ValueError(f"{trace_id}: one baseline readout per actually present role is required")
        present = set(baseline.source_role)
        if set(rows.source_role) != present or baseline.input_id.nunique() != 1 or baseline.total_input_tokens.nunique() != 1:
            raise ValueError(f"{trace_id}: inconsistent baseline inventory")
        for role in present:
            target = rows[rows.source_role == role]
            if target.operation.value_counts().to_dict() != {"swap": 5, "baseline": 1, "strip": 1, "move": 1}:
                raise ValueError(f"{trace_id}/{role}: require baseline, strip, move and five swaps")
            swaps = target[target.operation == "swap"]
            if set(swaps.destination_role) != set(ROLES) - {role} or swaps.destination_role.duplicated().any():
                raise ValueError(f"{trace_id}/{role}: require all five alternative role wrappers")
            for operation in ("baseline", "strip", "move"):
                destination = target.loc[target.operation == operation, "destination_role"].iloc[0]
                allowed = {"", role} | ({"none", "unwrapped"} if operation == "strip" else set())
                if destination not in allowed:
                    raise ValueError(f"{trace_id}/{role}: invalid non-swap destination")
            if target.body_tokens.nunique() != 1 or target.original_body_start.nunique() != 1:
                raise ValueError(f"{trace_id}/{role}: body denominator/original coordinate changed")
            original = target[target.operation == "baseline"].iloc[0]
            if original.original_body_start != original.new_body_start or original.new_body_start + original.body_tokens > original.total_input_tokens:
                raise ValueError(f"{trace_id}/{role}: invalid original body coordinates")
            moved = target[target.operation == "move"].iloc[0]
            if moved.original_body_start == moved.new_body_start:
                raise ValueError(f"{trace_id}/{role}: movement did not change body position")
        if len(rows) != 8 * len(present) or rows.input_id.nunique() != 1 + 7 * len(present):
            raise ValueError(f"{trace_id}: expected 8m summary rows and 1+7m distinct inputs")
        coverage.append({"trace_id": trace_id, "source_split": "external", "roles_present": [r for r in ROLES if r in present],
                         "roles_not_scored": [r for r in ROLES if r not in present], "summary_rows": len(rows),
                         "unique_inputs": int(rows.input_id.nunique())})
    if (frame.groupby("input_id").trace_id.nunique() != 1).any():
        raise ValueError("An input ID is reused across tasks")
    for input_id, rows in frame.groupby("input_id", sort=False):
        if len(rows) > 1 and (rows.operation != "baseline").any():
            raise ValueError(f"{input_id}: only a baseline union may have multiple target bodies")
    validation = {
        "stage": stage, "dataset": "IPI Arena", "cohort_type": "external clean tasks",
        "input_summary": str(path.resolve()), "input_summary_sha256": core.sha256(path),
        "n_traces": len(trace_ids), "split_trace_counts": {"external": len(trace_ids)},
        "summary_rows": len(frame), "unique_inputs": int(frame.input_id.nunique()), "role_order": list(ROLES),
        "full_condition_coverage": True, "conditions_per_present_source": 8,
        "unique_inputs_per_trace": "1 + 7m, where m is the number of actually present source roles",
        "body_token_denominator_invariant_within_target": True,
        "absent_source_roles": "Missing observations (NA), never fabricated or assigned probability zero",
        "destination_wrappers": "Same-trace native wrapper first; otherwise the frozen pilot catalog; all five destinations per present source",
        "body_token_identity_validation": "Identity is checked against saved native IDs by the independent validator and token-curve loader",
        "max_probability_sum_error": sum_error, "coverage": coverage,
        "reused_numerical_analyzer_sha256": core.sha256(Path(core.__file__)),
        "adapter_sha256": core.sha256(Path(__file__)),
    }
    return frame, validation


@contextmanager
def ipi_presentation():
    """Change labels at the adapter boundary, leaving frozen implementations intact."""
    old_label, old_save, old_make = core.cohort_label, core.save_figure, profiles.make_figure
    old_profile_save = profiles.save_figure

    def cohort_label(cohort: str, n: int) -> str:
        return "Single clean IPI Arena trace · descriptive pilot" if cohort == "pilot" else f"Clean IPI Arena tasks · external cohort · n = {n}"

    def save_figure(fig, destination, name):
        for text in fig.texts:
            text.set_text(text.get_text().replace("native benign AgentDojo traces", "clean IPI Arena traces"))
        for ax in fig.axes:
            ax.title.set_text(ax.get_title().replace("farthest trace boundary", "farthest parsed boundary"))
        return old_save(fig, destination, name)

    def make_figure(title, subtitle, footer):
        subtitle = subtitle.replace("Single native benign trace:", "Single clean IPI Arena trace:")
        subtitle = subtitle.replace("All traces · secondary cohort", "IPI Arena · external clean cohort")
        return old_make(title, subtitle, footer)

    def save_profile(fig, destination):
        for ax in fig.axes:
            ax.title.set_text(ax.get_title().replace("content · absent from this trace", "content · no scored first body"))
            for line in ax.lines:
                if len(np.atleast_1d(line.get_xdata())) == 1:
                    line.set_marker("o")
                    line.set_markersize(4)
        return old_profile_save(fig, destination)

    core.cohort_label, core.save_figure, profiles.make_figure = cohort_label, save_figure, make_figure
    profiles.save_figure = save_profile
    try:
        yield
    finally:
        core.cohort_label, core.save_figure, profiles.make_figure = old_label, old_save, old_make
        profiles.save_figure = old_profile_save


def load_token_curves(root: Path, stage: str, frame: pd.DataFrame) -> tuple[dict, dict]:
    """Read only saved probabilities and verify exact body-token alignment."""
    summary = frame[frame.operation.isin(profiles.OPERATIONS)].copy()
    keys = ["trace_id", "source_role", "operation"]
    if summary.duplicated(keys).any():
        raise ValueError("Repeated token-curve condition")
    expected = summary.set_index(keys)
    curves, hashes = {}, {}
    max_error = 0.0
    input_path = root / stage / "inputs.jsonl"
    with input_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            operation = record["operation"]
            if operation not in profiles.OPERATIONS:
                continue
            input_id, trace_id = record["input_id"], record["trace_id"]
            if not isinstance(input_id, str) or Path(input_id).name != input_id:
                raise ValueError("Unsafe input identifier")
            path = root / "readouts" / f"{input_id}.npz"
            with np.load(path, allow_pickle=False) as data:
                positions, original, token_ids = [data[name].copy() for name in ("positions", "original_positions", "token_ids")]
                probability, roles = data["probabilities"].astype(float), data["roles"].tolist()
            validate_opaque_target_mask(record, positions)
            if roles != list(ROLES) or not np.array_equal(positions, record["target_positions"]) or not np.array_equal(original, record["target_original_positions"]):
                raise ValueError(f"{input_id}: class order or coordinate map differs")
            if not np.array_equal(token_ids, np.asarray(record["input_ids"], dtype=np.int64)[positions]):
                raise ValueError(f"{input_id}: body tokens differ from original input")
            if probability.shape != (len(positions), 6) or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any() or not np.allclose(probability.sum(1), 1, atol=2e-5, rtol=0):
                raise ValueError(f"{input_id}: invalid probability matrix")
            hashes[input_id] = core.sha256(path)
            for role, mask in record["role_target_positions"].items():
                key = (trace_id, role, operation)
                if key not in expected.index or key in curves:
                    raise ValueError(f"Unexpected or repeated token curve: {key}")
                row = expected.loc[key]
                indices = np.flatnonzero(np.isin(positions, mask))
                if row.input_id != input_id or len(indices) != row.body_tokens or not np.array_equal(positions[indices], mask) or not np.array_equal(original[indices], record["original_role_target_positions"][role]):
                    raise ValueError(f"{key}: body alignment or denominator differs")
                if len(indices) > 1 and (np.diff(original[indices]) != 1).any():
                    raise ValueError(f"{key}: original body is not contiguous")
                values = probability[indices, ROLES.index(role)]
                error = abs(float(values.mean()) - float(row[f"p_{role}"]))
                max_error = max(max_error, error)
                if error > 3e-6:
                    raise ValueError(f"{key}: curve does not recover the primary mean")
                curves[key] = dict(input_id=input_id, trace_id=trace_id, source_split="external", source_role=role,
                                   operation=operation, body_tokens=len(indices), positions=positions[indices],
                                   original_positions=original[indices], token_ids=token_ids[indices], probability=values)
    if set(curves) != set(expected.index):
        raise ValueError("Incomplete native token-curve inventory")
    for trace_id, role in {key[:2] for key in curves}:
        baseline = curves[(trace_id, role, "baseline")]
        for operation in ("strip", "move"):
            current = curves[(trace_id, role, operation)]
            if not np.array_equal(current["original_positions"], baseline["original_positions"]) or not np.array_equal(current["token_ids"], baseline["token_ids"]):
                raise ValueError(f"{trace_id}/{role}: intervention changed body-token identity")
    validation = dict(stage=stage, dataset="IPI Arena", n_traces=frame.trace_id.nunique(), n_role_condition_curves=len(curves),
                      inputs_sha256=core.sha256(input_path), input_summary_sha256=core.sha256(root / stage / "input-summary.csv"),
                      readout_sha256=hashes, token_ids_and_original_positions_identical_across_conditions=True,
                      baseline_union_mask_split_using_role_target_positions=True, all_curve_means_match_primary_summary=True,
                      max_primary_mean_absolute_error=max_error, supplemental_only=True, new_model_inference=False)
    return curves, validation


def analyze(root: Path, stage: str, *, token_curves: bool = True) -> dict[str, Any]:
    root = root.resolve()
    stage_root = root / stage
    frame, validation = read_and_validate(stage_root / "input-summary.csv", stage)
    protocol_path = stage_root / "protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text())
        if "selected_traces" in protocol and set(protocol["selected_traces"]) != set(frame.trace_id):
            raise ValueError("Summary task inventory differs from the frozen protocol")
        validation["protocol_sha256"] = core.sha256(protocol_path)
    inputs_path = stage_root / "inputs.jsonl"
    if inputs_path.is_file():
        availability, wrapper_provenance, opaque_inputs, opaque_traces = {}, Counter(), 0, set()
        with inputs_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if validate_opaque_target_mask(row):
                    opaque_inputs += 1
                    opaque_traces.add(row["trace_id"])
                if row["operation"] == "baseline" and "role_availability" in row:
                    availability[row["trace_id"]] = row["role_availability"]
                if row["operation"] == "swap":
                    provenance = row.get("wrapper_provenance", {})
                    wrapper_provenance[provenance.get("kind", "unrecorded")] += 1
        validation["inputs_sha256"] = core.sha256(inputs_path)
        validation["native_role_availability"] = availability
        validation["swap_wrapper_provenance_counts"] = dict(wrapper_provenance)
        validation["opaque_terminal_context"] = dict(traces=len(opaque_traces), inputs=opaque_inputs,
            selected_suffix_tokens=0, target_masks_checked_without_native_reparse=True,
            policy="Entire final failed generation/open header is unlabelled context; eligible movement ends before it")
    analysis_root = stage_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    core.write_json(analysis_root / "validation.json", validation)
    cohort = "pilot" if stage == "pilot" else "all"
    destination = analysis_root / cohort
    with ipi_presentation():
        report = core.summarize(frame, destination, cohort, validation)
        # Replace the AgentDojo training-cohort caveat, without changing any values.
        report["dataset"] = "IPI Arena"
        report["cohort_type"] = "external clean tasks"
        report["limitations"] = [item for item in report["limitations"] if not item.startswith("Fit traces")]
        report["limitations"].extend([
            "IPI is one external dataset cohort; no FIT/DEV/TEST split or training-trace support is fabricated.",
            "A missing source role is NA. Alternative wrappers may use the frozen pilot catalog when absent from the same trace.",
            "Shared fixture templates can correlate tasks; trace-bootstrap intervals are not template-family uncertainty estimates.",
            "A final invalid_native_response is retained with its open header as opaque context. Its tokens supply no role labels, source bodies or replacement wrappers, and no later simulated tool response is generated.",
            "Movement ends at the farthest eligible parsed-prefix boundary, before any open header or opaque terminal suffix; displayed positions are divided by the full observed trace length, including that suffix.",
        ])
        core.write_json(destination / "report-data.json", report)
        if token_curves:
            curves, curve_validation = load_token_curves(root, stage, frame)
            if stage == "pilot":
                profiles.plot_pilot(curves, destination, curve_validation)
            else:
                profiles.plot_all(curves, sorted(frame.trace_id.unique()), destination, "all", curve_validation)
    return {"stage": stage, "dataset": "IPI Arena", "n_traces": validation["n_traces"],
            "role_support": report["bootstrap"]["role_support"], "output": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("pilot", "all"))
    parser.add_argument("--skip-token-curves", action="store_true", help="Produce primary statistics/plots only")
    args = parser.parse_args()
    core.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                             "axes.labelsize": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    print(json.dumps(analyze(args.output, args.stage, token_curves=not args.skip_token_curves)))


if __name__ == "__main__":
    main()
