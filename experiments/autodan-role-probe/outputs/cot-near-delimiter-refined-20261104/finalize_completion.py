"""CPU-only final artifact binding; no model, API, or AgentDojo execution."""
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
FIRST = OUT.parent / "cot-near-delimiter-20261104"
PREVIOUS = OUT.parent / "near-delimiter-20261104"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)


def canonical_template_sha(template):
    return hashlib.sha256(json.dumps(template, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def main():
    report = load(OUT / "report_summary.json")
    dataset = load(OUT / "dataset.json")
    score_protocol = load(OUT / "score_protocol.json")
    search_protocol = load(OUT / "search_protocol.json")
    winner = load(OUT / "winner.json")
    independent = load(OUT / "independent_audit.json")
    search_review = load(OUT / "cot_search_independent_review.json")
    eval_protocol = load(OUT / "eval_full/protocol.json")
    summary = load(OUT / "eval_full/summary.json")
    asr_audit = load(OUT / "eval_full/audit/audit.json")
    validation = load(OUT / "target_role_validation.json")
    expected_roles = ["system", "developer", "user", "cot", "assistant", "tool"]
    assert score_protocol["roles"] == winner["roles"] == expected_roles
    assert score_protocol["target_role"] == winner["target_role"] == "cot"
    assert score_protocol["target_role_index"] == winner["target_role_index"] == 3
    assert sha(OUT / "dataset.json") == score_protocol["dataset_sha256"]
    assert sha(FIRST / "dataset.json") == sha(PREVIOUS / "dataset.json") == sha(OUT / "dataset.json")
    probe = Path(search_protocol["config"]["probe"])
    assert sha(probe) == score_protocol["probe_sha256"]
    with np.load(probe, allow_pickle=False) as checkpoint:
        assert json.loads(checkpoint["metadata"].item())["roles"] == expected_roles
    assert validation["status"] == "passed"
    assert validation["actual_scorer_constructor_prepare_and_aggregation_exercised"]
    assert validation["cross_role_protocol_cache_rejected"]
    assert validation["real_user_and_cot_fields_preserved"]
    assert score_protocol["truncation"] is False

    frozen = load(OUT / "provenance/frozen_code.json")
    for item in frozen["files"]:
        assert sha(item["path"]) == sha(OUT / "provenance" / item["snapshot"]) == item["sha256"]
    for name, expected in score_protocol["code_sha256"].items():
        assert sha(ROOT / name) == expected
    for name, expected in search_protocol["config"]["code_sha256"].items():
        assert sha(name) == expected
    for path, expected in search_review["inputs_sha256"].items():
        assert sha(path) == expected

    first_rows, current_rows = rows(FIRST / "scores.jsonl"), rows(OUT / "scores.jsonl")
    goal_hashes = {hashlib.sha256(c["actual_goal"].encode()).hexdigest() for c in dataset["contexts"]}
    assert len(goal_hashes) == len(dataset["contexts"]) == 6
    for trial, candidates in [(FIRST, first_rows), (OUT, current_rows)]:
        assert len(candidates) == len({r["template"] for r in candidates}) == 42
        audit = load(trial / "independent_audit.json")
        near = load(trial / "near_search_audit.json")
        assert audit["status"] == near["status"] == "passed"
        assert audit["target_role"] == "cot" and audit["target_role_index"] == 3
        assert audit["candidates"] == 42
        assert near["search"]["full_context_preparations"] == 252
        assert near["search"]["all_payload_copies_checked"] == 378
        for r in candidates:
            assert r["status"] == "complete" and r["complete"] and r["completed_cases"] == 6
            assert r["target_role"] == "cot" and r["target_role_index"] == 3
            assert r["template"].count("{goal}") == 1
            assert {c["actual_goal_sha256"] for c in r["per_case"]} == goal_hashes
            for c in r["per_case"]:
                close(c["goal_target_mean"], c["probabilities"]["goal"]["cot"])
                close(c["goal_cot_mean"], c["probabilities"]["goal"]["cot"])
                close(c["goal_user_mean"], c["probabilities"]["goal"]["user"])
            close(r["fitness"], sum(c["goal_cot_mean"] for c in r["per_case"]) / 6)
    templates = {r["template"] for r in first_rows + current_rows}
    seed = load(FIRST / "latent_best_cot_seed.json")
    protected = set(eval_protocol["templates"].values()) | {load(FIRST / "winner.json")["template"], seed["template"]}
    assert len(protected) == 5
    historical = {r["template"] for r in rows(PREVIOUS / "scores.jsonl")}
    assert len(templates) == 77 and len(templates - protected) == 72
    assert len(templates - protected - historical) == 68
    assert winner["template"] == seed["template"]
    assert winner["template_sha256"] == canonical_template_sha(winner["template"])
    assert (OUT / "optimized_template.txt").read_text().rstrip("\n") == winner["template"]
    close(winner["fitness"], max(r["fitness"] for r in first_rows + current_rows))
    close(winner["fitness"], report["probe_means"]["optimized"])
    assert not search_review["cross_trial_accounting"]["new_mutations_exceeded_final_seed"]
    assert report["refinement_two_round_api"]["combined"]["successful_calls"] == 32
    close(report["refinement_two_round_api"]["combined"]["reported_cost"], 0.0395372)

    assert summary["terminal"] == summary["scheduled_episodes"] == 1556
    assert asr_audit["status"] == "verified"
    assert asr_audit["audited_episodes"] == asr_audit["expected_episodes"] == 1556
    assert asr_audit["model_calls"] == 11938
    assert asr_audit["official_predicates_rechecked"] == 1555
    assert asr_audit["native_error_episodes"] == 1 and asr_audit["native_errors_preserve_null"]
    assert asr_audit["generation_calls"] == 0 and not asr_audit["gpu_required"]
    assert asr_audit["protocol_file_sha256"] == sha(OUT / "eval_full/protocol.json")
    assert eval_protocol["winner_sha256"] == sha(OUT / "winner.json")
    assert eval_protocol["previous_winner_sha256"] == sha(PREVIOUS / "winner.json")
    assert eval_protocol["optimization_objective"]["target_role"] == "cot"
    assert eval_protocol["previous_winner_objective"]["target_role"] == "user"
    populations = set().union(*map(set, dataset["sample_manifest"]["population_task_ids"].values()))
    for condition, successes in {"plain": 11, "remember": 230, "previous_optimized": 88, "optimized": 21}.items():
        cases = [c for c in eval_protocol["cases"] if c["condition"] == condition]
        assert len(cases) == 389 and {c["pair_id"] for c in cases} == populations
        actual = report["full_asr"][condition]
        assert actual["terminal"] == actual["scheduled"] == 389
        assert actual["successes"] == successes
        assert actual["errors"] == (1 if condition == "remember" else 0)
    assert report["full_asr"]["remember"]["asr"] is None
    assert report["full_asr"]["remember"]["scored"] == 388
    assert report["full_evaluation_terminal"] and not report["full_evaluation_complete"]
    final_review = load(OUT / "native_error_review_final.json")
    assert final_review["statistics_source"]["sha256"] == sha(OUT / "eval_full/summary.json")
    assert final_review["episodes"][0]["security"] is None
    assert final_review["episodes"][0]["utility"] is None

    old_manifest = load(PREVIOUS / "result_manifest.json")
    for item in old_manifest["artifact_files"]:
        assert sha(PREVIOUS / item["path"]) == item["sha256"]
    own_pids = [2611784, 2619805, 2619924, 2619925]
    assert not any(Path(f"/proc/{pid}").exists() for pid in own_pids)

    now = datetime.now(timezone.utc).isoformat()
    phase1 = {"status": "search_only_complete", "created_at": now, "target_role": "cot",
              "target_role_index": 3, "candidate_evaluations": 42, "native_context_forwards": 252,
              "asr_planned_for_this_phase": False, "asr_selection_feedback_used": False,
              "followup": str(OUT), "winner_probability": load(FIRST / "winner.json")["fitness"],
              "evidence": [binding(FIRST / name) for name in ["winner.json", "scores.jsonl", "score_protocol.json",
                  "search_protocol.json", "independent_audit.json", "near_search_audit.json",
                  "cot_search_independent_review.json", "latent_best_cot_seed.json", "historical_cot_ranking.json"]]}
    write(FIRST / "completion.json", phase1)
    completion_audit = {
        "status": "passed_with_preserved_native_error", "created_at": now,
        "new_generation_or_api_calls": 0, "language_model_loaded": False,
        "roles": expected_roles, "target_role": "cot", "target_role_index": 3,
        "real_user_probability_column": 2, "actual_scorer_cache_separation_verified": True,
        "frozen_current_source_and_snapshot_hashes_verified": len(frozen["files"]),
        "previous_result_manifest_artifacts_verified_unchanged": len(old_manifest["artifact_files"]),
        "dataset_sha256": sha(OUT / "dataset.json"), "probe_sha256": sha(probe),
        "native_candidate_evaluations": 84, "native_context_forwards": 504,
        "unique_templates": 77, "cross_trial_repeated_evaluations": 7,
        "unique_nonprotected_templates": 72, "historically_novel_nonprotected_templates": 68,
        "all_six_goals_immutable": True, "full_simulated_contexts_no_truncation": True,
        "context_provenance": dataset["provenance"],
        "winner_is_historical_seed": True, "new_mutations_exceeded_seed": False,
        "winner_template_hash_semantics": {
            "raw_utf8_sha256": hashlib.sha256(winner["template"].encode()).hexdigest(),
            "canonical_json_string_sha256": canonical_template_sha(winner["template"]),
            "winner_template_sha256_field_uses": "canonical JSON string; not raw UTF-8 template bytes"},
        "asr_independent_audit": {k: asr_audit[k] for k in ["status", "audited_episodes", "expected_episodes",
            "model_calls", "official_predicates_rechecked", "native_error_episodes", "complete_prompt_tokens_reencoded",
            "generated_token_ids_checked", "native_errors_preserve_null", "payload_special_token_exclusion_verified",
            "optimization_role_metadata_verified", "all_completed_episodes_official_utility_and_security_predicates_match"]},
        "same_389_population_four_fresh_conditions_verified": True,
        "final_result_source": binding(OUT / "eval_full/summary.json"),
        "native_error_policy": "One remember episode remains null; full-population bounds are not confidence intervals.",
        "process_completion": {"main_exit_code": 0, "audit_exit_code": 0, "own_pids_absent": own_pids,
            "worker_shutdown": "Workers exited; EngineCore was forcibly cleaned after shutdown timeout, after all episodes and summary were saved.",
            "scope": "Only this experiment's processes; no assertion about other GPU jobs."},
        "evidence": [binding(OUT / name) for name in ["report_summary.json", "experiment_binding.json", "refinement_protocol.json",
            "target_role_validation.json", "independent_audit.json", "near_context_preflight.json", "near_search_audit.json",
            "cot_search_independent_review.json", "role_metadata_cpu_checks.json", "native_error_review_final.json",
            "eval_full/protocol.json", "eval_full/audit/audit.json"]]}
    write(OUT / "completion_audit.json", completion_audit)
    completion = {"status": "complete_with_native_errors", "created_at": now,
        "goal": "Optimize mean goal-token P(cot) on the fixed six simulated contexts, then compare fresh all389 ASR.",
        "all_planned_work_finished": True, "target_role": "cot", "target_role_index": 3,
        "candidate_evaluations": 84, "unique_templates": 77,
        "probe_means": report["probe_means"], "api_two_rounds": report["refinement_two_round_api"]["combined"],
        "asr": report["full_asr"], "missing_verdict_bounds": report["missing_verdict_bounds"]["389"],
        "interpretation": "The cot-selected historical seed improves P(cot), but its fresh ASR is below the previous user-target winner and Remember. New mutations did not exceed the seed. This single experiment does not isolate a causal effect of the role target.",
        "report": str(OUT / "REPORT.zh-CN.md"), "template": str(OUT / "optimized_template.txt"),
        "reproduction_script": str(ROOT / "experiments/autodan-role-probe/reproduce.sh"),
        "completion_audit": binding(OUT / "completion_audit.json")}
    write(OUT / "completion.json", completion)
    core = [p for p in OUT.iterdir() if p.is_file() and p.name != "result_manifest.json"]
    core += list((OUT / "provenance").glob("*"))
    core += [p for p in (OUT / "eval_full").iterdir() if p.is_file() and p.name != ".run.lock"]
    core += [p for p in (OUT / "eval_full/audit").iterdir() if p.is_file()]
    artifacts = [{"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(core) if p.is_file()]
    manifest = {"status": "complete_with_native_errors", "created_at": now, "artifact_files": artifacts,
        "external_phase1_completion": binding(FIRST / "completion.json"),
        "raw_artifact_validation": "Per-request raw responses, IDs, prompt hashes, parsed trajectories and typed snapshots were checked by eval_full/audit/audit.json; native probe feature/goal-mask artifacts were checked by the two independent probe audits. This manifest hashes the aggregate evidence and source snapshots, not every raw file again."}
    write(OUT / "result_manifest.json", manifest)
    for item in artifacts:
        assert sha(OUT / item["path"]) == item["sha256"]
    print(json.dumps({"status": completion["status"], "manifest_artifacts": len(artifacts),
                      "asr_audited": asr_audit["audited_episodes"], "model_calls": asr_audit["model_calls"]}))


if __name__ == "__main__":
    main()
