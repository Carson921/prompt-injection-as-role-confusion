#!/usr/bin/env python3
"""CPU-only sampled GPU resource appendix; never read arrays, predictions or text.

Completed windows come only from receipt.stage_marker_snapshot. Reused pilot
stages follow migration.parent_run_dir to their original receipts. Global device
samples are observations, never an attribution of pure model memory/compute.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

BASE = Path("/workspace/role-retraining-20261026")
STAGES = ("pilot.scan", "pilot.extract", "pilot.train", "formal.extract", "formal.train",
          "formal.extract.test", "formal.extract.external", "formal.dense")


def require(value, message):
    if not value:
        raise ValueError(message)


def utc(value):
    return None if value is None else datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite resource time/value")
    return result


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "mtime_unix": path.stat().st_mtime}


def snapshot_csv(path, monitor_stopped=False):
    """Hash the exact initial byte prefix even while the monitor keeps appending."""
    path = Path(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(before.st_size)
        after = os.fstat(stream.fileno())
    require(len(raw) == before.st_size and after.st_size >= before.st_size,
            "Monitoring CSV was truncated during snapshot")
    require((before.st_dev, before.st_ino) == (path.stat().st_dev, path.stat().st_ino), "Monitoring CSV was replaced during snapshot")
    if monitor_stopped:
        require(after.st_size == before.st_size and path.stat().st_size == before.st_size and
                path.stat().st_mtime_ns == before.st_mtime_ns, "Declared-final CSV changed during read")
    complete_end = raw.rfind(b"\n") + 1
    require(complete_end > 0, "Monitoring CSV has no complete header")
    discarded = len(raw) - complete_end
    require(not monitor_stopped or discarded == 0, "Declared-final CSV has an incomplete final row")
    reader = csv.DictReader(io.StringIO(raw[:complete_end].decode("utf-8")))
    require(reader.fieldnames == ["unix_time", "gpu", "memory_used_mib", "utilization_percent"], "Unknown monitor CSV schema")
    samples = {gpu: [] for gpu in range(4)}
    for row in reader:
        require(set(row) == set(reader.fieldnames) and all(value is not None for value in row.values()), "Malformed complete monitor row")
        gpu = int(row["gpu"])
        require(gpu in samples, "Unexpected GPU index")
        t, memory, utilization = [finite(row[key]) for key in ("unix_time", "memory_used_mib", "utilization_percent")]
        require(t > 0 and memory >= 0 and 0 <= utilization <= 100, "Invalid monitor sample")
        require(not samples[gpu] or t > samples[gpu][-1][0], "GPU monitor timestamps are not strictly increasing")
        samples[gpu].append((t, memory, utilization))
    source = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "snapshot_bytes": len(raw),
              "source_bytes_after_read": after.st_size,
              "identity_is_final": bool(monitor_stopped),
              "sha256_scope": "entire stopped-monitor CSV" if monitor_stopped else "provisional exact prefix only; NOT final raw CSV identity",
              "discarded_partial_tail_bytes": discarded, "sample_count": sum(map(len, samples.values())),
              "first_sample_unix": min((rows[0][0] for rows in samples.values() if rows), default=None),
              "last_sample_unix": max((rows[-1][0] for rows in samples.values() if rows), default=None)}
    source["first_sample_utc"], source["last_sample_utc"] = utc(source["first_sample_unix"]), utc(source["last_sample_unix"])
    return samples, source


def gardening_window(claim, report, observed_unix, completion_observation=None):
    claim, report = Path(claim), Path(report)
    if not claim.exists():
        return {"status": "not_recorded", "note": "No waiter claim; absence does not exclude unrelated CUDA contexts."}
    start = finite(read(claim)["claimed_unix"])
    result = {"claim": identity(claim), "start_unix": start, "start_utc": utc(start),
              "scope": "Conservative possible concurrency: waiter claim before tmux through report completion; includes CPU check/report, not exact inference duration."}
    observation = Path(completion_observation) if completion_observation else None
    if observation and observation.exists():
        require(report.exists(), "Root completion observation exists without report-complete")
        value = read(observation)
        report_identity = identity(report)
        require(value["report_complete_sha256"] == report_identity["sha256"], "Root completion observation report SHA mismatch")
        require(value["actual_infer_processes_absent"] is True, "Root observation does not confirm inference processes absent")
        end = finite(value["observed_unix"])
        require(start <= end <= observed_unix, "Root completion observation is outside the observation window")
        return {**result, "report_complete": report_identity, "completion_observation": identity(observation),
                "status": "complete_conservative_bound", "end_unix": end, "end_utc": utc(end),
                "end_source": "root_report_hash_bound_completion_observation"}
    if observation:
        require(start <= observed_unix, "Gardening claim lies in the future")
        return {**result, "status": "awaiting_root_completion_observation", "expected_observation_path": str(observation),
                "end_unix": observed_unix, "end_utc": utc(observed_unix), "end_source": "open_ended_as_of_appendix_observation"}
    if report.exists():
        result["report_complete"] = identity(report)
        end, source = report.stat().st_mtime, "original_report_file_mtime_observed_on_this_host"
        require(start <= end <= observed_unix, "Gardening completion observation is outside the observation window")
        result.update(status="complete_conservative_bound", end_unix=end, end_utc=utc(end), end_source=source)
        result["mtime_limitation"] = "File mtime is an original-host completion proxy; copying can change it. Prefer a root hash-bound completion observation."
    else:
        require(start <= observed_unix, "Gardening claim lies in the future")
        result.update(status="open_ended_possible_concurrency", end_unix=observed_unix,
                      end_utc=utc(observed_unix), end_source="open_ended_as_of_appendix_observation")
    return result


def overlap(a, b, gardening):
    if "start_unix" not in gardening:
        return 0.0
    return max(0.0, min(b, gardening["end_unix"]) - max(a, gardening["start_unix"]))


def gpu_stats(samples, start, end, period, gardening):
    rows = [r for r in samples if start <= r[0] <= end]
    result = {"sample_count": len(rows), "sampled_memory_peak_mib": None, "partial_coverage": True,
              "first_sample_unix": None, "last_sample_unix": None, "leading_unobserved_seconds": end - start,
              "trailing_unobserved_seconds": None, "long_internal_gaps": []}
    if not rows:
        result["missing_sample_note"] = "No monitor samples inside this completed stage window."
        return result
    times = [r[0] for r in rows]
    gaps = [(a, b) for a, b in zip(times, times[1:]) if b - a > 2 * period]
    memory = max(r[1] for r in rows)
    peaks = [r[0] for r in rows if r[1] == memory]
    in_gardening = lambda t: "start_unix" in gardening and gardening["start_unix"] <= t <= gardening["end_unix"]
    result.update(first_sample_unix=times[0], first_sample_utc=utc(times[0]), last_sample_unix=times[-1], last_sample_utc=utc(times[-1]),
        leading_unobserved_seconds=max(0.0, times[0] - start), trailing_unobserved_seconds=max(0.0, end - times[-1]),
        observed_sample_span_seconds=times[-1] - times[0], observed_sample_span_fraction=(times[-1] - times[0]) / (end - start),
        maximum_sampling_gap_seconds=max((b - a for a, b in zip(times, times[1:])), default=None),
        long_internal_gaps=[{"from_unix": a, "to_unix": b, "seconds": b - a} for a, b in gaps],
        sampled_memory_peak_mib=memory, peak_first_unix=peaks[0], peak_first_utc=utc(peaks[0]),
        peak_last_unix=peaks[-1], peak_last_utc=utc(peaks[-1]), peak_occurrences=len(peaks),
        sampled_utilization_peak_percent=max(r[2] for r in rows), utilization_sample_mean_percent=sum(r[2] for r in rows) / len(rows),
        samples_in_possible_gardening_concurrency=sum(in_gardening(r[0]) for r in rows),
        memory_peak_may_include_gardening=any(in_gardening(t) for t in peaks),
        partial_coverage=bool(gaps or times[0] - start > 2 * period or end - times[-1] > 2 * period))
    result["coverage_note"] = "Span and gaps describe discrete samples, not continuous coverage or a guaranteed maximum."
    return result


def original_receipt(run_root, name):
    root, chain, seen = Path(run_root), [], set()
    for _ in range(8):
        path = root / "campaign/receipts" / (name + ".json")
        if not path.is_file():
            return None, chain, "receipt_missing"
        require(str(path.resolve()) not in seen, "Migration receipt cycle")
        seen.add(str(path.resolve()))
        receipt = read(path)
        require(receipt["name"] == name, "Receipt name differs")
        adopted = receipt.get("execution", {}).get("adopted_verified_atomic_artifacts") is True
        chain.append({**identity(path), "adopted": adopted, "source_run_root": str(root.resolve())})
        if adopted and name.startswith("pilot."):
            migration = root / "pilot/migration.json"
            if not migration.is_file():
                return None, chain, "adopted_pilot_origin_unresolved"
            value = read(migration)
            chain[-1]["migration"] = identity(migration)
            root = Path(value["parent_run_dir"]).parent
            continue
        return receipt, chain, None
    raise ValueError("Too many migration ancestors")


def stage_report(run_root, name, samples, period, gardening):
    receipt, chain, missing = original_receipt(run_root, name)
    result = {"name": name, "receipt_chain": chain, "timing_rule": "stage_marker_snapshot.started_unix + seconds only"}
    if missing:
        result.update(status="pending" if missing == "receipt_missing" else "origin_unresolved", reason=missing)
        status = Path(run_root) / name.split(".")[0] / "status.json"
        if status.exists():
            value = read(status)
            if len(name.split(".")) == 2 and value.get("stage") == name.split(".")[1]:
                result["current_stage_status"] = {key: value.get(key) for key in ("stage", "phase", "status", "started_unix")}
                result["current_status_source"] = identity(status)
        return result
    marker = receipt.get("stage_marker_snapshot", {})
    if not (marker.get("status") == "complete" and "started_unix" in marker and "seconds" in marker):
        result.update(status="complete_timing_unavailable", reason="Committed receipt lacks a complete absolute stage-marker window; no execution/candidate timing substitution.")
        return result
    phase, stage = name.split(".")[:2]
    require(marker.get("phase") == phase and marker.get("stage") == stage, "Stage marker identity mismatch")
    start, seconds = finite(marker["started_unix"]), finite(marker["seconds"])
    require(start > 0 and seconds > 0, "Invalid stage-marker window")
    end = start + seconds
    result.update(status="complete", original_version=Path(chain[-1]["source_run_root"]).name,
                  source_receipt_sha256=chain[-1]["sha256"], start_unix=start, start_utc=utc(start), end_unix=end, end_utc=utc(end),
                  seconds=seconds, reused_without_new_execution=len(chain) > 1,
                  possible_gardening_overlap_seconds=overlap(start, end, gardening),
                  gpu={str(gpu): gpu_stats(rows, start, end, period, gardening) for gpu, rows in samples.items()})
    return result


def preflight_report(run_root):
    root, origins = Path(run_root) / "pilot", []
    for _ in range(8):
        migration = root / "migration.json"
        if not migration.exists():
            break
        origins.append(identity(migration))
        parent = Path(read(migration)["parent_run_dir"])
        require(parent.resolve() != root.resolve(), "Preflight migration cycle")
        root = parent
    else:
        raise ValueError("Too many preflight migration ancestors")
    path = root / "preflight.json"
    if not path.exists():
        return {"status": "missing", "origin_chain": origins}
    value = read(path)
    require(len(value["device_wide_peak_mib"]) == 4 and len(value["pytorch_peak"]) == 4, "Expected four-device preflight peaks")
    result = {"status": "recorded_separately", "source": identity(path), "origin_chain": origins,
              "original_version": root.parent.name, "device_wide_peak_mib": value["device_wide_peak_mib"],
              "pytorch_peak": value["pytorch_peak"], "started_unix": value.get("started_unix"), "seconds": value.get("seconds"),
              "producer_memory_note": value.get("memory_note"),
              "note": "Separate preflight record; PyTorch counters begin after weight loading. Device-wide peaks include all CUDA contexts. Inherited copies are not new preflight executions."}
    for phase in ("pilot", "formal"):
        copy = Path(run_root) / phase / "preflight.json"
        if copy.exists():
            record = identity(copy)
            require(record["sha256"] == result["source"]["sha256"], "Inherited preflight bytes differ from origin")
            result[phase + "_inherited_copy"] = record
    inheritance = Path(run_root) / "formal/pilot-inheritance.json"
    if inheritance.exists():
        require(read(inheritance)["preflight_sha256"] == result["source"]["sha256"], "Formal preflight inheritance SHA differs")
        result["formal_inheritance_pointer"] = identity(inheritance)
    return result


def build_report(csv_path, run_root, claim, report, *, observed_unix=None, completion_observation=None, monitor_summary=None, monitor_stopped=False):
    samples, source = snapshot_csv(csv_path, monitor_stopped)
    observed = time.time() if observed_unix is None else finite(observed_unix)
    summary_path = Path(monitor_summary) if monitor_summary else Path(csv_path).with_name("gpu-all-stages-summary.json")
    period = 5.0
    summary = {"status": "missing", "period_note": "Assumed producer target 5 seconds; actual gaps are reported."}
    if summary_path.exists():
        summary = identity(summary_path)
        period = finite(read(summary_path)["sample_period_seconds"])
    require(period > 0, "Invalid monitor sampling period")
    garden = gardening_window(claim, report, observed, completion_observation)
    stages = [stage_report(run_root, name, samples, period, garden) for name in STAGES]
    # If an alias points to an already-accounted physical receipt, keep its proof
    # and no second set of samples/duration. No total is inferred from aliases.
    accounted = {}
    for stage in stages:
        if stage["status"] != "complete":
            continue
        key = stage["receipt_chain"][-1]["path"], stage["source_receipt_sha256"]
        if key in accounted:
            stage.update(status="alias", alias_of=accounted[key])
            stage.pop("gpu", None)
        else:
            accounted[key] = stage["name"]
    return {"format_version": "synthetic-agent-resource-appendix-v1",
            "report_status": "final_monitor_stopped_by_root" if monitor_stopped else "provisional_prefix_metadata_only",
            "observed_unix": observed, "observed_utc": utc(observed),
            "csv": source, "monitor_summary": summary, "nominal_sample_period_seconds": period,
            "gardening_possible_concurrency": garden, "preflight": preflight_report(run_root), "stages": stages,
            "limitations": ["Sampled global device peaks are lower bounds on unobserved continuous maxima; include every CUDA context.",
                "Prefix/tail gaps and gaps over twice the nominal sampling period are explicit; no interpolation or missing-peak imputation.",
                "FIT+DEV extraction is one combined CLI window; individual candidates have no absolute timing windows and receive no per-candidate peaks.",
                "Garden overlap is a conservative claim-to-report interval including CPU check/report, never exact inference time or pure-model memory.",
                "Wall time is stage-marker elapsed time, not GPU kernel time or utilization-weighted GPU hours; unfinished stages remain pending."]}


def markdown(value):
    garden = value["gardening_possible_concurrency"]
    garden_note = (garden.get("start_utc", "—") + " → " + garden.get("end_utc", "—") + "；" + garden["status"])
    title = "# GPU 资源附录" if value["csv"]["identity_is_final"] else "# GPU 资源临时元数据（不是最终原始 CSV 身份）"
    lines = [title, "", "观察时间：" + value["observed_utc"], "",
             "峰值均为设备全局离散采样值，包含其他 CUDA contexts；不能当作连续真实峰值或纯模型占用。", "",
             "CSV：`" + value["csv"]["path"] + "`；SHA256：`" + value["csv"]["sha256"] + "`；字节：" + str(value["csv"]["snapshot_bytes"]) + "；范围：" + value["csv"]["sha256_scope"] + "。", "",
             "监测首末样本：" + str(value["csv"]["first_sample_utc"]) + " → " + str(value["csv"]["last_sample_utc"]) + "。", "",
             "Gardening 可能并发：" + garden_note + "（保守范围，含 CPU check/report）。", "",
             "| 阶段 | 原版本 | UTC 起止 | 秒 | GPU0/1/2/3 采样峰值 MiB | 覆盖/并发说明 |",
             "|---|---|---|---:|---|---|"]
    for s in value["stages"]:
        if s["status"] != "complete":
            lines.append(f"| {s['name']} | — | {s['status']} | — | — | {s.get('reason', s.get('alias_of', ''))} |")
            continue
        peaks = "/".join(str(s["gpu"][str(g)]["sampled_memory_peak_mib"]) for g in range(4))
        notes = []
        for g in range(4):
            d = s["gpu"][str(g)]
            notes.append(f"GPU{g}: n={d['sample_count']}, 前缺={d['leading_unobserved_seconds']:.3f}s, 后缺={d['trailing_unobserved_seconds'] if d['trailing_unobserved_seconds'] is None else round(d['trailing_unobserved_seconds'],3)}s, partial={d['partial_coverage']}")
        if s["reused_without_new_execution"]:
            notes.append("复用原执行，未重复计时")
        if s["possible_gardening_overlap_seconds"]:
            notes.append(f"可能叠加 gardening {s['possible_gardening_overlap_seconds']:.3f}s（含 CPU check/report）")
        lines.append(f"| {s['name']} | {s['original_version']} | {s['start_utc']} → {s['end_utc']} | {s['seconds']:.3f} | {peaks} | {'; '.join(notes)} |")
    lines += ["", "各卡首末样本、内部采样缺口、峰值发生时间及 receipt/migration 原始路径和 SHA 见配套 JSON。", ""]
    p = value["preflight"]
    if p["status"] == "recorded_separately":
        lines += ["独立 preflight：设备全局峰值 MiB=" + str(p["device_wide_peak_mib"]) + "；PyTorch allocated/reserved 原始字节见 JSON。计数从权重加载后开始，继承文件不表示重跑。", ""]
    lines += ["- " + item for item in value["limitations"]]
    return "\n".join(lines) + "\n"


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=BASE / "runs-v5/campaign/gpu-all-stages.csv")
    p.add_argument("--run-root", type=Path, default=BASE / "runs-v7")
    p.add_argument("--gardening-claim", type=Path, default=Path("/root/role-retraining-20261026/gardening-first-trigger/dispatch-claim.json"))
    p.add_argument("--gardening-report", type=Path, default=BASE / "diagnostics/gardening-pilot-vs-first-formal/report-complete.json")
    p.add_argument("--gardening-completion-observation", type=Path,
                   help="Root JSON: observed_unix, report_complete_sha256, actual_infer_processes_absent=true")
    p.add_argument("--monitor-summary", type=Path)
    p.add_argument("--monitor-stopped", action="store_true", help="Root declares monitor stopped; require stable complete CSV and label identity final. Does not stop any process.")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    value = build_report(a.csv, a.run_root, a.gardening_claim, a.gardening_report,
                         completion_observation=a.gardening_completion_observation, monitor_summary=a.monitor_summary,
                         monitor_stopped=a.monitor_stopped)
    value["implementation_sha256"] = identity(Path(__file__))["sha256"]
    write_atomic(a.output_dir / "resource-appendix.json", json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_atomic(a.output_dir / "resource-appendix.md", markdown(value))


if __name__ == "__main__":
    main()
