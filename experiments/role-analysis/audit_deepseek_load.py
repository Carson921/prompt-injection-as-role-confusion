"""Audit DeepSeek V4 Flash loading evidence without importing Torch or a model.

Run repeatedly while loading/preflight is in progress. Incomplete evidence yields
``running``; only a completed load and a passed GPU preflight yield ``passed``.
Every read is bound to a byte count and SHA256, including the full log prefix
visible at the start of that read. Existing artifacts and model files are read
only. The sole output is an atomically replaced load-integrity-audit.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

LAYERS, EXPERTS, HEADS, HEAD_DIM = 43, 256, 64, 512
EXPECTED_FUSED_KEYS = 1736
TOP_KEYS = {"embed.weight", "head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"}
EXPERT_KEY = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
PROGRESS = re.compile(r"Loading weights:\s*\d+%[^\r\n]*?\b(\d+)/(\d+)\b")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def snapshot(path):
    """Read exactly the complete file prefix present at open, even if it grows."""
    path = Path(path).resolve()
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            remaining, chunks, digest = before.st_size, [], hashlib.sha256()
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"File shrank while auditing: {path}")
                chunks.append(block)
                digest.update(block)
                remaining -= len(block)
            after = os.fstat(stream.fileno())
        return b"".join(chunks), {"path": str(path), "exists": True,
                                  "bytes": before.st_size, "sha256": digest.hexdigest(),
                                  "size_after_read": after.st_size,
                                  "appended_during_read": after.st_size > before.st_size}
    except FileNotFoundError:
        return None, {"path": str(path), "exists": False}


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, condition, detail=None, *, pending=False):
        state = "passed" if condition else ("running" if pending else "failed")
        self.items.append({"name": name, "status": state, "detail": detail})
        return bool(condition)

    @property
    def status(self):
        states = {item["status"] for item in self.items}
        return "failed" if "failed" in states else ("running" if "running" in states else "passed")


def index_inventory(index, config):
    """Derive expected loaded keys from the official per-expert checkpoint keys."""
    keys = set(index["weight_map"])
    mtp = {key for key in keys if key.startswith("mtp.")}
    backbone = keys - mtp
    groups, routed_keys, layer_counts = defaultdict(set), set(), Counter()
    other_top = set()
    for key in backbone:
        if key.startswith("layers."):
            layer_counts[int(key.split(".", 2)[1])] += 1
        else:
            other_top.add(key)
        match = EXPERT_KEY.fullmatch(key)
        if match:
            layer, expert, projection, kind = match.groups()
            groups[(int(layer), projection, kind)].add(int(expert))
            routed_keys.add(key)
    bad_groups = []
    for layer in range(LAYERS):
        for projection in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                observed = groups.get((layer, projection, kind), set())
                if observed != set(range(EXPERTS)):
                    bad_groups.append({"layer": layer, "projection": projection, "kind": kind,
                                       "missing_experts": sorted(set(range(EXPERTS)) - observed),
                                       "extra_experts": sorted(observed - set(range(EXPERTS)))})
    out_of_range_groups = [list(group) for group in groups if group[0] not in range(LAYERS)]
    # w1/w3 weights and scales merge into gate_up, while w2 becomes down: four
    # target tensors per layer replace 256*3*2 original expert tensors.
    fused_keys = len(backbone) - len(routed_keys) + LAYERS * 4
    geometry = [config.get(name) for name in (
        "num_hidden_layers", "n_routed_experts", "num_attention_heads", "head_dim")]
    valid = (geometry == [LAYERS, EXPERTS, HEADS, HEAD_DIM]
             and set(layer_counts) == set(range(LAYERS)) and other_top == TOP_KEYS
             and not bad_groups and not out_of_range_groups
             and len(routed_keys) == LAYERS * EXPERTS * 3 * 2
             and fused_keys == EXPECTED_FUSED_KEYS)
    return {"valid": valid, "geometry": geometry, "checkpoint_keys": len(keys),
            "ignored_mtp_keys": len(mtp), "backbone_checkpoint_keys": len(backbone),
            "routed_expert_checkpoint_keys": len(routed_keys),
            "fused_expert_target_keys": LAYERS * 4, "expected_loaded_keys": fused_keys,
            "fusion_equation": f"{len(backbone)} - {len(routed_keys)} + {LAYERS * 4} = {fused_keys}",
            "layers": len(layer_counts), "layer_key_counts": dict(sorted(layer_counts.items())),
            "top_level_backbone_keys": sorted(other_top),
            "incomplete_expert_groups": bad_groups, "out_of_range_groups": out_of_range_groups}


def inspect_log(text, expected=EXPECTED_FUSED_KEYS):
    text = ANSI.sub("", text)
    events, progresses, starts = [], [], []
    for number, line in enumerate(text.splitlines()):
        for match in PROGRESS.finditer(line):
            progresses.append({"line": number, "loaded": int(match[1]), "total": int(match[2])})
        if line.lstrip().startswith("{"):
            try:
                event = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and "stage" in event:
                events.append((number, event))
                if event["stage"] == "loading_deepseek_v4":
                    starts.append(number)
    start = starts[-1] if starts else None
    current = [p for p in progresses if start is None or p["line"] >= start]
    loaded_events = [(line, event) for line, event in events
                     if event["stage"] == "deepseek_v4_loaded" and start is not None and line > start]
    loaded = loaded_events[-1] if loaded_events else None
    # A later classifier/scan failure does not retrospectively change successful
    # loading. Check loading exceptions through its completion event only.
    lines = text.splitlines()
    loading_lines = lines[(start or 0):(loaded[0] + 1 if loaded else len(lines))]
    diagnostics = [{"line": number, "text": line[:1000]}
                   for number, line in enumerate(lines)
                   if re.search(r"\bLOAD REPORT\b|\bMISSING\b|\bMISMATCH\b|\bCONVERSION\b|\bUNEXPECTED\b", line)]
    errors = [line[:1000] for line in loading_lines
              if "Traceback (most recent call last)" in line or "Error(s) in loading state_dict" in line]
    finished = bool(current and current[-1]["loaded"] == current[-1]["total"] == expected)
    return {"loading_attempts": len(starts), "loading_start_seen": start is not None,
            "latest_progress": current[-1] if current else None,
            "progress_totals_match": all(p["total"] == expected and p["loaded"] <= expected for p in current),
            "weights_progress_complete": finished, "loaded_event_seen": loaded is not None,
            "loaded_event": loaded[1] if loaded else None,
            "loading_diagnostics": diagnostics, "loading_errors": errors,
            "preflight_forward_events": [event for line, event in events
                                         if event["stage"] == "preflight_forward"
                                         and loaded is not None and line > loaded[0]]}


def _cuda_devices(mapping):
    devices = set()
    for value in mapping:
        if isinstance(value, bool):
            return None
        value = str(value)
        if value.isdigit():
            devices.add(f"cuda:{value}")
        elif re.fullmatch(r"cuda:[0-3]", value):
            devices.add(value)
        else:
            return None
    return devices


def check_backbone(checks, data, config, prefix):
    for key in ("parameters_frozen", "all_parameters_on_cuda", "native_packed_fp4_fp8"):
        checks.add(prefix + key, data.get(key) is True)
    q = data.get("quantization_config", {})
    checks.add(prefix + "native_quantization", isinstance(q, dict) and q.get("quant_method") == "fp8"
               and q.get("expert_dtype") == "fp4" and q.get("scale_fmt") == "ue8m0"
               and q.get("dequantize") is False and q.get("weight_block_size") == [128, 128], q)
    checks.add(prefix + "expert_backend", data.get("expert_backends") == ["FP8Experts"]
               and data.get("experts_implementation") == "batched_mm")
    checks.add(prefix + "attention_backend", data.get("attention_implementation") == "deepseek_role_probe_tiled_eager")
    counts = data.get("dtype_bytes", {})
    physical_expert_bytes = LAYERS * EXPERTS * 3 * config["hidden_size"] * config["moe_intermediate_size"] // 2
    checks.add(prefix + "packed_expert_byte_count", counts.get("torch.int8") == physical_expert_bytes,
               {"expected": physical_expert_bytes, "observed": counts.get("torch.int8")})
    checks.add(prefix + "fp8_and_native_scales", counts.get("torch.float8_e4m3fn", 0) > 0
               and counts.get("torch.float8_e8m0fnu", 0) >= physical_expert_bytes // 16)
    by_device = data.get("parameter_bytes_by_device_dtype", {})
    checks.add(prefix + "four_cuda_devices", set(by_device) == {f"cuda:{i}" for i in range(4)})
    checks.add(prefix + "byte_totals", data.get("parameter_bytes_total", 0) > 0
               and data.get("parameter_bytes_total") == sum(counts.values())
               == sum(sum(values.values()) for values in by_device.values()))


def audit(run_dir, model_dir=None, minimum_preflight_tokens=8309):
    run_dir = Path(run_dir).resolve()
    checks, inputs, values = Checks(), {}, {}

    def read_json(label, path, *, pending=False):
        raw, binding = snapshot(path)
        inputs[label] = binding
        if raw is None:
            checks.add(label + "_present", False, pending=pending)
            return None
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            checks.add(label + "_valid_json", False, str(error), pending=pending)
            return None
        checks.add(label + "_present", True)
        values[label] = result
        return result

    run_config = read_json("run_config", run_dir / "config.json", pending=True)
    if model_dir is None and run_config:
        model_dir = run_config.get("model")
    inventory = log_info = None
    if model_dir is None:
        checks.add("model_path_available", False, pending=True)
    else:
        model_dir = Path(model_dir).resolve()
        config = read_json("official_config", model_dir / "config.json")
        index = read_json("official_index", model_dir / "model.safetensors.index.json")
        if config is not None and index is not None:
            try:
                inventory = index_inventory(index, config)
                checks.add("official_backbone_key_inventory", inventory["valid"], inventory["fusion_equation"])
            except (KeyError, TypeError, ValueError) as error:
                checks.add("official_backbone_key_inventory", False, str(error))
    raw_log, binding = snapshot(run_dir / "run.log")
    inputs["run_log"] = binding
    checks.add("run_log_present", raw_log is not None, pending=True)
    if raw_log is not None:
        log_info = inspect_log(raw_log.decode("utf-8", errors="replace"))
        checks.add("complete_log_start", log_info["loading_start_seen"], pending=True)
        checks.add("load_diagnostics_absent", not log_info["loading_diagnostics"], log_info["loading_diagnostics"])
        checks.add("load_exceptions_absent", not log_info["loading_errors"], log_info["loading_errors"])
        checks.add("observed_progress_total", log_info["progress_totals_match"], log_info["latest_progress"])
        checks.add("weights_progress_complete", log_info["weights_progress_complete"], pending=True)
        checks.add("loaded_event_present", log_info["loaded_event_seen"], pending=True)

    backend = read_json("model_backend", run_dir / "model_backend.json", pending=True)
    preflight = read_json("preflight", run_dir / "preflight.json", pending=True)
    if backend is not None and "official_config" in values and "official_index" in values:
        config = values["official_config"]
        check_backbone(checks, backend, config, "backend_")
        checks.add("backend_config_binding", backend.get("config_sha256") == inputs["official_config"]["sha256"])
        checks.add("backend_index_binding", backend.get("index_sha256") == inputs["official_index"]["sha256"])
        checks.add("backend_model_path_binding", Path(backend.get("model_path", "")).resolve() == model_dir)
        checks.add("backend_geometry", [backend.get(k) for k in ("layers", "heads_per_layer", "head_dim")]
                   == [LAYERS, HEADS, HEAD_DIM])
        checks.add("head_capture", backend.get("capture") == "attention-head-output-before-o_a_proj"
                   and backend.get("capture_after_inverse_rope") is True)
        checks.add("query_tiling", backend.get("attention_query_tile_size") == 256)
        checks.add("four_gpu_device_map", _cuda_devices(backend.get("device_map", {}).values())
                   == {f"cuda:{i}" for i in range(4)})
        _, inputs["backend_source"] = snapshot(Path(__file__).with_name("deepseek_role_probe_model.py"))
        checks.add("backend_source_binding", backend.get("backend_source_sha256") == inputs["backend_source"].get("sha256"))
        if log_info and log_info["loaded_event"]:
            checks.add("loaded_event_parameter_bytes", log_info["loaded_event"].get("parameter_bytes")
                       == backend.get("parameter_bytes_total"))
    if preflight is not None:
        checks.add("preflight_passed", preflight.get("status") == "passed", preflight.get("status"))
        checks.add("preflight_frozen_forward", preflight.get("use_cache") is False
                   and preflight.get("lm_head_skipped") is True and preflight.get("batch_size") == 1)
        forward_checks = preflight.get("checks", [])
        valid_forwards = len(forward_checks) == 2 and all(
            check.get("all_layers_captured") == LAYERS
            and isinstance(check.get("hook_output_bitexact"), bool)
            and math.isfinite(check.get("hook_output_max_abs_difference", float("inf")))
            and check.get("tokens", 0) > 0 for check in forward_checks)
        checks.add("preflight_all_layer_hooks", valid_forwards)
        logged_forwards = [] if log_info is None else log_info["preflight_forward_events"]
        complete_forward_log = len(logged_forwards) == len(forward_checks) == 2
        checks.add("preflight_completion_in_log", complete_forward_log, pending=True)
        if complete_forward_log:
            checks.add("preflight_log_artifact_agreement",
                       [{key: value for key, value in event.items() if key != "stage"}
                        for event in logged_forwards] == forward_checks)
        maximum = preflight.get("maximum_input_tokens", 0)
        checks.add("preflight_long_input", maximum >= minimum_preflight_tokens
                   and maximum == max((check.get("tokens", 0) for check in forward_checks), default=0),
                   {"required_minimum": minimum_preflight_tokens, "observed": maximum})
        if "official_config" in values:
            check_backbone(checks, preflight.get("backbone_audit", {}), values["official_config"], "preflight_")
        if backend is not None:
            for key in ("dtype_bytes", "parameter_bytes_total", "quantization_config",
                        "parameter_bytes_by_device_dtype", "experts_implementation", "attention_implementation"):
                checks.add("preflight_backend_agreement_" + key,
                           preflight.get("backbone_audit", {}).get(key) == backend.get(key))

    verified = read_json("weight_verification", run_dir / "weight-verification.json", pending=True)
    if verified is not None:
        checks.add("weight_verification_passed", verified.get("status") == "passed")
        verified_files = {item["path"]: item for item in verified.get("files", [])}
        for label, filename in (("official_config", "config.json"), ("official_index", "model.safetensors.index.json")):
            if label in inputs and inputs[label].get("exists"):
                checks.add("verified_" + label + "_binding",
                           verified_files.get(filename, {}).get("sha256") == inputs[label]["sha256"])
        if "official_index" in values:
            shards = set(values["official_index"]["weight_map"].values())
            rows = [verified_files.get(name, {}) for name in shards]
            checks.add("all_indexed_shards_official_sha_verified", all(
                row.get("official_lfs_verified") is True and row.get("size", 0) > 0
                and re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) for row in rows),
                {"indexed_shards": len(shards), "manifest_shards": sum(bool(row) for row in rows)})
            checks.add("verified_safetensors_byte_count", verified.get("safetensors_bytes")
                       == sum(row.get("size", 0) for row in rows))
    _, inputs["audit_source"] = snapshot(__file__)
    return {"format_version": "deepseek-load-integrity-audit-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": checks.status,
            "run_dir": str(run_dir), "model_dir": None if model_dir is None else str(model_dir),
            "checks": checks.items, "inputs": inputs, "index_inventory": inventory,
            "log_evidence": log_info,
            "evidence_scope": "Completed backbone loading and GPU preflight; checkpoint verification manifest is SHA-bound",
            "completion_rule": "All required checks passed; partial log or absent completion artifacts remain running"}


def write_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--minimum-preflight-tokens", type=int, default=8309)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-passed", action="store_true")
    args = parser.parse_args()
    result = audit(args.run_dir, args.model_dir, args.minimum_preflight_tokens)
    destination = args.output or args.run_dir / "load-integrity-audit.json"
    write_result(destination, result)
    print(json.dumps({"status": result["status"], "audit": str(destination.resolve()),
                      "failed_checks": [c["name"] for c in result["checks"] if c["status"] == "failed"],
                      "pending_checks": [c["name"] for c in result["checks"] if c["status"] == "running"]}))
    return 1 if result["status"] == "failed" else (2 if args.require_passed and result["status"] != "passed" else 0)


if __name__ == "__main__":
    sys.exit(main())
