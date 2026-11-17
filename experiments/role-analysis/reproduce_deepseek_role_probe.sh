#!/usr/bin/env bash
# Resume the pinned native DeepSeek-V4-Flash-0731 role-probe experiment.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)
RUN_DIR="${SCRIPT_DIR}/outputs/deepseek-v4-flash-0731-role-probe-20261115"
MODEL_DIR=/scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731
OLD_MODEL_DIR=/scratch/wz55191/models/gpt-oss-120b
REPO_PYTHON="${REPO_ROOT}/.venv/bin/python"
MODEL_PYTHON=/scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python
OFFICIAL_REPO=deepseek-ai/DeepSeek-V4-Flash-0731
OFFICIAL_REVISION=7872f01b1d1fe23eabc4c98b48bffcef5a386062
PERFORMANCE_CACHE_DIR=/home/wz55191/.cache/triton
PERFORMANCE_VALIDATION="${RUN_DIR}/performance-cache-gpu-validation.json"
CHECK_ONLY=0

usage() {
    cat <<'USAGE'
Usage: reproduce_deepseek_role_probe.sh [--check-only | --help]

Without arguments: verify the existing official model and pinned environment,
verify/reuse recovered source data and prepared inputs, prepare native kernel
performance caches from measured short references, resume the four-H100 pipeline,
then produce the load-integrity audit, independently audited report, and plots.

--check-only  Read-only model/cache/data/environment checks; never loads a model.
--help        Show this usage text.

Uses the existing official model at /scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731.
The model revision, output directory, Python runtimes, and package versions are
pinned in this script. Missing or changed official model files cause an error;
model downloading and package installation are separate setup operations.
All stages stop immediately on failure. Existing feature, head-scan, and probe
checkpoints are resumed by run_deepseek_role_probe.py. TEST is released only
after the selected probe is frozen.

Cold native-performance cache setup must be run separately on an available
H100 before this script. Missing measured eight-token references stop execution;
the script does not bootstrap them with the runner's long preflight.

From the repository root, generate only real S=8*6 native-kernel measurements:
  RUN=experiments/role-analysis/outputs/deepseek-v4-flash-0731-role-probe-20261115
  PY=/scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python
  CACHE=/home/wz55191/.cache/triton
  CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR="$CACHE" "$PY" \
    experiments/role-analysis/bootstrap_deepseek_performance_reference.py \
    --cache-dir "$CACHE" --output "$RUN/performance-cache-reference-bootstrap.json"

Then validate native cache dispatch against direct same-configuration outputs:
  CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR="$CACHE" "$PY" \
    experiments/role-analysis/prepare_deepseek_performance_cache.py \
    --data-dir "$RUN/data" --cache-dir "$CACHE" --reference-tokens 8 \
    --protected-tokens 8309 --validate-gpu \
    --output "$RUN/performance-cache-gpu-validation.json"

The first command is synthetic native-kernel calibration, not a model forward.
The second uses bounded fixed-configuration checks at 8/257/1029/8309 tokens;
it never performs an 8309-token Bayesian search. Normal reproduction fills only
absent cache entries, preserves true workload sizes, and invents no timings.
USAGE
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        --check-only) CHECK_ONLY=1 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
fi
[[ $# -eq 0 ]] || { usage >&2; exit 2; }
[[ -x "${REPO_PYTHON}" ]] || { echo "Missing repository Python: ${REPO_PYTHON}" >&2; exit 1; }
[[ -x "${MODEL_PYTHON}" ]] || { echo "Missing persistent model Python: ${MODEL_PYTHON}" >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
cd -- "${REPO_ROOT}"

# Reading small metadata and file stats never initializes a CUDA context.
"${REPO_PYTHON}" - "${RUN_DIR}" "${MODEL_DIR}" "${OFFICIAL_REPO}" "${OFFICIAL_REVISION}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

run, model = Path(sys.argv[1]), Path(sys.argv[2])
repo_id, revision = sys.argv[3:5]
manifest = json.loads((run / "download-manifest.json").read_text())
assert manifest["repo_id"] == repo_id and manifest["revision"] == revision, "Official repository/revision changed"
assert Path(manifest["destination"]).resolve() == model.resolve(), "Model destination changed"
assert len(manifest["files"]) == 74, "Expected the complete 74-file official release"
verified = json.loads((run / "download-file-verification.json").read_text())
expected = sorted(({"path": f["path"], "size": int(f["size"]),
                    "lfs_sha256": (f.get("lfs") or {}).get("sha256")} for f in manifest["files"]),
                  key=lambda f: f["path"])
binding = {"repo_id": repo_id, "revision": revision, "expected_files": expected}
assert verified["status"] == "passed" and verified["binding"] == binding, "Official file verification is incomplete or stale"
rows = {f["path"]: f for f in verified["files"]}
assert set(rows) == {f["path"] for f in manifest["files"]}, "Verified official file inventory changed"
for f in manifest["files"]:
    path, row = model / f["path"], rows[f["path"]]
    stat = path.stat()
    assert (row["size"], row["mtime_ns"], row["inode"]) == (stat.st_size, stat.st_mtime_ns, stat.st_ino), f"Stale SHA cache: {path}"
    assert stat.st_size == f["size"], f"Incomplete official model file: {path}"
    if f.get("lfs"):
        assert row["sha256"] == f["lfs"]["sha256"] and row["official_lfs_verified"], f"Official LFS mismatch: {path}"
data_path = run / "data/manifest.json"
if data_path.exists():
    data = json.loads(data_path.read_text())
    assert data["roles"] == ["system", "user", "cot", "assistant", "tool"], "Native role mapping changed"
    assert data["protocol"]["model_path"] == str(model.resolve()), "Prepared data belongs to another model"
    for name, digest in data["files_sha256"].items():
        path = run / "data" / name
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        assert observed == digest, f"Prepared data changed: {path}"
    print(json.dumps({"stage": "verified_existing_inputs", "records": data["records"], "roles": data["roles"]}), flush=True)
print(json.dumps({"stage": "verified_existing_official_model", "revision": revision,
                  "files": len(rows), "bytes": verified["verified_bytes"]}), flush=True)
PY

CUDA_VISIBLE_DEVICES='' "${MODEL_PYTHON}" - "${RUN_DIR}/environment.json" "${MODEL_PYTHON}" <<'PY'
import importlib
import json
from pathlib import Path
import sys

snapshot = json.loads(Path(sys.argv[1]).read_text())
assert Path(snapshot["executable"]) == Path(sys.argv[2]), "Persistent runtime changed"
expected = {"torch": "2.9.1", "triton": "3.5.1", "transformers": "5.23.0",
            "kernels": "0.18.2", "tokenizers": "0.23.2"}
actual = {}
for name, version in expected.items():
    module = importlib.import_module(name)
    observed = str(module.__version__)
    recorded = snapshot["packages"][name]
    assert observed.split("+", 1)[0] == version, f"Runtime {name} version changed: {observed}"
    assert observed == recorded["version"], f"Runtime {name} differs from environment.json"
    assert Path(module.__file__).resolve() == Path(recorded["path"]).resolve(), f"Runtime {name} import location changed"
    actual[name] = observed
print(json.dumps({"stage": "verified_persistent_runtime", "python": sys.executable, "packages": actual}), flush=True)
PY

# This is deliberately a CPU inspection of existing evidence. The performance
# helper's own dry run queries CUDA, so --check-only must never invoke it.
"${REPO_PYTHON}" - "${PERFORMANCE_VALIDATION}" "${PERFORMANCE_CACHE_DIR}" "${RUN_DIR}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

report_path, cache_dir, run_dir = map(Path, sys.argv[1:4])
def require(condition, message):
    if not condition:
        raise SystemExit(message + "\nCold-cache setup: run the dedicated eight-token "
                         "bootstrap_deepseek_performance_reference.py and bounded --validate-gpu "
                         "commands shown by this script's --help. Do not use the runner's default long preflight as bootstrap.")
require(report_path.is_file(), "Measured short-reference validation report is missing")
report = json.loads(report_path.read_text())
require(report.get("status") == "validation_passed" and report.get("validation", {}).get("status") == "passed",
        "Native performance-reference mechanism validation is incomplete")
identity = report["native_identity"]
require(identity["torch"].split("+", 1)[0] == "2.9.1" and identity["triton"] == "3.5.1",
        "Measured reference runtime differs from the pinned runtime")
require(identity["target"] == {"backend": "cuda", "arch": 90, "warp_size": 32},
        "Measured reference is not from an H100 SM90")
for kind in ("kernel", "tuner"):
    source = Path(identity[f"{kind}_source_path"])
    require(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == identity[f"{kind}_source_sha256"],
            f"Original native {kind} source changed after mechanism validation")
checks = report["validation"].get("checks", [])
require({(c["projection"], c["tokens"]) for c in checks}
        == {(p, n) for p in ("gate_up", "down") for n in (8, 257, 1029, 8309)},
        "Native reference validation does not cover the actual short and longest shapes")
require(all(c["native_disk_cache_hit"] and not c["search_called"] and c["finite"] and c["bitexact"]
            and c["maximum_absolute_difference"] == 0 for c in checks),
        "Native reference dispatch/direct-output equivalence did not pass")
sources = [s for s in report["plan"]["sources"].values() if s["source_tokens"] == 8]
require({s["projection"] for s in sources} == {"gate_up", "down"}, "Both measured eight-token native references are required")
dtypes = ["torch.bfloat16", "torch.uint8", "torch.bfloat16", "torch.uint8", "torch.int64"]
for source in sources:
    n, k = (4096, 4096) if source["projection"] == "gate_up" else (4096, 2048)
    require(source["source_key"] == [n, k, 48, *dtypes], "Short reference has a different full tuning key")
    path = Path(source["path"])
    require(path.resolve().is_relative_to(cache_dir.resolve()), "Short reference is outside the selected native cache")
    require(path.is_file(), f"Measured eight-token cache is missing: {path}")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == source["sha256"] and len(raw) == source["bytes"],
            "Measured eight-token reference changed after validation")
    entry = json.loads(raw)
    timings = entry.get("timings", [])
    measured = [ms for config, ms in timings if config == entry.get("best")
                and isinstance(ms, (int, float)) and math.isfinite(ms) and ms > 0]
    require(len(timings) > 1 and measured and entry["best"] == source["best"]
            and source["measured_candidates"] == len(timings),
            "Reference does not contain a genuine measured multi-candidate native winner")
manifest = json.loads((run_dir / "data/manifest.json").read_text())
for bound in report["prepared_inputs"]:
    path = Path(bound["path"])
    require(path.resolve().parent == (run_dir / "data").resolve()
            and manifest["files_sha256"].get(path.name) == bound["sha256"],
            "Native mechanism-validation inputs differ from the prepared experiment")
print(json.dumps({"stage": "verified_measured_eight_token_references", "source_projections": [s["projection"] for s in sources],
                  "mechanism_validation": "disk-cache hits and bitexact same-config outputs at 8/257/1029/8309",
                  "cuda_initialized": False}), flush=True)
PY

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo "Read-only checks passed; no model was loaded."
    exit 0
fi

# Prevent a second reproduction process or an already-running direct pipeline.
command -v flock >/dev/null
exec 9>"${RUN_DIR}/.reproduce.lock"
flock --nonblock 9 || { echo "Another reproduction process holds the run lock." >&2; exit 1; }
"${REPO_PYTHON}" - "${SCRIPT_DIR}/run_deepseek_role_probe.py" <<'PY'
from pathlib import Path
import os
import sys

target = Path(sys.argv[1]).name
active = []
for path in Path("/proc").glob("[0-9]*/cmdline"):
    if int(path.parent.name) == os.getpid():
        continue
    try:
        args = path.read_bytes().split(b"\0")
        if any(Path(arg.decode(errors="replace")).name == target for arg in args if arg):
            active.append(int(path.parent.name))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
assert not active, f"A role-probe pipeline is already running (PIDs {active}); wait for it before resuming"
PY

# Existing manifests make these stages validate/reuse their exact source bytes.
"${REPO_PYTHON}" "${SCRIPT_DIR}/deepseek_role_probe_data.py" restore-articles --output "${RUN_DIR}"
"${REPO_PYTHON}" "${SCRIPT_DIR}/deepseek_role_probe_data.py" prepare \
    --output "${RUN_DIR}" --model "${MODEL_DIR}" --old-model "${OLD_MODEL_DIR}" --max-input-tokens 16384

# No pipeline is active here. Including 8309 is safe because the independent
# bounded mechanism validation covers that true shape, and existing entries
# are atomically protected against overwrite. No target timings are fabricated.
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR="${PERFORMANCE_CACHE_DIR}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${MODEL_PYTHON}" "${SCRIPT_DIR}/prepare_deepseek_performance_cache.py" \
    --data-dir "${RUN_DIR}/data" --cache-dir "${PERFORMANCE_CACHE_DIR}" \
    --reference-tokens 8 --protected-tokens '' --apply \
    --output "${RUN_DIR}/performance-cache-reproduce-applied.json"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TRITON_CACHE_DIR="${PERFORMANCE_CACHE_DIR}" \
    "${MODEL_PYTHON}" -u "${SCRIPT_DIR}/run_deepseek_role_probe.py" \
    --stage all --model "${MODEL_DIR}" --output "${RUN_DIR}" 2>&1 | tee -a "${RUN_DIR}/run.log"
"${REPO_PYTHON}" "${SCRIPT_DIR}/audit_deepseek_load.py" \
    --run-dir "${RUN_DIR}" --model-dir "${MODEL_DIR}" --minimum-preflight-tokens 8309 --require-passed
"${MODEL_PYTHON}" "${SCRIPT_DIR}/deepseek_role_probe_report.py" --output "${RUN_DIR}" --plots
"${REPO_PYTHON}" - "${RUN_DIR}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
assert (root / "runtime-final.json").is_file(), "Complete model/probe pipeline receipt is missing"
load_audit = json.loads((root / "load-integrity-audit.json").read_text())
assert load_audit["status"] == "passed", "Final backbone loading/preflight integrity audit did not pass"
validation = json.loads((root / "independent-validation.json").read_text())
assert validation["status"] == "passed" and validation["complete"], "Final independent report audit did not pass"
assert validation.get("plots", {}).get("status") == "created", "Final report plots were not created"
print(f"Reproduction completed: {root / 'REPORT.zh-CN.md'}")
PY
