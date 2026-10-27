"""Frozen MXFP4 GPT-OSS-120B runtime for the new six-role experiment.

No historical probe, head ranking, or natural conversation is imported here.
Every retained activation is a sampled article-body token before ``o_proj``.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np

ROLES = ["system", "developer", "user", "cot", "assistant", "tool"]
LAYERS, HEADS, HEAD_DIM, TOP_K = 36, 64, 64, 16
KERNEL_REVISION = "9fbc1b48e710ac8abf18ea2650ee690d52dc3859"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _get(config, key, default):
    """Runtime values can be top-level or inside the ``gpu`` section."""
    return config.get("gpu", {}).get(key, config.get(key, default))


def model_binding(model_dir):
    """Bind model/tokenizer bytes; shard content checks are recorded separately.

    The index, config, and all tokenizer assets are hashed. The physical weight
    file list and sizes detect a different layout; preflight can optionally hash
    every large shard using ``verify_weight_sha256``.
    """
    root = Path(model_dir).resolve()
    index = root / "model.safetensors.index.json"
    require(index.exists(), "GPT-OSS safetensors index is missing")
    raw = json.loads(index.read_text())
    names = sorted(set(raw["weight_map"].values()))
    assets = [p for p in root.iterdir() if p.is_file() and p.name != "weight-verification.json" and
              (p.name.endswith(".json") or p.name.endswith(".jinja") or p.name.endswith(".tiktoken"))]
    result = {
        "format": "gpt-oss-120b-native-mxfp4", "kernel_revision": KERNEL_REVISION,
        "assets_sha256": {p.name: digest(p) for p in sorted(assets)},
        "weights": [{"name": name, "bytes": (root / name).stat().st_size} for name in names],
    }
    require(len(names) == 15, "Expected the 15 official GPT-OSS-120B weight shards")
    require(all((root / name).is_file() for name in names), "A weight shard is missing")
    verification_path = root / "weight-verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        verified = {row["file"]: row for row in verification["shards"]}
        require(verification.get("all_15_shards_sha256_verified") is True and set(verified) == set(names),
                "Model verification does not cover exactly the 15 official shards")
        require(all(int(verified[name]["size"]) == (root / name).stat().st_size for name in names),
                "Verified weight size changed")
        result["repository"] = verification["repository"]
        result["revision"] = verification["revision"]
        result["weight_shards_sha256"] = {name: verified[name]["sha256"] for name in names}
        proof = {"repository": verification["repository"], "revision": verification["revision"],
                 "shards": [{"file": name, "size": int(verified[name]["size"]),
                             "sha256": verified[name]["sha256"]} for name in names]}
        result["weight_verification_identity_sha256"] = object_digest(proof)
    result["identity_sha256"] = object_digest(result)
    return result


def load_model(model_dir, run_dir, devices="0,1,2,3", cache_dir=None):
    """Load all weights on four A40 GPUs; never permit CPU weight offload."""
    if isinstance(devices, (list, tuple)):
        devices = ",".join(map(str, devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cache_dir:
        os.environ["HF_HOME"] = str(Path(cache_dir) / "huggingface")
    import torch
    from transformers import AutoModelForCausalLM, Mxfp4Config
    from transformers.integrations import hub_kernels
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    require(torch.cuda.device_count() == 4, "Exactly four CUDA devices must be visible")
    require(all("A40" in torch.cuda.get_device_name(i) for i in range(4)), "This protocol requires four A40s")
    original_get = getattr(hub_kernels, "_synthetic_original_get_kernel", hub_kernels.get_kernel)
    hub_kernels._synthetic_original_get_kernel = original_get

    def pinned(name, *args, **kwargs):
        if name == "kernels-community/gpt-oss-triton-kernels":
            kwargs["revision"] = KERNEL_REVISION
        return original_get(name, *args, **kwargs)

    hub_kernels.get_kernel = pinned
    original_validate = getattr(Mxfp4HfQuantizer, "_synthetic_original_validate", Mxfp4HfQuantizer.validate_environment)
    Mxfp4HfQuantizer._synthetic_original_validate = original_validate

    def strict(self, *args, **kwargs):
        original_validate(self, *args, **kwargs)
        require(not self.quantization_config.dequantize, "Refusing silent BF16 dequantization of MXFP4 weights")

    Mxfp4HfQuantizer.validate_environment = strict
    device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0,
                  "model.norm": 3, "lm_head": 3}
    device_map.update({f"model.layers.{i}": i // 9 for i in range(LAYERS)})
    started = time.monotonic()
    print(json.dumps({"stage": "loading_model", "model_dir": str(model_dir)}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=False), device_map=device_map,
        attn_implementation="eager", use_kernels=False,
    ).eval().requires_grad_(False)
    require(model.config.num_hidden_layers == LAYERS and model.config.num_attention_heads == HEADS,
            "Unexpected GPT-OSS model geometry")
    backends = sorted({type(layer.mlp.experts).__name__ for layer in model.model.layers})
    require(backends == ["Mxfp4GptOssExperts"], "Experts are not native MXFP4: " + str(backends))
    require(not model.hf_quantizer.quantization_config.dequantize, "Weights were dequantized")
    require(all(p.device.type == "cuda" for p in model.parameters()), "CPU/disk model weight offload is forbidden")
    require(not any(p.requires_grad for p in model.parameters()), "The backbone is not frozen")
    import accelerate.hooks
    import accelerate.utils.operations as operations
    original_send = getattr(operations, "_synthetic_original_send", operations.send_to_device)
    operations._synthetic_original_send = original_send

    def staged_send(tensor, device, non_blocking=False, skip_keys=None):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
            target = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
            if target.type == "cuda" and tensor.device != target:
                return tensor.cpu().to(target, non_blocking=False)
        return original_send(tensor, device, non_blocking=non_blocking, skip_keys=skip_keys)

    operations.send_to_device = staged_send
    accelerate.hooks.send_to_device = staged_send
    backend = {
        "model_binding": model_binding(model_dir), "expert_backends": backends,
        "device_map": model.hf_device_map, "load_path": str(Path(model_dir).resolve()),
        "gpu_transfer": "synchronous CPU staging between distinct CUDA devices; weights remain on GPUs",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "gpus": [torch.cuda.get_device_name(i) for i in range(4)],
        "load_seconds": time.monotonic() - started,
    }
    verification_path = Path(model_dir) / "weight-verification.json"
    if verification_path.exists():
        # Provenance preserves the exact audit artifact, while cache identity
        # ignores its incidental execution times, reuse flags, and local paths.
        backend["verification_artifact_sha256"] = digest(verification_path)
    atomic_json(Path(run_dir) / "model_backend.json", backend)
    model._synthetic_binding = backend["model_binding"]
    return model


class MemoryMonitor:
    """Device-wide memory includes CUDA contexts outside this process."""
    def __enter__(self):
        self.stop = threading.Event()
        self.peak = [0] * 4
        self.errors = []

        def poll():
            while not self.stop.is_set():
                try:
                    raw = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        text=True, timeout=10)
                    vals = [int(x.strip()) for x in raw.strip().splitlines()]
                    if len(vals) == 4:
                        self.peak = [max(a, b) for a, b in zip(self.peak, vals)]
                except Exception as error:
                    self.errors.append(str(error))
                self.stop.wait(.25)

        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=15)


def _forward(model, input_ids):
    import torch
    tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
    mask = torch.ones_like(tensor)
    return model.model(input_ids=tensor, attention_mask=mask, use_cache=False).last_hidden_state


class HeadCapture:
    """Collect only sampled body positions and optionally selected heads."""
    def __init__(self, positions, selected=None, receive=None):
        self.positions = np.asarray(positions, dtype=np.int64)
        self.selected = None if selected is None else np.asarray(selected, dtype=np.int64)
        self.layers = list(range(LAYERS)) if selected is None else sorted(set(self.selected // HEADS))
        self.receive = receive
        self.blocks = {}
        self.seen = set()

    def hook(self, layer):
        def capture(module, args):
            import torch
            require(layer not in self.seen, "A head hook was invoked twice")
            self.seen.add(layer)
            value = args[0].detach()
            require(value.ndim == 3 and value.shape[0] == 1 and value.shape[-1] == HEADS * HEAD_DIM,
                    "Unexpected pre-o_proj geometry")
            require(len(self.positions) and self.positions.min() >= 0 and self.positions.max() < value.shape[1],
                    "Body mask position is outside the input")
            x = value.reshape(-1, HEADS, HEAD_DIM).index_select(
                0, torch.as_tensor(self.positions, dtype=torch.long, device=value.device))
            if self.selected is not None:
                indices = np.flatnonzero(self.selected // HEADS == layer)
                x = x.index_select(1, torch.as_tensor(self.selected[indices] % HEADS, device=value.device))
            require(bool(torch.isfinite(x).all()), f"Nonfinite pre-o_proj activation at layer {layer}")
            if self.receive:
                self.receive(layer, x)
            else:
                require(bool((x.abs() <= np.finfo(np.float16).max).all()), "Activation cannot be represented in FP16")
                self.blocks[layer] = x.to(device="cpu", dtype=torch.float16).numpy()
            # Returning None is essential: observing never replaces activations.
        return capture

    @contextmanager
    def install(self, model):
        handles = [model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(self.hook(layer))
                   for layer in self.layers]
        try:
            yield self
            require(self.seen == set(self.layers), "A required pre-o_proj hook did not execute")
        finally:
            for handle in handles:
                handle.remove()

    def matrix(self):
        require(self.selected is not None, "All-head activations must not be retained as a full feature bank")
        result = np.empty((len(self.positions), len(self.selected), HEAD_DIM), dtype=np.float16)
        for layer, block in self.blocks.items():
            result[:, np.flatnonzero(self.selected // HEADS == layer)] = block
        return result.reshape(len(self.positions), -1)


def _check_hook(model, ids, selected=None):
    import torch
    with torch.inference_mode():
        before = _forward(model, ids).cpu()
        samples = sorted(set([0, len(ids) // 2, len(ids) - 1]))
        capture = HeadCapture(samples, selected=selected, receive=lambda layer, x: None)
        with capture.install(model):
            after = _forward(model, ids).cpu()
        require(bool(torch.isfinite(before).all()) and torch.equal(before, after), "Hooks changed model hidden states")
        logits = model.lm_head(after[:, -1:, :].to("cuda:3")).float().cpu()
        require(bool(torch.isfinite(logits).all()) and float(logits.std()) > .01, "Invalid model output logits")
        return {"input_tokens": len(ids), "hidden_states_bitexact": True,
                "last_token_logits_finite": True, "logit_std": float(logits.std()),
                "hook_layers": [int(x) for x in capture.layers]}


def preflight(model_dir, run_dir, config=None, devices="0,1,2,3", cache_dir=None, model=None):
    import torch
    from transformers import AutoTokenizer
    config = config or {}
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    binding = model_binding(model_dir)
    if _get(config, "verify_weight_sha256", False):
        hashes = {row["name"]: digest(Path(model_dir) / row["name"]) for row in binding["weights"]}
        atomic_json(run_dir / "weight-verification.json", {"sha256": hashes, "binding": binding})
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    short = tok.encode("<|start|>system<|message|>An ordinary article about forests.<|end|>"
                       "<|start|>developer<|message|>A second article passage.<|end|>"
                       "<|start|>user<|message|>The trees grow each spring.<|end|>", add_special_tokens=False)
    maximum = int(_get(config, "max_input_tokens", 4096))
    require(maximum == 4096, "The planned longest-input preflight must use 4096 tokens")
    body = tok.encode("Forests contain many kinds of plants and animals. ", add_special_tokens=False)
    prefix = tok.encode("<|start|>system<|message|>", add_special_tokens=False)
    suffix = tok.encode("<|end|>", add_special_tokens=False)
    long_ids = (prefix + (body * (maximum // len(body) + 1)))[:maximum - len(suffix)] + suffix
    require(len(long_ids) == maximum, "Longest-input construction failed")
    owned = model is None
    report = {"status": "running", "started_unix": time.time(), "model_binding": binding,
              "forward_batch_size": 1, "maximum_input_tokens": maximum}
    atomic_json(run_dir / "preflight.json", report)
    started = time.monotonic()
    try:
        with MemoryMonitor() as monitor:
            if owned:
                model = load_model(model_dir, run_dir, devices, cache_dir)
            for i in range(4):
                torch.cuda.reset_peak_memory_stats(i)
            checks = []
            for ids in [short, long_ids]:
                t = time.monotonic()
                check = _check_hook(model, ids)
                check["all_head_hook_check"] = True
                check["selected_head_hook"] = _check_hook(model, ids, selected=np.arange(TOP_K) * 137 % (LAYERS * HEADS))
                check["seconds"] = time.monotonic() - t
                checks.append(check)
                print(json.dumps({"stage": "preflight_forward", **check}), flush=True)
            # Exercise the actual all-head F64 statistic path at the maximum
            # possible number of retained positions. These artificial labels
            # are solely a workspace check and never enter head selection.
            moments = OnlineMoments()
            positions = np.arange(maximum, dtype=np.int64)
            labels = positions % len(ROLES)
            scan_capture = HeadCapture(positions, receive=lambda layer, x:
                                       moments.update(layer, x, labels, np.ones(maximum)))
            t = time.monotonic()
            with torch.inference_mode(), scan_capture.install(model):
                hidden = _forward(model, long_ids)
                require(bool(torch.isfinite(hidden).all()), "Nonfinite full-head statistic preflight output")
            arrays = moments.export()
            require(all(np.isfinite(array).all() for array in arrays.values()), "Nonfinite scan sufficient statistics")
            scan_workspace = {"input_tokens": maximum, "sampled_positions": maximum,
                              "all_2304_heads": True, "statistics_finite": True,
                              "statistics_bytes": sum(array.nbytes for array in arrays.values()),
                              "seconds": time.monotonic() - t,
                              "labels": "artificial modulo-six labels solely for workspace validation; discarded"}
            del moments, arrays, scan_capture, hidden
            peaks = [{"device": i, "allocated_bytes": torch.cuda.max_memory_allocated(i),
                      "reserved_bytes": torch.cuda.max_memory_reserved(i)} for i in range(4)]
        report.update(status="passed", checks=checks, pytorch_peak=peaks,
                      full_head_scan_workspace=scan_workspace,
                      device_wide_peak_mib=monitor.peak, memory_monitor_errors=monitor.errors,
                      seconds=time.monotonic() - started,
                      memory_note="Device-wide peaks include other CUDA contexts; the PyTorch counter begins after weight loading.")
        require(not monitor.errors, "Device-wide GPU memory monitoring failed")
        atomic_json(run_dir / "preflight.json", report)
        return report
    except Exception as error:
        report.update(status="failed", error=repr(error), seconds=time.monotonic() - started)
        atomic_json(run_dir / "preflight.json", report)
        raise
    finally:
        if owned:
            del model
            gc.collect()
            torch.cuda.empty_cache()


def _input_path(data_dir, phase, split):
    return Path(data_dir) / phase / f"inputs-{split}.jsonl"


def _records(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _label(target):
    value = target["label"]
    result = ROLES.index(value) if isinstance(value, str) else int(value)
    require(0 <= result < len(ROLES), "Unknown six-role label")
    if "role" in target:
        require(target["role"] == ROLES[result], "Role and integer label disagree")
    return result


def _checked_record(record):
    ids, targets = record["input_ids"], record["targets"]
    require(0 < len(ids) <= 4096, "Input exceeds the preflight limit")
    positions = [int(row["token_ix"]) for row in targets]
    require(len(positions) == len(set(positions)), "Duplicate supervised body positions")
    require(all(0 <= i < len(ids) for i in positions), "Invalid supervised token index")
    require(all(np.isfinite(float(row.get("weight", 1))) and float(row.get("weight", 1)) > 0 for row in targets),
            "Training row weights must be finite and positive")
    return record


def _source_split(cluster_id):
    number = int(hashlib.sha256(("head-inner-v1:" + str(cluster_id)).encode()).hexdigest()[:16], 16)
    return "inner" if number % 5 == 0 else "statistics"


def _scan_plan(path, maximum_rows, maximum_sources=None):
    """Fixed, role-stratified sample selected without inspecting activations."""
    import heapq
    eligible = sorted({str(r.get("cluster_id", r["source_id"])) for r in _records(path)},
                      key=lambda value: object_digest(["scan-source-v1", value]))
    if maximum_sources:
        eligible = eligible[:int(maximum_sources)]
    eligible = set(eligible)
    quota = max(1, int(maximum_rows) // len(ROLES))
    heaps = {(part, role): [] for part in ["statistics", "inner"] for role in range(len(ROLES))}
    clusters = {"statistics": set(), "inner": set()}
    for record_ix, record in enumerate(_records(path)):
        _checked_record(record)
        require(record["split"] == "fit", "Head scan attempted to use non-FIT data")
        cluster = str(record.get("cluster_id", record["source_id"]))
        if cluster not in eligible:
            continue
        part = _source_split(cluster)
        clusters[part].add(cluster)
        for target_ix, target in enumerate(record["targets"]):
            role = _label(target)
            key = f"{record.get('id', record_ix)}:{target['token_ix']}:scan-v1"
            priority = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
            item = (-priority, record_ix, target_ix)
            heap = heaps[(part, role)]
            if len(heap) < quota:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    require(not clusters["statistics"] & clusters["inner"], "Head scan source-cluster leakage")
    selected = {"statistics": {}, "inner": {}}
    counts = {}
    for (part, role), heap in heaps.items():
        require(heap, f"FIT internal {part} subset has no {ROLES[role]} body targets")
        counts[f"{part}:{ROLES[role]}"] = len(heap)
        for _, record_ix, target_ix in heap:
            selected[part].setdefault(record_ix, []).append(target_ix)
    for part in selected:
        selected[part] = {str(k): sorted(v) for k, v in selected[part].items()}
    inputs = {part: {"records": len(rows), "input_tokens": 0} for part, rows in selected.items()}
    for ix, record in enumerate(_records(path)):
        for part in selected:
            if str(ix) in selected[part]:
                inputs[part]["input_tokens"] += len(record["input_ids"])
    return {"records": selected, "counts": counts,
            "clusters": {key: sorted(value) for key, value in clusters.items()},
            "source_partition": "SHA256('head-inner-v1:'+cluster_id) modulo 5; 0 held out; 1..4 statistics",
            "sample_rule": "Lowest deterministic target hashes, with a separate equal six-role quota in each source partition",
            "maximum_rows_per_partition": int(maximum_rows), "inputs": inputs,
            "source_subset_rule": "Lowest SHA256(['scan-source-v1',cluster_id]); independent of features/labels",
            "maximum_sources": maximum_sources}


class OnlineMoments:
    """Pooled weighted F64 covariance, never a bank of all-head activations."""
    def __init__(self, arrays=None):
        self.blocks = {}
        self.initial = arrays

    def update(self, layer, values, labels, weights):
        import torch
        device = values.device
        if layer not in self.blocks:
            if self.initial is None:
                self.blocks[layer] = (
                    torch.zeros(6, dtype=torch.float64, device=device),
                    torch.zeros(HEADS, 6, HEAD_DIM, dtype=torch.float64, device=device),
                    torch.zeros(HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float64, device=device))
            else:
                self.blocks[layer] = tuple(torch.as_tensor(self.initial[key][layer], device=device).clone()
                                           for key in ["n", "sum", "xx"])
        count, total, xx = self.blocks[layer]
        x = values.to(torch.float64)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        weights = torch.as_tensor(weights, dtype=torch.float64, device=device)
        xx.add_(torch.bmm(x.permute(1, 2, 0), (x * weights[:, None, None]).permute(1, 0, 2)))
        for role in range(6):
            mask = labels == role
            count[role].add_(weights[mask].sum())
            if bool(mask.any()):
                total[:, role].add_((x[mask] * weights[mask, None, None]).sum(0))

    def export(self):
        if not self.blocks and self.initial is not None:
            return {key: np.array(value, copy=True) for key, value in self.initial.items()}
        require(set(self.blocks) == set(range(LAYERS)), "A model layer has no accumulated scan statistics")
        return {key: np.stack([self.blocks[i][j].cpu().numpy() for i in range(LAYERS)])
                for j, key in enumerate(["n", "sum", "xx"])}


def fit_head_lda(arrays):
    count = np.repeat(arrays["n"], HEADS, axis=0)
    total = arrays["sum"].reshape(LAYERS * HEADS, 6, HEAD_DIM)
    xx = arrays["xx"].reshape(LAYERS * HEADS, HEAD_DIM, HEAD_DIM)
    require((count > 0).all(), "Every head needs all six classes in its FIT statistics subset")
    mean = total / count[:, :, None]
    within = xx - np.einsum("hcd,hce,hc->hde", total, total, 1 / count, optimize=True)
    within /= count.sum(1)[:, None, None]
    within = .5 * (within + within.transpose(0, 2, 1))
    trace = np.trace(within, axis1=1, axis2=2) / HEAD_DIM
    floor = np.maximum(trace, 1e-12)
    covariance = .9 * within + .1 * floor[:, None, None] * np.eye(HEAD_DIM)
    weight = np.linalg.solve(covariance, mean.transpose(0, 2, 1)).transpose(0, 2, 1)
    # Uniform class priors isolate role separability rather than article sampling frequencies.
    bias = -.5 * np.einsum("hcd,hcd->hc", mean, weight)
    require(np.isfinite(weight).all() and np.isfinite(bias).all(), "Nonfinite head LDA parameters")
    return weight, bias


def _head_scores(confusion):
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    precision = np.divide(diagonal, confusion.sum(1), out=np.zeros_like(diagonal), where=confusion.sum(1) > 0)
    recall = np.divide(diagonal, confusion.sum(2), out=np.zeros_like(diagonal), where=confusion.sum(2) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(diagonal), where=precision + recall > 0)
    return f1.mean(1), np.minimum(precision, recall).min(1)


def _scan_signature(data_dir, model_dir, config, plan):
    return {"pilot_fit_sha256": digest(_input_path(data_dir, "pilot", "fit")),
            "model_binding": model_binding(model_dir), "config_sha256": object_digest(config),
            "scan_plan_sha256": object_digest(plan), "capture": "pre-o_proj", "roles": ROLES,
            "implementation_sha256": digest(__file__)}


def _preflight_required(run_dir, model_dir):
    path = Path(run_dir) / "preflight.json"
    require(path.exists(), "Run the 4096-token GPU preflight before scan/extract")
    result = json.loads(path.read_text())
    require(result.get("status") == "passed" and result.get("maximum_input_tokens") == 4096,
            "The longest-input preflight did not pass")
    require(result["model_binding"] == model_binding(model_dir), "Preflight model/tokenizer identity changed")


def scan(data_dir, run_dir, config, phase="pilot", model_dir=None, devices="0,1,2,3", cache_dir=None,
         resume=False, model=None):
    """Rank all heads using only two disjoint source partitions of pilot FIT."""
    import torch
    require(phase == "pilot", "Head discovery is restricted to pilot FIT")
    require(model_dir is not None, "model_dir is required")
    run_dir = Path(run_dir)
    _preflight_required(run_dir, model_dir)
    folder = run_dir / "head-scan"
    folder.mkdir(parents=True, exist_ok=True)
    path = _input_path(data_dir, "pilot", "fit")
    plan = _scan_plan(path, _get(config, "scan_max_rows", 65536), _get(config, "scan_max_sources", None))
    signature = _scan_signature(data_dir, model_dir, config, plan)
    selection_path = run_dir / "head-selection.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text())
        require(existing["signature"] == signature, "Existing selected heads belong to a different run")
        require(resume, "Head selection already exists; use --resume")
        return existing
    atomic_json(folder / "plan.json", {"signature": signature, **plan})
    state_path = folder / "progress.json"
    state = {"phase": "statistics", "next_record": 0, "signature": signature}
    arrays = None
    if state_path.exists():
        require(resume, "An interrupted scan exists; use --resume")
        state = json.loads(state_path.read_text())
        require(state["signature"] == signature, "Scan checkpoint identity mismatch")
        snap = folder / state["checkpoint"]
        require(digest(snap) == state["checkpoint_sha256"], "Scan checkpoint was modified")
        with np.load(snap, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ["n", "sum", "xx"]}
    owned = model is None
    started = time.monotonic()
    try:
        if owned:
            model = load_model(model_dir, run_dir, devices, cache_dir)
        moments = OnlineMoments(arrays)
        if state["phase"] == "statistics":
            chosen = plan["records"]["statistics"]
            processed = 0
            input_tokens = 0
            phase_started = time.monotonic()
            for ix, record in enumerate(_records(path)):
                if ix < state["next_record"] or str(ix) not in chosen:
                    continue
                targets = [record["targets"][j] for j in chosen[str(ix)]]
                labels = [_label(target) for target in targets]
                weights = [float(target.get("weight", 1)) for target in targets]
                capture = HeadCapture([row["token_ix"] for row in targets], receive=
                    lambda layer, x: moments.update(layer, x, labels, weights))
                with torch.inference_mode(), capture.install(model):
                    hidden = _forward(model, record["input_ids"])
                    require(bool(torch.isfinite(hidden).all()), "Nonfinite head-scan model output")
                processed += 1
                input_tokens += len(record["input_ids"])
                if processed % int(_get(config, "scan_checkpoint_records", 16)) == 0:
                    arrays = moments.export()
                    checkpoint = f"statistics-{ix + 1:08d}.npz"
                    atomic_npz(folder / checkpoint, **arrays)
                    state.update(next_record=ix + 1, checkpoint=checkpoint, checkpoint_sha256=digest(folder / checkpoint))
                    atomic_json(state_path, state)
                    seconds = time.monotonic() - phase_started
                    print(json.dumps({"stage": "scan_statistics", "record": ix + 1,
                                      "processed_records": processed, "input_tokens": input_tokens,
                                      "seconds": seconds, "input_tokens_per_second": input_tokens / max(seconds, 1e-9)}), flush=True)
            arrays = moments.export()
            atomic_npz(folder / "statistics-final.npz", **arrays)
            state.update(phase="inner", next_record=0, checkpoint="statistics-final.npz",
                         checkpoint_sha256=digest(folder / "statistics-final.npz"))
            atomic_json(state_path, state)
        weight, bias = fit_head_lda(arrays)
        del moments
        torch.cuda.empty_cache()
        # Evaluation stores per-head confusion only, never all-head feature rows.
        confusion = np.zeros((LAYERS * HEADS, 6, 6), dtype=np.float64)
        if state.get("inner_checkpoint"):
            require(digest(folder / state["inner_checkpoint"]) == state["inner_sha256"], "Inner scan checkpoint was modified")
            with np.load(folder / state["inner_checkpoint"], allow_pickle=False) as saved:
                confusion = saved["confusion"]
        cached = {}
        chosen = plan["records"]["inner"]
        processed = 0
        input_tokens = 0
        phase_started = time.monotonic()
        for ix, record in enumerate(_records(path)):
            if ix < state["next_record"] or str(ix) not in chosen:
                continue
            targets = [record["targets"][j] for j in chosen[str(ix)]]
            labels = np.asarray([_label(target) for target in targets])
            weights = np.asarray([float(target.get("weight", 1)) for target in targets])

            def receive(layer, values):
                if layer not in cached:
                    sl = slice(layer * HEADS, (layer + 1) * HEADS)
                    cached[layer] = (torch.as_tensor(weight[sl], device=values.device),
                                     torch.as_tensor(bias[sl], device=values.device))
                w, b = cached[layer]
                logits = torch.einsum("nhd,hcd->nhc", values.double(), w) + b[None]
                pred = logits.argmax(-1).cpu().numpy()
                for head in range(HEADS):
                    np.add.at(confusion[layer * HEADS + head], (labels, pred[:, head]), weights)

            capture = HeadCapture([row["token_ix"] for row in targets], receive=receive)
            with torch.inference_mode(), capture.install(model):
                _forward(model, record["input_ids"])
            processed += 1
            input_tokens += len(record["input_ids"])
            if processed % int(_get(config, "scan_checkpoint_records", 16)) == 0:
                checkpoint = f"inner-{ix + 1:08d}.npz"
                atomic_npz(folder / checkpoint, confusion=confusion)
                state.update(next_record=ix + 1, inner_checkpoint=checkpoint, inner_sha256=digest(folder / checkpoint))
                atomic_json(state_path, state)
                seconds = time.monotonic() - phase_started
                print(json.dumps({"stage": "scan_inner", "record": ix + 1,
                                  "processed_records": processed, "input_tokens": input_tokens,
                                  "seconds": seconds, "input_tokens_per_second": input_tokens / max(seconds, 1e-9)}), flush=True)
        macro, weakest = _head_scores(confusion)
        order = sorted(range(LAYERS * HEADS), key=lambda h: (-macro[h], -weakest[h], h))
        selection = {"format_version": "synthetic-agent-heads-v1", "roles": ROLES,
                     "heads": order[:TOP_K], "head_order": "descending held-out FIT macro-F1; weakest precision/recall; global head ID",
                     "feature_dim": TOP_K * HEAD_DIM, "signature": signature,
                     "ranking": [{"head": h, "layer": h // HEADS, "within_layer": h % HEADS,
                                  "macro_f1": float(macro[h]), "weakest_precision_recall": float(weakest[h])} for h in order],
                     "scan_seconds": time.monotonic() - started, "counts": plan["counts"],
                     "scan_inputs": plan["inputs"], "source_clusters": plan["clusters"],
                     "no_dev_test_or_historical_probe_used": True,
                     "lda": {"covariance": "0.9 weighted pooled within-class covariance + 0.1 max(trace/D,1e-12) I",
                             "priors": "uniform six-class", "retained": "online F64 sufficient statistics only"}}
        atomic_npz(folder / "inner-confusion.npz", confusion=confusion)
        atomic_json(selection_path, selection)
        state.update(phase="complete", heads=selection["heads"])
        atomic_json(state_path, state)
        return selection
    finally:
        if owned:
            del model
            gc.collect()
            torch.cuda.empty_cache()


def _verify_shard(root, entry, signature):
    folder = root / entry["path"]
    complete = folder / "complete.json"
    require(complete.is_file() and digest(complete) == entry["sha256"], "Feature shard metadata was modified")
    meta = json.loads(complete.read_text())
    require(meta["signature"] == signature and meta["rows"] == entry["rows"], "Feature shard belongs to a different run")
    for name, expected in meta["files"].items():
        require(digest(folder / name) == expected, "Corrupt feature shard: " + str(folder / name))
    x = np.load(folder / "x.npy", mmap_mode="r", allow_pickle=False)
    require(x.shape == (entry["rows"], TOP_K * HEAD_DIM) and x.dtype == np.float16,
            "Feature shard geometry/dtype changed")
    return meta


def _save_shard(root, split, index, buffers, record_start, record_end, signature):
    name = f"{split}-{index:06d}"
    destination = root / name
    temporary = root / (name + ".tmp")
    require(not destination.exists(), "Refusing to overwrite a completed feature shard")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    files = {}
    arrays = {key: np.concatenate(value, axis=0) for key, value in buffers.items()}
    rows = len(arrays["y"])
    require(arrays["x"].shape == (rows, TOP_K * HEAD_DIM), "Incomplete selected-head feature output")
    require(np.isfinite(arrays["x"]).all(), "Nonfinite features cannot be committed")
    for key, value in arrays.items():
        path = temporary / f"{key}.npy"
        with path.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        files[path.name] = digest(path)
    metadata = {"format_version": "synthetic-agent-feature-shard-v1", "rows": rows,
                "record_start": record_start, "record_end": record_end,
                "signature": signature, "files": files}
    atomic_json(temporary / "complete.json", metadata)
    temporary.replace(destination)
    return {"path": name, "rows": rows, "sha256": digest(destination / "complete.json")}


def _row_arrays(record, x):
    targets = record["targets"]
    source = str(record.get("cluster_id", record["source_id"]))
    groups = []
    for row in targets:
        groups.append(json.dumps({
            "profile": record.get("profile"), "template_id": record.get("template_id"),
            "rounds": len(record.get("round_segments", [])),
            "history_policy": record.get("history_policy"), "snapshot_stage": record.get("snapshot_stage"),
            "position_bin": min(3, int(row["token_ix"]) * 4 // len(record["input_ids"])),
        }, separators=(",", ":"), sort_keys=True))
    return {
        "x": x, "y": np.asarray([_label(row) for row in targets], dtype=np.int64),
        "structure": np.asarray([str(row.get("structure", record.get("profile", "") + ":" + record.get("template_id", ""))) for row in targets]),
        "source": np.asarray([source] * len(targets)),
        "family": np.asarray([str(row["family_id"]) for row in targets]),
        "weight": np.asarray([float(row.get("weight", 1)) for row in targets], dtype=np.float32),
        "groups": np.asarray(groups),
    }


def extract(data_dir, run_dir, config, phase="pilot", model_dir=None, devices="0,1,2,3", cache_dir=None,
            resume=False, splits=("fit", "dev"), model=None):
    """Write immutable, independently hashed FP16 shards, resuming per shard.

    A crash before a shard commit repeats that shard's forwards. A crash after
    the directory rename but before manifest publication adopts the verified
    orphan, rather than accidentally counting its body tokens twice.
    """
    import torch
    require(model_dir is not None, "model_dir is required")
    run_dir = Path(run_dir)
    _preflight_required(run_dir, model_dir)
    selection_path = run_dir / "head-selection.json"
    require(selection_path.exists(), "Select heads from pilot FIT before extraction")
    selection = json.loads(selection_path.read_text())
    heads = selection["heads"]
    require(len(heads) == TOP_K and len(set(heads)) == TOP_K and min(heads) >= 0 and max(heads) < LAYERS * HEADS,
            "Exactly 16 distinct ordered heads are required")
    require(selection["signature"]["model_binding"] == model_binding(model_dir), "Selected heads belong to a different backbone")
    pilot_fit = _input_path(data_dir, "pilot", "fit")
    require(pilot_fit.exists(), "The original pilot FIT inputs are required to verify inherited selected heads")
    pilot_fit_sha = digest(pilot_fit)
    require(selection["signature"]["pilot_fit_sha256"] == pilot_fit_sha,
            "Selected heads were discovered from different pilot FIT inputs; old data-version rankings cannot be reused")
    selection_sha = digest(selection_path)
    root = Path(_get(config, "feature_dir", str(run_dir / "features" / phase)))
    root.mkdir(parents=True, exist_ok=True)
    signature = {"model_binding": model_binding(model_dir), "heads": heads,
                 "selection_sha256": selection_sha, "pilot_fit_sha256": pilot_fit_sha,
                 "config_sha256": object_digest(config),
                 "phase": phase, "capture": "pre-o_proj", "implementation_sha256": digest(__file__)}
    primary = {split: digest(_input_path(data_dir, phase, split)) for split in ["fit", "dev"]
               if _input_path(data_dir, phase, split).exists()}
    manifest = {"format_version": "synthetic-agent-features-v1", "roles": ROLES,
                "feature_dim": TOP_K * HEAD_DIM, "feature_position": "pre-o_proj",
                "model_binding": signature["model_binding"], "heads": heads,
                "selection_sha256": selection_sha, "pilot_fit_sha256": pilot_fit_sha,
                "data_hash": object_digest(primary), "config_hash": object_digest(config),
                "signature": signature, "splits": {}, "split_sources": {}, "progress": {},
                "source_identity": "near-duplicate cluster_id, falling back to source_id",
                "group_format": "JSON strings containing profile/template_id/rounds/history_policy/snapshot_stage/position_bin"}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        require(resume, "Feature manifest already exists; use --resume")
        existing = json.loads(manifest_path.read_text())
        require(existing["signature"] == signature and existing["data_hash"] == manifest["data_hash"],
                "Feature cache identity mismatch; configurations/models/data cannot be mixed")
        manifest = existing
    else:
        atomic_json(manifest_path, manifest)
    owned = model is None
    selected_checked = False
    started = time.monotonic()
    try:
        for split in splits:
            require(split in ["fit", "dev", "test", "external"], "Unknown split")
            input_path = _input_path(data_dir, phase, split)
            require(input_path.is_file(), f"{split} input is unavailable or remains sealed")
            input_sha = digest(input_path)
            split_signature = {**signature, "split": split, "input_sha256": input_sha}
            if split in manifest["split_sources"]:
                require(manifest["split_sources"][split] == input_sha, "Split inputs changed during extraction")
            manifest["split_sources"][split] = input_sha
            entries = manifest["splits"].setdefault(split, [])
            cursor = 0
            for entry in entries:
                meta = _verify_shard(root, entry, split_signature)
                require(meta["record_start"] == cursor, "Feature shards do not cover a contiguous input prefix")
                cursor = meta["record_end"]
            # Recover the one possible durable-but-unpublished shard.
            orphan = root / f"{split}-{len(entries):06d}" / "complete.json"
            if orphan.exists():
                require(resume, "Unpublished feature shard exists; use --resume")
                meta = json.loads(orphan.read_text())
                entry = {"path": orphan.parent.name, "rows": meta["rows"], "sha256": digest(orphan)}
                _verify_shard(root, entry, split_signature)
                require(meta["record_start"] == cursor, "Unpublished shard is not the next input prefix")
                entries.append(entry)
                cursor = meta["record_end"]
                atomic_json(manifest_path, manifest)
            if manifest["progress"].get(split, {}).get("complete"):
                require(manifest["progress"][split]["records"] >= cursor, "Completed feature progress was modified")
                if manifest["progress"][split]["records"] != cursor:
                    suffix = [(ix, record) for ix, record in enumerate(_records(input_path)) if ix >= cursor]
                    require(len(suffix) == manifest["progress"][split]["records"] - cursor
                            and not any(record["targets"] for _, record in suffix),
                            "Completed cache omitted supervised body targets")
                continue
            buffers = {key: [] for key in ["x", "y", "structure", "source", "family", "weight", "groups"]}
            buffered = 0
            record_start = cursor
            max_rows = int(_get(config, "extraction_shard_rows", 16384))
            require(max_rows > 0, "extraction_shard_rows must be positive")
            records_seen, input_tokens = cursor, 0
            split_started = time.monotonic()
            for ix, record in enumerate(_records(input_path)):
                records_seen = ix + 1
                if ix < cursor:
                    continue
                _checked_record(record)
                require(record["split"] == split, "Cross-split input in feature extractor")
                input_tokens += len(record["input_ids"])
                if record["targets"]:
                    if model is None:
                        model = load_model(model_dir, run_dir, devices, cache_dir)
                    if not selected_checked:
                        # Active selected hooks are checked with actual input and actual ordered heads.
                        check = _check_hook(model, record["input_ids"], selected=heads)
                        atomic_json(root / "selected-hook-validation.json", check)
                        selected_checked = True
                    capture = HeadCapture([row["token_ix"] for row in record["targets"]], selected=heads)
                    with torch.inference_mode(), capture.install(model):
                        hidden = _forward(model, record["input_ids"])
                        require(bool(torch.isfinite(hidden).all()), "Nonfinite feature extraction model output")
                    arrays = _row_arrays(record, capture.matrix())
                    for key, value in arrays.items():
                        buffers[key].append(value)
                    buffered += len(record["targets"])
                if buffered >= max_rows:
                    entry = _save_shard(root, split, len(entries), buffers, record_start, ix + 1, split_signature)
                    entries.append(entry)
                    manifest["progress"][split] = {"complete": False, "records": ix + 1,
                                                   "rows": sum(e["rows"] for e in entries)}
                    atomic_json(manifest_path, manifest)
                    buffers = {key: [] for key in buffers}
                    buffered = 0
                    record_start = ix + 1
                if (ix + 1 - cursor) % 16 == 0:
                    seconds = time.monotonic() - split_started
                    print(json.dumps({"stage": "extract", "phase": phase, "split": split,
                                      "record": ix + 1, "input_tokens": input_tokens,
                                      "seconds": seconds, "input_tokens_per_second": input_tokens / max(seconds, 1e-9),
                                      "committed_rows": sum(e["rows"] for e in entries), "buffered_rows": buffered}), flush=True)
            if buffered:
                entries.append(_save_shard(root, split, len(entries), buffers, record_start, records_seen, split_signature))
            # An empty-target suffix requires no additional feature shard; retain its explicit count.
            manifest["progress"][split] = {"complete": True, "records": records_seen,
                                           "rows": sum(e["rows"] for e in entries),
                                           "newly_processed_input_tokens": input_tokens}
            atomic_json(manifest_path, manifest)
        manifest["last_extraction_seconds"] = time.monotonic() - started
        manifest["feature_root"] = str(root.resolve())
        atomic_json(manifest_path, manifest)
        return manifest
    finally:
        if owned and model is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()
