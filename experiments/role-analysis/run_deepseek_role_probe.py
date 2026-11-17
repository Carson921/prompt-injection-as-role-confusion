"""Train a frozen DeepSeek V4 native-precision head-output role readout.

The backbone remains on four H100s. Head discovery uses two source-disjoint
subsets of FIT only; classifier selection uses DEV; TEST is extracted only after
the selected classifier is frozen. Completed feature shards are resumable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import deepseek_role_probe_model as backend

BASE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE / "outputs/deepseek-v4-flash-0731-role-probe-20261115"
CAPTURE = "attention-head-output-before-o_a_proj"
SPLITS = ("fit", "dev", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    backend._atomic_json(path, value)


def save_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source(record):
    for key in ("source_cluster", "cluster_id", "source_id"):
        if key in record:
            return str(record[key])
    raise ValueError("Record has no source-cluster identity")


def labels_for(record, roles):
    result = []
    for target in record["targets"]:
        label = target["label"]
        label = roles.index(label) if isinstance(label, str) else int(label)
        require(0 <= label < len(roles), "Target label outside the declared native role schema")
        require(target.get("role", roles[label]) == roles[label], "Role label and integer disagree")
        result.append(label)
    return np.asarray(result, np.int64)


def validate_record(record, split, roles):
    require(record["split"] == split, "Record occurs in the wrong split")
    require(record["domain"] in ("c4_dolma3", "agentdojo"), "Unexpected domain")
    positions = [int(t["token_ix"]) for t in record["targets"]]
    require(len(positions) == len(set(positions)), "Duplicate body-token target")
    require(all(0 <= p < len(record["input_ids"]) for p in positions), "Invalid body-token target")
    labels_for(record, roles)


def model_binding(model_dir):
    root = Path(model_dir)
    return {"model_id": "deepseek-ai/DeepSeek-V4-Flash-0731", "path": str(root.resolve()),
            "config_sha256": sha(root / "config.json"),
            "index_sha256": sha(root / "model.safetensors.index.json"),
            "tokenizer_sha256": sha(root / "tokenizer.json"),
            "official_encoder_sha256": sha(root / "encoding/encoding_dsv4.py"),
            "native_experts": "packed FP4", "native_dense": "FP8", "offload": False}


def verify_download(output, model_dir):
    """Verify every official file, including LFS SHA256 of all model shards."""
    manifest = read(output / "download-manifest.json")
    old_path = output / "weight-verification.json"
    binding = {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
               "expected_files_sha256": object_sha([
                   {"path": f["path"], "size": f["size"], "lfs_sha256": (f.get("lfs") or {}).get("sha256")}
                   for f in sorted(manifest["files"], key=lambda f: f["path"])])}
    if old_path.exists():
        old = read(old_path)
        if old.get("binding") == binding and all(
                (model_dir / f["path"]).is_file() and (model_dir / f["path"]).stat().st_size == f["size"]
                for f in manifest["files"]):
            return old
    incremental_path = output / "download-file-verification.json"
    incremental = read(incremental_path) if incremental_path.exists() else {}
    cached_files = {f["path"]: f for f in incremental.get("files", [])}
    results = []
    for i, entry in enumerate(manifest["files"]):
        path = model_dir / entry["path"]
        require(path.is_file() and path.stat().st_size == entry["size"], f"Incomplete model file: {path}")
        stat = path.stat()
        cached = cached_files.get(entry["path"], {})
        cached_matches = (incremental.get("revision") == manifest["revision"]
                          and cached.get("size") == stat.st_size
                          and cached.get("mtime_ns") == stat.st_mtime_ns
                          and cached.get("inode") == stat.st_ino)
        observed = cached["sha256"] if cached_matches else sha(path)
        expected = (entry.get("lfs") or {}).get("sha256")
        require(expected is None or observed == expected, f"Official LFS checksum mismatch: {path}")
        results.append({"path": entry["path"], "size": entry["size"], "sha256": observed,
                        "official_lfs_verified": expected is not None})
        if path.suffix == ".safetensors":
            print(json.dumps({"stage": "verify_weight", "file": path.name, "index": i}), flush=True)
    result = {"status": "passed", "binding": binding, "files": results,
              "safetensors_bytes": sum(x["size"] for x in results if x["path"].endswith(".safetensors"))}
    write(old_path, result)
    return result


def make_scan_plan(path, roles, config):
    """Preselect source strata and positions without observing model outputs."""
    strata = defaultdict(set)
    available = {}
    for ix, record in enumerate(records(path)):
        validate_record(record, "fit", roles)
        cluster = source(record)
        part = "inner" if int(object_sha(["head-inner-v1", cluster])[:16], 16) % 5 == 0 else "statistics"
        available[ix] = (cluster, part)
        for label in set(labels_for(record, roles)):
            strata[(part, record["domain"], int(label))].add(cluster)
    chosen_sources = set()
    for (part, domain, label), values in sorted(strata.items()):
        limit = int(config["scan_statistics_sources_per_stratum"] if part == "statistics"
                    else config["scan_inner_sources_per_stratum"])
        ranked = sorted(values, key=lambda s: object_sha(["scan-source-v1", s]))
        chosen_sources.update(ranked[:limit])
    candidates = defaultdict(list)
    clusters = {"statistics": set(), "inner": set()}
    for ix, record in enumerate(records(path)):
        cluster, part = available[ix]
        if cluster not in chosen_sources:
            continue
        clusters[part].add(cluster)
        for j, label in enumerate(labels_for(record, roles)):
            key = ["scan-position-v1", record["id"], record["targets"][j]["token_ix"]]
            candidates[(part, record["domain"], int(label))].append((object_sha(key), ix, j))
    plan = {"statistics": defaultdict(list), "inner": defaultdict(list)}
    counts = {}
    limit = int(config["scan_rows_per_domain_role"])
    for (part, domain, label), positions in sorted(candidates.items()):
        retained = sorted(positions)[:limit]
        counts[f"{part}/{domain}/{roles[label]}"] = len(retained)
        for _, ix, j in retained:
            plan[part][str(ix)].append(j)
    for part in plan:
        require(all(any(key.startswith(part + "/") and key.endswith("/" + role) and count > 0
                        for key, count in counts.items()) for role in roles), f"Missing FIT {part} role")
        plan[part] = {ix: sorted(js) for ix, js in plan[part].items()}
    require(not clusters["statistics"] & clusters["inner"], "Source leakage inside FIT head discovery")
    return {"records": plan, "counts": counts, "clusters": {p: sorted(c) for p, c in clusters.items()},
            "source_partition": "20% deterministic source-hash holdout within FIT",
            "source_selection": "lowest fixed source hashes, independently per domain/role/partition",
            "position_selection": "lowest fixed position hashes within retained source strata",
            "no_dev_or_test": True}


class OnlineMoments:
    """F64 pooled within-class LDA statistics, distributed alongside layers."""
    def __init__(self, classes, initial=None):
        self.classes, self.initial, self.blocks = classes, initial, {}

    def update(self, layer, values, labels):
        import torch
        if layer not in self.blocks:
            shapes = ((self.classes,), (backend.HEADS, self.classes, backend.HEAD_DIM),
                      (backend.HEADS, backend.HEAD_DIM, backend.HEAD_DIM))
            self.blocks[layer] = tuple(torch.zeros(shape, dtype=torch.float64, device=values.device)
                                       if self.initial is None else torch.as_tensor(self.initial[key][layer],
                                       device=values.device).clone()
                                       for key, shape in zip(("n", "sum", "xx"), shapes))
        count, total, xx = self.blocks[layer]
        x = values.double()
        y = torch.as_tensor(labels, dtype=torch.long, device=x.device)
        xx.add_(torch.bmm(x.permute(1, 2, 0), x.permute(1, 0, 2)))
        for role in range(self.classes):
            mask = y == role
            count[role].add_(mask.sum())
            if bool(mask.any()):
                total[:, role].add_(x[mask].sum(0))

    def export(self):
        if not self.blocks and self.initial is not None:
            return {key: np.array(value, copy=True) for key, value in self.initial.items()}
        require(set(self.blocks) == set(range(backend.LAYERS)), "A scanned layer is missing")
        return {key: np.stack([self.blocks[i][j].cpu().numpy() for i in range(backend.LAYERS)])
                for j, key in enumerate(("n", "sum", "xx"))}


def fit_lda(arrays, classes):
    import torch
    weights, biases = [], []
    for layer in range(backend.LAYERS):
        device = f"cuda:{layer * 4 // backend.LAYERS}"
        count = torch.as_tensor(arrays["n"][layer], device=device)
        total = torch.as_tensor(arrays["sum"][layer], device=device)
        xx = torch.as_tensor(arrays["xx"][layer], device=device)
        require(bool((count > 0).all()), "Each FIT statistics class must be present")
        mean = total / count[None, :, None]
        within = (xx - torch.einsum("hcd,hce,c->hde", total, total, 1 / count)) / count.sum()
        within = .5 * (within + within.transpose(-1, -2))
        scale = within.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
        cov = .9 * within + .1 * scale[:, None, None] * torch.eye(backend.HEAD_DIM, device=device)
        weight = torch.linalg.solve(cov, mean.transpose(-1, -2)).transpose(-1, -2)
        bias = -.5 * (mean * weight).sum(-1)
        require(bool(torch.isfinite(weight).all()), "Nonfinite LDA coefficients")
        weights.append(weight.cpu().numpy()); biases.append(bias.cpu().numpy())
    return np.concatenate(weights), np.concatenate(biases)


def select_heads(model, data_dir, output, config, roles, binding):
    import torch
    folder = output / "head-scan"
    folder.mkdir(exist_ok=True)
    path = data_dir / "inputs-fit.jsonl"
    plan = make_scan_plan(path, roles, config)
    signature = {"pilot_fit_sha256": sha(path), "model_binding": binding,
                 "config_sha256": object_sha(config), "scan_plan_sha256": object_sha(plan), "capture": CAPTURE,
                 "runner_implementation_sha256": sha(__file__),
                 "capture_implementation_sha256": sha(backend.__file__)}
    selected_path = output / "head-selection.json"
    if selected_path.exists():
        result = read(selected_path)
        require(result["signature"] == signature, "Selected heads belong to different inputs or model")
        return result
    write(folder / "plan.json", {"signature": signature, **plan})
    state_path = folder / "progress.json"
    state = read(state_path) if state_path.exists() else {"phase": "statistics", "next_record": 0, "signature": signature}
    require(state["signature"] == signature, "Head scan resume identity changed")
    stat_path = folder / "statistics.npz"
    started = time.monotonic()
    if state["phase"] == "statistics":
        initial = None
        if state.get("checkpoint"):
            snapshot = folder / state["checkpoint"]
            require(sha(snapshot) == state["checkpoint_sha256"], "Scan statistics checkpoint changed")
            with np.load(snapshot, allow_pickle=False) as z:
                initial = {key: z[key] for key in z.files}
        moments = OnlineMoments(len(roles), initial)
        processed = 0
        for ix, record in enumerate(records(path)):
            if ix < state["next_record"] or str(ix) not in plan["records"]["statistics"]:
                continue
            targets = plan["records"]["statistics"][str(ix)]
            y = labels_for(record, roles)[targets]
            backend.scan_heads(model, record["input_ids"], [record["targets"][j]["token_ix"] for j in targets],
                               lambda layer, x: moments.update(layer, x, y))
            processed += 1
            print(json.dumps({"stage": "head_statistics", "record": ix, "processed": processed,
                              "total_records": len(plan["records"]["statistics"]),
                              "input_tokens": len(record["input_ids"]), "seconds": time.monotonic() - started}), flush=True)
            if processed % int(config["scan_checkpoint_records"]) == 0:
                snapshot = folder / f"statistics-{ix + 1:08d}.npz"
                save_npz(snapshot, **moments.export())
                state.update(next_record=ix + 1, checkpoint=snapshot.name, checkpoint_sha256=sha(snapshot))
                write(state_path, state)
        save_npz(stat_path, **moments.export())
        state.update(phase="inner", next_record=0)
        write(state_path, state)
        del moments, initial
        gc.collect(); torch.cuda.empty_cache()
    lda_path = folder / "lda.npz"
    if not lda_path.exists():
        with np.load(stat_path, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        weight, bias = fit_lda(arrays, len(roles))
        save_npz(lda_path, weight=weight, bias=bias)
        del arrays
        gc.collect(); torch.cuda.empty_cache()
    with np.load(lda_path, allow_pickle=False) as z:
        weight, bias = z["weight"], z["bias"]
    confusion_path = folder / "inner-confusion.npz"
    confusion = np.zeros((backend.LAYERS * backend.HEADS, len(roles), len(roles)), np.int64)
    if state.get("inner_checkpoint"):
        snapshot = folder / state["inner_checkpoint"]
        require(sha(snapshot) == state["inner_checkpoint_sha256"], "FIT inner checkpoint changed")
        with np.load(snapshot, allow_pickle=False) as z:
            confusion = z["confusion"]
    cache = {}
    processed = 0
    for ix, record in enumerate(records(path)):
        if ix < state["next_record"] or str(ix) not in plan["records"]["inner"]:
            continue
        targets = plan["records"]["inner"][str(ix)]
        y = labels_for(record, roles)[targets]

        def receive(layer, values):
            if layer not in cache:
                sl = slice(layer * backend.HEADS, (layer + 1) * backend.HEADS)
                cache[layer] = (torch.as_tensor(weight[sl], device=values.device),
                                torch.as_tensor(bias[sl], device=values.device))
            w, b = cache[layer]
            pred = (torch.einsum("nhd,hcd->nhc", values.double(), w) + b[None]).argmax(-1).cpu().numpy()
            for head in range(backend.HEADS):
                np.add.at(confusion[layer * backend.HEADS + head], (y, pred[:, head]), 1)

        backend.scan_heads(model, record["input_ids"], [record["targets"][j]["token_ix"] for j in targets], receive)
        processed += 1
        print(json.dumps({"stage": "head_inner", "record": ix, "processed": processed,
                          "total_records": len(plan["records"]["inner"]), "seconds": time.monotonic() - started}), flush=True)
        if processed % int(config["scan_checkpoint_records"]) == 0:
            snapshot = folder / f"inner-{ix + 1:08d}.npz"
            save_npz(snapshot, confusion=confusion)
            state.update(next_record=ix + 1, inner_checkpoint=snapshot.name,
                         inner_checkpoint_sha256=sha(snapshot))
            write(state_path, state)
    save_npz(confusion_path, confusion=confusion)
    diagonal = confusion.diagonal(axis1=1, axis2=2)
    precision = np.divide(diagonal, confusion.sum(1), out=np.zeros_like(diagonal, dtype=float), where=confusion.sum(1) > 0)
    recall = np.divide(diagonal, confusion.sum(2), out=np.zeros_like(diagonal, dtype=float), where=confusion.sum(2) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=precision + recall > 0)
    macro, weakest = f1.mean(1), np.minimum(precision, recall).min(1)
    order = sorted(range(len(confusion)), key=lambda h: (-macro[h], -weakest[h], h))
    result = {"format_version": "deepseek-role-head-selection-v1", "roles": roles,
              "heads": order[:16], "feature_dim": 16 * backend.HEAD_DIM, "head_dim": backend.HEAD_DIM,
              "feature_position": CAPTURE, "signature": signature, "counts": plan["counts"],
              "ranking": [{"head": h, "layer": h // backend.HEADS, "within_layer": h % backend.HEADS,
                           "macro_f1": float(macro[h]), "weakest_precision_recall": float(weakest[h])} for h in order],
              "no_dev_test_or_historical_probe_used": True, "seconds": time.monotonic() - started,
              "lda": "F64, 0.9 pooled within-class covariance + 0.1 trace/D identity; uniform role priors"}
    write(selected_path, result)
    state.update(phase="complete", heads=result["heads"])
    write(state_path, state)
    del cache
    gc.collect(); torch.cuda.empty_cache()
    return result


def preflight(model, data_dir, output, config):
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    short = tokenizer.encode("The forests contain many plants and animals.", add_special_tokens=True)
    # TEST content must never enter a backbone forward before classifier freeze.
    max_record = max((r for s in ("fit", "dev") for r in records(data_dir / f"inputs-{s}.jsonl")),
                     key=lambda r: len(r["input_ids"]))
    long_ids = max_record["input_ids"]
    checks = []
    for ids in (short, long_ids):
        start = time.monotonic()
        before = backend._forward(model, ids).cpu()
        positions = sorted(set((0, len(ids) // 2, len(ids) - 1)))
        captured = {}
        capture = backend.HeadCapture(positions, receive=lambda layer, x: captured.update({layer: tuple(x.shape)}))
        with capture.install(model):
            after = backend._forward(model, ids).cpu()
        require(bool(torch.isfinite(before).all() and torch.isfinite(after).all()), "Nonfinite preflight output")
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)
        checks.append({"tokens": len(ids), "all_layers_captured": len(captured),
                       "hook_output_bitexact": bool(torch.equal(before, after)),
                       "hook_output_max_abs_difference": float((before - after).abs().max()),
                       "seconds": time.monotonic() - start})
        print(json.dumps({"stage": "preflight_forward", **checks[-1]}), flush=True)
    audit = backend.audit_model(model)
    result = {"status": "passed", "checks": checks, "backbone_audit": audit,
              "gpu_memory": backend.memory_report(), "maximum_input_tokens": len(long_ids),
              "batch_size": 1, "use_cache": False, "lm_head_skipped": True}
    write(output / "preflight.json", result)
    return result


def feature_arrays(record, x, roles):
    targets = record["targets"]
    families = [str(t.get("family", t.get("family_id", f"{record['id']}:{t['token_ix']}"))) for t in targets]
    groups = [json.dumps({"domain": record["domain"], "corpus": record.get("corpus", "agentdojo"),
                          "suite": record.get("suite", "N/A"), "scope": t.get("scope", "article_body")},
                         sort_keys=True) for t in targets]
    return {"x": x, "y": labels_for(record, roles),
            "domain": np.full(len(targets), record["domain"]),
            "source": np.full(len(targets), source(record)), "family": np.asarray(families),
            "groups": np.asarray(groups)}


def save_feature_shard(root, split, ix, buffers, start, end, signature):
    folder = root / f"{split}-{ix:06d}"
    temp = folder.with_name(folder.name + ".tmp")
    require(not folder.exists(), "Refusing to overwrite a committed feature shard")
    temp.mkdir(exist_ok=True)
    files = {}
    arrays = {key: np.concatenate(parts) for key, parts in buffers.items()}
    require(np.isfinite(arrays["x"]).all(), "Nonfinite cached features")
    for key, value in arrays.items():
        path = temp / f"{key}.npy"
        with path.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        files[path.name] = sha(path)
    metadata = {"signature": signature, "rows": len(arrays["y"]), "record_start": start,
                "record_end": end, "files": files}
    write(temp / "complete.json", metadata)
    temp.replace(folder)
    return {"path": folder.name, "rows": len(arrays["y"]), "sha256": sha(folder / "complete.json")}


def verify_feature_shard(root, entry, signature):
    folder = root / entry["path"]
    require(sha(folder / "complete.json") == entry["sha256"], "Feature-shard metadata changed")
    meta = read(folder / "complete.json")
    require(meta["signature"] == signature and meta["rows"] == entry["rows"], "Wrong feature-shard identity")
    for name, expected in meta["files"].items():
        require(sha(folder / name) == expected, f"Corrupt feature file: {folder / name}")
    return meta


def extract(model, data_dir, output, config, roles, binding, selection, splits):
    root = output / "features"
    root.mkdir(exist_ok=True)
    manifest_path = root / "manifest.json"
    split_sources = {s: sha(data_dir / f"inputs-{s}.jsonl") for s in SPLITS}
    signature = {"model_binding": binding, "heads": selection["heads"], "capture": CAPTURE,
                 "selection_sha256": sha(output / "head-selection.json"),
                 "pilot_fit_sha256": split_sources["fit"], "config_sha256": object_sha(config),
                 "runner_implementation_sha256": sha(__file__),
                 "capture_implementation_sha256": sha(backend.__file__)}
    manifest = {"format_version": "deepseek-role-features-v1", "roles": roles, "feature_dim": 16 * backend.HEAD_DIM,
                "head_dim": backend.HEAD_DIM, "heads": selection["heads"], "feature_position": CAPTURE,
                "model_binding": binding, "selection_sha256": signature["selection_sha256"],
                "pilot_fit_sha256": split_sources["fit"], "data_hash": object_sha(split_sources),
                "config_hash": object_sha(config), "signature": signature, "split_sources": split_sources,
                "splits": {}, "progress": {}}
    if manifest_path.exists():
        existing = read(manifest_path)
        require(existing["signature"] == signature and existing["split_sources"] == split_sources,
                "Feature cache belongs to different model/data/heads/config")
        manifest = existing
    write(manifest_path, manifest)
    for split in splits:
        if split == "test":
            import deepseek_role_probe_train as trainer
            require((output / "freeze.json").is_file(), "TEST extraction is sealed until classifier freeze")
            trainer.verify_freeze(output)
        split_signature = {**signature, "split": split, "input_sha256": split_sources[split]}
        entries = manifest["splits"].setdefault(split, [])
        cursor = 0
        for entry in entries:
            meta = verify_feature_shard(root, entry, split_signature)
            require(meta["record_start"] == cursor, "Feature cache has a record gap")
            cursor = meta["record_end"]
        orphan = root / f"{split}-{len(entries):06d}" / "complete.json"
        if orphan.exists():
            entry = {"path": orphan.parent.name, "rows": read(orphan)["rows"], "sha256": sha(orphan)}
            meta = verify_feature_shard(root, entry, split_signature)
            require(meta["record_start"] == cursor, "Orphan feature shard is out of sequence")
            entries.append(entry); cursor = meta["record_end"]
            write(manifest_path, manifest)
        if manifest["progress"].get(split, {}).get("complete"):
            continue
        buffers = {key: [] for key in ("x", "y", "domain", "source", "family", "groups")}
        buffered, start, count, tokens = 0, cursor, cursor, 0
        started = time.monotonic()
        for ix, record in enumerate(records(data_dir / f"inputs-{split}.jsonl")):
            count = ix + 1
            if ix < cursor:
                continue
            validate_record(record, split, roles)
            if record["targets"]:
                x = backend.extract_selected(model, record["input_ids"],
                                             [t["token_ix"] for t in record["targets"]], selection["heads"])
                for key, values in feature_arrays(record, x, roles).items():
                    buffers[key].append(values)
                buffered += len(x)
            tokens += len(record["input_ids"])
            if buffered >= int(config["extraction_shard_rows"]):
                entries.append(save_feature_shard(root, split, len(entries), buffers, start, count, split_signature))
                manifest["progress"][split] = {"complete": False, "records": count, "rows": sum(e["rows"] for e in entries)}
                write(manifest_path, manifest)
                buffers = {key: [] for key in buffers}; buffered = 0; start = count
            if (ix + 1 - cursor) % 16 == 0:
                print(json.dumps({"stage": "extract", "split": split, "records": count, "input_tokens": tokens,
                                  "committed_rows": sum(e["rows"] for e in entries), "buffered_rows": buffered,
                                  "seconds": time.monotonic() - started}), flush=True)
        if buffered:
            entries.append(save_feature_shard(root, split, len(entries), buffers, start, count, split_signature))
        manifest["progress"][split] = {"complete": True, "records": count, "rows": sum(e["rows"] for e in entries)}
        write(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(backend.DEFAULT_MODEL))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("verify", "preflight", "scan", "extract", "all"), default="all")
    parser.add_argument("--skip-weight-hashes", action="store_true", help="Only for repeated local diagnostics after verification")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    config = {"model": str(args.model.resolve()), "seed": 20261115, "top_k": 16,
              "scan_statistics_sources_per_stratum": 16, "scan_inner_sources_per_stratum": 8,
              "scan_rows_per_domain_role": 2048, "scan_checkpoint_records": 64,
              "extraction_shard_rows": 8192, "native_roles": "system,user,cot,assistant,tool",
              "classifier": {"device": "cuda:0", "batch_size": 4096, "learning_rate": .001,
                             "weight_decay": .001, "dropout": .1, "gradient_clip": 5.,
                             "max_epochs": 45, "min_epochs": 10, "patience": 8,
                             "checkpoint_steps": 100, "bootstrap_replicates": 2000,
                             "cache_features_on_gpu": True,
                             "profile_weights": {"c4_dolma3": .5, "agentdojo": .5}}}
    config_path = output / "config.json"
    if config_path.exists():
        require(read(config_path) == config, "Run config changed; use a fresh output directory")
    else:
        write(config_path, config)
    (output / ".gitignore").write_text("data/\nfeatures/\nhead-scan/*.npz\n*.log\n")
    if not args.skip_weight_hashes:
        verify_download(output, args.model)
    if args.stage == "verify":
        return
    manifest = read(data_dir / "manifest.json")
    roles = manifest["roles"]
    binding = model_binding(args.model)
    model = backend.load_model(args.model, output)
    try:
        preflight(model, data_dir, output, config)
        if args.stage == "preflight":
            return
        selection = select_heads(model, data_dir, output, config, roles, binding)
        if args.stage == "scan":
            return
        extract(model, data_dir, output, config, roles, binding, selection, ("fit", "dev"))
        if args.stage == "extract":
            return
        import deepseek_role_probe_train as trainer
        if (output / "freeze.json").exists():
            trainer.verify_freeze(output)
        else:
            trainer.train_candidates(output / "features", output, config, resume=True)
            trainer.freeze(output)
        import torch
        gc.collect(); torch.cuda.empty_cache()
        frozen_sha = sha(output / "checkpoint.npz")
        release_path = output / "test-release.json"
        if release_path.exists():
            require(read(release_path)["checkpoint_sha256"] == frozen_sha, "TEST release belongs to another probe")
        else:
            write(release_path, {"checkpoint_sha256": frozen_sha, "released_unix": time.time(),
                                 "selection_uses_dev_only": True})
        extract(model, data_dir, output, config, roles, binding, selection, ("test",))
        require(sha(output / "checkpoint.npz") == frozen_sha, "Checkpoint changed after TEST release")
        trainer.evaluate(output / "features", output, "test")
        write(output / "runtime-final.json", {"backbone_audit": backend.audit_model(model),
                                               "gpu_memory": backend.memory_report(),
                                               "checkpoint_sha256": frozen_sha, "finished_unix": time.time()})
    finally:
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
