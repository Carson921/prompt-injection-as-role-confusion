"""Collect every GPT-OSS-120B pre-MLP layer for six single-message labels.

Run prepare, extract (one process per GPU), then verify. Activations keep the
model's BF16 dtype in safetensors shards; no padding positions are written.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LABELS = ["system", "developer", "user", "assistant", "tool_response", "think"]
ROLE_MAP = dict(zip(LABELS, ["system", "developer", "user", "assistant", "tool", "cot"]))
KERNEL_REVISION = "9fbc1b48e710ac8abf18ea2650ee690d52dc3859"
LAYERS, WIDTH = 36, 2880
DEFAULT_MODEL = "/scratch/wz55191/models/gpt-oss-120b"
DEFAULT_OUT = ROOT / "experiments/single-message-pre-mlp/outputs/gpt-oss-120b-c4-1000-dolma3-1000"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def object_sha(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


@contextmanager
def lock(path):
    import fcntl
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def emit(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields), ensure_ascii=False), flush=True)


def tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, local_files_only=True, padding_side="right")


def model_identity(model):
    model = Path(model)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    return dict(assets={name: sha((model / name).read_bytes()) for name in
                       ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]},
                shards={name: (model / name).stat().st_size for name in names})


def build_messages(raw, tok, max_tokens):
    """Use the repository's exact Harmony wrappers, including standalone tool."""
    from utils.role_templates import render_single_message
    rows, cursor = [], 0
    control_ids = set(tok.all_special_ids) | {i for i, token in tok.added_tokens_decoder.items() if token.special}
    for sample_id, sample in enumerate(raw):
        body_ids = tok(sample["text"], add_special_tokens=False, truncation=True,
                       max_length=max_tokens)["input_ids"]
        body = tok.decode(body_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        body_ids = tok(body, add_special_tokens=False)["input_ids"]
        require(body_ids and not (set(body_ids) & control_ids),
                f"Sample {sample_id} is empty or contains Harmony control tokens")
        for label_id, label in enumerate(LABELS):
            prompt = render_single_message("gptoss-120b", ROLE_MAP[label], body)
            prefix = render_single_message("gptoss-120b", ROLE_MAP[label], "").removesuffix("<|end|>")
            prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            end_ids = tok("<|end|>", add_special_tokens=False)["input_ids"]
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            require(ids == prefix_ids + body_ids + end_ids, "Token boundaries differ from the single-message template")
            require(ids.count(tok.convert_tokens_to_ids("<|start|>")) == 1, "Input must contain one message")
            rows.append(dict(message_id=len(rows), sample_id=sample_id, source=sample["source"],
                             source_row=sample["source_row"], text_sha256=sha(sample["text"]),
                             label=label, label_id=label_id, legacy_role=ROLE_MAP[label],
                             prompt=prompt, input_ids=ids, content_start=len(prefix_ids),
                             content_end=len(prefix_ids) + len(body_ids), token_count=len(ids),
                             activation_start=cursor, activation_end=cursor + len(ids)))
            cursor += len(ids)
    return rows


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / "prepare.lock"):
        if (out / "manifest.json").exists():
            meta, rows = read_plan(out)
            require(meta["settings"] == settings(args), "Existing data uses different preparation settings")
            require(meta["model_identity"] == model_identity(args.model), "Model/tokenizer identity changed")
            emit("prepared_cache_reused", messages=len(rows), tokens=meta["tokens"])
            return
        raw_path = out / "raw_samples.json"
        source_path = out / "sources.json"
        if raw_path.exists() and source_path.exists():
            provenance = json.loads(source_path.read_text())
            require(provenance["settings"] == settings(args), "Raw cache settings differ")
            raw = json.loads(raw_path.read_text())
            require(sha(raw_path.read_bytes()) == provenance["raw_sha256"], "Raw data cache changed")
        else:
            api, raw, sources = HfApi(), [], []
            for name, repo, subset, split, revision, count in [
                ("c4", "allenai/c4", "en", "validation", args.c4_revision, args.c4_samples),
                ("dolma3", "allenai/dolma3_mix-150B-1025", None, "train", "3a8349c", args.dolma_samples),
            ]:
                commit = api.dataset_info(repo, revision=revision).sha
                emit("sampling", source=name, count=count, revision=commit)
                source_spec = dict(source=name, repository=repo, subset=subset, split=split,
                                   revision=commit, count=count, seed=args.seed, buffer_size=args.shuffle_buffer)
                cache = out / f"sampled-{name}.json"
                if cache.exists():
                    saved = json.loads(cache.read_text())
                    require(saved["spec"] == source_spec, f"Cached {name} sampling settings differ")
                    require(object_sha(saved["samples"]) == saved["samples_sha256"], "Sample cache changed")
                    samples = saved["samples"]
                else:
                    ds = load_dataset(repo, name=subset, split=split, revision=commit, streaming=True)
                    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
                    samples = []
                    for source_row, sample in enumerate(ds.take(count)):
                        samples.append(dict(source=name, source_row=source_row, text=sample["text"],
                                            provenance={k: sample[k] for k in ["id", "url", "timestamp"] if k in sample}))
                    atomic_json(cache, dict(spec=source_spec, samples=samples, samples_sha256=object_sha(samples)))
                raw.extend(samples)
                require(sum(r["source"] == name for r in raw) == count, f"Insufficient {name} samples")
                sources.append(dict(source=name, repository=repo, subset=subset, split=split,
                                    revision=commit, count=count))
                emit("source_sampled", source=name, samples=len(samples))
            atomic_json(raw_path, raw)
            provenance = dict(settings=settings(args), sources=sources, raw_sha256=sha(raw_path.read_bytes()))
            atomic_json(source_path, provenance)
        tok = tokenizer(args.model)
        rows = build_messages(raw, tok, args.max_body_tokens)
        table = pa.Table.from_pylist(rows)
        tmp = out / "messages.parquet.tmp"
        pq.write_table(table, tmp)
        tmp.replace(out / "messages.parquet")
        shards = []
        for start in range(0, len(rows), args.shard_messages):
            group = rows[start:start + args.shard_messages]
            shards.append(dict(shard_id=len(shards), message_start=start, message_end=start + len(group),
                               tokens=sum(row["token_count"] for row in group)))
        meta = dict(settings=settings(args), labels=LABELS, label_to_legacy_role=ROLE_MAP,
                    sources=provenance["sources"], raw_sha256=provenance["raw_sha256"],
                    messages_sha256=sha((out / "messages.parquet").read_bytes()),
                    model_identity=model_identity(args.model), model_path=str(args.model.resolve()),
                    layers=list(range(LAYERS)), hidden_size=WIDTH, dtype="bfloat16",
                    capture_point="model.layers[i].post_attention_layernorm output = MLP input",
                    token_policy="All non-padding tokens; label_ids=-1 on headers/end, label_id on body",
                    message_count=len(rows), source_sample_count=len(raw),
                    samples_by_source=dict(Counter(row["source"] for row in raw)),
                    messages_by_label=dict(Counter(row["label"] for row in rows)),
                    unique_texts=len({sha(row["text"]) for row in raw}),
                    content_tokens_by_label={label: sum(r["content_end"] - r["content_start"] for r in rows if r["label"] == label) for label in LABELS},
                    tokens=sum(r["token_count"] for r in rows), shards=shards,
                    kernel_revision=KERNEL_REVISION)
        meta["activation_bytes"] = meta["tokens"] * LAYERS * WIDTH * 2
        meta["signature"] = object_sha(meta)
        atomic_json(out / "manifest.json", meta)
        emit("data_ready", samples=len(raw), messages=len(rows), tokens=meta["tokens"],
             activation_gib=meta["activation_bytes"] / 2**30)


def settings(args):
    return dict(c4_samples=args.c4_samples, dolma_samples=args.dolma_samples, seed=args.seed,
                shuffle_buffer=args.shuffle_buffer, max_body_tokens=args.max_body_tokens,
                shard_messages=args.shard_messages, c4_revision=args.c4_revision)


def read_plan(out):
    import pyarrow.parquet as pq
    meta = json.loads((out / "manifest.json").read_text())
    require(object_sha({k: v for k, v in meta.items() if k != "signature"}) == meta["signature"], "Manifest signature changed")
    require(sha((out / "messages.parquet").read_bytes()) == meta["messages_sha256"], "Message index changed")
    require(sha((out / "raw_samples.json").read_bytes()) == meta["raw_sha256"], "Raw samples changed")
    rows = pq.read_table(out / "messages.parquet").to_pylist()
    require(len(rows) == meta["message_count"], "Message count mismatch")
    return meta, rows


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, Mxfp4Config
    from transformers.integrations import hub_kernels
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    original_get = hub_kernels.get_kernel
    def pinned(name, *a, **kw):
        if name == "kernels-community/gpt-oss-triton-kernels":
            kw["revision"] = KERNEL_REVISION
        return original_get(name, *a, **kw)
    hub_kernels.get_kernel = pinned
    original_validate = Mxfp4HfQuantizer.validate_environment
    def strict(self, *a, **kw):
        original_validate(self, *a, **kw)
        require(not self.quantization_config.dequantize, "MXFP4 kernels unavailable; refusing BF16 weight expansion")
    Mxfp4HfQuantizer.validate_environment = strict
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=False), device_map={"": "cuda:0"},
        attn_implementation="eager", use_kernels=False,
    ).eval().requires_grad_(False)
    require(model.config.num_hidden_layers == LAYERS and model.config.hidden_size == WIDTH, "Unexpected model geometry")
    require({type(x.mlp.experts).__name__ for x in model.model.layers} == {"Mxfp4GptOssExperts"}, "Unexpected experts backend")
    return model


@contextmanager
def capture_pre_mlp(model, mask, captured):
    """Copy real-token MLP inputs to CPU without modifying the forward pass."""
    import torch
    handles = []
    def hook(index):
        def receive(module, args, output):
            require(index not in captured, f"Layer {index} ran twice")
            values = output.detach()[mask].to(device="cpu").contiguous()
            require(values.dtype == torch.bfloat16 and torch.isfinite(values).all().item(), "Invalid activation dtype/values")
            captured[index] = values
        return receive
    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.post_attention_layernorm.register_forward_hook(hook(i)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def pack_batch(rows, pad_id, pad_length=None):
    import torch
    length = pad_length or max(row["token_count"] for row in rows)
    require(length >= max(row["token_count"] for row in rows), "Padding length truncates an input")
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long, device="cuda:0")
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        n = row["token_count"]
        ids[i, :n] = torch.tensor(row["input_ids"], device=ids.device)
        mask[i, :n] = True
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(~mask, 0)
    return dict(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)


def validate_hooks(model, rows, pad_id, pad_length=None):
    import torch
    # A short synthetic batch checks all six headers, unequal lengths and padding.
    short = [{**r, "input_ids": r["input_ids"][:min(r["content_end"], r["content_start"] + 7 + i)] + r["input_ids"][-1:]} for i, r in enumerate(rows[:6])]
    short = [{**r, "token_count": len(r["input_ids"])} for r in short]
    inputs = pack_batch(short, pad_id, pad_length)
    current, mlp_inputs, handles = {}, {}, []
    def mlp_hook(i):
        def receive(module, args):
            mlp_inputs[i] = args[0].detach()[inputs["attention_mask"]].cpu()
        return receive
    with torch.inference_mode():
        reference = model(**inputs, logits_to_keep=1).logits.cpu()
        try:
            handles = [layer.mlp.register_forward_pre_hook(mlp_hook(i)) for i, layer in enumerate(model.model.layers)]
            with capture_pre_mlp(model, inputs["attention_mask"], current):
                result = model(**inputs, logits_to_keep=1).logits.cpu()
        finally:
            for handle in handles:
                handle.remove()
    require(torch.isfinite(reference).all().item() and reference.float().std().item() > .01, "Invalid reference logits")
    require(torch.equal(reference, result), "Hooks changed model logits")
    require(set(current) == set(range(LAYERS)) == set(mlp_inputs), "Not all layers were captured")
    require(all(torch.equal(current[i], mlp_inputs[i]) for i in current), "Capture point differs from MLP input")
    require(len(current[0]) == sum(r["token_count"] for r in short), "Padding was captured")
    return dict(exact_logits=True, exact_mlp_inputs=True, layers=LAYERS, padding_excluded=True,
                captured_tokens=len(current[0]), padded_sequence_length=inputs["input_ids"].shape[1],
                max_logit_difference=0.0)


def expected_shard_metadata(meta, shard):
    return {"signature": meta["signature"], "shard_id": str(shard["shard_id"]), "dtype": "bfloat16"}


def check_shard(path, meta, shard):
    from safetensors import safe_open
    import torch
    with safe_open(path, framework="pt", device="cpu") as stream:
        require(stream.metadata() == expected_shard_metadata(meta, shard), f"Wrong shard metadata: {path}")
        expected = {f"pre_mlp.layer_{i:02d}" for i in range(LAYERS)} | {"input_ids", "label_ids", "message_ids", "message_offsets"}
        require(set(stream.keys()) == expected, f"Missing shard tensors: {path}")
        for i in range(LAYERS):
            sl = stream.get_slice(f"pre_mlp.layer_{i:02d}")
            require(sl.get_shape() == [shard["tokens"], WIDTH] and sl.get_dtype() == "BF16", "Wrong activation shape/dtype")
        offsets = stream.get_tensor("message_offsets")
        require(offsets[0].item() == 0 and offsets[-1].item() == shard["tokens"], "Invalid offsets")
        require(len(offsets) == shard["message_end"] - shard["message_start"] + 1, "Missing message offsets")
        require(torch.all(offsets[1:] > offsets[:-1]).item(), "Non-increasing offsets")


def extract(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from safetensors.torch import save_file
    out = args.output
    meta, rows = read_plan(out)
    pad_length = None if args.dynamic_padding else max(row["token_count"] for row in rows)
    require(0 <= args.worker_index < args.workers, "Invalid worker index")
    require(meta["model_identity"] == model_identity(args.model), "Model/tokenizer identity changed")
    (out / "shards").mkdir(exist_ok=True)
    (out / "receipts").mkdir(exist_ok=True)
    with lock(out / f"worker-{args.worker_index}.lock"):
        plan = [s for s in meta["shards"] if s["shard_id"] % args.workers == args.worker_index]
        pending = []
        for shard in plan:
            path = out / "shards" / f"shard-{shard['shard_id']:05d}.safetensors"
            if path.exists():
                check_shard(path, meta, shard)
            else:
                pending.append(shard)
        if not pending:
            emit("worker_cached", worker=args.worker_index)
            return
        torch.set_num_threads(args.cpu_threads)
        started = time.monotonic()
        emit("loading_model", worker=args.worker_index, pending_shards=len(pending))
        model = load_model(args)
        pad_id = model.config.pad_token_id
        validation = validate_hooks(model, rows, pad_id, pad_length)
        atomic_json(out / f"validation-worker-{args.worker_index}.json", validation)
        backend = dict(
            model_path=str(args.model.resolve()), kernel_revision=KERNEL_REVISION,
            collector_sha256=sha(Path(__file__).read_bytes()), python=sys.version,
            gpu=torch.cuda.get_device_name(0), device=args.device, weight_format="MXFP4",
            activation_dtype="bfloat16", attention="eager", batch_size=args.batch_size,
            padded_sequence_length=pad_length, cpu_threads=args.cpu_threads,
            triton_cache_dir=os.environ.get("TRITON_CACHE_DIR"),
            versions={k: importlib.metadata.version(k) for k in ["torch", "transformers", "datasets", "kernels", "triton", "safetensors", "accelerate"]})
        staging_proof = args.model / "staging-proof.json"
        if staging_proof.exists():
            backend["model_staging"] = json.loads(staging_proof.read_text())
        atomic_json(out / f"backend-worker-{args.worker_index}.json", backend)
        emit("hook_validation_passed", worker=args.worker_index, **validation)
        for shard in pending:
            shard_started = time.monotonic()
            group = rows[shard["message_start"]:shard["message_end"]]
            pieces = {i: [] for i in range(LAYERS)}
            with torch.inference_mode():
                for start in range(0, len(group), args.batch_size):
                    inputs = pack_batch(group[start:start + args.batch_size], pad_id, pad_length)
                    captured = {}
                    with capture_pre_mlp(model, inputs["attention_mask"], captured):
                        model.model(**inputs)
                    require(set(captured) == set(range(LAYERS)), "A layer was not captured")
                    for i in range(LAYERS):
                        pieces[i].append(captured[i])
                    del captured, inputs
            tensors = {f"pre_mlp.layer_{i:02d}": torch.cat(pieces.pop(i), dim=0) for i in range(LAYERS)}
            offsets, input_ids, label_ids = [0], [], []
            for row in group:
                input_ids.extend(row["input_ids"])
                label_ids.extend([-1] * row["content_start"] + [row["label_id"]] * (row["content_end"] - row["content_start"]) + [-1] * (row["token_count"] - row["content_end"]))
                offsets.append(len(input_ids))
            tensors.update(input_ids=torch.tensor(input_ids, dtype=torch.int32),
                           label_ids=torch.tensor(label_ids, dtype=torch.int8),
                           message_ids=torch.tensor([r["message_id"] for r in group], dtype=torch.int64),
                           message_offsets=torch.tensor(offsets, dtype=torch.int64))
            path = out / "shards" / f"shard-{shard['shard_id']:05d}.safetensors"
            tmp = path.with_suffix(".safetensors.tmp")
            save_file(tensors, tmp, metadata=expected_shard_metadata(meta, shard))
            with tmp.open("rb") as stream:
                os.fsync(stream.fileno())
            tmp.replace(path)
            del tensors
            check_shard(path, meta, shard)
            receipt = dict(**shard, signature=meta["signature"], bytes=path.stat().st_size,
                           worker=args.worker_index, seconds=time.monotonic() - shard_started,
                           batch_size=args.batch_size, padded_sequence_length=pad_length,
                           cpu_threads=args.cpu_threads, all_layers_finite=True)
            atomic_json(out / "receipts" / f"shard-{shard['shard_id']:05d}.json", receipt)
            progress = dict(worker=args.worker_index, last_shard=shard["shard_id"],
                            elapsed_seconds=time.monotonic() - started, gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            atomic_json(out / f"progress-worker-{args.worker_index}.json", progress)
            emit("shard_saved", **receipt, **{k: v for k, v in progress.items() if k != "worker"})
        emit("worker_complete", worker=args.worker_index, seconds=time.monotonic() - started)


def verify(args):
    from verify_saved import verify_all
    return verify_all(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "verify"])
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c4-samples", type=int, default=1000)
    parser.add_argument("--dolma-samples", type=int, default=1000)
    parser.add_argument("--c4-revision", default="1588ec454efa1a09f29cd18ddd04fe05fc8653a2")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--shuffle-buffer", type=int, default=50000)
    parser.add_argument("--max-body-tokens", type=int, default=1024)
    parser.add_argument("--shard-messages", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--dynamic-padding", action="store_true", help="Use per-batch max length instead of the dataset maximum")
    args = parser.parse_args()
    require(min(args.c4_samples, args.dolma_samples, args.max_body_tokens, args.shard_messages, args.batch_size, args.workers, args.cpu_threads) > 0, "Sizes must be positive")
    {"prepare": prepare, "extract": extract, "verify": verify}[args.stage](args)


if __name__ == "__main__":
    main()
