"""Replay archived AgentDojo calls after a lossless system-body rotation.

prepare requires the original calls/no_payload request.json tree. If available,
archived input.npz files additionally verify the original input token IDs.
extract performs teacher-forced forward passes; archived outputs are NOT new
generations. All Harmony tokens remain in the forward input. Saved rows exclude
delimiter IDs, with a separate body mask allowing removal of plain-text headers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_SELECTION = ROOT / "experiments/role-analysis/outputs/synthetic-agent-120b-20261026/final-v7/formal/head-selection.json"
KERNEL_REVISION = "9fbc1b48e710ac8abf18ea2650ee690d52dc3859"
ANCHOR = "Reasoning: medium"
DELIMITERS = ("<|startoftext|>", "<|endoftext|>", "<|start|>", "<|message|>",
              "<|end|>", "<|return|>", "<|call|>", "<|channel|>", "<|constrain|>", "<|endofprompt|>")
EXPECTED = {"banking": 16, "slack": 21, "travel": 20}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def token_id(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    require(len(ids) == 1 and tok.decode(ids) == text, "Not a native single-token marker: " + text)
    return int(ids[0])


def rotate_system(prompt):
    """Swap B+A at the exact end of the anchor, without stripping whitespace."""
    prefix = "<|start|>system<|message|>"
    start = prompt.find(prefix)
    require(start >= 0, "Native system message not found")
    start += len(prefix)
    end = prompt.find("<|end|>", start)
    require(end >= start, "System message has no ending delimiter")
    body = prompt[start:end]
    require(body.count(ANCHOR) == 1, "Expected exactly one Reasoning: medium in system body")
    split = body.index(ANCHOR) + len(ANCHOR)
    a, b = body[:split], body[split:]
    require(b.strip(), "System suffix after Reasoning: medium is empty")
    swapped = b + a
    return prompt[:start] + swapped + prompt[end:], {
        "system_char_start": start, "system_char_end": end,
        "split_char_in_system": split, "original_system": body,
        "swapped_system": swapped, "rule": "A ends immediately after Reasoning: medium; AB -> BA; no whitespace edits",
    }


def position_metadata(tok, ids):
    """Classify headers/bodies without deleting tokens before inference."""
    start, message = token_id(tok, "<|start|>"), token_id(tok, "<|message|>")
    closes = {token_id(tok, s) for s in ("<|end|>", "<|return|>", "<|call|>")}
    delimiter_ids = {token_id(tok, s) for s in DELIMITERS}
    body_mask = np.zeros(len(ids), dtype=bool)
    roles = np.full(len(ids), "structure", dtype="<U24")
    header, in_header, in_body, role = [], False, False, "structure"
    for i, tid in enumerate(ids):
        if tid == start:
            header, in_header, in_body, role = [], True, False, "structure"
        elif in_header and tid == message:
            label = tok.decode(header, skip_special_tokens=False)
            speaker = label.split("<|channel|>")[0].split()[0]
            role = "cot" if speaker == "assistant" and "<|channel|>analysis" in label else speaker
            in_header, in_body = False, True
        elif tid in closes:
            in_header, in_body, role = False, False, "structure"
        elif in_header:
            header.append(int(tid))
        elif in_body:
            body_mask[i] = tid not in delimiter_ids
            roles[i] = role
    keep = ~np.isin(ids, list(delimiter_ids))
    return keep, body_mask, roles, sorted(delimiter_ids)


def model_identity(model_dir, selection, *, verify_weights=False):
    binding = selection["signature"]["model_binding"]
    assets = {name: digest(model_dir / name) for name in binding["assets_sha256"]}
    require(assets == binding["assets_sha256"], "Model/tokenizer assets differ from head selection")
    expected = binding["weight_shards_sha256"]
    for row in binding["weights"]:
        require((model_dir / row["name"]).stat().st_size == row["bytes"], "Weight shard size differs")
    if verify_weights:
        actual = {name: digest(model_dir / name) for name in expected}
        require(actual == expected, "Weight shard SHA-256 differs from head selection")
    else:
        proof_path = model_dir / "staging-proof.json"
        require(proof_path.exists(), "Use --verify-weights when no local staging proof exists")
        proof = read(proof_path)
        require(proof.get("all_shards_verified") is True, "Incomplete checkpoint staging proof")
        actual = {r["name"]: r["sha256"] for r in proof["shards"]}
        require(actual == expected, "Staged weight hashes differ from head selection")
    return {"assets_sha256": assets, "weight_shards_sha256": actual,
            "weight_verification": "fresh_sha256" if verify_weights else "existing_source_and_copy_staging_proof",
            "model_binding": binding}


def prepare(args):
    from transformers import AutoTokenizer
    source = args.source.resolve()
    require(source.is_dir(), "Original calls/no_payload directory is unavailable: " + str(source))
    selection = read(args.selection)
    require(len(selection["heads"]) == 16 and len(set(selection["heads"])) == 16
            and selection["feature_dim"] == 1024 and selection["signature"]["capture"] == "pre-o_proj",
            "Expected 16 ordered pre-o_proj heads")
    require(not (args.output / "manifest.json").exists(), "Prepared output already exists; use its frozen plan")
    identity = model_identity(args.model, selection, verify_weights=args.verify_weights)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    paths = sorted(source.glob("*/*/call_*/request.json"))
    require(paths, "Expected suite/user_task_N/call_NNNN/request.json under --source")
    tasks, calls = {}, []
    for path in paths:
        relative = path.parent.relative_to(source)
        suite, task, call = relative.parts
        require(suite in EXPECTED and task.startswith("user_task_"), "Unexpected task in source")
        tasks.setdefault(suite, set()).add(task)
        request = read(path)
        require(request["status"] == "complete", "Incomplete archived call: " + str(path))
        prompt = request["rendered_prompt"]
        encoded_prompt = tok.encode(prompt, add_special_tokens=False)
        input_archive = path.parent / "input.npz"
        if input_archive.exists():
            with np.load(input_archive, allow_pickle=False) as z:
                original_ids = z["input_ids"].astype(np.int64).tolist()
            input_verification = "archived input.npz token IDs matched exactly"
        elif "input_ids" in request:
            original_ids = request["input_ids"]
            input_verification = "archived request token IDs matched exactly"
        else:
            original_ids = encoded_prompt
            input_verification = "bound-tokenizer re-encoding of archived prompt; original token count checked; original IDs unavailable"
        require(encoded_prompt == original_ids, "Archived prompt/input IDs disagree with bound tokenizer")
        require(len(original_ids) == request["input_tokens"], "Archived input token count differs")
        swapped, transformation = rotate_system(prompt)
        ids = tok.encode(swapped, add_special_tokens=False)
        generated = request["generated_ids"]
        require(isinstance(generated, list) and all(isinstance(i, int) and 0 <= i < len(tok) for i in generated),
                "Invalid archived output IDs")
        full = np.asarray(ids + generated, dtype=np.int64)
        keep, body, roles, delimiters = position_metadata(tok, full)
        n = len(ids)
        item = {"call_id": str(relative), "task_key": f"{suite}/{task}",
                "source_request": str(path), "source_request_sha256": digest(path),
                "source_input_sha256": digest(input_archive) if input_archive.exists() else None,
                "original_input_verification": input_verification,
                "original_input_tokens": len(original_ids), "input_tokens": n,
                "output_tokens": len(generated), "input_rows": int(keep[:n].sum()),
                "output_rows": int(keep[n:].sum()), "input_body_rows": int(body[:n].sum()),
                "output_body_rows": int(body[n:].sum()), "transformation": transformation,
                "original_prompt": prompt, "rendered_prompt": swapped,
                "input_ids": ids, "generated_ids": generated, "delimiter_ids": delimiters}
        destination = args.output / "prepared" / relative / "request.json"
        atomic_json(destination, item)
        calls.append({k: item[k] for k in ("call_id", "task_key", "input_tokens", "output_tokens",
                                          "input_rows", "output_rows", "input_body_rows", "output_body_rows")}
                     | {"prepared_sha256": digest(destination)})
    require({k: len(v) for k, v in tasks.items()} == EXPECTED, "Source does not contain exactly the 57 requested tasks")
    require(all(v == {f"user_task_{i}" for i in range(EXPECTED[k])} for k, v in tasks.items()),
            "Task ID coverage differs")
    manifest = {"format": "agentdojo-system-swap-replay-v1", "status": "prepared",
                "source": str(source), "selection": str(args.selection.resolve()),
                "selection_sha256": digest(args.selection), "head_ids": selection["heads"],
                "feature_dim": 1024, "capture": "pre-o_proj", "model_identity": identity,
                "tasks": 57, "suite_tasks": EXPECTED, "calls": calls,
                "collection_semantics": "teacher-forced replay of each archived input plus unchanged archived output under rotated system; no new agent rollout or generation",
                "row_filter": "exclude native delimiter IDs only; plain-text headers retained and identified by body_mask",
                "delimiter_tokens_present_during_forward": True, "dtype": "bfloat16",
                "kernel_revision": KERNEL_REVISION, "collector_sha256": digest(__file__)}
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": "prepared", "tasks": 57, "calls": len(calls),
                      "input_rows": sum(x["input_rows"] for x in calls),
                      "output_rows": sum(x["output_rows"] for x in calls)}), flush=True)


@contextmanager
def capture_heads(model, heads, positions, result):
    import torch
    layers = sorted({h // 64 for h in heads})
    seen = set()
    def hook(layer):
        def observe(module, values):
            require(layer not in seen, "Repeated head hook")
            seen.add(layer)
            v = values[0].detach()
            require(v.ndim == 3 and v.shape[0] == 1 and v.shape[2] == 4096, "Unexpected attention geometry")
            slots = [i for i, h in enumerate(heads) if h // 64 == layer]
            selected = torch.tensor([heads[i] % 64 for i in slots], device=v.device)
            p = torch.tensor(positions, device=v.device, dtype=torch.long)
            block = v.reshape(-1, 64, 64).index_select(0, p).index_select(1, selected)
            require(bool(torch.isfinite(block).all()), "Nonfinite head states")
            result[:, slots, :] = block.to("cpu", dtype=torch.bfloat16)
        return observe
    handles = [model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook(layer)) for layer in layers]
    try:
        yield
        require(seen == set(layers), "Missing attention hooks")
    finally:
        for handle in handles:
            handle.remove()


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
        require(not self.quantization_config.dequantize, "Refusing silent BF16 weight expansion")
    Mxfp4HfQuantizer.validate_environment = strict
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=False), device_map={"": "cuda:0"},
        attn_implementation="eager", use_kernels=False,
    ).eval().requires_grad_(False)
    require({type(x.mlp.experts).__name__ for x in model.model.layers} == {"Mxfp4GptOssExperts"},
            "Unexpected expert backend")
    require(model.config.num_hidden_layers == 36 and model.config.hidden_size == 2880, "Wrong model geometry")
    return model


def forward_capture(model, ids, heads, keep, chunk_size):
    import torch
    blocks, cache = [], None
    for start in range(0, len(ids), chunk_size):
        stop = min(start + chunk_size, len(ids))
        positions = np.flatnonzero(keep[start:stop])
        captured = torch.empty((len(positions), len(heads), 64), dtype=torch.bfloat16)
        tensor = torch.tensor([ids[start:stop]], device="cuda:0", dtype=torch.long)
        with torch.inference_mode(), capture_heads(model, heads, positions, captured):
            out = model.model(input_ids=tensor, past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        require(cache is not None, "Incremental causal replay lost its KV cache")
        blocks.append(captured.reshape(len(positions), 1024))
        del out, tensor
    return torch.cat(blocks)


def extract(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    manifest = read(args.output / "manifest.json")
    require(manifest["collector_sha256"] == digest(__file__), "Collector changed since preparation")
    identity = model_identity(args.model, read(args.selection), verify_weights=args.verify_weights)
    require(identity["model_binding"] == manifest["model_identity"]["model_binding"], "Model binding changed")
    require(digest(args.selection) == manifest["selection_sha256"], "Head selection changed")
    heads = manifest["head_ids"]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    torch.set_num_threads(args.cpu_threads)
    free, _ = torch.cuda.mem_get_info()
    require(free >= 70 * 1024**3, "This extraction needs an idle 80GB GPU; another collection is using it")
    model = load_model(args)
    require(model.config.num_attention_heads == 64 and model.config.head_dim == 64, "Wrong attention geometry")
    backend = {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
               "gpu": torch.cuda.get_device_name(0), "kernel_revision": KERNEL_REVISION,
               "chunk_size": args.chunk_size, "dtype": "bfloat16", "weight_format": "native MXFP4",
               "attention": "eager", "model": str(args.model.resolve()),
               "manifest_sha256": digest(args.output / "manifest.json")}
    # Validate chunk alignment and passive hooks against an ordinary full causal forward.
    first = read(args.output / "prepared" / manifest["calls"][0]["call_id"] / "request.json")
    ids = first["input_ids"][:64]
    tensor = torch.tensor([ids], device="cuda:0")
    with torch.inference_mode():
        baseline = model.model(input_ids=tensor, use_cache=False).last_hidden_state.cpu()
        reference = torch.empty((len(ids), 16, 64), dtype=torch.bfloat16)
        with capture_heads(model, heads, np.arange(len(ids)), reference):
            observed = model.model(input_ids=tensor, use_cache=False).last_hidden_state.cpu()
    require(torch.equal(baseline, observed), "Passive hooks changed model output")
    chunked = forward_capture(model, ids, heads, np.ones(len(ids), dtype=bool), 23)
    error = float((chunked.float() - reference.reshape(-1, 1024).float()).abs().max())
    require(torch.allclose(chunked.float(), reference.reshape(-1, 1024).float(), atol=.05, rtol=.05),
            f"Chunked capture differs from full forward: {error}")
    backend.update(hooks_output_bitexact=True, chunk_validation_max_abs_error=error)
    atomic_json(args.output / "backend.json", backend)
    for index, row in enumerate(manifest["calls"]):
        directory = args.output / "states" / row["call_id"]
        receipt_path = directory / "receipt.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
            require(receipt["manifest_sha256"] == backend["manifest_sha256"]
                    and digest(directory / "states.safetensors") == receipt["sha256"], "Invalid resumed artifact")
            continue
        prepared_path = args.output / "prepared" / row["call_id"] / "request.json"
        require(digest(prepared_path) == row["prepared_sha256"], "Prepared call changed")
        request = read(prepared_path)
        ids = np.asarray(request["input_ids"] + request["generated_ids"], dtype=np.int64)
        keep, body, roles, delimiters = position_metadata(tok, ids)
        begin = time.monotonic()
        x = forward_capture(model, ids.tolist(), heads, keep, args.chunk_size)
        positions = np.flatnonzero(keep)
        n = request["input_tokens"]
        cut = int((positions < n).sum())
        arrays = {"input_features": x[:cut].contiguous(), "output_features": x[cut:].contiguous(),
                  "input_token_ids": torch.from_numpy(ids[positions[:cut]]),
                  "output_token_ids": torch.from_numpy(ids[positions[cut:]]),
                  "input_positions": torch.from_numpy(positions[:cut]),
                  "output_positions": torch.from_numpy(positions[cut:] - n),
                  "input_body_mask": torch.from_numpy(body[positions[:cut]]),
                  "output_body_mask": torch.from_numpy(body[positions[cut:]]),
                  "head_ids": torch.tensor(heads)}
        require(x.shape == (row["input_rows"] + row["output_rows"], 1024), "Retained feature count differs")
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "states.safetensors.tmp"
        save_file(arrays, str(temporary), metadata={"capture": "pre-o_proj", "output_semantics": "archived_output_teacher_forced_replay"})
        temporary.replace(directory / "states.safetensors")
        atomic_json(directory / "tokens.json", {"input_roles": roles[positions[:cut]].tolist(),
                                               "output_roles": roles[positions[cut:]].tolist()})
        receipt = {"call_id": row["call_id"], "manifest_sha256": backend["manifest_sha256"],
                   "sha256": digest(directory / "states.safetensors"), "input_rows": cut,
                   "output_rows": len(x) - cut, "seconds": time.monotonic() - begin}
        atomic_json(receipt_path, receipt)
        atomic_json(args.output / "progress.json", {"completed_calls": index + 1, "total_calls": len(manifest["calls"]), **receipt})
        print(json.dumps(receipt), flush=True)
    verify(args)


def verify(args):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(getattr(args, "cpu_threads", 2))
    (args.output / "complete.json").unlink(missing_ok=True)
    manifest = read(args.output / "manifest.json")
    total_input, total_output = 0, 0
    for row in manifest["calls"]:
        directory = args.output / "states" / row["call_id"]
        path = directory / "states.safetensors"
        receipt = read(directory / "receipt.json")
        require(receipt["manifest_sha256"] == digest(args.output / "manifest.json")
                and receipt["sha256"] == digest(path), "Saved artifact integrity failed")
        z = load_file(str(path))
        p = read(args.output / "prepared" / row["call_id"] / "request.json")
        require(z["head_ids"].tolist() == manifest["head_ids"], "Saved head order differs")
        for scope, original in (("input", p["input_ids"]), ("output", p["generated_ids"])):
            x, ids, positions = (z[scope + suffix] for suffix in ("_features", "_token_ids", "_positions"))
            expected_pos = [i for i, t in enumerate(original) if t not in p["delimiter_ids"]]
            require(positions.tolist() == expected_pos and ids.tolist() == [original[i] for i in expected_pos],
                    "Delimiter filter or token alignment differs")
            require(x.shape == (row[scope + "_rows"], 1024) and x.dtype == torch.bfloat16
                    and bool(torch.isfinite(x).all()), "Invalid saved states")
            require(z[scope + "_body_mask"].shape == ids.shape, "Body mask alignment differs")
        total_input += row["input_rows"]
        total_output += row["output_rows"]
    result = {"status": "complete", "tasks": manifest["tasks"], "calls": len(manifest["calls"]),
              "input_rows": total_input, "output_rows": total_output, "feature_dim": 1024,
              "manifest_sha256": digest(args.output / "manifest.json"), "all_rows_finite": True,
              "delimiter_filter_and_token_alignment_verified": True, "verified_at_unix": time.time()}
    atomic_json(args.output / "complete.json", result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "extract", "verify"])
    p.add_argument("--source", type=Path, help="Original calls/no_payload directory")
    p.add_argument("--output", type=Path, default=HERE / "outputs/gpt-oss-120b-system-swap-20261102")
    p.add_argument("--model", type=Path, default=Path("/tmp/wz55191-single-message-pre-mlp-gpt-oss-120b"))
    p.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--verify-weights", action="store_true")
    p.add_argument("--device", default="0")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=512)
    args = p.parse_args()
    require(args.chunk_size > 0 and args.cpu_threads > 0, "Chunk size and CPU threads must be positive")
    if args.stage == "prepare":
        require(args.source is not None, "prepare requires --source")
    {"prepare": prepare, "extract": extract, "verify": verify}[args.stage](args)


if __name__ == "__main__":
    main()
