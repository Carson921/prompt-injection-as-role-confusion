"""Native GPT-OSS-120B generation and selected-head capture for fresh AgentDojo57.

Initial requests and simulated tool execution come directly from delimeter-token.
Each generated non-delimiter token is forwarded to obtain its OWN hidden state.
The stored outputs are generated under the swapped system, never teacher-forced
historical responses. Raw requests, exact IDs, roles, and receipts are retained.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("system_swap_collection", HERE / "collect.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ROLES = ["system", "developer", "user", "cot", "assistant", "tool"]
RUN_DATE = "2026-11-02"


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sources():
    return {name: base.digest(HERE / name) for name in ("run_fresh.py", "collect.py", "agentdojo_bridge.py")}


def emit(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def render(tok, request):
    original = tok.apply_chat_template(
        request["messages"], tools=request["tools"], tokenize=False,
        add_generation_prompt=True, reasoning_effort="medium",
        strftime_now=lambda pattern: RUN_DATE)
    swapped, proof = base.rotate_system(original)
    return {"call_id": request["call_id"], "original_prompt": original,
            "rendered_prompt": swapped, "input_ids": tok.encode(swapped, add_special_tokens=False),
            "transformation": proof, "raw_request_sha256": signature(request)}


def prepare(args):
    from transformers import AutoTokenizer
    prepared = base.read(args.output / "initial-requests-manifest.json")
    base.require(prepared["task_count"] == 57 and prepared["injections"] == {}, "Expected 57 no-payload initial requests")
    base.require(not (args.output / "manifest.json").exists(), "Frozen run already prepared")
    selection = base.read(args.selection)
    base.require(selection["feature_dim"] == 1024 and len(selection["heads"]) == 16
                 and selection["signature"]["capture"] == "pre-o_proj", "Unexpected selection geometry")
    identity = base.model_identity(args.model, selection, verify_weights=args.verify_weights)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    records = []
    for item in prepared["tasks"]:
        path = args.output / "initial_requests" / item["task_key"] / "request.json"
        base.require(base.digest(path) == item["request_sha256"], "Initial request changed")
        request = base.read(path)
        base.require(request["injections"] == {}, "Unexpected attack payload")
        value = render(tok, request)
        _, _, _, delimiters = base.position_metadata(tok, value["input_ids"])
        base.atomic_json(path.with_name("rendered.json"), value)
        records.append({"task_key": item["task_key"], "request_sha256": item["request_sha256"],
                        "rendered_sha256": base.digest(path.with_name("rendered.json")),
                        "input_tokens": len(value["input_ids"])})
    manifest = {"format": "agentdojo57-system-swap-native-generation-v1", "task_count": 57,
                "suite_tasks": base.EXPECTED, "head_ids": selection["heads"], "feature_dim": 1024,
                "capture": "pre-o_proj", "activation_dtype": "bfloat16", "model_identity": identity,
                "selection_sha256": base.digest(args.selection), "code_sha256": sources(),
                "agentdojo_source_sha256": base.digest(args.output / "agentdojo-source.json"),
                "initial_requests_manifest_sha256": base.digest(args.output / "initial-requests-manifest.json"),
                "initial_requests": records, "run_date": RUN_DATE, "injections": {},
                "generation": {"temperature": 0.0, "do_sample": False, "max_new_tokens": args.max_new_tokens,
                               "reasoning_effort": "medium", "prefill": "entire_sequence",
                               "attention_query_tile": args.query_tile},
                "delimiter_ids": delimiters, "row_filter": "remove only native delimiter tokens; header text retained with separate body mask",
                "output_capture": "each emitted ordinary token forwarded through native decode; its own pre-o_proj state captured",
                "system_rotation": "split immediately after Reasoning: medium; AB -> BA, retaining all whitespace exactly",
                "tools": "unmodified AgentDojo v1.2.1 simulated tools, YAML results; no injected payload",
                "model_backbone_training": False}
    base.atomic_json(args.output / "manifest.json", manifest)
    emit("prepared", tasks=57, initial_input_tokens=sum(r["input_tokens"] for r in records),
         min_input_tokens=min(r["input_tokens"] for r in records), max_input_tokens=max(r["input_tokens"] for r in records))


class NativeCapture:
    """Copy one ordered selected-head block per forward, with one CPU transfer."""
    def __init__(self, model, heads):
        import torch
        self.model, self.heads, self.handles = model, heads, []
        self.layers = sorted({h // 64 for h in heads})
        self.slots = {layer: [i for i, h in enumerate(heads) if h // 64 == layer] for layer in self.layers}
        self.head_indices = {layer: torch.tensor([heads[i] % 64 for i in self.slots[layer]], device="cuda:0")
                             for layer in self.layers}

    def begin(self, ids, delimiter_ids):
        import torch
        self.positions = np.flatnonzero(~np.isin(ids, delimiter_ids))
        self.position_tensor = torch.tensor(self.positions, device="cuda:0", dtype=torch.long)
        self.value = torch.empty((len(self.positions), 16, 64), device="cuda:0", dtype=torch.bfloat16)
        self.seen = set()
        self.expected_ids = list(ids)
        self.root_seen = False

    def root_hook(self, module, args, kwargs):
        actual = kwargs.get("input_ids", args[0] if args else None)
        base.require(actual is not None and actual.shape[0] == 1 and actual[0].tolist() == self.expected_ids,
                     "Observed model input IDs differ from the tokens owning this capture")
        self.root_seen = True

    def hook(self, layer):
        def observe(module, args):
            base.require(layer not in self.seen, "Duplicate hook within a forward")
            self.seen.add(layer)
            if not len(self.positions):
                return
            value = args[0].detach()
            base.require(value.ndim == 3 and value.shape[0] == 1 and value.shape[2] == 4096,
                         "Unexpected pre-o_proj geometry")
            block = value.reshape(-1, 64, 64).index_select(0, self.position_tensor)
            block = block.index_select(1, self.head_indices[layer])
            self.value[:, self.slots[layer], :] = block
        return observe

    def finish(self):
        base.require(self.root_seen and self.seen == set(self.layers), "A required input/head hook did not run")
        return self.value.reshape(len(self.positions), 1024).cpu()

    def __enter__(self):
        self.handles = [self.model.model.register_forward_pre_hook(self.root_hook, with_kwargs=True)]
        self.handles.extend(self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(self.hook(layer))
                            for layer in self.layers)
        return self

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()


def install_tiled_attention(query_tile):
    """Evaluate the unchanged eager attention equation in bounded query tiles.

    The complete sequence still traverses every layer and MoE together. Right
    alignment describes both a full prefill and native growing/sliding caches.
    No dense N-by-N mask is materialized. This has no approximation or truncation.
    """
    import torch
    from transformers.models.gpt_oss import modeling_gpt_oss as implementation
    original = implementation.eager_attention_forward
    def tiled(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        base.require(not module.training and dropout == 0.0, "Capture requires inference attention")
        keys = implementation.repeat_kv(key, module.num_key_value_groups)
        values = implementation.repeat_kv(value, module.num_key_value_groups)
        batch, heads, q_length, width = query.shape
        k_length = keys.shape[-2]
        result = torch.empty((batch, heads, q_length, width), device=query.device, dtype=values.dtype)
        key_positions = torch.arange(k_length, device=query.device)
        for start in range(0, q_length, query_tile):
            end = min(start + query_tile, q_length)
            logits = torch.matmul(query[:, :, start:end], keys.transpose(2, 3)) * scaling
            if attention_mask is not None:
                mask = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[..., start:end, :]
            else:
                query_positions = torch.arange(start, end, device=query.device) + k_length - q_length
                allowed = key_positions[None, :] <= query_positions[:, None]
                if module.sliding_window is not None:
                    allowed &= key_positions[None, :] > query_positions[:, None] - module.sliding_window
                mask = torch.zeros((end - start, k_length), device=query.device, dtype=query.dtype)
                mask.masked_fill_(~allowed, torch.finfo(query.dtype).min)
            logits = logits + mask
            sinks = module.sinks.reshape(1, -1, 1, 1).expand(batch, -1, end - start, -1)
            combined = torch.cat([logits, sinks], dim=-1)
            combined = combined - combined.max(dim=-1, keepdim=True).values
            probabilities = torch.nn.functional.softmax(combined, dim=-1, dtype=combined.dtype)
            weights = probabilities[..., :-1].to(values.dtype)
            result[:, :, start:end] = torch.matmul(weights, values)
        return result.transpose(1, 2).contiguous(), None
    implementation.eager_attention_forward = tiled
    return original


def generate(model, ids, heads, delimiters, query_tile, max_new_tokens, *, progress=None, native_masks=False):
    import torch
    cache, input_blocks, output_blocks, generated, last = None, [], [], [], None
    stops = {200002, 200012, 199999}
    with torch.inference_mode(), NativeCapture(model, heads) as capture:
        masks = None if native_masks else {"full_attention": None, "sliding_attention": None}
        # One entire-sequence prefill preserves the original MoE batch geometry.
        capture.begin(ids, delimiters)
        output = model.model(input_ids=torch.tensor([ids], device="cuda:0"), attention_mask=masks,
                             past_key_values=None, use_cache=True, return_dict=True)
        input_blocks.append(capture.finish())
        cache = output.past_key_values
        last = output.last_hidden_state[:, -1:, :]
        del output
        logits = model.lm_head(last).float()
        for step in range(max_new_tokens):
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if token in stops:
                break
            # The emitted token is now an input to its own decode forward.
            # Capturing the preceding prediction step would shift the states.
            capture.begin([token], delimiters)
            output = model.model(input_ids=torch.tensor([[token]], device="cuda:0"),
                                 attention_mask=masks, past_key_values=cache, use_cache=True, return_dict=True)
            output_blocks.append(capture.finish())
            cache = output.past_key_values
            logits = model.lm_head(output.last_hidden_state[:, -1:, :]).float()
            del output
            if progress and (step + 1) % 128 == 0:
                progress(step + 1)
    input_features = torch.cat(input_blocks)
    output_features = torch.cat(output_blocks) if output_blocks else torch.empty((0, 1024), dtype=torch.bfloat16)
    base.require(bool(torch.isfinite(input_features).all()) and bool(torch.isfinite(output_features).all()),
                 "Nonfinite native activations")
    return generated, input_features, output_features


def parse_response(tok, generated, call_id):
    text = "<|start|>assistant" + tok.decode(generated, skip_special_tokens=False)
    pattern = r"<\|start\|>(.*?)<\|message\|>(.*?)(?=<\|(?:end|return|call)\|>|$)"
    frames = list(re.finditer(pattern, text, flags=re.DOTALL))
    content, tools, status = [], [], "parsed"
    for frame in frames:
        header, body = frame.groups()
        recipient = re.search(r"(?:^|\s)to=functions\.([\w]+)", header)
        if recipient:
            try:
                arguments = json.loads(body.strip())
                base.require(isinstance(arguments, dict), "Tool arguments must be an object")
            except (ValueError, RuntimeError):
                status = "invalid_tool_json"
                content.append(body)
                continue
            tools.append({"name": recipient.group(1), "arguments": arguments,
                          "id": "call_" + hashlib.sha256((call_id + str(len(tools))).encode()).hexdigest()[:20]})
        elif "<|channel|>final" in header or "<|channel|>commentary" in header:
            content.append(body)
    if not tools and not content:
        status = "no_final_or_tool_frame"
        content = [frames[-1].group(2) if frames else tok.decode(generated, skip_special_tokens=False)]
    return {"call_id": call_id, "content": "\n".join(content) or None,
            "tool_calls": tools, "parse_status": status}


def metadata(tok, full_ids, delimiter_ids):
    keep, body, roles, actual = base.position_metadata(tok, full_ids)
    base.require(actual == delimiter_ids, "Delimiter inventory differs")
    role_ids = np.full(len(full_ids), -1, dtype=np.int8)
    for i, role in enumerate(ROLES):
        role_ids[roles == role] = i
    role_ids[np.char.startswith(roles, "functions.")] = ROLES.index("tool")
    return keep, body, role_ids


def validate_native(model, tok, ids, heads, delimiters, query_tile):
    import torch
    # Cross the 128-token sliding-attention window as well as chunk boundaries.
    short = ids[:384]
    tensor = torch.tensor([short], device="cuda:0")
    with torch.inference_mode():
        before = model.model(input_ids=tensor, use_cache=False).last_hidden_state.cpu()
        with NativeCapture(model, heads) as capture:
            capture.begin(short, delimiters)
            after = model.model(input_ids=tensor, use_cache=False).last_hidden_state.cpu()
            selected = capture.finish()
    base.require(torch.equal(before, after), "Capture changed model output")
    # The reference follows the SAME native prefill/decode geometry; a full
    # teacher-forced replay has different BF16/MXFP4 MoE batching and is not a
    # valid bit-exact reference for generated own-token states.
    generated_reference, reference_input, reference_output = generate(
        model, short, heads, delimiters, 0, 16, native_masks=True)
    install_tiled_attention(query_tile)
    generated, actual_input, decoded = generate(model, short, heads, delimiters, 0, 16)
    base.require(torch.equal(selected, reference_input), "Original cached/full prefills differ")
    base.require(torch.equal(reference_input, actual_input), "Query-tiled prefill differs from the original eager model")
    base.require(generated == generated_reference and torch.equal(reference_output, decoded),
                 "Query-tiled native decoding differs from the original eager model")
    return {"hooks_output_bitexact": True, "prefill_max_absolute_difference": float((reference_input.float() - actual_input.float()).abs().max()),
            "native_decode_own_token_max_absolute_difference": float((reference_output.float() - decoded.float()).abs().max()) if decoded.numel() else 0,
            "native_decode_input_ids_verified": True, "own_token_alignment_checked": True,
            "prefill_and_native_decode_bitexact_to_original_eager": True,
            "validation_input_tokens": len(short), "validation_query_tile": query_tile,
            "validation_generated_tokens": len(generated), "generated_non_delimiter_rows": len(decoded)}


def live_pid(pid, start_ticks):
    if not pid:
        return False
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] not in ("Z", "X") and fields[19] == str(start_ticks)
    except FileNotFoundError:
        return False


def wait_gpu(args):
    first = True
    while True:
        live = live_pid(args.wait_for_pid, args.wait_for_start_ticks)
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(args.device)], text=True)
        free = int(raw.strip())
        if not live and free >= 71680:
            return
        status = {"state": "waiting_for_gpu", "other_pid": args.wait_for_pid,
                  "other_pid_start_ticks": args.wait_for_start_ticks, "other_process_verified_live": live,
                  "free_mib": free, "checked_at_unix": time.time(), "runner_pid": os.getpid()}
        base.atomic_json(args.output / "progress.json", status)
        if first:
            emit("waiting_for_gpu", **status)
            first = False
        time.sleep(5)


def save_call(args, manifest, tok, request, rendered, generated, inp, out, seconds):
    import torch
    from safetensors.torch import save_file
    directory = args.output / "calls" / request["call_id"]
    ids, n = rendered["input_ids"] + generated, len(rendered["input_ids"])
    keep, body, roles = metadata(tok, np.asarray(ids), manifest["delimiter_ids"])
    positions = np.flatnonzero(keep)
    cut = int((positions < n).sum())
    base.require(inp.shape == (cut, 1024) and out.shape == (len(positions) - cut, 1024), "Native features/token counts differ")
    arrays = {"head_ids": torch.tensor(manifest["head_ids"])}
    for scope, features, ps, shift in (("input", inp, positions[:cut], 0), ("output", out, positions[cut:], n)):
        arrays.update({scope + "_features": features.contiguous(), scope + "_token_ids": torch.tensor(np.asarray(ids)[ps]),
                       scope + "_positions": torch.tensor(ps - shift), scope + "_body_mask": torch.tensor(body[ps]),
                       scope + "_role_ids": torch.tensor(roles[ps])})
    temporary = directory / "states.safetensors.tmp"
    save_file(arrays, str(temporary), metadata={"capture": "pre-o_proj", "output_capture": "native_decode_own_token_forward"})
    temporary.replace(directory / "states.safetensors")
    response = parse_response(tok, generated, request["call_id"])
    result = {**rendered, "status": "complete", "input_tokens": n, "generated_ids": generated,
              "generated_text": tok.decode(generated, skip_special_tokens=False), "head_ids": manifest["head_ids"],
              "feature_dim": 1024, "roles": ROLES, "delimiter_ids": manifest["delimiter_ids"],
              "input_rows": cut, "output_rows": len(out), "input_body_rows": int(body[positions[:cut]].sum()),
              "output_body_rows": int(body[positions[cut:]].sum()),
              "termination": "tool_call" if generated[-1] == 200012 else "return" if generated[-1] == 200002 else "endoftext" if generated[-1] == 199999 else "max_new_tokens",
              "output_capture": "native_decode_own_token_forward", "parse_status": response["parse_status"],
              "model_input_ids_verified_per_forward": True, "prefill_mode": "entire_sequence",
              "attention_query_tile": manifest["generation"]["attention_query_tile"]}
    base.atomic_json(directory / "request.json", result)
    base.atomic_json(directory / "response.json", response)
    receipt = {"call_id": request["call_id"], "raw_request_sha256": signature(request),
               "manifest_sha256": base.digest(args.output / "manifest.json"),
               "files": {name: base.digest(directory / name) for name in ("states.safetensors", "request.json", "response.json", "original_request.json")},
               "input_rows": cut, "output_rows": len(out), "seconds": seconds}
    base.atomic_json(directory / "receipt.json", receipt)
    return response, receipt


def run(args):
    manifest = base.read(args.output / "manifest.json")
    base.require(manifest["code_sha256"] == sources(), "Prepared source code changed")
    base.require(base.digest(args.selection) == manifest["selection_sha256"], "Selection changed")
    base.require(base.digest(args.output / "agentdojo-source.json") == manifest["agentdojo_source_sha256"], "AgentDojo source manifest changed")
    wait_gpu(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import importlib.metadata
    from transformers import AutoTokenizer
    torch.set_num_threads(args.cpu_threads)
    base.model_identity(args.model, base.read(args.selection), verify_weights=args.verify_weights)
    base.atomic_json(args.output / "progress.json", {"state": "loading_model", "runner_pid": os.getpid(), "unix_time": time.time()})
    model = base.load_model(args)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    first = base.read(args.output / "initial_requests" / manifest["initial_requests"][0]["task_key"] / "rendered.json")
    validation = validate_native(model, tok, first["input_ids"], manifest["head_ids"], manifest["delimiter_ids"],
                                 manifest["generation"]["attention_query_tile"])
    base.atomic_json(args.output / "backend.json", {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                     "transformers": importlib.metadata.version("transformers"), "attention": "original eager equation, 256-query tiles, entire-sequence MoE prefill", "weights": "native MXFP4",
                     "kernel_revision": base.KERNEL_REVISION, "activation_dtype": "bfloat16", **validation})
    emit("model_validated", **validation)
    command = [str(args.bridge_python), str(HERE / "agentdojo_bridge.py"), "run", "--repo", str(args.repo), "--output", str(args.output)]
    with (args.output / "bridge.log").open("a") as errors:
        bridge = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)
        completed, calls = set(), 0
        try:
            for line in bridge.stdout:
                event = json.loads(line)
                if event["event"] == "request":
                    request = base.read(event["path"])
                    directory = args.output / "calls" / request["call_id"]
                    if (directory / "receipt.json").exists():
                        receipt = base.read(directory / "receipt.json")
                        base.require(receipt["manifest_sha256"] == base.digest(args.output / "manifest.json")
                                     and receipt["raw_request_sha256"] == signature(request), "Resumed request provenance differs")
                        for name, expected in receipt["files"].items():
                            base.require(base.digest(directory / name) == expected, "Resumed call artifact changed")
                        response = base.read(directory / "response.json")
                    else:
                        rendered = render(tok, request)
                        generation = manifest["generation"]
                        started = time.monotonic()
                        def progress(step):
                            base.atomic_json(args.output / "progress.json", {"state": "generating", "call_id": request["call_id"],
                                             "generated_tokens": step, "completed_tasks": len(completed), "runner_pid": os.getpid(), "unix_time": time.time()})
                        generated, inp, out = generate(model, rendered["input_ids"], manifest["head_ids"], manifest["delimiter_ids"],
                                                      generation["attention_query_tile"], generation["max_new_tokens"], progress=progress)
                        response, receipt = save_call(args, manifest, tok, request, rendered, generated, inp, out, time.monotonic() - started)
                        emit("call_saved", **{k: receipt[k] for k in ("call_id", "input_rows", "output_rows", "seconds")})
                    calls += 1
                    bridge.stdin.write(json.dumps(response, ensure_ascii=False) + "\n")
                    bridge.stdin.flush()
                elif event["event"] in ("task_complete", "task_reused"):
                    completed.add(event["task_key"])
                    base.atomic_json(args.output / "progress.json", {"state": "running", "completed_tasks": len(completed),
                                     "task_count": 57, "calls_this_process": calls, "last_task": event["task_key"], "runner_pid": os.getpid(), "unix_time": time.time()})
                    emit(**event)
                elif event["event"] == "task_error":
                    raise RuntimeError(str(event))
                elif event["event"] == "all_tasks_complete":
                    base.require(len(completed) == 57, "Missing completed tasks")
            base.require(bridge.wait() == 0, "AgentDojo bridge failed; inspect bridge.log")
        finally:
            if bridge.poll() is None:
                bridge.terminate()
                bridge.wait(timeout=30)
    del model
    torch.cuda.empty_cache()
    verify(args)


def verify(args):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    torch.set_num_threads(args.cpu_threads)
    (args.output / "complete.json").unlink(missing_ok=True)
    manifest = base.read(args.output / "manifest.json")
    base.require(manifest["code_sha256"] == sources(), "Collector code differs from frozen run")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    results = sorted((args.output / "tasks").glob("*/*/result.json"))
    base.require({base.read(p)["task_key"] for p in results} == {r["task_key"] for r in manifest["initial_requests"]}, "Incomplete task coverage")
    total_in, total_out, count, limits, utilities, parse_errors, files = 0, 0, 0, 0, 0, 0, []
    for result_path in results:
        task = base.read(result_path)
        utilities += int(task["utility"])
        base.require(task["injections"] == {} and task["calls"] > 0, "Invalid no-payload task result")
        for index in range(task["calls"]):
            directory = args.output / "calls/no_payload" / task["task_key"] / f"call_{index:04d}"
            receipt, request = base.read(directory / "receipt.json"), base.read(directory / "request.json")
            raw = base.read(directory / "original_request.json")
            base.require(receipt["manifest_sha256"] == base.digest(args.output / "manifest.json")
                         and receipt["raw_request_sha256"] == signature(raw) and raw["injections"] == {}, "Call provenance differs")
            for name, expected in receipt["files"].items():
                base.require(base.digest(directory / name) == expected, "Call file integrity failed")
            expected = render(tok, raw)
            for key in ("original_prompt", "rendered_prompt", "input_ids", "transformation"):
                base.require(request[key] == expected[key], "Actual request does not have the required system rotation")
            z = load_file(str(directory / "states.safetensors"))
            base.require(z["head_ids"].tolist() == manifest["head_ids"], "Head ordering differs")
            full = np.asarray(request["input_ids"] + request["generated_ids"])
            keep, body, roles = metadata(tok, full, manifest["delimiter_ids"])
            positions = np.flatnonzero(keep)
            n = request["input_tokens"]
            for scope, ps, shift in (("input", positions[positions < n], 0), ("output", positions[positions >= n], n)):
                features = z[scope + "_features"]
                base.require(features.dtype == torch.bfloat16 and features.shape == (len(ps), 1024)
                             and bool(torch.isfinite(features).all()), "Invalid feature values or geometry")
                base.require(z[scope + "_positions"].tolist() == (ps - shift).tolist()
                             and z[scope + "_token_ids"].tolist() == full[ps].tolist()
                             and z[scope + "_body_mask"].tolist() == body[ps].tolist()
                             and z[scope + "_role_ids"].tolist() == roles[ps].tolist(), "Token/mask alignment differs")
            base.require(request["output_capture"] == "native_decode_own_token_forward", "Unexpected generated state source")
            base.require(request["model_input_ids_verified_per_forward"] and request["prefill_mode"] == "entire_sequence"
                         and request["attention_query_tile"] == manifest["generation"]["attention_query_tile"],
                         "Missing actual model-input or full-sequence prefill provenance")
            total_in += request["input_rows"]
            total_out += request["output_rows"]
            limits += int(request["termination"] == "max_new_tokens")
            parse_errors += int(request["parse_status"] != "parsed")
            count += 1
            files.append({"call_id": request["call_id"], "states_sha256": receipt["files"]["states.safetensors"]})
    base.require(count == len(list((args.output / "calls").glob("*/*/*/call_*/receipt.json"))), "Unexpected extra calls")
    validation = base.read(args.output / "backend.json")
    base.require(validation["hooks_output_bitexact"] and validation["generated_non_delimiter_rows"] > 0
                 and validation["own_token_alignment_checked"] and validation["prefill_and_native_decode_bitexact_to_original_eager"],
                 "Native own-token validation is missing")
    report = {"status": "complete", "tasks": 57, "calls": count, "input_rows": total_in, "output_rows": total_out,
              "feature_dim": 1024, "capture": "pre-o_proj", "dtype": "bfloat16", "head_ids": manifest["head_ids"],
              "manifest_sha256": base.digest(args.output / "manifest.json"), "all_features_finite": True,
              "all_system_swaps_and_token_filters_verified": True, "fresh_model_generation": True,
              "utility_successes": utilities, "max_token_terminations": limits, "nonstandard_parse_calls": parse_errors,
              "native_validation": validation, "files": files, "verified_at_unix": time.time()}
    base.atomic_json(args.output / "complete.json", report)
    base.atomic_json(args.output / "progress.json", {"state": "complete", "tasks": 57, "calls": count, "unix_time": time.time()})
    emit("verified_complete", **{k: report[k] for k in ("tasks", "calls", "input_rows", "output_rows", "max_token_terminations")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "run", "verify"])
    p.add_argument("--output", type=Path, default=HERE / "outputs/gpt-oss-120b-system-swap-fresh57-20261102")
    p.add_argument("--model", type=Path, default=Path("/tmp/wz55191-single-message-pre-mlp-gpt-oss-120b"))
    p.add_argument("--selection", type=Path, default=base.DEFAULT_SELECTION)
    p.add_argument("--repo", type=Path, default=Path.home() / "Github/delimeter-token")
    p.add_argument("--bridge-python", type=Path, default=Path("/scratch/wz55191/venvs/chatinject-table1/bin/python"))
    p.add_argument("--device", default="0")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--query-tile", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--verify-weights", action="store_true")
    p.add_argument("--wait-for-pid", type=int)
    p.add_argument("--wait-for-start-ticks")
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    base.require(args.query_tile > 0 and args.max_new_tokens > 0 and args.cpu_threads > 0, "Invalid run limits")
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"prepare": prepare, "run": run, "verify": verify}[args.stage](args)


if __name__ == "__main__":
    main()
