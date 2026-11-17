"""Native DeepSeek V4 AgentDojo generation and observational role readouts.

The frozen probe experiment's source is imported without modification. This
adapter enables the *actual* causal LM head and the official V4 DynamicCache,
and captures a token only when that token is an input to a real forward.
generate_batch deliberately executes serial B=1 requests: left-padded V4
HCA/CSA equivalence has not been established and is not claimed here.

No model is loaded by importing this module. Production construction loads the
official packed-FP4/FP8 checkpoint across four H100s. Tests may explicitly inject
a tiny CPU model; their identity is marked engineering_test_only.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np


ROLES = ("system", "user", "cot", "assistant", "tool")
LAYERS, HEADS, HEAD_DIM = 43, 64, 512
SELECTED_HEADS = 16
FEATURE_POSITION = "attention-head-output-before-o_a_proj"


class AdapterInfrastructureError(RuntimeError):
    category = "infrastructure"

    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response


class NativeContextOverflow(ValueError):
    category = "model"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def normalize_tool_call(call):
    """Accept semantic AgentDojo FunctionCall dumps or OpenAI function calls."""
    call = copy.deepcopy(call)
    function = call.get("function")
    if isinstance(function, dict):
        name, arguments = function.get("name"), function.get("arguments", "{}")
    else:
        name = function or call.get("name")
        arguments = call.get("args", call.get("arguments", {}))
    require(isinstance(name, str) and bool(name), "Tool call needs a function name")
    if not isinstance(arguments, str):
        arguments = canonical(arguments)
    result = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if call.get("id") is not None:
        result["id"] = str(call["id"])
    return result


def normalize_tools(tools):
    result = []
    for tool in tools or []:
        function = tool.get("function", tool)
        require(isinstance(function, dict) and isinstance(function.get("name"), str), "Malformed function schema")
        result.append({"type": "function", "function": copy.deepcopy(function)})
    return result


def normalize_messages(messages):
    require(isinstance(messages, list) and bool(messages), "messages must be a nonempty list")
    result = []
    for original in messages:
        require(isinstance(original, dict), "Each message must be an object")
        role = original.get("role")
        require(role in ("system", "developer", "user", "assistant", "tool", "latest_reminder"),
                f"Unsupported semantic role: {role}")
        message = copy.deepcopy(original)
        if role == "tool":
            content = message.get("error") or message.get("content", "")
            require(isinstance(content, str), "AgentDojo tool content must remain an unescaped string")
            message["content"] = content
        else:
            require(message.get("content") is None or isinstance(message.get("content"), str),
                    "Message content must be text or null")
            message["content"] = message.get("content") or ""
        if role == "assistant":
            message["reasoning_content"] = message.get("reasoning_content") or ""
            require(isinstance(message["reasoning_content"], str), "reasoning_content must be text")
            if message.get("tool_calls"):
                message["tool_calls"] = [normalize_tool_call(call) for call in message["tool_calls"]]
        if message.get("tools"):
            message["tools"] = normalize_tools(message["tools"])
        result.append(message)
    return result


def normalize_request(request):
    require(isinstance(request, dict), "Request must be an object")
    require(set(request) <= {"messages", "tools", "metadata"}, "Request contains unsupported fields")
    metadata = copy.deepcopy(request.get("metadata", {}))
    require(isinstance(metadata, dict), "metadata must be an object")
    for key in ("case_id", "call_id"):
        require(isinstance(metadata.get(key), str) and bool(metadata[key]), f"metadata.{key} is required")
    # Ensure the exact server/client canonical request hash is well-defined.
    object_sha(request)
    return {"messages": normalize_messages(request.get("messages")),
            "tools": normalize_tools(request.get("tools", [])), "metadata": metadata}


class OfficialCodec:
    def __init__(self, model_dir):
        from tokenizers import Tokenizer
        self.model_dir = Path(model_dir).resolve()
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        encoder_path = self.model_dir / "encoding/encoding_dsv4.py"
        spec = importlib.util.spec_from_file_location("native_dsv4_encoder_" + file_sha(encoder_path)[:16], encoder_path)
        self.encoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.encoder)
        self.special_tokens = {text: self.tokenizer.encode(text, add_special_tokens=False).ids for text in (
            self.encoder.bos_token, self.encoder.eos_token, self.encoder.USER_SP_TOKEN,
            self.encoder.ASSISTANT_SP_TOKEN, self.encoder.thinking_start_token,
            self.encoder.thinking_end_token, self.encoder.dsml_token)}
        require(all(len(ids) == 1 for ids in self.special_tokens.values()), "A native delimiter is not a single token")
        self.eos_token_id = self.special_tokens[self.encoder.eos_token][0]
        self.binding = {"tokenizer_sha256": file_sha(self.model_dir / "tokenizer.json"),
                        "official_encoder_sha256": file_sha(encoder_path), "special_tokens": self.special_tokens}

    def encode(self, messages, tools=None, thinking_mode="thinking", reasoning_effort="low"):
        normalized = normalize_messages(messages)
        schemas = normalize_tools(tools)
        if schemas:
            system = next((m for m in normalized if m["role"] == "system"), None)
            if system is None:
                system = {"role": "system", "content": ""}
                normalized.insert(0, system)
            system["tools"] = schemas
        prompt = self.encoder.encode_messages(normalized, thinking_mode=thinking_mode,
                                              reasoning_effort=reasoning_effort, drop_thinking=False)
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        return {"prompt": prompt, "input_ids": encoded.ids, "offsets": encoded.offsets,
                "normalized_messages": normalized, "tools": schemas}

    def parse(self, completion_text, call_id, thinking_mode="thinking"):
        parsed = self.encoder.parse_message_from_completion_text(completion_text, thinking_mode=thinking_mode)
        for index, call in enumerate(parsed.get("tool_calls") or []):
            call["id"] = f"{call_id}-tool-{index}"
            # Native parser supplies a JSON string. Validate it, but do not
            # repair malformed model output or substitute an older trajectory.
            require(isinstance(json.loads(call["function"]["arguments"]), dict), "Tool arguments must be an object")
        return parsed

    def decode(self, ids):
        return self.tokenizer.decode(list(map(int, ids)), skip_special_tokens=False)


def load_official_codec(model_dir):
    return OfficialCodec(model_dir)


class TokenizerView:
    """Expose ordinary ID lists for independent preflight, without retemplating."""
    def __init__(self, codec):
        self.codec = codec
        self.eos_token_id = codec.eos_token_id

    def encode(self, text, add_special_tokens=False):
        return self.codec.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.codec.tokenizer.decode(list(map(int, ids)), skip_special_tokens=skip_special_tokens)


def as_uint16_bf16(value):
    """Return exact native BF16 bits; never convert a different floating dtype."""
    import torch
    require(value.dtype == torch.bfloat16, f"Expected native BF16 head states, got {value.dtype}")
    return value.detach().contiguous().view(torch.uint16).cpu().numpy()


class ForwardCapture:
    """One forward only. Selected-head order and input-row order are preserved."""
    def __init__(self, model, selected_heads, positions=None, all_head_positions=None, all_head_writer=None,
                 *, num_layers=LAYERS, num_heads=HEADS, head_dim=HEAD_DIM):
        self.model = model
        self.selected = np.asarray(selected_heads, dtype=np.int64)
        require(self.selected.ndim == 1 and len(self.selected) > 0
                and len(np.unique(self.selected)) == len(self.selected), "Invalid selected head list")
        require(self.selected.min() >= 0 and self.selected.max() < num_layers * num_heads, "Selected head out of range")
        self.positions = None if positions is None else np.asarray(positions, dtype=np.int64)
        self.all_head_positions = None if all_head_positions is None else np.asarray(all_head_positions, dtype=np.int64)
        for positions_array in (self.positions, self.all_head_positions):
            require(positions_array is None or (positions_array.ndim == 1 and len(positions_array) > 0
                    and positions_array.min() >= 0), "Capture positions must be nonempty nonnegative vectors")
        require((all_head_writer is None) == (all_head_positions is None), "Full heads need positions and a writer")
        self.all_head_writer = all_head_writer
        self.num_layers, self.num_heads, self.head_dim = num_layers, num_heads, head_dim
        self.layers = set(map(int, self.selected // num_heads))
        if all_head_writer is not None:
            self.layers.update(range(num_layers))
        self.blocks, self.seen, self.handles = {}, set(), []

    def _hook(self, layer):
        def hook(module, args):
            import torch
            require(layer not in self.seen, "A ForwardCapture instance cannot span multiple forwards")
            self.seen.add(layer)
            value = args[0].detach()
            require(value.ndim == 4 and value.shape[0] == 1
                    and value.shape[-2] * value.shape[-1] == self.num_heads * self.head_dim,
                    f"Unexpected native o_a_proj input: {tuple(value.shape)}")
            value = value.reshape(1, value.shape[1], self.num_heads, self.head_dim)[0]
            length = len(value)
            if self.all_head_writer is not None:
                require(self.all_head_positions.max() < length, "Full-head position outside this forward")
                sampled = value.index_select(0, torch.as_tensor(self.all_head_positions, device=value.device))
                require(bool(torch.isfinite(sampled).all()), "Nonfinite full native head states")
                self.all_head_writer(layer, as_uint16_bf16(sampled))
                del sampled
            columns = np.flatnonzero(self.selected // self.num_heads == layer)
            if len(columns):
                heads = torch.as_tensor(self.selected[columns] % self.num_heads, device=value.device)
                selected = value.index_select(1, heads)
                if self.positions is not None:
                    require(self.positions.max() < length, "Selected-head position outside this forward")
                    selected = selected.index_select(0, torch.as_tensor(self.positions, device=value.device))
                require(bool(torch.isfinite(selected).all())
                        and bool((selected.abs() <= np.finfo(np.float16).max).all()), "Invalid FP16 selected head state")
                self.blocks[layer] = selected.to(dtype=torch.float16, device="cpu").numpy()
            return None
        return hook

    def __enter__(self):
        require(not self.handles, "Capture already installed")
        try:
            for layer in sorted(self.layers):
                site = self.model.model.layers[layer].self_attn.o_a_proj
                self.handles.append(site.register_forward_pre_hook(self._hook(layer)))
        except Exception as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def __exit__(self, kind, value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if kind is None:
            require(self.seen == self.layers, "A required native head hook did not execute")

    @property
    def features(self):
        require(self.seen == self.layers and bool(self.blocks), "Capture incomplete")
        count = len(next(iter(self.blocks.values())))
        result = np.empty((count, len(self.selected), self.head_dim), np.float16)
        for layer, block in self.blocks.items():
            result[:, np.flatnonzero(self.selected // self.num_heads == layer)] = block
        return result.reshape(count, -1)


class NativePerformanceCache:
    """Reuse measured native configurations for each *true* query length."""
    def __init__(self, output_dir, cache_dir=None):
        import prepare_deepseek_performance_cache as performance
        self.performance = performance
        self.cache_dir = Path(cache_dir or os.environ.get("TRITON_CACHE_DIR", "~/.cache/triton")).expanduser().resolve()
        actual = Path(os.environ.get("TRITON_CACHE_DIR", "~/.cache/triton")).expanduser().resolve()
        require(self.cache_dir == actual, "performance_cache_dir must equal the actual TRITON_CACHE_DIR")
        self.runtime = performance.initialize_runtime()
        self.sources = {}
        self.seen = set()
        self.events_path = Path(output_dir) / "generation-performance-cache.jsonl"
        for projection in performance.PROJECTIONS:
            key = performance.tuning_key(projection, 8)
            path = performance.native_cache_path(self.cache_dir, self.runtime["identity"], key)
            source = performance.measured_winner(path, key, self.runtime["configs"])
            require(source is not None, "Missing genuinely measured 8-token native reference; run the bootstrap first")
            self.sources[projection] = source

    def identity(self):
        return {"native_identity": self.runtime["identity"], "measured_reference_tokens": 8,
                "sources": self.sources, "policy": "true query S with original measured config; target timings empty"}

    def ensure(self, query_tokens):
        p = self.performance
        query_tokens = int(query_tokens)
        require(query_tokens > 0, "Cannot cache an empty forward")
        if query_tokens in self.seen:
            return
        events = []
        for projection in p.PROJECTIONS:
            key = p.tuning_key(projection, query_tokens)
            path = p.native_cache_path(self.cache_dir, self.runtime["identity"], key)
            if path.exists():
                value = json.loads(path.read_text())
                require(any(config.all_kwargs() == value.get("best") for config in self.runtime["configs"]),
                        f"Existing native performance cache is incompatible: {path}")
                action = "existing"
            else:
                source = self.sources[projection]
                require(file_sha(source["path"]) == source["sha256"], "Measured source cache changed")
                created = p.create_absent_cache(path, {"best": source["best"], "timings": []})
                action = "created" if created else "race_existing"
            events.append({"query_tokens": query_tokens, "projection": projection, "target_key": key,
                           "path": str(path), "action": action, "source_winner_sha256": self.sources[projection]["sha256"]})
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a") as stream:
            for event in events:
                stream.write(canonical(event) + "\n")
        self.seen.add(query_tokens)


def default_payload_locator(messages, **kwargs):
    from deepseek_agentdojo_protocol import locate_first_payload
    return locate_first_payload(messages, **kwargs)


class DeepSeekAgentDojoModel:
    supported_batch_size = 1
    batch_supported = False

    def __init__(self, model_dir, probe_dir, output_dir, **options):
        self.model_dir, self.probe_dir, self.output_dir = map(lambda p: Path(p).resolve(), (model_dir, probe_dir, output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calls_dir = self.output_dir / "calls"
        self.calls_dir.mkdir(exist_ok=True)
        self.max_new_tokens = int(options.pop("max_new_tokens", 1024))
        self.max_model_len = int(options.pop("max_model_len", 32768))
        self.temperature = float(options.pop("temperature", 0))
        self.seed = int(options.pop("seed", 0))
        self.thinking_mode = options.pop("thinking_mode", "thinking")
        self.reasoning_effort = options.pop("reasoning_effort", "low")
        self.locator = options.pop("locator", default_payload_locator)
        devices = options.pop("devices", "0,1,2,3")
        native_cache_dir = options.pop("performance_cache_dir", None)
        hf_cache_dir = options.pop("cache_dir", None)
        self.cpu_test = bool(options.pop("_cpu_test", False))
        injected_model, injected_probe, injected_codec = (options.pop("_model", None), options.pop("_probe", None), options.pop("_codec", None))
        performance_policy = options.pop("_performance_policy", None)
        require(not options, f"Unsupported adapter options: {sorted(options)}")
        require(self.temperature == 0 and self.seed == 0, "This fixed protocol requires greedy temperature=0 and seed=0")
        require(self.thinking_mode == "thinking" and self.reasoning_effort == "low", "Protocol requires thinking+low")
        require(0 < self.max_new_tokens < self.max_model_len, "Invalid model/generation budget")
        self._lock = threading.RLock()
        self._closed = False
        self._exposure_seen = set()
        self.counters = {"calls": 0, "replayed_calls": 0, "input_tokens": 0, "generated_tokens": 0, "full_payload_captures": 0}
        self.codec = injected_codec or load_official_codec(self.model_dir)
        self.tokenizer = TokenizerView(self.codec)
        if injected_probe is None:
            import deepseek_role_probe_train as trainer
            trainer.verify_freeze(self.probe_dir)
            self.probe = trainer.Probe(self.probe_dir / "checkpoint.npz")
        else:
            require(self.cpu_test, "Injected probes are restricted to explicit engineering tests")
            self.probe = injected_probe
        self.selected_heads = list(map(int, self.probe.metadata["heads"]))
        require(tuple(self.probe.roles) == ROLES, "Probe native roles differ from the fixed five-role schema")
        if not self.cpu_test:
            require(len(self.selected_heads) == SELECTED_HEADS and self.probe.feature_dim == SELECTED_HEADS * HEAD_DIM,
                    "Expected the frozen sixteen-head 8192-dimensional probe")
            binding = self.probe.metadata["model_binding"]
            for key, filename in (("config_sha256", "config.json"), ("index_sha256", "model.safetensors.index.json"),
                                  ("tokenizer_sha256", "tokenizer.json"), ("official_encoder_sha256", "encoding/encoding_dsv4.py")):
                require(file_sha(self.model_dir / filename) == binding[key], f"Probe/model binding mismatch: {key}")
        if injected_model is None:
            import deepseek_role_probe_model as original
            self.model = original.load_model(self.model_dir, run_dir=None, devices=devices, cache_dir=hf_cache_dir)
        else:
            require(self.cpu_test, "Injected model restricted to explicit engineering tests")
            self.model = injected_model.eval().requires_grad_(False)
        import torch
        torch.manual_seed(self.seed)
        if not self.cpu_test:
            torch.cuda.manual_seed_all(self.seed)
            import deepseek_role_probe_model as original
            audit = original.audit_model(self.model)
            self.performance = performance_policy or NativePerformanceCache(self.output_dir, native_cache_dir)
        else:
            audit = {"engineering_test_only": True}
            self.performance = performance_policy
        self.geometry = {"num_layers": len(self.model.model.layers),
                         "num_heads": int(self.model.config.num_attention_heads), "head_dim": int(self.model.config.head_dim)}
        self.backend = {"format_version": "deepseek-agentdojo-native-model-v1", "engineering_test_only": self.cpu_test,
                        "model_dir": str(self.model_dir), "probe_dir": str(self.probe_dir),
                        "implementation_sha256": file_sha(__file__), "codec": self.codec.binding,
                        "roles": list(ROLES), "heads": self.selected_heads, "capture": FEATURE_POSITION,
                        "capture_after_inverse_rope": True, "supported_batch_size": 1, "real_batch_supported": False,
                        "forward_target": "DeepseekV4ForCausalLM with logits_to_keep=1", "use_cache": True,
                        "cache": "V4 DynamicCache; new per assistant turn; within-turn incremental decode",
                        "generation": {"max_new_tokens": self.max_new_tokens, "max_model_len": self.max_model_len,
                                       "temperature": self.temperature, "seed": self.seed,
                                       "thinking_mode": self.thinking_mode, "reasoning_effort": self.reasoning_effort},
                        "native_backbone_audit": audit,
                        "performance_cache": None if self.performance is None else self.performance.identity()}
        for name, path in (("probe_checkpoint_sha256", self.probe_dir / "checkpoint.npz"),
                           ("probe_freeze_sha256", self.probe_dir / "freeze.json"),
                           ("loader_implementation_sha256", Path(__file__).with_name("deepseek_role_probe_model.py")),
                           ("probe_consumer_implementation_sha256", Path(__file__).with_name("deepseek_role_probe_train.py")),
                           ("payload_locator_implementation_sha256", Path(__file__).with_name("deepseek_agentdojo_protocol.py")),
                           ("performance_cache_implementation_sha256", Path(__file__).with_name("prepare_deepseek_performance_cache.py"))):
            self.backend[name] = file_sha(path) if path.is_file() else None
        self.identity_sha256 = object_sha(self.backend)
        atomic_json(self.output_dir / "generation-backend.json", self.backend)
        self._restore_exposures()

    def identity(self):
        return copy.deepcopy(self.backend)

    def prepare_query_length(self, query_tokens):
        """Prepare true-S native kernel configs before an independent forward."""
        require(int(query_tokens) > 0, "Query length must be positive")
        if self.performance is not None:
            self.performance.ensure(int(query_tokens))

    def _restore_exposures(self):
        for receipt_path in self.output_dir.rglob("receipt.json"):
            if any(part.startswith(".tmp-") for part in receipt_path.relative_to(self.output_dir).parts):
                continue
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("format_version") != "deepseek-agentdojo-call-v1":
                continue
            if (receipt.get("identity_sha256") == self.identity_sha256
                    and receipt.get("status") in ("complete", "error") and receipt.get("full_payload_captured")):
                for name, expected in receipt["files"].items():
                    require(file_sha(receipt_path.parent / name) == expected,
                            f"Saved first-exposure artifact changed: {receipt_path.parent / name}")
                self._exposure_seen.add(tuple(receipt["episode_key"]))

    def _episode_key(self, metadata):
        folder = self._call_path(metadata["call_id"], metadata.get("recording_dir"))
        scope = folder
        for _ in range(len(Path(metadata["call_id"]).parts) + 1):
            scope = scope.parent
        return str(scope), str(metadata.get("condition", "")), metadata["case_id"]

    def _call_path(self, call_id, recording_dir=None):
        parts = Path(call_id).parts
        require(not Path(call_id).is_absolute() and parts
                and not any(part in (".", "..") or part.startswith(".tmp-") for part in parts), "Invalid call_id")
        path = Path(recording_dir).resolve() if recording_dir is not None else (self.calls_dir / call_id).resolve()
        require(path != self.output_dir and path.is_relative_to(self.output_dir), "recording_dir escapes output directory")
        suffix = ("calls", *parts)
        require(path.parts[-len(suffix):] == suffix, "recording_dir must end in calls/<call_id>")
        scope = path
        for _ in suffix:
            scope = scope.parent
        require(scope.is_relative_to(self.output_dir), "recording scope escapes output directory")
        return path

    def _probabilities(self, features):
        probabilities = np.asarray(self.probe.predict_proba(features), dtype=np.float32)
        require(probabilities.shape == (len(features), len(ROLES)) and np.isfinite(probabilities).all()
                and np.all(probabilities >= 0) and np.allclose(probabilities.sum(1), 1, atol=2e-5), "Invalid role probabilities")
        return probabilities

    def _forward(self, ids, cache, absolute_start, *, full_head_positions=None, full_head_writer=None):
        """Actual cached CausalLM forward, never the historical feature-only _forward."""
        import torch
        self.prepare_query_length(len(ids))
        device = self.model.model.embed_tokens.weight.device
        inputs = torch.as_tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, absolute_start + len(ids)), dtype=torch.long, device=device)
        positions = torch.arange(absolute_start, absolute_start + len(ids), device=device).unsqueeze(0)
        capture = ForwardCapture(self.model, self.selected_heads, all_head_positions=full_head_positions,
                                 all_head_writer=full_head_writer, **self.geometry)
        with torch.inference_mode(), capture:
            output = self.model(input_ids=inputs, attention_mask=attention_mask, position_ids=positions,
                                past_key_values=cache, use_cache=True, logits_to_keep=1,
                                output_attentions=False, output_hidden_states=False, return_dict=True)
        require(output.past_key_values is not None, "Native model did not return a DynamicCache")
        features = capture.features
        return output.logits[:, -1, :], output.past_key_values, features

    def _locate(self, encoded, request):
        metadata = request["metadata"]
        if not metadata.get("actual_goal"):
            return None
        return self.locator(encoded["normalized_messages"], condition=metadata.get("condition"),
                            actual_goal=metadata["actual_goal"], rendered_prompt=encoded["prompt"],
                            offsets=encoded["offsets"], input_ids=encoded["input_ids"],
                            render_callback=lambda messages: self.codec.encode(messages, request["tools"],
                                    self.thinking_mode, self.reasoning_effort)["prompt"])

    def _replay(self, folder, request_sha):
        receipt_path = folder / "receipt.json"
        require(receipt_path.is_file(), "Existing call directory is not a committed call")
        receipt = json.loads(receipt_path.read_text())
        require(receipt["identity_sha256"] == self.identity_sha256
                and receipt["raw_request_sha256"] == request_sha, "Saved call identity/request differs")
        for name, expected in receipt["files"].items():
            require(file_sha(folder / name) == expected, f"Saved call file changed: {name}")
        self.counters["replayed_calls"] += 1
        if receipt.get("full_payload_captured"):
            self._exposure_seen.add(tuple(receipt["episode_key"]))
        return json.loads((folder / "response.json").read_text())

    def _commit(self, temporary, folder, response, *, request_sha, metadata, episode_key,
                full_captured, payload_exposed, status):
        atomic_json(temporary / "response.json", response)
        files = {str(path.relative_to(temporary)): file_sha(path)
                 for path in sorted(temporary.rglob("*")) if path.is_file()}
        receipt = {"format_version": "deepseek-agentdojo-call-v1", "status": status,
                   "identity_sha256": self.identity_sha256, "raw_request_sha256": request_sha,
                   "metadata": metadata, "episode_key": list(episode_key), "full_payload_captured": full_captured,
                   "payload_exposed": payload_exposed, "files": files,
                   "forward_started": response.get("forward_started", True),
                   "input_forward_completed": response.get("input_forward_completed", True),
                   "generated_token_alignment": "probability row j belongs to the forward consuming generated token j",
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(temporary / "receipt.json", receipt)
        folder.parent.mkdir(parents=True, exist_ok=True)
        require(not folder.exists(), "Concurrent call commit collision")
        temporary.replace(folder)
        if full_captured:
            self._exposure_seen.add(episode_key)
            self.counters["full_payload_captures"] += 1
        self.counters["calls"] += 1
        self.counters["input_tokens"] += response.get("input_tokens", 0)
        self.counters["generated_tokens"] += response.get("output_tokens", 0)
        return response

    def _generate_one(self, original_request):
        import torch
        request_sha = object_sha(original_request)
        request = normalize_request(original_request)
        metadata = request["metadata"]
        if not self.cpu_test:
            require(metadata.get("condition") in ("plain", "plain_delimiter"), "Missing native payload condition")
            require(isinstance(metadata.get("actual_goal"), str) and bool(metadata["actual_goal"]),
                    "Native attacker actual_goal is required even before the first tool call")
        call_id, episode_key = metadata["call_id"], self._episode_key(metadata)
        folder = self._call_path(call_id, metadata.get("recording_dir"))
        if folder.exists():
            return self._replay(folder, request_sha)
        temporary = self.calls_dir / (".tmp-" + object_sha(call_id)[:16] + "-" + uuid.uuid4().hex)
        temporary.mkdir()
        atomic_json(temporary / "original_request.json", original_request)
        started = time.monotonic()
        raw_heads = None
        cache = None
        full_captured = False
        forward_started, input_forward_completed = False, False
        payload = None
        ids, generated, generated_probabilities, observed = [], [], [], []
        try:
            encoded = self.codec.encode(request["messages"], request["tools"], self.thinking_mode, self.reasoning_effort)
            ids = list(map(int, encoded["input_ids"]))
            require(len(ids) > 0, "Empty native prompt")
            if len(ids) + self.max_new_tokens > self.max_model_len:
                raise NativeContextOverflow(f"Native context overflow: input {len(ids)} + generation "
                                            f"{self.max_new_tokens} > {self.max_model_len}; no truncation")
            payload = self._locate(encoded, request)
            atomic_json(temporary / "request.json", {"raw_request_sha256": request_sha, "metadata": metadata,
                        "messages": encoded["normalized_messages"], "tools": encoded["tools"],
                        "rendered_prompt": encoded["prompt"], "prompt_sha256": hashlib.sha256(encoded["prompt"].encode()).hexdigest(),
                        "input_tokens": len(ids), "payload": payload, "generation": self.backend["generation"]})
            (temporary / "prompt.txt").write_text(encoded["prompt"])
            capture_payload = payload is not None and episode_key not in self._exposure_seen
            payload_positions = np.asarray(payload["input_token_indices"], dtype=np.int64) if capture_payload else None
            layers_written = set()
            if capture_payload:
                require(len(payload_positions) > 0, "Full payload has no token positions")
                raw_heads = np.lib.format.open_memmap(temporary / "payload-heads-bf16-uint16.npy", mode="w+", dtype=np.uint16,
                        shape=(len(payload_positions), self.geometry["num_layers"], self.geometry["num_heads"], self.geometry["head_dim"]))

                def full_writer(layer, values):
                    raw_heads[:, layer] = values
                    layers_written.add(layer)
            else:
                full_writer = None
            forward_started = True
            logits, cache, input_features = self._forward(ids, None, 0, full_head_positions=payload_positions,
                                                         full_head_writer=full_writer)
            input_probabilities = self._probabilities(input_features)
            input_forward_completed = True
            if capture_payload:
                require(layers_written == set(range(self.geometry["num_layers"])), "Full payload head capture incomplete")
                raw_heads.flush()
                raw_heads._mmap.close()
                raw_heads = None
                np.save(temporary / "payload-selected-fp16.npy", input_features[payload_positions], allow_pickle=False)
                atomic_json(temporary / "payload-capture.json", {"positions": payload_positions.tolist(),
                            "proof": payload, "native_dtype": "bfloat16", "storage_dtype": "uint16",
                            "storage_layout": "[payload_token, layer, head, head_dimension]",
                            "shape": [len(payload_positions), self.geometry["num_layers"], self.geometry["num_heads"], self.geometry["head_dim"]],
                            "selected_heads": self.selected_heads, "capture": FEATURE_POSITION,
                            "capture_after_inverse_rope": True, "role_probabilities": input_probabilities[payload_positions].tolist()})
                full_captured = True
            np.savez(temporary / "input.npz", token_ids=np.asarray(ids, np.int64), token_indices=np.arange(len(ids), dtype=np.int64),
                     offsets=np.asarray(encoded["offsets"], np.int64), probabilities=input_probabilities,
                     observed=np.ones(len(ids), bool), roles=np.asarray(ROLES))
            del input_features, input_probabilities
            finish_reason = "length"
            for step in range(self.max_new_tokens):
                token = int(torch.argmax(logits[0]).item())
                generated.append(token)
                generated_probabilities.append(np.full(len(ROLES), np.nan, np.float32))
                observed.append(False)
                if token == self.codec.eos_token_id:
                    finish_reason = "stop"
                    break
                # Consume the token that was actually sampled. This is also the
                # terminal forward on a normal last token at the length limit.
                logits, cache, features = self._forward([token], cache, len(ids) + step)
                generated_probabilities[-1] = self._probabilities(features)[0]
                observed[-1] = True
                del features
            completion = self.codec.decode(generated)
            probabilities = np.asarray(generated_probabilities, np.float32).reshape(-1, len(ROLES))
            np.savez(temporary / "output.npz", token_ids=np.asarray(generated, np.int64),
                     token_indices=np.arange(len(ids), len(ids) + len(generated), dtype=np.int64),
                     probabilities=probabilities, observed=np.asarray(observed, bool), roles=np.asarray(ROLES))
            (temporary / "completion.txt").write_text(completion)
            if finish_reason == "stop":
                try:
                    parsed = self.codec.parse(completion, call_id, self.thinking_mode)
                    parse_error = None
                except Exception as error:
                    parsed = {"content": None, "reasoning_content": None, "tool_calls": []}
                    parse_error = {"type": "NativeCompletionParseError", "category": "model",
                                   "parser_error_type": type(error).__name__,
                                   "message": str(error)}
            else:
                parsed = {"content": None, "reasoning_content": None, "tool_calls": []}
                parse_error = {"type": "IncompleteNativeCompletion", "category": "model",
                               "message": "Generation budget exhausted before native EOS; completion was not repaired"}
            relative = str(folder.relative_to(self.output_dir))
            response = {"content": parsed.get("content"), "reasoning_content": parsed.get("reasoning_content"),
                        "tool_calls": parsed.get("tool_calls") or [], "finish_reason": finish_reason,
                        "raw_completion": completion, "raw_request_sha256": request_sha,
                        "artifact_refs": {"directory": relative, "receipt": relative + "/receipt.json",
                                          "input": relative + "/input.npz", "output": relative + "/output.npz"},
                        "input_tokens": len(ids), "output_tokens": len(generated), "execution_batch_size": 1,
                        "forward_started": forward_started, "input_forward_completed": input_forward_completed,
                        "full_payload_captured": full_captured, "payload_exposed": payload is not None,
                        "eos_state_policy": "EOS is not forwarded: NaN probability row with observed=false",
                        "seconds": time.monotonic() - started}
            if parse_error:
                response["error"] = parse_error
            return self._commit(temporary, folder, response, request_sha=request_sha, metadata=metadata,
                                episode_key=episode_key, full_captured=full_captured, payload_exposed=payload is not None,
                                status="error" if parse_error else "complete")
        except Exception as error:
            if raw_heads is not None:
                raw_heads.flush()
                raw_heads._mmap.close()
                raw_heads = None
            atomic_json(temporary / "error.json", {"type": type(error).__name__, "message": str(error),
                        "raw_request_sha256": request_sha, "metadata": metadata, "full_payload_captured": full_captured,
                        "seconds": time.monotonic() - started})
            # Retain partial actual sampled tokens even if consuming one failed.
            # Its state, like unconsumed EOS, remains NaN with observed=false.
            if generated:
                np.savez(temporary / "output.npz", token_ids=np.asarray(generated, np.int64),
                         token_indices=np.arange(len(ids), len(ids) + len(generated), dtype=np.int64),
                         probabilities=np.asarray(generated_probabilities, np.float32),
                         observed=np.asarray(observed, bool), roles=np.asarray(ROLES))
            native_failure = isinstance(error, NativeContextOverflow)
            if native_failure:
                (temporary / "prompt.txt").write_text(encoded["prompt"])
                atomic_json(temporary / "request.json", {"raw_request_sha256": request_sha, "metadata": metadata,
                            "messages": encoded["normalized_messages"], "tools": encoded["tools"],
                            "rendered_prompt": encoded["prompt"], "input_tokens": len(ids), "payload": None,
                            "generation": self.backend["generation"], "forward_started": False})
                np.savez(temporary / "unforwarded-input.npz", token_ids=np.asarray(ids, np.int64),
                         offsets=np.asarray(encoded["offsets"], np.int64))
            error_folder = folder if native_failure else temporary
            relative = str(error_folder.relative_to(self.output_dir))
            response = {"content": None, "reasoning_content": None, "tool_calls": [], "finish_reason": "error",
                        "raw_request_sha256": request_sha, "input_tokens": len(ids), "output_tokens": len(generated),
                        "execution_batch_size": 1, "full_payload_captured": full_captured,
                        "forward_started": forward_started, "input_forward_completed": input_forward_completed,
                        "payload_exposed": payload is not None,
                        "artifact_refs": {"directory": relative, "error": relative + "/error.json"},
                        "error": {"type": type(error).__name__, "message": str(error),
                                  "category": "model" if native_failure else "infrastructure"},
                        "seconds": time.monotonic() - started}
            if native_failure:
                response["artifact_refs"]["receipt"] = relative + "/receipt.json"
                return self._commit(temporary, folder, response, request_sha=request_sha, metadata=metadata,
                                    episode_key=episode_key, full_captured=False, payload_exposed=False, status="error")
            atomic_json(temporary / "response.json", response)
            raise AdapterInfrastructureError(str(error), response) from error
        finally:
            if raw_heads is not None:
                raw_heads.flush()
                raw_heads._mmap.close()
            cache = None

    def generate_batch(self, requests):
        require(isinstance(requests, list), "generate_batch expects a list")
        with self._lock:
            require(not self._closed, "Model adapter is closed")
            # A real batch is not claimed until left-padding equivalence for
            # both V4 compressor cache types has been tested end to end.
            try:
                return [self._generate_one(request) for request in requests]
            except AdapterInfrastructureError:
                raise
            except Exception as error:
                raise AdapterInfrastructureError(str(error)) from error

    def close(self):
        with self._lock:
            if self._closed:
                return
            result = {"identity_sha256": self.identity_sha256, "counters": self.counters,
                      "finished_at_utc": datetime.now(timezone.utc).isoformat(), "engineering_test_only": self.cpu_test}
            if not self.cpu_test:
                import deepseek_role_probe_model as original
                result.update(backbone_audit=original.audit_model(self.model), gpu_memory=original.memory_report())
            atomic_json(self.output_dir / "generation-final.json", result)
            self.model = None
            gc.collect()
            if not self.cpu_test:
                import torch
                for device in range(torch.cuda.device_count()):
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
            self._closed = True
