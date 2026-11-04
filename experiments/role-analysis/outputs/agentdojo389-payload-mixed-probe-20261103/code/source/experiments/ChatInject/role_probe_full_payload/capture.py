"""Trusted payload spans and read-only native-BF16 pre-o_proj capture."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np

from experiments.ChatInject.role_probe.runtime import normalize_messages
from experiments.ChatInject.role_probe_benign.analyze import native_input_spans
from experiments.ChatInject.role_probe_consolidation.build import flexible_text

LAYERS, HEADS, HEAD_DIM = 36, 64, 64
ALL_HEAD_IDS = np.arange(LAYERS * HEADS, dtype=np.int64)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def bf16_bits_to_fp32(bits):
    bits = np.asarray(bits)
    require(bits.dtype == np.uint16, "Native BF16 storage must be uint16 bits")
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def selected_fp16(bits, head_ids):
    """Select ordered global head IDs, then reproduce training's FP16 cast."""
    bits = np.asarray(bits)
    heads = np.asarray(head_ids)
    require(bits.ndim == 4 and bits.shape[1:] == (LAYERS, HEADS, HEAD_DIM), "Wrong all-head BF16 layout")
    require(heads.ndim == 1 and heads.dtype.kind in "iu" and bool(heads.size)
            and len(np.unique(heads)) == len(heads) and bool(((heads >= 0) & (heads < LAYERS * HEADS)).all()),
            "Invalid ordered head IDs")
    values = bf16_bits_to_fp32(bits.reshape(len(bits), LAYERS * HEADS, HEAD_DIM)[:, heads, :])
    require(bool(np.isfinite(values).all()) and bool((np.abs(values) <= np.finfo(np.float16).max).all()),
            "Selected native heads are not representable in FP16")
    return np.ascontiguousarray(values.astype(np.float16).reshape(len(bits), len(heads) * HEAD_DIM))


@dataclass(frozen=True)
class PayloadSelection:
    input_token_indices: tuple[int, ...]
    goal_token_indices: tuple[int, ...]
    stem_token_indices: tuple[int, ...]
    payload_char_span: tuple[int, int]
    goal_char_span: tuple[int, int]
    tool_message_index: int
    native_outer_frame_index: int

    def metadata(self):
        return {name: list(value) if isinstance(value, tuple) else value for name, value in vars(self).items()}


def locate_first_payload(item, tokenizer, *, prompt_date="2026-10-21"):
    """Find the first full payload only inside a template-proved real tool body.

    The complete input is already tokenized.  Neither the payload nor any
    fragment is re-tokenized; indices come from its original offset mapping.
    Literal fake Harmony headers inside messages cannot create tool chunks.
    """
    metadata = item.metadata
    payload, goal, template = (metadata.get(key) for key in ("payload_text", "actual_goal", "payload_template"))
    require(all(isinstance(value, str) and value for value in (payload, goal, template)),
            "Payload metadata require nonempty payload_text/actual_goal/payload_template")
    require(template.count("{goal}") == 1 and template.endswith("{goal}")
            and template.format(goal=goal) == payload, "Payload metadata do not describe stem plus exact goal")
    prefix, suffix = template.split("{goal}")
    pattern = re.compile(flexible_text(prefix) + "(?P<goal>" + flexible_text(goal) + ")" + flexible_text(suffix))
    normalized = normalize_messages(item.messages)
    request = dict(rendered_prompt=item.prompt, normalized_messages=normalized, tools=item.tools,
                   generation=dict(prompt_date=prompt_date, reasoning_effort="medium"))
    spans = native_input_spans(request, tokenizer)
    tool_spans = [span for span in spans if span.tag == "tool"]
    tool_messages = [(index, message) for index, message in enumerate(normalized) if message.get("role") == "tool"]
    require(len(tool_spans) == len(tool_messages), "Trusted tool chunks do not map one-to-one to original messages")
    offsets = np.asarray(item.offsets, dtype=np.int64)
    require(offsets.shape == (len(item.ids), 2), "Whole-input token offsets have wrong shape")
    for span, (message_index, message) in zip(tool_spans, tool_messages):
        body, content = item.prompt[span.start:span.end], message["content"]
        require(json.loads(body) == content and body == json.dumps(content, ensure_ascii=False),
                "Trusted native JSON tool body differs from the actual tool message")
        match = pattern.search(content)
        if match is None:
            continue
        character_map, used = [], 0
        for character in content:
            encoded = json.dumps(character, ensure_ascii=False)[1:-1]
            character_map.append((used, used + len(encoded)))
            used += len(encoded)
        def original_span(a, b):
            return span.start + 1 + character_map[a][0], span.start + 1 + character_map[b - 1][1]
        start, end = original_span(*match.span())
        goal_start, goal_end = original_span(*match.span("goal"))
        indices = tuple(int(i) for i in np.flatnonzero((offsets[:, 1] > start) & (offsets[:, 0] < end)
                                                       & (offsets[:, 1] > offsets[:, 0])))
        require(bool(indices), "A full observed payload has no original token positions")
        require(offsets[indices[0], 0] <= start and offsets[indices[-1], 1] >= end,
                "Original token offsets do not cover the complete payload")
        # A token overlapping the stem/goal boundary belongs to the goal part.
        goals = tuple(i for i in indices if offsets[i, 1] > goal_start and offsets[i, 0] < goal_end)
        stems = tuple(i for i in indices if i not in set(goals))
        require(bool(goals) and bool(stems), "Payload must contain both stem and goal token positions")
        return PayloadSelection(indices, goals, stems, (start, end), (goal_start, goal_end), message_index, span.frame_index)
    return None


class NativeAllHeadCollector:
    """Capture selected token rows from all 36 layers, without casting BF16."""
    def __init__(self):
        self.enabled = False
        self.positions = np.empty(0, np.int64)
        self.bits = None
        self.seen = set()
        self._indices = {}

    def begin(self, positions):
        raw = np.asarray(positions)
        require(raw.ndim == 1 and raw.dtype.kind in "iu" and bool(raw.size)
                and len(np.unique(raw)) == len(raw) and bool((raw >= 0).all()), "Invalid all-head capture positions")
        self.positions = raw.astype(np.int64, copy=True)
        self.bits = np.empty((len(raw), LAYERS, HEADS, HEAD_DIM), np.uint16)
        self.seen, self._indices, self.enabled = set(), {}, True

    def hook(self, layer):
        require(0 <= layer < LAYERS, "Invalid layer")
        def receive(module: Any, args):
            if not self.enabled:
                return None
            import torch
            value = args[0]
            require(layer not in self.seen and value.ndim == 3 and value.shape[-1] == HEADS * HEAD_DIM,
                    "Duplicate layer call or wrong native pre-o_proj layout")
            require(value.dtype == torch.bfloat16, "All-head capture requires native BF16 activations, without a cast")
            require(int(self.positions.max()) < value.shape[0] * value.shape[1], "All-head position exceeds native forward")
            device = str(value.device)
            if device not in self._indices:
                self._indices[device] = torch.tensor(self.positions, dtype=torch.long, device=value.device)
            sample = value.detach().reshape(-1, HEADS, HEAD_DIM).index_select(0, self._indices[device])
            require(bool(torch.isfinite(sample).all()), "Native all-head capture contains nonfinite values")
            self.bits[:, layer] = sample.contiguous().view(torch.uint16).cpu().numpy()
            self.seen.add(layer)
            return None
        return receive

    def finish(self):
        require(self.enabled and self.seen == set(range(LAYERS)) and self.bits is not None,
                "Native all-head capture did not cover all 36 layers")
        self.enabled, self._indices = False, {}
        return self.bits

    def install(self, model):
        handles = []
        try:
            for layer in range(LAYERS):
                handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(self.hook(layer)))
            return handles
        except BaseException:
            for handle in handles:
                handle.remove()
            raise
