"""CPU-only outer-frame readers that preserve opaque native tool names.

The official template may concatenate a model-produced function name containing
literal Harmony strings into a real header. Those bytes remain unmodified.
Tool ownership comes from the template branch / original message objects,
never by parsing or repairing the opaque name as another channel or frame.
"""
from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace

BEGIN = "\ue000FULLPAYLOAD_REAL_TOOL_BEGIN_V2\ue001"
END = "\ue000FULLPAYLOAD_REAL_TOOL_END_V2\ue001"


def require(value, message):
    if not value:
        raise ValueError(message)


def tool_spans_from_template(request, tokenizer):
    """Mark only the official tool-body output expression, not any header."""
    prompt = request["rendered_prompt"]
    require(BEGIN not in prompt and END not in prompt, "Tool-body sentinel collision")
    count = 0

    def mark(match):
        nonlocal count
        expression = match.group()
        if "message.content|tojson" not in expression:
            return expression
        require('" to=assistant<|channel|>commentary<|message|>"' in expression
                and expression.count("<|message|>") == expression.count("<|end|>") == 1,
                "Unexpected official tool-body template expression")
        count += 1
        return expression.replace("<|message|>", "<|message|>" + BEGIN).replace("<|end|>", END + "<|end|>")

    template = re.sub(r"{{[\s\S]*?}}", mark, tokenizer.chat_template)
    require(count == 1, "Official tool branch must have one known output expression")
    generation = request.get("generation", {})
    rendered = tokenizer.apply_chat_template(
        request["normalized_messages"], tools=request.get("tools") or None,
        chat_template=template, tokenize=False, add_generation_prompt=True,
        reasoning_effort=generation.get("reasoning_effort", "medium"),
        strftime_now=lambda fmt: generation.get("prompt_date", "2026-10-21"),
    )
    require(rendered.replace(BEGIN, "").replace(END, "") == prompt,
            "Tool-only instrumentation must reproduce the exact saved prompt")
    spans, position, cursor, pending = [], 0, 0, None
    for match in re.finditer(re.escape(BEGIN) + "|" + re.escape(END), rendered):
        position += match.start() - cursor
        if match.group() == BEGIN:
            require(pending is None, "Nested real tool-body instrumentation")
            pending = position
        else:
            require(pending is not None, "Tool-body end without begin")
            spans.append(SimpleNamespace(start=pending, end=position, tag="tool"))
            pending = None
        cursor = match.end()
    require(pending is None, "Unterminated real tool-body instrumentation")
    tools = [message for message in request["normalized_messages"] if message["role"] == "tool"]
    require(len(tools) == len(spans), "Actual tool-message/body count differs")
    for message, span in zip(tools, spans):
        require(prompt[span.start:span.end] == json.dumps(message["content"], ensure_ascii=False),
                "Instrumented real tool body is not the original JSON string")
    return spans


def independent_frames_from_messages(request, tokenizer):
    """Independently rebuild every frame sequentially, preserving all name bytes."""
    messages = copy.deepcopy(request["messages"])
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = "".join(block["text"] for block in content if block.get("type") == "text")
        elif content is None:
            message["content"] = ""
        for call in message.get("tool_calls", []):
            if isinstance(call["function"]["arguments"], str):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    require(messages == request["normalized_messages"], "Independent message normalization differs")
    require(bool(messages) and messages[0]["role"] in {"system", "developer"}, "Missing native system/developer input")
    prompt = request["rendered_prompt"]
    generation = request.get("generation", {})
    prefix = tokenizer.apply_chat_template(
        messages[:1], tools=request.get("tools") or None, tokenize=False,
        add_generation_prompt=False, reasoning_effort=generation.get("reasoning_effort", "medium"),
        strftime_now=lambda fmt: generation.get("prompt_date", "2026-10-21"),
    )
    require(prompt.startswith(prefix), "Native fixed system/developer prefix differs")
    system_header, ending = "<|start|>system<|message|>", "<|end|>"
    require(prefix.startswith(system_header), "Missing native automatic system header")
    system_end = prefix.index(ending, len(system_header))
    frames = [dict(start=len(system_header), end=system_end, frame_index=0, role="system",
                   channel=None, recipient=None, tag="system", structural_tool_call=False)]
    cursor = system_end + len(ending)
    if cursor < len(prefix):
        header = "<|start|>developer<|message|>"
        require(prefix.startswith(header, cursor) and prefix.endswith(ending), "Native developer prefix differs")
        frames.append(dict(start=cursor+len(header), end=len(prefix)-len(ending), frame_index=1,
                           role="developer", channel=None, recipient=None, tag="developer", structural_tool_call=False))
        cursor = len(prefix)
    require(cursor == len(prefix), "Unaccounted native prefix bytes")

    def add(header, body, terminator, *, role, tag, channel=None, recipient=None, structural=False):
        nonlocal cursor
        segment = header + body + terminator
        require(prompt.startswith(segment, cursor), f"Sequential original native frame differs at {cursor}")
        start = cursor + len(header)
        frames.append(dict(start=start, end=start+len(body), frame_index=len(frames), role=role,
                           channel=channel, recipient=recipient, tag=tag, structural_tool_call=structural))
        cursor += len(segment)

    last_tool = None
    following = messages[1:]
    for index, message in enumerate(following):
        role = message["role"]
        if role == "user":
            add("<|start|>user<|message|>", message["content"], ending, role="user", tag="user")
        elif role == "tool":
            require(bool(last_tool), "Tool message without a preceding actual tool call")
            add(f"<|start|>functions.{last_tool} to=assistant<|channel|>commentary<|message|>",
                json.dumps(message["content"], ensure_ascii=False), ending,
                role="functions."+last_tool, channel="commentary", recipient="assistant", tag="tool")
        elif role == "assistant" and "tool_calls" in message:
            future_final = any(other["role"] == "assistant" and "tool_calls" not in other for other in following[index+1:])
            require(not (message.get("content") and message.get("thinking")), "Ambiguous native assistant reasoning fields")
            reasoning = message.get("content") or message.get("thinking")
            if reasoning and not future_final:
                add("<|start|>assistant<|channel|>analysis<|message|>", reasoning, ending,
                    role="assistant", tag="assistant_analysis", channel="analysis")
            call = message["tool_calls"][0]
            function = call.get("function", call)
            last_tool = function["name"]
            subtype = function.get("content_type", "json")
            add(f"<|start|>assistant to=functions.{last_tool}<|channel|>commentary {subtype}<|message|>",
                json.dumps(function["arguments"], ensure_ascii=False), "<|call|>",
                role="assistant", tag="assistant_commentary", channel="commentary",
                recipient="functions."+last_tool, structural=True)
        elif role == "assistant":
            add("<|start|>assistant<|channel|>final<|message|>", message["content"], ending,
                role="assistant", tag="assistant_final", channel="final")
            last_tool = None
        else:
            raise ValueError(f"Unsupported original outer role: {role}")
    require(prompt[cursor:] == "<|start|>assistant", "Unaccounted native frame or changed generation prefix")
    return frames


def install():
    from experiments.ChatInject.role_probe_full_payload import analyze, verify
    analyze.native_input_spans = tool_spans_from_template
    verify.input_frames = independent_frames_from_messages
