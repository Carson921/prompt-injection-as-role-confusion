"""Role-probe inputs for Qwen3.5 (official chat template), reusing the DeepSeek data pipeline.

Same sources, splits, fragments and role mapping as deepseek_role_probe_data.py (GPT-OSS developer ->
system, GPT-OSS system dropped); only the renderer differs: the model's own ``chat_template.jinja``.
  system    <|im_start|>system\n[# Tools ... block when tools are given\n\n]BODY<|im_end|>
  user      <|im_start|>user\nBODY<|im_end|>
  cot       <|im_start|>assistant\n<think>\nBODY\n</think>\n\n...
  assistant ...</think>\n\nBODY<|im_end|>
  tool      <|im_start|>user\n<tool_response>\nBODY\n</tool_response><|im_end|>
The template trims message bodies, so surrounding whitespace of a body is removed before rendering.
Single-role article records need a user query for the template: an empty user turn is added AFTER a system
body and BEFORE a tool body (the causal model cannot see what follows the labeled tokens).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import deepseek_role_probe_data as D

BASE = Path(__file__).resolve().parent
MODEL = Path("/scratch/wz55191/Hugginface/Qwen3.5-397B-A17B-GPTQ-Int4")
OUTPUT = BASE / "outputs/qwen35-397b-gptq-int4-role-probe-20261120"
D.FORMAT_VERSION = "qwen35-native-role-inputs-v1-developer-as-system"
MARKED = re.compile("([0-9a-f]+L)\\s*(.*?)\\s*([0-9a-f]+R)", re.S)


def clean_body(value):
    if "" in value:
        return MARKED.sub(lambda m: m[1] + m[2] + m[3], value)
    return value.strip()


class QwenEncoder:
    def __init__(self, model_dir):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.template = (Path(model_dir) / "chat_template.jinja").read_text()

    def prepare(self, messages):
        messages, tools = copy.deepcopy(messages), None
        for message in messages:
            if "tools" in message:
                tools = message.pop("tools")
            for field in ("content", "reasoning_content"):
                if isinstance(message.get(field), str):
                    message[field] = clean_body(message[field])
            for call in message.get("tool_calls") or []:
                function = call.get("function", call)
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"] or "{}")
        roles = [m["role"] for m in messages]
        if roles == ["system"]:
            messages.append({"role": "user", "content": ""})
        elif roles == ["tool"]:
            messages.insert(0, {"role": "user", "content": ""})
        return messages, tools

    def encode_messages(self, messages, **_):
        messages, tools = self.prepare(messages)
        return self.tokenizer.apply_chat_template(messages, tools=tools, chat_template=self.template, tokenize=False,
                                                  add_generation_prompt=False)


D.load_encoder = QwenEncoder

# ---------------------------------------------------------------------------------------------------------------
# v2: multi-role article conversations.
# The reused GPT-OSS data has one labeled role per article record, and its AgentDojo calls keep no reasoning in the
# history and have a single user turn.  A probe trained on it alone (run ...-role-probe-20261120, heads in decoder
# layers 3 / 7) labeled Qwen's own conversations wrongly: everything after a kept <think> block stayed "cot", and a
# genuine second user turn after a tool result stayed "tool" (P(user) 0.06).  These records teach role *switches*:
# the same article fragments, several per conversation, in agent-like role orders (reasoning followed by answers,
# tool results and further turns; user turns after tool results).  Splits follow the source articles.
MULTI = bool(int(__import__("os").environ.get("QWEN_MULTI", "1")))
_captured = []
_tokenize_record = D.tokenize_record


def tokenize_record(tokenizer, encoder, messages, fields, identity, **kwargs):
    if identity.get("domain") == "c4_dolma3" and identity.get("structure") == "c4_dolma3:single":
        field = fields[0]
        _captured.append((identity["split"], identity["source_cluster"], identity["source_id"], identity["corpus"],
                          messages[field["message_index"]][field["field"]]))
    return _tokenize_record(tokenizer, encoder, messages, fields, identity, **kwargs)


def halves(text):
    """Two pieces of an article fragment, each cut at whitespace and at most ~700 characters."""
    middle = text.find(" ", len(text) // 2)
    parts = [text[:middle], text[middle + 1:]] if middle > 0 else [text]
    return [part[:700].rsplit(" ", 1)[0].strip() for part in parts if len(part.strip()) > 80]


def conversation(rng, bodies):
    """Agent-like role order over article pieces -> (messages, fields without ids)."""
    messages, fields = [], []

    def add(role, label, field="content", **extra):
        if not bodies:
            return False
        if field == "reasoning_content":
            messages[-1]["reasoning_content"] = bodies.pop()
        else:
            messages.append({"role": role, "content": bodies.pop(), **extra})
            if role == "assistant":
                messages[-1]["reasoning_content"] = ""
        fields.append({"message_index": len(messages) - 1, "field": field, "source_role": label})
        return True

    if rng.random() < 0.5:
        add("system", "developer")
    add("user", "user")
    final_user = rng.random() < 0.5      # the chat template keeps reasoning only after the LAST user turn
    if final_user:                       # earlier rounds: no reasoning; ends with a user turn after assistant / tool text
        for _ in range(rng.randint(1, 2)):
            add("assistant", "assistant")
            if rng.random() < 0.6:
                add("tool", "tool", tool_call_id="call")
                if rng.random() < 0.5:
                    add("assistant", "assistant")
        add("user", "user")
        if rng.random() < 0.5 and add("assistant", "assistant") and rng.random() < 0.7:
            add("assistant", "cot", "reasoning_content")
    else:                                # agent loop after the only user turn: reasoning, answer, tool result, again
        for _ in range(rng.randint(2, 3)):
            if not add("assistant", "assistant"):
                break
            if rng.random() < 0.75:
                add("assistant", "cot", "reasoning_content")
            if rng.random() < 0.8:
                add("tool", "tool", tool_call_id="call")
    return messages, fields


_agentdojo_records = D.agentdojo_records


def agentdojo_records(args, tokenizer, encoder, assignments, audit):
    yield from _agentdojo_records(args, tokenizer, encoder, assignments, audit)
    if not MULTI:
        return
    import random
    by_split = {}
    for row in _captured:
        by_split.setdefault(row[0], []).append(row)
    count = 0
    for split, rows in sorted(by_split.items()):
        for start in range(0, len(rows) - 3, 4):
            group = rows[start:start + 4]
            rng = random.Random(int(D.sha(["multi-v1", group[0][2]])[:16], 16))
            bodies = [piece for row in group for piece in halves(row[4])]
            rng.shuffle(bodies)
            messages, fields = conversation(rng, bodies)
            for number, field in enumerate(fields):
                field["event_id"] = f"{group[0][2]}/multi/m{number}/{field['source_role']}"
                field["scope"] = "article_body"
            identity = {"id": group[0][2] + "/multi/native-qwen", "domain": "c4_dolma3", "split": split, "source_id": group[0][2],
                        "source_cluster": group[0][1], "corpus": group[0][3], "structure": "c4_dolma3:multi",
                        "member_clusters": [row[1] for row in group], "roles": [f["source_role"] for f in fields]}
            yield D.tokenize_record(tokenizer, encoder, messages, fields, identity, max_tokens=args.max_input_tokens, target_cap=240)
            count += 1
    audit["multi_role_article_conversations"] = count


# ---------------------------------------------------------------------------------------------------------------
# v3: agent-loop records in Qwen's own layout.
# The v2 probe (run ...-multiturn-20261123) still failed on real Qwen conversations: tool-result tokens flickered
# between tool and cot when reasoning was kept in the history, and a genuine user turn after a tool result got
# P(user) 0.54.  A layer/head scan (qwen35-richevo/ds_diag_layers.py) showed that no layer transfers that case, so
# the data, not the head choice, is missing it.  Qwen's template keeps the reasoning of every assistant turn after
# the last user query, while the reused GPT-OSS calls carry no reasoning in their history; and no reused record
# has a user turn after a tool result.  From the same 359 frozen benign AgentDojo calls (same task splits):
#   agentdojo:kept-reasoning  the call's history with the archived reasoning of each earlier response restored
#   agentdojo:followup        the history up to its last tool result, optionally a short assistant reply, then a
#                             further user turn (another task's request from the same split, or an article piece
#                             of the same split), optionally followed by a reasoning + answer turn
AGENT = bool(int(__import__("os").environ.get("QWEN_AGENT", "1")))
REPLIES = ("OK.", "Sure.", "All set.", "That is finished.", "I have completed that.", "Here is what I found.",
           "The task is complete.", "Okay, noted.", "Finished. Anything else?", "I've taken care of it.")
_multi_agentdojo_records = agentdojo_records


def words(rng, text, low, high):
    return " ".join(text.split()[:rng.randint(low, high)])


def agent_records(args, tokenizer, encoder, assignments, audit):
    import random
    tasks = {}
    for path in sorted((args.agentdojo / "calls").glob("*/*/*/call_*/request.json")):
        request = D.read(path)
        meta = request["metadata"]
        tasks.setdefault(f"{meta['suite']}/{meta['user_task_id']}", []).append((meta["call_index"], request))
    prompts, pieces, reasoning, answers = {}, {}, {}, {}
    for split, cluster, source, corpus, text in _captured:
        pieces.setdefault(split, []).extend(halves(text))
    for task, calls in tasks.items():
        split = assignments["agentdojo:" + task]
        for _, request in calls:
            response = D.normalize_message(request["response"])
            key = tuple(call.get("id", "") for call in response.get("tool_calls") or [])
            if key and response.get("reasoning_content"):
                reasoning[task, key] = response["reasoning_content"]
            if not key and len(response["content"].strip()) > 20:
                answers.setdefault(split, []).append(response["content"].strip())
        user = next(m for m in calls[0][1]["normalized_messages"] if m["role"] == "user")
        prompts.setdefault(split, []).append((task, D.normalize_message(user)["content"].strip()))
    counts = {"agentdojo:kept-reasoning": 0, "agentdojo:followup": 0, "skipped_too_long": 0}
    for task, calls in sorted(tasks.items()):
        cluster = "agentdojo:" + task
        split = assignments[cluster]
        for call_index, request in sorted(calls, key=lambda item: item[0]):
            if call_index == 0:
                continue
            rng = random.Random(int(D.sha(["agent-v3", request["metadata"]["call_id"]])[:16], 16))
            base = [D.normalize_message(m) for m in request["normalized_messages"]]
            base = [m for m in base if m["role"] not in D.DROPPED_SOURCE_ROLES]
            for message in base:
                if message["role"] == "developer":
                    message["role"] = "system"
                    message["tools"] = copy.deepcopy(request["tools"])
                if message["role"] == "assistant":
                    if message.get("content") in ("None", None):
                        message["content"] = ""
                    key = tuple(call.get("id", "") for call in message.get("tool_calls") or [])
                    message["reasoning_content"] = reasoning.get((task, key), "") if key else ""
            if base[-1]["role"] != "tool":
                continue
            variants = []
            # (a) history with restored reasoning + the archived response
            kept = copy.deepcopy(base) + [D.normalize_message(request["response"])]
            variants.append(("agentdojo:kept-reasoning", kept, range(len(kept)), 240))
            # (b) follow-up user turn after the last tool result
            follow = copy.deepcopy(base)
            for message in follow:           # the template drops reasoning before the last user query
                message.pop("reasoning_content", None)
            start = max(i for i, m in enumerate(follow) if m["role"] != "tool") + 1   # the last block of tool results
            choice = rng.random()
            if choice < 0.6:
                reply = (rng.choice(REPLIES) if choice < 0.2 else rng.choice(answers[split]) if choice < 0.4 and answers.get(split)
                         else words(rng, rng.choice(pieces[split]), 2, 14))
                follow.append({"role": "assistant", "content": reply})
            others = [text for other, text in prompts[split] if other != task]
            follow.append({"role": "user", "content": rng.choice(others) if rng.random() < 0.5 and others
                           else words(rng, rng.choice(pieces[split]), 6, 60)})
            if rng.random() < 0.4:
                follow.append({"role": "assistant", "content": words(rng, rng.choice(pieces[split]), 8, 80),
                               "reasoning_content": words(rng, rng.choice(pieces[split]), 8, 80)})
            variants.append(("agentdojo:followup", follow, [1] + list(range(start, len(follow))), None))
            for structure, messages, indices, cap in variants:
                fields = []
                for index in indices:
                    message = messages[index]
                    for field in ("content", "reasoning_content"):
                        if not (message.get(field) or "").strip():
                            continue
                        role = "cot" if field == "reasoning_content" else {"system": "developer"}.get(message["role"], message["role"])
                        fields.append({"message_index": index, "field": field, "source_role": role, "scope": "actual_input",
                                       "event_id": f"{task}/{structure}/c{call_index}/m{index}/{field}"})
                identity = {"id": request["metadata"]["call_id"] + "/" + structure.split(":")[1] + "/native-qwen", "domain": "agentdojo",
                            "split": split, "source_id": task, "source_cluster": cluster, "task_id": task,
                            "suite": request["metadata"]["suite"], "call_index": call_index, "structure": structure,
                            "trajectory_policy": "frozen_gptoss_teacher_forced_replay"}
                try:
                    yield D.tokenize_record(tokenizer, encoder, messages, fields, identity, max_tokens=args.max_input_tokens, target_cap=cap)
                    counts[structure] += 1
                except ValueError as error:
                    if "tokens >" not in str(error):
                        raise
                    counts["skipped_too_long"] += 1
    audit["agent_loop_records"] = counts


def agentdojo_records_v3(args, tokenizer, encoder, assignments, audit):
    yield from _multi_agentdojo_records(args, tokenizer, encoder, assignments, audit)
    if AGENT:
        yield from agent_records(args, tokenizer, encoder, assignments, audit)


if MULTI:
    D.tokenize_record, D.agentdojo_records = tokenize_record, agentdojo_records_v3
    D.FORMAT_VERSION = "qwen35-native-role-inputs-v2-multi-role-articles"
    OUTPUT = BASE / "outputs/qwen35-397b-gptq-int4-role-probe-multiturn-20261123"
    if AGENT:
        D.FORMAT_VERSION = "qwen35-native-role-inputs-v3-agent-loop"
        OUTPUT = BASE / "outputs/qwen35-397b-gptq-int4-role-probe-agentloop-20261123"

if __name__ == "__main__":
    parser = D.parser()
    parser.set_defaults(output=OUTPUT, model=MODEL)
    arguments = parser.parse_args()
    if arguments.stage == "restore-articles":
        D.restore_articles(arguments)
    else:
        manifest = D.prepare(arguments)
        D.atomic_json(arguments.output / "data/qwen-rendering-note.json", {
            "renderer": "chat_template.jinja of the checkpoint via tokenizer.apply_chat_template (tools passed separately)",
            "role_mapping": "GPT-OSS developer -> system; GPT-OSS system dropped; cot = <think> block; tool = <tool_response> inside a user turn",
            "note": "protocol.role_mapping_evidence in manifest.json was written by the shared DeepSeek preparation code and describes the DeepSeek encoder; this file describes the Qwen rendering actually used",
            "chat_template_sha256": D.file_sha(arguments.model / "chat_template.jinja"), "implementation_sha256": D.file_sha(__file__)})
