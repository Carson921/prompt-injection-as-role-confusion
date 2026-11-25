"""Role-probe inputs for Llama-4-Scout-17B-16E-Instruct (official chat template), reusing the DeepSeek data pipeline.

Same sources, splits and fragments as deepseek_role_probe_data.py; the renderer is the checkpoint's own
``chat_template.jinja``.  Llama 4 Scout has no reasoning channel, so the probe has FOUR roles:
  system    <|header_start|>system<|header_end|>\n\nBODY[\nHere is a list of functions ... TOOLS JSON]<|eot|>
  user      <|header_start|>user<|header_end|>\n\nBODY<|eot|>
  assistant <|header_start|>assistant<|header_end|>\n\nBODY[ [tool_name(arg="value")] ]<|eot|>
  tool      <|header_start|>ipython<|header_end|>\n\n"BODY as one JSON string"<|eot|>
Role mapping (the user's rule for DeepSeek / Qwen): GPT-OSS developer -> system, GPT-OSS system dropped.
GPT-OSS reasoning (cot) has no Llama 4 counterpart: article records assigned to it are dropped and archived
reasoning is removed from AgentDojo histories (the template would not render it anyway).

The template trims the system body and JSON-encodes tool bodies (newlines become the two characters
backslash-n); labels follow the rendered characters of each body.

Like the final Qwen probe data (qwen35_role_probe_data.py v3), two record families teach role switches that the
reused GPT-OSS data lacks (one labeled role per article record; no user turn after a tool result):
  c4_dolma3:multi      several article pieces in one conversation, in agent-like role orders
  agentdojo:followup   a frozen AgentDojo history up to its last tool result, optionally a short assistant
                       reply, then a further user turn, optionally followed by an assistant answer
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import re

import deepseek_role_probe_data as D

BASE = Path(__file__).resolve().parent
MODEL = Path("/scratch/wz55191/Hugginface/Llama-4-Scout-17B-16E-Instruct")
OUTPUT = BASE / "outputs/llama4-scout-17b-16e-role-probe-20261125"
D.FORMAT_VERSION = "llama4-native-role-inputs-v1-developer-as-system-no-cot-agent-loop"
D.ROLES = ("system", "user", "assistant", "tool")
D.ROLE_MAPPING = {"developer": "system", "user": "user", "assistant": "assistant", "tool": "tool"}
D.DROPPED_SOURCE_ROLES = ("system", "cot")
MARKED = re.compile("([0-9a-f]+L)\\s*(.*?)\\s*([0-9a-f]+R)", re.S)


def clean_body(value):
    """Strip a body the way the template's trim would, also when span markers surround it."""
    if "" in value:
        return MARKED.sub(lambda m: m[1] + m[2] + m[3], value)
    return value.strip()


class LlamaEncoder:
    def __init__(self, model_dir):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.template = (Path(model_dir) / "chat_template.jinja").read_text()

    def prepare(self, messages):
        messages, tools = copy.deepcopy(messages), None
        for message in messages:
            if "tools" in message:
                tools = message.pop("tools")
            message.pop("reasoning_content", None)
            if isinstance(message.get("content"), str):
                message["content"] = clean_body(message["content"])
            for call in message.get("tool_calls") or []:
                function = call.get("function", call)
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"] or "{}")
            if not message.get("tool_calls"):
                message.pop("tool_calls", None)  # the template tests the key, not its length, in one branch
        return messages, tools

    def encode_messages(self, messages, **_):
        messages, tools = self.prepare(messages)
        return self.tokenizer.apply_chat_template(messages, tools=tools, chat_template=self.template, tokenize=False,
                                                  add_generation_prompt=False)


D.load_encoder = LlamaEncoder
_normalize_message = D.normalize_message


def normalize_message(message):
    result = _normalize_message(message)
    result.pop("reasoning_content", None)
    if result["role"] == "assistant" and result.get("content") in ("None", None):
        result["content"] = ""
    return result


D.normalize_message = normalize_message

# ---------------------------------------------------------------------------------------------------------------
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

    def add(role, label, **extra):
        if not bodies:
            return False
        messages.append({"role": role, "content": bodies.pop(), **extra})
        fields.append({"message_index": len(messages) - 1, "field": "content", "source_role": label})
        return True

    if rng.random() < 0.5:
        add("system", "developer")
    add("user", "user")
    if rng.random() < 0.5:               # ends with a user turn after assistant / tool text
        for _ in range(rng.randint(1, 2)):
            add("assistant", "assistant")
            if rng.random() < 0.6:
                add("tool", "tool", tool_call_id="call")
                if rng.random() < 0.5:
                    add("assistant", "assistant")
        add("user", "user")
        if rng.random() < 0.5:
            add("assistant", "assistant")
    else:                                # agent loop after the only user turn: answer, tool result, again
        for _ in range(rng.randint(2, 3)):
            if not add("assistant", "assistant"):
                break
            if rng.random() < 0.8:
                add("tool", "tool", tool_call_id="call")
    return messages, fields


REPLIES = ("OK.", "Sure.", "All set.", "That is finished.", "I have completed that.", "Here is what I found.",
           "The task is complete.", "Okay, noted.", "Finished. Anything else?", "I've taken care of it.")
def base_agentdojo_records(args, tokenizer, encoder, assignments, audit):
    # copy of D.agentdojo_records; only change: calls without a new non-reasoning body are skipped instead of rejected
    """Teacher-force frozen benign trajectories; supervise each body event once.

    Appending the current archived response lets a single causal forward capture
    new input and output body positions. Prior input tokens cannot attend to the
    appended response. This is cross-model replay, not a new DeepSeek rollout.
    """
    paths = sorted((args.agentdojo / "calls").glob("*/*/*/call_*/request.json"))
    D.require(len(paths) == 359, "Expected all original 359 benign calls")
    original_audit = D.read(args.original / "agentdojo-source-audit.json")["consumed_source_files"]
    seen, fingerprints, tasks = set(), {}, set()
    for path in paths:
        relative = path.relative_to(args.agentdojo).as_posix()
        D.require(relative in original_audit and D.file_sha(path) == original_audit[relative]["sha256"],
                "Historical AgentDojo request differs from the original mixed experiment: " + relative)
        request = D.read(path)
        D.require(request["status"] == "complete", "Incomplete historical AgentDojo request")
        meta = request["metadata"]
        task = f"{meta['suite']}/{meta['user_task_id']}"
        cluster = "agentdojo:" + task
        D.require(cluster in assignments, "Unknown original AgentDojo task")
        tasks.add(cluster)
        messages = [D.normalize_message(message) for message in request["normalized_messages"]]
        audit["dropped_source_system_messages"] = audit.get("dropped_source_system_messages", 0) + sum(
            message["role"] in D.DROPPED_SOURCE_ROLES for message in messages)
        messages = [message for message in messages if message["role"] not in D.DROPPED_SOURCE_ROLES]
        if meta["call_index"] == 0:
            audit.setdefault("first_call_structure", []).append({"task": task,
                "roles": [message["role"] for message in messages],
                "content_characters": [len(message["content"]) for message in messages]})
        instructions = next((message for message in messages if message["role"] == "developer"), None)
        D.require(instructions is not None, "Task instructions are missing")
        instructions["tools"] = copy.deepcopy(request["tools"])
        response_index = len(messages)
        messages.append(D.normalize_message(request["response"]))
        source_roles = [message["role"] for message in messages]
        for message in messages:  # developer bodies are rendered in the native system slot
            if message["role"] == "developer":
                message["role"] = "system"
        fields = []
        for index, message in enumerate(messages):
            source_role = source_roles[index]
            D.require(source_role in D.ROLE_MAPPING, "Unexpected AgentDojo source role")
            for field in ("content", "reasoning_content"):
                body = message.get(field) or ""
                if not body:
                    continue
                role = "cot" if field == "reasoning_content" else source_role
                event = D.body_event_id(task, {**message, "role": source_role}, field, body)
                signature = (role, D.sha(body))
                if event in fingerprints:
                    D.require(fingerprints[event] == signature, f"Historical body changed for stable event {event}")
                fingerprints[event] = signature
                if event in seen:
                    continue
                seen.add(event)
                fields.append({"message_index": index, "field": field, "event_id": event,
                               "source_role": role,
                               "scope": "actual_generated" if index == response_index else "actual_input"})
        if not fields:  # the only new body of this call was GPT-OSS reasoning, which Llama 4 has no slot for
            audit["calls_without_new_body_fields"] = audit.get("calls_without_new_body_fields", 0) + 1
            continue
        identity = {"id": meta["call_id"] + "/native-llama4", "domain": "agentdojo",
                    "split": assignments[cluster], "source_id": task, "source_cluster": cluster,
                    "task_id": task, "suite": meta["suite"], "call_index": meta["call_index"],
                    "structure": "agentdojo:native", "source_request_sha256": D.file_sha(path),
                    "source_request": str(path), "trajectory_policy": "frozen_gptoss_teacher_forced_replay"}
        try:
            record = D.tokenize_record(tokenizer, encoder, messages, fields, identity, max_tokens=args.max_input_tokens)
        except ValueError as error:  # e.g. a one-word tool result that tokenizes together with its JSON quotes
            if "No native body tokens remain" not in str(error):
                raise
            audit["calls_without_pure_body_tokens"] = audit.get("calls_without_pure_body_tokens", 0) + 1
            continue
        audit.setdefault("source_requests", {})[meta["call_id"]] = D.file_sha(path)
        yield record
    D.require(tasks == set(assignments), "Missing original AgentDojo task assignments")
    audit["tasks"], audit["calls"], audit["first_occurrence_body_events"] = len(tasks), len(paths), len(seen)


_agentdojo_records = base_agentdojo_records


def words(rng, text, low, high):
    return " ".join(text.split()[:rng.randint(low, high)])


def multi_records(args, tokenizer, encoder, audit):
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
            identity = {"id": group[0][2] + "/multi/native-llama4", "domain": "c4_dolma3", "split": split, "source_id": group[0][2],
                        "source_cluster": group[0][1], "corpus": group[0][3], "structure": "c4_dolma3:multi",
                        "member_clusters": [row[1] for row in group], "roles": [f["source_role"] for f in fields]}
            yield D.tokenize_record(tokenizer, encoder, messages, fields, identity, max_tokens=args.max_input_tokens, target_cap=240)
            count += 1
    audit["multi_role_article_conversations"] = count


def followup_records(args, tokenizer, encoder, assignments, audit):
    tasks = {}
    for path in sorted((args.agentdojo / "calls").glob("*/*/*/call_*/request.json")):
        request = D.read(path)
        meta = request["metadata"]
        tasks.setdefault(f"{meta['suite']}/{meta['user_task_id']}", []).append((meta["call_index"], request))
    prompts, pieces, answers = {}, {}, {}
    for split, cluster, source, corpus, text in _captured:
        pieces.setdefault(split, []).extend(halves(text))
    for task, calls in tasks.items():
        split = assignments["agentdojo:" + task]
        for _, request in calls:
            response = D.normalize_message(request["response"])
            if not response.get("tool_calls") and len(response["content"].strip()) > 20:
                answers.setdefault(split, []).append(response["content"].strip())
        user = next(m for m in calls[0][1]["normalized_messages"] if m["role"] == "user")
        prompts.setdefault(split, []).append((task, D.normalize_message(user)["content"].strip()))
    counts = {"agentdojo:followup": 0, "skipped_too_long": 0}
    for task, calls in sorted(tasks.items()):
        cluster = "agentdojo:" + task
        split = assignments[cluster]
        for call_index, request in sorted(calls, key=lambda item: item[0]):
            if call_index == 0:
                continue
            rng = random.Random(int(D.sha(["agent-v3", request["metadata"]["call_id"]])[:16], 16))
            follow = [D.normalize_message(m) for m in request["normalized_messages"]]
            follow = [m for m in follow if m["role"] not in D.DROPPED_SOURCE_ROLES]
            for message in follow:
                if message["role"] == "developer":
                    message["role"] = "system"
                    message["tools"] = copy.deepcopy(request["tools"])
            if follow[-1]["role"] != "tool":
                continue
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
                follow.append({"role": "assistant", "content": words(rng, rng.choice(pieces[split]), 8, 80)})
            fields = []
            for index in [1] + list(range(start, len(follow))):
                message = follow[index]
                if not (message.get("content") or "").strip():
                    continue
                role = {"system": "developer"}.get(message["role"], message["role"])
                fields.append({"message_index": index, "field": "content", "source_role": role, "scope": "actual_input",
                               "event_id": f"{task}/agentdojo:followup/c{call_index}/m{index}/content"})
            identity = {"id": request["metadata"]["call_id"] + "/followup/native-llama4", "domain": "agentdojo",
                        "split": split, "source_id": task, "source_cluster": cluster, "task_id": task,
                        "suite": request["metadata"]["suite"], "call_index": call_index, "structure": "agentdojo:followup",
                        "trajectory_policy": "frozen_gptoss_teacher_forced_replay"}
            try:
                yield D.tokenize_record(tokenizer, encoder, follow, fields, identity, max_tokens=args.max_input_tokens)
                counts["agentdojo:followup"] += 1
            except ValueError as error:
                if "tokens >" not in str(error):
                    raise
                counts["skipped_too_long"] += 1
    audit["agent_loop_records"] = counts


def agentdojo_records(args, tokenizer, encoder, assignments, audit):
    yield from _agentdojo_records(args, tokenizer, encoder, assignments, audit)
    yield from multi_records(args, tokenizer, encoder, audit)
    yield from followup_records(args, tokenizer, encoder, assignments, audit)


D.tokenize_record, D.agentdojo_records = tokenize_record, agentdojo_records

if __name__ == "__main__":
    parser = D.parser()
    parser.set_defaults(output=OUTPUT, model=MODEL)
    arguments = parser.parse_args()
    if arguments.stage == "restore-articles":
        D.restore_articles(arguments)
    else:
        manifest = D.prepare(arguments)
        D.atomic_json(arguments.output / "data/llama4-rendering-note.json", {
            "renderer": "chat_template.jinja of the checkpoint via tokenizer.apply_chat_template (tools passed separately)",
            "roles": list(D.ROLES),
            "role_mapping": "GPT-OSS developer -> system; GPT-OSS system dropped; GPT-OSS cot dropped (no reasoning channel); "
                            "tool = ipython turn whose body is one JSON string",
            "note": "protocol.role_mapping_evidence in manifest.json was written by the shared DeepSeek preparation code and describes "
                    "the DeepSeek encoder; this file describes the Llama 4 rendering actually used",
            "chat_template_sha256": D.file_sha(arguments.model / "chat_template.jinja"), "implementation_sha256": D.file_sha(__file__)})
