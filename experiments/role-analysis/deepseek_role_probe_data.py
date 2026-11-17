"""Prepare provenance-bound native DeepSeek-V4 role-probe inputs on CPU.

The original mixed experiment's source clusters and task assignments are fixed.
No GPT-OSS activation or probability is an input to this pipeline. Its tokenizer
is used only to reproduce the exact original article fragment as ordinary text;
all model inputs and supervision coordinates are retokenized with DeepSeek.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import time


BASE = Path(__file__).resolve().parent
ORIGINAL = BASE / "outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103"
ARTICLE_META = BASE / "outputs/synthetic-agent-120b-20261026/final-v7/metadata/data/formal"
OUTPUT = BASE / "outputs/deepseek-v4-flash-0731-role-probe-20261115"
MODEL = Path("/scratch/wz55191/Hugginface/DeepSeek-V4-Flash-0731")
OLD_MODEL = Path("/scratch/wz55191/models/gpt-oss-120b")
AGENTDOJO = Path("/scratch/wz55191/agentdojo-trace-source-20261112/full57_no_payload_first_occurrence_probe_v7_20261027")
ROLES = ("system", "user", "cot", "assistant", "tool")
SOURCE_ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
ROLE_MAPPING = {role: ("user" if role == "developer" else role) for role in SOURCE_ROLES}
SPLITS = ("fit", "dev", "test")
FORMAT_VERSION = "deepseek-native-role-inputs-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    part.replace(path)


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def restore_articles(args):
    """Recover exact historical articles from pinned, unshuffled HF streams.

    Only historically retained candidates are materialized. SHA mismatches fail
    immediately rather than silently replacing a document. Per-source JSONL is
    append-only and flushed after every retained document; retries can replay
    the deterministic stream while skipping already verified documents.
    """
    import synthetic_agent_data as old

    root = args.output / "data" / "sources"
    root.mkdir(parents=True, exist_ok=True)
    keys = read(args.article_metadata / "source-keys.json")
    require(len(keys) == 5632, "Expected the original 5,632 source articles")
    configuration = read(args.article_metadata / "data-manifest.json")["configuration"]
    identity = {"source_keys_sha256": file_sha(args.article_metadata / "source-keys.json"),
                "source_specifications": configuration["sources"],
                "restore_rule": "Exact original candidate index and UTF-8 text SHA256; no replacement"}
    pin = root / "restore-protocol.json"
    if pin.exists():
        require(read(pin) == identity, "Source recovery protocol changed")
    else:
        atomic_json(pin, identity)
    complete_path = root / "restore-complete.json"
    if complete_path.exists():
        completed = read(complete_path)
        require(completed["identity"] == identity and completed["documents"] == 5632,
                "Completed source recovery identity changed")
        for source in completed["sources"]:
            require(file_sha(root / f"articles-{source['source']}.jsonl") == source["sha256"],
                    "Completed recovered source file changed")
        print(json.dumps({"stage": "restored_articles_reused", "documents": 5632}), flush=True)
        return completed

    def one_source(name):
        wanted = {int(item["candidate_index"]): item for item in keys if item["source"] == name}
        target = root / f"articles-{name}.jsonl"
        saved = {}
        # Repair only a trailing interrupted write, never a complete bad record.
        if target.exists():
            with target.open("rb+") as stream:
                cursor = 0
                while line := stream.readline():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        require(not stream.read(), "Corrupt nonterminal source cache row")
                        stream.truncate(cursor)
                        break
                    require(item["source_id"] not in saved, "Duplicate recovered source")
                    require(sha(item["text"]) == item["text_sha256"], "Recovered text hash changed")
                    require(item["text_sha256"] == wanted[int(item["candidate_index"])]["text_sha256"],
                            "Recovered source does not match the original source key")
                    saved[item["source_id"]] = item
                    cursor = stream.tell()
        if len(saved) == len(wanted):
            complete = root / f"restore-{name}-complete.json"
            if complete.exists():
                result = read(complete)
                require(result["sha256"] == file_sha(target), "Completed source file changed")
                return result
            return {"source": name, "documents": len(saved), "status": "complete", "sha256": file_sha(target)}
        provenance = {}
        maximum = max(wanted)
        started = time.time()
        with target.open("a") as stream:
            for candidate_index, row in enumerate(old._source_stream(name, configuration["sources"][name], root, provenance)):
                if candidate_index > maximum:
                    break
                key = wanted.get(candidate_index)
                if key and key["source_id"] not in saved:
                    require(isinstance(row.get("text"), str), "Original source text is missing")
                    require(sha(row["text"]) == key["text_sha256"],
                            f"Pinned {name} candidate {candidate_index} text differs from historical SHA")
                    if key.get("source_group"):
                        require(row.get("_synthetic_data_file_group") == key["source_group"], "Dolma source group changed")
                    document = {**key, "text": row["text"]}
                    stream.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    saved[key["source_id"]] = document
                if candidate_index % 5000 == 0 or candidate_index == maximum:
                    report = {"stage": "restore_articles", "source": name, "candidate_index": candidate_index,
                              "maximum_candidate_index": maximum, "recovered": len(saved), "expected": len(wanted),
                              "seconds": time.time() - started}
                    atomic_json(root / f"restore-{name}-progress.json", report)
                    print(json.dumps(report), flush=True)
                if len(saved) == len(wanted):
                    break
        require(len(saved) == len(wanted), f"{name} stream exhausted before exact source recovery")
        result = {"source": name, "documents": len(saved), "status": "complete", "sha256": file_sha(target),
                  "provenance": provenance}
        atomic_json(root / f"restore-{name}-complete.json", result)
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(one_source, ("c4", "dolma3")))
    result = {"status": "complete", "documents": sum(item["documents"] for item in results),
              "sources": results, "identity": identity}
    atomic_json(root / "restore-complete.json", result)
    print(json.dumps(result), flush=True)
    return result


def load_encoder(model_dir):
    """Load the locally pinned official encoder without copying its templates."""
    path = Path(model_dir) / "encoding/encoding_dsv4.py"
    require(path.is_file(), "Download the official encoding/encoding_dsv4.py first")
    spec = importlib.util.spec_from_file_location("_role_probe_official_deepseek_encoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_with_spans(encoder, messages, fields):
    """Track ordinary field spans through the unmodified official renderer.

    Temporary unique markers are removed before tokenization; the resulting
    string must exactly equal an independent official rendering without markers.
    This follows tool-result merging and tools-schema insertion faithfully.
    """
    original = copy.deepcopy(messages)
    marked = copy.deepcopy(messages)
    markers = []
    for number, field in enumerate(fields):
        index, key = int(field["message_index"]), field["field"]
        value = marked[index].get(key) or ""
        require(isinstance(value, str) and value, "Supervised field must be nonempty text")
        marker_id = sha([number, index, key, value])
        left, right = "\ue000" + marker_id + "L\ue001", "\ue000" + marker_id + "R\ue001"
        require(all(left not in str(message) and right not in str(message) for message in original), "Marker collision")
        marked[index][key] = left + value + right
        markers.append((left, right, dict(field)))
    kwargs = {"thinking_mode": "thinking", "drop_thinking": False,
              "reasoning_effort": "low", "add_default_bos_token": True}
    clean = encoder.encode_messages(original, **kwargs)
    prompt = encoder.encode_messages(marked, **kwargs)
    events = []
    for left, right, field in markers:
        require(prompt.count(left) == 1 and prompt.count(right) == 1,
                "Official encoder dropped or duplicated a supervised field")
        events.extend([(prompt.index(left), left, "start", field), (prompt.index(right), right, "end", field)])
    events.sort(key=lambda event: event[0])
    pieces, spans, active, cursor, length = [], [], None, 0, 0
    for position, marker, kind, field in events:
        piece = prompt[cursor:position]
        pieces.append(piece)
        length += len(piece)
        if kind == "start":
            require(active is None, "Supervised body fields overlap")
            active = {**field, "char_start": length}
        else:
            require(active is not None and active["event_id"] == field["event_id"], "Unbalanced body markers")
            spans.append({**active, "char_end": length})
            active = None
        cursor = position + len(marker)
    pieces.append(prompt[cursor:])
    require(active is None and "".join(pieces) == clean,
            "Marker instrumentation changed official DeepSeek encoding")
    return clean, spans


def tokenize_record(tokenizer, encoder, messages, fields, identity, *, max_tokens, target_cap=None):
    prompt, spans = render_with_spans(encoder, messages, fields)
    result = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = result["input_ids"], result["offset_mapping"]
    require(len(ids) <= max_tokens,
            f"{identity['id']} requires {len(ids)} tokens > {max_tokens}; full messages must not be truncated")
    require(len(ids) > 0, "Empty native input")
    targets, boundary = [], 0
    for position, (left, right) in enumerate(offsets):
        if right <= left:
            continue
        overlaps = [span for span in spans if left < span["char_end"] and right > span["char_start"]]
        if not overlaps:
            continue
        if len(overlaps) != 1 or left < overlaps[0]["char_start"] or right > overlaps[0]["char_end"]:
            boundary += 1
            continue
        span = overlaps[0]
        role = ROLE_MAPPING[span["source_role"]]
        family = f"{identity['source_cluster']}/{span['event_id']}/{position}"
        targets.append({"token_ix": position, "label": ROLES.index(role), "role": role,
                        "source_role": span["source_role"], "event_id": span["event_id"],
                        "scope": span.get("scope", "article_body"), "family_id": family,
                        "source_id": identity["source_cluster"], "weight": 1.0,
                        "char_start": left, "char_end": right})
    if target_cap is not None and len(targets) > target_cap:
        keep = set(sorted(range(len(targets)), key=lambda i: sha([20261115, targets[i]["family_id"]]))[:target_cap])
        targets = [target for i, target in enumerate(targets) if i in keep]
    require(targets, f"No native body tokens remain in {identity['id']}")
    require(len({target["token_ix"] for target in targets}) == len(targets), "Duplicate labeled token")
    return {**identity, "format_version": FORMAT_VERSION, "input_ids": ids, "input_tokens": len(ids),
            "positions": [target["token_ix"] for target in targets],
            "labels": [target["label"] for target in targets], "targets": targets,
            "body_spans": spans, "excluded_boundary_tokens": boundary,
            "native_prompt_sha256": sha(prompt), "messages_sha256": sha(messages)}


def article_records(args, tokenizer, encoder, assignments):
    """Reuse the historical article fragments, then retokenize natively."""
    import synthetic_agent_data as old
    from transformers import AutoTokenizer

    old_tokenizer = AutoTokenizer.from_pretrained(str(args.old_model), local_files_only=True)
    complete = read(args.output / "data/sources/restore-complete.json")
    require(complete["status"] == "complete" and complete["documents"] == 5632,
            "Restore all exact historical articles before native preparation")
    seen = set()
    for source in ("c4", "dolma3"):
        path = args.output / "data/sources" / f"articles-{source}.jsonl"
        for document in jsonl(path):
            require(sha(document["text"]) == document["text_sha256"], "Recovered article changed")
            cluster = "article:" + document["cluster_id"]
            require(cluster in assignments and cluster not in seen, "Unknown/repeated original article cluster")
            seen.add(cluster)
            source_id = document["source_id"]
            native_role = old.SYMBOLS[old.balanced_block_choice("SDUCAT", document["source_ordinal"],
                         20261026, "single-role-v1", source)]
            old_ids = old_tokenizer.encode(document["text"], add_special_tokens=False)
            rng = random.Random(int(old.sha(f"20261026:{source_id}:single")[:16], 16))
            body, fragment = old._extract_fragment(old_tokenizer, old_ids, rng, "single", 1.0)
            if native_role in ("cot", "assistant"):
                field = "reasoning_content" if native_role == "cot" else "content"
                messages = [{"role": "user", "content": ""},
                            {"role": "assistant", "content": "", "reasoning_content": ""}]
                messages[1][field] = body
                index = 1
            else:
                field, index = "content", 0
                messages = [{"role": native_role, "content": body}]
                if native_role == "tool":
                    messages[0]["tool_call_id"] = source_id + "/single"
            event = source_id + "/single/r0/" + native_role
            fields = [{"message_index": index, "field": field, "event_id": event, "source_role": native_role}]
            identity = {"id": source_id + "/single/native-deepseek", "domain": "c4_dolma3",
                        "split": assignments[cluster], "source_id": source_id, "source_cluster": cluster,
                        "corpus": source, "structure": "c4_dolma3:single", "source_role": native_role,
                        "article_text_sha256": document["text_sha256"], "fragment_text_sha256": sha(body),
                        "original_fragment": fragment, "origin_partition": document["split"]}
            yield tokenize_record(tokenizer, encoder, messages, fields, identity,
                                  max_tokens=args.max_input_tokens, target_cap=97)
    require(seen == set(assignments), "Not all original article assignments were reproduced")


def normalize_message(message):
    """Keep original semantic fields; convert JSON tool arguments for encoder."""
    result = copy.deepcopy(message)
    content = result.get("content")
    if isinstance(content, list):
        require(all(block.get("type") == "text" for block in content), "Nontext AgentDojo body")
        result["content"] = "\n".join(block.get("text", "") for block in content)
    elif content is None:
        result["content"] = ""
    require(isinstance(result.get("content"), str), "Invalid normalized message content")
    for call in result.get("tool_calls") or []:
        function = call.get("function", call)
        if not isinstance(function.get("arguments"), str):
            function["arguments"] = json.dumps(function.get("arguments", {}), ensure_ascii=False)
    return result


def body_event_id(task, message, field, body):
    """Stable event IDs survive original context resets without merging calls."""
    source_role = message["role"]
    if source_role == "tool":
        anchor = "tool_result/" + message["tool_call_id"]
    elif source_role == "assistant":
        tool_ids = [call.get("id", "") for call in message.get("tool_calls") or []]
        if tool_ids:
            require(all(tool_ids), "Assistant tool-call identity is missing")
            anchor = "assistant_tool_calls/" + "/".join(tool_ids)
        else:
            anchor = "assistant_text/" + sha(message.get("content") or body)
    else:
        anchor = source_role + "/" + sha(body)
    return f"{task}/{anchor}/{field}"


def agentdojo_records(args, tokenizer, encoder, assignments, audit):
    """Teacher-force frozen benign trajectories; supervise each body event once.

    Appending the current archived response lets a single causal forward capture
    new input and output body positions. Prior input tokens cannot attend to the
    appended response. This is cross-model replay, not a new DeepSeek rollout.
    """
    paths = sorted((args.agentdojo / "calls").glob("*/*/*/call_*/request.json"))
    require(len(paths) == 359, "Expected all original 359 benign calls")
    original_audit = read(args.original / "agentdojo-source-audit.json")["consumed_source_files"]
    seen, fingerprints, tasks = set(), {}, set()
    for path in paths:
        relative = path.relative_to(args.agentdojo).as_posix()
        require(relative in original_audit and file_sha(path) == original_audit[relative]["sha256"],
                "Historical AgentDojo request differs from the original mixed experiment: " + relative)
        request = read(path)
        require(request["status"] == "complete", "Incomplete historical AgentDojo request")
        meta = request["metadata"]
        task = f"{meta['suite']}/{meta['user_task_id']}"
        cluster = "agentdojo:" + task
        require(cluster in assignments, "Unknown original AgentDojo task")
        tasks.add(cluster)
        messages = [normalize_message(message) for message in request["normalized_messages"]]
        if meta["call_index"] == 0:
            audit.setdefault("first_call_structure", []).append({"task": task,
                "roles": [message["role"] for message in messages],
                "content_characters": [len(message["content"]) for message in messages]})
        instructions = next((message for message in messages if message["role"] in ("system", "developer")), None)
        require(instructions is not None, "Task instructions are missing")
        instructions["tools"] = copy.deepcopy(request["tools"])
        response_index = len(messages)
        messages.append(normalize_message(request["response"]))
        fields = []
        for index, message in enumerate(messages):
            source_role = message["role"]
            require(source_role in SOURCE_ROLES, "Unexpected AgentDojo source role")
            for field in ("content", "reasoning_content"):
                body = message.get(field) or ""
                if not body:
                    continue
                role = "cot" if field == "reasoning_content" else source_role
                event = body_event_id(task, message, field, body)
                signature = (role, sha(body))
                if event in fingerprints:
                    require(fingerprints[event] == signature, f"Historical body changed for stable event {event}")
                fingerprints[event] = signature
                if event in seen:
                    continue
                seen.add(event)
                fields.append({"message_index": index, "field": field, "event_id": event,
                               "source_role": role,
                               "scope": "actual_generated" if index == response_index else "actual_input"})
        require(fields, "Call contains no newly observed ordinary body fields")
        identity = {"id": meta["call_id"] + "/native-deepseek", "domain": "agentdojo",
                    "split": assignments[cluster], "source_id": task, "source_cluster": cluster,
                    "task_id": task, "suite": meta["suite"], "call_index": meta["call_index"],
                    "structure": "agentdojo:native", "source_request_sha256": file_sha(path),
                    "source_request": str(path), "trajectory_policy": "frozen_gptoss_teacher_forced_replay"}
        record = tokenize_record(tokenizer, encoder, messages, fields, identity, max_tokens=args.max_input_tokens)
        audit.setdefault("source_requests", {})[meta["call_id"]] = file_sha(path)
        yield record
    require(tasks == set(assignments), "Missing original AgentDojo task assignments")
    audit["tasks"], audit["calls"], audit["first_occurrence_body_events"] = len(tasks), len(paths), len(seen)


def prepare(args):
    from transformers import AutoTokenizer

    root = args.output / "data"
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    require(tokenizer.is_fast, "Fast tokenizer offsets are required for pure body masks")
    encoder = load_encoder(args.model)
    binding = {path.name: file_sha(path) for path in [args.model / "tokenizer.json", args.model / "tokenizer_config.json",
               args.model / "encoding/encoding_dsv4.py", args.model / "encoding/README.md"] if path.exists()}
    old_binding = read(args.article_metadata / "data-manifest.json")["configuration"]["tokenizer_file_binding"]["files_sha256"]
    fragment_binding = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        require((args.old_model / name).is_file(), "Original article-fragment tokenizer asset is missing: " + name)
        fragment_binding[name] = file_sha(args.old_model / name)
        require(fragment_binding[name] == old_binding[name], "Original article-fragment tokenizer differs: " + name)
    original = read(args.original / "split-assignment.json")
    assignments = original["domains"]
    require(len(assignments["c4_dolma3"]) == 5632 and len(assignments["agentdojo"]) == 57,
            "Original mixed source/task assignments changed")
    protocol = {"format_version": FORMAT_VERSION, "roles": list(ROLES), "source_role_mapping": ROLE_MAPPING,
        "tokenizer_binding": binding, "model_path": str(args.model.resolve()),
        "article_fragment_tokenizer_binding": fragment_binding,
        "original_split_assignment_sha256": file_sha(args.original / "split-assignment.json"),
        "article_source_keys_sha256": file_sha(args.article_metadata / "source-keys.json"),
        "article_restore_sha256": file_sha(root / "sources/restore-complete.json"),
        "implementation_sha256": file_sha(__file__), "maximum_input_tokens": args.max_input_tokens,
        "article_target_cap": 97, "article_sampling_seed": 20261115,
        "article_target_rule": "Lowest deterministic family hashes after native token/body offset transfer",
        "article_fragment_rule": "Reproduce original single-message role assignment and text fragment with original seed 20261026 and byte-verified original tokenizer; native DeepSeek retokenization follows",
        "domain_weights": {"c4_dolma3": 0.5, "agentdojo": 0.5},
        "sampling": "Uniform domain, then uniform actually present native role, then uniform source event position",
        "head_selection": "FIT only, with inner source-cluster split; DEV for classifier selection only",
        "test_policy": "No TEST head selection, scaler fit, candidate selection, or tuning; freeze selected probe before TEST evaluation",
        "role_mapping_evidence": {
            "developer": "Official render_message emits the identical <｜User｜> prefix for developer/user; collapse to user",
            "cot": "assistant.reasoning_content between native <think> and </think>",
            "tool": "Official merge_tool_messages embeds tool result bodies inside native user <tool_result> blocks",
            "documentation": str(args.model / "encoding/README.md"),
            "caveat": "Developer is an internal search-agent input role, not a distinct model token or public API role"},
        "agentdojo_adaptation": "Use normalized semantic task messages and official tools schemas; exclude GPT-OSS injected system identity/channel instructions and namespace serialization; no artificial system targets; cross-model teacher-forced replay"}
    if (root / "manifest.json").exists():
        previous = read(root / "manifest.json")
        require(previous["protocol"] == protocol, "Prepared data protocol changed; use a fresh output")
        require(all(file_sha(root / name) == digest for name, digest in previous["files_sha256"].items()), "Prepared input changed")
        print(json.dumps({"stage": "prepared_data_reused", "records": previous["records"]}), flush=True)
        return previous
    atomic_json(root / "preparation-protocol.json", protocol)
    atomic_json(root / "split-assignment.json", original)
    streams = {split: (root / f"inputs-{split}.jsonl.part").open("w") for split in SPLITS}
    counts, roles, clusters, input_lengths = Counter(), Counter(), defaultdict(set), Counter()
    audit = {"status": "running"}
    try:
        for iterator in [article_records(args, tokenizer, encoder, assignments["c4_dolma3"]),
                         agentdojo_records(args, tokenizer, encoder, assignments["agentdojo"], audit)]:
            for record in iterator:
                split, domain = record["split"], record["domain"]
                streams[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[f"{split}/{domain}/records"] += 1
                counts[f"{split}/{domain}/targets"] += len(record["targets"])
                clusters[split].add(record["source_cluster"])
                input_lengths[domain] = max(input_lengths[domain], record["input_tokens"])
                roles.update(f"{split}/{domain}/{target['role']}" for target in record["targets"])
                if sum(value for key, value in counts.items() if key.endswith("/records")) % 250 == 0:
                    print(json.dumps({"stage": "native_data", "counts": counts, "maximum_input_tokens": input_lengths}), flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    overlap = {f"{a}/{b}": len(clusters[a] & clusters[b]) for a, b in (("fit", "dev"), ("fit", "test"), ("dev", "test"))}
    require(not any(overlap.values()), "Cross-split source/task leakage")
    for split in SPLITS:
        require(all(roles[f"{split}/c4_dolma3/{role}"] for role in ROLES), "Article split missing a native role")
        (root / f"inputs-{split}.jsonl.part").replace(root / f"inputs-{split}.jsonl")
    audit.update(status="passed", counts=dict(counts), targets_by_split_domain_role=dict(roles),
                 maximum_input_tokens=dict(input_lengths), cross_split_cluster_overlap=overlap,
                 present_roles={f"{split}/{domain}": [role for role in ROLES if roles[f"{split}/{domain}/{role}"]]
                                for split in SPLITS for domain in ("c4_dolma3", "agentdojo")})
    atomic_json(root / "data-audit.json", audit)
    manifest = {"format_version": FORMAT_VERSION, "protocol": protocol, "roles": list(ROLES),
                "records": sum(counts[f"{split}/{domain}/records"] for split in SPLITS for domain in ("c4_dolma3", "agentdojo")),
                "counts": dict(counts), "maximum_input_tokens": dict(input_lengths),
                "files_sha256": {name: file_sha(root / name) for name in
                    ["inputs-fit.jsonl", "inputs-dev.jsonl", "inputs-test.jsonl", "split-assignment.json", "data-audit.json"]}}
    atomic_json(root / "manifest.json", manifest)
    print(json.dumps({"stage": "native_data_complete", **manifest}), flush=True)
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=["restore-articles", "prepare"])
    result.add_argument("--output", type=Path, default=OUTPUT)
    result.add_argument("--original", type=Path, default=ORIGINAL)
    result.add_argument("--article-metadata", type=Path, default=ARTICLE_META)
    result.add_argument("--model", type=Path, default=MODEL)
    result.add_argument("--old-model", type=Path, default=OLD_MODEL)
    result.add_argument("--agentdojo", type=Path, default=AGENTDOJO)
    result.add_argument("--max-input-tokens", type=int, default=16384)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.stage == "restore-articles":
        restore_articles(arguments)
    else:
        prepare(arguments)
