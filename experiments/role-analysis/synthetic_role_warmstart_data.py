"""CPU-only, source-disjoint article preparation for the six-hour warm start.

Article text and token IDs are private experiment caches, not Git artifacts.
No historical agent/conversation text is read by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import itertools
import json
from pathlib import Path
import random
import re
import time

import synthetic_agent_data as base

FORMAT = "synthetic-role-warm-data-v1"
SEED = 20261027
LENGTHS = (128, 256, 512, 1024)
WRAPPERS = ("article_before_tools", "article_after_tools", "article_description")
CONTEXTS = ("SDUA", "SDUCA", "SDUCQ", "SDUQTCA", "SDUQTA", "SD+UA+UQTCA", "SD+UCA+UQTA+UCQ")


def warm_article_rejection(text):
    """Additional corpus-wide quality rules established by source auditing.

    These detect page/list structure, not model predictions or source topics.
    News, product descriptions, fiction/rules prose and isolated captions are
    not rejected merely for their subject matter.
    """
    phone_notices = re.findall(r"\bcall\s+\+?[\d ()-]{6,}\s+for\s+(?:more\s+)?information\b", text, re.I)
    event_listings = re.findall(r"\b(?:is|are)\s+hosting\b|\b(?:upcoming\s+events|event\s+calendar)\b", text, re.I)
    if len(phone_notices) >= 3 and len(event_listings) >= 3:
        return "repeated_event_listing_phone_notices"
    posted = re.findall(r"\bwas\s+posted\b", text, re.I)
    views = re.findall(r"\b(?:has\s+)?viewed\s+by\s+\d+\s+(?:visitors?|people|users?)\b", text, re.I)
    if len(posted) >= 3 and len(views) >= 3:
        return "repeated_posted_view_count_gallery"
    related_tail = re.search(r"\b(?:featured|related|recommended)\s+(?:industries|products|categories|articles|pages|resources)\b", text, re.I)
    ellipses = len(re.findall(r"\.\.\.|…", text))
    merged_titles = re.findall(r"\b[a-z]{2,}[A-Z][a-z]+(?:[- ][A-Z][A-Za-z]+)*", text)
    if related_tail and ellipses >= 2 and len(merged_titles) >= 2:
        return "truncated_related_cards_and_merged_titles"
    listing_markers = [r"\bmultiple\s+listing\s+service\b", r"\bbuyer\s+to\s+verify\b",
                       r"\bproperties\s+may\s+or\s+may\s+not\s+be\s+listed\b",
                       r"\blisting\s+provided\s+courtesy\b", r"\bproperties\s+are\s+subject\s+to\s+prior\s+sale\b",
                       r"\binformation\s+deemed\s+reliable\s+but\s+not\s+guaranteed\b"]
    if (sum(bool(re.search(marker, text, re.I)) for marker in listing_markers) >= 3
            and re.search(r"\b(?:bedrooms?|bathrooms?|en-suite|great\s+room|covered\s+deck|patio|realty|realtors)\b", text, re.I)):
        return "property_listing_with_repeated_mls_disclaimers"
    inventory = re.findall(r"\b\d+(?:\s*/\s*\d+)?\s*[- ]?\s*(?:bedrooms?|bath(?:rooms?)?|living[- ]rooms?|car[- ]garage|sqft|sq\.?\s*ft|square\s*(?:feet|foot)|season\s+sun\s+porch)\b", text, re.I)
    transaction_markers = [r"\brental\s+(?:property|investment)\b", r"\btenants?\b", r"\blandlords?\b",
                           r"\bowner\s+occup", r"\bsee\s+it\s+today\b", r"\bturn[- ]?key\b",
                           r"\basking\s+price\b", r"\bfor\s+(?:sale|rent)\b", r"\blisting\b"]
    feature = re.compile(r"\b(?:bedrooms?|bath(?:rooms?)?|kitchens?|garages?|living\s+rooms?|sun\s+porch|master\s+bedroom|eat-in|sq(?:uare)?\s*(?:ft|feet)|sqft|fenced\s+lawn|woodstove|deck|patio)\b", re.I)
    clauses = re.split(r"(?<=[.!?;])\s+|\n+", text)
    total_words = max(len(base.ALPHA_WORDS.findall(text)), 1)
    inventory_coverage = sum(len(base.ALPHA_WORDS.findall(clause)) for clause in clauses if feature.search(clause)) / total_words
    if (len(inventory) >= 3 and sum(bool(re.search(marker, text, re.I)) for marker in transaction_markers) >= 2
            and len(feature.findall(text)) >= 6 and inventory_coverage >= .35):
        return "property_transaction_room_inventory"
    return None


def quality_exclusions(old_documents, supplementary, old_sources_path, supplementary_path=None, manual_rejections=None):
    manual = {}
    manual_hashes = {}
    if manual_rejections:
        payload = json.loads(Path(manual_rejections).read_text())
        entries = payload.get("rejected_sources", payload.get("reviews", [])) if isinstance(payload, dict) else payload
        for entry in entries:
            if entry.get("decision", "reject").startswith("pass"):
                continue
            if entry.get("source_id"):
                manual[entry["source_id"]] = entry.get("reason", "explicit_full_article_semantic_rejection")
            if entry.get("text_sha256"):
                manual_hashes[entry["text_sha256"]] = entry.get("reason", "explicit_full_article_semantic_rejection")
    rejected = []
    for document in old_documents + supplementary:
        digest = base.sha(document["text"])
        reason = manual.get(document["source_id"]) or manual_hashes.get(digest)
        kind = "manual_full_article_review" if reason else "general_quality_rule"
        reason = reason or warm_article_rejection(document["text"])
        if reason:
            rejected.append({"source_id": document["source_id"], "cluster_id": document["cluster_id"],
                             "source": document["source"], "split": document["split"], "text_sha256": digest,
                             "reason": reason, "decision_origin": kind})
    excluded = {r["cluster_id"] for r in rejected}
    return {"format_version": "warmstart-source-quality-exclusions-v1", "code_sha256": base.file_sha(__file__),
            "old_sources_sha256": base.file_sha(old_sources_path),
            "supplementary_sources_sha256": base.file_sha(supplementary_path) if supplementary_path else None,
            "manual_rejections_sha256": base.file_sha(manual_rejections) if manual_rejections else None,
            "assessed_sources": len(old_documents) + len(supplementary),
            "excluded_clusters": sorted(excluded), "reasons": rejected,
            "counts_by_reason": dict(Counter(r["reason"] for r in rejected)),
            "counts_by_source_split": dict(Counter(f"{r['source']}/{r['split']}" for r in rejected)),
            "scope": "Every retainedV7 and supplementary candidate, irrespective of FIT/DEV/TEST use; no probe scores consulted"}


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def _rng(*parts):
    return random.Random(int(base.sha(list(parts))[:16], 16))


def read_documents(path):
    return list(base.read_jsonl(path))


def _length_qualified(document, tokenizer, minimum):
    ids = tokenizer.encode(document["text"], add_special_tokens=False)
    return len(ids) >= minimum, ids


def select_documents(old_documents, supplementary, tokenizer, fit_count=512, dev_count=128, test_count=128):
    """Long bins allocate first; output order makes FIT256 a balanced prefix.

    An original TEST document can never become development data. Supplementary
    sources already have a deterministic split. Insufficient length eligibility
    is an explicit error, never repetition or a shorter mislabeled fragment.
    """
    quotas = {"fit": fit_count, "dev": dev_count, "test": test_count}
    if any(n % 16 for n in quotas.values()):
        raise ValueError("Every source quota must be divisible by 16")
    old = [dict(x, origin="retained_v7") for x in old_documents if x["split"] in ("fit", "dev")]
    old_ids = {d["source_id"] for d in old_documents}
    extra = [dict(x, origin="fresh_pinned_stream") for x in supplementary]
    if any(d["source_id"] in old_ids for d in extra):
        raise ValueError("Supplementary source collides with a retained historical source")
    if len({d["source_id"] for d in extra}) != len(extra):
        raise ValueError("Supplementary source IDs repeat")
    pool = old + extra
    token_ids = {d["source_id"]: tokenizer.encode(d["text"], add_special_tokens=False) for d in pool}
    selected, deficits = [], {}
    for split, count in quotas.items():
        for source in ("c4", "dolma3"):
            candidates = [d for d in pool if d["split"] == split and d["source"] == source]
            candidates.sort(key=lambda d: (d["origin"] != "retained_v7", base.sha([SEED, split, source, d["source_id"]])))
            used = set()
            buckets = {}
            for length in reversed(LENGTHS):
                qualified = [d for d in candidates if d["source_id"] not in used and len(token_ids[d["source_id"]]) >= length]
                wanted = count // 8
                chosen = qualified[:wanted]
                if len(chosen) != wanted:
                    deficits[f"{split}/{source}/{length}"] = wanted - len(chosen)
                buckets[length] = []
                for d in chosen:
                    used.add(d["source_id"])
                    buckets[length].append(dict(d, paired_length=length))
            # Round robin guarantees each first half remains corpus/length balanced.
            for ordinal in range(count // 8):
                for length in LENGTHS:
                    if ordinal < len(buckets[length]):
                        selected.append(dict(buckets[length][ordinal], cell_ordinal=ordinal))
    if deficits:
        raise ValueError("Missing article length/source/split cells: " + json.dumps(deficits, sort_keys=True))
    selected.sort(key=lambda d: (d["split"], d["cell_ordinal"], LENGTHS.index(d["paired_length"]), d["source"]))
    base.validate_source_partitions(selected)
    return selected


def _fragment(document, tokenizer, length, namespace):
    ids = tokenizer.encode(document["text"], add_special_tokens=False)
    if len(ids) < length:
        raise ValueError(f"{document['source_id']} has fewer than {length} article tokens")
    positions = len(ids) - length + 1
    start = _rng(SEED, document["source_id"], namespace).randrange(positions)
    body = None
    for offset in range(positions):
        candidate_start = (start + offset) % positions
        candidate = tokenizer.decode(ids[candidate_start:candidate_start + length], skip_special_tokens=False)
        # Byte-BPE slices may bisect a Unicode character. Such a decode can
        # invent U+FFFD; use an exact original-text substring instead of silently
        # training on replacement characters that were absent from the article.
        if candidate.strip() and not base.CONTROL.search(candidate) and candidate in document["text"]:
            start, body = candidate_start, candidate
            break
    if body is None:
        raise ValueError("No exact original-text slice of the requested token length")
    return body, {"source_id": document["source_id"], "article_token_start": start,
                  "article_token_length": length, "body_sha256": base.sha(body)}


def _tool_definition(count=1):
    tail = "".join(f"type lookup_{i} = (_: {{ document_id: string, query: string }}) => any;\n" for i in range(1, count))
    return base.TOOL_DEFINITION.replace("} // namespace functions", tail + "} // namespace functions")


def _parts(symbol, body, wrapper, tool_count):
    if symbol != "D":
        return "", body, ""
    tools = _tool_definition(tool_count)
    if wrapper == "article_before_tools":
        return "", body, tools
    if wrapper == "article_after_tools":
        return tools + "\n\n", body, ""
    if wrapper == "article_description":
        if "*/" in body:
            raise ValueError("Article cannot be placed inside a tool documentation comment")
        prefix = "\n\n# Tools\n\n## functions\n\nnamespace functions {\n\n/**\n"
        suffix = "\n*/\ntype lookup = (_: {document_id: string, query: string}) => any;\n"
        suffix += "".join(f"type lookup_{i} = (_: {{ document_id: string, query: string }}) => any;\n" for i in range(1, tool_count))
        suffix += "} // namespace functions\n"
        return prefix, body, suffix
    raise ValueError(wrapper)


def render(messages):
    pieces, spans, cursor = [], [], 0
    for i, message in enumerate(messages):
        symbol = message["symbol"]
        if symbol == "Q":
            header, ending = "assistant to=functions.lookup<|channel|>commentary", "<|call|>"
        elif symbol == "C":
            header, ending = "assistant<|channel|>analysis", "<|end|>"
        elif symbol == "A":
            header = "assistant<|channel|>final"
            ending = "<|return|>" if i == len(messages) - 1 else "<|end|>"
        elif symbol == "T":
            header, ending = "functions.lookup", "<|end|>"
        else:
            header, ending = base.SYMBOLS[symbol], "<|end|>"
        tag = "<|start|>" + header + "<|message|>"
        prefix, body, suffix = message.get("prefix", ""), message["body"], message.get("suffix", "")
        begin = cursor + len(tag) + len(prefix)
        piece = tag + prefix + body + suffix + ending
        spans.append({"symbol": symbol, "role": base.SYMBOLS.get(symbol),
                      "header_start": cursor, "full_body_start": cursor + len(tag),
                      "body_start": begin, "body_end": begin + len(body),
                      "message_end": cursor + len(piece), "supervised": symbol != "Q",
                      "message_id": message["event_id"], "event_id": message["event_id"],
                      "round_ix": message.get("round_ix", 0), "call_id": message.get("call_id"),
                      "tool_definition": symbol == "D" and bool(prefix or suffix),
                      "article_provenance": message.get("article_provenance")})
        pieces.append(piece)
        cursor += len(piece)
    text = "".join(pieces)
    parsed = base.parse_native(text)
    if len(parsed) != len(spans):
        raise ValueError("Harmony construction/parser message count mismatch")
    for expected, actual in zip(spans, parsed):
        if any(expected[k] != actual[k] for k in ("role", "symbol", "header_start", "message_end")):
            raise ValueError("Harmony construction/parser mismatch")
        if not actual["body_start"] <= expected["body_start"] <= expected["body_end"] <= actual["body_end"]:
            raise ValueError("Article span escapes parsed message body")
    return text, spans


def encode(record, tokenizer, messages, max_input_tokens=4096):
    text, spans = render(messages)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) > max_input_tokens:
        raise OverflowError(f"Whole input {len(ids)} exceeds {max_input_tokens}")
    targets, boundaries = [], []
    for span in spans:
        if not span["supervised"]:
            continue
        pure = []
        for token_ix, (left, right) in enumerate(offsets):
            if right <= left or right <= span["body_start"] or left >= span["body_end"]:
                continue
            if left < span["body_start"] or right > span["body_end"]:
                boundaries.append({"token_ix": token_ix, "event_id": span["event_id"],
                                   "reason": "crosses_article_structure_boundary"})
                continue
            pure.append((token_ix, left - span["body_start"], right - span["body_start"]))
        n = len(pure)
        span_occurrences = Counter()
        for body_position, (token_ix, left, right) in enumerate(pure):
            token_id = ids[token_ix]
            occurrence = span_occurrences[left, right, token_id]
            span_occurrences[left, right, token_id] += 1
            alignment = f"{left}:{right}:{token_id}:{occurrence}"
            targets.append({"token_ix": token_ix, "label": base.ROLES.index(span["role"]),
                            "role": span["role"], "source_id": record["source_id"],
                            "message_id": span["message_id"], "event_id": span["event_id"],
                            "body_position": body_position, "body_tokens": n,
                            "body_char_start": left, "body_char_end": right,
                            "body_token_id": token_id, "paired_alignment_key": alignment,
                            "body_quartile": min(3, body_position * 4 // max(n, 1)),
                            "family_id": f"{span['event_id']}:{alignment}",
                            "structure": record["profile"] + ":" + record["template_id"],
                            "body_length_bin": str(next((v for v in LENGTHS if n <= v), ">1024")),
                            "allocated_paired_length": record["paired_length"],
                            "wrapper_type": record["wrapper_type"], "profile": record["profile"],
                            "weight": 1.0})
    record.update(text=text, input_ids=ids, input_tokens=len(ids), message_spans=spans,
                  targets=targets, boundary_tokens=boundaries, format_version=FORMAT)
    return record


def _quartile_sample(targets, namespace):
    """At most24 actual targets per body quartile; never interpolate tokens."""
    chosen = []
    for quarter in range(4):
        available = [t for t in targets if t["body_quartile"] == quarter]
        available.sort(key=lambda t: base.sha([SEED, namespace, t["paired_alignment_key"]]))
        chosen.extend(available[:24])
    return sorted(chosen, key=lambda t: t["token_ix"])


def make_records(document, tokenizer, ordinal=0):
    sid, length = document["source_id"], document["paired_length"]
    body, provenance = _fragment(document, tokenizer, length, "paired")
    paired = []
    for symbol in "SDUCAT":
        event = f"{sid}/paired/{symbol}"
        message = {"symbol": symbol, "body": body, "event_id": event, "article_provenance": provenance}
        record = {"id": event, "source_id": sid, "cluster_id": document["cluster_id"],
                  "source": document["source"], "split": document["split"],
                  "profile": "paired", "template_id": symbol, "paired_length": length,
                  "wrapper_type": "plain", "round_segments": [], "history_policy": "independent",
                  "snapshot_stage": "output", "paired_family_id": sid + "/paired"}
        paired.append(encode(record, tokenizer, [message]))
    # Same character-span positions across all six roles, excluding BPE boundaries.
    common = set.intersection(*[{t["paired_alignment_key"] for t in r["targets"]} for r in paired])
    anchor = [t for t in paired[0]["targets"] if t["paired_alignment_key"] in common]
    sampled = _quartile_sample(anchor, sid + "/paired-common")
    positions = {t["paired_alignment_key"] for t in sampled}
    for record in paired:
        record["targets"] = [t for t in record["targets"] if t["paired_alignment_key"] in positions]
        record["common_paired_positions"] = len(positions)
        if len(record["targets"]) != len(positions) or not positions:
            raise ValueError("Six-role paired article positions do not align")
    # Independent balanced permutation namespaces avoid binding S length to
    # wrapper style or D length to tool count through a shared modulo counter.
    source_ordinal = ordinal // 2
    def choice(options, namespace):
        return base.balanced_block_choice(list(options), source_ordinal, SEED, namespace, document["source"])
    template = choice(CONTEXTS, "warm-context-template")
    segments = template[3:].split("+") if template.startswith("SD+") else [template[2:]]
    wrapper = choice(WRAPPERS, "warm-developer-wrapper")
    tool_count = choice((1, 4, 12), "warm-tool-count")
    # A bounded, predeclared allocation covers long D without truncating messages.
    s_length = choice((64, 128, 256), "warm-system-length")
    d_length = choice((128, 512, 1024), "warm-developer-length")
    available = len(tokenizer.encode(document["text"], add_special_tokens=False))
    s_length, d_length = min(s_length, available), min(d_length, available)
    other_length = 64 if len(segments) > 1 else 128
    messages = []

    def add(symbol, round_ix, occurrence=0):
        event = f"{sid}/context/r{round_ix}/{symbol}{occurrence}"
        if symbol == "Q":
            message = {"symbol": symbol, "body": json.dumps({"document_id": sid, "query": f"excerpt-{round_ix}"}, sort_keys=True),
                       "event_id": event, "round_ix": round_ix, "call_id": f"{sid}/context/r{round_ix}/call"}
        else:
            target_length = s_length if symbol == "S" else d_length if symbol == "D" else min(other_length, available)
            fragment, prov = _fragment(document, tokenizer, target_length, event)
            prefix, fragment, suffix = _parts(symbol, fragment, wrapper, tool_count)
            message = {"symbol": symbol, "body": fragment, "prefix": prefix, "suffix": suffix,
                       "event_id": event, "round_ix": round_ix, "article_provenance": prov}
            if symbol == "T":
                message["call_id"] = f"{sid}/context/r{round_ix}/call"
        messages.append(message)

    add("S", 0)
    add("D", 0)
    for round_ix, segment in enumerate(segments, 1):
        for symbol in segment:
            add(symbol, round_ix)
    context = {"id": sid + "/context", "source_id": sid, "cluster_id": document["cluster_id"],
               "source": document["source"], "split": document["split"], "profile": "context",
               "template_id": template, "paired_length": length, "wrapper_type": wrapper,
               "round_segments": segments, "history_policy": "native_context", "snapshot_stage": "output",
               "context_lengths": {"system": s_length, "developer": d_length, "other": other_length},
               "tool_count": tool_count}
    context = encode(context, tokenizer, messages)
    by_role = defaultdict(list)
    for target in context["targets"]:
        by_role[target["role"]].append(target)
    context["targets"] = sorted([t for role, values in by_role.items() for t in _quartile_sample(values, sid + "/context/" + role)], key=lambda t: t["token_ix"])
    return paired + [context]


def audit_records(records, tokenizer, documents=None):
    counts = Counter()
    maximum, boundaries = 0, 0
    seen_ids, source_splits = set(), {}
    paired = defaultdict(list)
    source_map = {d["source_id"]: d for d in documents} if documents is not None else None
    article_cache = {}
    for record in records:
        if record["id"] in seen_ids:
            raise ValueError("Duplicate record ID")
        seen_ids.add(record["id"])
        sid, split = record["source_id"], record["split"]
        if sid in source_splits and source_splits[sid] != split:
            raise ValueError("Source crossed partitions")
        source_splits[sid] = split
        encoded = tokenizer(record["text"], add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] != record["input_ids"]:
            raise ValueError("Retokenization differs")
        if len(record["input_ids"]) > 4096:
            raise ValueError("Oversize complete input")
        parsed = base.parse_native(record["text"])
        for span in record["message_spans"]:
            if not span["supervised"]:
                continue
            body = record["text"][span["body_start"]:span["body_end"]]
            provenance = span["article_provenance"]
            if provenance["source_id"] != sid or base.sha(body) != provenance["body_sha256"]:
                raise ValueError("Article span differs from its source provenance")
            if source_map is not None:
                key = (sid, provenance["article_token_start"], provenance["article_token_length"])
                if key not in article_cache:
                    article_ids = tokenizer.encode(source_map[sid]["text"], add_special_tokens=False)
                    start, length = key[1:]
                    article_cache[key] = tokenizer.decode(article_ids[start:start + length], skip_special_tokens=False)
                if article_cache[key] != body:
                    raise ValueError("Supervised text is not the claimed original article slice")
                if body not in source_map[sid]["text"]:
                    raise ValueError("Supervised text is not an unmodified source substring")
        if record["profile"] == "context":
            if [p["symbol"] for p in parsed].count("S") != 1 or [p["symbol"] for p in parsed].count("D") != 1:
                raise ValueError("Context must contain exactly one opening S/D")
            if [p["symbol"] for p in parsed[:2]] != ["S", "D"]:
                raise ValueError("Context S/D prefix is out of order")
            pending = None
            for span in record["message_spans"]:
                if span["symbol"] == "Q":
                    if pending:
                        raise ValueError("Two unmatched tool calls")
                    pending = span["call_id"]
                elif span["symbol"] == "T":
                    if not pending or pending != span["call_id"]:
                        raise ValueError("Tool reply has no matching native call")
                    pending = None
        events, roles = Counter(), Counter()
        for target in record["targets"]:
            left, right = encoded["offset_mapping"][target["token_ix"]]
            candidates = [s for s in record["message_spans"] if s["supervised"] and s["body_start"] <= left < right <= s["body_end"]]
            if len(candidates) != 1 or candidates[0]["role"] != target["role"] or base.ROLES[target["label"]] != target["role"]:
                raise ValueError("Non-article target or incorrect native role")
            span = candidates[0]
            if record["text"][span["body_start"]:span["body_end"]] == "":
                raise ValueError("Empty supervised article")
            events[target["event_id"]] += 1
            roles[target["role"]] += 1
            counts[f"{split}/{record['profile']}/{target['role']}"] += 1
        if any(n > 96 for n in events.values()):
            raise ValueError("More than96 targets in one message")
        if any(n > 96 for n in roles.values()):
            raise ValueError("More than96 targets for one role in one record")
        if record["profile"] == "paired":
            paired[sid].append(record)
        maximum = max(maximum, len(record["input_ids"]))
        boundaries += len(record["boundary_tokens"])
    for sid, variants in paired.items():
        if len(variants) != 6 or {r["template_id"] for r in variants} != set("SDUCAT"):
            raise ValueError("Incomplete six-role paired family")
        bodies = [r["text"][r["message_spans"][0]["body_start"]:r["message_spans"][0]["body_end"]] for r in variants]
        positions = [{t["paired_alignment_key"] for t in r["targets"]} for r in variants]
        if len(set(bodies)) != 1 or any(p != positions[0] for p in positions):
            raise ValueError("Paired roles do not share identical content and positions")
    return {"valid": True, "records": len(records), "sources": len(source_splits), "targets": sum(counts.values()),
            "max_input_tokens": maximum, "excluded_mixed_boundary_tokens": boundaries,
            "targets_by_split_profile_role": dict(counts), "six_role_paired_families": len(paired),
            "retokenized_all_records": True, "all_targets_inside_article_spans": True,
            "original_source_slices_verified": source_map is not None}


def prepare(old_data_dir, tokenizer_dir, output_dir, supplementary=None, fit_count=512, dev_count=128, test_count=128, development_only=False, manual_rejections=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "data-manifest.json"
    if manifest_path.exists():
        raise FileExistsError("A completed preparation exists; use an independent directory")
    tokenizer = load_tokenizer(tokenizer_dir)
    old_path = Path(old_data_dir) / "sources.jsonl"
    old = read_documents(old_path)
    extra = read_documents(supplementary) if supplementary else []
    quality = quality_exclusions(old, extra, old_path, supplementary, manual_rejections)
    base.atomic_json(output_dir / "quality-excluded-clusters.json", quality)
    excluded = set(quality["excluded_clusters"])
    selected = select_documents([d for d in old if d["cluster_id"] not in excluded],
                                [d for d in extra if d["cluster_id"] not in excluded],
                                tokenizer, fit_count, dev_count, 0 if development_only else test_count)
    base.write_jsonl(output_dir / "sources.jsonl", selected)
    all_reports, hashes, split_stats = {}, {}, {}
    hashes["quality-excluded-clusters.json"] = base.file_sha(output_dir / "quality-excluded-clusters.json")
    fit_records = []
    for split in ("fit", "dev", "test"):
        docs = [d for d in selected if d["split"] == split]
        if not docs:
            continue
        records = []
        for ordinal, document in enumerate(docs):
            records.extend(make_records(document, tokenizer, ordinal))
        report = audit_records(records, tokenizer, docs)
        report["input_tokens"] = sum(r["input_tokens"] for r in records)
        report["sources_by_corpus_length"] = dict(Counter(f"{d['source']}/{d['paired_length']}" for d in docs))
        path = output_dir / f"inputs-{split}.jsonl"
        base.write_jsonl(path, records)
        hashes[path.name] = base.file_sha(path)
        all_reports[split] = report
        split_stats[split] = {"sources": len(docs), "records": len(records), "targets": report["targets"],
                              "input_tokens": report["input_tokens"], "max_input_tokens": report["max_input_tokens"]}
        print(json.dumps({"stage": "prepared_split", "split": split, **split_stats[split]}), flush=True)
        if split == "fit" and fit_count == 512:
            conservative_ids = {d["source_id"] for d in docs[:256]}
            conservative = [r for r in records if r["source_id"] in conservative_ids]
            base.write_jsonl(output_dir / "inputs-fit-conservative.jsonl", conservative)
            hashes["inputs-fit-conservative.jsonl"] = base.file_sha(output_dir / "inputs-fit-conservative.jsonl")
            all_reports["fit_conservative"] = audit_records(conservative, tokenizer, docs[:256])
            all_reports["fit_conservative"]["input_tokens"] = sum(r["input_tokens"] for r in conservative)
        if split == "fit":
            fit_records = records
    hashes["sources.jsonl"] = base.file_sha(output_dir / "sources.jsonl")
    base.atomic_json(output_dir / "mask-audit.json", all_reports)
    hashes["mask-audit.json"] = base.file_sha(output_dir / "mask-audit.json")
    base.atomic_json(output_dir / "source-keys.json", [{k: v for k, v in d.items() if k != "text"} for d in selected])
    hashes["source-keys.json"] = base.file_sha(output_dir / "source-keys.json")
    fit_sources = [d for d in selected if d["split"] == "fit"]
    pilot_ids = {d["source_id"] for d in fit_sources[:64]}
    pilot_records = [r for r in fit_records if r["source_id"] in pilot_ids]
    base.write_jsonl(output_dir / "inputs-pilot.jsonl", pilot_records)
    hashes["inputs-pilot.jsonl"] = base.file_sha(output_dir / "inputs-pilot.jsonl")
    base.atomic_json(output_dir / "pilot-input-manifest.json", {"format_version": FORMAT,
        "purpose": "Timing-only fixed64 FIT sources; no scores used to choose formal scale",
        "source_ids": sorted(pilot_ids), "sources": len(pilot_ids), "records": len(pilot_records),
        "input_tokens": sum(r["input_tokens"] for r in pilot_records),
        "input_sha256": hashes["inputs-pilot.jsonl"], "source_manifest_sha256": hashes["sources.jsonl"]})
    hashes["pilot-input-manifest.json"] = base.file_sha(output_dir / "pilot-input-manifest.json")
    # Audit the nested FIT256 pool so the same actual50 reviews apply to both scales.
    audit_fit_ids = {d["source_id"] for d in fit_sources[:256]}
    review = []
    for split, source, quota in (("fit", "c4", 15), ("fit", "dolma3", 15), ("dev", "c4", 10), ("dev", "dolma3", 10)):
        strata = {n: sorted([d for d in selected if d["split"] == split and d["source"] == source and d["paired_length"] == n
                            and (split != "fit" or d["source_id"] in audit_fit_ids)],
                            key=lambda d: base.sha([SEED, "article-audit", d["source_id"]])) for n in LENGTHS}
        chosen = []
        for ordinal in range(quota):
            length = LENGTHS[ordinal % 4]
            offset = ordinal // 4
            if offset >= len(strata[length]):
                raise ValueError("Too few sources for the fixed50 FIT/DEV semantic audit")
            chosen.append(strata[length][offset])
        review.extend(chosen)
    review_dir = output_dir / "article-audit-sample"
    review_dir.mkdir(exist_ok=True)
    sample = [{"sample_index": index, "source_id": d["source_id"], "source": d["source"], "split": d["split"],
               "paired_length": d["paired_length"], "text_sha256": base.sha(d["text"]),
               "url": d["normalized_url"], "text": d["text"], "review_status": "pending_llm_review"}
              for index, d in enumerate(review)]
    base.write_jsonl(review_dir / "articles.jsonl", sample)
    sample_metadata = [{k: v for k, v in entry.items() if k != "text"} for entry in sample]
    base.atomic_json(review_dir / "selection.json", {"seed": SEED, "selection": "Source-score-independent stratified source/split/length round robin, 15FIT+10DEV per corpus",
                                                      "samples": sample_metadata, "payload_sha256": base.file_sha(review_dir / "articles.jsonl"),
                                                      "reviewer_type": "LLM agent, not human gold", "requires_actual_full_article_review": True})
    hashes["article-audit-sample/articles.jsonl"] = base.file_sha(review_dir / "articles.jsonl")
    hashes["article-audit-sample/selection.json"] = base.file_sha(review_dir / "selection.json")
    manifest = {"format_version": FORMAT, "seed": SEED, "roles": base.ROLES,
                "code_sha256": base.file_sha(__file__), "base_builder_sha256": base.file_sha(base.__file__),
                "old_sources_sha256": base.file_sha(old_path),
                "supplementary_sha256": base.file_sha(supplementary) if supplementary else None,
                "tokenizer": base.tokenizer_file_binding(tokenizer),
                "quotas": {"fit": fit_count, "dev": dev_count, "test": 0 if development_only else test_count},
                "statistics": split_stats, "sha256": hashes,
                "split_cluster_ids": {split: sorted(d["cluster_id"] for d in selected if d["split"] == split)
                                      for split in ("fit", "dev", "test")},
                "source_partition_validation": base.validate_source_partitions(selected),
                "supervision": "Unmodified article spans only; schema, control, header and Q JSON excluded",
                "paired_rule": "Identical article content and common character-span targets in all6 native roles",
                "fit_conservative_rule": "First256 FIT sources in balanced corpus/length round-robin order",
                "article_audit_status": "requires_semantic_article_audit",
                "test_opened": not development_only, "development_only": development_only}
    base.atomic_json(manifest_path, manifest)
    if fit_count == 512:
        conservative_sources = [d for d in selected if d["split"] != "fit" or d["source_id"] in audit_fit_ids]
        base.write_jsonl(output_dir / "sources-conservative.jsonl", conservative_sources)
        conservative_manifest = copy.deepcopy(manifest)
        conservative_manifest["parent_manifest_sha256"] = base.file_sha(manifest_path)
        conservative_manifest["quotas"]["fit"] = 256
        conservative_manifest["input_files"] = {s: f"inputs-{s}.jsonl" for s in split_stats}
        conservative_manifest["input_files"]["fit"] = "inputs-fit-conservative.jsonl"
        conservative_manifest["source_file"] = "sources-conservative.jsonl"
        r = all_reports["fit_conservative"]
        conservative_manifest["statistics"]["fit"] = {"sources": r["sources"], "records": r["records"],
            "targets": r["targets"], "input_tokens": r["input_tokens"], "max_input_tokens": r["max_input_tokens"]}
        conservative_manifest["source_partition_validation"] = base.validate_source_partitions(conservative_sources)
        conservative_manifest["split_cluster_ids"] = {split: sorted(d["cluster_id"] for d in conservative_sources if d["split"] == split)
                                                     for split in ("fit", "dev", "test")}
        conservative_manifest["sha256"].pop("inputs-fit.jsonl")
        conservative_manifest["sha256"].pop("sources.jsonl")
        conservative_manifest["sha256"]["sources-conservative.jsonl"] = base.file_sha(output_dir / "sources-conservative.jsonl")
        base.atomic_json(output_dir / "data-manifest-conservative.json", conservative_manifest)
    return manifest


def _seed_deduplicator(documents):
    dedup = base.Deduplicator()
    dedup.raw_hashes = set()
    for d in documents:
        if d.get("normalized_sha256"):
            dedup.texts.add(d["normalized_sha256"])
        for key in ("normalized_url", "canonical_source_url", "source_url", "url"):
            if d.get(key):
                dedup.urls.add(base.normalize_url(d[key]))
        for key in ("text_sha256", "full_text_sha256", "full_body_sha256", "body_sha256", "short_body_sha256",
                    "prefix192_sha256", "prefix1024_sha256", "truncated_body_sha256"):
            if d.get(key):
                dedup.raw_hashes.add(d[key])
        if d.get("simhash64"):
            fingerprint = int(d["simhash64"], 16)
            for band in range(4):
                dedup.bands[(band, (fingerprint >> (16 * band)) & 65535)].append(fingerprint)
    return dedup


def prepare_pilot(old_data_dir, tokenizer_dir, output_dir):
    """Fixed64-source timing corpus; separate from formal preparation identity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "pilot-input-manifest.json").exists():
        raise FileExistsError("Pilot preparation already exists")
    tokenizer = load_tokenizer(tokenizer_dir)
    old = read_documents(Path(old_data_dir) / "sources.jsonl")
    docs = select_documents(old, [], tokenizer, 64, 0, 0)
    records = [record for ordinal, document in enumerate(docs) for record in make_records(document, tokenizer, ordinal)]
    report = audit_records(records, tokenizer, docs)
    base.write_jsonl(output_dir / "inputs-pilot.jsonl", records)
    base.write_jsonl(output_dir / "sources.jsonl", docs)
    manifest = {"format_version": FORMAT, "purpose": "Fixed64 FIT-source corpus used for timing only, not scores or formal cache reuse",
                "source_ids": sorted(d["source_id"] for d in docs), "statistics": report,
                "input_tokens": sum(r["input_tokens"] for r in records),
                "sources_by_corpus_length": dict(Counter(f"{d['source']}/{d['paired_length']}" for d in docs)),
                "input_sha256": base.file_sha(output_dir / "inputs-pilot.jsonl"),
                "sources_sha256": base.file_sha(output_dir / "sources.jsonl"),
                "code_sha256": base.file_sha(__file__), "tokenizer": base.tokenizer_file_binding(tokenizer)}
    base.atomic_json(output_dir / "pilot-input-manifest.json", manifest)
    return manifest


def collect(old_data_dir, tokenizer_dir, output_dir, source_config=None, history_keys=(), fit_count=512, dev_count=128, test_count=128, max_candidates=2000000, seed_sources=None, source_offsets=None, manual_rejections=None):
    """Collect supplementary sources only; no GPU is imported or initialized.

    Explicit source_config can select other splits at the same pinned revisions.
    Stream provenance records that choice instead of pretending validation/train
    are interchangeable. Only content-independent hash partition chooses split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    old = read_documents(Path(old_data_dir) / "sources.jsonl")
    tokenizer = load_tokenizer(tokenizer_dir)
    seeded_documents = read_documents(seed_sources) if seed_sources else []
    quality = quality_exclusions(old, seeded_documents, Path(old_data_dir) / "sources.jsonl", seed_sources, manual_rejections)
    excluded = set(quality["excluded_clusters"])
    eligible_old = [d for d in old if d["cluster_id"] not in excluded]
    manual_hashes = {d["text_sha256"] for d in quality["reasons"] if d["decision_origin"] == "manual_full_article_review"}
    historical = list(old)
    for path in history_keys:
        value = json.loads(Path(path).read_text())
        if isinstance(value, dict):
            value = value.get("sources", value.get("documents", value.get("keys", [])))
        if not isinstance(value, list):
            raise ValueError(f"Historical article keys must be a list: {path}")
        historical.extend(v for v in value if isinstance(v, dict))
    dedup = _seed_deduplicator(historical)
    if source_config:
        config = json.loads(Path(source_config).read_text())
        specs = config.get("sources", config)
    else:
        specs = json.loads((Path(old_data_dir) / "preparation-config.json").read_text())["sources"]
    if set(specs) != {"c4", "dolma3"} or any(len(s.get("revision", "")) != 40 for s in specs.values()):
        raise ValueError("Two source specifications with full pinned revision hashes are required")
    wanted = {"fit": fit_count // 8, "dev": dev_count // 8, "test": test_count // 8}
    # Count actual length-cell allocation from original development sources.
    remaining = {}
    for source in ("c4", "dolma3"):
        for split in ("fit", "dev", "test"):
            pool = [d for d in eligible_old if d["source"] == source and d["split"] == split and split != "test"]
            lengths = sorted([len(tokenizer.encode(d["text"], add_special_tokens=False)) for d in pool], reverse=True)
            for length in reversed(LENGTHS):
                eligible = [n for n in lengths if n >= length]
                take = min(wanted[split], len(eligible))
                remaining[source, split, length] = wanted[split] - take
                for _ in range(take):
                    lengths.remove(eligible.pop())
    provenance, filters = {}, Counter()
    supplemental = [d for d in seeded_documents if d["cluster_id"] not in excluded]
    if len({d["source_id"] for d in supplemental}) != len(supplemental):
        raise ValueError("Seed source IDs repeat")
    for document in supplemental:
        key = (document["source"], document["split"], document["reserved_length_bin"])
        if remaining.get(key, 0) <= 0:
            raise ValueError("Seed source cache exceeds the declared source/length quota")
        remaining[key] -= 1
    dedup = _seed_deduplicator(historical + seeded_documents)
    offsets = json.loads(Path(source_offsets).read_text()) if source_offsets else {}
    if set(offsets) - {"c4", "dolma3"} or any(not isinstance(n, int) or n < 0 for n in offsets.values()):
        raise ValueError("Invalid explicit stream offsets")
    destination = output_dir / "supplementary-sources.jsonl"
    log_path = output_dir / "supplementary-filter-log.jsonl"
    if destination.exists():
        raise FileExistsError("Supplementary source cache already exists; inspect it before retry")
    print(json.dumps({"stage": "collection_deficits", "cells": {"/".join(map(str, k)): v for k, v in remaining.items() if v}}), flush=True)
    with destination.open("w") as accepted, log_path.open("w") as log:
        for document in supplemental:
            accepted.write(json.dumps(document, ensure_ascii=False) + "\n")
        accepted.flush()
        streams = {name: itertools.islice(base._source_stream(name, spec, output_dir, provenance), offsets.get(name, 0), None)
                   for name, spec in specs.items()}
        processed = Counter(offsets)
        active = list(streams)
        while any(remaining.values()):
            if not active:
                raise RuntimeError("Pinned source streams exhausted before source/length quotas")
            for source in list(active):
                if not any(value for (name, _, _), value in remaining.items() if name == source):
                    active.remove(source)
                    continue
                try:
                    row = next(streams[source])
                except StopIteration:
                    active.remove(source)
                    continue
                processed[source] += 1
                if processed[source] > max_candidates:
                    raise RuntimeError("Supplementary source candidate limit reached")
                reason = base.article_filter(row)
                text = row.get("text", "")
                if reason is None:
                    reason = warm_article_rejection(text)
                if reason is None and base.sha(text) in manual_hashes:
                    reason = "manual_full_article_semantic_rejection"
                if reason is None and (base.sha(text) in dedup.raw_hashes or base.sha(text.strip()) in dedup.raw_hashes):
                    reason = "historical_raw_body_hash"
                url_info = base.source_url_info(row)
                info = None
                if reason is None:
                    info, reason = dedup.add(text, url_info["url"])
                split = base.assigned_split(info["cluster_id"], SEED) if info else None
                length = len(tokenizer.encode(text, add_special_tokens=False)) if info else 0
                slot = next((n for n in reversed(LENGTHS) if split and length >= n and remaining.get((source, split, n), 0) > 0), None)
                if reason is None and slot is None:
                    reason = "split_length_quota_filled"
                filters[f"{source}/{reason or 'accepted'}"] += 1
                log.write(json.dumps({"source": source, "candidate_index": processed[source] - 1,
                                      "text_sha256": base.sha(text) if isinstance(text, str) else None,
                                      "reason": reason or "accepted", "split": split, "qualified_tokens": length}) + "\n")
                if reason is None:
                    document = {"source_id": f"{source}-{base.sha(text)[:24]}", "source": source,
                                "split": split, "source_ordinal": sum(d["source"] == source and d["split"] == split for d in supplemental),
                                "text": text, "text_sha256": base.sha(text), **info,
                                "candidate_index": processed[source] - 1, "prose_metrics": base.prose_quality_metrics(text),
                                "source_url_available": url_info["available"], "source_url_field": url_info["field"],
                                "source_url_missing_reason": url_info["missing_reason"], "source_url_value_sha256": url_info["value_sha256"],
                                "source_url_metadata_type": url_info["metadata_type"], "source_url_metadata_parse": url_info["metadata_parse"],
                                "source_url_rejected_fields": url_info["rejected_fields"], "reserved_length_bin": slot,
                                "source_group": row.get("_synthetic_data_file_group")}
                    supplemental.append(document)
                    accepted.write(json.dumps(document, ensure_ascii=False) + "\n")
                    accepted.flush()
                    remaining[source, split, slot] -= 1
                if processed[source] % 5000 == 0:
                    print(json.dumps({"stage": "collect", "processed": dict(processed), "accepted": len(supplemental),
                                      "remaining": sum(remaining.values())}), flush=True)
    report = {"format_version": FORMAT, "seed": SEED, "source_provenance": provenance,
              "processed": dict(processed), "filters": dict(filters), "accepted": len(supplemental),
              "remaining": {"/".join(map(str, k)): v for k, v in remaining.items()},
              "source_sha256": base.file_sha(destination), "filter_log_sha256": base.file_sha(log_path),
              "old_sources_sha256": base.file_sha(Path(old_data_dir) / "sources.jsonl"),
              "seed_sources_sha256": base.file_sha(seed_sources) if seed_sources else None,
              "quality_filter_code_sha256": base.file_sha(__file__),
              "manual_rejections_sha256": base.file_sha(manual_rejections) if manual_rejections else None,
              "seed_quality_excluded_clusters": sorted(excluded),
              "source_offsets": offsets,
              "historical_keys": [{"path": str(p), "sha256": base.file_sha(p)} for p in history_keys],
              "source_partition_validation": base.validate_source_partitions(supplemental),
              "collection_only_no_test_predictions": True}
    base.atomic_json(output_dir / "supplementary-source-provenance.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "collect"):
        p = sub.add_parser(action)
        p.add_argument("--old-data-dir", type=Path, required=True)
        p.add_argument("--tokenizer-dir", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--fit-count", type=int, default=512)
        p.add_argument("--dev-count", type=int, default=128)
        p.add_argument("--test-count", type=int, default=128)
    sub.choices["prepare"].add_argument("--fresh-test-sources", "--supplementary", dest="supplementary", type=Path)
    sub.choices["prepare"].add_argument("--development-only", action="store_true")
    sub.choices["prepare"].add_argument("--manual-rejections", type=Path)
    sub.choices["collect"].add_argument("--source-config", type=Path)
    sub.choices["collect"].add_argument("--history-keys", type=Path, action="append", default=[])
    sub.choices["collect"].add_argument("--max-candidates", type=int, default=2000000)
    sub.choices["collect"].add_argument("--seed-sources", type=Path)
    sub.choices["collect"].add_argument("--source-offsets", type=Path)
    sub.choices["collect"].add_argument("--manual-rejections", type=Path)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--old-data-dir", type=Path, required=True)
    pilot.add_argument("--tokenizer-dir", type=Path, required=True)
    pilot.add_argument("--output-dir", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    action = args.pop("action")
    result = prepare(**args) if action == "prepare" else collect(**args) if action == "collect" else prepare_pilot(**args)
    print(json.dumps({"status": "complete", "action": action, "format_version": result["format_version"]}), flush=True)


if __name__ == "__main__":
    main()
