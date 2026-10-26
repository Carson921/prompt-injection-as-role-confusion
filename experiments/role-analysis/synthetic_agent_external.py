"""Post-freeze authentic OASST/ToxicChat evaluation, with explicit provenance.

``prepare_external_protocol`` reads only historical ID/hash ledgers.  New public
conversation bodies and legacy CSV bodies are read exclusively after a verified
six-role classifier freeze.  UA is scored; the added S/D paragraphs are context.
"""
from __future__ import annotations

from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata

import synthetic_agent_data as data

VERSION = "synthetic-agent-external-v1"
LEGACY_HASH_VERSION = "synthetic-agent-legacy-hashes-v1"
HASH_PRIVACY_STATEMENT = "Hash-only local transformation; no historical text, fragments, token IDs or reversible text encoding."
HERE = Path(__file__).resolve().parent
PINS = {
    "oasst": {
        "repo": "OpenAssistant/oasst1", "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "file": "data/train-00000-of-00001-b42a775f407cee45.parquet",
        "sha256": "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e",
    },
    "toxicchat": {
        "repo": "lmsys/toxic-chat", "revision": "29df8e4dba60e1f4af4b4075c0705c5b313548a8",
        "file": "data/0124/toxic-chat_annotation_train.csv",
        "sha256": "702eb9b7cac96c3c35e28b9b95855a71f26f21afba8666f9d243f1fa469e81ed",
    },
}
HISTORY_METADATA = (
    "outputs/local-gptoss120b/systemness-natural-transcripts/dev-ledger.json",
    "outputs/local-gptoss120b/systemness-natural-test/test-ledger.json",
    "outputs/local-gptoss120b/systemness-complementary-natural-confirmation/test-ledger.json",
    "outputs/local-gptoss120b/systemness-complementary-natural-confirmation/natural-source-keys.json",
)
LEGACY_CSV = (
    "data/conversations/gptoss-20b.csv",
    "data/conversations/gptoss-20b-raw.csv",
    "outputs/local-gptoss20b/user_queries.csv",
)
BODY_HASH_FIELDS = {
    "raw_sha256", "normalized_raw_sha256", "clean_sha256", "capped_sha256",
    "serialized_body_sha256", "root_raw_sha256", "root_normalized_sha256",
    "body_sha256", "normalized_body_sha256", "user_input_sha256",
}
# Original, factual expository paragraphs fixed before any external text is read.
# They are not claimed to be the context under which the source dialogue arose.
PREFIXES = {
    "system": (
        "Water moves between the atmosphere, the land, and the oceans. Sunlight supplies energy "
        "for evaporation, and water vapour can condense into small droplets or ice crystals. "
        "Precipitation returns water to the surface. Some of this water enters rivers, some "
        "soaks into soil, and some is stored as snow or ice. These processes occur at different "
        "rates and connect local weather with the broader water cycle."
    ),
    "developer": (
        "Sedimentary rocks form as material accumulates and becomes consolidated. The material "
        "may include mineral grains, fragments of older rocks, or the remains of organisms. "
        "Layers can preserve differences in the conditions under which the deposits formed. "
        "Later erosion may expose these layers at the surface, where their textures and "
        "relationships provide evidence about the history of a landscape."
    ),
}
CONTROL = re.compile(r"<\|[^\n<>]{1,80}\|>")
OLD_SEGMENT = re.compile(
    r"<\|start\|>assistant<\|channel\|>(?:analysis|final)<\|message\|>(.*?)"
    r"(?:<\|end\|>|<\|return\|>|<\|call\|>|$)", re.S)


def _sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalized(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()


def _body_keys(text):
    return {_sha(text), _sha(_normalized(text))} if text.strip() else set()


def _dataset(value):
    value = str(value).casefold()
    if value.startswith("oasst"):
        return "oasst"
    if value.startswith("toxicchat") or value.startswith("toxic-chat"):
        return "toxicchat"
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def _collect_history(value, body_hashes, native_ids, message_ids, dataset=None):
    """Collect a whitelist of metadata fields, never use a prose body field."""
    if isinstance(value, list):
        for item in value:
            _collect_history(item, body_hashes, native_ids, message_ids, dataset)
    elif isinstance(value, dict):
        dataset = _dataset(value.get("dataset", dataset or ""))
        uid = value.get("source_uid")
        if isinstance(uid, str) and ":" in uid:
            ds, native = uid.split(":", 1)
            native_ids.add((_dataset(ds), native))
        if value.get("native_id") is not None and dataset in PINS:
            native_ids.add((dataset, str(value["native_id"])))
        for key in ("root_message_id", "source_message_id"):
            if value.get(key) is not None and dataset == "oasst":
                message_ids.add(str(value[key]))
        for key, item in value.items():
            if key in BODY_HASH_FIELDS and isinstance(item, str) and re.fullmatch(r"[a-f0-9]{64}", item):
                body_hashes.add(item)
            elif isinstance(item, (dict, list)):
                _collect_history(item, body_hashes, native_ids, message_ids, dataset)


def prepare_external_protocol(config, data_dir, run_dir, history_root=None):
    """Freeze source revisions, prefixes, eligibility, seeds and old exclusions.

    This function is safe before classifier freezing: it never opens new
    OASST/ToxicChat files and never decodes a historical CSV body.  Run locally
    where the archived metadata exists; copy data_dir/external to the run host.
    """
    options = dict(config.get("external", {}))
    history_root = Path(history_root or options.get("history_root", HERE)).resolve()
    root = Path(data_dir) / "external"
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json"
    index_path = root / "history-index.json"
    pointer_path = Path(run_dir) / "external-protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["history_index_sha256"] != _file_sha(index_path):
            raise ValueError("Historical exclusion index changed")
        if protocol["seed"] != int(options.get("seed", 2026102601)):
            raise ValueError("External sampling seed changed after protocol declaration")
        if protocol["prefixes"] != PREFIXES or protocol["source_pins"] != PINS or protocol["implementation_sha256"] != _file_sha(__file__):
            raise ValueError("External prefix/source protocol differs from implementation")
        if not pointer_path.is_file():
            if (Path(run_dir) / "freeze.json").exists():
                raise ValueError("An external protocol cannot be attached after model freezing")
            _atomic_json(pointer_path, {
                "protocol_path": str(protocol_path.resolve()), "protocol_sha256": _file_sha(protocol_path),
                "history_index_sha256": _file_sha(index_path), "new_external_bodies_read": False})
        else:
            pointer = json.loads(pointer_path.read_text())
            if pointer.get("protocol_sha256") != _file_sha(protocol_path) or pointer.get("history_index_sha256") != _file_sha(index_path):
                raise ValueError("Run directory binds a different external protocol")
        return protocol
    if (Path(run_dir) / "freeze.json").exists():
        raise ValueError("Declare the external protocol before model freezing")
    hashes, ids, messages = set(), set(), set()
    metadata_files = options.get("history_metadata", list(HISTORY_METADATA))
    if not metadata_files:
        raise ValueError("Historical source/hash ledgers are mandatory")
    file_pins = []
    for relative in metadata_files:
        path = history_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Historical exclusion metadata is required: {path}")
        _collect_history(json.loads(path.read_text()), hashes, ids, messages)
        file_pins.append({"path": str(relative), "sha256": _file_sha(path)})
    if not hashes or not ids:
        raise ValueError("Historical ledgers contain no body hashes/native source IDs")
    csv_pins = []
    for relative in options.get("legacy_csv", list(LEGACY_CSV)):
        path = history_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Legacy natural CSV is required for post-freeze de-duplication: {path}")
        # Only hash file bytes here; no CSV body field is decoded pre-freeze.
        csv_pins.append({"path": str(relative), "sha256": _file_sha(path)})
    index = {"body_sha256": sorted(hashes), "native_ids": sorted([list(x) for x in ids]),
             "oasst_message_ids": sorted(messages), "metadata_files": file_pins,
             "scope": "All archived natural FIT, DEV and TEST source/body metadata are excluded conservatively."}
    _atomic_json(index_path, index)
    protocol = {
        "format_version": VERSION, "declared_at_unix": time.time(),
        "implementation_sha256": _file_sha(__file__),
        "seed": int(options.get("seed", 2026102601)), "count_per_source": 128,
        "source_pins": PINS, "prefixes": PREFIXES,
        "prefix_provenance": "Fixed original factual paragraphs added for this evaluation; not original source generation context. Unscored in authentic-dialogue metrics.",
        "scored_roles": ["user", "assistant"], "unobserved_roles": ["system", "developer", "cot", "tool"],
        "max_input_tokens": 4096, "max_oasst_user_turns": 3,
        "eligibility": {
            "oasst": "English, ready_for_export root; every selected message nondeleted and nonsynthetic; deterministic alternating UA path of at most 3 user turns, ending in authentic assistant.",
            "toxicchat": "Authentic user_input and model_output from pinned corpus; nonempty pair; no new model generation.",
            "both": "Reject reserved Harmony control strings, historical native/message IDs or exact/NFKC-casefold-whitespace body overlap, cross-selected source/body overlap, and complete inputs exceeding 4096 tokens. No message truncation.",
        },
        "ordering": "SHA256(seed:dataset:revision:native_id); path children ranked by SHA256(seed:message_id). No role predictions or scores are consulted.",
        "history_index_sha256": _file_sha(index_path), "history_metadata": file_pins,
        "legacy_csv": csv_pins, "history_root_at_declaration": str(history_root),
        "new_external_body_read_before_classifier_freeze": False,
        "final_external_results_may_select_classifier": False,
    }
    _atomic_json(protocol_path, protocol)
    _atomic_json(pointer_path, {
        "protocol_path": str(protocol_path.resolve()), "protocol_sha256": _file_sha(protocol_path),
        "history_index_sha256": _file_sha(index_path), "new_external_bodies_read": False})
    return protocol


def _check_freeze(run_dir, protocol, protocol_sha256=None, *, verify_pointer=True):
    root = Path(run_dir)
    path = root / "freeze.json"
    if not path.is_file():
        raise RuntimeError("External conversation bodies may only be opened after freeze.json exists")
    frozen = json.loads(path.read_text())
    if frozen.get("probe_sha256") != _file_sha(root / "best.npz"):
        raise ValueError("Frozen classifier was modified")
    if frozen.get("selection_sha256") != _file_sha(root / "selection.json"):
        raise ValueError("Frozen DEV selection was modified")
    if verify_pointer:
        pointer = root / "external-protocol.json"
        if not pointer.is_file() or frozen.get("external_protocol_pointer_sha256") != _file_sha(pointer):
            raise ValueError("External protocol pointer was not bound at model freeze")
        declared = json.loads(pointer.read_text())
        if declared.get("protocol_sha256") != protocol_sha256:
            raise ValueError("External protocol pointer binds a different protocol")
    if not protocol_sha256 or frozen.get("external_protocol_sha256") != protocol_sha256:
        raise ValueError("External protocol differs from its model-freeze binding")
    if frozen.get("external_history_index_sha256") != protocol["history_index_sha256"]:
        raise ValueError("External historical exclusions differ from model freeze")
    if frozen.get("metadata", {}).get("engineering_smoke_only"):
        raise ValueError("An engineering smoke classifier cannot open external evaluation")
    if protocol["declared_at_unix"] >= frozen["frozen_at_unix"]:
        raise ValueError("Declare the external protocol before classifier freezing")
    return frozen


def _snapshot(pin, options, cache_dir):
    # Cache files are never opened by protocol-only preparation.
    dataset_key = next(name for name, spec in PINS.items() if spec == pin)
    sources = options.get("source_files", {})
    supplied = sources.get(dataset_key, sources.get(pin["repo"], sources.get(pin["repo"].split("/")[-1])))
    if supplied:
        path = Path(supplied)
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=pin["repo"], repo_type="dataset", revision=pin["revision"],
                                    filename=pin["file"], cache_dir=str(cache_dir)))
    if _file_sha(path) != pin["sha256"]:
        raise ValueError(f"Public source snapshot SHA256 differs: {pin['repo']}")
    return path


def _extend_legacy_hashes(protocol, options, tokenizer, index, *, history_root_override=None):
    """Post-freeze only; old body text is hashed and never reaches a readout."""
    history_root = Path(history_root_override or options.get("history_root", protocol["history_root_at_declaration"]))
    hashes = set(index["body_sha256"])
    old_users = {"oasst": set(), "toxicchat": set()}
    occurrences = 0
    for pin in protocol["legacy_csv"]:
        path = history_root / pin["path"]
        if _file_sha(path) != pin["sha256"]:
            raise ValueError(f"Historical natural CSV changed: {path}")
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                ds = _dataset(row.get("dataset", ""))
                if row.get("user_query") and ds in old_users:
                    old_users[ds].update(_body_keys(row["user_query"]))
                texts = [row.get(k, "") for k in ("user_query", "assistant", "cot", "model_prefix")]
                texts += OLD_SEGMENT.findall(row.get("state_text", ""))
                for text in texts:
                    if not text.strip():
                        continue
                    occurrences += 1
                    hashes.update(_body_keys(text))
                    # Old training capped bodies at 192 tokens. Include their
                    # exact and normalized versions to catch source fragments.
                    clean = CONTROL.sub(lambda m: "[" + m.group()[2:-2] + "]", text)
                    hashes.update(_body_keys(clean))
                    if tokenizer is not None:
                        ids = tokenizer.encode(clean, add_special_tokens=False)[:192]
                        if ids:
                            hashes.update(_body_keys(tokenizer.decode(ids, skip_special_tokens=False)))
    return hashes, old_users, {"legacy_body_occurrences_hashed": occurrences,
                               "historical_body_hashes": len(hashes), "model_predictions_read": False,
                               "legacy_192_token_cap_hashes_added": tokenizer is not None}


def _augmentation_binding(protocol, protocol_path, run_dir, frozen):
    return {
        "protocol_sha256": _file_sha(protocol_path),
        "history_index_sha256": protocol["history_index_sha256"],
        "freeze_sha256": _file_sha(Path(run_dir) / "freeze.json"),
        "probe_sha256": frozen["probe_sha256"],
        "implementation_sha256": _file_sha(__file__),
        "legacy_csv_sha256": {pin["path"]: pin["sha256"] for pin in protocol["legacy_csv"]},
    }


def _hash_list(values, name):
    if not isinstance(values, list) or any(
            not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value) for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain only distinct irreversible SHA256 digests")
    return set(values)


def export_legacy_hashes(config, data_dir, run_dir, *, history_root_override=None,
                         tokenizer=None, output_path=None):
    """LOCAL ONLY: after freeze, convert archived CSV bodies to hash sets.

    No source text, text fragment, token IDs, or reversible encoding is written.
    Only this hash-only JSON artifact is intended for synchronization. The raw
    CSV files remain at ``history_root_override`` or the local configured root.
    ``tokenizer`` is optional; existing archived capped-body hashes remain in
    the exclusion index, and a supplied local tokenizer adds legacy 192-token
    cap hashes without sending the tokenizer any text off the local host.
    """
    options = dict(config.get("external", {}))
    root = Path(data_dir) / "external"
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("implementation_sha256") != _file_sha(__file__):
        raise ValueError("Legacy hash exporter differs from the frozen external protocol")
    # The remote pointer contains a remote filesystem path. For this local
    # transformation, bind the actual protocol hash directly to the copied
    # freeze manifest rather than requiring an identical local path pointer.
    frozen = _check_freeze(run_dir, protocol, _file_sha(protocol_path), verify_pointer=False)
    index_path = root / "history-index.json"
    if _file_sha(index_path) != protocol["history_index_sha256"]:
        raise ValueError("Historical metadata index changed")
    index = json.loads(index_path.read_text())
    hashes, users, audit = _extend_legacy_hashes(protocol, options, tokenizer, index,
                                               history_root_override=history_root_override)
    binding = _augmentation_binding(protocol, protocol_path, run_dir, frozen)
    record = {"format_version": LEGACY_HASH_VERSION, "binding": binding,
              "body_sha256": sorted(hashes),
              "user_prompt_sha256": {ds: sorted(users[ds]) for ds in ("oasst", "toxicchat")},
              "audit": audit,
              "privacy": HASH_PRIVACY_STATEMENT}
    _hash_list(record["body_sha256"], "body_sha256")
    for ds, values in record["user_prompt_sha256"].items():
        _hash_list(values, f"user_prompt_sha256.{ds}")
    output = Path(output_path) if output_path else root / "legacy-hashes.json"
    _atomic_json(output, record)
    return {"path": str(output.resolve()), "sha256": _file_sha(output), "binding": binding,
            "body_hashes": len(hashes), "prompt_hashes": {ds: len(values) for ds, values in users.items()},
            "historical_text_exported": False}


def _load_legacy_hashes(path, protocol, protocol_path, run_dir, frozen):
    """REMOTE SAFE: consume digests only; never fall back to archived CSVs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Hash-only legacy augmentation is required; run export_legacy_hashes locally after freeze and synchronize only legacy-hashes.json")
    record = json.loads(path.read_text())
    allowed = {"format_version", "binding", "body_sha256", "user_prompt_sha256", "audit", "privacy"}
    if set(record) != allowed or record["format_version"] != LEGACY_HASH_VERSION:
        raise ValueError("Unknown legacy hash-only augmentation format")
    if record["binding"] != _augmentation_binding(protocol, protocol_path, run_dir, frozen):
        raise ValueError("Legacy hash augmentation differs from frozen protocol/model/CSV/implementation bindings")
    hashes = _hash_list(record["body_sha256"], "body_sha256")
    if set(record["user_prompt_sha256"]) != {"oasst", "toxicchat"}:
        raise ValueError("Legacy prompt hashes must identify exactly the two public sources")
    users = {ds: _hash_list(values, f"user_prompt_sha256.{ds}")
             for ds, values in record["user_prompt_sha256"].items()}
    audit = record["audit"]
    if not isinstance(audit, dict) or set(audit) != {
            "legacy_body_occurrences_hashed", "historical_body_hashes", "model_predictions_read",
            "legacy_192_token_cap_hashes_added"} or record["privacy"] != HASH_PRIVACY_STATEMENT:
        raise ValueError("Invalid legacy hash audit")
    if type(audit["legacy_body_occurrences_hashed"]) is not int or type(audit["historical_body_hashes"]) is not int or audit["model_predictions_read"] is not False or type(audit["legacy_192_token_cap_hashes_added"]) is not bool:
        raise ValueError("Legacy hash audit must contain only declared counts and boolean flags")
    return hashes, users, {**record["audit"], "augmentation_sha256": _file_sha(path),
                           "historical_csv_read_on_run_host": False}


def _oasst_candidates(path, protocol, old_users):
    import pandas as pd
    frame = pd.read_parquet(path)
    required = {"message_id", "parent_id", "message_tree_id", "role", "lang", "tree_state", "deleted", "synthetic", "text"}
    if not required.issubset(frame.columns):
        raise ValueError("Pinned OASST schema changed")
    by_id = {str(row["message_id"]): row for row in frame.to_dict("records")}
    children = {}
    for key, row in by_id.items():
        if pd.notna(row["parent_id"]):
            children.setdefault(str(row["parent_id"]), []).append(key)
    for key in children:
        children[key].sort(key=lambda child: _sha(f"{protocol['seed']}:{child}"))
    trees = {}
    for row in by_id.values():
        trees.setdefault(str(row["message_tree_id"]), []).append(row)
    for native, rows in trees.items():
        all_hashes = set().union(*(_body_keys(row["text"]) for row in rows
                                  if isinstance(row["text"], str) and row["text"].strip()))
        historical_prompt = bool(all_hashes & old_users["oasst"])
        roots = [row for row in rows if pd.isna(row["parent_id"])]
        if len(roots) != 1:
            continue
        root = roots[0]
        if root["role"] != "prompter" or root["tree_state"] != "ready_for_export":
            continue
        path_rows = []
        current = str(root["message_id"])
        for position in range(6):
            row = by_id[current]
            if row["lang"] != "en" or bool(row["deleted"]) or bool(row["synthetic"]) or not isinstance(row["text"], str) or not row["text"].strip():
                break
            expected = "prompter" if position % 2 == 0 else "assistant"
            if row["role"] != expected:
                break
            path_rows.append(row)
            candidates = [key for key in children.get(current, [])
                          if by_id[key]["role"] != expected and by_id[key]["lang"] == "en"
                          and not bool(by_id[key]["deleted"]) and not bool(by_id[key]["synthetic"])]
            if not candidates:
                break
            current = candidates[0]
        if len(path_rows) % 2:
            path_rows.pop()  # Entire incomplete trailing U is not a completed exchange.
        if len(path_rows) < 2:
            continue
        yield {"dataset": "oasst", "native_id": native, "source_message_ids": [str(row["message_id"]) for row in rows],
               "messages": [{"symbol": "U" if row["role"] == "prompter" else "A", "body": str(row["text"]),
                             "source_message_id": str(row["message_id"]), "source_synthetic": False,
                             "source_deleted": False} for row in path_rows],
               "all_tree_body_hashes": all_hashes, "historical_prompt_match": historical_prompt}


def _toxicchat_candidates(path, protocol, old_users):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"conv_id", "user_input", "model_output"}.issubset(reader.fieldnames or []):
            raise ValueError("Pinned ToxicChat schema changed")
        for row in reader:
            bodies = [row["user_input"], row["model_output"]]
            keys = set().union(*(_body_keys(text) for text in bodies))
            yield {"dataset": "toxicchat", "native_id": str(row["conv_id"]), "source_message_ids": [],
                   "messages": [{"symbol": "U", "body": bodies[0], "source_message_id": row["conv_id"] + ":user"},
                                {"symbol": "A", "body": bodies[1], "source_message_id": row["conv_id"] + ":model",
                                 "source_model_response": True}],
                   "all_tree_body_hashes": keys,
                   "historical_prompt_match": bool(_body_keys(bodies[0]) & old_users["toxicchat"])}


def _encode(candidate, protocol, tokenizer):
    source = f"external:{candidate['dataset']}:{candidate['native_id']}"
    messages = []
    provenance = []
    raw_messages = [{"symbol": "S", "body": protocol["prefixes"]["system"]},
                    {"symbol": "D", "body": protocol["prefixes"]["developer"]}] + candidate["messages"]
    for i, message in enumerate(raw_messages):
        if not message["body"].strip() or CONTROL.search(message["body"]):
            raise ValueError("empty_or_reserved_control_body")
        event_id = f"{source}:message-{i}"
        messages.append({**message, "event_id": event_id, "message_id": event_id,
                         "round_ix": max(0, (i - 2) // 2)})
        provenance.append({"event_id": event_id, "source_message_id": message.get("source_message_id"),
                           "raw_sha256": _sha(message["body"]),
                           "normalized_raw_sha256": _sha(_normalized(message["body"])),
                           "origin": "fixed added article prefix; not original source context" if i < 2 else "authentic source conversation",
                           "scored": i >= 2})
    record = {"id": source, "snapshot_id": source + ":native", "source_id": source,
              "cluster_id": source, "split": "test", "profile": "external",
              "template_id": candidate["dataset"], "snapshot_stage": "authentic_conversation",
              "history_policy": "no_C_in_source", "call_chain_id": source,
              "round_segments": ["UA"] * (len(candidate["messages"]) // 2),
              "messages": messages, "dataset": candidate["dataset"], "native_id": candidate["native_id"],
              "source_revision": protocol["source_pins"][candidate["dataset"]]["revision"],
              "message_provenance": provenance, "prefix_provenance": protocol["prefix_provenance"],
              "external_scoring_scope": "authentic_UA_only"}
    encoded = data.encode_record(record, tokenizer, max_tokens=protocol["max_input_tokens"])
    encoded["prefix_diagnostic_targets"] = [target for target in encoded["targets"] if target["role"] in {"system", "developer"}]
    encoded["targets"] = [target for target in encoded["targets"] if target["role"] in {"user", "assistant"}]
    if set(target["role"] for target in encoded["targets"]) != {"user", "assistant"}:
        raise ValueError("authentic_UA_body_missing")
    return encoded


def prepare_external(config, tokenizer, data_dir, run_dir):
    """Read external bodies only after a verified freeze; write GPU-ready JSONL."""
    options = dict(config.get("external", {}))
    root = Path(data_dir) / "external"
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError("Declare the external protocol before classifier freezing")
    protocol = json.loads(protocol_path.read_text())
    frozen = _check_freeze(run_dir, protocol, _file_sha(protocol_path))
    if protocol["source_pins"] != PINS or protocol["prefixes"] != PREFIXES or protocol["implementation_sha256"] != _file_sha(__file__):
        raise ValueError("External protocol differs from fixed source/prefix implementation")
    if _file_sha(root / "history-index.json") != protocol["history_index_sha256"]:
        raise ValueError("Historical exclusion index changed")
    index = json.loads((root / "history-index.json").read_text())
    augmentation_path = Path(options.get("legacy_hash_artifact", root / "legacy-hashes.json"))
    hashes, old_users, legacy_audit = _load_legacy_hashes(augmentation_path, protocol, protocol_path, run_dir, frozen)
    if not set(index["body_sha256"]).issubset(hashes):
        raise ValueError("Legacy augmentation omitted frozen archived body hashes")
    existing = root / "manifest.json"
    if existing.exists():
        result = json.loads(existing.read_text())
        if result["protocol_sha256"] != _file_sha(protocol_path) or result["probe_sha256"] != frozen["probe_sha256"]:
            raise ValueError("External inputs belong to another frozen experiment")
        if result["inputs_sha256"] != _file_sha(root / "inputs-test.jsonl"):
            raise ValueError("External inputs changed")
        if result["historical_de_duplication"].get("augmentation_sha256") != legacy_audit["augmentation_sha256"]:
            raise ValueError("External inputs bind a different legacy hash augmentation")
        return result
    native_ids = {tuple(x) for x in index["native_ids"]}
    message_ids = set(index["oasst_message_ids"])
    cache_dir = Path(options.get("cache_dir", root / "source-cache"))
    paths = {ds: _snapshot(pin, options, cache_dir) for ds, pin in PINS.items()}
    factories = {"oasst": _oasst_candidates, "toxicchat": _toxicchat_candidates}
    selected = []
    used = set()
    counts, exclusions = Counter(), Counter()
    for ds in ("oasst", "toxicchat"):
        candidates = list(factories[ds](paths[ds], protocol, old_users))
        candidates.sort(key=lambda item: _sha(f"{protocol['seed']}:{ds}:{PINS[ds]['revision']}:{item['native_id']}"))
        seen_native = set()
        for candidate in candidates:
            native = candidate["native_id"]
            if native in seen_native:
                raise ValueError("Duplicate native source ID in pinned public source")
            seen_native.add(native)
            if counts[ds] >= 128:
                break
            key = (ds, native)
            body_keys = candidate["all_tree_body_hashes"]
            if key in native_ids or candidate["historical_prompt_match"]:
                exclusions[ds + ":historical_native_or_prompt"] += 1
                continue
            if set(candidate["source_message_ids"]) & message_ids:
                exclusions[ds + ":historical_message_id"] += 1
                continue
            if body_keys & hashes:
                exclusions[ds + ":historical_body_hash"] += 1
                continue
            if body_keys & used:
                exclusions[ds + ":cross_selected_body_hash"] += 1
                continue
            try:
                record = _encode(candidate, protocol, tokenizer)
            except (ValueError, OverflowError) as error:
                exclusions[ds + ":" + ("input_too_long" if isinstance(error, OverflowError) else str(error))] += 1
                continue
            counts[ds] += 1
            selected.append(record)
            used.update(body_keys)
        if counts[ds] != 128:
            raise RuntimeError(f"Only {counts[ds]} eligible historical-disjoint {ds} conversations; protocol requires 128")
    output = root / "inputs-test.jsonl"
    temporary = output.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for record in selected:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, output)
    target_counts = Counter(target["role"] for record in selected for target in record["targets"])
    result = {"format_version": VERSION, "created_at_unix": time.time(),
              "protocol_sha256": _file_sha(protocol_path), "probe_sha256": frozen["probe_sha256"],
              "freeze_sha256": _file_sha(Path(run_dir) / "freeze.json"),
              "inputs_sha256": _file_sha(output), "inputs": output.name,
              "counts_by_source": dict(counts), "scored_body_tokens_by_role": dict(target_counts),
              "unobserved_roles": {r: {"recall": None, "status": "N/A; false positives still reported"}
                                   for r in ("system", "developer", "cot", "tool")},
              "prefix_provenance": protocol["prefix_provenance"], "exclusions": dict(exclusions),
              "historical_de_duplication": legacy_audit,
              "historical_metadata_sha256": protocol["history_index_sha256"],
              "source_snapshots": {ds: {**PINS[ds], "path": str(paths[ds])} for ds in paths},
              "new_model_generation": False, "readout_selection_permitted": False}
    _atomic_json(existing, result)
    _atomic_json(Path(run_dir) / "external-preparation.json", result)
    return result
