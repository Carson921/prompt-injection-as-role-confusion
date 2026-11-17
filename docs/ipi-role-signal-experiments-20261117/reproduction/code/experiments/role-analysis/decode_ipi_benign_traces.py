"""Export display-only native text from the committed IPI token-ID traces.

This is a CPU tokenizer operation. The original token-ID JSONL remains the
authoritative inference source; decoded text is never fed back into inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


CAVEAT = ("Display only: decoded with skip_special_tokens=False. Original source "
          "token IDs, not this text or a retokenization, are authoritative. "
          "Opaque terminal context remains uninterpreted text.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ids_sha(ids):
    """SHA-256 of the compact ASCII JSON integer array, in original order."""
    require(isinstance(ids, list) and all(type(value) is int for value in ids),
            "Source input_ids must be the original JSON integer array")
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def write_identical_or_new(path, data):
    if path.exists():
        require(path.read_bytes() == data, f"Existing readable export differs: {path}")
    else:
        path.write_bytes(data)


def export(output, model=None):
    import tokenizers

    output = Path(output).resolve()
    source_path = output / "sources/traces.jsonl"
    manifest_path = output / "sources/all-collection-manifest.json"
    prepared_path = output / "prepared/manifest.json"
    collection_path = output / "collection/protocol.json"
    manifest, prepared, collection = read(manifest_path), read(prepared_path), read(collection_path)
    require(manifest["status"] == "complete" and manifest["traces"] == 41,
            "All 41 source traces must be committed before text export")
    require(manifest["source_records_sha256"] == sha(source_path)
            and manifest["collection_protocol_sha256"] == sha(collection_path)
            and collection["prepared_manifest_sha256"] == sha(prepared_path),
            "Committed source/collection/prepared hashes differ")
    model = Path(model) if model is not None else Path(prepared["model_path"])
    tokenizer_path = model / "tokenizer.json"
    tokenizer_sha = sha(tokenizer_path)
    require(tokenizer_sha == prepared["tokenizer_sha256"]["tokenizer.json"],
            "Tokenizer differs from the frozen prepared inputs")
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    sources = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    expected = {row["trace_id"] for row in prepared["rows"]}
    require(len(sources) == 41 and len(expected) == 41
            and {row["trace_id"] for row in sources} == expected,
            "Readable export must contain exactly the fixed 41 trace identities")
    rows = []
    for source in sources:
        ids = source["input_ids"]
        rows.append(dict(trace_id=source["trace_id"], source_split=source["source_split"],
            category=source["category"], token_count=len(ids),
            source_token_ids_sha256=ids_sha(ids),
            token_ids_hash_encoding="SHA-256 of compact ASCII JSON integer array, original order",
            native_text=tokenizer.decode(ids, skip_special_tokens=False),
            opaque_terminal_start=source.get("opaque_terminal_start"),
            allow_terminal_body=source.get("allow_terminal_body", False),
            caveat=CAVEAT))
    target = output / "sources/native-traces-readable.jsonl"
    encoded = ("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                         for row in rows) + "\n").encode("utf-8")
    write_identical_or_new(target, encoded)

    # Re-read the actual delivery and audit every row against authoritative IDs.
    delivered = [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
    require(len(delivered) == len(sources), "Readable delivery row count differs")
    inventory = []
    for source, row in zip(sources, delivered):
        ids = source["input_ids"]
        require(row["trace_id"] == source["trace_id"]
                and row["source_split"] == source["source_split"]
                and row["category"] == source["category"]
                and row["token_count"] == len(ids)
                and row["source_token_ids_sha256"] == ids_sha(ids)
                and row["native_text"] == tokenizer.decode(ids, skip_special_tokens=False)
                and row["opaque_terminal_start"] == source.get("opaque_terminal_start")
                and row["allow_terminal_body"] == source.get("allow_terminal_body", False)
                and row["caveat"] == CAVEAT,
                f"Readable delivery differs from source: {source['trace_id']}")
        inventory.append(dict(trace_id=row["trace_id"], token_count=row["token_count"],
                              source_token_ids_sha256=row["source_token_ids_sha256"]))
    result = dict(format_version="ipi-native-trace-readable-export-v1", status="complete",
        traces=len(delivered), total_tokens=sum(row["token_count"] for row in delivered),
        caveat=CAVEAT, source_untouched=True, retokenization_for_inference=False,
        decoder="tokenizers.Tokenizer.from_file(tokenizer.json).decode(ids, skip_special_tokens=False)",
        tokenizers_version=tokenizers.__version__, python_version=sys.version.split()[0],
        tokenizer=dict(path=str(tokenizer_path.resolve()), sha256=tokenizer_sha),
        exporter_sha256=sha(Path(__file__)),
        source_hashes={"sources/traces.jsonl": sha(source_path),
                       "sources/all-collection-manifest.json": sha(manifest_path),
                       "collection/protocol.json": sha(collection_path),
                       "prepared/manifest.json": sha(prepared_path)},
        output=dict(path="sources/native-traces-readable.jsonl", sha256=sha(target),
                    bytes=target.stat().st_size),
        validation=dict(status="passed", rows=len(delivered),
                        all_token_counts_and_source_id_hashes_verified=True,
                        all_native_text_redecoded_exactly=True,
                        complete_fixed_41_identity_inventory_verified=True),
        trace_inventory=inventory)
    destination = output / "sources/native-traces-readable-manifest.json"
    write_identical_or_new(destination, (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    result = export(args.output, args.model)
    print(json.dumps({key: result[key] for key in ["status", "traces", "total_tokens", "output", "validation"]}))


if __name__ == "__main__":
    main()
