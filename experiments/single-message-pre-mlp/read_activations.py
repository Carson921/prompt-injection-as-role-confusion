"""Read one message/layer without loading an entire activation shard."""
import argparse
from pathlib import Path


def read_message(output, message_id, layer, content_only=True):
    import json
    import pyarrow.parquet as pq
    from safetensors import safe_open
    output = Path(output)
    meta = json.loads((output / "manifest.json").read_text())
    if not 0 <= message_id < meta["message_count"] or layer not in meta["layers"]:
        raise ValueError("Invalid message ID or layer")
    rows = pq.read_table(output / "messages.parquet").to_pylist()
    row = rows[message_id]
    shard = next(s for s in meta["shards"] if s["message_start"] <= message_id < s["message_end"])
    path = output / "shards" / f"shard-{shard['shard_id']:05d}.safetensors"
    with safe_open(path, framework="pt", device="cpu") as stream:
        local_message = message_id - shard["message_start"]
        offsets = stream.get_tensor("message_offsets")
        start, end = offsets[local_message].item(), offsets[local_message + 1].item()
        if content_only:
            end = start + row["content_end"]
            start += row["content_start"]
        states = stream.get_slice(f"pre_mlp.layer_{layer:02d}")[start:end].clone()
        ids = stream.get_tensor("input_ids")[start:end].clone()
    return row, ids, states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("message_id", type=int)
    parser.add_argument("layer", type=int)
    parser.add_argument("--include-headers", action="store_true")
    args = parser.parse_args()
    row, ids, states = read_message(args.output, args.message_id, args.layer, not args.include_headers)
    print({"message_id": row["message_id"], "source": row["source"], "sample_id": row["sample_id"],
           "label": row["label"], "shape": list(states.shape), "dtype": str(states.dtype),
           "finite": bool(states.isfinite().all())})
