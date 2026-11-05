"""Verify safetensors with bounded pread calls, avoiding slow shared-FS mmap faults."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import struct

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_exact(fd, size, offset):
    chunks = []
    while size:
        chunk = os.pread(fd, size, offset)
        require(bool(chunk), "Unexpected end of safetensors file")
        chunks.append(chunk)
        size -= len(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def verify_shard(path, meta, shard, rows):
    """Validate the complete index and read actual BF16 samples from every layer."""
    layers, width, n = meta["layers"], meta["hidden_size"], shard["tokens"]
    with Path(path).open("rb", buffering=0) as stream:
        fd, size = stream.fileno(), os.fstat(stream.fileno()).st_size
        header_size = struct.unpack("<Q", read_exact(fd, 8, 0))[0]
        require(0 < header_size <= 16 << 20 and header_size + 8 <= size, "Invalid safetensors header length")
        header = json.loads(read_exact(fd, header_size, 8))
        start = 8 + header_size
        expected_keys = {f"pre_mlp.layer_{i:02d}" for i in layers} | {
            "input_ids", "label_ids", "message_ids", "message_offsets", "__metadata__"}
        require(set(header) == expected_keys, f"Missing or extra tensors in {path}")
        require(header["__metadata__"] == dict(signature=meta["signature"],
                shard_id=str(shard["shard_id"]), dtype="bfloat16"), "Shard metadata mismatch")
        item_sizes = {"BF16": 2, "I32": 4, "I8": 1, "I64": 8}
        spans = []
        for key, info in header.items():
            if key == "__metadata__":
                continue
            require(set(info) == {"dtype", "shape", "data_offsets"}, "Invalid tensor header")
            require(info["dtype"] in item_sizes and all(isinstance(x, int) and x >= 0 for x in info["shape"]), "Invalid tensor type/shape")
            begin, end = info["data_offsets"]
            require(0 <= begin <= end <= size - start, "Tensor exceeds file bounds")
            require(end - begin == math.prod(info["shape"]) * item_sizes[info["dtype"]], "Tensor byte size mismatch")
            spans.append((begin, end))
        cursor = 0
        for begin, end in sorted(spans):
            require(begin == cursor, "Overlapping tensor offsets or unindexed data")
            cursor = end
        require(start + cursor == size, "Trailing or missing tensor data")

        def tensor(name, dtype, shape, numpy_dtype):
            info = header[name]
            require(info["dtype"] == dtype and info["shape"] == shape, f"Wrong dtype/shape: {name}")
            begin, end = info["data_offsets"]
            return np.frombuffer(read_exact(fd, end - begin, start + begin), dtype=numpy_dtype)

        m = len(rows)
        require(m == shard["message_end"] - shard["message_start"], "Message count mismatch")
        message_ids = tensor("message_ids", "I64", [m], "<i8")
        offsets = tensor("message_offsets", "I64", [m + 1], "<i8")
        ids = tensor("input_ids", "I32", [n], "<i4")
        labels = tensor("label_ids", "I8", [n], "i1")
        require(message_ids.tolist() == [r["message_id"] for r in rows], "Message IDs are misaligned")
        expected_offsets = [0]
        for row in rows:
            expected_offsets.append(expected_offsets[-1] + row["token_count"])
        require(offsets.tolist() == expected_offsets and expected_offsets[-1] == n, "Message offsets are misaligned")
        for i, row in enumerate(rows):
            begin, end = expected_offsets[i:i + 2]
            require(ids[begin:end].tolist() == row["input_ids"], "Input token alignment failed")
            a, b = row["content_start"], row["content_end"]
            require(0 <= a <= b <= end - begin, "Invalid content span")
            actual = labels[begin:end]
            require(np.all(actual[:a] == -1) and np.all(actual[a:b] == row["label_id"]) and
                    np.all(actual[b:] == -1), "Token labels are misaligned")
        for layer in layers:
            info = header[f"pre_mlp.layer_{layer:02d}"]
            require(info["dtype"] == "BF16" and info["shape"] == [n, width], "Activation shape/dtype mismatch")
            sample_bytes = min(16, n) * width * 2
            bits = np.frombuffer(read_exact(fd, sample_bytes, start + info["data_offsets"][0]), dtype="<u2")
            # BF16 has an eight-bit exponent at bits 7..14; all ones means Inf/NaN.
            require(np.all((bits & 0x7F80) != 0x7F80), "Non-finite saved activations")
    return dict(shard_id=shard["shard_id"], tokens=n, bytes=size)


def verify_all(output, workers=8):
    import collect as spec
    output = Path(output)
    meta, rows = spec.read_plan(output)
    (output / "complete.json").unlink(missing_ok=True)
    missing, jobs = [], []
    for shard in meta["shards"]:
        path = output / "shards" / f"shard-{shard['shard_id']:05d}.safetensors"
        if path.exists():
            jobs.append((path, shard))
        else:
            missing.append(shard["shard_id"])
    total_bytes, total_tokens, done = 0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(verify_shard, path, meta, shard,
                   rows[shard["message_start"]:shard["message_end"]]): shard["shard_id"] for path, shard in jobs}
        for future in as_completed(futures):
            result = future.result()
            total_bytes += result["bytes"]
            total_tokens += result["tokens"]
            done += 1
            if done % 25 == 0 or done == len(jobs):
                progress = dict(checked_shards=done, planned_shards=len(meta["shards"]), checked_tokens=total_tokens)
                spec.atomic_json(output / "verification-progress.json", progress)
                spec.emit("verification_progress", **progress)
    result = dict(complete=not missing, signature=meta["signature"], missing_shards=missing,
                  saved_shards=done, planned_shards=len(meta["shards"]), saved_tokens=total_tokens,
                  bytes=total_bytes, activation_shape=[total_tokens, len(meta["layers"]), meta["hidden_size"]],
                  dtype="bfloat16", token_index_verified=True, saved_layer_samples_finite=True,
                  sample_tokens_per_layer_per_shard=16, reader="bounded POSIX pread; 8-byte safetensors header; BF16 exponent-bit finite check")
    spec.atomic_json(output / "verification.json", result)
    if not missing:
        require(total_tokens == meta["tokens"], "Total token count mismatch")
        spec.atomic_json(output / "complete.json", result)
    spec.emit("verification", **result)
    require(not missing, "Extraction is incomplete")
    return result


if __name__ == "__main__":
    import collect as spec
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=spec.DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    verify_all(args.output, args.workers)
