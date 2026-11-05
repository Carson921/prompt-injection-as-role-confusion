"""Copy checkpoint shards to local storage, recording source and copy hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(32 << 20):
            h.update(data)
    return h.hexdigest()


def stage(source, target, workers):
    target.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json",
                 "generation_config.json", "special_tokens_map.json", "chat_template.jinja"]:
        if (source / name).exists():
            shutil.copyfile(source / name, target / name)
    names = sorted(set(json.loads((source / "model.safetensors.index.json").read_text())["weight_map"].values()))
    started = time.monotonic()
    def copy(name):
        begin = time.monotonic()
        h = hashlib.sha256()
        temporary = target / (name + ".tmp")
        with (source / name).open("rb") as r, temporary.open("wb") as w:
            while data := r.read(32 << 20):
                h.update(data)
                w.write(data)
            w.flush()
            os.fsync(w.fileno())
        if digest(temporary) != h.hexdigest():
            raise RuntimeError(f"Staged shard differs from source: {name}")
        temporary.replace(target / name)
        result = dict(name=name, bytes=(target / name).stat().st_size, sha256=h.hexdigest(),
                      source_and_copy_sha256_equal=True, seconds=time.monotonic() - begin)
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        proof = list(pool.map(copy, names))
    record = dict(source=str(source.resolve()), target=str(target.resolve()), shards=proof,
                  all_shards_verified=True, seconds=time.monotonic() - started)
    (target / "staging-proof.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.source.resolve() == args.target.resolve():
        raise ValueError("Source and target must differ")
    stage(args.source, args.target, args.workers)
