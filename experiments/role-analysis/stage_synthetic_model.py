"""Copy the user's mounted checkpoint to local SSD and verify official hashes.

The mounted checkpoint remains the durable source; this cache is reconstructible.
The expected manifest comes from the project's earlier official-weight audit,
never from a trained probe or its feature ranking.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            value.update(block)
    return value.hexdigest()


def stage(source, target, expected, workers=4, download=False):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(expected).read_text())
    started = time.time()

    def copy(row):
        destination = target / row['file']
        if destination.exists() and destination.stat().st_size == row['size'] and sha256(destination) == row['sha256']:
            return dict(row, reused=True)
        temporary = destination.with_suffix(destination.suffix + '.partial')
        h = hashlib.sha256()
        if download:
            url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{row['file']}"
            src = urlopen(Request(url, headers={'User-Agent': 'six-role-retraining/1'}), timeout=120)
        else:
            src = (source / row['file']).open('rb')
        with src, temporary.open('wb') as dst:
            for block in iter(lambda: src.read(1 << 20), b''):
                h.update(block)
                dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        if temporary.stat().st_size != row['size'] or h.hexdigest() != row['sha256']:
            raise RuntimeError('Official checkpoint checksum mismatch: ' + row['file'])
        temporary.replace(destination)
        print(json.dumps({'stage': 'weight_verified', 'file': row['file'], 'seconds': time.time()-started}), flush=True)
        return dict(row, reused=False)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        checked = list(pool.map(copy, manifest['shards']))
    for asset in source.iterdir():
        if asset.is_file() and asset.suffix in {'.json', '.jinja'}:
            shutil.copy2(asset, target / asset.name)
    result = dict(repository=manifest['repository'], revision=manifest['revision'],
                  all_15_shards_sha256_verified=True, shards=checked,
                  durable_source=str(source), execution_cache=str(target),
                  seconds=time.time()-started,
                  note='Local SSD cache can be rebuilt from the durable source after host restart.')
    temporary = target / 'weight-verification.json.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(target / 'weight-verification.json')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--expected', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--download', action='store_true', help='Fetch the same official revision over HTTPS instead of the mounted source')
    args = p.parse_args()
    stage(args.source, args.target, args.expected, args.workers, args.download)
