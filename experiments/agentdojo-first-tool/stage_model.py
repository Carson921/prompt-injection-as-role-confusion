"""Copy the bound native checkpoint to node-local storage, verifying every byte."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('/scratch/wz55191/models/gpt-oss-120b'))
    p.add_argument('--destination', type=Path, default=Path('/tmp/wz55191-agentdojo-first-tool-model'))
    p.add_argument('--selection', type=Path, default=Path(__file__).resolve().parents[1] / 'role-analysis/outputs/synthetic-agent-120b-20261026/final-v7/formal/head-selection.json')
    a = p.parse_args()
    binding = json.loads(a.selection.read_text())['signature']['model_binding']
    expected = binding['assets_sha256'] | binding['weight_shards_sha256']
    a.destination.mkdir(exist_ok=True)

    def copy(item):
        name, digest = item
        src, dst = a.source / name, a.destination / name
        tmp = dst.with_name(dst.name + '.copying')
        h = hashlib.sha256()
        with src.open('rb') as fin, tmp.open('wb') as fout:
            for block in iter(lambda: fin.read(16 << 20), b''):
                h.update(block)
                fout.write(block)
        assert h.hexdigest() == digest, name
        h2 = hashlib.sha256()
        with tmp.open('rb') as stream:
            for block in iter(lambda: stream.read(16 << 20), b''):
                h2.update(block)
        assert h2.hexdigest() == digest, name
        tmp.replace(dst)
        print(json.dumps({'copied': name, 'sha256': digest}), flush=True)
        return {'name': name, 'sha256': digest, 'bytes': dst.stat().st_size}

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(copy, expected.items()))
    proof = {'all_shards_verified': True, 'source': str(a.source), 'pid': os.getpid(),
             'shards': [r for r in rows if r['name'].endswith('.safetensors')], 'files': rows}
    (a.destination / 'staging-proof.json').write_text(json.dumps(proof, indent=2) + '\n')


if __name__ == '__main__':
    main()
