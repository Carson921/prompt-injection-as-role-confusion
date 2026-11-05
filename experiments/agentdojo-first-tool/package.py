"""Package a terminal, fully audited collection with archive byte readback."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('/home/wz55191/Github/delimeter-token/results/ChatInject/agentdojo_first_tool_gptoss120b/full_20261103'))
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    source = a.source.resolve()
    output = a.output or source.with_name(source.name + '.tar.gz')
    assert output.resolve().is_relative_to(Path('/home/wz55191/Github/delimeter-token').resolve())
    assert not output.exists()
    audit = json.loads((source / 'audit.json').read_text())
    assert audit['status'] == 'full_scope_verified' and audit['terminal_cases'] == audit['expected_cases'] == 1898
    assert json.loads((source / 'probe_inputs/manifest.json').read_text())['status'] == 'complete'
    paths = {str(f.relative_to(source)): f for f in sorted(source.rglob('*'))
             if f.is_file() and f.name != 'run.lock' and not f.name.endswith(('.tmp', '.log')) and '__pycache__' not in f.parts}
    hashes = {name: digest(path) for name, path in paths.items()}
    checksums = ''.join(f'{value}  {name}\n' for name, value in sorted(hashes.items())).encode()
    hashes['SHA256SUMS'] = hashlib.sha256(checksums).hexdigest()
    tmp = output.with_name(output.name + '.tmp')
    with tarfile.open(tmp, 'w:gz', compresslevel=1) as tar:
        for name, path in paths.items():
            tar.add(path, arcname=source.name + '/' + name, recursive=False)
        info = tarfile.TarInfo(source.name + '/SHA256SUMS')
        info.size, info.mode = len(checksums), 0o644
        tar.addfile(info, io.BytesIO(checksums))
    seen = set()
    with tarfile.open(tmp, 'r:gz') as tar:
        for member in tar:
            assert member.isfile()
            name = str(Path(member.name).relative_to(source.name))
            assert name in hashes and name not in seen
            h = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    h.update(block)
            assert h.hexdigest() == hashes[name], name
            seen.add(name)
    assert seen == set(hashes)
    tmp.replace(output)
    checksum = digest(output)
    output.with_name(output.name + '.sha256').write_text(f'{checksum}  {output.name}\n')
    receipt = {'status': 'complete', 'archive': str(output), 'archive_sha256': checksum,
        'archive_bytes': output.stat().st_size, 'all_member_bytes_readback_verified': True,
        'files': len(hashes), 'audit_sha256': digest(source / 'audit.json'),
        'expected_cases': audit['expected_cases'], 'captured_cases': audit['captured_cases'],
        'errors': audit['errors'], 'completed_at_unix': time.time()}
    output.with_name(output.name + '.complete.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
