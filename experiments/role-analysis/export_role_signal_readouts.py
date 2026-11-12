"""Stream saved factorial body-token readouts to a portable Parquet table.

This exporter reads inputs.jsonl and readouts/<input_id>.npz only. It neither
loads a model/probe nor reconstructs logits or probabilities. Coordinates are
zero-based, body_end is exclusive, and excluded=False identifies the complete
predefined body region; wrapper/filler/ending tokens never enter the table.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROLES = ('system', 'developer', 'user', 'cot', 'assistant', 'tool')
EXPECTED_SPLITS = {'pilot': 120, 'formal': 1200}
IDENTIFIERS = ('input_id', 'pair_id', 'split', 'filler_id', 'text_style', 'wrapper')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'{path}:{number}: invalid JSON') from error


def inspect_manifest(path):
    ids, pair_ids = set(), {'pilot': set(), 'formal': set()}
    splits, token_counts = Counter(), Counter()
    shortest, longest = None, None
    for row in jsonlines(path):
        name = row.get('input_id')
        require(isinstance(name, str) and name and Path(name).name == name,
                f'Invalid input_id: {name!r}')
        require(name not in ids, 'Duplicate manifested input_id: ' + name)
        require(row.get('split') in EXPECTED_SPLITS, 'Unexpected split: ' + name)
        require(all(isinstance(row.get(key), str) for key in IDENTIFIERS),
                'Missing string identity fields: ' + name)
        require(row.get('text_style') in ('styled', 'destyled'), 'Unexpected text style: ' + name)
        require(row.get('wrapper') in ('cot', 'user'), 'Unexpected wrapper: ' + name)
        require(row.get('filler_id') in ('A', 'B'), 'Unexpected filler: ' + name)
        require(row.get('position') in (32, 256, 768), 'Unexpected position: ' + name)
        start, end = row.get('body_start'), row.get('body_end')
        require(isinstance(start, int) and isinstance(end, int) and 0 <= start < end,
                'Invalid body coordinates: ' + name)
        require(start == row['position'], 'Body start differs from target position: ' + name)
        require(isinstance(row.get('input_ids'), list) and end <= len(row['input_ids']),
                'Body extends beyond input: ' + name)
        body = row.get('body_token_ids')
        require(isinstance(body, list) and len(body) == end-start == row.get('body_tokens'),
                'Body token count mismatch: ' + name)
        require(body == row['input_ids'][start:end], 'Body token IDs differ from full input: ' + name)
        require(all(isinstance(token, int) and token >= 0 for token in body),
                'Invalid body token IDs: ' + name)
        require(row.get('excluded_boundary_tokens', 0) == 0,
                'This schema requires a complete body mask without excluded boundary tokens: ' + name)
        ids.add(name)
        splits[row['split']] += 1
        token_counts[row['split']] += len(body)
        pair_ids[row['split']].add(row['pair_id'])
        shortest = len(body) if shortest is None else min(shortest, len(body))
        longest = len(body) if longest is None else max(longest, len(body))
    require(dict(splits) == EXPECTED_SPLITS, f'Expected {EXPECTED_SPLITS}; observed {dict(splits)}')
    require(len(pair_ids['pilot']) == 10 and len(pair_ids['formal']) == 50,
            'Expected 10 pilot and 50 formal semantic units')
    require(not pair_ids['pilot'] & pair_ids['formal'], 'Pilot/formal pair IDs overlap')
    return ids, dict(splits), dict(token_counts), shortest, longest


def table_schema():
    fields = [pa.field(key, pa.string(), nullable=False) for key in IDENTIFIERS]
    fields += [pa.field('position', pa.int32(), nullable=False),
               pa.field('token_position', pa.int64(), nullable=False),
               pa.field('body_token_offset', pa.int32(), nullable=False),
               pa.field('token_id', pa.int64(), nullable=False),
               pa.field('excluded', pa.bool_(), nullable=False)]
    fields += [pa.field(prefix + role, pa.float32(), nullable=False)
               for prefix in ('z_', 'p_') for role in ROLES]
    return pa.schema(fields, metadata={
        b'format': b'role-signal-body-token-readouts-v1',
        b'roles': json.dumps(ROLES).encode(),
        b'coordinates': b'zero-based token_position; zero-based body_token_offset; body_end exclusive',
        b'values': b'saved float32 logits and probabilities copied directly; no recomputation',
        b'scope': b'1320 manifested factorial inputs only: 120 pilot and 1200 formal; body tokens only',
    })


def input_table(row, archive, schema):
    name = row['input_id']
    count = row['body_tokens']
    with np.load(archive, allow_pickle=False) as arrays:
        require({'positions', 'roles', 'logits', 'probabilities'} <= set(arrays.files),
                'Missing readout arrays: ' + name)
        roles = arrays['roles']
        require(roles.ndim == 1 and tuple(roles.tolist()) == ROLES,
                'Readout role order differs: ' + name)
        positions = arrays['positions']
        require(positions.dtype == np.int64 and positions.shape == (count,),
                'Readout position dtype/shape differs: ' + name)
        require(np.array_equal(positions, np.arange(row['body_start'], row['body_end'], dtype=np.int64)),
                'Readout positions differ from manifested body: ' + name)
        logits, probs = arrays['logits'], arrays['probabilities']
        for label, values in [('logits', logits), ('probabilities', probs)]:
            require(values.dtype == np.float32 and values.shape == (count, len(ROLES)),
                    f'Readout {label} must be float32 with shape ({count},6): {name}')
            require(np.isfinite(values).all(), f'Nonfinite {label}: {name}')
        require(np.all((probs >= 0) & (probs <= 1)), 'Saved probabilities out of range: ' + name)
        require(np.allclose(probs.sum(axis=1), 1, rtol=0, atol=1e-6),
                'Saved probability row sums differ from one: ' + name)
        columns = {key: pa.array([row[key]] * count, type=pa.string()) for key in IDENTIFIERS}
        columns.update(position=pa.array(np.full(count, row['position'], dtype=np.int32)),
                       token_position=pa.array(positions, type=pa.int64()),
                       body_token_offset=pa.array(np.arange(count, dtype=np.int32)),
                       token_id=pa.array(row['body_token_ids'], type=pa.int64()),
                       excluded=pa.array(np.zeros(count, dtype=np.bool_)))
        for prefix, values in [('z_', logits), ('p_', probs)]:
            for column, role in enumerate(ROLES):
                columns[prefix + role] = pa.array(values[:, column], type=pa.float32())
        # This retains only the current input's body arrays. The 1024-dimensional
        # feature array in the NPZ is never accessed or decompressed.
        return pa.Table.from_pydict(columns, schema=schema)


def temporary_path(directory, suffix):
    handle = tempfile.NamedTemporaryFile(prefix='.role-signal-export-', suffix=suffix,
                                         dir=directory, delete=False)
    handle.close()
    return Path(handle.name)


def export(output):
    output = Path(output).resolve()
    started = time.monotonic()
    inputs = output / 'inputs.jsonl'
    require(inputs.is_file(), 'Missing input manifest: ' + str(inputs))
    input_sha = digest(inputs)
    expected_ids, splits, tokens, shortest, longest = inspect_manifest(inputs)
    archives = output / 'readouts'
    actual_ids = {path.stem for path in archives.glob('*.npz')}
    require(actual_ids == expected_ids,
            f'Readout files differ from inputs: missing={sorted(expected_ids-actual_ids)[:8]}, '
            f'extra={sorted(actual_ids-expected_ids)[:8]}')
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        require(protocol.get('inputs_sha256') == input_sha, 'Protocol input SHA256 mismatch')
    schema = table_schema()
    parquet_path = output / 'token-readouts.parquet'
    manifest_path = output / 'export-manifest.json'
    parquet_temp = temporary_path(output, '.parquet')
    manifest_temp = temporary_path(output, '.json')
    readout_hashes, rows_written = {}, 0
    try:
        with pq.ParquetWriter(parquet_temp, schema=schema, compression='zstd',
                              use_dictionary=list(IDENTIFIERS), write_statistics=True) as writer:
            for row in jsonlines(inputs):
                source = archives / (row['input_id'] + '.npz')
                source_sha = digest(source)
                table = input_table(row, source, schema)
                require(digest(source) == source_sha, 'Readout changed during export: ' + row['input_id'])
                writer.write_table(table, row_group_size=table.num_rows)
                rows_written += table.num_rows
                readout_hashes['readouts/' + source.name] = source_sha
        require(digest(inputs) == input_sha, 'Input manifest changed during export')
        require({path.stem for path in archives.glob('*.npz')} == expected_ids,
                'Readout file inventory changed during export')
        metadata = pq.read_metadata(parquet_temp)
        require(metadata.num_rows == rows_written == sum(tokens.values()), 'Parquet row count mismatch')
        require(metadata.num_row_groups == len(expected_ids) == 1320, 'Parquet row-group count mismatch')
        require(pq.read_schema(parquet_temp).equals(schema, check_metadata=True), 'Parquet schema mismatch')
        identities = {'inputs.jsonl': input_sha, 'export_role_signal_readouts.py': digest(__file__)}
        for name in ('protocol.json', 'pairs.jsonl', 'run-manifest.json'):
            path = output / name
            if path.exists():
                identities[name] = digest(path)
        manifest = dict(format_version='role-signal-body-token-export-v1', status='complete',
                        parquet_file=parquet_path.name, parquet_sha256=digest(parquet_temp),
                        inputs=len(expected_ids), input_counts_by_split=splits,
                        token_rows=rows_written, token_rows_by_split=tokens,
                        row_groups=metadata.num_row_groups, source_readout_files=len(readout_hashes),
                        source_file_inventory_exact=True, excluded_rows=0,
                        minimum_body_tokens=shortest, maximum_body_tokens=longest,
                        roles=ROLES, coordinate_system='zero-based absolute token_position and body_token_offset',
                        value_policy='Logits and probabilities copied directly as float32 from saved NPZ arrays; no model/probe loading or probability derivation.',
                        excluded_scopes=['construction-inputs.jsonl', 'normal-checks/'],
                        columns={field.name: str(field.type) for field in schema},
                        source_hashes=identities, readout_hashes=readout_hashes,
                        elapsed_seconds=time.monotonic()-started,
                        software=dict(numpy=np.__version__, pyarrow=pa.__version__))
        manifest_temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                                 encoding='utf-8')
        os.replace(parquet_temp, parquet_path)
        os.replace(manifest_temp, manifest_path)
        print(json.dumps({key: manifest[key] for key in ('status', 'inputs', 'token_rows', 'row_groups', 'parquet_sha256')}))
    finally:
        parquet_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Frozen experiment output directory')
    export(parser.parse_args().output)


if __name__ == '__main__':
    main()
