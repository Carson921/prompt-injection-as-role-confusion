"""Independent CPU/source audit. No archive, model, GPU or formal writes."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, '/root/prompt-injection-as-role-confusion/experiments/role-analysis')
import numpy as np
import pandas as pd
import torch
import systemness_head_confirm as h

torch.set_num_threads(2)
source_sha = h.data.digest(h.__file__)
checks = []
inventory = h.exclusion_inputs()
assert str(h.legacy.OUT / 'document-keys.json') in inventory
assert str(h.data.OUT / 'document-keys.json') in inventory
assert str(h.long_data.OUT / 'document-keys.json') in inventory
assert not any(h.OUT in Path(path).parents for path in inventory)
assert not any('natural-source-keys' in Path(path).name for path in inventory)
manifest_counts = {}
hashes, urls, lines = set(), set(), set()
for path in inventory:
    if Path(path).name == 'document-keys.json':
        records = h.read_json(path)
        a,b,c = h.key_fingerprints(records)
        hashes |= a; urls |= b; lines |= c
        manifest_counts[path] = len(records)
assert manifest_counts[str(h.legacy.OUT / 'document-keys.json')] == 464
checks.append('actual_prior_C4_manifest_inventory_includes_latest464_with_valid_hash_line_identity')

with tempfile.TemporaryDirectory(prefix='head-confirm-exclusions-independent-', dir='/tmp') as temp:
    temp = Path(temp)
    original = h.data.runtime.OUT, h.data.OUT, h.long_data.OUT, h.legacy.OUT, h.OUT
    try:
        h.data.runtime.OUT = temp / 'runtime'
        h.data.OUT = h.data.runtime.OUT / 'systemness-expansion'
        h.long_data.OUT = h.data.runtime.OUT / 'systemness-long-full'
        h.legacy.OUT = h.long_data.OUT / 'confirmation'
        h.OUT = h.data.runtime.OUT / 'systemness-head-confirmation'
        required = [h.data.OUT / 'document-keys.json', h.long_data.OUT / 'document-keys.json',
            h.data.runtime.OUT / 'feature-search/fresh/document-keys.json',
            h.data.runtime.OUT / 'training_inputs.parquet',
            h.data.runtime.OUT.parent / 'local-gptoss20b/raw_training_data.json',
            h.legacy.OUT / 'document-keys.json', h.legacy.OUT / 'data-complete.json', h.legacy.OUT / 'complete.json',
            h.data.runtime.OUT / 'prepared-unreported/nested/document-keys.json',
            h.OUT / 'document-keys.json', h.OUT / 'nested/document-keys.json',
            h.data.runtime.OUT / 'systemness-natural-sources/natural-source-keys.json']
        for path in required:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text('[]')
        observed = h.exclusion_inputs()
        assert str(h.data.runtime.OUT / 'prepared-unreported/nested/document-keys.json') in observed
        assert str(h.legacy.OUT / 'document-keys.json') in observed
        assert not any(h.OUT in Path(path).parents for path in observed)
        assert not any('natural-source-keys' in Path(path).name for path in observed)
        checks.append('recursive_prepared_keys_included_selfOUT_and_distinct_natural_manifest_omitted')
    finally:
        h.data.runtime.OUT, h.data.OUT, h.long_data.OUT, h.legacy.OUT, h.OUT = original

selected = np.asarray([3*64+7, 1, 2*64+3, 64+7, 0, 2*64+7, 64, 3*64], dtype=np.int64)
positions = np.asarray([0,3,7], dtype=np.int64)
rng = np.random.default_rng(917)
raw = {layer: torch.from_numpy(rng.normal(size=(2,4,4096)).astype(np.float32) + 10*layer)
       for layer in sorted(set(selected // 64))}
legacy_raw = {name: torch.from_numpy(rng.normal(size=(2,4,2880)).astype(np.float32)) for name in h.SPECS}
capture = h.Capture(selected); capture.begin(positions, False)
for layer, value in raw.items():
    saved = value.clone(); assert capture.head_hook(layer)(None,(value,)) is None
    assert torch.equal(saved,value)
for name,value in legacy_raw.items(): capture.legacy(name,value)
capture.finish(); assert not capture.values and capture.heads is None
capture.begin(positions, True)
for layer,value in raw.items(): capture.head_hook(layer)(None,(value,))
for name,value in legacy_raw.items(): capture.legacy(name,value)
capture.finish()
expected = np.concatenate([raw[int(head//64)].reshape(-1,64,64)[positions,int(head%64)].half().numpy()
                           for head in selected],axis=1)
assert np.array_equal(capture.heads, expected)
for name,value in legacy_raw.items():
    assert torch.equal(capture.values[name], value.reshape(-1,2880)[positions].half())
checks.append('layer_distinct_unsorted_eight_heads_and_six_legacy_points_sample_same_current_tokens')

docrows = []
counts = {}
for model_ix,name in enumerate(h.NAMES):
    counts[name] = []
    for doc in range(3):
        row = dict(tp=5+model_ix+doc, fp=1+doc, fn=7-model_ix, tn=19+doc,
            correct=22+model_ix+doc, tokens=32+3*doc)
        counts[name].append([row[k] for k in ['tp','fp','fn','tn','correct','tokens']])
        docrows.append(dict(model=name,profile='single',group_id=f'doc-{doc}',**row))
comparisons,intervals = h.bootstrap(pd.DataFrame(docrows),expected_counts=False)
assert len(comparisons)==36 and len(intervals)==24
ix=np.random.RandomState(h.data.SEED+11464).randint(3,size=(2000,3))
def f1(array):
    return 2*array[...,0]/(2*array[...,0]+array[...,1]+array[...,2])
before=np.asarray(counts['dense128'],dtype=float)
after=np.asarray(counts['head-selected'],dtype=float)
draw=(f1(after[ix].sum(1))-f1(before[ix].sum(1)))*100
reference=np.quantile(draw,[.025,.975])
entry=next(r for r in comparisons if r['baseline_model']=='dense128' and r['comparison_model']=='head-selected' and r['metric']=='system_f1')
assert np.allclose(reference,[entry['ci95_low'],entry['ci95_high']],atol=1e-12,rtol=0)
assert abs(entry['gain_percentage_points']-100*(f1(after.sum(0))-f1(before.sum(0))))<1e-12
checks.append('nonzero_four_model_six_pair_bootstrap_matches_independent_shared_doc_draws')

assert h.data.digest(h.__file__)==source_sha
report=dict(source_sha256=source_sha,source_sha256_at_import=source_sha,
    source_sha256_at_completion=h.data.digest(h.__file__),checks=checks,actual_prior_key_manifest_counts=manifest_counts,
    actual_key_hashes=len(hashes),actual_known_urls=len(urls),actual_known_lines=len(lines),
    cpu_only=True,source_archive_read=False,model_loaded=False,gpu_run=False,formal_outputs_written=False,
    existing_or_new_test_probabilities_read=False)
Path('/tmp/feature_review_head_confirm_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
