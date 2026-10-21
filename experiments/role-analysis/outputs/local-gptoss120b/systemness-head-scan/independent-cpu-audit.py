"""Reproducible read-only CPU audit of the frozen scanner source.

Run from /root/prompt-injection-as-role-confusion with .venv/bin/python -B,
HF_HOME=/root/.cache/huggingface-role-probe and
ROLE_ANALYSIS_LOAD_PATH=/dev/shm/role-analysis-gptoss120b. No model is loaded;
only existing fit/validation caches are read. No formal result is written.
The captured successful output is independent-cpu-audit.log/json. The JSON
records when these artifacts were saved from the original successful audit.
"""
import sys
import json

sys.path.insert(0, 'experiments/role-analysis')
import numpy as np
import pandas as pd
import systemness_head_scan as h
from sklearn.metrics import roc_auc_score

assert h.data.digest(h.__file__) == '6095e77d8ecf9cc22fb51169bdd126ec1f69efe9d3b59575b9e4b9094ba29ecd'
h.require(h.data.digest(h.long_probes.BANK / 'rows.parquet') ==
          h.read(h.long_data.OUT / 'extraction-complete.json')['rows_sha256'], 'Bank manifest hash mismatch')
tok = h.long_data.load_tokenizer()
bundles = h.scan_data(tok)
fit, disc = bundles['fit'], bundles['discovery']
fit_ids, disc_ids = set(fit.rows.document_id), set(disc.rows.document_id)
assert len(fit_ids) == 101 and len(disc_ids) == 31 and not fit_ids & disc_ids
keys = h.read(h.long_data.OUT / 'document-keys.json')
assert fit_ids | disc_ids == {k['document_id'] for k in keys if k['split'] == 'fit'}
assert not (fit_ids | disc_ids) & {k['document_id'] for k in keys if k['split'] == 'validation'}
for bundle in bundles.values():
    for batch in bundle.batches():
        packed = bundle.collate(batch)
        actual = packed['input_ids'].reshape(-1)[packed['positions']].numpy()
        expected = bundle.rows.iloc[packed['targets']].token_id.to_numpy()
        assert np.array_equal(actual, expected)
    for count in [0, 1, 16, len(bundle.batches())]:
        if count > len(bundle.batches()):
            continue
        meta = h.checkpoint_meta(bundle, count, 'fixture', plan_sha256='audit')
        prefix_uids = [uid for part in bundle.batches()[:count] for uid in part.uid]
        expected = bundle.rows[bundle.rows.uid.isin(prefix_uids)]
        assert meta['rows_written'] == len(expected)
        assert meta['class_counts'] == np.bincount(h.probes.labels(expected), minlength=5).tolist()
val = h.validation_data(tok)
original = pd.read_parquet(h.long_probes.BANK / 'rows.parquet',
                          filters=[('split', '=', 'validation')]).reset_index(drop=True)
assert len(val.rows) == 173032 and val.rows.sample_ix.tolist() == original.sample_ix.tolist()
assert val.rows.cache_ix.tolist() == original.cache_ix.tolist()
assert set(val.rows.profile) == set(h.long_probes.PROFILES)
for part in val.batches():
    packed = val.collate(part)
    assert np.array_equal(packed['input_ids'].reshape(-1)[packed['positions']].numpy(),
                          val.rows.iloc[packed['targets']].token_id.to_numpy())
rng = np.random.default_rng(122)
d = disc.rows.copy()
p = rng.random((len(d), h.TOTAL_HEADS)).astype(np.float32)
order, bydoc, all_mean, tail_mean, score = h.document_auc(d, p)
for head in [0, 64, 2289]:
    a, t = [], []
    for doc in bydoc['document_ids']:
        mask = d.document_id.to_numpy() == doc
        for target, sub in [(a, mask), (t, mask & (d.body_offset.to_numpy() >= 512))]:
            target.append(roc_auc_score(d.role.to_numpy()[sub] == 'system', p[sub, head]))
    assert abs(all_mean[head] - np.mean(a)) < 1e-12
    assert abs(tail_mean[head] - np.mean(t)) < 1e-12
assert np.array_equal(order, np.lexsort((np.arange(h.TOTAL_HEADS), -tail_mean, -score)))
print(json.dumps(dict(source_sha256=h.data.digest(h.__file__), fit_docs=len(fit_ids),
    discovery_docs=len(disc_ids), refit_union_docs=len(fit_ids | disc_ids), validation_rows=len(val.rows),
    validation_profiles=val.rows.groupby('profile').size().to_dict(),
    validation_sources=val.rows.groupby('source_set').size().to_dict(), validation_order_preserved=True,
    all_bundle_padding_tokens_exact=True, checkpoint_prefixes_exact=True, auc_sklearn_reference_exact=True,
    model_loaded=False, gpu_run=False, formal_output_written=False), indent=2))
