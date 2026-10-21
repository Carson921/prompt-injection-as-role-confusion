"""Small independent report interface audit; does not render or write formally.

Actual development validation is used only for row/gallery indexing checks. Its
dense probabilities are deliberately aliased under five names as an interface
fixture, never as a claim about future head candidates. No GPU/test rows.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock

sys.path.insert(0, '/root/prompt-injection-as-role-confusion/experiments/role-analysis')
import numpy as np
import pandas as pd
import pyarrow as pa
import systemness_head_probes_report as r

pa.set_cpu_count(1); pa.set_io_thread_count(1)
source_sha=r.data.digest(r.__file__)
checks=[]
with tempfile.TemporaryDirectory(prefix='head-probes-report-independent-',dir='/tmp') as tmp:
    missing=Path(tmp)/'must-not-be-created'
    def unfinished(): raise RuntimeError('Synthetic unfinished head search')
    with mock.patch.object(r,'OUT',missing), mock.patch.object(r.heads,'verified_completion',side_effect=unfinished), \
         mock.patch.object(r.pd,'read_parquet',side_effect=AssertionError('Rows read before completion gate')):
        try: r.write()
        except RuntimeError as error: assert 'unfinished' in str(error)
        else: raise AssertionError('Unfinished search accepted')
        assert not missing.exists()
    checks.append('unfinished_search_rejected_before_any_output_or_row_read')

rows_path=r.long_probes.BANK/'rows.parquet'
dense_record_path=r.long_data.OUT/'candidates/dense-mlp128.json'
dense_record=r.read(dense_record_path)
rows=pd.read_parquet(rows_path,filters=[('split','=','validation')]).reset_index(drop=True)
assert len(rows)==173032 and rows.index.equals(pd.RangeIndex(len(rows)))
inputs_path=r.data.CACHE/'inputs.parquet'
meta=r.read(r.data.OUT/'data-complete.json')
assert r.data.digest(inputs_path)==meta['cache_sha256']['inputs.parquet']
inputs=pd.read_parquet(inputs_path,filters=[('split','=','validation'),('wrapper','=','native'),('profile','=','single')])
chosen,single=r.gallery_sources(rows,inputs)
ids=sorted(single.document_id.unique().tolist())
expected=[ids[i] for i in np.linspace(0,len(ids)-1,12,dtype=int)]
assert len(ids)==200 and chosen==expected
assert r.gallery_sources(rows,inputs.iloc[::-1])[0]==chosen
bad=rows.copy(); index=bad[bad.profile.eq('single')&bad.role.eq('cot')].index[0]
bad.loc[index,'token_id']+=1
try:r.gallery_sources(bad,inputs)
except RuntimeError:pass
else:raise AssertionError('Mismatched per-role body token accepted')
checks.append('actual200_sources_fixed12_ids_input_order_independent_same64_tokens_per_role')

collision=pd.DataFrame(dict(source_set=['main','main','supplement','supplement'],
    prompt_ix=[7,7,7,7],message_ix=[0,0,0,0],token_ix=[10,13,20,25]))
assert r.body_positions(collision).tolist()==[0,3,0,5]
checks.append('body_positions_group_source_prompt_message_separately')

prob_path=r.long_probes.BANK/'dense-mlp128-validation.npz'
p=r.probability_file(prob_path,rows.sample_ix.to_numpy(),dense_record['validation_probabilities_sha256'])
try:r.probability_file(prob_path,rows.sample_ix.to_numpy()[::-1],dense_record['validation_probabilities_sha256'])
except RuntimeError:pass
else:raise AssertionError('Reordered probability row identity accepted')
checks.append('saved_validation_probability_sample_ix_order_strict')
probabilities={name:p for name in r.NAMES}
thresholds={name:dense_record['validation']['system_threshold'] for name in r.NAMES}
records={name:dict(validation=copy.deepcopy(dense_record['validation'])) for name in r.NAMES}
metrics,positions=r.metrics_at_frozen_thresholds(rows,probabilities,thresholds,records)
assert len(metrics)==30
tail=[entry for entry in metrics if entry['profile']=='long-full-tail']
assert len(tail)==5 and all(entry['tokens']==9920 and entry['source_documents']==31 for entry in tail)
assert all(entry['threshold']==dense_record['validation']['system_threshold'] for entry in metrics)
assert all(entry['system_f1']==dense_record['validation']['long_full_tail']['system_f1'] for entry in tail)
assert len(positions)==30 and all(entry['source_documents']==31 for entry in positions)
checks.append('actual173032_global_index_tail9920_all5_alias_consumers_match_frozen_dense_metrics')
for value,label in [(thresholds['dense128']+.0001,'reject_supplied_threshold_drift_before_metric_scoring'),
                    (float('nan'),'reject_nonfinite_threshold_before_metric_scoring')]:
    changed=dict(thresholds,dense128=value)
    try:r.metrics_at_frozen_thresholds(rows,probabilities,changed,records)
    except RuntimeError as error:assert 'threshold differs' in str(error)
    else:raise AssertionError(label)
    checks.append(label)
assert r.data.digest(r.__file__)==source_sha
report=dict(source_sha256_at_import=source_sha,source_sha256_at_completion=r.data.digest(r.__file__),
    checks=checks,actual_validation_rows=173032,actual_gallery_source_documents=200,
    fixed_gallery_source_ids=[int(x) for x in chosen],actual_sampled_tail_tokens=9920,
    actual_tail_source_documents=31,
    fixture_note='Dense validation probabilities aliased under five model names only to test indexing/threshold interfaces, not future head performance.',
    inputs_sha256={str(path):r.data.digest(path) for path in [rows_path,inputs_path,dense_record_path,prob_path]},
    complete_plot_fixture_not_repeated=True,formal_outputs_written=False,gpu_run=False,test_rows_read=False)
Path('/tmp/feature_review_head_probes_report_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
