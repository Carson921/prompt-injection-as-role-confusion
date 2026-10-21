"""Frozen-threshold development-validation error overlap; no model selection.

Reads only fit/validation-bank row metadata filtered to validation, two saved
validation probability arrays, and frozen model/provenance metadata. Never reads
old/new test rows, test probabilities, raw source text, or GPU/model activations.
Oracle values use the true labels to choose between models per token: they are
nondeployable upper bounds and do not establish a learnable or valid gate.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['HF_HOME'] = '/root/.cache/huggingface-role-probe'
os.environ['ROLE_ANALYSIS_LOAD_PATH'] = '/dev/shm/role-analysis-gptoss120b'
sys.path.insert(0, '/root/prompt-injection-as-role-confusion/experiments/role-analysis')
import numpy as np
import pandas as pd
import pyarrow as pa
import systemness_data as data
import systemness_long_data as long_data
import systemness_long_probes as long_probes
import systemness_head_scan as scanner
import systemness_probes as probes

pa.set_cpu_count(1); pa.set_io_thread_count(1)
def read(path): return json.loads(Path(path).read_text())
def digest_array(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def ratio(a,b): return float(a/b) if b else None
def intersection(a,b):
    return dict(both_correct=int((a&b).sum()), dense_only_correct=int((a&~b).sum()),
        head_only_correct=int((~a&b).sum()), both_wrong=int((~a&~b).sum()))
def complement(cells):
    dense_errors=cells['head_only_correct']+cells['both_wrong']
    head_errors=cells['dense_only_correct']+cells['both_wrong']
    union=cells['head_only_correct']+cells['dense_only_correct']+cells['both_wrong']
    return dict(dense_errors=dense_errors,head_errors=head_errors,
        head_repairs_fraction_of_dense_errors=ratio(cells['head_only_correct'],dense_errors),
        dense_repairs_fraction_of_head_errors=ratio(cells['dense_only_correct'],head_errors),
        error_set_jaccard=ratio(cells['both_wrong'],union))
def binary_metrics(y,pred):
    tp=int((y&pred).sum()); fp=int((~y&pred).sum()); fn=int((y&~pred).sum()); tn=int((~y&~pred).sum())
    div=lambda a,b:float(a/b) if b else 0.
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=div(tp,tp+fp),recall=div(tp,tp+fn),
        f1=div(2*tp,2*tp+fp+fn),false_positive_rate=div(fp,fp+tn),
        balanced_accuracy=.5*(div(tp,tp+fn)+div(tn,tn+fp)),accuracy=div(tp+tn,len(y)))
def validate_p(p):
    assert p.shape==(173032,5) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    assert np.allclose(p.sum(1),1,atol=2e-5,rtol=0)

dense_candidate_path=long_data.OUT/'candidates/dense-mlp128.json'
dense_candidate=read(dense_candidate_path)
head_metrics_path=scanner.OUT/'validation-metrics.json'
head_metrics=read(head_metrics_path)
head_done=read(scanner.OUT/'validation-complete.json')
head_scan_done=read(scanner.OUT/'scan-complete.json')
bank_rows_path=long_probes.BANK/'rows.parquet'
extraction=read(long_data.OUT/'extraction-complete.json')
dense_p_path=long_probes.BANK/'dense-mlp128-validation.npz'
head_p_path=scanner.CACHE/'validation-probabilities.npz'
paths=dict(bank_rows=bank_rows_path,dense_candidate=dense_candidate_path,
    dense_model=long_data.OUT/'models/dense-mlp128.npz',dense_validation_probabilities=dense_p_path,
    dense_search_plan=long_data.OUT/'search-plan.json',long_extraction=long_data.OUT/'extraction-complete.json',
    head_model=scanner.OUT/'models/top8-lda.npz',head_plan=scanner.OUT/'plan.json',
    head_scan_complete=scanner.OUT/'scan-complete.json',head_selection=scanner.OUT/'head-selection.json',
    head_validation_complete=scanner.OUT/'validation-complete.json',head_validation_metrics=head_metrics_path,
    head_validation_probabilities=head_p_path)
hashes={name:data.digest(path) for name,path in paths.items()}
assert hashes['bank_rows']==extraction['rows_sha256']
assert hashes['dense_model']==dense_candidate['model_sha256']
assert hashes['dense_validation_probabilities']==dense_candidate['validation_probabilities_sha256']
assert hashes['dense_search_plan']==dense_candidate['search_plan_sha256']
assert hashes['head_model']==head_done['model_sha256']==head_scan_done['model_sha256']==head_metrics['model_sha256']
assert hashes['head_plan']==head_done['plan_sha256']==head_scan_done['plan_sha256']==head_metrics['plan_sha256']
assert hashes['head_scan_complete']==head_done['scan_complete_sha256']
assert hashes['head_selection']==head_scan_done['head_selection_sha256']==head_metrics['head_selection_sha256']
assert hashes['head_validation_probabilities']==head_done['probabilities_sha256']==head_metrics['probabilities_sha256']
assert hashes['head_validation_metrics']==head_done['metrics_sha256']
assert head_done['tests_read'] is False and head_scan_done['tests_read'] is False and dense_candidate['tests_read'] is False
assert head_metrics['data']==head_done['data']

rows=pd.read_parquet(bank_rows_path,filters=[('split','=','validation')]).reset_index(drop=True)
assert len(rows)==173032 and set(rows.split)=={'validation'} and set(rows.wrapper)=={'native'}
assert set(rows.profile)==set(long_probes.PROFILES) and rows.sample_ix.is_unique
rows['uid']=rows.source_set+':'+rows.prompt_ix.astype(str)
assert scanner.frame_digest(rows,['sample_ix','uid','token_ix','token_id','role','document_id'])==head_done['data']['rows_sha256']
assert digest_array(rows.sample_ix.to_numpy())==head_done['data']['sample_ix_sha256']
probabilities={}
for name,path in [('dense128',dense_p_path),('head_top8_lda',head_p_path)]:
    with np.load(path,allow_pickle=False) as archive:
        assert np.array_equal(archive['sample_ix'],rows.sample_ix.to_numpy())
        probabilities[name]=archive['p'].copy()
    validate_p(probabilities[name])
thresholds=dict(dense128=float(dense_candidate['validation']['system_threshold']),
    head_top8_lda=float(head_metrics['validation']['system_threshold']))
assert all(np.isfinite(value) for value in thresholds.values())
labels=probes.labels(rows); assert (labels>=0).all()
full=rows[rows.profile.eq('long-full')]
offset=full.token_ix-full.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
tail_ix=full.index[offset.ge(512)].to_numpy()
assert len(tail_ix)==9920 and full.loc[tail_ix].document_id.nunique()==31
groups={profile:np.flatnonzero(rows.profile.to_numpy()==profile) for profile in long_probes.PROFILES}
groups['long-full-tail']=tail_ix
results={}
for profile,ix in groups.items():
    y=labels[ix]; truth=y==0
    d=probabilities['dense128'][ix]; h=probabilities['head_top8_lda'][ix]
    dp=d[:,0]>=thresholds['dense128']; hp=h[:,0]>=thresholds['head_top8_lda']
    dc=dp==truth; hc=hp==truth
    d5=d.argmax(1); h5=h.argmax(1); dc5=d5==y; hc5=h5==y
    cells=intersection(dc,hc); cells5=intersection(dc5,hc5)
    assert sum(cells.values())==sum(cells5.values())==len(ix)
    own={}
    for name,p,record in [('dense128',d,dense_candidate['validation']),('head_top8_lda',h,head_metrics['validation'])]:
        computed=probes.metrics(p,y,thresholds[name])
        frozen=record['long_full_tail'] if profile=='long-full-tail' else record['profiles'][profile]
        for key in ['accuracy','system_threshold','system_precision','system_recall','system_f1',
                    'system_false_positive_rate','system_balanced_accuracy','system_confusion','confusion','recall']:
            assert computed[key]==frozen[key],(profile,name,key)
        own[name]=dict(system=binary_metrics(truth,p[:,0]>=thresholds[name]),
            five_way_accuracy=float((p.argmax(1)==y).mean()))
    oracle_system_pred=np.where(dc,dp,hp)
    oracle_system=binary_metrics(truth,oracle_system_pred)
    assert int((oracle_system_pred==truth).sum())==len(ix)-cells['both_wrong']
    five_oracle_accuracy=float((dc5|hc5).mean())
    results[profile]=dict(tokens=len(ix),source_groups=int(rows.iloc[ix].group_id.nunique()),
        actual_system_tokens=int(truth.sum()),models=own,
        system_binary=dict(correctness_intersection=cells,complementarity=complement(cells),
            true_system=intersection(dc[truth],hc[truth]),true_nonsystem=intersection(dc[~truth],hc[~truth])),
        five_way=dict(correctness_intersection=cells5,complementarity=complement(cells5),
            per_true_role={role:intersection(dc5[y==i],hc5[y==i]) for i,role in enumerate(data.ROLES) if np.any(y==i)}),
        nondeployable_label_aware_oracle_upper_bound=dict(system=oracle_system,
            system_f1_gain_vs_dense_percentage_points=100*(oracle_system['f1']-own['dense128']['system']['f1']),
            five_way_accuracy=five_oracle_accuracy,
            five_way_accuracy_gain_vs_dense_percentage_points=100*(five_oracle_accuracy-own['dense128']['five_way_accuracy'])))
assert all(data.digest(path)==hashes[name] for name,path in paths.items())
report=dict(purpose='Frozen-threshold error complementarity on previously reused development validation only.',
    diagnostic_only=True,not_a_deployable_model=True,not_used_to_reselect_existing_models=True,
    no_threshold_search_or_fitting_performed=True,thresholds=thresholds,validation_rows=173032,
    validation_sample_ix_sha256=digest_array(rows.sample_ix.to_numpy()),
    validation_row_descriptor=head_done['data'],
    inputs_sha256={name:dict(path=str(paths[name]),sha256=sha) for name,sha in hashes.items()},
    source_script_sha256=data.digest(__file__),results=results,
    definitions=dict(system_correct='Frozen threshold prediction agrees with formal system-vs-nonsystem label.',
        five_way_correct='Argmax of the frozen five-class probabilities equals the formal role.',
        head_repairs_fraction_of_dense_errors='head_only_correct / (head_only_correct + both_wrong).',
        error_set_jaccard='both_wrong / (dense_only_correct + head_only_correct + both_wrong).',
        oracle='Uses true labels per token to select a correct prediction when either model is correct; binary oracle uses dense when dense is correct, otherwise head. Five-way oracle reports accuracy only.'),
    limitations=['Label-aware oracle is an unattainable deployment claim without a separately learned and independently validated gate.',
        'Complementarity does not establish that a bounded pointwise system-logit delta can identify the repairable cases.',
        'Dense128 and scanner LDA differ in training distribution, feature geometry and classifier family; this is not a matched causal ablation.',
        'Validation has already been reused for development. No old/new test, test model analysis, generation or GPU was used.'],
    gpu_run=False,test_rows_read=False,test_probabilities_read=False,formal_results_written=False)
Path('/tmp/feature_review_validation_complementarity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(output='/tmp/feature_review_validation_complementarity.json',thresholds=thresholds,
    summary={profile:dict(system_cells=r['system_binary']['correctness_intersection'],
        system_head_repairs_dense_errors=r['system_binary']['complementarity']['head_repairs_fraction_of_dense_errors'],
        dense_f1=r['models']['dense128']['system']['f1'],head_f1=r['models']['head_top8_lda']['system']['f1'],
        oracle_system_f1=r['nondeployable_label_aware_oracle_upper_bound']['system']['f1'],
        five_way_cells=r['five_way']['correctness_intersection'],
        oracle_5way_accuracy=r['nondeployable_label_aware_oracle_upper_bound']['five_way_accuracy']) for profile,r in results.items()}),indent=2))
