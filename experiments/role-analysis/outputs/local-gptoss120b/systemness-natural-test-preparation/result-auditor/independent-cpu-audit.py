"""Synthetic CPU-only review of future natural-result auditor; no actual mode."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score

SOURCE=Path('/tmp/confirmation_review_natural_result_audit.py')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
before=sha(SOURCE)
spec=importlib.util.spec_from_file_location('natural_result_auditor_review',SOURCE)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
checks=[]
assert 'systemness_natural_test_confirm' not in sys.modules
checks.append('import_has_no_test_runner_or_source_materialization')
sys.path.insert(0,'/root/prompt-injection-as-role-confusion/experiments/role-analysis')
import systemness_natural_test_confirm as runner
rows,p,thresholds,ledger=a.synthetic()
records,counts,messages,macro=a.independent_summary(rows,p,thresholds)
expected,expected_counts,expected_messages=runner.summarize(rows,p,thresholds)
metric_error=a.compare_records(records,expected,['stratum','dataset','model'],'synthetic metric parity')
a.compare_frame(counts,expected_counts,['stratum','dataset','model','source_uid'],'synthetic source counts',0)
message_error=a.compare_frame(messages,expected_messages,['stratum','model','source_uid','message_ix','role','family_id','dependency_component_id'],'synthetic message p',2e-6)
assert len(records)==24
checks.append('24metrics_all_clean_SUA_NA_source_and_message_schema_match_frozen_runner')
c,i,components=a.independent_bootstrap(counts);ec,ei,ecomponents=runner.bootstrap(expected_counts)
pair_error=a.compare_records(c,ec,['stratum','dataset','baseline_model','comparison_model','metric'],'all9metrics sixpairs component paired2000')
interval_error=a.compare_records(i,ei,['stratum','dataset','model','metric'],'all9metrics model intervals')
a.compare_frame(components,ecomponents,['stratum','dataset','model','dependency_component_id'],'synthetic component counts',0)
assert len(c)==6*6*9 and len(i)==6*4*9
checks.append('all9metrics_fourmodels_sixpairs_2000_component_draws_exact')
v=a.metrics(p[a.NAMES[0]],a.row_labels(rows),.5);y=a.row_labels(rows);pred=p[a.NAMES[0]].argmax(1)
assert v['confusion']==confusion_matrix(y,pred,labels=np.arange(5)).tolist()
assert abs(v['macro_f1']-f1_score(y,pred,labels=np.unique(y),average='macro'))<1e-15
assert abs(v['macro_f1']-f1_score(y,pred,average='macro'))>.05
assert v['system_recall']!=v['recall']['system'] and v['recall']['cot'] is None and v['recall']['tool'] is None
checks.append('truth_supported_macro_and_threshold_vs_argmax_S_recall_definitions_preserved')
one=counts.copy();one['dependency_component_id']='same-component';c1,i1,_=a.independent_bootstrap(one)
assert all(not r['ci_available'] and r['ci95_low'] is None and r['ci95_high'] is None and r['ci_unavailable_reason']=='fewer_than_two_dependency_components' for r in c1+i1)
checks.append('single_dependency_component_withholds_confidence_intervals')
never=counts.copy();mask=never.model.eq(a.NAMES[0]);never.loc[mask,'fn']+=never.loc[mask,'tp'];never.loc[mask,'tn']+=never.loc[mask,'fp'];never.loc[mask,['tp','fp']]=0
c1,i1,_=a.independent_bootstrap(never)
assert all(r['value'] is None and r['defined_bootstrap_replicates']==0 for r in i1 if r['model']==a.NAMES[0] and r['metric']=='system_precision')
assert all(r['gain_percentage_points'] is None and r['defined_bootstrap_replicates']==0 for r in c1 if r['baseline_model']==a.NAMES[0] and r['metric']=='system_precision')
checks.append('undefined_precision_and_paired_deltas_remain_NA_with_explicit_defined_draw_count')
empty=rows.iloc[:0].copy();zero={name:np.empty((0,5),np.float32) for name in a.NAMES}
obs,empty_counts,_,_=a.independent_summary(empty,zero,thresholds)
assert len(obs)==8 and all(r['tokens']==0 and r['accuracy'] is None for r in obs)
assert not a.independent_bootstrap(empty_counts)[0]
checks.append('allfailed_empty_schema_has_no_invented_observations_or_bootstrap')
folder=Path(tempfile.mkdtemp(prefix='feature-review-natural-auditor-gate-',dir='/tmp'))
calls=[]
def sentinel(path):
    if Path(path).resolve()==SOURCE.resolve():return sha(SOURCE)
    calls.append(str(path));raise RuntimeError('first-builder-manifest-after-terminal')
fake_runner=SimpleNamespace(OUT=folder/'runner');fake_runner.OUT.mkdir()
fake_builder=SimpleNamespace(OUT=folder/'builder')
with patch.dict(sys.modules,{'systemness_natural_test_confirm':fake_runner,'systemness_natural_test':fake_builder}),patch.object(a,'digest',sentinel):
    for state in ['missing-complete','nonterminal']:
        if state=='nonterminal':
            (fake_runner.OUT/'complete.json').write_text('{}');(fake_runner.OUT/'status.json').write_text('{"stage":"inference"}')
        try:a.actual(folder/'never.json')
        except RuntimeError:pass
        else:raise AssertionError('Partial terminal state accepted')
        assert not calls and not (folder/'never.json').exists()
    (fake_runner.OUT/'status.json').write_text('{"stage":"complete"}')
    try:a.actual(folder/'never.json')
    except RuntimeError as error:assert str(error)=='first-builder-manifest-after-terminal'
    else:raise AssertionError('Manifest sentinel did not run after terminal')
    assert calls==[str(fake_builder.OUT/'data-complete.json')]
checks.append('actual_complete_and_status_gates_precede_builder_manifest_and_any_probability_access')
assert before==sha(SOURCE)
result=dict(source_sha256_at_start=before,source_sha256_at_end=sha(SOURCE),checks=checks,checks_passed=len(checks),
    maximum_metric_difference=metric_error,maximum_message_difference=message_error,maximum_paired_difference=pair_error,
    maximum_interval_difference=interval_error,test_data_or_probabilities_read=False,c4_partial_artifacts_read=False,
    actual_result_audit_run=False,gpu_run=False,formal_files_modified=False)
out=Path('/tmp/feature_review_natural_result_auditor_audit.json');out.write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(dict(report=str(out),report_sha256=sha(out),**result),indent=2))
