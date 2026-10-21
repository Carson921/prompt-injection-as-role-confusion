#!/usr/bin/env python3
"""Small read-only interpretation of audited frozen DEVELOPMENT tables. No probability/model opens."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path('/root/prompt-injection-as-role-confusion')
OUT=ROOT/'experiments/role-analysis/outputs/local-gptoss120b/systemness-long-mix-residual'
AUDIT=OUT/'actual-audit/actual-development-audit.json'
AUDIT_SHA='6dadc3a34ff9ea2c5c3415930051876165b529f3983d77eed757456259a1e58e'
DEST=Path('/tmp/feature_review_long_mix_residual_interpretation.json')

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())

assert digest(AUDIT)==AUDIT_SHA
audit=read(AUDIT)
assert audit['stage']=='actual_completed_long_mix_residual_development_audit' and audit['audit_passed'] is True
assert audit['test_data_or_results_read'] is False and audit['gpu_used'] is False
paths=[OUT/p for p in ['complete.json','protocol.json','selection.json','records.json','replication.json','validation-metrics.csv']]
paths += [OUT/folder/(arm+'.json') for folder in ['training','candidates'] for arm in ['new','matched']]
provenance={str(AUDIT):AUDIT_SHA,str(Path(__file__).resolve()):digest(__file__)}
for path in paths:
    assert digest(path)==audit['provenance_sha256'][str(path.resolve())]
    provenance[str(path.resolve())]=digest(path)
index={r['record']:r['validation'] for r in read(OUT/'records.json')}
selection=read(OUT/'selection.json');replication=read(OUT/'replication.json')
assert selection['candidate']=='new-max33' and selection['promoted'] is False and selection['primary_changed'] is False
assert replication['all_weight_arrays_bitexact'] is True and replication['all_probability_arrays_bitexact'] is True
with (OUT/'validation-metrics.csv').open(newline='') as f:csv_rows=list(csv.DictReader(f))
assert len(csv_rows)==144

keys=['system_precision','system_recall','system_f1','system_balanced_accuracy','accuracy']
groups=['old-natural','old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']
chosen=['new-max33','matched-max33','priorBinary-max33']
summary={}
for name in chosen:
    v=index[name];summary[name]=dict(system_threshold=v['system_threshold'],min33=v['min33'],old5_mean_f1=v['old5_mean_f1'],groups={})
    for g in groups:
        m=v['groups'][g];entry=dict(tokens=m['tokens'],**{k:m[k] for k in keys})
        if g=='old-natural':
            c=m['confusion'];entry.update(cot_support=sum(c[2]),cot_correct=c[2][2],cot_recall=m['recall']['cot'],
                cot_to_system=c[2][0],cot_to_assistant=c[2][3],system_confusion=m['system_confusion'],tool_truth_support=sum(c[4]))
        summary[name]['groups'][g]=entry
assert index['matched-max33']==index['priorBinary-max33']
delta={g:{k:100*(index['new-max33']['groups'][g][k]-index['matched-max33']['groups'][g][k]) for k in keys} for g in groups}
delta['old-natural']['cot_recall']=100*(index['new-max33']['groups']['old-natural']['recall']['cot']-index['matched-max33']['groups']['old-natural']['recall']['cot'])
training={};raw={}
for arm in ['new','matched']:
    t=read(OUT/'training'/(arm+'.json'));c=read(OUT/'candidates'/(arm+'.json'))
    training[arm]={k:t[k] for k in ['epochs','best_epoch','best_steps','actual_steps','actual_instances']}
    raw[arm]={}
    for bank,proof in c['raw_invariance'].items():
        assert proof['raw_non_system_sha256']==proof['base_raw_non_system_sha256']
        assert proof['raw_non_system_argmax_changes']==proof['non_system_to_different_non_system_p_argmax']==0
        raw[arm][bank]={k:proof[k] for k in ['tokens','raw_non_system_sha256','base_raw_non_system_sha256',
            'raw_non_system_argmax_changes','non_system_to_different_non_system_p_argmax',
            'non_system_to_system_p_argmax','system_to_non_system_p_argmax']}
assert training['new']['epochs']==26 and training['new']['best_epoch']==18 and training['new']['best_steps']==2826 and training['new']['actual_steps']==4082
assert training['matched']['epochs']==11 and training['matched']['best_epoch']==3 and training['matched']['best_steps']==471 and training['matched']['actual_steps']==1727

report=dict(stage='read_only_audited_long_mix_development_interpretation',independent_new_metrics_audit=False,
    actual_audit_sha256=AUDIT_SHA,provenance_sha256=provenance,source_sha256=digest(__file__),
    probabilities_or_weights_read=False,test_data_or_results_read=False,gpu_used=False,formal_outputs_modified=False,
    promoted=False,primary_changed=False,rejection_reasons=selection['rejection_reasons'],
    selected_frozen_records=summary,new_minus_matched_percentage_points=delta,
    min33_change_new_minus_matched=index['new-max33']['min33']-index['matched-max33']['min33'],
    exact_training_budgets=training,per_epoch_instances=641493,per_epoch_updates=157,
    same_per_epoch_budget=True,same_actual_total_updates=False,
    matched_reproduces_prior_binary_NEW=dict(all_weight_arrays_bitexact=True,all_three_probability_banks_bitexact=True,
        max33_threshold_and_metrics_equal=True),raw_non_system_and_system_transition_evidence=raw,
    interpretation=[
        'The constant-long-weight replacement improves new-C4 tail precision and F1 at the frozen selected operating point, but recall falls. Its minimum33 is worse than the matched/prior binary baseline and below90%.',
        'Legacy natural-system recall remains100%; its system precision falls from96.2209% to88.8591%, causing F1 to fall from98.0741% to94.1009%. This exceeds the unchanged2pp F1 floor. The negative result must be retained.',
        'CoT recall is79.4609% versus matched79.7205%. Four non-system raw logits and their conditional ordering stay fixed, and observed different-non-system winner swaps remain0; this still allows system/non-system winner transitions and changes in truth-CoT errors.',
        'The matched arm reproduces all prior binary-NEW parameter arrays and three-bank probabilities bit-for-bit. Its operating point and metrics agree; reference recomputation drift is not the observed effect.',
        'Only per-epoch instances, role/half labels and update budget are matched. New trains26epochs/best18/4082actual updates, matched11/best3/1727actual updates; do not call actual total updates matched.',
        'Document breadth, per-document density/body grid, style and realized early stopping vary together. These reused-development observations do not isolate a causal source-count effect and do not establish independent generalization.'
    ],
    next_step='Keep the failed trial and current primary unchanged. No fresh-test confirmation for this candidate; any future fit-only feature/context experiment needs its own prior declaration and unchanged guards.')
assert all(digest(path)==value for path,value in provenance.items())
DEST.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(report=str(DEST),report_sha256=digest(DEST),source_sha256=digest(__file__),
    new_min33=index['new-max33']['min33'],matched_min33=index['matched-max33']['min33'],
    new_minus_matched_percentage_points=delta,training=training)))
