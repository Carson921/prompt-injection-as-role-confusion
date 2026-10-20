"""Freeze a validation-only primary before preparing the 464-source confirmation.

Nothing runs by default. Declare the comparison before training results exist;
select only after both trials finish. Confirmation inputs must remain unopened.
"""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import systemness_attention as attention
import systemness_data as data
import systemness_long_data as long_data
import systemness_long_probes as dense

OUT=data.runtime.OUT/'systemness-final-readout'
TRIALS={'dense':long_data.OUT,'paired':data.runtime.OUT/'systemness-paired-pilot'}
SOURCES={'dense':'systemness_long_probes.py','paired':'systemness_paired_probes.py'}
DEPENDENCIES=['systemness_final_select.py','systemness_long_probes.py',
    'systemness_paired_probes.py','systemness_probes.py','systemness_long_confirm.py']
require=long_data.require


def read(path):
    return json.loads(Path(path).read_text())


def no_confirmation():
    folder=long_data.OUT/'confirmation'
    require(not any((folder/name).exists() for name in
        ['preparation-lock.json','data-complete.json','inference-complete.json','complete.json']),
        'Confirmation preparation has started; final selection cannot be changed.')


def specification():
    protocol=read(long_data.OUT/'confirmation-plan.json')
    require(protocol['protocol_version']==2 and
        protocol['final_selection']=='systemness-final-readout/selection.json',
        'The pre-confirmation protocol must declare this final selector.')
    return dict(version=1,trial_selection_paths={k:str(v/'selection.json') for k,v in TRIALS.items()},
        choice='Keep dense selection unless paired passes its matched-control promotion and all final gates.',
        gates=dict(paired_trial_promoted=True,mean_f1_gain_strictly_positive=True,
            maximum_original_profile_f1_loss=.02,long_full_f1_loss=0.,long_full_tail_f1_loss=0.,
            maximum_each_profile_fpr=.05,numeric_tolerance_for_inclusive_loss_and_fpr=1e-12,
            original_mean_accuracy_floor='Published attention mean accuracy minus .01'),
        dependencies_sha256={name:data.digest(data.HERE/name) for name in DEPENDENCIES},
        source_sha256=data.digest(__file__),rows_sha256=data.digest(dense.BANK/'rows.parquet'),
        long_data_sha256=data.digest(long_data.OUT/'data-complete.json'),
        published_attention_selection_sha256=data.digest(attention.OUT/'selection.json'),
        standardization_sha256=data.digest(attention.OUT/'standardization.npz'),
        confirmation_plan_sha256=data.digest(long_data.OUT/'confirmation-plan.json'),
        tests_read=False)


def declare():
    no_confirmation();value=specification();path=OUT/'plan.json'
    if path.exists():
        require({k:v for k,v in read(path).items() if k!='declared_unix_time'}==value,
            'Declared final selection dependencies or rules changed.')
    else:
        # This is called before the dense and paired trial selections exist.
        require(all(not (folder/'selection.json').exists() for folder in TRIALS.values()),
            'Declare final comparison before either trial selection exists.')
        OUT.mkdir(parents=True,exist_ok=True)
        data.atomic_json(path,value|dict(declared_unix_time=time.time()))


def verified_trial(name,valid):
    folder=TRIALS[name];selection=read(folder/'selection.json');plan=read(folder/'search-plan.json')
    module=importlib.import_module(Path(SOURCES[name]).stem)
    module.verify_plan(plan)
    require(selection['test_used_for_selection'] is False and type(selection['promoted']) is bool,
        f'{name} selection was not frozen from validation only.')
    require(selection['source_sha256']==plan['source_sha256']==data.digest(data.HERE/SOURCES[name]),
        f'{name} source changed.')
    require(selection['search_plan_sha256']==data.digest(folder/'search-plan.json'),f'{name} plan changed.')
    for source,sha in plan['dependencies_sha256'].items():
        require(data.digest(data.HERE/source)==sha,f'{name} dependency changed: {source}.')
    require(plan['extraction_sha256']==data.digest(long_data.OUT/'extraction-complete.json'),
        f'{name} used a different feature extraction.')
    for key,sha in selection['candidates'].items():
        require(data.digest(folder/'candidates'/f'{key}.json')==sha,f'{name} candidate changed: {key}.')
    key=selection['candidate'];candidate=module.candidate_record(key,plan)
    model=folder/'models'/f'{key}.npz'
    require(Path(selection['model_path'])==model and data.digest(model)==selection['model_sha256']==candidate['model_sha256'],
        f'{name} selected model changed.')
    require(candidate['source_sha256']==plan['source_sha256'] and candidate['tests_read'] is False
        and candidate['portable_probability_max_difference']<5e-4,f'{name} selected export is not verified.')
    default=dense.BANK/f'{key}-validation.npz' if name=='dense' else data.CACHE/'paired-pilot'/f'{key}-validation.npz'
    probability=Path(candidate.get('validation_probabilities_path',str(default)))
    require(data.digest(probability)==candidate['validation_probabilities_sha256'],f'{name} validation probabilities changed.')
    with np.load(probability,allow_pickle=False) as archive:
        p=archive['p'].copy()
        require(np.array_equal(archive['sample_ix'],valid.sample_ix),f'{name} validation rows differ.')
    require(p.shape==(len(valid),5) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        and np.allclose(p.sum(1),1,atol=2e-5,rtol=0),f'{name} probabilities invalid.')
    metric=dense.validation_metrics(valid,p)
    require(metric==candidate['validation']==selection['validation'],f'{name} validation metrics do not recompute.')
    require(selection['system_threshold']==metric['system_threshold'],f'{name} threshold differs.')
    with np.load(model,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        require(metadata['roles']==data.ROLES and metadata['feature_spec']==attention.SPECS,
            f'{name} model has a different role or feature order.')
    return selection


def choose(base,pair,accuracy_floor):
    a,b=base['validation'],pair['validation']
    gains=dict(mean_f1=b['mean_system_f1']-a['mean_system_f1'],
        full_f1=b['profiles']['long-full']['system_f1']-a['profiles']['long-full']['system_f1'],
        tail_f1=b['long_full_tail']['system_f1']-a['long_full_tail']['system_f1'],
        maximum_original_profile_loss=max(a['profiles'][k]['system_f1']-b['profiles'][k]['system_f1']
            for k in dense.ORIGINAL_PROFILES))
    gates=dict(paired_trial_promoted=pair['promoted'],mean_f1_gain=gains['mean_f1']>0,
        full_f1_non_decreasing=gains['full_f1']>=0,tail_f1_non_decreasing=gains['tail_f1']>=0,
        original_profiles_preserved=gains['maximum_original_profile_loss']<=.02+1e-12,
        accuracy_guard=b['original_mean_accuracy']>=accuracy_floor,
        fpr_guard=all(v['system_false_positive_rate']<=.05+1e-12 for v in b['profiles'].values()))
    return ('paired' if all(gates.values()) else 'dense'),gates,gains


def select():
    no_confirmation();declare()
    rows=pd.read_parquet(dense.BANK/'rows.parquet')
    valid=rows[rows.split=='validation'].reset_index(drop=True)
    require(len(valid)==173032 and set(valid.profile)==set(dense.PROFILES),'Validation rows changed.')
    trials={name:verified_trial(name,valid) for name in TRIALS}
    floor=read(attention.OUT/'selection.json')['validation']['mean_accuracy']-.01
    name,gates,gains=choose(trials['dense'],trials['paired'],floor);chosen=trials[name]
    result={key:chosen[key] for key in
        ['candidate','model_path','model_sha256','system_threshold','validation','promoted']}
    result.update(originating_trial=name,test_used_for_selection=False,confirmation_required=True,
        comparison_gates=gates,paired_gains_vs_dense=gains,accuracy_floor=floor,
        trial_selection_sha256={k:data.digest(v/'selection.json') for k,v in TRIALS.items()},
        upstream_artifacts={k:dict(selection_sha256=data.digest(folder/'selection.json'),
            search_plan_sha256=data.digest(folder/'search-plan.json'),
            model_sha256=trials[k]['model_sha256'],
            validation_probabilities_sha256=read(folder/'candidates'/f'{trials[k]["candidate"]}.json')['validation_probabilities_sha256'])
            for k,folder in TRIALS.items()},
        validation_sample_ix_sha256=hashlib.sha256(np.ascontiguousarray(valid.sample_ix.to_numpy()).tobytes()).hexdigest(),
        source_sha256=data.digest(__file__),final_plan_sha256=data.digest(OUT/'plan.json'))
    path=OUT/'selection.json'
    if path.exists():require(read(path)==result,'Final selection already differs; refusing to overwrite.')
    else:data.atomic_json(path,result)
    lines=['Validation-only final primary selection',f'Origin: {name}; candidate: {chosen["candidate"]}',
        f'Originating-trial promoted: {chosen["promoted"]}',
        'No confirmation inputs prepared or results read. Original global validation threshold retained.',
        'Paired requires its own matched-control promotion, greater mean F1, non-decreasing full/tail F1,',
        'at most .02 F1 loss in each original profile, accuracy floor, and every-profile FPR <= .05.',
        json.dumps(dict(gates=gates,gains=gains),indent=2)]
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    print((OUT/'report.txt').read_text(),flush=True)


def preflight():
    metric=dict(mean_system_f1=.8,original_mean_accuracy=.9,
        profiles={k:dict(system_f1=.8,system_false_positive_rate=.01) for k in dense.PROFILES},
        long_full_tail=dict(system_f1=.7))
    base=dict(validation=metric,promoted=False);pair=copy.deepcopy(base);pair['promoted']=True
    pair['validation']['mean_system_f1']=.82
    require(choose(base,pair,.88)[0]=='paired','Passing fixture failed.')
    boundary=copy.deepcopy(pair);boundary['validation']['profiles']['long-single']['system_f1']=.78
    require(choose(base,boundary,.88)[0]=='paired','Inclusive two-percentage-point boundary failed.')
    for field in ['promotion','mean','full','tail','profile','accuracy','fpr']:
        bad=copy.deepcopy(pair);v=bad['validation']
        if field=='promotion':bad['promoted']=False
        elif field=='mean':v['mean_system_f1']=.8
        elif field=='full':v['profiles']['long-full']['system_f1']=.79
        elif field=='tail':v['long_full_tail']['system_f1']=.69
        elif field=='profile':v['profiles']['long-single']['system_f1']=.77
        elif field=='accuracy':v['original_mean_accuracy']=.87
        else:v['profiles']['single']['system_false_positive_rate']=.051
        require(choose(base,bad,.88)[0]=='dense',f'Guard fixture failed: {field}.')
    print(json.dumps(dict(cpu_only=True,gate_fixtures=9,formal_output_written=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--declare',action='store_true');action.add_argument('--select',action='store_true')
    action.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.declare:declare()
    elif args.select:select()
    elif args.preflight:preflight()
    else:parser.print_help()
