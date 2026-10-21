#!/usr/bin/env python3
"""Exact F32 threshold-event DEVELOPMENT diagnostic. No model/selection writes, tests, LM or GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

ROOT = Path('/root/prompt-injection-as-role-confusion')
HERE = ROOT / 'experiments/role-analysis'
OUT = HERE / 'outputs/local-gptoss120b/systemness-binary-residual'
CACHE = HERE / 'outputs/local-gptoss120b/systemness-expansion/cache/binary-residual'
AUDIT = Path('/tmp/long_data_binary_residual_results_audit.json')
AUDIT_SHA = '2600d086c711e65bb320b70e66539fa775164cba8c192b84a2f9c52a75a9aa67'
SOURCE_SHA = 'fd49e6c8f199b9163d7ca2a5b7c4c66f2d0092b270d66a037394b0b2511cf9e9'
ROLES = ['system', 'user', 'cot', 'assistant', 'tool']
OLD5 = ['old-' + p for p in ['single', 'multi', 'natural', 'long-single', 'long-full']]
PUBLIC = ['public-natural-all', 'public-natural-oasst', 'public-natural-toxicchat0124']
FULL = ['old-long-full', 'old-long-full-tail', 'new-C4-long-full', 'new-C4-long-full-tail']
GROUPS = OLD5 + ['old-long-full-tail', 'new-C4-long-full', 'new-C4-long-full-tail'] + PUBLIC
FEASIBLE = OLD5 + ['new-C4-long-full'] + PUBLIC
GUARDS = ['matched-max33', 'oldhead-max33', 'ensemble-max33', 'oldhead-oldmean',
          'ensemble-original', 'zero-oldmean', 'zero-max33']
STRICT = ['matched-max33', 'oldhead-max33', 'ensemble-max33', 'zero-max33']
ROWS = {'old': Path('/dev/shm/role-analysis-top8-heads/rows.parquet'),
        'c4': Path('/dev/shm/role-analysis-long-augmentation-heads/rows.parquet'),
        'public': Path('/dev/shm/role-analysis-natural-dev-heads/rows.parquet')}
N = {'old': 173032, 'c4': 61440, 'public': 16713}
TOL = 1e-12

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def require(ok, why):
    if not bool(ok):
        raise RuntimeError(why)

def events(scores):
    values = np.unique(np.concatenate([np.asarray(s, np.float32) for s in scores]))
    require(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all(), 'Invalid system scores.')
    grid = np.unique(np.r_[np.float32(0), values,
                           np.nextafter(values, np.float32(np.inf)), np.float32(1.000001)]).astype(np.float32)
    # Each possible >= classification is represented: include a tie at its score,
    # and exclude that complete tie immediately afterwards, plus both endpoints.
    return grid, values

def curve(score, truth, grid):
    score = np.asarray(score, np.float32); truth = np.asarray(truth, bool)
    require(grid.dtype == score.dtype == np.float32, 'Consumer comparisons must use effective F32 thresholds.')
    pos, neg = np.sort(score[truth]), np.sort(score[~truth])
    require(len(pos) and len(neg), 'Every group must contain both system and non-system truth.')
    tp = len(pos) - np.searchsorted(pos, grid, side='left')
    fp = len(neg) - np.searchsorted(neg, grid, side='left')
    recall = tp / len(pos)
    return dict(tp=tp, fp=fp,
                P=np.divide(tp, tp + fp, out=np.full(len(grid), np.nan), where=(tp + fp) > 0),
                R=recall, BAcc=(recall + (len(neg) - fp) / len(neg)) / 2,
                F1=2 * tp / (2 * tp + fp + len(pos) - tp), FPR=fp / len(neg))

def direct(score, truth, tau):
    detect = np.asarray(score, np.float32) >= float(tau); truth = np.asarray(truth, bool)
    a = int((detect & truth).sum()); b = int((detect & ~truth).sum())
    p = int(truth.sum()); n = len(truth) - p
    return dict(tp=a, fp=b, P=a / (a+b) if a+b else math.nan, R=a/p,
                BAcc=(a/p + (n-b)/n)/2, F1=2*a/(2*a+b+p-a), FPR=b/n)

def best(mask, minimum, mean, grid):
    indices = np.flatnonzero(mask & np.isfinite(minimum))
    if not len(indices): return None
    indices = indices[minimum[indices] == minimum[indices].max()]
    indices = indices[mean[indices] == mean[indices].max()]
    return int(indices[np.argmin(grid[indices])])

def group_indices(rows):
    old, c4, pub = rows['old'], rows['c4'], rows['public']
    result = {name: ('old', np.flatnonzero(old.profile.eq(profile)))
              for name, profile in zip(OLD5, ['single','multi','natural','long-single','long-full'])}
    for bank, f, name in [('old', old, FULL[1]), ('c4', c4, FULL[3])]:
        full = f[f.profile.eq('long-full')]
        offset = full.body_offset if 'body_offset' in full else full.token_ix - full.groupby(['source_set','prompt_ix','message_ix']).token_ix.transform('min')
        require(np.isfinite(offset.to_numpy()).all(), 'Undefined full-body offsets.')
        result[name] = bank, full.index.to_numpy()[offset.to_numpy() >= 512]
    result[FULL[2]] = 'c4', np.flatnonzero(c4.profile.eq('long-full'))
    result[PUBLIC[0]] = 'public', np.arange(len(pub))
    for name, ds in zip(PUBLIC[1:], ['oasst','toxicchat0124']):
        result[name] = 'public', np.flatnonzero(pub.dataset.eq(ds))
    require(set(result) == set(GROUPS) and all(len(ix) for _, ix in result.values()), 'Missing group.')
    return result

def gate():
    # No probability or row arrays may be read before the approved terminal certificate.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
        try: args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (OSError, ProcessLookupError): continue
        require(not any(Path(a.decode(errors='replace')).name in
                        {'systemness_binary_residual.py','long_data_binary_residual_results_audit.py'} for a in args if a),
                'Trainer or independent actual auditor is live.')
    require(sha(AUDIT) == AUDIT_SHA, 'Approved actual audit changed.')
    audit = read(AUDIT)
    require(audit['stage'] == 'actual_completed_binary_residual_development_audit'
            and audit['audit_passed'] is True and audit['test_data_or_results_read'] is False,
            'Passing approved development audit required.')
    require(read(OUT / 'status.json')['stage'] == 'validation_complete'
            and sha(OUT / 'complete.json') == audit['trainer_complete_sha256']
            and sha(OUT / 'protocol.json') == audit['trainer_protocol_sha256']
            and sha(HERE / 'systemness_binary_residual.py') == audit['trainer_source_sha256'] == SOURCE_SHA,
            'Frozen producer no longer matches its terminal audit.')
    return audit

def preflight():
    """Hand-computable tied F32 events, then exhaustive toy event/direct parity."""
    checks=[]
    a=np.float32(.5); b=np.nextafter(a,np.float32(1))
    score=np.array([0,a,a,b,1,1],np.float32); truth=np.array([0,1,0,1,1,0],bool)
    grid, _=events([score[:2],score[2:4],score[4:]])
    c=curve(score,truth,grid)
    hand={np.float32(0):(3,3),a:(3,2),b:(2,1),np.nextafter(b,np.float32(np.inf)):(1,1),np.nextafter(np.float32(1),np.float32(np.inf)):(0,0)}
    for tau, pair in hand.items():
        i=int(np.flatnonzero(grid==tau)[0]);require((int(c['tp'][i]),int(c['fp'][i]))==pair,'Hand toy tie boundary failed.')
    checks.append('Hand tied scores including nextafter(.5),0,1 give exact five TP/FP partitions')
    for i,tau in enumerate(grid):
        d=direct(score,truth,tau)
        for k in d:require(np.array_equal(np.asarray(c[k][i]),np.asarray(d[k]),equal_nan=True),'Toy direct F32 mismatch: '+k)
    checks.append('All toy event counts/P/R/BAcc/F1/FPR equal direct >= consumer including undefined precision')
    require(np.isnan(c['P'][-1]),'No-prediction precision must remain undefined.')
    checks.append('Undefined precision cannot masquerade as a finite maximin candidate')
    # Coarse all-float probability grids have exactly the same distinct decision masks
    # as explicit event points; ties cannot be split by any threshold.
    candidates=np.unique(np.r_[grid,np.linspace(0,1,1001).astype(np.float32)])
    patterns={tuple((score>=float(t)).tolist()) for t in grid}
    require(all(tuple((score>=float(t)).tolist()) in patterns for t in candidates),'Missing decision partition.')
    checks.append('Event set covers all toy binary decision partitions without splitting ties')
    print(json.dumps(dict(stage='synthetic_exact_F32_threshold_event_preflight',checks_passed=len(checks),checks=checks,
                          source_sha256=sha(__file__),actual_probabilities_read=False,test_data_read=False,gpu_used=False)))
    return checks

def actual(checks):
    audit=gate(); bindings={str(AUDIT):AUDIT_SHA,str(Path(__file__).resolve()):sha(__file__)}
    def bind(path,expected):
        path=Path(path);require(sha(path)==expected,'Bound input changed: '+str(path));bindings[str(path.resolve())]=expected
    for name in ['records.json','selection.json','controls.json','zero-controls.json','complete.json','protocol.json']:
        path=OUT/name;bind(path,audit['provenance_sha256'][str(path.resolve())])
    rows={};p={}
    for bank,path in ROWS.items():
        bind(path,audit['provenance_sha256'][str(path.resolve())])
        rows[bank]=pd.read_parquet(path);rows[bank]=rows[bank][rows[bank].split.eq('validation')].reset_index(drop=True)
        require(len(rows[bank])==N[bank] and rows[bank].sample_ix.is_unique and rows[bank].role.isin(ROLES).all(),'Invalid development row ownership.')
        path=CACHE/('new-'+bank+'-validation.npz');bind(path,audit['probability_bindings'][str(path.resolve())])
        with np.load(path,allow_pickle=False) as z:
            require(set(z.files)=={'p','sample_ix'} and np.array_equal(z['sample_ix'],rows[bank].sample_ix.to_numpy()),'Saved sample order differs.')
            p[bank]=z['p'].copy()
        require(p[bank].shape==(N[bank],5) and p[bank].dtype==np.float32 and np.isfinite(p[bank]).all()
                and (p[bank]>=0).all() and (p[bank]<=1).all() and np.allclose(p[bank].sum(1),1,atol=2e-5,rtol=0),'Invalid audited probability.')
    records={r['record']:r['validation'] for r in read(OUT/'records.json')}
    controls={name:records[name] for name in GUARDS if name in records}
    controls.update(read(OUT/'zero-controls.json')['records'])
    controls['original-savedGPU']=read(OUT/'controls.json')['original_savedGPU_validation']
    mapping=group_indices(rows); truth={b:rows[b].role.eq('system').to_numpy() for b in rows}
    grid,unique=events([p[b][:,0] for b in ['old','c4','public']]); cur={};support={};accuracy={}
    for g,(b,ix) in mapping.items():
        cur[g]=curve(p[b][ix,0],truth[b][ix],grid)
        y=rows[b].iloc[ix].role.map({r:i for i,r in enumerate(ROLES)}).to_numpy()
        accuracy[g]=float((p[b][ix].argmax(1)==y).mean())
        require(accuracy[g]==records['new-max33']['groups'][g]['accuracy'],'Fixed p.argmax accuracy no longer reproduces.')
        support[g]=dict(tokens=len(ix),system=int(truth[b][ix].sum()),non_system=int((~truth[b][ix]).sum()))
    minimum=np.min(np.stack([cur[g][k] for g in GROUPS for k in ['P','R','BAcc']]),axis=0)
    minimum[~np.isfinite(minimum)]=-np.inf
    mean=np.mean(np.stack([cur[g]['F1'] for g in OLD5]),axis=0)
    primary_fpr=np.logical_and.reduce([cur[g]['FPR']<=.05 for g in FEASIBLE])
    primary_defined=np.isfinite(minimum)
    masks={};constant_failures={}
    for name,control in controls.items():
        legacy=name=='original-savedGPU'; pieces={};fixed=[]
        for g in OLD5[:4]:
            pieces['old4_F1/'+g]=cur[g]['F1']>=control['groups'][g]['system_f1']-.02-TOL
            if control['groups'][g]['accuracy']-accuracy[g]>.01+TOL:fixed.append('old4_five_accuracy/'+g)
        pieces['old5_mean_F1']=mean>=control['old5_mean_f1']-.005-TOL
        for g in FULL:pieces['full_tail_F1/'+g]=cur[g]['F1']>=control['groups'][g]['system_f1']-TOL
        if not legacy:
            for g in PUBLIC:
                pieces['public_F1/'+g]=cur[g]['F1']>=control['groups'][g]['system_f1']-.02-TOL
                if control['groups'][g]['accuracy']-accuracy[g]>.01+TOL:fixed.append('public_five_accuracy/'+g)
        pieces['FPR']=np.logical_and.reduce([cur[g]['FPR']<=.05+TOL for g in FEASIBLE])
        pieces['finite_F1_FPR']=np.logical_and.reduce([np.isfinite(cur[g]['F1'])&np.isfinite(cur[g]['FPR']) for g in GROUPS])
        masks[name]=np.logical_and.reduce(list(pieces.values())) & (not fixed)
        constant_failures[name]=fixed
    all_guards=np.logical_and.reduce(list(masks.values()))
    strict=np.logical_and.reduce([minimum-controls[n]['min33']>TOL for n in STRICT])
    target=minimum>=.90
    permitted=primary_defined & primary_fpr & all_guards & strict & target
    def point(i):
        if i is None:return None
        group_values={}
        # Independently verify actual consumer counts at each reported real point.
        for g,(b,ix) in mapping.items():
            direct_values=direct(p[b][ix,0],truth[b][ix],grid[i])
            require(all(np.array_equal(np.asarray(cur[g][k][i]),np.asarray(v),equal_nan=True) for k,v in direct_values.items()),'Reported real point direct-count mismatch.')
            group_values[g]={k:(None if not np.isfinite(v) else int(v) if k in ['tp','fp'] else float(v)) for k,v in direct_values.items()}
            group_values[g]['five_accuracy_fixed']=accuracy[g]
        return dict(effective_F32_threshold=float(grid[i]),threshold_F32_bits=int(grid[i].view(np.uint32)),
                    min33=float(minimum[i]),old5_mean_F1=float(mean[i]),groups=group_values,
                    original_primary_FPR_feasible=bool(primary_fpr[i]),absolute90_pass=bool(target[i]),
                    seven_plus_savedGPU_guards_pass=bool(all_guards[i]),four_strict_controls_pass=bool(strict[i]),
                    strict_min33_changes={n:float(minimum[i]-controls[n]['min33']) for n in STRICT},
                    failing_guard_controls=[n for n in controls if not masks[n][i]])
    best_any=best(primary_defined,minimum,mean,grid)
    best_fpr=best(primary_defined & primary_fpr,minimum,mean,grid)
    best_guard=best(primary_defined & primary_fpr & all_guards,minimum,mean,grid)
    best_guard_strict=best(primary_defined & primary_fpr & all_guards & strict,minimum,mean,grid)
    selected_tau=np.float32(records['new-max33']['system_threshold'])
    # A frozen-grid threshold may fall strictly between score events. Re-evaluate it
    # separately; it is diagnostic input only and never replaced.
    saved_groups={g:direct(p[b][ix,0],truth[b][ix],selected_tau) for g,(b,ix) in mapping.items()}
    frozen_min=min(saved_groups[g][k] for g in GROUPS for k in ['P','R','BAcc'])
    require(frozen_min==records['new-max33']['min33'],'Frozen current threshold does not reproduce.')
    # Broad per-stat exact intervals provide a small, concrete P/R conflict certificate.
    boundaries={}
    for g in GROUPS:
        for k in ['P','R','BAcc']:
            valid=np.flatnonzero(np.isfinite(cur[g][k]) & (cur[g][k]>=.90))
            boundaries[g+'/'+k]=dict(feasible_event_count=int(len(valid)),first_threshold=float(grid[valid[0]]) if len(valid) else None,
                                     last_threshold=float(grid[valid[-1]]) if len(valid) else None,
                                     feasible_sets_may_have_gaps=(k=='P'))
    f1_floor={g:max(control['groups'][g]['system_f1'] for control in controls.values()) for g in FULL}
    counts=dict(total_event_thresholds=len(grid),distinct_underlying_scores=len(unique),defined_min33=int(primary_defined.sum()),
                original_FPR_feasible=int((primary_defined&primary_fpr).sum()),absolute90=int((primary_defined&target).sum()),
                absolute90_and_FPR=int((primary_defined&target&primary_fpr).sum()),all_guards=int((primary_defined&primary_fpr&all_guards).sum()),
                all_guards_and_strict=int((primary_defined&primary_fpr&all_guards&strict).sum()),
                absolute90_and_all_guards=int((primary_defined&primary_fpr&target&all_guards).sum()),
                all_requirements=int(permitted.sum()))
    lowerbound=max(control['min33'] for n,control in controls.items() if n in STRICT)
    report=dict(stage='exact_all_F32_score_event_threshold_development_diagnostic',diagnostic_only=True,
                model_or_threshold_or_selection_changed=False,candidate_promoted=False,new_independent_confirmation_required=True,
                test_data_or_results_read=False,gpu_used=False,model_or_LM_loaded=False,formal_outputs_modified=False,
                audit_sha256=AUDIT_SHA,source_sha256=sha(__file__),provenance_sha256=bindings,
                threshold_event_definition='Each distinct F32 score from the three underlying banks exactly once, its F32 nextafter(+inf),0 and all-negative1.000001 sentinel; >= uses actual F32 consumer semantics. Overlapping groups are not duplicated in the score pool.',
                undefined_precision='Undefined P excludes a threshold from finite min33; never replace with zero.',
                exact_objective='Max min33, tie higher old5 mean F1, then smaller effective F32 threshold; all five-way p.argmax statistics remain fixed.',
                groups=GROUPS,group_support=support,threshold_counts=counts,
                exact_unconstrained_max=point(best_any),exact_original_FPR_feasible_max=point(best_fpr),
                exact_all_guards_max=point(best_guard),exact_all_guards_plus_four_strict_max=point(best_guard_strict),
                exists_threshold_meeting_90_all_original_guards_and_four_strict=bool(permitted.any()),
                diagnostic_feasible_witness=point(best(permitted,minimum,mean,grid)),
                frozen_grid_result=dict(effective_threshold=float(selected_tau),min33=float(frozen_min)),
                event_search_min33_change_from_frozen_grid=float(minimum[best_fpr]-frozen_min),
                fixed_five_way_guard_failures=constant_failures,
                per_control_guard_feasible_events={n:int((primary_defined&primary_fpr&mask).sum()) for n,mask in masks.items()},
                strict_min33_must_exceed=lowerbound,strict_improvement_tolerance=TOL,
                full_tail_F1_must_reach=f1_floor,at_least90_statistic_event_boundaries=boundaries,
                toy_checks=checks,
                limitation='Exact only for thresholds applied to these fixed saved probabilities and frozen controls on reused development data. It is not a new promoted candidate, not independent-test evidence, and not an upper bound on other models/features/contextual readouts.')
    require(all(sha(path)==want for path,want in bindings.items()),'Inputs/source changed during diagnosis.')
    dest=Path('/tmp/feature_review_binary_exact_threshold_diagnosis.json')
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(dest),report_sha256=sha(dest),source_sha256=sha(__file__),counts=counts,
                         best_FPR_min33=float(minimum[best_fpr]),best_FPR_tau=float(grid[best_fpr]),
                         best_allguards_min33=None if best_guard is None else float(minimum[best_guard]),
                         all_requirements_exist=bool(permitted.any()))))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--preflight',action='store_true');group.add_argument('--actual',action='store_true');args=parser.parse_args()
    if args.preflight:preflight()
    elif args.actual:actual(preflight())
    else:parser.print_help()
