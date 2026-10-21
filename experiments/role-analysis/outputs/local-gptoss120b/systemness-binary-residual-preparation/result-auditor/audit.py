"""Independent binary-system residual development results auditor.

Default help; --preflight is synthetic CPU/tmp only. --actual requires a pinned
final trainer, declared protocol and terminal completion before loading results.
No production metrics, calibration, promotion guards or completion verifiers are
called. No tests, training, GPU or writes to formal output directories.
"""
import argparse,copy,hashlib,importlib.metadata,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path('/root/prompt-injection-as-role-confusion')
HERE=ROOT/'experiments/role-analysis';BASE=HERE/'outputs/local-gptoss120b'
OUT=BASE/'systemness-binary-residual';CACHE=BASE/'systemness-expansion/cache/binary-residual'
SOURCE=HERE/'systemness_binary_residual.py'
SOURCE_SHA='fd49e6c8f199b9163d7ca2a5b7c4c66f2d0092b270d66a037394b0b2511cf9e9'
PROTOCOL_SHA='84ebfc2dff0138dab829cb7756660b819e4d734823508c6219ec9de6c93cba1a'
DESIGN=BASE/'systemness-binary-residual-preparation/root-prospective-design.json'
DESIGN_SHA='c68a32d55e732f678f77a7ef91d8d966a416f118d43663f913b3c9f2d14e09c5'
CURRENT_SHA='27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
CURRENT_TAU=.620969696969697;ENSEMBLE_TAU=.4747979797979798
ROLES=['system','user','cot','assistant','tool']
PROFILES=['single','multi','natural','long-single','long-full']
OLD5=['old-'+p for p in PROFILES];ORIGINAL=OLD5[:4]
PUBLIC=['public-natural-all','public-natural-oasst','public-natural-toxicchat0124']
FULL=['old-long-full','old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']
GROUPS=OLD5+['old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']+PUBLIC
FEASIBLE=OLD5+['new-C4-long-full']+PUBLIC
RECORDS=['new-oldmean','new-max33','matched-oldmean','matched-max33','oldhead-oldmean','oldhead-max33','ensemble-original','ensemble-max33','naturalCE-oldmean','naturalCE-max33']
GUARD_CONTROLS=['matched-max33','oldhead-max33','ensemble-max33','oldhead-oldmean','ensemble-original','zero-oldmean','zero-max33']
STRICT_CONTROLS=['matched-max33','oldhead-max33','ensemble-max33','zero-max33']
ZERO_RECORDS=['zero-oldmean','zero-max33']
PRIOR_AUDIT_SHA='b8de402c326f052e1ed3e2e90a313fb8f58268c1802c66c1ff3570de40c67617'
MEMBERS=[20261021,20261022,20261023,20261024]
ROLE_COUNTS={'fit':{'system':7935,'user':12660,'assistant':42808},'validation':{'system':2032,'user':3596,'assistant':11085}}
BANK_PATHS={'old':Path('/dev/shm/role-analysis-top8-heads'),'c4':Path('/dev/shm/role-analysis-long-augmentation-heads'),'public':Path('/dev/shm/role-analysis-natural-dev-heads')}
MAX_DIFFERENCE=0.




def require(value, message):
    if not value: raise RuntimeError(message)

def read(path): return json.loads(Path(path).read_text())

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def array_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def canonical(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def labels(frame):
    value = frame.role.map(dict(zip(ROLES, range(5)))).to_numpy()
    require(pd.notna(value).all(), 'Unknown formal role.'); return value.astype(np.int64)

def divide(a, b): return float(a / b) if b else None

def close(actual, expected, label='', *, tolerance=1e-7):
    global MAX_DIFFERENCE
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Mapping keys differ: ' + label)
        for key in expected: close(actual[key], expected[key], label + '/' + str(key), tolerance=tolerance)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), 'List dimensions differ: ' + label)
        for index, value in enumerate(expected): close(actual[index], value, label + '/' + str(index), tolerance=tolerance)
    elif expected is None:
        require(actual is None, 'Undefined value was replaced: ' + label)
    elif isinstance(expected, (bool, str, int)):
        require(actual == expected, 'Discrete value differs: ' + label)
    else:
        require(actual is not None and np.isfinite(float(actual)) and np.isfinite(float(expected)), 'Nonfinite comparison: ' + label)
        difference = abs(float(actual) - float(expected))
        require(difference <= (2e-6 if label.endswith('/nll') else tolerance), 'Numeric value differs: ' + label + '/' + str(difference))
        MAX_DIFFERENCE = max(MAX_DIFFERENCE, difference)

def validate_p(p, count):
    require(p.shape == (count, 5) and p.dtype == np.float32 and np.isfinite(p).all()
        and (p >= 0).all() and (p <= 1.00001).all() and np.allclose(p.sum(1), 1., atol=2e-5, rtol=0), 'Invalid F32 five-class probability cache.')

def rankings(score, truth):
    """Independent tied-score AUROC/AP from grouped descending thresholds."""
    positives, negatives = int(truth.sum()), int((~truth).sum())
    if not positives: return None, None
    order = np.argsort(-score, kind='stable'); sorted_p, y = score[order], truth[order]
    ends = np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]), len(y) - 1]
    tp = np.cumsum(y, dtype=np.int64)[ends]; fp = ends + 1 - tp
    pos_group, neg_group = np.diff(np.r_[0, tp]), np.diff(np.r_[0, fp])
    ap = float(np.sum((pos_group / positives) * (tp / (ends + 1))))
    auc = float(np.sum(pos_group * (negatives - fp + .5 * neg_group)) / (positives * negatives)) if negatives else None
    return auc, ap

def metrics(p, y, tau):
    validate_p(p, len(y)); require(len(y) and ((y >= 0) & (y < 5)).all() and np.isfinite(tau), 'Invalid labels/threshold.')
    pred = p.argmax(1); matrix = np.zeros((5, 5), np.int64)
    np.add.at(matrix, (y, pred), 1)
    support, predicted, diagonal = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    class_f1 = [2 * int(diagonal[i]) / int(support[i] + predicted[i]) if support[i] + predicted[i] else 0. for i in range(5)]
    positive, detected = y == 0, p[:, 0] >= float(tau)
    tp = int(np.count_nonzero(positive & detected)); fn = int(np.count_nonzero(positive & ~detected))
    fp = int(np.count_nonzero(~positive & detected)); tn = int(np.count_nonzero(~positive & ~detected))
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp); auc, ap = rankings(p[:, 0], positive)
    return dict(tokens=len(y), accuracy=float(np.count_nonzero(pred == y) / len(y)),
        macro_f1=float(np.mean([class_f1[i] for i in range(5) if support[i] > 0])),
        nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
        recall={role: divide(int(diagonal[i]), int(support[i])) for i, role in enumerate(ROLES)},
        confusion=matrix.tolist(), system_threshold=float(tau), system_accuracy=float((tp + tn) / len(y)),
        system_balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
        system_precision=divide(tp, tp + fp), system_recall=recall, system_f1=divide(2 * tp, 2 * tp + fp + fn),
        system_false_positive_rate=divide(fp, fp + tn), system_auroc=auc, system_average_precision=ap,
        system_detection_by_role={role: float(detected[y == i].mean()) if np.any(y == i) else None for i, role in enumerate(ROLES)},
        system_confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn))

def legacy_mean(old_rows, p):
    """Preserve original F64 sorted-search grid exactly; do not fix legacy."""
    grid = np.unique(np.r_[np.linspace(.001, .999, 199), np.quantile(p[:, 0], np.linspace(0, 1, 301)), 1.000001])
    scores, fprs = [], []; y = labels(old_rows)
    for profile in PROFILES:
        ix = np.flatnonzero(old_rows.profile.eq(profile)); truth = y[ix] == 0
        pos, neg = np.sort(p[ix, 0][truth]), np.sort(p[ix, 0][~truth])
        tp, fp = len(pos) - np.searchsorted(pos, grid, side='left'), len(neg) - np.searchsorted(neg, grid, side='left')
        scores.append(2 * tp / np.maximum(2 * tp + fp + len(pos) - tp, 1)); fprs.append(fp / max(len(neg), 1))
    ranked = np.mean(scores, axis=0); ranked[np.max(fprs, axis=0) > .05] = -1
    return float(grid[int(np.argmax(ranked))])

def read_probability(path, expected_sha, rows):
    require(digest(path) == expected_sha, 'Probability byte SHA mismatch.')
    with np.load(path, allow_pickle=False) as values:
        require(set(values.files) == {'p', 'sample_ix'} and np.array_equal(values['sample_ix'], rows.sample_ix), 'Probability sample order/namespace mismatch.')
        p = values['p'].copy()
    validate_p(p, len(rows)); return p

def model_arrays(path, mean, scale, expected_metadata=None):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata'])); require(meta['kind'] == 'mlp' and meta['hidden'] == 128
            and meta['dimensions'] == 512 and meta['heads'] == 8 and meta['head_dim'] == 64 and meta['roles'] == ROLES, 'Portable model geometry/role order mismatch.')
        require(np.array_equal(archive['mean'], mean) and np.array_equal(archive['scale'], scale), 'Frozen old fit-only embedded scaler mismatch.')
        weights = {key[8:]: archive[key].copy() for key in archive.files if key.startswith('weight::')}
        expected = {'encoder.0.weight': (128, 512), 'encoder.0.bias': (128,), 'head.weight': (5, 128), 'head.bias': (5,)}
        require(set(weights) == set(expected) and all(weights[key].shape == shape and weights[key].dtype == np.float32
            and np.isfinite(weights[key]).all() for key, shape in expected.items()), 'Portable weights changed shape/dtype/finiteness.')
        if expected_metadata:
            for key, value in expected_metadata.items(): close(meta[key], value, 'model/' + key)
    return meta, weights

def group_indices(rows):
    for b,f in rows.items():require(np.array_equal(f.index,np.arange(len(f))), 'Validation local row index changed: '+b)
    old,cv,pub=rows['old'],rows['c4'],rows['public']
    result={g:('old',np.flatnonzero(old.profile.eq(p))) for g,p in zip(OLD5,PROFILES)}
    for bank,frame,name in [('old',old,'old-long-full-tail'),('c4',cv,'new-C4-long-full-tail')]:
        full=frame[frame.profile.eq('long-full')]
        offset=full.body_offset if 'body_offset' in full else full.token_ix-full.groupby(['source_set','prompt_ix','message_ix']).token_ix.transform('min')
        result[name]=(bank,full.index.to_numpy()[offset.to_numpy()>=512])
    result['new-C4-long-full']=('c4',np.flatnonzero(cv.profile.eq('long-full')))
    result[PUBLIC[0]]=('public',np.arange(len(pub)))
    for name,dataset in zip(PUBLIC[1:],['oasst','toxicchat0124']):result[name]=('public',np.flatnonzero(pub.dataset.eq(dataset)))
    require(set(result)==set(GROUPS) and all(len(ix) for _,ix in result.values()),'Missing declared group support.')
    return result

def evaluate(rows,p,tau):
    ys={b:labels(frame) for b,frame in rows.items()}
    groups={g:metrics(p[b][ix],ys[b][ix],tau) for g,(b,ix) in group_indices(rows).items()}
    primary=[groups[g][k] for g in GROUPS for k in ['system_precision','system_recall','system_balanced_accuracy']]
    return dict(system_threshold=float(tau),groups=groups,
        min33=min(primary) if all(v is not None and np.isfinite(v) for v in primary) else None,
        old5_mean_f1=float(np.mean([groups[g]['system_f1'] for g in OLD5])),
        separate_bank_descriptive={b:metrics(p[b],ys[b],tau) for b in rows},
        pooled_descriptive_only=metrics(np.concatenate([p[b] for b in rows]),np.concatenate([ys[b] for b in rows]),tau))

def direct_grid(rows,p):
    """Independent brute-force actual F32 consumer at every original F64 tau."""
    grid=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(np.concatenate([p[b][:,0] for b in ['old','c4','public']]),np.linspace(0,1,301)),1.000001])
    curves={};keys=['P','R','BAcc','F1','FPR']
    for name,(bank,ix) in group_indices(rows).items():
        validate_p(p[bank],len(rows[bank]));truth=labels(rows[bank].iloc[ix])==0;score=p[bank][ix,0]
        pos,neg=int(truth.sum()),int((~truth).sum());require(pos>0 and neg>0,'Objective group lacks positive/negative support.')
        values=[]
        for tau in grid:
            chosen=score>=float(tau);tp=int(np.count_nonzero(truth&chosen));fp=int(np.count_nonzero(~truth&chosen));fn=pos-tp;tn=neg-fp
            values.append([tp/(tp+fp) if tp+fp else np.nan,tp/pos,(tp/pos+tn/neg)/2,2*tp/(2*tp+fp+fn),fp/neg])
        curves[name]=np.asarray(values)
    minimum=np.min([curves[g][:,k] for g in GROUPS for k in [0,1,2]],axis=0);minimum[~np.isfinite(minimum)]=-np.inf
    oldmean=np.mean([curves[g][:,3] for g in OLD5],axis=0)
    feasible=np.max([curves[g][:,4] for g in FEASIBLE],axis=0)<=.05
    valid=np.flatnonzero(feasible&np.isfinite(minimum));require(len(valid),'No defined FPR-feasible max33 threshold.')
    best=sorted(valid,key=lambda i:(-float(minimum[i]),-float(oldmean[i]),float(grid[i])))[0]
    proof=dict(grid_thresholds=len(grid),groups=11,primary_statistics=len(grid)*33,additional_F1_FPR_statistics=len(grid)*22,
        maximum_direct_difference=0.,actual_F32_comparison=True,all_grid_defined_semantics_equal=True,
        all_grid_feasibility_equal=True,global_argmax_equal=True,min33=float(minimum[best]),
        old5_mean_f1=float(oldmean[best]),feasible_thresholds=int(feasible.sum()))
    annex=dict(grid_sha256=array_sha(grid),effective_F32_grid_sha256=array_sha(grid.astype(np.float32)),
        direct_curves_sha256={g:array_sha(curves[g]) for g in GROUPS},stat_order=keys,
        undefined_precision_thresholds={g:int(np.isnan(curves[g][:,0]).sum()) for g in GROUPS})
    return float(grid[best]),proof,annex

def guard(result,control,legacy=False):
    tol=1e-12;reasons=[]
    f1={g:control['groups'][g]['system_f1']-result['groups'][g]['system_f1'] for g in ORIGINAL}
    accuracy={g:control['groups'][g]['accuracy']-result['groups'][g]['accuracy'] for g in ORIGINAL}
    gains={g:result['groups'][g]['system_f1']-control['groups'][g]['system_f1'] for g in FULL}
    meanloss=control['old5_mean_f1']-result['old5_mean_f1']
    if max(f1.values())>.02+tol:reasons.append('old4_system_f1_loss')
    if max(accuracy.values())>.01+tol:reasons.append('old4_five_accuracy_loss')
    if meanloss>.005+tol:reasons.append('old5_mean_f1_loss')
    if any(v < -tol for v in gains.values()):reasons.append('C4_full_tail_f1_regression')
    pubf1={g:control['groups'][g]['system_f1']-result['groups'][g]['system_f1'] for g in PUBLIC}
    pubacc={g:control['groups'][g]['accuracy']-result['groups'][g]['accuracy'] for g in PUBLIC}
    if not legacy:
        if max(pubf1.values())>.02+tol:reasons.append('public_system_f1_loss')
        if max(pubacc.values())>.01+tol:reasons.append('public_five_accuracy_loss')
    if any(result['groups'][g]['system_false_positive_rate']>.05+tol for g in FEASIBLE):reasons.append('FPR_above_limit')
    if not all(result['groups'][g][k] is not None and np.isfinite(result['groups'][g][k]) for g in GROUPS for k in ['system_f1','accuracy','system_false_positive_rate']):reasons.append('nonfinite_metrics')
    return dict(eligible=not reasons,rejection_reasons=reasons,old4_f1_losses=f1,old4_accuracy_losses=accuracy,
        old5_mean_f1_loss=meanloss,C4_full_tail_f1_gains=gains,public_f1_losses=pubf1,public_accuracy_losses=pubacc,legacy_savedGPU_only=legacy)

def history_audit(history,best_epoch):
    require(1<=best_epoch<=len(history)<=45,'Invalid epoch evidence.');best=-1.;chosen=-1
    for epoch,row in enumerate(history,1):
        require(row['epoch']==epoch and row['steps']==157 and row['instances']==641493 and row['cumulative_steps']==157*epoch
            and np.isfinite(row['loss']) and np.isfinite(row['old5_mean_f1']),'Epoch update/instance evidence mismatch.')
        if row['old5_mean_f1']>best+1e-5:best,chosen=row['old5_mean_f1'],epoch
        if epoch<len(history):require(not(epoch>=10 and epoch-chosen>=8),'History continued after early-stop trigger.')
    require(chosen==best_epoch and (len(history)==45 or len(history)>=10 and len(history)-chosen>=8),'Best/terminal epoch mismatch.')
    return dict(best_epoch=chosen,best_steps=chosen*157,epochs=len(history),actual_steps=len(history)*157,
        actual_instances=len(history)*641493,instances_per_epoch=641493,steps_per_epoch=157,last_batch_instances=2517)

def fixed_mean(values):
    require(len(values)==4,'Exactly four inherited members required.');total=np.zeros(values[0].shape,np.float64)
    for p in values:validate_p(p,len(total));require(p.shape==total.shape,'Unaligned member rows.');total+=p
    result=(total/4).astype(np.float32);validate_p(result,len(total));return result

def matched_draw(fit,nf):
    require(fit.cache_ix.is_unique and fit.split.eq('fit').all() and nf.split.eq('fit').all()
        and set(nf.role)<={'system','user','assistant'},'Invalid role-slot pool or validation contamination.')
    rng=np.random.RandomState(20516085);cache=np.empty(len(nf),np.int64);sample=np.empty(len(nf),np.int64)
    for role in ['system','user','assistant']:
        slots=np.flatnonzero(nf.role.eq(role));pool=fit[fit.role.eq(role)];require(len(pool),'Empty matched role pool.')
        picks=rng.choice(len(pool),size=len(slots),replace=True);cache[slots]=pool.iloc[picks].cache_ix;sample[slots]=pool.iloc[picks].sample_ix
    frame=pd.DataFrame(dict(extra_slot=np.arange(len(nf)),natural_fit_sample_ix=nf.sample_ix.to_numpy(),
        natural_source_uid=nf.source_uid.to_numpy(),role=nf.role.to_numpy(),matched_old_cache_ix=cache,matched_old_sample_ix=sample))
    require(np.array_equal(fit.set_index('cache_ix').loc[cache].role,nf.role),'Matched role slots changed.')
    proof=dict(seed=20516085,draw_once=True,replacement=True,rows=len(nf),role_counts={r:int(nf.role.eq(r).sum()) for r in ['system','user','assistant']},
        matched_cache_ix_sha256=array_sha(cache),matched_sample_ix_sha256=array_sha(sample),natural_fit_sample_ix_sha256=array_sha(nf.sample_ix.to_numpy()),
        extra_labels_sha256=array_sha(labels(nf)),no_validation_rows_sampled=True,torch_rng_consumed=False)
    return frame,proof

def drift(rows,reference,cpu,tau):
    groups={g:(b,ix) for g,(b,ix) in group_indices(rows).items() if b!='public'}
    groups.update({'all-old':('old',np.arange(len(rows['old']))),'all-new':('c4',np.arange(len(rows['c4'])))})
    result={}
    for g,(bank,ix) in groups.items():
        a,b=reference[bank][ix],cpu[bank][ix];validate_p(a,len(ix));validate_p(b,len(ix));delta=np.abs(a-b);maximum=float(delta.max())
        require(maximum<5e-4,'Full reference drift exceeds technical tolerance: '+g)
        result[g.replace('new-C4-long','new-long')]=dict(tokens=len(ix),maximum_probability_difference=maximum,
            mean_probability_difference=float(delta.mean()),absolute_difference_quantiles={str(q):float(np.quantile(delta,q)) for q in [.5,.9,.99,1.]},
            argmax_changes=int(np.count_nonzero(a.argmax(1)!=b.argmax(1))),
            original_threshold_decision_changes=int(np.count_nonzero((a[:,0]>=tau)!=(b[:,0]>=tau))))
    return result

def source_partitions(old,c4,natural):
    for frame,count in [(old,755282),(c4,389120),(natural,80116)]:
        require(len(frame)==count and np.array_equal(frame.cache_ix,np.arange(count)) and frame.sample_ix.is_unique
            and set(frame.split)=={'fit','validation'} and frame.wrapper.eq('native').all(),'Bank namespace/count/ownership changed.')
    raw=old[old.split.eq('fit')];ids=set(raw.loc[raw.profile.eq('long-full'),'document_id'])
    removed=raw[raw.profile.eq('long-single')&raw.document_id.isin(ids)]
    fit=raw[~raw.cache_ix.isin(removed.cache_ix)].copy();ov=old[old.split.eq('validation')].copy().reset_index(drop=True)
    cv=c4[c4.split.eq('validation')].copy().reset_index(drop=True)
    nf=natural[natural.split.eq('fit')].copy().reset_index(drop=True);pv=natural[natural.split.eq('validation')].copy().reset_index(drop=True)
    require(len(fit)==578090 and len(ov)==173032 and len(cv)==61440 and len(nf)==63403 and len(pv)==16713
        and len(removed)==4160 and removed.document_id.nunique()==26 and len(ids)==132,'Densefit replacement/split geometry changed.')
    require(natural.source_set.eq('natural-dev-clean').all() and natural.sample_ix.min()>=4_000_000_000_000
        and natural.original_cache_ix.is_monotonic_increasing and natural.sample_ix.is_monotonic_increasing
        and set(natural.role)=={'system','user','assistant'} and natural.semantic_role.eq(natural.role).all()
        and nf.eligible_fit.eq(True).all() and pv.eligible_clean_validation.eq(True).all()
        and not natural.old_generated_collision.any() and not natural.cross_split_assistant_collision.any(),'Clean native natural fit/validation ownership changed.')
    for key in ['source_uid','group_id','family_id','dependency_component_id']:
        require(not set(nf[key])&set(pv[key]),'Natural fit/public leakage: '+key)
    for split,frame,count in [('fit',nf,255),('validation',pv,64)]:
        require(frame.source_uid.nunique()==count and {r:int(frame.role.eq(r).sum()) for r in ROLE_COUNTS[split]}==ROLE_COUNTS[split],'Natural source/role support changed.')
    require(not set(fit.group_id)&set(ov.group_id) and not set(cv.group_id)&set(fit.group_id)
        and not set(natural.sample_ix)&set(old.sample_ix) and c4.source_set.eq('long-augmentation').all()
        and cv.profile.eq('long-full').all() and c4.sample_ix.min()>=6_000_000_000_000,'Original/new/public split or namespace overlap.')
    partition=dict(fit_rows=len(fit),validation_rows=len(ov),removed_sparse_rows=len(removed),removed_sparse_documents=26,
        fit_cache_ix_sha256=array_sha(fit.cache_ix.to_numpy()),fit_sample_ix_sha256=array_sha(fit.sample_ix.to_numpy()),fit_labels_sha256=array_sha(labels(fit)),
        validation_cache_ix_sha256=array_sha(ov.cache_ix.to_numpy()),validation_sample_ix_sha256=array_sha(ov.sample_ix.to_numpy()),validation_labels_sha256=array_sha(labels(ov)))
    return fit,dict(old=ov,c4=cv,public=pv),nf,partition

def old_validation_record(value,rows):
    profiles={p:value['groups']['old-'+p] for p in PROFILES};tail=dict(value['groups']['old-long-full-tail'])
    full=rows[rows.profile.eq('long-full')];offset=full.token_ix-full.groupby(['source_set','prompt_ix','message_ix']).token_ix.transform('min')
    tail.update(documents=int(full.loc[offset.ge(512),'document_id'].nunique()),body_start=512,body_end=1023)
    return dict(system_threshold=value['system_threshold'],mean_system_f1=value['old5_mean_f1'],
        original_mean_system_f1=float(np.mean([profiles[p]['system_f1'] for p in PROFILES[:4]])),
        original_mean_accuracy=float(np.mean([profiles[p]['accuracy'] for p in PROFILES[:4]])),profiles=profiles,long_full_tail=tail)

def parity_proof(proof,nf):
    require(proof['fit_only'] is True and proof['validation_probabilities_computed'] is False and proof['sampled_tokens']==256
        and proof['target_cache_ix_sha256']==array_sha(nf.cache_ix.to_numpy()[:256])
        and proof['sample_ix_sha256']==array_sha(nf.sample_ix.to_numpy()[:256])
        and np.isfinite(proof['maximum_probability_difference']) and 0<=proof['maximum_probability_difference']<5e-4,'FIT-only portable scope or numerical proof changed.')

def independent_predict(weights,mean,scale,bank,rows):
    """Every raw F16 row, no producer model loader or predictor."""
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    weights={k:torch.from_numpy(v) for k,v in weights.items()};mt,st=torch.from_numpy(mean),torch.from_numpy(scale);parts=[]
    with torch.inference_mode():
        for start in range(0,len(rows),4096):
            raw=torch.from_numpy(np.asarray(bank[rows.cache_ix.to_numpy()[start:start+4096]],np.float32))
            x=(raw-mt)/st;require(torch.isfinite(x).all(),'Nonfinite raw/scaled independent prediction vector.')
            z=torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x,weights['encoder.0.weight'],weights['encoder.0.bias'])),weights['head.weight'],weights['head.bias'])
            parts.append(torch.log_softmax(z,dim=1).exp().numpy().astype(np.float32))
    p=np.concatenate(parts);validate_p(p,len(rows));return p


def selection(records,zero,legacy):
    require(len(records)==10 and {r['record'] for r in records}==set(RECORDS),'Ten formal records required.')
    require(len(zero)==2 and {r['record'] for r in zero}==set(ZERO_RECORDS),'Two separate zero controls required.')
    index={r['record']:r for r in records+zero};candidate=index['new-max33']['validation'];minimum=candidate['min33'];reasons=[];guards={};gains={}
    if minimum is None or not np.isfinite(minimum) or minimum<.9:reasons.append('absolute_min33_below_90_percent')
    for name in GUARD_CONTROLS:
        control=index[name]['validation'];g=guard(candidate,control);guards[name]=g
        reasons.extend(name+':'+r for r in g['rejection_reasons'])
        if name in STRICT_CONTROLS:
            gain=minimum-control['min33'] if minimum is not None and control['min33'] is not None else None;gains[name]=gain
            if gain is None or not np.isfinite(gain) or gain<=1e-12:reasons.append(name+':min33_not_strictly_improved')
    g=guard(candidate,legacy,True);guards['original-savedGPU']=g;reasons.extend('original-savedGPU:'+r for r in g['rejection_reasons'])
    return dict(candidate='new-max33',promoted=not reasons,no_fallback=True,min33_gains=gains,guards=guards,rejection_reasons=reasons,
        naturalCE_is_diagnostic_only=True,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)


def verify_csv(frame,records,zero=False):
    expected_names=ZERO_RECORDS if zero else RECORDS
    require(len(records)==len(expected_names) and {r['record'] for r in records}==set(expected_names),'CSV record coverage changed.')
    expected={(r['record'],g):v for r in records for g,v in dict(r['validation']['groups'],pooled_descriptive_only=r['validation']['pooled_descriptive_only']).items()}
    require(len(frame)==len(records)*12 and not frame.duplicated(['record','group']).any() and set(zip(frame.record,frame.group))==set(expected),'CSV group/pool coverage changed.')
    for row in frame.to_dict('records'):
        want=expected[(row['record'],row['group'])]
        for k,v in want.items():
            if isinstance(v,(dict,list)):continue
            close(None if pd.isna(row[k]) else row[k],v,'csv/'+row['record']+'/'+row['group']+'/'+k)
        for role in ROLES:
            x=row['role_recall_'+role];close(None if pd.isna(x) else x,want['recall'][role],'csv/recall/'+role)


def residual_replay(base,residual,mean,scale,bank,rows,*,require_no_swaps=True):
    """Own rawF16->base+delta consumer; no producer loader or predictor.

    Returns full p and compact raw-logit invariance/actual-p transition proof.
    Non-system raw byte equality and raw argmax equality are structural checks.
    Actual F32 probability class swaps are separately measured and fail closed.
    """
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    wb={k:torch.from_numpy(v) for k,v in base.items()};wr={k:torch.from_numpy(v) for k,v in residual.items()}
    mt,st=torch.from_numpy(mean),torch.from_numpy(scale);parts=[];base_parts=[]
    oldhash,newhash=hashlib.sha256(),hashlib.sha256();conditional_changes=0;non_system_swaps=0;cot_assistant=0
    transitions=np.zeros((5,5),np.int64);by_truth={r:np.zeros((5,5),np.int64) for r in ROLES}
    prob_ties=0;underflow=0;non_sys_underflow=0;maxdelta=0.;delta_min=4.;delta_max=-4.;delta_sum=0.;conditional_p_changes=0;conditional_p_ties=0
    with torch.inference_mode():
        for start in range(0,len(rows),4096):
            frame=rows.iloc[start:start+4096];raw=torch.from_numpy(np.asarray(bank[frame.cache_ix.to_numpy()],np.float32));x=(raw-mt)/st
            require(torch.isfinite(x).all(),'Nonfinite residual input.')
            z=torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x,wb['encoder.0.weight'],wb['encoder.0.bias'])),wb['head.weight'],wb['head.bias'])
            hidden=torch.nn.functional.gelu(torch.nn.functional.linear(x,wr['encoder.0.weight'],wr['encoder.0.bias']))
            # eval dropout is identity; explicitly no stochastic operation here.
            score=torch.nn.functional.linear(hidden,wr['head.weight'],wr['head.bias']).reshape(-1)
            delta=4*torch.tanh(score);corrected=z.clone();corrected[:,0]+=delta
            require(torch.isfinite(corrected).all() and torch.isfinite(delta).all() and bool((delta.abs()<=4).all()),'Residual bound/finiteness violated.')
            maxdelta=max(maxdelta,float(delta.abs().max()));dv=delta.numpy()
            delta_min=min(delta_min,float(dv.min()));delta_max=max(delta_max,float(dv.max()));delta_sum+=float(dv.astype(np.float64).sum())
            before,after=z.numpy(),corrected.numpy();a,b=np.ascontiguousarray(before[:,1:]),np.ascontiguousarray(after[:,1:])
            require(a.dtype==b.dtype==np.float32 and np.array_equal(a.view(np.uint32),b.view(np.uint32)),'Non-system raw logits changed at bit level.')
            oldhash.update(a.tobytes());newhash.update(b.tobytes());conditional_changes+=int(np.count_nonzero(a.argmax(1)!=b.argmax(1)))
            oldp=torch.log_softmax(z,dim=1).exp().numpy().astype(np.float32);p=torch.log_softmax(corrected,dim=1).exp().numpy().astype(np.float32)
            validate_p(oldp,len(frame));validate_p(p,len(frame));oldpred,newpred=oldp.argmax(1),p.argmax(1)
            changed=(oldpred>0)&(newpred>0)&(oldpred!=newpred);non_system_swaps+=int(changed.sum());cot_assistant+=int(np.count_nonzero((oldpred==2)&(newpred==3)))
            conditional_p_changes+=int(np.count_nonzero(oldp[:,1:].argmax(1)!=p[:,1:].argmax(1)))
            conditional_p_ties+=int(np.count_nonzero((p[:,1:]==p[:,1:].max(1,keepdims=True)).sum(1)>1))
            non_sys_underflow+=int(np.count_nonzero(p[:,1:]==0))
            np.add.at(transitions,(oldpred,newpred),1)
            for r_ix,role in enumerate(ROLES):
                mask=labels(frame)==r_ix;np.add.at(by_truth[role],(oldpred[mask],newpred[mask]),1)
            # Report all complete-probability ties/zeros, never perturb them.
            order=np.sort(p,axis=1);prob_ties+=int(np.count_nonzero(order[:,-1]==order[:,-2]));underflow+=int(np.count_nonzero(p==0))
            parts.append(p);base_parts.append(oldp)
    proof=dict(rows=len(rows),non_system_raw_before_sha256=oldhash.hexdigest(),non_system_raw_after_sha256=newhash.hexdigest(),
        raw_non_system_bitexact=oldhash.hexdigest()==newhash.hexdigest(),conditional_raw_argmax_changes=conditional_changes,
        p_non_system_different_class_swaps=non_system_swaps,p_cot_to_assistant_swaps=cot_assistant,
        five_class_transition=transitions.tolist(),transitions_by_true_role={r:m.tolist() for r,m in by_truth.items()},
        full_p_argmax_tie_rows=prob_ties,p_zero_elements=underflow,maximum_abs_delta=maxdelta,
        delta_min=delta_min,delta_max=delta_max,delta_mean=delta_sum/len(rows),
        p_conditional_non_system_argmax_changes=conditional_p_changes,p_non_system_tie_rows=conditional_p_ties,
        p_non_system_underflow_values=non_sys_underflow)
    require(proof['raw_non_system_bitexact'] and conditional_changes==0,'Non-system conditional raw classifier changed.')
    if require_no_swaps:require(non_system_swaps==0,'Actual F32 p introduced a non-system class swap; no epsilon repair allowed.')
    return np.concatenate(parts),np.concatenate(base_parts),proof


def weight_state_sha(state,sort=False):
    pairs=sorted(state.items()) if sort else list(state.items())
    return canonical([dict(name=n,shape=list(v.shape),dtype=str(v.dtype),sha256=array_sha(v)) for n,v in pairs])


def binary_model_arrays(path,oldbase,mean,scale,expected_metadata=None):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));require(meta['kind']=='binary-residual' and meta['dimensions']==512 and meta['base_hidden']==128
            and meta['residual_hidden']==32 and meta['delta_formula']=='4*tanh(g)' and meta['roles']==ROLES
            and meta['base_model_sha256']==CURRENT_SHA,'Portable binary geometry/formula/base changed.')
        require(np.array_equal(z['mean'].view(np.uint32),mean.view(np.uint32)) and np.array_equal(z['scale'].view(np.uint32),scale.view(np.uint32)),'Embedded original scaler changed bits.')
        full={k[8:]:z[k].copy() for k in z.files if k.startswith('weight::')}
    expected={'base.encoder.0.weight':(128,512),'base.encoder.0.bias':(128,),'base.head.weight':(5,128),'base.head.bias':(5,),
        'correction.0.weight':(32,512),'correction.0.bias':(32,),'correction.3.weight':(1,32),'correction.3.bias':(1,)}
    require(set(full)==set(expected) and all(v.shape==expected[k] and v.dtype==np.float32 and np.isfinite(v).all() for k,v in full.items()),'Malformed portable base/correction arrays.')
    base={k[5:]:v for k,v in full.items() if k.startswith('base.')}
    for key in oldbase:require(np.array_equal(base[key].view(np.uint32),oldbase[key].view(np.uint32)),'Trained portable base weight changed bits: '+key)
    require(weight_state_sha(base,True)==meta['embedded_base_weight_sha256'] and array_sha(mean)==meta['mean_sha256'] and array_sha(scale)==meta['scale_sha256'],'Portable embedded array digests changed.')
    if expected_metadata:
        for key,v in expected_metadata.items():close(meta[key],v,'portable/'+key)
    residual={k:v for k,v in [('encoder.0.weight',full['correction.0.weight']),('encoder.0.bias',full['correction.0.bias']),('head.weight',full['correction.3.weight']),('head.bias',full['correction.3.bias'])]}
    return meta,base,residual


def initial_binary_state(oldbase):
    import torch
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Sequential(torch.nn.Linear(512,128),torch.nn.GELU(),torch.nn.Dropout(.1));self.head=torch.nn.Linear(128,5)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.base=Base();self.correction=torch.nn.Sequential(torch.nn.Linear(512,32),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(32,1))
            torch.nn.init.zeros_(self.correction[-1].weight);torch.nn.init.zeros_(self.correction[-1].bias)
            self.base.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in oldbase.items()})
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261021);model=Model();state={n:v.detach().numpy().copy() for n,v in model.state_dict().items()}
    return state,weight_state_sha(state)


def processes():
    names={'systemness_binary_residual.py','systemness_natural_dev_extract.py'};active=[]
    for e in Path('/proc').iterdir():
        if not e.name.isdigit() or int(e.name)==os.getpid():continue
        try:args=(e/'cmdline').read_bytes().split(b'\0')
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(x.decode(errors='replace')).name in names for x in args if x):active.append(int(e.name))
    return active


def terminal_gate():
    require(not processes(),'Binary trainer/extractor live: no partial result reads.')
    require((OUT/'complete.json').is_file() and (OUT/'status.json').is_file(),'Formal completion/status absent.')
    require(read(OUT/'status.json')['stage']=='validation_complete','Trainer not terminal complete.')


def probability_drift(reference,p,tau):
    validate_p(reference,len(p));validate_p(p,len(reference));difference=np.abs(reference.astype(np.float64)-p.astype(np.float64))
    return dict(rows=len(p),maximum_probability_difference=float(difference.max()),mean_probability_difference=float(difference.mean()),
        changed_argmax=int(np.count_nonzero(reference.argmax(1)!=p.argmax(1))),
        changed_original_threshold_decisions=int(np.count_nonzero((reference[:,0]>=tau)!=(p[:,0]>=tau))),threshold=float(tau))


def compare_raw_evidence(saved,computed,label):
    t=np.asarray(computed['five_class_transition']);n=computed['rows']
    expected=dict(tokens=n,raw_non_system_sha256=computed['non_system_raw_after_sha256'],base_raw_non_system_sha256=computed['non_system_raw_before_sha256'],
        raw_non_system_argmax_changes=computed['conditional_raw_argmax_changes'],delta_min=computed['delta_min'],delta_max=computed['delta_max'],delta_mean=computed['delta_mean'],
        base_to_residual_p_argmax=t.tolist(),p_conditional_non_system_argmax_changes=computed['p_conditional_non_system_argmax_changes'],
        p_non_system_tie_rows=computed['p_non_system_tie_rows'],p_non_system_underflow_values=computed['p_non_system_underflow_values'],
        CoT_to_assistant_p_argmax=int(t[2,3]),non_system_to_system_p_argmax=int(t[1:,0].sum()),system_to_non_system_p_argmax=int(t[0,1:].sum()),
        non_system_to_different_non_system_p_argmax=int(t[1:,1:].sum()-np.trace(t[1:,1:])))
    require(expected['non_system_to_different_non_system_p_argmax']==0 and expected['raw_non_system_argmax_changes']==0
        and expected['raw_non_system_sha256']==expected['base_raw_non_system_sha256'],'Actual raw/p preservation failed.')
    for k,v in expected.items():close(saved[k],v,label+'/'+k)
    descriptions=dict(metric_convention='Unmodified savedF32 p.argmax; raw-tail bitidentity does not guarantee p rounding preserves non-system argmax.',
        transition_definition='Rows index original base p.argmax; columns index residual p.argmax, regardless of ground-truth role. CoT_to_assistant is a newly introduced winner transition, not total truth-CoT misclassification.')
    require(set(saved)==set(expected)|set(descriptions),'Unexpected raw evidence field scope.')
    for k,v in descriptions.items():require(saved[k]==v,'Raw evidence transition semantics changed: '+k)


def verified_layout(inherited,bind):
    """Independent explicit prepared-bank identity; no global verifiers."""
    upstream=inherited['upstream'];paths={'old':BASE/'systemness-head-features','c4':BASE/'systemness-long-augmentation','public':BASE/'systemness-natural-dev-features'}
    records={b:read(path/'extraction-complete.json') for b,path in paths.items()}
    bind(paths['old']/'extraction-complete.json',upstream['old_extraction_sha256']);bind(paths['c4']/'extraction-complete.json',upstream['new_extraction_sha256'])
    bind(paths['c4']/'data-complete.json',upstream['new_data_sha256']);bind(paths['public']/'extraction-complete.json',inherited['natural_complete_sha256'])
    close(records['c4']['signature'],upstream['new_extraction_signature'],'C4 extraction signature');close(records['public']['signature'],inherited['natural_signature'],'natural extraction signature')
    heads=[458,462,2045,460,459,2047,456,997]
    require(inherited['selected_head_ids']==heads and all(v['selected_head_ids']==heads and v['all_rows_written'] is True and v['all_finite'] is True for v in records.values()),'Raw feature completeness/head rank changed.')
    tables={};banks={}
    for b,count in [('old',755282),('c4',389120),('public',80116)]:
        folder=BANK_PATHS[b];record=records[b]
        require((folder/'features.f16').stat().st_size==count*512*2,'Raw feature size changed.')
        bind(folder/'rows.parquet',record['rows_sha256']);bind(folder/'features.f16',record['features_sha256']);tables[b]=pd.read_parquet(folder/'rows.parquet')
        banks[b]=np.memmap(folder/'features.f16',mode='r',dtype=np.float16,shape=(count,512))
    require(upstream['old_features_sha256']==records['old']['features_sha256'] and upstream['new_features_sha256']==records['c4']['features_sha256']
        and upstream['old_rows_sha256']==records['old']['rows_sha256'] and upstream['new_rows_sha256']==records['c4']['rows_sha256']
        and inherited['natural_features_sha256']==records['public']['features_sha256'] and inherited['natural_rows_sha256']==records['public']['rows_sha256'],'Inherited feature binding changed.')
    fit,rows,nf,partition=source_partitions(tables['old'],tables['c4'],tables['public']);close(upstream['old_partitions'],partition,'upstream partitions')
    return fit,rows,nf,partition,banks


def actual(output):
    require(SOURCE_SHA is not None and PROTOCOL_SHA is not None,'Final binary source/protocol not pinned: actual forbidden.')
    terminal_gate()
    require(Path(output).resolve().is_relative_to(Path('/tmp')),'Actual independent auditor writes /tmp only.')
    bindings={};probability_bindings={}
    def bind(path,want=None):
        path=Path(path);require(path.is_file(),'Missing bound artifact: '+str(path));got=digest(path)
        if want is not None:require(got==want,'Bound artifact SHA changed: '+str(path))
        bindings[str(path.resolve())]=got;return got
    def p_read(path,want,frame):
        path=Path(path);probability_bindings[str(path.resolve())]=bind(path,want);return read_probability(path,want,frame)
    bind(__file__);bind(SOURCE,SOURCE_SHA);bind(DESIGN,DESIGN_SHA);bind(OUT/'protocol.json',PROTOCOL_SHA);bind(OUT/'status.json')
    complete=read(OUT/'complete.json');complete_sha=bind(OUT/'complete.json');plan=read(OUT/'search-plan.json');plan_sha=bind(OUT/'search-plan.json')
    declaration=read(OUT/'protocol.json');static=declaration['static'];sig=plan['signature'];proto=declaration['protocol'];design=read(DESIGN)
    artifacts={'protocol.json','search-plan.json','draw-manifest.json','controls.json','zero-controls.json','records.json','selection.json','validation-metrics.csv','zero-control-metrics.csv','source-counts.json','report.txt'}
    require(complete['signature']==sig and complete['source_sha256']==SOURCE_SHA and complete['tests_read'] is False and complete['primary_changed'] is False
        and complete['new_independent_confirmation_required'] is True and set(complete['artifact_sha256'])==artifacts
        and set(complete['candidate_sha256'])=={'new','matched'},'Completion contract changed.')
    for n,h in complete['artifact_sha256'].items():bind(OUT/n,h)
    require(plan['protocol']==proto and sig['source_sha256']==SOURCE_SHA and sig['protocol_sha256']==PROTOCOL_SHA and sig['static_sha256']==canonical(static)
        and static['source_sha256']==SOURCE_SHA and static['design_sha256']==DESIGN_SHA and static['validation_probability_arrays_read'] is False
        and proto['tests_read'] is False and proto['primary_changed'] is False and proto['records']==RECORDS and proto['groups']==GROUPS
        and proto['feasible']==FEASIBLE and proto['max_FPR']==.05 and proto['zero_controls']==ZERO_RECORDS,'Protocol/source/static binding changed.')
    expected_neural=dict(seed=20261021,residual_hidden=32,dropout=.1,optimizer='AdamW',lr=.001,weight_decay=.001,clip=5.,batch=4096,instances=641493,steps=157,last_batch=2517,max_epochs=45,min_epochs=10,patience=8,improvement=1e-5)
    close(proto['neural'],expected_neural,'protocol/neural')
    require(proto['CPU']==dict(threads=4,float32_matmul_precision='highest',dtype='float32',chunk=4096)
        and proto['promotion']['strict_controls']==STRICT_CONTROLS and proto['promotion']['guard_controls']==GUARD_CONTROLS
        and proto['promotion']['absolute_min33']==.9 and proto['frozen_base']['model_sha256']==CURRENT_SHA,'Frozen numerical/control protocol changed.')
    def declared_maps(value):
        if not isinstance(value,dict):return
        for key,item in value.items():
            if key in {'metadata_sha256','dependencies_sha256'}:
                for name,want in item.items():bind(Path(name) if Path(name).is_absolute() else HERE/name,want)
            elif key in {'prior','base','original'}:declared_maps(item)
    declared_maps(static)
    require(static['packages']=={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','pyarrow','scikit-learn']},'Portable packages changed.')
    prior_out=BASE/'systemness-natural-fit-probes';prior_audit_path=prior_out/'preparation/audits/actual-development-audit.json';bind(prior_audit_path,PRIOR_AUDIT_SHA)
    prior=read(prior_audit_path)
    require(static['prior_actual_audit_sha256']==PRIOR_AUDIT_SHA==design['background']['latest_actual_audit_sha256']
        and prior['stage']=='actual_completed_natural_fit_development_audit' and prior['audit_passed'] is True and prior['test_data_or_results_read'] is False
        and prior['trainer_complete_sha256']==digest(prior_out/'complete.json') and prior['trainer_protocol_sha256']==digest(prior_out/'protocol.json')
        and prior['trainer_selection_sha256']==digest(prior_out/'selection.json') and prior['trainer_records_sha256']==digest(prior_out/'records.json'),'Prior actual certificate closure changed.')
    for path,want in prior['provenance_sha256'].items():bind(path,want)
    for path,want in prior['probability_bindings'].items():
        probability_bindings[str(Path(path).resolve())]=bind(path,want)
    bind(prior['auditor_source_path'],prior['auditor_source_sha256'])
    fit,rows,nf,partition,banks=verified_layout(sig['inherited'],bind);close(partition,static['original_partitions'],'original partitions')
    for name,frame in [('fit',fit),('old',rows['old']),('c4',rows['c4']),('natural-fit',nf),('public',rows['public'])]:
        require(array_sha(frame.sample_ix.to_numpy())==static['sample_ix_sha256'][name]==design['data']['source_sample_sha256'][name],'Source sample sequence changed: '+name)
    draw,summary=matched_draw(fit,nf);draw_path=Path(static['draw_source_path']);draw_sha=bind(draw_path,static['draw_slots_sha256'])
    require(draw_sha==design['matching']['draw_sha256'],'Prospective exact role-slot draw changed.')
    bind(CACHE/'matched-draw.parquet',draw_sha);pd.testing.assert_frame_equal(pd.read_parquet(CACHE/'matched-draw.parquet'),draw,check_exact=True)
    close(summary,static['draw_summary'],'matched summary');descriptor=read(OUT/'draw-manifest.json')
    close(descriptor,summary|dict(slots_path=str(CACHE/'matched-draw.parquet'),slots_sha256=draw_sha,source_path=str(draw_path)),'draw descriptor')
    scaler_path=BASE/'systemness-head-probes/standardization.npz';scaler_sha=bind(scaler_path,static['standardization_sha256'])
    with np.load(scaler_path,allow_pickle=False) as z:mean,scale=z['mean'].copy(),z['scale'].copy()
    require(mean.shape==scale.shape==(512,) and mean.dtype==scale.dtype==np.float32 and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Frozen old scaler malformed.')
    base_path=Path(static['base_model_path']);bind(base_path,CURRENT_SHA);_,base=model_arrays(base_path,mean,scale)
    require(static['base_model_sha256']==sig['base_model_sha256']==CURRENT_SHA and weight_state_sha(base,True)==static['embedded_base_weight_sha256']==sig['embedded_base_weight_sha256']
        and array_sha(mean)==static['mean_sha256'] and array_sha(scale)==static['scale_sha256'],'Frozen base/scaler identity changed.')
    initial,init=initial_binary_state(base);require(init==static['initial_weight_sha256']==plan['initial_weight_sha256'],'Full base+correction initial state changed.')
    probs={};budgets={};model_hashes={};replays={};frozen_training={}
    def load_ps(name,hashes):
        require(set(hashes)==set(rows),'Probability bank coverage changed.')
        return {b:p_read(CACHE/f'{name}-{b}-validation.npz',hashes[b],frame) for b,frame in rows.items()}
    def expected_meta(arm):
        return dict(kind='binary-residual',arm=arm,dimensions=512,base_hidden=128,residual_hidden=32,delta_formula='4*tanh(g)',roles=ROLES,
            base_model_sha256=CURRENT_SHA,embedded_base_weight_sha256=static['embedded_base_weight_sha256'],mean_sha256=static['mean_sha256'],scale_sha256=static['scale_sha256'],
            selected_head_ids=sig['selected_head_ids'],signature=sig,source_sha256=SOURCE_SHA,plan_sha256=plan_sha,initial_weight_sha256=init,standardization_sha256=scaler_sha,tests_read=False)
    def replay(name,model_path,model_sha,raw_proof,metadata):
        model_hashes[name]=bind(model_path,model_sha);_,bb,rr=binary_model_arrays(model_path,base,mean,scale,metadata);proofs={}
        for b,frame in rows.items():
            computed,oldp,inv=residual_replay(bb,rr,mean,scale,banks[b],frame);err=float(np.abs(computed-probs[name][b]).max())
            require(err<5e-6,'Independent full binary portable replay mismatch: '+name+'/'+b)
            compare_raw_evidence(raw_proof[b],inv,name+'/'+b)
            proofs[b]=dict(tokens=len(frame),maximum_probability_difference=err,independent_probability_sha256=array_sha(computed),saved_probability_sha256=array_sha(probs[name][b]),
                base_probability_sha256=array_sha(oldp),raw_invariance=inv)
        replays[name]=proofs;return rr
    for arm in ['new','matched']:
        cp=OUT/'candidates'/f'{arm}.json';bind(cp,complete['candidate_sha256'][arm]);candidate=read(cp);tp=OUT/'training'/f'{arm}.json';frozen_training[arm]=bind(tp,candidate['training_sha256']);training=read(tp)
        amount=history_audit(training['history'],training['best_epoch']);budgets[arm]=amount
        for obj in [candidate,training]:
            require(obj['arm']==arm and obj['signature']==sig and obj['plan_sha256']==plan_sha and obj['initial_weight_sha256']==init,'Arm training ownership/source/initialization changed.')
            for k,v in amount.items():close(obj[k],v,arm+'/budget/'+k)
        require(candidate['source_sha256']==SOURCE_SHA and candidate['tests_read'] is False and candidate['model_sha256']==training['model_sha256']
            and candidate['gpu_old_probability_sha256']==training['gpu_old_probability_sha256'] and training['new_C4_predictions_read'] is False
            and training['public_validation_predictions_read'] is False and training['embedded_base_weight_sha256']==static['embedded_base_weight_sha256']
            and training['base_eval_all_epochs'] is True and training['base_gradient_observations']==0 and training['optimizer_only_residual'] is True
            and candidate['cuda_visible_devices'] in ['0','1'] and 'A40' in candidate['cuda_device_name'],'Frozen-base/device/best-before-new evidence changed.')
        parity_proof(candidate['portable_evidence'],nf);probs[arm]=load_ps(arm,candidate['probability_sha256'])
        gp=p_read(CACHE/f'{arm}-gpu-old-validation.npz',candidate['gpu_old_probability_sha256'],rows['old']);tau=legacy_mean(rows['old'],gp)
        oldvalue=evaluate(rows,dict(probs[arm],old=gp),tau);close(old_validation_record(oldvalue,rows['old']),candidate['old_early_stop_validation'],arm+'/GPUold')
        close(oldvalue['old5_mean_f1'],training['history'][training['best_epoch']-1]['old5_mean_f1'],arm+'/best/history')
        driftproof=probability_drift(gp,probs[arm]['old'],CURRENT_TAU);close(driftproof,candidate['full_old_GPU_CPU_drift'],arm+'/GPUCPUdrift')
        require(driftproof['maximum_probability_difference']<5e-4,'Candidate full GPUold/CPUold drift exceeds gate.')
        replay(arm,OUT/'models'/f'{arm}.npz',candidate['model_sha256'],candidate['raw_invariance'],expected_meta(arm)|dict(best_epoch=amount['best_epoch'],best_steps=amount['best_steps']))
    controls=read(OUT/'controls.json')
    require(controls['signature']==sig and controls['prior_actual_audit_sha256']==PRIOR_AUDIT_SHA and controls['reference_probabilities']==static['reference_probabilities']
        and controls['reference_probabilities_recomputed'] is False and controls['naturalCE_diagnostic_only'] is True and controls['tests_read'] is False,'Inherited reference controls changed.')
    require(controls['physical_model_sha256']==dict(oldhead=CURRENT_SHA,ensemble=static['prior']['model_sha256'],naturalCE='204f4a9dbfb7855460ccb210ca703aca721b974a87059be9134f55c4f7b5cba5'),'Inherited physical model identities changed.')
    for name,bindings_ in static['reference_probabilities'].items():
        require(name in {'oldhead','ensemble','naturalCE'} and set(bindings_)==set(rows),'Inherited model/bank scope changed.')
        probs[name]={}
        for b,item in bindings_.items():
            require(item['sample_ix_sha256']==array_sha(rows[b].sample_ix.to_numpy()),'Inherited reference row identity changed.')
            path=str(Path(item['path']).resolve());require(prior['probability_bindings'][path]==item['sha256'],'Reference not bound by prior actual audit.')
            probs[name][b]=p_read(path,item['sha256'],rows[b])
    require(set(probs)=={'new','matched','oldhead','ensemble','naturalCE'},'Missing three inherited physical controls.')
    prior_controls=read(prior_out/'controls.json');members=[]
    for seed in MEMBERS:
        h=prior_controls['members'][str(seed)]['probability_sha256'];member={b:p_read(prior_out.parent/'systemness-expansion/cache/natural-fit-probes'/f'seed-{seed}-{b}-validation.npz',h[b],rows[b]) for b in rows};members.append(member)
        model_hashes['seed-'+str(seed)]=bind(static['prior']['model_paths'][str(seed)],prior_controls['model_sha256'][str(seed)])
    require(all(np.array_equal(fixed_mean([m[b] for m in members]),probs['ensemble'][b]) for b in rows),'Inherited fixed4seed ensemble arithmetic changed.')
    require(all(np.array_equal(members[0][b],probs['oldhead'][b]) for b in rows),'Inherited oldhead member alias changed.')
    ce_sha='204f4a9dbfb7855460ccb210ca703aca721b974a87059be9134f55c4f7b5cba5';bind(static['naturalCE_model_path'],ce_sha)
    model_hashes['oldhead']=CURRENT_SHA;model_hashes['naturalCE']=ce_sha
    require(static['naturalCE_model_sha256']==ce_sha==prior['model_sha256']['new'],'Prior CE model proof changed.')
    original_refs=static['old_reference'];legacy_p={b:p_read(original_refs[k+'_path'],original_refs[k+'_sha256'],rows[b]) for b,k in [('old','old'),('c4','new')]}
    legacy=evaluate(rows,dict(legacy_p,public=probs['oldhead']['public']),CURRENT_TAU);close(controls['original_savedGPU_validation'],legacy,'inherited originalGPU floors')
    close(prior_controls['original_savedGPU_validation'],legacy,'prior originalGPU floors')
    zero=read(OUT/'zero-controls.json');require(zero['signature']==sig and zero['both_arm_training_sha256']==frozen_training and zero['original_threshold']==CURRENT_TAU
        and zero['trained_record'] is False and zero['tests_read'] is False and zero['model_path']==str(OUT/'models/zero.npz'),'Zero control closure/bestfreeze changed.')
    parity_proof(zero['portable_evidence'],nf);probs['zero']=load_ps('zero',zero['probability_sha256'])
    zr=replay('zero',OUT/'models/zero.npz',zero['model_sha256'],zero['raw_invariance'],expected_meta('zero')|dict(zero_residual=True))
    require(np.array_equal(zr['encoder.0.weight'],initial['correction.0.weight']) and np.array_equal(zr['encoder.0.bias'],initial['correction.0.bias'])
        and not np.any(zr['head.weight']) and not np.any(zr['head.bias']),'Zero initial residual weights changed.')
    for b in rows:
        inv=replays['zero'][b]['raw_invariance'];require(inv['delta_min']==inv['delta_max']==inv['delta_mean']==0,'Zero delta is nonzero.')
        d=probability_drift(probs['oldhead'][b],probs['zero'][b],CURRENT_TAU);close(d,zero['oldhead_commonCPU_drift'][b],'zero/drift/'+b)
        require(d['maximum_probability_difference']<5e-4,'Zero samepath oldhead drift exceeds gate.')
    ztau,zproof,zannex=direct_grid(rows,probs['zero']);zero_records=[dict(record='zero-oldmean',validation=evaluate(rows,probs['zero'],CURRENT_TAU)),dict(record='zero-max33',validation=evaluate(rows,probs['zero'],ztau))]
    close(zero['records'],{r['record']:r['validation'] for r in zero_records},'zero/records');close(zero['max33_full_grid_proof'],zproof,'zero/fullgrid')
    saved=read(OUT/'records.json');index={r['record']:r for r in saved};require(len(saved)==10 and set(index)==set(RECORDS),'Ten formal records missing.')
    computed=[];grid_proofs={'zero':dict(proof=zproof,annex=zannex)}
    for name in RECORDS:
        model,objective=name.split('-');p=probs[model]
        if objective=='max33':tau,proof,annex=direct_grid(rows,p);grid_proofs[model]=dict(proof=proof,annex=annex)
        elif model=='oldhead':tau,proof=CURRENT_TAU,None
        elif model=='ensemble':tau,proof=ENSEMBLE_TAU,None
        elif model=='naturalCE':tau,proof=.7772222222222223,None
        else:tau,proof=legacy_mean(rows['old'],p['old']),None
        if name=='naturalCE-max33':require(tau==.6008080808080808,'Inherited CE max33 threshold changed.')
        record=dict(record=name,model=model,objective=objective,validation=evaluate(rows,p,tau),full_grid_proof=proof,diagnostic=name!='new-max33',tests_read=False)
        close(index[name],record,name);computed.append(record)
    selected=selection(computed,zero_records,legacy);published=read(OUT/'selection.json')
    for k,v in selected.items():close(published[k],v,'selection/'+k)
    require(published['source_sha256']==SOURCE_SHA and published['plan_sha256']==plan_sha and published['records_sha256']==complete['artifact_sha256']['records.json']
        and published['controls_sha256']==complete['artifact_sha256']['controls.json'] and published['zero_controls_sha256']==complete['artifact_sha256']['zero-controls.json']
        and published['model_path']==str(OUT/'models/new.npz') and published['model_sha256']==model_hashes['new']
        and published['system_threshold']==index['new-max33']['validation']['system_threshold'],'Published selection identity changed.')
    verify_csv(pd.read_csv(OUT/'validation-metrics.csv'),computed);verify_csv(pd.read_csv(OUT/'zero-control-metrics.csv'),zero_records,True)
    counts=read(OUT/'source-counts.json');previous_counts=read(prior_out/'source-counts.json')
    for k,v in previous_counts.items():
        if k!='limitation':close(counts[k],v,'source-counts/'+k)
    require(counts['experiment']=='binary-residual' and counts['same_existing_rows'] is True and counts['inherited_naturalCE_diagnostic_only'] is True
        and counts['per_epoch']==641493 and counts['steps']==157 and counts['last_batch']==2517,'Source/update counts changed.')
    source_counts={g:dict(bank=b,tokens=len(ix),sources=int(rows[b].iloc[ix].group_id.nunique()),role_counts={r:int(rows[b].iloc[ix].role.eq(r).sum()) for r in ROLES},sample_ix_sha256=array_sha(rows[b].iloc[ix].sample_ix.to_numpy())) for g,(b,ix) in group_indices(rows).items()}
    terminal_gate()
    for path,want in bindings.items():require(digest(path)==want,'Bound input changed during audit: '+path)
    report=dict(stage='actual_completed_binary_residual_development_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=digest(__file__),
        trainer_source_sha256=SOURCE_SHA,trainer_complete_sha256=complete_sha,trainer_protocol_sha256=PROTOCOL_SHA,
        trainer_selection_sha256=bindings[str((OUT/'selection.json').resolve())],trainer_records_sha256=bindings[str((OUT/'records.json').resolve())],
        trainer_model_record_sha256={a:bindings[str((OUT/'candidates'/f'{a}.json').resolve())] for a in ['new','matched']},
        prior_actual_audit_sha256=PRIOR_AUDIT_SHA,model_sha256=model_hashes,provenance_sha256=bindings,probability_bindings=probability_bindings,
        ten_records=computed,zero_records=zero_records,selection=selected,independent_grid_proofs=grid_proofs,source_counts=source_counts,
        training_budgets=budgets,full_independent_CPU_predictions=replays,maximum_metric_numeric_difference=MAX_DIFFERENCE,
        csv_primary_group_rows=110,csv_descriptive_pool_rows=10,zero_csv_primary_group_rows=22,zero_csv_pool_rows=2,
        original_savedGPU_NPZ_and_threshold_unchanged=True,fixed_four_member_ensemble=True,only_new_max33_eligible=True,
        test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False,reused_development_validation=True,new_independent_confirmation_required=True)
    output=Path(output);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(output),report_sha256=digest(output),source_sha256=report['auditor_source_sha256'],selection=selected,training_budgets=budgets)),flush=True)


def synthetic():
    frames={b:[] for b in ['old','c4','public']}
    for p_ix,profile in enumerate(PROFILES):
        for r_ix,role in enumerate(ROLES):
            if profile=='natural' and role=='tool':continue
            for offset in [0,341,682,1023]:
                frames['old'].append(dict(profile=profile,role=role,token_ix=17+offset,prompt_ix=p_ix*5+r_ix,
                    message_ix=0,document_id=p_ix,group_id='old-'+str(p_ix),source_set='fixture',cache_ix=len(frames['old']),sample_ix=len(frames['old'])))
    for r_ix,role in enumerate(ROLES):
        for offset in [0,341,682,1023]:
            frames['c4'].append(dict(profile='long-full',role=role,token_ix=29+offset,body_offset=offset,prompt_ix=100+r_ix,
                message_ix=0,document_id=6000000,group_id='new',source_set='long-augmentation',cache_ix=len(frames['c4']),sample_ix=6_000_000_000_000+len(frames['c4'])))
    for d_ix,dataset in enumerate(['oasst','toxicchat0124']):
        for role in ['system','user','assistant']:
            for offset in range(4):frames['public'].append(dict(role=role,dataset=dataset,source_uid=f'{dataset}:{d_ix}',
                cache_ix=len(frames['public']),sample_ix=4_000_000_000_000+len(frames['public'])))
    rows={b:pd.DataFrame(v) for b,v in frames.items()};p={}
    for b,f in rows.items():
        y=labels(f);value=np.full((len(y),5),.02,np.float32);value[np.arange(len(y)),y]=.92
        for i in np.flatnonzero(y==0)[::3]:value[i]=np.array([.50000006,.124999985,.124999985,.124999985,.124999985],np.float32)
        p[b]=value
    return rows,p


def preflight():
    import torch
    from sklearn.metrics import roc_auc_score,average_precision_score
    torch.set_num_threads(2);folder=Path(tempfile.mkdtemp(prefix='long-data-binary-residual-audit-preflight-',dir='/tmp'));before=digest(__file__);checks=[]
    def check(name,value=True):require(bool(value),'CPU fixture: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,AssertionError,KeyError,ValueError):checks.append(name);return
        raise RuntimeError('Fault accepted: '+name)
    rows,p=synthetic();tau,proof,annex=direct_grid(rows,p);v=evaluate(rows,p,tau)
    check('11groups33_direct_F32_grid',len(v['groups'])==11 and proof['primary_statistics']==33*proof['grid_thresholds'] and v['min33']==proof['min33'])
    y=labels(rows['old']);close(v['groups']['old-natural']['system_auroc'],roc_auc_score(labels(rows['old'][rows['old'].profile.eq('natural')])==0,p['old'][rows['old'].profile.eq('natural'),0]),'fixture/AUC')
    close(metrics(p['old'],y,tau)['system_average_precision'],average_precision_score(y==0,p['old'][:,0]),'fixture/AP')
    check('independent_tied_rank_metrics_and_legacy_CoT',v['groups']['old-natural']['recall']['cot'] is not None and all(v['groups'][g]['recall']['cot'] is None and v['groups'][g]['recall']['tool'] is None for g in PUBLIC))
    rng=np.random.default_rng(20516085);raw=(rng.normal(size=(8,512))*.15).astype(np.float16)
    f=pd.DataFrame(dict(cache_ix=[7,1,6,2],role=['system','cot','assistant','tool']))
    mean=np.full(512,.01,np.float32);scale=np.full(512,.7,np.float32)
    base={'encoder.0.weight':(rng.normal(size=(128,512))*.02).astype(np.float32),'encoder.0.bias':np.zeros(128,np.float32),
        'head.weight':(rng.normal(size=(5,128))*.02).astype(np.float32),'head.bias':np.array([-.2,-.1,.3,.05,-.5],np.float32)}
    residual={'encoder.0.weight':(rng.normal(size=(32,512))*.02).astype(np.float32),'encoder.0.bias':np.zeros(32,np.float32),
        'head.weight':np.zeros((1,32),np.float32),'head.bias':np.zeros(1,np.float32)}
    q,old,zero=residual_replay(base,residual,mean,scale,raw,f)
    check('zero_delta_fullp_bitexact_and_rawrestSHA',np.array_equal(q.view(np.uint32),old.view(np.uint32)) and zero['raw_non_system_bitexact'] and zero['conditional_raw_argmax_changes']==zero['p_non_system_different_class_swaps']==0 and zero['maximum_abs_delta']==0)
    plain=independent_predict(base,mean,scale,raw,f);check('zero_same_original_numerical_path',np.array_equal(plain,q))
    state,init=initial_binary_state(base);again,init2=initial_binary_state(base)
    check('deterministic_full_initial_hash_and_zero_final',init==init2 and all(np.array_equal(v,again[k]) for k,v in state.items())
        and not np.any(state['correction.3.weight']) and not np.any(state['correction.3.bias']))
    meta=dict(kind='binary-residual',dimensions=512,base_hidden=128,residual_hidden=32,delta_formula='4*tanh(g)',roles=ROLES,
        base_model_sha256=CURRENT_SHA,embedded_base_weight_sha256=weight_state_sha(base,True),mean_sha256=array_sha(mean),scale_sha256=array_sha(scale))
    portable=folder/'synthetic-model.npz'
    def save_toy(current=state,metadata=meta):
        np.savez(portable,mean=mean,scale=scale,metadata=np.array(json.dumps(metadata)),**{'weight::'+k:v for k,v in current.items()})
    save_toy();_,bb,rr=binary_model_arrays(portable,base,mean,scale)
    check('portable_selfcontained_base_residual_scaler_roundtrip',all(np.array_equal(bb[k].view(np.uint32),base[k].view(np.uint32)) for k in base) and np.all(rr['head.weight']==0))
    malformed=copy.deepcopy(state);malformed['base.head.bias'][0]=np.nextafter(malformed['base.head.bias'][0],np.float32(1))
    save_toy(malformed);reject('portable_rejects_mutated_frozen_base',lambda:binary_model_arrays(portable,base,mean,scale))
    save_toy(state,dict(meta,delta_formula='4*tanh(g/4)'));reject('portable_rejects_division_inside_tanh',lambda:binary_model_arrays(portable,base,mean,scale))
    save_toy()
    residual['head.bias']=np.array([.3],np.float32);q,old,invariance=residual_replay(base,residual,mean,scale,raw,f)
    check('delta4tanh_no_div4',abs(invariance['maximum_abs_delta']-float(4*torch.tanh(torch.tensor(.3))))<1e-6 and abs(invariance['maximum_abs_delta']-float(4*torch.tanh(torch.tensor(.3/4))))>.5)
    check('nonzero_delta_rawrest_bitexact_conditional_fixed',invariance['raw_non_system_bitexact'] and invariance['conditional_raw_argmax_changes']==0 and invariance['p_non_system_different_class_swaps']==0)
    broken=copy.deepcopy(invariance);broken['five_class_transition']=[[0]*5 for _ in range(5)];broken['five_class_transition'][2][3]=1
    reject('actual_consumer_non_system_swap_fails_before_savedproof',lambda:compare_raw_evidence({},broken,'synthetic-swap'))
    check('full5_transition_truth_role_counts',sum(sum(r) for r in invariance['five_class_transition'])==4 and sum(sum(sum(r) for r in m) for m in invariance['transitions_by_true_role'].values())==4)
    # Corrected five-class CE and binary BCE differ only by a constant when
    # all non-system logits are frozen; verify exact parameter gradients.
    logits=torch.randn(19,5,dtype=torch.float64);target=torch.arange(19)%5;g=torch.linspace(-.3,.4,19,dtype=torch.float64,requires_grad=True)
    delta=4*torch.tanh(g);fixed=logits.clone();fixed[:,0]+=delta
    ce=torch.nn.functional.cross_entropy(fixed,target);gc=torch.autograd.grad(ce,g,retain_graph=True)[0]
    binary=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0]+delta-torch.logsumexp(logits[:,1:],1),(target==0).double());gb=torch.autograd.grad(binary,g)[0]
    check('frozenconditional_BCE_CE_gradient_equivalence',float((gc-gb).abs().max())<1e-14)
    def result(score):return dict(min33=score,old5_mean_f1=score,groups={g:dict(system_f1=score,accuracy=score,system_false_positive_rate=.01) for g in GROUPS})
    records=[dict(record=n,validation=result(.93 if n=='new-max33' else .9)) for n in RECORDS];zeros=[dict(record=n,validation=result(.9)) for n in ZERO_RECORDS]
    check('10records_zero2_4strict_7guards_GPU',selection(records,zeros,result(.9))['promoted'])
    ce=copy.deepcopy(records)
    for record in ce:
        if record['record'].startswith('naturalCE-'):record['validation']=result(.99)
    check('naturalCE_strongdiagnostic_not_hardguard',selection(ce,zeros,result(.9))['promoted'])
    for name in STRICT_CONTROLS:
        rr,zz=copy.deepcopy(records),copy.deepcopy(zeros)
        next(r for r in rr+zz if r['record']==name)['validation']['min33']=.93
        check('strict_tie_rejected_'+name,not selection(rr,zz,result(.9))['promoted'])
    bad=copy.deepcopy(records);next(r for r in bad if r['record']=='new-max33')['validation']['min33']=.89999999999
    check('absolute90_floor',not selection(bad,zeros,result(.8))['promoted'])
    for name in GUARD_CONTROLS:
        rr,zz=copy.deepcopy(records),copy.deepcopy(zeros)
        next(r for r in rr+zz if r['record']==name)['validation']['groups']['old-natural']['accuracy']=.99
        check('seven_hardguards_'+name,not selection(rr,zz,result(.9))['promoted'])
    check('mandatory_original_GPU_guard',not selection(records,zeros,result(.99))['promoted'])
    allrecords=[dict(record=n,validation=v) for n in RECORDS];zero_records=[dict(record=n,validation=v) for n in ZERO_RECORDS]
    def flat(entries):
        out=[]
        for r in entries:
            for g,m in dict(r['validation']['groups'],pooled_descriptive_only=r['validation']['pooled_descriptive_only']).items():out.append(dict(record=r['record'],group=g,**{k:x for k,x in m.items() if not isinstance(x,(dict,list))},**{'role_recall_'+k:x for k,x in m['recall'].items()}))
        return pd.DataFrame(out)
    main,zcsv=flat(allrecords),flat(zero_records);verify_csv(main,allrecords);verify_csv(zcsv,zero_records,True)
    check('120mainCSV_and24zeroCSV_NA_roles',len(main)==120 and len(zcsv)==24)
    wrong=zcsv.copy();wrong.loc[0,'system_f1']=.1;reject('zeroCSV_tamper',lambda:verify_csv(wrong,zero_records,True))
    h=[dict(epoch=e,steps=157,instances=641493,cumulative_steps=e*157,loss=.5,old5_mean_f1=.8) for e in range(1,11)]
    amount=history_audit(h,1);check('same_budget_best157_actual1570_last2517',amount['best_steps']==157 and amount['actual_steps']==1570 and amount['last_batch_instances']==2517)
    with patch(__name__+'.SOURCE_SHA','fake'),patch(__name__+'.PROTOCOL_SHA','fake'),patch(__name__+'.processes',return_value=[123]),patch(__name__+'.read',side_effect=AssertionError('Unexpected real result read')):
        reject('actual_live_gate_before_reads',lambda:actual(folder/'forbidden.json'))
    with patch(__name__+'.PROTOCOL_SHA',None):reject('actual_unpinned_forbidden',lambda:actual(folder/'forbidden.json'))
    check('source_unchanged',before==digest(__file__))
    report=dict(stage='synthetic_binary_residual_auditor_preflight',source_sha256=before,audit_passed=True,checks=checks,check_count=len(checks),
        final_trainer_source_pinned=SOURCE_SHA,protocol_pinned=PROTOCOL_SHA,design_sha256=DESIGN_SHA,fullgrid=proof,
        zero_invariance=zero,nonzero_invariance=invariance,maximum_metric_difference=MAX_DIFFERENCE,
        actual_probabilities_or_models_read=False,test_data_read=False,GPU_used=False,formal_outputs_written=False,actual_artifact_chain_ready=True)
    path=folder/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(path),report_sha256=digest(path),**report)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true');action.add_argument('--actual',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('/tmp/long_data_binary_residual_results_audit.json'));args=parser.parse_args()
    if args.preflight:preflight()
    elif args.actual:actual(args.output)
    else:parser.print_help()
