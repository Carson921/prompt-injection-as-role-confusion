"""Independent natural-fit development results auditor.

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
OUT=BASE/'systemness-natural-fit-probes';CACHE=BASE/'systemness-expansion/cache/natural-fit-probes'
SOURCE=HERE/'systemness_natural_fit_probes.py'
SOURCE_SHA='7c0674adbce3ef4980a1f29077a6ca36396b93aa992c32ad848ff9e56be4aea3'
PROTOCOL_SHA='1edc202c1f8f2651e730a650590489c5a8f1f8da1b9da4ab730f0b949b59c104'
DESIGN=BASE/'systemness-natural-fit-preparation/root-prospective-design.json'
DESIGN_SHA='9e8cae74eef9317265cfe4417b807d856d9d2332ca081695780e657629c39f40'
CURRENT_SHA='27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
CURRENT_TAU=.620969696969697;ENSEMBLE_TAU=.4747979797979798
ROLES=['system','user','cot','assistant','tool']
PROFILES=['single','multi','natural','long-single','long-full']
OLD5=['old-'+p for p in PROFILES];ORIGINAL=OLD5[:4]
PUBLIC=['public-natural-all','public-natural-oasst','public-natural-toxicchat0124']
FULL=['old-long-full','old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']
GROUPS=OLD5+['old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']+PUBLIC
FEASIBLE=OLD5+['new-C4-long-full']+PUBLIC
RECORDS=['new-oldmean','new-max33','matched-oldmean','matched-max33','oldhead-oldmean','oldhead-max33','ensemble-original','ensemble-max33']
GUARD_CONTROLS=['matched-max33','oldhead-max33','ensemble-max33','oldhead-oldmean','ensemble-original']
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

def portable_check(weights, mean, scale, bank, rows, p):
    import torch
    chosen = np.linspace(0, len(rows) - 1, 32, dtype=int)
    value = torch.from_numpy((np.asarray(bank[rows.iloc[chosen].cache_ix.to_numpy()], dtype=np.float32) - mean) / scale)
    with torch.inference_mode():
        hidden = torch.nn.functional.gelu(torch.nn.functional.linear(value, torch.from_numpy(weights['encoder.0.weight']), torch.from_numpy(weights['encoder.0.bias'])))
        actual = torch.softmax(torch.nn.functional.linear(hidden, torch.from_numpy(weights['head.weight']), torch.from_numpy(weights['head.bias'])), dim=1).numpy()
    difference = float(np.abs(actual - p[chosen]).max()); require(difference < 5e-4, 'Independent raw-F16/scaler/weights probability check failed.')
    return dict(sampled_tokens=32, maximum_difference=difference)


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


def selection(records,legacy):
    require(len(records)==8 and {r['record'] for r in records}==set(RECORDS),'All eight fixed records required.')
    index={r['record']:r for r in records};candidate=index['new-max33']['validation'];minimum=candidate['min33'];reasons=[];guards={};gains={}
    if minimum is None or not np.isfinite(minimum) or minimum<.9:reasons.append('absolute_min33_below_90_percent')
    for name in GUARD_CONTROLS:
        control=index[name]['validation'];g=guard(candidate,control);guards[name]=g;reasons.extend(name+':'+x for x in g['rejection_reasons'])
        if name in GUARD_CONTROLS[:3]:
            gain=minimum-control['min33'] if minimum is not None and control['min33'] is not None else None;gains[name]=gain
            if gain is None or not np.isfinite(gain) or gain<=1e-12:reasons.append(name+':min33_not_strictly_improved')
    saved=guard(candidate,legacy,True);guards['original-savedGPU']=saved;reasons.extend('original-savedGPU:'+x for x in saved['rejection_reasons'])
    return dict(candidate='new-max33',promoted=not reasons,no_fallback=True,min33_gains=gains,guards=guards,rejection_reasons=reasons,
        primary_changed=False,tests_read=False,new_independent_confirmation_required=True)


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


def initialization_sha():
    import torch
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Sequential(torch.nn.Linear(512,128),torch.nn.GELU(),torch.nn.Dropout(.1));self.head=torch.nn.Linear(128,5)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261021);net=Net()
        values=[dict(name=n,shape=list(v.shape),dtype=str(v.numpy().dtype),sha256=array_sha(v.numpy())) for n,v in net.state_dict().items()]
    return canonical(values)


def verify_csv(frame,records):
    expected={(r['record'],g):v for r in records for g,v in dict(r['validation']['groups'],pooled_descriptive_only=r['validation']['pooled_descriptive_only']).items()}
    require(len(frame)==96 and not frame.duplicated(['record','group']).any() and set(zip(frame.record,frame.group))==set(expected),'CSV must contain88 formal+8 pooled rows.')
    for row in frame.to_dict('records'):
        want=expected[(row['record'],row['group'])]
        for k,v in want.items():
            if isinstance(v,(dict,list)):continue
            actual=None if pd.isna(row[k]) else row[k];close(actual,v,'csv/'+row['record']+'/'+row['group']+'/'+k)
        for role in ROLES:
            actual=row['role_recall_'+role];close(None if pd.isna(actual) else actual,want['recall'][role],'csv/recall/'+role)


def processes():
    names={'systemness_natural_fit_probes.py','systemness_natural_dev_extract.py'};active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(v.decode(errors='replace')).name in names for v in args if v):active.append(int(entry.name))
    return active


def terminal_gate():
    require(not processes(),'Trainer/extractor live: no partial results or probability reads.')
    require((OUT/'complete.json').is_file() and (OUT/'status.json').is_file(),'Formal completion/status absent.')
    require(read(OUT/'status.json')['stage']=='validation_complete','Trainer has not completed.')


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


def actual(output):
    # Fail closed while trainer is still under development. Root pins source and
    # formal metadata protocol only after independent source review.
    require(SOURCE_SHA is not None and PROTOCOL_SHA is not None,'Final trainer/protocol not pinned: actual forbidden.')
    terminal_gate()
    require(Path(output).resolve().is_relative_to(Path('/tmp')),'Independent audit writes only /tmp.')
    bindings={};probability_bindings={}
    def bind(path,want=None):
        p=Path(path);require(p.is_file(),'Missing bound artifact: '+str(p));value=digest(p)
        if want is not None:require(value==want,'Byte SHA mismatch: '+str(p))
        bindings[str(p.resolve())]=value;return value
    bind(__file__);bind(SOURCE,SOURCE_SHA);bind(DESIGN,DESIGN_SHA);bind(OUT/'protocol.json',PROTOCOL_SHA)
    complete=read(OUT/'complete.json');complete_sha=bind(OUT/'complete.json');bind(OUT/'status.json')
    plan=read(OUT/'search-plan.json');plan_sha=bind(OUT/'search-plan.json');declaration=read(OUT/'protocol.json');static=declaration['static'];signature=plan['signature']
    artifacts={'protocol.json','search-plan.json','draw-manifest.json','controls.json','records.json','selection.json','validation-metrics.csv','source-counts.json','report.txt'}
    require(complete['source_sha256']==SOURCE_SHA and complete['signature']==signature and complete['tests_read'] is False
        and complete['primary_changed'] is False and complete['new_independent_confirmation_required'] is True
        and set(complete['artifact_sha256'])==artifacts and set(complete['candidate_sha256'])=={'new','matched'},'Complete artifact contract changed.')
    for name,want in complete['artifact_sha256'].items():bind(OUT/name,want)
    require(plan['protocol']==declaration['protocol'] and signature['source_sha256']==SOURCE_SHA and signature['protocol_sha256']==PROTOCOL_SHA
        and signature['static_sha256']==canonical(static) and static['source_sha256']==SOURCE_SHA and static['design_sha256']==DESIGN_SHA
        and static['validation_probability_arrays_read'] is False and declaration['protocol']['tests_read'] is False
        and declaration['protocol']['primary_changed'] is False,'Declared static source/design/probability boundary changed.')
    protocol=declaration['protocol'];neural=protocol['neural']
    expected_neural=dict(seed=20261021,kind='mlp',hidden=128,dropout=.1,optimizer='AdamW',lr=.001,weight_decay=.001,clip=5.,batch=4096,instances=641493,steps=157,last_batch=2517,max_epochs=45,min_epochs=10,patience=8,improvement=1e-5)
    close(neural,expected_neural,'protocol/neural')
    require(protocol['records']==RECORDS and protocol['groups']==GROUPS and protocol['feasible']==FEASIBLE and protocol['max_FPR']==.05
        and protocol['CPU']==dict(threads=4,float32_matmul_precision='highest',dtype='float32',chunk=4096)
        and protocol['ensemble']['members']==MEMBERS and protocol['ensemble']['weights']==[.25]*4
        and protocol['ensemble']['original_tau']==ENSEMBLE_TAU,'Fixed records/group/CPU/ensemble protocol changed.')
    # Explicit, recursively nested declared maps only: no dynamic rglob inventory
    # and no legacy or current production completion verifier.
    def declared_maps(value):
        if not isinstance(value,dict):return
        for key,item in value.items():
            if key in {'metadata_sha256','dependencies_sha256'}:
                for name,want in item.items():bind(Path(name) if Path(name).is_absolute() else HERE/name,want)
            elif key in {'base','original'}:declared_maps(item)
    declared_maps(static)
    require(static['packages']=={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','pyarrow','scikit-learn']},'Numerical packages changed.')
    upstream=signature['upstream'];paths={'old':BASE/'systemness-head-features','c4':BASE/'systemness-long-augmentation'}
    oldcomplete=read(paths['old']/'extraction-complete.json');c4complete=read(paths['c4']/'extraction-complete.json')
    nc=read(BASE/'systemness-natural-dev-features/extraction-complete.json')
    bind(paths['old']/'extraction-complete.json',upstream['old_extraction_sha256']);bind(paths['c4']/'extraction-complete.json',upstream['new_extraction_sha256'])
    bind(paths['c4']/'data-complete.json',upstream['new_data_sha256']);bind(BASE/'systemness-natural-dev-features/extraction-complete.json',signature['natural_complete_sha256'])
    close(c4complete['signature'],upstream['new_extraction_signature'],'C4 extraction signature');close(nc['signature'],signature['natural_signature'],'natural extraction signature')
    require(signature['selected_head_ids']==oldcomplete['selected_head_ids']==c4complete['selected_head_ids']==nc['selected_head_ids']==[458,462,2045,460,459,2047,456,997]
        and nc['rows']==80116 and nc['dimensions']==512 and nc['all_rows_written'] is True and nc['all_finite'] is True
        and nc['tests_read'] is False and nc['readout_training'] is False,'Extraction geometry/completeness/head order changed.')
    rowtables={};banks={}
    for b,count,record in [('old',755282,oldcomplete),('c4',389120,c4complete),('public',80116,nc)]:
        folder=BANK_PATHS[b];require((folder/'features.f16').stat().st_size==count*512*2,'Raw bank size/header differs.')
        bind(folder/'rows.parquet',record['rows_sha256']);bind(folder/'features.f16',record['features_sha256']);rowtables[b]=pd.read_parquet(folder/'rows.parquet')
        banks[b]=np.memmap(folder/'features.f16',mode='r',dtype=np.float16,shape=(count,512))
    require(upstream['old_features_sha256']==oldcomplete['features_sha256'] and upstream['new_features_sha256']==c4complete['features_sha256']
        and upstream['old_rows_sha256']==oldcomplete['rows_sha256'] and upstream['new_rows_sha256']==c4complete['rows_sha256']
        and signature['natural_features_sha256']==nc['features_sha256'] and signature['natural_rows_sha256']==nc['rows_sha256'],'Plan bank SHA binding changed.')
    fit,rows,nf,partition=source_partitions(rowtables['old'],rowtables['c4'],rowtables['public']);close(partition,static['original_partitions'],'old partitions');close(partition,upstream['old_partitions'],'upstream partitions')
    for name,frame in [('fit',fit),('old',rows['old']),('c4',rows['c4']),('natural-fit',nf),('public',rows['public'])]:
        require(array_sha(frame.sample_ix.to_numpy())==static['sample_ix_sha256'][name],'Frozen sample row identity changed: '+name)
    draw,summary=matched_draw(fit,nf);manifest=read(OUT/'draw-manifest.json');bind(CACHE/'matched-draw.parquet',static['draw_slots_sha256'])
    pd.testing.assert_frame_equal(pd.read_parquet(CACHE/'matched-draw.parquet'),draw,check_exact=True)
    for k,v in summary.items():close(manifest[k],v,'matched/'+k)
    require(bind(OUT/'draw-manifest.json')==static['draw_manifest_sha256'] and manifest['slots_path']==str(CACHE/'matched-draw.parquet')
        and manifest['slots_sha256']==static['draw_slots_sha256'],'Matched draw identity changed.')
    scaler_path=BASE/'systemness-head-probes/standardization.npz';scaler_sha=bind(scaler_path,signature['standardization_sha256'])
    require(scaler_sha==static['standardization_sha256'],'Old scaler closure differs.')
    with np.load(scaler_path,allow_pickle=False) as z:mean,scale=z['mean'].copy(),z['scale'].copy()
    require(mean.shape==scale.shape==(512,) and mean.dtype==scale.dtype==np.float32 and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Original fit scaler malformed.')
    init=initialization_sha();require(init==plan['initial_weight_sha256']==static['initial_weight_sha256'],'Original common initial weights changed.')
    probability={};models={};budgets={};portable={};candidate_records={}
    def load_ps(name,proof):
        require(set(proof)==set(rows),'Probability bank coverage differs.')
        result={}
        for b,frame in rows.items():
            path=CACHE/f'{name}-{b}-validation.npz';probability_bindings[str(path)]=bind(path,proof[b]);result[b]=read_probability(path,proof[b],frame)
        return result
    def recompute(name,path,expected,metadata=None):
        models[name]=bind(path,expected);meta,weights=model_arrays(path,mean,scale,metadata);checks={}
        require(meta['selected_head_ids']==signature['selected_head_ids'],'Portable selected-head dimension order changed.')
        for b,frame in rows.items():
            computed=independent_predict(weights,mean,scale,banks[b],frame);error=float(np.abs(computed-probability[name][b]).max())
            require(error<5e-6,'Full independent CPU prediction mismatch: '+name+'/'+b)
            checks[b]=dict(tokens=len(frame),maximum_probability_difference=error,independent_prediction_sha256=array_sha(computed),saved_prediction_sha256=array_sha(probability[name][b]))
        portable[name]=checks
    for arm in ['new','matched']:
        path=OUT/'candidates'/f'{arm}.json';bind(path,complete['candidate_sha256'][arm]);record=read(path);candidate_records[arm]=record
        tp=OUT/'training'/f'{arm}.json';bind(tp,record['training_sha256']);training=read(tp);amounts=history_audit(training['history'],training['best_epoch']);budgets[arm]=amounts
        for e in [record,training]:
            require(e['arm']==arm and e['signature']==signature and e['plan_sha256']==plan_sha and e['initial_weight_sha256']==init,'Arm/source/plan/init evidence changed.')
            for k,v in amounts.items():close(e[k],v,arm+'/budget/'+k)
        require(record['source_sha256']==SOURCE_SHA and record['tests_read'] is False and training['new_C4_predictions_read'] is False
            and training['public_validation_predictions_read'] is False and record['model_sha256']==training['model_sha256']
            and record['gpu_old_probability_sha256']==training['gpu_old_probability_sha256']
            and record['cuda_visible_devices'] in ['0','1'] and 'A40' in record['cuda_device_name'],'Candidate freeze/read/device evidence changed.')
        parity_proof(record['portable_evidence'],nf);probability[arm]=load_ps(arm,record['probability_sha256'])
        gp=CACHE/f'{arm}-gpu-old-validation.npz';probability_bindings[str(gp)]=bind(gp,record['gpu_old_probability_sha256']);gpu_p=read_probability(gp,record['gpu_old_probability_sha256'],rows['old'])
        gpu_tau=legacy_mean(rows['old'],gpu_p);gpu_values=evaluate(rows,dict(probability[arm],old=gpu_p),gpu_tau)
        close(old_validation_record(gpu_values,rows['old']),record['old_early_stop_validation'],arm+'/GPUold')
        close(gpu_values['old5_mean_f1'],training['history'][training['best_epoch']-1]['old5_mean_f1'],arm+'/best/history')
        error=float(np.abs(gpu_p-probability[arm]['old']).max());close(error,record['full_old_GPU_CPU_max_difference'],arm+'/fullGPUCPU');require(error<5e-4,'Candidate GPU/CPU drift exceeds technical gate.')
        recompute(arm,OUT/'models'/f'{arm}.npz',record['model_sha256'],dict(arm=arm,source_sha256=SOURCE_SHA,signature=signature,
            plan_sha256=plan_sha,initial_weight_sha256=init,best_epoch=amounts['best_epoch'],best_steps=amounts['best_steps'],
            selected_head_ids=signature['selected_head_ids'],standardization_sha256=scaler_sha,tests_read=False))
    control=read(OUT/'controls.json');require(control['signature']==signature and control['tests_read'] is False
        and control['model_sha256']==static['model_sha256'] and control['member_order']==MEMBERS and control['weights']==[.25]*4,'Inherited controls ownership/order changed.')
    for seed in MEMBERS:
        key=str(seed);name='seed-'+key;record=control['members'][key];parity_proof(record['portable_evidence'],nf)
        require(record['model_sha256']==static['model_sha256'][key],'Inherited member model changed.')
        probability[name]=load_ps(name,record['probability_sha256']);recompute(name,Path(static['model_paths'][key]),record['model_sha256'])
    require(static['model_sha256']['20261021']==CURRENT_SHA,'Original head identity changed.')
    probability['oldhead']=probability['seed-20261021'];models['oldhead']=CURRENT_SHA
    ensemble={b:fixed_mean([probability['seed-'+str(s)][b] for s in MEMBERS]) for b in rows}
    probability['ensemble']=load_ps('ensemble',control['ensemble_probability_sha256'])
    require(all(np.array_equal(ensemble[b],probability['ensemble'][b]) for b in rows),'Fixed four-member arithmetic changed.')
    def reference(spec):
        result={}
        for b,k in [('old','old'),('c4','new')]:
            path=Path(spec[k+'_path']);probability_bindings[str(path)]=bind(path,spec[k+'_sha256']);result[b]=read_probability(path,spec[k+'_sha256'],rows[b])
        return result
    ref=reference(static['old_reference']);eref=reference(static['ensemble_reference'])
    close(control['original_savedGPU_reference'],static['old_reference'],'oldGPU reference');close(control['inherited_ensemble_reference'],static['ensemble_reference'],'ensemble reference')
    close(control['oldhead_drift'],drift(rows,ref,probability['oldhead'],CURRENT_TAU),'oldhead full drift')
    close(control['ensemble_drift'],drift(rows,eref,ensemble,ENSEMBLE_TAU),'ensemble full drift')
    legacy=evaluate(rows,dict(ref,public=probability['oldhead']['public']),CURRENT_TAU);close(control['original_savedGPU_validation'],legacy,'mandatory savedGPU')
    original=read(BASE/'systemness-head-probes/candidates/joint-mlp128.json');bind(BASE/'systemness-head-probes/candidates/joint-mlp128.json')
    require(original['model_sha256']==CURRENT_SHA and original['validation_probabilities_sha256']==static['old_reference']['old_sha256']
        and legacy_mean(rows['old'],ref['old'])==CURRENT_TAU,'Original savedGPU identity/calibration altered.')
    close(original['validation'],old_validation_record(legacy,rows['old']),'original legacy validation')
    saved=read(OUT/'records.json');require(len(saved)==8 and {r['record'] for r in saved}==set(RECORDS),'Eight record names missing.')
    saved_index={r['record']:r for r in saved};computed=[];grid_proofs={}
    for name in RECORDS:
        model,objective=name.split('-');p=probability[model]
        if objective=='max33':tau,proof,annex=direct_grid(rows,p);grid_proofs[model]=dict(proof=proof,annex=annex)
        elif model=='oldhead':tau,proof=CURRENT_TAU,None
        elif model=='ensemble':tau,proof=ENSEMBLE_TAU,None
        else:tau,proof=legacy_mean(rows['old'],p['old']),None
        result=evaluate(rows,p,tau);record=dict(record=name,model=model,objective=objective,validation=result,
            full_grid_proof=proof,diagnostic=name!='new-max33',tests_read=False);close(saved_index[name],record,name);computed.append(record)
    selected=selection(computed,legacy);published=read(OUT/'selection.json')
    for k,v in selected.items():close(published[k],v,'selection/'+k)
    require(published['source_sha256']==SOURCE_SHA and published['plan_sha256']==plan_sha and published['records_sha256']==complete['artifact_sha256']['records.json']
        and published['controls_sha256']==complete['artifact_sha256']['controls.json'] and published['model_path']==str(OUT/'models/new.npz')
        and published['model_sha256']==models['new'] and published['system_threshold']==saved_index['new-max33']['validation']['system_threshold'],'Published selection identity/tau changed.')
    verify_csv(pd.read_csv(OUT/'validation-metrics.csv'),computed)
    counts=read(OUT/'source-counts.json');expected_counts=dict(old_fit=len(fit),natural_fit=len(nf),per_epoch=641493,steps=157,last_batch=2517,
        validation_rows={b:len(f) for b,f in rows.items()},natural_fit_source_roots=255,public_source_roots=64,
        natural_fit_role_support={r:int(nf.role.eq(r).sum()) for r in ROLES},public_role_support={r:int(rows['public'].role.eq(r).sum()) for r in ROLES},
        legacy_natural_role_support={r:int((rows['old'].profile.eq('natural')&rows['old'].role.eq(r)).sum()) for r in ROLES},
        natural_fit_system_families=int(nf.family_id.nunique()),natural_fit_dependency_components=int(nf.dependency_component_id.nunique()),
        public_system_families=int(rows['public'].family_id.nunique()),public_dependency_components=int(rows['public'].dependency_component_id.nunique()),
        public_dataset_source_roots={d:int(rows['public'].loc[rows['public'].dataset.eq(d),'source_uid'].nunique()) for d in ['oasst','toxicchat0124']},
        reused_development_only=True,tails_and_public_source_groups_overlap=True)
    for k,v in expected_counts.items():close(counts[k],v,'source-counts/'+k)
    sources={g:dict(bank=b,tokens=len(ix),documents=int(rows[b].iloc[ix].group_id.nunique()),role_counts={r:int(rows[b].iloc[ix].role.eq(r).sum()) for r in ROLES},
        sample_ix_sha256=array_sha(rows[b].iloc[ix].sample_ix.to_numpy())) for g,(b,ix) in group_indices(rows).items()}
    require(sources['old-natural']['role_counts']['cot']>0 and sources['old-natural']['role_counts']['tool']==0
        and all(sources[g]['role_counts']['cot']==sources[g]['role_counts']['tool']==0 for g in PUBLIC),'Public SUA/legacy CoT support collapsed.')
    terminal_gate()
    for path,want in bindings.items():require(digest(path)==want,'Bound artifact changed during audit: '+path)
    report=dict(stage='actual_completed_natural_fit_development_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=digest(__file__),
        trainer_source_sha256=SOURCE_SHA,trainer_complete_sha256=complete_sha,trainer_protocol_sha256=PROTOCOL_SHA,
        trainer_selection_sha256=bindings[str((OUT/'selection.json').resolve())],trainer_records_sha256=bindings[str((OUT/'records.json').resolve())],
        trainer_model_record_sha256={a:bindings[str((OUT/'candidates'/f'{a}.json').resolve())] for a in ['new','matched']},
        model_sha256=models,provenance_sha256=bindings,probability_bindings=probability_bindings,eight_records=computed,selection=selected,
        independent_grid_proofs=grid_proofs,source_counts=sources,training_budgets=budgets,full_independent_CPU_predictions=portable,
        maximum_metric_numeric_difference=MAX_DIFFERENCE,csv_primary_group_rows=88,csv_descriptive_pool_rows=8,
        original_savedGPU_NPZ_and_threshold_unchanged=True,fixed_four_member_ensemble=True,only_new_max33_eligible=True,
        test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False,reused_development_validation=True,
        new_independent_confirmation_required=True,generalization_confidence_intervals_claimed=False)
    output=Path(output);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(output),report_sha256=digest(output),training_budgets=budgets,selection=selected,source_sha256=report['auditor_source_sha256'])),flush=True)


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
    from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,f1_score,log_loss
    checks=[];before=digest(__file__);folder=Path(tempfile.mkdtemp(prefix='long-data-natural-fit-audit-preflight-',dir='/tmp'))
    def check(name,value=True):require(bool(value),'CPU fixture: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,AssertionError,ValueError,KeyError):checks.append(name);return
        raise RuntimeError('Fault accepted: '+name)
    rng=np.random.default_rng(57149)
    for i in range(8):
        y=rng.integers(0,5 if i%2 else 3,400);p=rng.integers(1,10,(400,5)).astype(np.float32);p/=p.sum(1,keepdims=True)
        value=metrics(p,y,.23);matrix=confusion_matrix(y,p.argmax(1),labels=np.arange(5))
        check('metrics_sklearn_confusion_'+str(i),value['confusion']==matrix.tolist())
        close(value['system_auroc'],roc_auc_score(y==0,p[:,0]),'fixture/auc')
        close(value['system_average_precision'],average_precision_score(y==0,p[:,0]),'fixture/ap')
        close(value['accuracy'],float((y==p.argmax(1)).mean()),'fixture/acc')
        close(value['macro_f1'],f1_score(y,p.argmax(1),labels=np.unique(y),average='macro'),'fixture/macro')
        close(value['nll'],log_loss(y,p,labels=np.arange(5)),'fixture/nll')
    check('random_tied_AUC_AP_supported_macro_NLL')
    p=np.tile(np.array([.1,.1,.6,.1,.1],np.float32),(6,1));y=np.array([0,0,1,1,3,3]);v=metrics(p,y,.9)
    check('undefined_precision_and_SUA_CoT_tool_NA',v['system_precision'] is None and v['recall']['cot'] is None and v['recall']['tool'] is None)
    rows,p=synthetic();tau,proof,annex=direct_grid(rows,p);value=evaluate(rows,p,tau)
    check('eleven_groups_min33_direct_everygrid',len(value['groups'])==11 and value['min33']==proof['min33'] and proof['primary_statistics']==33*proof['grid_thresholds'])
    check('nine_FPR_groups_exclude_tails',len(FEASIBLE)==9 and all('tail' not in x for x in FEASIBLE))
    check('public_CoT_tool_NA_legacy_CoT_supported',all(value['groups'][g]['recall']['cot'] is None and value['groups'][g]['recall']['tool'] is None for g in PUBLIC) and value['groups']['old-natural']['recall']['cot'] is not None)
    groups=group_indices(rows)
    check('tail_indices_global_saved_bank_order',np.array_equal(groups['old-long-full-tail'][1],np.flatnonzero(rows['old'].profile.eq('long-full')&rows['old'].token_ix.ge(529))) and np.array_equal(groups['new-C4-long-full-tail'][1],np.flatnonzero(rows['c4'].body_offset.ge(512))))
    adjacent=np.nextafter(np.float32(.5),np.float32(1));boundary=.5+(float(adjacent)-.5)/6;scores=np.array([.5,adjacent],np.float32)
    check('actual_F32_vs_legacy_F64_boundary',np.count_nonzero(scores>=boundary)==2 and len(scores)-np.searchsorted(np.sort(scores),np.array([boundary]),side='left')[0]==1)
    # Explicitly change non-system probabilities: original same-order averaging
    # must retain full five-way arrays, not average only the binary score.
    values=[np.roll(p['public'],i,axis=1).copy() for i in range(4)];mean=fixed_mean(values)
    check('fixed_all4_F64_mean_castF32_exact',np.array_equal(mean,(sum(v.astype(np.float64) for v in values)/4).astype(np.float32)))
    reject('reject_pruned_ensemble',lambda:fixed_mean(values[:3]))
    original=rows['old'].copy();original['source_uid']='old';original['split']='fit';nf=rows['public'].copy();nf['split']='fit'
    draw,summary=matched_draw(original,nf);again,_=matched_draw(original,nf)
    pd.testing.assert_frame_equal(draw,again,check_exact=True)
    check('fixed_role_slot_draw_once_replacement',np.array_equal(original.set_index('cache_ix').loc[draw.matched_old_cache_ix].role,nf.role) and summary['torch_rng_consumed'] is False)
    import torch
    state=torch.random.get_rng_state().clone();matched_draw(original,nf);check('private_numpy_draw_preserves_torch_rng',torch.equal(state,torch.random.get_rng_state()))
    history=[dict(epoch=e,steps=157,instances=641493,cumulative_steps=e*157,loss=.5,old5_mean_f1=.8) for e in range(1,11)]
    h=history_audit(history,1);check('157_steps_last2517_best_actual_differ',h['best_steps']==157 and h['actual_steps']==1570 and h['last_batch_instances']==2517)
    reject('history_min10',lambda:history_audit(history[:8],1));reject('history_continues_after_patience',lambda:history_audit(history+[dict(history[-1],epoch=11,cumulative_steps=1727)],1))
    def g(score):return dict(min33=score,old5_mean_f1=score,groups={name:dict(system_f1=score,accuracy=score,system_false_positive_rate=.01) for name in GROUPS})
    records=[dict(record=n,validation=g(.9)) for n in RECORDS];next(r for r in records if r['record']=='new-max33')['validation']=g(.93)
    selected=selection(records,g(.9));check('positive_only_new_max33',selected['promoted'] and selected['candidate']=='new-max33')
    changes=[('absolute90',lambda v:v.update(min33=.899999999999)),('strict_matched',lambda v:v.update(min33=.9)),
        ('old4_F1',lambda v:v['groups']['old-single'].update(system_f1=.87)),('old4_accuracy',lambda v:v['groups']['old-multi'].update(accuracy=.88)),
        ('old5_mean',lambda v:v.update(old5_mean_f1=.89)),('oldtail',lambda v:v['groups']['old-long-full-tail'].update(system_f1=.89)),
        ('newtail',lambda v:v['groups']['new-C4-long-full-tail'].update(system_f1=.89)),('publicF1',lambda v:v['groups'][PUBLIC[1]].update(system_f1=.87)),
        ('public_accuracy',lambda v:v['groups'][PUBLIC[2]].update(accuracy=.88)),('FPR',lambda v:v['groups'][PUBLIC[0]].update(system_false_positive_rate=.051))]
    for name,change in changes:
        bad=copy.deepcopy(records);change(next(r for r in bad if r['record']=='new-max33')['validation']);s=selection(bad,g(.9));check('guard_'+name,not s['promoted'] and s['candidate']=='new-max33')
    legacy=g(.96);s=selection(records,legacy);check('mandatory_savedGPU_guard_independent','original-savedGPU:old5_mean_f1_loss' in s['rejection_reasons'])
    f_records=[dict(record=n,validation=value) for n in RECORDS];flat=[]
    for r in f_records:
        for name,m in dict(r['validation']['groups'],pooled_descriptive_only=r['validation']['pooled_descriptive_only']).items():
            flat.append(dict(record=r['record'],group=name,**{k:v for k,v in m.items() if not isinstance(v,(dict,list))},**{'role_recall_'+role:v for role,v in m['recall'].items()}))
    frame=pd.DataFrame(flat);verify_csv(frame,f_records);check('96CSV_88formal_8pooled_NAs')
    wrong=frame.copy();wrong.loc[0,'system_recall']=.123;reject('CSV_numeric_tamper',lambda:verify_csv(wrong,f_records))
    target=folder/'p.npz';np.savez(target,p=p['public'],sample_ix=rows['public'].sample_ix.to_numpy());check('NPZ_sample_SHA',np.array_equal(read_probability(target,digest(target),rows['public']),p['public']))
    wrong=rows['public'].copy();wrong['sample_ix']+=1;reject('NPZ_wrong_sample_namespace',lambda:read_probability(target,digest(target),wrong))
    bad=copy.deepcopy(p);bad['old'][0,0]+=.001
    reject('complete_reference_drift_technical_failure',lambda:drift(rows,p,bad,CURRENT_TAU))
    with patch(__name__+'.SOURCE_SHA','fake'),patch(__name__+'.PROTOCOL_SHA','fake'),patch(__name__+'.processes',return_value=[123]),patch(__name__+'.read',side_effect=AssertionError('Result read before terminal')):
        reject('actual_live_gate_before_result_read',lambda:actual(folder/'forbidden.json'))
    with patch(__name__+'.PROTOCOL_SHA',None):
        reject('actual_unpinned_final_trainer_forbidden',lambda:actual(folder/'forbidden.json'))
    # End-to-end independent rawF16 -> old scaler -> MLP -> p, fixture weights
    # only. NumPy/SciPy evaluates the same mathematical network separately.
    from scipy.special import erf
    bank=(rng.normal(size=(8,512))*.1).astype(np.float16);tiny=pd.DataFrame(dict(cache_ix=[7,1,4]))
    mu=np.full(512,.013,np.float32);sigma=np.full(512,.7,np.float32)
    w={'encoder.0.weight':(rng.normal(size=(128,512))*.01).astype(np.float32),'encoder.0.bias':np.full(128,.01,np.float32),
        'head.weight':(rng.normal(size=(5,128))*.01).astype(np.float32),'head.bias':np.linspace(-.1,.1,5,dtype=np.float32)}
    actual_p=independent_predict(w,mu,sigma,bank,tiny);x=(bank[tiny.cache_ix].astype(np.float32)-mu)/sigma
    z=x.astype(np.float64)@w['encoder.0.weight'].T+w['encoder.0.bias'];z=.5*z*(1+erf(z/np.sqrt(2)))
    logits=z@w['head.weight'].T+w['head.bias'];prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    check('full_CPU_rawF16_scaler_GELU_logits_numpy_scipy_parity',float(np.abs(actual_p-prob).max())<1e-6)
    check('independent_seeded_initial_weight_hash',len(initialization_sha())==64)
    check('source_unchanged',digest(__file__)==before)
    report=dict(stage='synthetic_natural_fit_result_auditor_preflight',audit_passed=True,source_sha256=before,
        checks=checks,check_count=len(checks),synthetic_only=True,actual_probabilities_or_models_read=False,
        test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False,
        trainer_final_source_pinned=SOURCE_SHA,protocol_pinned=PROTOCOL_SHA,fullgrid=proof,
        fullgrid_annex=annex,maximum_metric_difference=MAX_DIFFERENCE)
    target=folder/'report.json';target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(target),report_sha256=digest(target),**report)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true');action.add_argument('--actual',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('/tmp/long_data_natural_fit_results_audit.json'))
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.actual:actual(args.output)
    else:parser.print_help()
