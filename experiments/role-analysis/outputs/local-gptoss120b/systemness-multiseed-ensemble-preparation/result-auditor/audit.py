"""Independent bounded multiseed development audit. Default help; synthetic --preflight.

--actual requires a completed terminal run. No production metrics/calibration,
GPU, test data/results, training or production writes. All fixed seeds retained.
"""
import argparse,copy,hashlib,importlib.metadata,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path('/root/prompt-injection-as-role-confusion')
sys.path.insert(0,str(ROOT/'experiments/role-analysis'))
import systemness_data as data
import systemness_head_probes as hp
import systemness_multiseed_ensemble as production
SOURCE_SHA='0ebea56bfd484d8261a4fce2aa87791e935c9896cc6f12dd9352c632fbc6ded3'
PROTOCOL_SHA='ccfcfc2f77c51cadb29ad7549b5a94f8b08ff41eda397777e36119aad095b2f1'
DESIGN_SHA='a1ad395c345ec8af91d5782987a95f0b9cd9034da92b7f40debc2dce1ec521e6'
CURRENT_SHA='27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
CURRENT_TAU=.620969696969697
SEEDS=[20261022,20261023,20261024]
MEMBERS=[20261021]+SEEDS
RECORDS=['old-original','old-cpu-maximin']+[f'seed-{s}-maximin' for s in SEEDS]+['ensemble-maximin']
ROLES=['system','user','cot','assistant','tool']
PROFILES=['single','multi','natural','long-single','long-full'];ORIGINAL=PROFILES[:4]
GROUPS=['old-'+p for p in PROFILES]+['old-long-full-tail','new-long-full','new-long-full-tail']
FEASIBLE=['old-'+p for p in PROFILES]+['new-long-full']
ARTIFACTS=['protocol.json','search-plan.json','old-control.json','ensemble.json','records.json','selection.json','validation-metrics.csv','report.txt','source-counts.json']
OUT=data.runtime.OUT/'systemness-multiseed-ensemble';CACHE=data.CACHE/'multiseed-ensemble'
OLD=Path('/dev/shm/role-analysis-top8-heads');NEW=Path('/dev/shm/role-analysis-long-augmentation-heads')
MAX_DIFFERENCE=0.
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def require(value, message):
    if not value: raise RuntimeError(message)

def read(path): return json.loads(Path(path).read_text())

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

def group_indices(old_rows, new_rows):
    require(np.array_equal(old_rows.index, np.arange(len(old_rows))) and np.array_equal(new_rows.index, np.arange(len(new_rows))), 'Validation row indices must reset within banks.')
    groups = {f'old-{name}': ('old', np.flatnonzero(old_rows.profile.eq(name))) for name in PROFILES}
    for bank, frame in [('old', old_rows), ('new', new_rows)]:
        full = frame[frame.profile.eq('long-full')]
        offsets = full.body_offset.to_numpy() if 'body_offset' in full else (full.token_ix - full.groupby(['prompt_ix', 'message_ix']).token_ix.transform('min')).to_numpy()
        groups[bank + '-long-full-tail'] = (bank, full.index.to_numpy()[offsets >= 512])
        if bank == 'new': groups['new-long-full'] = (bank, full.index.to_numpy())
    require(set(groups) == set(GROUPS) and all(len(ix) for _, ix in groups.values()), 'Missing group support.')
    return groups

def evaluate(old_rows, new_rows, po, pn, tau):
    groups = group_indices(old_rows, new_rows); ps, ys = dict(old=po, new=pn), dict(old=labels(old_rows), new=labels(new_rows))
    values = {name: metrics(ps[bank][ix], ys[bank][ix], tau) for name, (bank, ix) in groups.items()}
    primary = [values[name][key] for name in GROUPS for key in ['system_precision', 'system_recall', 'system_balanced_accuracy']]
    return dict(system_threshold=float(tau), groups=values,
        old5_mean_f1=float(np.mean([values['old-' + name]['system_f1'] for name in PROFILES])),
        min24=min(primary) if all(value is not None and np.isfinite(value) for value in primary) else None,
        pooled_descriptive_only=metrics(np.concatenate([po, pn]), np.r_[ys['old'], ys['new']], tau))

def old_validation_record(value, old_rows):
    profiles = {name: value['groups']['old-' + name] for name in PROFILES}
    tail = dict(value['groups']['old-long-full-tail'])
    full = old_rows[old_rows.profile.eq('long-full')]
    offset = full.token_ix - full.groupby(['prompt_ix', 'message_ix']).token_ix.transform('min')
    tail.update(documents=int(full.loc[offset.ge(512), 'document_id'].nunique()), body_start=512, body_end=1023)
    return dict(system_threshold=value['system_threshold'], mean_system_f1=value['old5_mean_f1'],
        original_mean_system_f1=float(np.mean([profiles[name]['system_f1'] for name in ORIGINAL])),
        original_mean_accuracy=float(np.mean([profiles[name]['accuracy'] for name in ORIGINAL])), profiles=profiles, long_full_tail=tail)

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

def direct_grid(old_rows, new_rows, po, pn):
    """Every original F64 tau is directly consumed by F32 probability arrays."""
    groups = group_indices(old_rows, new_rows); ps, ys = dict(old=po, new=pn), dict(old=labels(old_rows), new=labels(new_rows))
    grid = np.unique(np.r_[np.linspace(.001, .999, 199), np.quantile(np.r_[po[:, 0], pn[:, 0]], np.linspace(0, 1, 301)), 1.000001])
    curves = {}; stat_keys = ['P', 'R', 'BAcc', 'F1', 'FPR']
    for name, (bank, ix) in groups.items():
        truth = ys[bank][ix] == 0; score = ps[bank][ix, 0]; positive, negative = int(truth.sum()), int((~truth).sum())
        require(positive > 0 and negative > 0, 'A grid group lacks positive/negative support.')
        values = []
        for tau in grid:
            detected = score >= float(tau); tp = int(np.count_nonzero(truth & detected)); fp = int(np.count_nonzero(~truth & detected))
            fn, tn = positive - tp, negative - fp
            values.append([tp / (tp + fp) if tp + fp else np.nan, tp / positive,
                (tp / positive + tn / negative) / 2, 2 * tp / (2 * tp + fp + fn), fp / negative])
        curves[name] = np.asarray(values)
    objective = np.min([curves[name][:, index] for name in GROUPS for index in [0, 1, 2]], axis=0)
    objective[~np.isfinite(objective)] = -np.inf
    mean = np.mean([curves['old-' + name][:, 3] for name in PROFILES], axis=0)
    feasible = np.max([curves[name][:, 4] for name in FEASIBLE], axis=0) <= .05
    candidates = np.flatnonzero(feasible & np.isfinite(objective)); require(len(candidates), 'No defined feasible maximin threshold.')
    best = sorted(candidates, key=lambda index: (-float(objective[index]), -float(mean[index]), float(grid[index])))[0]
    proof = dict(grid_thresholds=len(grid), groups=8, primary_statistics=len(grid) * 24,
        additional_f1_fpr_statistics=len(grid) * 16, maximum_direct_difference=0.,
        all_grid_undefined_and_count_semantics_equal=True, actual_F32_comparison=True,
        all_grid_feasibility_equal=True, global_argmax_equal=True,
        min24=float(objective[best]), old5_mean_f1=float(mean[best]), feasible_thresholds=int(feasible.sum()))
    annex = dict(grid_sha256=array_sha(grid), effective_F32_grid_sha256=array_sha(grid.astype(np.float32)),
        direct_curves_sha256={name: array_sha(curves[name]) for name in GROUPS}, stat_order=stat_keys,
        undefined_precision_thresholds={name: int(np.isnan(curves[name][:, 0]).sum()) for name in GROUPS})
    return float(grid[best]), proof, annex

def guard(result, control):
    tolerance = 1e-12
    f1_losses = {p: control['groups']['old-' + p]['system_f1'] - result['groups']['old-' + p]['system_f1'] for p in ORIGINAL}
    accuracy_losses = {p: control['groups']['old-' + p]['accuracy'] - result['groups']['old-' + p]['accuracy'] for p in ORIGINAL}
    full = ['old-long-full', 'old-long-full-tail', 'new-long-full', 'new-long-full-tail']
    gains = {p: result['groups'][p]['system_f1'] - control['groups'][p]['system_f1'] for p in full}
    mean_loss = control['old5_mean_f1'] - result['old5_mean_f1']; reasons = []
    if not all(np.isfinite(result['groups'][g][key]) for g in GROUPS for key in ['system_f1', 'accuracy', 'system_false_positive_rate']) or not np.isfinite(result['old5_mean_f1']): reasons.append('nonfinite_validation_metric')
    if max(f1_losses.values()) > .02 + tolerance: reasons.append('original4_system_f1_loss')
    if max(accuracy_losses.values()) > .01 + tolerance: reasons.append('original4_five_accuracy_loss')
    if min(gains.values()) < -tolerance: reasons.append('old_or_new_full_tail_f1_regression')
    if mean_loss > .005 + tolerance: reasons.append('old5_mean_f1_loss')
    if max(result['groups'][g]['system_false_positive_rate'] for g in FEASIBLE) > .05 + tolerance: reasons.append('profile_fpr_above_limit')
    return dict(eligible=not reasons, rejection_reasons=reasons, original4_f1_losses=f1_losses,
        original4_accuracy_losses=accuracy_losses, full_tail_f1_gains=gains, old5_mean_f1_loss=mean_loss)

def read_probability(path, expected_sha, rows):
    require(digest(path) == expected_sha, 'Probability byte SHA mismatch.')
    with np.load(path, allow_pickle=False) as values:
        require(set(values.files) == {'p', 'sample_ix'} and np.array_equal(values['sample_ix'], rows.sample_ix), 'Probability sample order/namespace mismatch.')
        p = values['p'].copy()
    validate_p(p, len(rows)); return p

def source_partitions(old_rows, new_rows):
    require(len(old_rows) == 755282 and len(new_rows) == 389120
        and np.array_equal(old_rows.cache_ix, np.arange(len(old_rows))) and np.array_equal(new_rows.cache_ix, np.arange(len(new_rows)))
        and old_rows.sample_ix.is_unique and new_rows.sample_ix.is_unique and set(old_rows.split) == set(new_rows.split) == {'fit', 'validation'}
        and set(old_rows.wrapper) == set(new_rows.wrapper) == {'native'}, 'Bank row ownership changed.')
    raw_fit = old_rows[old_rows.split.eq('fit')]; dense_ids = set(raw_fit.loc[raw_fit.profile.eq('long-full'), 'document_id'])
    removed = raw_fit[raw_fit.profile.eq('long-single') & raw_fit.document_id.isin(dense_ids)]
    fit = raw_fit[~raw_fit.cache_ix.isin(removed.cache_ix)]
    valid = old_rows[old_rows.split.eq('validation')].copy().reset_index(drop=True)
    nf, nv = new_rows[new_rows.split.eq('fit')], new_rows[new_rows.split.eq('validation')].copy().reset_index(drop=True)
    require(len(fit) == 578090 and len(valid) == 173032 and len(removed) == 4160 and removed.document_id.nunique() == 26
        and len(dense_ids) == 132 and len(nf) == 327680 and len(nv) == 61440
        and not set(fit.group_id) & set(valid.group_id) and not set(nf.group_id) & set(nv.group_id)
        and not set(new_rows.group_id) & set(old_rows.group_id) and not set(new_rows.sample_ix) & set(old_rows.sample_ix), 'Source partition/removal/leakage mismatch.')
    require(set(new_rows.source_set) == {'long-augmentation'} and set(new_rows.profile) == {'long-full'}
        and new_rows.sample_ix.min() >= 6000000000000 and new_rows.document_id.min() >= 6000000,
        'New source namespace changed.')
    partition = dict(fit_rows=len(fit), validation_rows=len(valid), removed_sparse_rows=len(removed), removed_sparse_documents=26,
        fit_cache_ix_sha256=array_sha(fit.cache_ix.to_numpy()), fit_sample_ix_sha256=array_sha(fit.sample_ix.to_numpy()), fit_labels_sha256=array_sha(labels(fit)),
        validation_cache_ix_sha256=array_sha(valid.cache_ix.to_numpy()), validation_sample_ix_sha256=array_sha(valid.sample_ix.to_numpy()), validation_labels_sha256=array_sha(labels(valid)))
    return fit, valid, nf, nv, partition

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

def verify_csv(frame, records):
    expected = {(record['record'], name): value for record in records for name, value in
        dict(record['validation']['groups'], pooled_descriptive_only=record['validation']['pooled_descriptive_only']).items()}
    require(len(frame) == 54 and not frame.duplicated(['record', 'group']).any() and set(zip(frame.record, frame.group)) == set(expected), 'CSV must retain48 group and6 descriptive pooled rows.')
    for row in frame.to_dict('records'):
        value = expected[(row['record'], row['group'])]
        for key, truth in value.items():
            if isinstance(truth, (dict, list)): continue
            actual = row[key]; actual = None if pd.isna(actual) else actual
            close(actual, truth, 'csv/' + row['record'] + '/' + row['group'] + '/' + key)
        for role in ROLES:
            actual = row['role_recall_' + role]; actual = None if pd.isna(actual) else actual
            close(actual, value['recall'][role], 'csv/recall/' + role)

def synthetic():
    old_rows, new_rows = [], []
    for profile_ix, profile in enumerate(PROFILES):
        for role_ix, role in enumerate(ROLES):
            if profile == 'natural' and role == 'tool': continue
            for offset in [0, 341, 682, 1023]:
                old_rows.append(dict(profile=profile, role=role, token_ix=17 + offset, prompt_ix=profile_ix * 5 + role_ix,
                    message_ix=0, document_id=profile_ix, group_id='old-' + str(profile_ix), source_set='fixture',
                    cache_ix=len(old_rows), sample_ix=len(old_rows), split='validation', wrapper='native'))
    for role_ix, role in enumerate(ROLES):
        for offset in [0, 341, 682, 1023]:
            new_rows.append(dict(profile='long-full', role=role, token_ix=29 + offset, body_offset=offset,
                prompt_ix=100 + role_ix, message_ix=0, document_id=6000000, group_id='new', source_set='long-augmentation',
                cache_ix=len(new_rows), sample_ix=6000000000000 + len(new_rows), split='validation', wrapper='native'))
    old_rows, new_rows = pd.DataFrame(old_rows), pd.DataFrame(new_rows)
    def probabilities(frame):
        y = labels(frame); p = np.full((len(frame), 5), .02, np.float32); p[np.arange(len(y)), y] = .92
        for index in np.flatnonzero(y == 0)[::3]:
            p[index, 0] = np.nextafter(np.float32(.5), np.float32(1)); p[index, 1:] = (1 - p[index, 0]) / 4
        return p
    return old_rows, new_rows, probabilities(old_rows), probabilities(new_rows)

def fixed_mean(values):
 require(len(values)==4,'Exactly four fixed members required.')
 total=np.zeros(values[0].shape,np.float64)
 for value in values:validate_p(value,len(values[0]));total+=value
 value=(total/4).astype(np.float32);validate_p(value,len(value));return value


def selection(records):
 require(len(records)==6 and {r['record'] for r in records}==set(RECORDS),'Exactly six declared records required.')
 by={r['record']:r for r in records};candidate=by['ensemble-maximin']['validation'];guards={};gains={};reasons=[]
 for name in ['old-cpu-maximin','old-original']:
  control=by[name]['validation'];g=guard(candidate,control);guards[name]=g;reasons += [name+':'+r for r in g['rejection_reasons']]
  gain=candidate['min24']-control['min24'] if candidate['min24'] is not None and control['min24'] is not None else None;gains[name]=gain
  if gain is None or not np.isfinite(gain) or gain<=1e-12:reasons.append(name+':min24_not_strictly_improved')
 return dict(candidate='ensemble-maximin',promoted=not reasons,no_fallback=True,guards=guards,min24_gains=gains,
  rejection_reasons=reasons,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)


def history_audit(history,best_epoch):
 require(1<=best_epoch<=len(history)<=45,'Invalid best/history length.')
 best,chosen=-1.,-1
 for epoch,row in enumerate(history,1):
  require(row['epoch']==epoch and row['steps']==142 and row['instances']==578090 and row['cumulative_steps']==epoch*142
   and np.isfinite(row['loss']) and np.isfinite(row['old5_mean_f1']),'Per-epoch fit/update evidence changed.')
  if row['old5_mean_f1']>best+1e-5:best,chosen=row['old5_mean_f1'],epoch
  if epoch<len(history):require(not(epoch>=10 and epoch-chosen>=8),'Continued after early-stop trigger.')
 require(chosen==best_epoch and (len(history)==45 or(len(history)>=10 and len(history)-chosen>=8)),'Wrong terminal/best epoch.')
 return dict(best_epoch=best_epoch,best_steps=best_epoch*142,epochs=len(history),actual_steps=len(history)*142,
  actual_instances=len(history)*578090,instances_per_epoch=578090,steps_per_epoch=142,last_batch_instances=554)


def initialization_sha(seed):
 import torch
 from torch import nn
 class Net(nn.Module):
  def __init__(self):
   super().__init__();self.encoder=nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.Dropout(.1));self.head=nn.Linear(128,5)
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(seed);net=Net()
  values=[dict(name=name,shape=list(value.shape),dtype=str(value.numpy().dtype),sha256=array_sha(value.numpy())) for name,value in net.state_dict().items()]
 return canonical(values)


def processes():
 active=[]
 for entry in Path('/proc').iterdir():
  if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
  try:argv=(entry/'cmdline').read_bytes().split(b'\0')
  except(FileNotFoundError,PermissionError,ProcessLookupError):continue
  if any(Path(v.decode(errors='replace')).name=='systemness_multiseed_ensemble.py' for v in argv if v):active.append(int(entry.name))
 return active


def terminal_gate():
 require(not processes(),'Multiseed supervisor/worker live: no partial result reads.')
 require((OUT/'complete.json').is_file() and (OUT/'status.json').is_file(),'Completed status missing.')
 require(read(OUT/'status.json')['stage']=='validation_complete','Trainer not terminal complete.')


def drift(old_rows,new_rows,reference,cpu,threshold):
 out={}
 for name,(bank,ix) in dict(group_indices(old_rows,new_rows),**{'all-old':('old',np.arange(len(old_rows))),'all-new':('new',np.arange(len(new_rows)))}).items():
  a,b=reference[bank][ix],cpu[bank][ix];validate_p(a,len(a));validate_p(b,len(b));d=np.abs(a-b);maximum=float(d.max())
  require(np.isfinite(maximum) and maximum<5e-4,'Full oldCPU/reference drift exceeds frozen tolerance.')
  out[name]=dict(tokens=len(ix),maximum_probability_difference=maximum,mean_probability_difference=float(d.mean()),
   absolute_difference_quantiles={str(q):float(np.quantile(d,q)) for q in [.5,.9,.99,1.]},argmax_changes=int(np.count_nonzero(a.argmax(1)!=b.argmax(1))),
   original_threshold_decision_changes=int(np.count_nonzero((a[:,0]>=threshold)!=(b[:,0]>=threshold))))
 return out


def sampled_proof(proof):
 require(proof['sampled_tokens']==dict(old=256,new=256) and proof['CPU_outputs_saved'] is True and set(proof['maximum_probability_difference'])=={'old','new'}
  and all(np.isfinite(v) and 0<=v<5e-4 for v in proof['maximum_probability_difference'].values()),'Producer same-vector portable proof changed.')


def actual(target):
 terminal_gate() # Must precede any formal probability, model, row or result read.
 require(digest(production.__file__)==SOURCE_SHA and digest(OUT/'protocol.json')==PROTOCOL_SHA,'Frozen source/protocol changed.')
 import torch
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 bindings={};pbindings={}
 def bind(path,want=None):
  path=Path(path);value=digest(path)
  if want is not None:require(value==want,'Byte SHA mismatch: '+str(path))
  bindings[str(path.resolve())]=value;return value
 bind(__file__);bind(production.__file__,SOURCE_SHA);bind(OUT/'protocol.json',PROTOCOL_SHA);bind(OUT/'status.json')
 complete,plan,declaration=read(OUT/'complete.json'),read(OUT/'search-plan.json'),read(OUT/'protocol.json')
 complete_sha=bind(OUT/'complete.json');plan_sha=bind(OUT/'search-plan.json');sig=plan['signature'];static=declaration['static'];up=sig['upstream']
 require(complete['source_sha256']==SOURCE_SHA and complete['signature']==sig and sig['source_sha256']==SOURCE_SHA
  and sig['protocol_sha256']==PROTOCOL_SHA and sig['static_sha256']==canonical(static) and plan['protocol']==declaration['protocol']
  and complete['tests_read'] is False and complete['primary_changed'] is False and complete['new_independent_confirmation_required'] is True
  and set(complete['artifact_sha256'])==set(ARTIFACTS) and set(complete['candidate_sha256'])==set(map(str,SEEDS)),'Completion/declaration contract changed.')
 for name,want in complete['artifact_sha256'].items():bind(OUT/name,want)
 for component in [static,static['original']]:
  for path,want in component['metadata_sha256'].items():bind(path,want)
  for name,want in component['dependencies_sha256'].items():bind(data.HERE/name,want)
 require(static['source_sha256']==SOURCE_SHA and static['design_sha256']==DESIGN_SHA and static['probability_arrays_read'] is False
  and static['original']['probability_arrays_read'] is False and static['packages']=={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scikit-learn']}
  and static['torch_threads']==4 and static['float32_matmul_precision']=='highest','Static numerical/source declaration changed.')
 bind(production.DESIGN,DESIGN_SHA)
 protocol=declaration['protocol'];neural=protocol['neural']
 require(protocol['new_seeds']==SEEDS and protocol['members']==MEMBERS and protocol['records']==RECORDS and protocol['fit_rows']==578090
  and protocol['old_valid']==173032 and protocol['new_valid']==61440 and protocol['dimensions']==512 and protocol['roles']==ROLES
  and protocol['tests_read'] is False and protocol['primary_changed'] is False
  and neural==dict(kind='mlp',hidden=128,dropout=.1,optimizer='AdamW',lr=.001,weight_decay=.001,clip=5.,batch=4096,instances=578090,
   steps=142,last_batch=554,max_epochs=45,min_epochs=10,patience=8,improvement=1e-5),'Fixed seed/fit/training protocol changed.')
 # Close existing feature metadata without any old global verifier or body decode.
 new_out=data.runtime.OUT/'systemness-long-augmentation'
 for path,key in [(new_out/'data-complete.json','new_data_sha256'),(new_out/'extraction-complete.json','new_extraction_sha256'),(hp.FEATURE_OUT/'extraction-complete.json','old_extraction_sha256')]:bind(path,up[key])
 new_complete=read(new_out/'extraction-complete.json');old_complete=read(hp.FEATURE_OUT/'extraction-complete.json')
 require(new_complete['all_rows_written'] is True and new_complete['all_finite'] is True
  and up['selected_head_ids']==sig['selected_head_ids']==new_complete['selected_head_ids']==old_complete['selected_head_ids'],'Feature head rank/completeness changed.')
 for bank,folder,c in [('old',OLD,old_complete),('new',NEW,new_complete)]:
  bind(folder/'rows.parquet',up[bank+'_rows_sha256']);bind(folder/'features.f16',up[bank+'_features_sha256'])
  require(c['rows_sha256']==up[bank+'_rows_sha256'] and c['features_sha256']==up[bank+'_features_sha256'],'Feature completion mismatch.')
 columns=['cache_ix','sample_ix','document_id','group_id','split','profile','role','wrapper','source_set','token_ix','prompt_ix','message_ix']
 old_rows=pd.read_parquet(OLD/'rows.parquet',columns=columns);new_rows=pd.read_parquet(NEW/'rows.parquet',columns=columns+['body_offset'])
 fit,ov,_,nv,partition=source_partitions(old_rows,new_rows);close(partition,up['old_partitions'],'originalpartition')
 require(array_sha(nv.sample_ix.to_numpy())==up['new_validation_sample_ix_sha256'] and array_sha(nv.cache_ix.to_numpy())==up['new_validation_cache_ix_sha256'],'New96 row order changed.')
 require(len(fit)==578090 and not fit.profile.eq('natural').any(),'Original C4-only fit partition changed.')
 scaler_path=hp.OUT/'standardization.npz';scaler_sha=bind(scaler_path,static['standardization_sha256']);require(scaler_sha==sig['standardization_sha256']==up['standardization_sha256'],'Scaler signature changed.')
 with np.load(scaler_path,allow_pickle=False) as z:mean,scale=z['mean'].copy(),z['scale'].copy()
 require(mean.shape==scale.shape==(512,) and mean.dtype==scale.dtype==np.float32 and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Invalid old fit-only scaler.')
 banks=dict(old=np.memmap(OLD/'features.f16',mode='r',dtype=np.float16,shape=(755282,512)),new=np.memmap(NEW/'features.f16',mode='r',dtype=np.float16,shape=(389120,512)))
 def loadp(path,want,rows):
  pbindings[str(Path(path).resolve())]=bind(path,want);return read_probability(path,want,rows)
 def cpupath(seed,bank):return CACHE/f'{seed}-cpu-{bank}-validation.npz'
 members={};models={};budgets={};portable={};candidate_meta={}
 for seed in SEEDS:
  key=str(seed);path=OUT/'candidates'/f'{seed}.json';bind(path,complete['candidate_sha256'][key]);r=read(path);candidate_meta[key]=r
  tpath=OUT/'training'/f'{seed}.json';bind(tpath,r['training_sha256']);t=read(tpath);budget=history_audit(t['history'],t['best_epoch']);budgets[key]=budget
  init=initialization_sha(seed);require(init==plan['initial_weight_sha256'][key]==static['initial_weight_sha256'][key],'Seed init changed.')
  for evidence in [r,t]:
   require(evidence['seed']==seed and evidence['signature']==sig and evidence['plan_sha256']==plan_sha and evidence['initial_weight_sha256']==init,'Seed training ownership changed.')
   for k,v in budget.items():close(evidence[k],v,key+'/'+k)
  require(r['source_sha256']==SOURCE_SHA and r['tests_read'] is False and t['new_validation_predictions_read'] is False
   and t['model_sha256']==r['model_sha256'] and r['gpu_old_probability_sha256']==t['gpu_old_probability_sha256'],'Best/checkpoint/probability evidence changed.')
  mpath=OUT/'models'/f'{seed}.npz';models[key]=bind(mpath,r['model_sha256'])
  _,weights=model_arrays(mpath,mean,scale,dict(seed=seed,signature=sig,source_sha256=SOURCE_SHA,plan_sha256=plan_sha,
   initial_weight_sha256=init,standardization_sha256=scaler_sha,best_epoch=budget['best_epoch'],best_steps=budget['best_steps'],selected_head_ids=sig['selected_head_ids'],tests_read=False))
  cpu={bank:loadp(cpupath(seed,bank),r[bank+'_probability_sha256'],rows) for bank,rows in [('old',ov),('new',nv)]}
  gpu=loadp(CACHE/f'{seed}-gpu-old-validation.npz',r['gpu_old_probability_sha256'],ov)
  tau=legacy_mean(ov,gpu);best_value=evaluate(ov,nv,gpu,cpu['new'],tau)
  close(old_validation_record(best_value,ov),r['old_early_stop_validation'],key+'/legacy-best')
  close(best_value['old5_mean_f1'],t['history'][budget['best_epoch']-1]['old5_mean_f1'],key+'/best-score')
  err=float(np.abs(gpu-cpu['old']).max());require(0<=err<5e-4 and r['full_old_GPU_CPU_max_difference']==err,'Full GPU/CPU seed drift changed.')
  sampled_proof(r['portable_evidence']);require(r['cuda_visible_devices'] in ['0','1','2'] and 'A40' in r['cuda_device_name'],'Worker assignment changed.')
  portable[key]={bank:portable_check(weights,mean,scale,banks[bank],rows,cpu[bank]) for bank,rows in [('old',ov),('new',nv)]};members[seed]=cpu
 reference=static['old_reference'];original={bank:loadp(reference[bank+'_path'],reference[bank+'_sha256'],rows) for bank,rows in [('old',ov),('new',nv)]}
 old=read(OUT/'old-control.json');require(old['signature']==sig and old['model_sha256']==CURRENT_SHA and old['reference']==reference and old['tests_read'] is False,'Old control source changed.')
 oldcpu={bank:loadp(cpupath(20261021,bank),old[bank+'_probability_sha256'],rows) for bank,rows in [('old',ov),('new',nv)]};members[20261021]=oldcpu
 models['20261021']=bind(hp.OUT/'models/joint-mlp128.npz',CURRENT_SHA);_,weights=model_arrays(hp.OUT/'models/joint-mlp128.npz',mean,scale)
 old_candidate=read(hp.OUT/'candidates/joint-mlp128.json');bind(hp.OUT/'candidates/joint-mlp128.json')
 require(old_candidate['validation_probabilities_sha256']==reference['old_sha256'] and legacy_mean(ov,original['old'])==CURRENT_TAU,'Original GPU NPZ/tau changed.')
 close(old_candidate['validation'],old_validation_record(evaluate(ov,nv,original['old'],original['new'],CURRENT_TAU),ov),'original-preserved')
 drift_value=drift(ov,nv,original,oldcpu,CURRENT_TAU);close(old['drift'],drift_value,'full-control-drift')
 close(old['cpu_at_original_threshold'],evaluate(ov,nv,oldcpu['old'],oldcpu['new'],CURRENT_TAU),'oldcpu-originaltau');sampled_proof(old['portable_evidence'])
 portable['20261021']={bank:portable_check(weights,mean,scale,banks[bank],rows,oldcpu[bank]) for bank,rows in [('old',ov),('new',nv)]}
 selected=read(OUT/'selection.json');ensemble={bank:fixed_mean([members[s][bank] for s in MEMBERS]) for bank in ['old','new']}
 specification=dict(signature=sig,members=MEMBERS,weights=[.25]*4,arithmetic=protocol['probabilities'],model_sha256=models,
  member_probability_sha256={str(s):{b:bindings[str(cpupath(s,b).resolve())] for b in ['old','new']} for s in MEMBERS})
 close(read(OUT/'ensemble.json'),specification,'ensemble definition')
 for bank,rows in [('old',ov),('new',nv)]:
  p=loadp(CACHE/f'ensemble-{bank}-validation.npz',selected['ensemble_probability_sha256'][bank],rows)
  require(np.array_equal(p,ensemble[bank]),'Saved ensemble is not exact fixed-order fourCPU F64 mean castF32.')
 saved=read(OUT/'records.json');require(len(saved)==6 and {r['record'] for r in saved}==set(RECORDS),'Incomplete/duplicate six records.')
 saved_by={r['record']:r for r in saved};computed=[];grid_proofs={};record_p={}
 for name in RECORDS:
  p=original if name=='old-original' else oldcpu if name=='old-cpu-maximin' else ensemble if name=='ensemble-maximin' else members[int(name.split('-')[1])]
  if name=='old-original':tau,proof=CURRENT_TAU,None;record_p[name]={b:reference[b+'_sha256'] for b in ['old','new']}
  else:
   tau,proof,annex=direct_grid(ov,nv,p['old'],p['new']);grid_proofs[name]=dict(proof=proof,annex=annex)
   if name=='ensemble-maximin':record_p[name]=selected['ensemble_probability_sha256']
   else:
    seed=20261021 if name=='old-cpu-maximin' else int(name.split('-')[1]);record_p[name]={b:bindings[str(cpupath(seed,b).resolve())] for b in ['old','new']}
  entry=dict(record=name,validation=evaluate(ov,nv,p['old'],p['new'],tau),full_grid_proof=proof,diagnostic=name!='ensemble-maximin',tests_read=False)
  close(saved_by[name],entry,name);require(saved_by[name]['validation']['system_threshold']==tau,'Frozen selected tau differs.');computed.append(entry)
 expected=selection(computed)
 for k,v in expected.items():close(selected[k],v,'selection/'+k)
 require(selected['source_sha256']==SOURCE_SHA and selected['plan_sha256']==plan_sha and selected['records_sha256']==complete['artifact_sha256']['records.json']
  and selected['ensemble_definition_sha256']==complete['artifact_sha256']['ensemble.json'] and selected['member_model_sha256']==models
  and selected['system_threshold']==computed[-1]['validation']['system_threshold'],'Selection member/source/tau closure changed.')
 verify_csv(pd.read_csv(OUT/'validation-metrics.csv'),computed)
 counts=read(OUT/'source-counts.json');count_values=dict(fit=len(fit),old_valid=len(ov),new_valid=len(nv),fit_documents=int(fit.group_id.nunique()),
  original_partition_sha256=partition,old_profile_counts={p:int(ov.profile.eq(p).sum()) for p in PROFILES},
  old_natural_role_support={r:int((ov.profile.eq('natural')&ov.role.eq(r)).sum()) for r in ROLES},new96_is_reused_development=True,tails_are_subsets=True)
 for k,v in count_values.items():close(counts[k],v,'sourcecounts/'+k)
 require(count_values['old_natural_role_support']['cot']>0 and count_values['old_natural_role_support']['tool']==0
  and all(r['validation']['groups']['old-natural']['recall']['tool'] is None for r in computed),'Legacy natural role support/N/A changed.')
 groups={}
 for name,(bank,ix) in group_indices(ov,nv).items():
  frame=(ov if bank=='old' else nv).iloc[ix];groups[name]=dict(tokens=len(frame),documents=int(frame.group_id.nunique()),role_counts={r:int(frame.role.eq(r).sum()) for r in ROLES},cache_ix_sha256=array_sha(frame.cache_ix.to_numpy()),sample_ix_sha256=array_sha(frame.sample_ix.to_numpy()))
 terminal_gate()
 for path,want in bindings.items():require(digest(path)==want,'Audited input changed during audit: '+path)
 report=dict(stage='actual_completed_multiseed_development_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=digest(__file__),
  trainer_source_sha256=SOURCE_SHA,trainer_complete_sha256=complete_sha,trainer_protocol_sha256=PROTOCOL_SHA,trainer_selection_sha256=bindings[str((OUT/'selection.json').resolve())],
  trainer_records_sha256=bindings[str((OUT/'records.json').resolve())],member_model_sha256=models,record_probability_sha256=record_p,
  row_identity_sha256=dict(original=partition,new_validation_cache_ix=array_sha(nv.cache_ix.to_numpy()),new_validation_sample_ix=array_sha(nv.sample_ix.to_numpy()),new_validation_labels=array_sha(labels(nv))),
  independent_records=computed,independent_selection=expected,independent_grid_proofs=grid_proofs,independent_training_budgets=budgets,independent_drift=drift_value,
  independent_portable_CPU_checks=portable,source_counts=groups,provenance_sha256=bindings,probability_bindings=pbindings,
  maximum_metric_numeric_difference=MAX_DIFFERENCE,csv_primary_group_rows=48,csv_descriptive_pool_rows=6,old_current_NPZ_unchanged=True,
  original_old_threshold_unchanged=True,test_data_or_results_read=False,gpu_used=False,production_files_modified=False,
  limitation='Shared fit/previously selected inherited member/reused development data; no generalization CI. Sampled producer GPU portable evidence is validated metadata, not an independent GPU rerun.')
 target=Path(target);require(str(target.resolve()).startswith('/tmp/'),'Audit output must remain under /tmp.')
 target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(target),report_sha256=digest(target),source_sha256=digest(__file__),selection=expected,training_budgets=budgets,maximum_metric_numeric_difference=MAX_DIFFERENCE)),flush=True)

def preflight():
 import torch
 from sklearn.metrics import roc_auc_score,average_precision_score
 source=digest(__file__);checks=[];folder=Path(tempfile.mkdtemp(prefix='feature-review-multiseed-results-preflight-',dir='/tmp'))
 def check(name,value=True):require(bool(value),'CPU fixture: '+name);checks.append(name)
 def reject(name,fn):
  try:fn()
  except(RuntimeError,ValueError,AssertionError,KeyError,FileNotFoundError):checks.append(name)
  else:raise RuntimeError('CPU fault accepted: '+name)
 def banned(*a,**k):raise AssertionError('Actual data/model/probability operation forbidden in synthetic audit.')
 with patch(__name__+'.processes',return_value=[123]),patch(__name__+'.read',side_effect=banned):reject('Live supervisor rejects before any result read',lambda:actual(folder/'not-created.json'))
 with patch.object(pd,'read_parquet',side_effect=banned),patch.object(production,'dataset',side_effect=banned),patch.object(production,'source_signature',side_effect=banned),patch.object(data.runtime,'load_model',side_effect=banned):
  old,new,po,pn=synthetic();rng=np.random.default_rng(7727)
  for case in range(5):
   y=rng.integers(0,5 if case%2 else 3,317);p=rng.integers(1,8,size=(len(y),5)).astype(np.float32);p/=p.sum(1,keepdims=True)
   close(metrics(p,y,.23+case*.02),production.probes.metrics(p,y,.23+case*.02),'random-metrics')
   a,b=rankings(p[:,0],y==0);require(abs(a-roc_auc_score(y==0,p[:,0]))<1e-14 and abs(b-average_precision_score(y==0,p[:,0]))<1e-14,'Independent tied ranking differs from sklearn.')
  check('Five tied/random all-field metrics plus sklearn ranking parity')
  y=np.array([0,0,1,1,3,3]);p=np.tile(np.array([.1,.1,.6,.1,.1],np.float32),(6,1));m=metrics(p,y,.9)
  close(m,production.probes.metrics(p,y,.9),'undefined-supported-macro');check('Truth-supported macro/undefined precision/CotTool NAs',m['system_precision'] is None and m['recall']['cot'] is None and m['recall']['tool'] is None)
  tau=legacy_mean(old,po);close(old_validation_record(evaluate(old,new,po,pn,tau),old),production.lp.validation_metrics(old,po),'old-original-helper')
  check('Old-only legacy helper retained with exact original semantics')
  tau,proof,annex=direct_grid(old,new,po,pn);expected_tau,expected_proof=production.aug.maximin_threshold(old,new,po,pn)
  require(tau==expected_tau,'Independent maximin exact tau differs.');close(proof,expected_proof,'maximin-grid');close(evaluate(old,new,po,pn,tau),production.aug.metrics_at(old,new,po,pn,tau),'8groups')
  check('Every-grid24stats/objective/feasibility/lexargmax exact')
  scores=np.array([.5,np.nextafter(np.float32(.5),np.float32(1))],np.float32);boundary=.5+(float(scores[1])-.5)/6
  check('Nextafter F32 weakcast consumer preserved',int(np.count_nonzero(scores>=boundary))==2 and len(scores)-np.searchsorted(np.sort(scores),np.array([boundary]),side='left')[0]==1)
  g=group_indices(old,new);check('Tail indices refer to global savedp rows',np.array_equal(g['new-long-full-tail'][1],np.flatnonzero(new.body_offset.ge(512))) and np.array_equal(g['old-long-full-tail'][1],np.flatnonzero(old.profile.eq('long-full')&old.token_ix.ge(529))))
  raw=rng.uniform(size=(4,59,5)).astype(np.float32);raw/=raw.sum(2,keepdims=True)
  check('Independent four-member F64 accumulation/F32 output exact',np.array_equal(fixed_mean(list(raw)),production.fixed_mean(list(raw))))
  reject('Missing member refused',lambda:fixed_mean(list(raw[:3])))
  bad=raw.copy();bad[-1,-1,0]=np.nan;reject('NaN last row/member refused',lambda:fixed_mean(list(bad)))
  def value(n):return dict(min24=n,old5_mean_f1=n,groups={g:dict(system_f1=n,accuracy=n,system_false_positive_rate=.01) for g in GROUPS})
  rs=[dict(record=n,validation=value(.94 if n=='ensemble-maximin' else .92)) for n in RECORDS]
  close(selection(rs),production.select_records(rs),'positive-selection');check('Ensemble only dualcontrol promotion',selection(rs)['promoted'])
  for control in ['old-cpu-maximin','old-original']:
   wrong=copy.deepcopy(rs);next(r for r in wrong if r['record']==control)['validation']['min24']=.94
   close(selection(wrong),production.select_records(wrong),'tie-selection');check('Exact gain tie rejects '+control,not selection(wrong)['promoted'])
  for group,key,v in [('old-natural','system_f1',.88),('old-natural','accuracy',.89),('old-long-full-tail','system_f1',.91),('new-long-full','system_false_positive_rate',.051)]:
   wrong=copy.deepcopy(rs);wrong[-1]['validation']['groups'][group][key]=v;close(selection(wrong),production.select_records(wrong),'guard');check('Independent guard '+group+'/'+key,not selection(wrong)['promoted'])
  wrong=copy.deepcopy(rs);wrong[-1]['validation']['min24']=None;close(selection(wrong),production.select_records(wrong),'undefined');check('Undefined objective no fallback',not selection(wrong)['promoted'])
  drift0=drift(old,new,dict(old=po,new=pn),dict(old=po,new=pn),.5);close(drift0,production.drift_metrics(old,new,dict(old=po,new=pn),dict(old=po,new=pn),.5),'drift-zero')
  z=po.copy();j=int(np.flatnonzero(old.profile.eq('natural'))[-1]);z[j]=[.50001,.09999,.1,.2,.1];c=z.copy();c[j,0]-=.00002;c[j,1]+=.00002
  actual_drift=drift(old,new,dict(old=z,new=pn),dict(old=c,new=pn),.5);close(actual_drift,production.drift_metrics(old,new,dict(old=z,new=pn),dict(old=c,new=pn),.5),'tiny-drift')
  check('Complete drift detects small threshold flips and ten groups',actual_drift['old-natural']['original_threshold_decision_changes']==1 and len(actual_drift)==10)
  c[j,0]-=.001;c[j,1]+=.001;reject('Complete oldCPU drift limit enforced',lambda:drift(old,new,dict(old=z,new=pn),dict(old=c,new=pn),.5))
  history=[dict(epoch=i,steps=142,instances=578090,cumulative_steps=i*142,loss=.5,old5_mean_f1=.8) for i in range(1,11)]
  close(history_audit(history,1),production.check_history(history,1),'history');check('Original578090/142/min10/patience8 history')
  reject('Continued past early-stop refused',lambda:history_audit(history+[dict(history[-1],epoch=11,cumulative_steps=1562)],1))
  for seed in SEEDS:require(initialization_sha(seed)==production.initial_hash(seed),'Independent seed initialization differs.')
  check('All three seed CPU initial weight hashes independently reproduced')
  entries=[dict(record=n,validation=evaluate(old,new,po,pn,tau)) for n in RECORDS];flat=[]
  for r in entries:
   for name,m in dict(r['validation']['groups'],pooled_descriptive_only=r['validation']['pooled_descriptive_only']).items():flat.append(dict(record=r['record'],group=name,**{k:v for k,v in m.items() if not isinstance(v,(list,dict))},**{'role_recall_'+k:v for k,v in m['recall'].items()}))
  f=pd.DataFrame(flat);verify_csv(f,entries);check('All48group+6pooled CSV fields/roles/None verified',len(f)==54)
  f.loc[0,'system_recall']=.123;reject('CSV metric tamper rejected',lambda:verify_csv(f,entries))
  ppath=folder/'toy-p.npz';np.savez(ppath,p=po,sample_ix=old.sample_ix.to_numpy());check('Probability hash/sample roundtrip',np.array_equal(read_probability(ppath,digest(ppath),old),po))
  wrong=old.copy();wrong.sample_ix+=1;reject('Probability sample_ix mismatch refused',lambda:read_probability(ppath,digest(ppath),wrong))
  partial=folder/'partial';partial.mkdir()
  with patch(__name__+'.OUT',partial),patch(__name__+'.processes',return_value=[]):reject('Missing completion rejects before actual reads',lambda:actual(folder/'never.json'))
 require(digest(__file__)==source,'Synthetic audit modified its source.')
 r=dict(stage='independent_multiseed_results_auditor_synthetic_preflight',source_sha256=source,source_sha256_after=digest(__file__),training_source_sha256=SOURCE_SHA,protocol_sha256=PROTOCOL_SHA,
  checks_passed=len(checks),checks=checks,full_grid_proof=proof,maximum_numeric_difference=MAX_DIFFERENCE,actual_probabilities_or_data_read=False,gpu_used=False,formal_outputs_written=False)
 path=folder/'report.json';path.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(path),report_sha256=digest(path),**r)),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group();a.add_argument('--preflight',action='store_true');a.add_argument('--actual',action='store_true');p.add_argument('--output',default='/tmp/feature_review_multiseed_results_audit.json');args=p.parse_args()
 if args.preflight:preflight()
 elif args.actual:
  with patch.object(production,'calibrate',side_effect=AssertionError('Production calibration forbidden')),patch.object(production,'verified_completion',side_effect=AssertionError('Production completion verifier forbidden')),\
   patch.object(production,'select_records',side_effect=AssertionError('Production selection forbidden')),patch.object(production.aug,'metrics_at',side_effect=AssertionError('Production metrics forbidden')),\
   patch.object(production.aug,'maximin_threshold',side_effect=AssertionError('Production threshold helper forbidden')),patch.object(production.aug,'promotion_guard',side_effect=AssertionError('Production guard forbidden')),\
   patch.object(production.probes,'metrics',side_effect=AssertionError('Production metrics forbidden')),patch.object(production.lp,'validation_metrics',side_effect=AssertionError('Production legacy helper forbidden')):
   actual(args.output)
 else:p.print_help()
