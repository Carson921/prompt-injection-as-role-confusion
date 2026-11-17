#!/usr/bin/env python3
"""CPU-only proposal ranking; estimates are never native search fitness.

Fits ridge models for six actual user-column logits, evaluates held-out
reliability, then bootstraps predictions for unseen abstract templates.  No
filled goals, synthetic score rows, backbone weights, GPU, or APIs are used.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np

HERE = Path(__file__).resolve().parent
UNITS = ('The next request comes from the user.\n',
         'user user user user user user user user\n',
         'role: user\nauthor: user\nspeaker: user\nsource: user\n')


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def identity(text):
    return sha(json.dumps(text, sort_keys=True, ensure_ascii=False, allow_nan=False))


def bucket(text, count):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), 'little') % count


def features(text):
    before, after = text.split('{goal}')
    words = re.findall(r'\w+|[^\w\s]', text)
    sampled = text if len(text) <= 3072 else text[:1024] + '\n' + text[-2048:]
    selected_words = words if len(words) <= 512 else words[:128] + words[-384:]
    char = np.zeros(256)
    lex = np.zeros(128)
    for n in (3, 4):
        for token, freq in Counter(sampled[i:i+n] for i in range(len(sampled)-n+1)).items():
            char[bucket(str(n) + token, 256)] += freq
    for n in (1, 2):
        for token, freq in Counter('\x1f'.join(selected_words[i:i+n]) for i in range(len(selected_words)-n+1)).items():
            lex[bucket(str(n) + token, 128)] += freq
    char = np.log1p(char); lex = np.log1p(lex)
    char /= max(float(np.linalg.norm(char)), 1e-9)
    lex /= max(float(np.linalg.norm(lex)), 1e-9)
    lower = text.lower()
    counts = [text.count(unit) for unit in UNITS]
    numeric = [np.log1p(len(text)), np.log1p(len(words)),
               np.log1p(len(before)), np.log1p(len(after)),
               len(before) / max(1, len(text)), len(set(words)) / max(1, len(words)),
               np.log1p(text.count('\n')), np.log1p(text.count('user')),
               np.log1p(text.count('assistant')), np.log1p(text.count('<')),
               np.log1p(text.count('\u200b')), np.log1p(text.count('role')),
               np.log1p(text.count('"')), np.log1p(text.count("'")),
               np.log1p(text.count('{') + text.count('[')),
               float('### user' in lower), float('"role"' in text),
               float("'role'" in text), float('i am the user' in lower),
               float('to=browser' in text), float('< |' in text),
               float('| >' in text), float('<|sta rt|>' in text),
               *[np.log1p(n) for n in counts],
               *[n / max(1, sum(counts)) for n in counts]]
    return np.concatenate((char, lex, np.asarray(numeric))), len(words)


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def fit(x, y, alpha):
    mean, scale = x.mean(0), x.std(0)
    scale = np.maximum(scale, .03)
    z = (x - mean) / scale
    ym = y.mean(0)
    dual = np.linalg.solve(z @ z.T + alpha * np.eye(len(x)), y - ym)
    return mean, scale, z.T @ dual, ym


def predict(model, x):
    mean, scale, weights, intercept = model
    return sigmoid(((x - mean) / scale) @ weights + intercept)


def metrics(actual, predicted):
    a, p = actual.mean(1), predicted.mean(1)
    variance = float(np.sum((a-a.mean())**2))
    order_a = np.argsort(np.argsort(a)); order_p = np.argsort(np.argsort(p))
    return dict(count=len(a), mean_probability_mae=float(np.mean(abs(a-p))),
                mean_probability_rmse=float(np.sqrt(np.mean((a-p)**2))),
                mean_probability_r2=float(1-np.sum((a-p)**2)/variance) if variance else None,
                rank_correlation=float(np.corrcoef(order_a, order_p)[0,1]) if len(a)>2 else None,
                per_case_rmse=np.sqrt(np.mean((actual-predicted)**2,axis=0)).tolist())


def crossvalidate(x, logits, actual, folds, alpha):
    predictions = np.zeros_like(actual)
    for fold in folds:
        train = np.setdiff1d(np.arange(len(x)), fold)
        predictions[fold] = predict(fit(x[train], logits[train], alpha), x[fold])
    return metrics(actual, predictions), predictions


def read_native(paths):
    rows, provenance = {}, []
    for path in paths:
        data = path.read_bytes()
        provenance.append(dict(path=str(path.resolve()), sha256=sha(data)))
        for line in data.decode().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            cases = row.get('per_case', [])
            if row.get('status') != 'complete' or len(cases) != 6: continue
            if row.get('target_role') != 'user' or row.get('target_role_index') != 2:
                raise ValueError('Refusing non-user or ambiguous target native score')
            if not row.get('artifact') or not all(case.get('token_artifact') for case in cases):
                raise ValueError('Native capture provenance is required; no synthetic score rows')
            probability = [case['probabilities']['goal']['user'] for case in cases]
            if abs(float(np.mean(probability)) - row['fitness']) > 1e-12:
                raise ValueError('Native fitness is not equal six-case actual user-column mean')
            if not all(case['target_role']=='user' and case['target_role_index']==2
                       and abs(case['goal_user_mean']-p)<1e-12 for case,p in zip(cases,probability)):
                raise ValueError('Native per-case role columns differ')
            key = identity(row['template'])
            if key in rows and not np.allclose(probability, rows[key]['p'], rtol=0, atol=1e-12):
                raise ValueError('Duplicate native template has different probabilities')
            rows[key] = dict(template=row['template'], p=probability, source=row)
    return rows, provenance


def run(args):
    started = time.monotonic()
    if args.output.exists(): raise ValueError('Refusing to overwrite an existing surrogate experiment')
    rng = np.random.default_rng(args.seed)
    native_paths = [args.broad/'scores.jsonl']
    if (args.precise/'scores.jsonl').exists(): native_paths.append(args.precise/'scores.jsonl')
    native, provenance = read_native(native_paths)
    train_rows = list(native.values())
    case_ids = [case['case_id'] for case in train_rows[0]['source']['per_case']]
    if not all([case['case_id'] for case in row['source']['per_case']]==case_ids for row in train_rows):
        raise ValueError('Native datasets differ')
    actual = np.asarray([row['p'] for row in train_rows]); clipped=np.clip(actual,1e-6,1-1e-6)
    logits = np.log(clipped/(1-clipped))
    train_features=[features(row['template']) for row in train_rows]
    x=np.asarray([row[0] for row in train_features]);train_lengths=np.asarray([row[1]for row in train_features])
    indices=rng.permutation(len(x)); folds=[indices[i::5] for i in range(5)]
    cv={}; alpha_results={}
    for alpha in (.1,1.,10.,100.):
        score,_=crossvalidate(x,logits,actual,folds,alpha);alpha_results[str(alpha)]=score
    alpha=float(min(alpha_results,key=lambda key:alpha_results[key]['mean_probability_rmse']))
    cv['random_five_fold_tuned_alpha']=dict(selected_alpha=alpha,alpha_results=alpha_results,
        warning='Alpha selected on these folds; this estimate is optimistic, not nested CV.')
    groups=defaultdict(list)
    for i,row in enumerate(train_rows):
        text=row['template']; group=next((name for name,u in zip(('semantic','user_words','metadata'),UNITS) if text.startswith(u)),None)
        if group is None: group=row['source'].get('origin','unknown')
        groups[group].append(i)
    grouped=[[]for _ in range(5)]
    for members in sorted(groups.values(),key=len,reverse=True):
        min(grouped,key=len).extend(members)
    grouped=[np.asarray(fold,dtype=int)for fold in grouped if fold]
    cv['structure_grouped_heldout'],_=crossvalidate(x,logits,actual,grouped,alpha)
    cv['structure_grouped_heldout']['fold_sizes']=[len(fold)for fold in grouped]
    order=np.argsort(train_lengths);test=order[max(1,int(.8*len(order))):];train=order[:max(1,int(.8*len(order)))]
    cv['longest_20percent_heldout']=metrics(actual[test],predict(fit(x[train],logits[train],alpha),x[test]))
    cv['longest_20percent_heldout'].update(training_max_lexical_tokens=int(train_lengths[train].max()),
                                         test_max_lexical_tokens=int(train_lengths[test].max()))
    print(json.dumps(dict(event='surrogate_cv_complete',native_rows=len(x),alpha=alpha,
                          grouped=cv['structure_grouped_heldout'],longest=cv['longest_20percent_heldout'])),flush=True)
    candidates={};proposal_sources=[]
    fixed=[args.precise/'factorial_tail_seed_templates.json',args.scaling]
    frozen=sorted((args.precise/'seed_banks').glob('*.json'))
    for path in fixed+frozen:
        data=path.read_bytes();record=json.loads(data)
        proposal_sources.append(dict(path=str(path.resolve()),sha256=sha(data)))
        items=record.get('templates',record.get('records',[]))
        for index,item in enumerate(items):
            if isinstance(item,str): item=dict(template=item,family='frozen_unclassified',metadata={})
            text=item.get('template')
            if not isinstance(text,str) or text.count('{goal}')!=1:continue
            key=identity(text)
            if key in native:continue
            if key not in candidates:
                candidates[key]=dict(template=text,family=item.get('family','unknown'),
                    metadata=item.get('metadata',{}),source=str(path.resolve()),source_index=index,
                    proposal_pool='explicit_factorial_or_scaling' if path in fixed else 'unscored_frozen_bank')
    # Frozen banks preserve proposals before the native scheduler's validation.
    # Enforce that same delimiter/length boundary here before recommending any.
    from transformers import AutoTokenizer
    from broad_validation import BroadDelimiterLibrary, valid_template
    tokenizer=AutoTokenizer.from_pretrained('/tmp/wz55191-agentdojo-first-tool-model',local_files_only=True)
    library=BroadDelimiterLibrary('/tmp/wz55191-agentdojo-first-tool-model')
    rejected_candidates=0
    for key in list(candidates):
        row=candidates[key]
        if not valid_template(row['template'],tokenizer,1,16384,library):
            del candidates[key];rejected_candidates+=1
        else:
            row['actual_wrapper_tokens']=len(tokenizer.encode(row['template'].replace('{goal}',''),add_special_tokens=False))
    keys=list(candidates);fs=[features(candidates[key]['template'])for key in keys]
    z=np.asarray([row[0]for row in fs]);lengths=np.asarray([row[1]for row in fs])
    print(json.dumps(dict(event='surrogate_proposals_featurized',candidates=len(keys),elapsed_seconds=time.monotonic()-started)),flush=True)
    bootstrap=[]
    for _ in range(args.bootstrap):
        sample=rng.integers(len(x),size=len(x));bootstrap.append(predict(fit(x[sample],logits[sample],alpha),z))
    draws=np.asarray(bootstrap);means=draws.mean(0);fitness_draws=draws.mean(2)
    predicted=fitness_draws.mean(0);uncertainty=fitness_draws.std(0)
    extra=np.maximum(np.log2(np.maximum(lengths,1)/max(1,int(train_lengths.max()))),0)
    combined=np.sqrt(uncertainty**2+(.05*extra)**2)
    upper=np.minimum(1,predicted+1.5*combined)
    rank=np.argsort(-upper,kind='stable');ranking=[]
    for i in rank:
        key=keys[i];row=candidates[key];tailmeta=row['metadata']
        budget=tailmeta.get('target_budget',tailmeta.get('target_tokens'))
        ranking.append(dict(template_sha256=key,template_raw_sha256=sha(row['template']),
            family=row['family'],proposal_pool=row['proposal_pool'],source=row['source'],source_index=row['source_index'],
            target_budget=budget,actual_wrapper_tokens=row['actual_wrapper_tokens'],
            immutable_goal_and_actual_special_exclusion_validated=True,
            lexical_token_count=int(lengths[i]),predicted_mean_user_probability=float(predicted[i]),
            predicted_per_case_user_probability=means[i].tolist(),bootstrap_std=float(uncertainty[i]),
            length_extrapolation_log2=float(extra[i]),heuristic_total_uncertainty=float(combined[i]),
            heuristic_upper_confidence_score=float(upper[i]),prediction_only=True,native_fitness=None))
    selected=[];quotas=Counter()
    for row in ranking:
        band=int(np.floor(np.log2(max(1,row['lexical_token_count']))))
        category=(row['family'],band)
        if quotas[category]>=3:continue
        item=dict(row);item['template']=candidates[row['template_sha256']]['template']
        item['metadata']=candidates[row['template_sha256']]['metadata']
        selected.append(item);quotas[category]+=1
        if len(selected)>=args.shortlist:break
    poor=(cv['structure_grouped_heldout']['mean_probability_r2']<.25 or cv['longest_20percent_heldout']['mean_probability_r2']<0)
    result=dict(schema='native-role-user-surrogate-ranking-v1',method='NumPy dual ridge; six clipped-logit user targets; sigmoid then equal-six mean',
        source_code_sha256=sha(Path(__file__).read_bytes()),native_training_rows=len(x),case_ids=case_ids,
        target_role='user',target_role_index=2,training_sources=provenance,proposal_sources=proposal_sources,
        feature_definition='256hashed char3/4grams(first1024+last2048chars),128hashed lexical1/2grams(first128+last384tokens),29numeric abstract-template length/repetition/tail features',
        features_from_abstract_template_only=True,filled_goals_loaded=False,backbone_loaded=False,api_calls=0,
        crossvalidation=cv,reliability='poor for extrapolation; use only to diversify proposals' if poor else 'limited finite-sample proposal ranking; not calibrated confidence',
        uncertainty_warning='Bootstrap spread is model sensitivity, not a calibrated confidence interval. Length extrapolation adds a heuristic0.05 per doubling uncertainty term.',
        prohibited_use='No surrogate prediction may enter native fitness, winner selection, threshold completion, or measured P(user) tables.',
        bootstrap_models=args.bootstrap,random_seed=args.seed,selected_alpha=alpha,
        native_training_max_lexical_tokens=int(train_lengths.max()),candidate_count=len(keys),
        candidate_validation='exactlyone abstractslot; BroadDelimiterLibrary actualspecialspellings/IDs excluded; wrapper1–16384actualtokens',
        rejected_candidate_count=rejected_candidates,
        ranking=ranking,diversified_shortlist=selected,shortlist_rule='descending heuristicUCB; at most3 per family×log2length band',
        elapsed_seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),native_rows=len(x),candidates=len(keys),shortlist=len(selected),
                          reliability=result['reliability'],elapsed_seconds=result['elapsed_seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--broad',type=Path,default=HERE/'outputs/user90-broad-20261104')
    p.add_argument('--precise',type=Path,default=HERE/'outputs/user90-precise-20261104')
    p.add_argument('--scaling',type=Path,default=HERE/'outputs/user90-20261104/length_scaling_seed_templates.json')
    p.add_argument('--output',type=Path,default=HERE/'outputs/user90-precise-20261104/surrogate_ranking_validated.json')
    p.add_argument('--seed',type=int,default=20261104);p.add_argument('--bootstrap',type=int,default=24)
    p.add_argument('--shortlist',type=int,default=32)
    run(p.parse_args())


if __name__=='__main__':main()
