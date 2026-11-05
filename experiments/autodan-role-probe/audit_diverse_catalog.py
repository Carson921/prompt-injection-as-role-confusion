"""Index existing candidate artifacts and audit complete six-case scores.

This CPU-only tool neither generates text nor assigns proposal fitness.  Every
catalog entry points to an already saved JSON string and binds its file hash.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template, SCHEMA, THRESHOLDS

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE/'outputs'
ROLES = ['system','developer','user','cot','assistant','tool']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n')


def write_jsonl(path,rows):
    with Path(path).open('w')as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+'\n')


def compact(result):
    m=result.get('metrics') or {}
    r=m.get('repetition') or {}
    return dict(allowed=result['allowed'],reason_codes=result['reason_codes'],
                wrapper_tokens=m.get('wrapper_tokens'),goal_slots=m.get('goal_slots'),
                max_identical_body_line_copies=r.get('max_identical_body_line_copies'),
                max_identical_body_sentence_copies=r.get('max_identical_body_sentence_copies'),
                max_redundant_body_character_fraction=max([x['redundant_fraction'] for x in r.get('character_ngrams',[])],default=0),
                max_redundant_body_token_fraction=max([x['redundant_fraction'] for x in r.get('token_ngrams',[])],default=0),
                max_redundant_full_character_fraction=max([x['redundant_fraction'] for x in r.get('full_template_character_ngrams',[])],default=0),
                max_redundant_full_token_fraction=max([x['redundant_fraction'] for x in r.get('full_template_token_ngrams',[])],default=0),
                unicode=m.get('unicode'))


def native_score_check(path,dataset):
    """Recompute exact full-goal means from saved native probability matrices.

    This verifies numerical aggregation and record completeness; it does not
    re-run the backbone or independently recompute the frozen probe MLP.
    """
    d=json.loads(path.read_text())
    problems=[]
    def require(condition,code):
        if not condition:problems.append(code)
    require(d.get('status')=='complete' and d.get('complete') is True,'not_complete')
    require(d.get('expected_cases')==6 and d.get('completed_cases')==6,'not_all_six_cases')
    require(d.get('target_role')=='user' and d.get('target_role_index')==2,'wrong_target_role')
    cases=d.get('per_case',[])
    require(len(cases)==6,'case_count')
    means=[];case_evidence=[]
    for index,(case,context) in enumerate(zip(cases,dataset['contexts'])):
        local=[]
        def check(condition,code):
            if not condition:local.append(code)
        check(case.get('case_id')==context['case_id'],'wrong_case_id_or_order')
        check(case.get('status')=='complete','case_incomplete')
        check(case.get('target_role')=='user' and case.get('target_role_index')==2,'case_wrong_role')
        check(case.get('template')==d['template'],'case_template_mismatch')
        check(case.get('actual_goal_sha256')==raw_sha(context['actual_goal']),'goal_changed')
        check(case.get('location',{}).get('scored_payload_occurrence') is None,'unexpected_location_field')
        check(case.get('scored_payload_occurrence')==0,'not_first_payload_occurrence')
        sp=case.get('location',{}).get('payload_special_token_check',{})
        check(sp.get('passed') is True and sp.get('native_special_token_count')==0,'payload_special_check')
        token_path=path.parent/f'case_{index:02d}'/'tokens.npz'
        request_path=token_path.parent/'request.json'
        check(token_path.exists() and request_path.exists(),'missing_native_artifact')
        value=None
        if token_path.exists():
            with np.load(token_path,allow_pickle=False) as a:
                check(a['roles'].tolist()==ROLES,'native_role_order')
                check(str(a['target_role'].item())=='user' and int(a['target_role_index'])==2,'native_wrong_role')
                p=a['probabilities'];mask=a['goal_mask']
                check(mask.dtype==bool and mask.ndim==1 and len(mask)==len(p) and bool(mask.any()),'invalid_goal_mask')
                check(p.ndim==2 and p.shape[1]==6 and np.isfinite(p).all() and bool((p>=0).all()) and bool((p<=1).all()),'invalid_native_probabilities')
                check(int(mask.sum())==case.get('goal_tokens'),'not_full_saved_goal_mask')
                value=float(np.mean(p[mask,2],dtype=np.float64))
                check(abs(value-case.get('goal_user_mean',math.inf))<=1e-12,'case_native_mean_mismatch')
                check(abs(value-case.get('goal_target_mean',math.inf))<=1e-12,'case_target_mean_mismatch')
                check(abs(value-case.get('probabilities',{}).get('goal',{}).get('user',math.inf))<=1e-12,'case_prob_mean_mismatch')
            means.append(value)
        if local:problems.extend(f'case_{index:02d}:{code}'for code in local)
        case_evidence.append(dict(case_id=context['case_id'],goal_tokens=case.get('goal_tokens'),
                                  actual_goal_sha256=case.get('actual_goal_sha256'),
                                  native_probability_mean_user=value,passed=not local,problems=local))
    actual_mean=float(np.mean(means)) if len(means)==6 else None
    for key in ('fitness','goal_target_mean'):
        require(actual_mean is not None and abs(actual_mean-d.get(key,math.inf))<=1e-12,'aggregate_'+key+'_mismatch')
    require(actual_mean is not None and abs(actual_mean-d.get('goal_role_means',{}).get('user',math.inf))<=1e-12,'aggregate_role_mean_mismatch')
    return dict(passed=not problems,problems=problems,recomputed_six_case_mean=actual_mean,
                per_case=case_evidence,method='saved native probability matrices, complete goal masks, equal six-case mean',
                backbone_or_probe_rerun=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUTPUTS/'diverse8h-20261105')
    parser.add_argument('--model',default='/tmp/wz55191-agentdojo-first-tool-model')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    library=BroadDelimiterLibrary(args.model)
    dataset_path=OUTPUTS/'user90-content-20261105'/'dataset.json'
    dataset=json.loads(dataset_path.read_text())
    policy_sha=sha(HERE/'diverse_policy.py')
    score_roots=[OUTPUTS/'user90-precise-20261104',OUTPUTS/'user90-content-20261105']
    grouped={};source_counts={};mismatches=[]
    for root in score_roots:
        files=sorted(root.glob('scores/*/score.json'));source_counts[str(root)]=len(files)
        for path in files:
            d=json.loads(path.read_text());key=raw_sha(d['template'])
            if key in grouped:
                previous=grouped[key]
                if abs(previous['record']['fitness']-d['fitness'])>1e-12:
                    mismatches.append(dict(template_sha256=key,paths=[str(previous['path']),str(path)]))
                previous['aliases'].append(str(previous['path']))
                previous['path'],previous['record']=path,d
            else:grouped[key]=dict(path=path,record=d,aliases=[])
    score_rows=[];reasons=Counter();eligible=[]
    for key,item in sorted(grouped.items()):
        d,path=item['record'],item['path']
        admission=validate_diverse_template(d['template'],tokenizer,library,maximum=16384)
        native=native_score_check(path,dataset)
        row=dict(template_sha256=key,source_path=str(path),source_sha256=sha(path),source_aliases=item['aliases'],
                 policy=compact(admission),native_complete_score_audit=native,
                 eligible_for_native_cache=admission['allowed'] and native['passed'],
                 native_mean_p_user=d['fitness'] if native['passed'] else None,
                 predictions_or_partial_scores_as_fitness=False)
        score_rows.append(row)
        if row['eligible_for_native_cache']:eligible.append(row)
        else:reasons.update(set(admission['reason_codes'])|set(native['problems']))
    eligible.sort(key=lambda x:(-x['native_mean_p_user'],x['template_sha256']))
    intervals=[(0,.05),(.05,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.000001)]
    bins=[]
    for lower,upper in intervals:
        matching=[r for r in eligible if lower<=r['native_mean_p_user']<upper]
        bins.append(dict(lower_inclusive=lower,upper_exclusive=upper,count=len(matching),
                         candidates=[{k:r[k]for k in ('template_sha256','source_path','source_sha256','native_mean_p_user')}|
                                     {'wrapper_tokens':r['policy']['wrapper_tokens']} for r in matching[:3]]))
    score_audit=dict(schema='existing-native-scores-diverse-admission-audit-v1',created_utc=datetime.now(timezone.utc).isoformat(),
                     policy_sha256=policy_sha,policy_schema=SCHEMA,thresholds=THRESHOLDS,
                     source_counts=source_counts,unique_templates=len(grouped),duplicate_score_mismatches=mismatches,
                     complete_numeric_audits_passed=sum(r['native_complete_score_audit']['passed']for r in score_rows),
                     eligible_count=len(eligible),rejected_count=len(score_rows)-len(eligible),rejection_reasons=dict(reasons),
                     best=eligible[0]if eligible else None,probability_intervals=bins,rows=score_rows,
                     dataset_path=str(dataset_path),dataset_sha256=sha(dataset_path),
                     candidate_text_unchanged=True,new_candidate_generation=False)
    write(args.out/'legacy_score_admission.json',score_audit)
    write(args.out/'eligible_native_cache_index.json',dict(schema='diverse-eligible-existing-native-cache-index-v1',
          policy_sha256=policy_sha,score_audit_path=str(args.out/'legacy_score_admission.json'),
          score_audit_sha256=sha(args.out/'legacy_score_admission.json'),entries=eligible))
    write_jsonl(args.out/'legacy_cache_index.jsonl',[
        dict(source_score_path=r['source_path'],source_score_sha256=r['source_sha256'],
             raw_sha256=r['template_sha256'],fitness=r['native_mean_p_user'],
             allowed=r['eligible_for_native_cache'],wrapper_tokens=r['policy']['wrapper_tokens'],
             policy=r['policy'],native_evidence=r['native_complete_score_audit'],policy_sha256=policy_sha)
        for r in eligible])
    print(json.dumps(dict(stage='legacy_scores',unique=len(grouped),eligible=len(eligible),
                         native_audits_passed=score_audit['complete_numeric_audits_passed'],
                         best_p_user=eligible[0]['native_mean_p_user']if eligible else None)),flush=True)

    banks=[OUTPUTS/'user90-20261104'/name for name in ['bootstrap_templates.json','embedding_seed_templates.json',
            'broad_seed_templates.json','feedback/feedback_templates.json']]
    banks += [OUTPUTS/'user90-precise-20261104'/name for name in ['factorial_tail_seed_templates.json','surrogate_shortlist_templates.json']]
    banks += [OUTPUTS/'user90-content-20261105'/'content_seed_preflight_bank.json']
    catalog={};bank_stats=[];all_reasons=Counter();all_rows=[]
    for bank in banks:
        data=json.loads(bank.read_text());records=data['templates'];bank_hash=sha(bank)
        stat=dict(path=str(bank),sha256=bank_hash,entries=len(records),allowed_entries=0,rejection_reasons=Counter())
        for index,record in enumerate(records):
            text=record if isinstance(record,str)else record['template']
            key=raw_sha(text)
            admission=validate_diverse_template(text,tokenizer,library,maximum=16384)
            location=dict(source_path=str(bank),source_sha256=bank_hash,
                          json_pointer=f'/templates/{index}'+(''if isinstance(record,str)else'/template'))
            row=dict(template_sha256=key,source=location,family=record.get('family')if isinstance(record,dict)else None,
                     policy=compact(admission),proposal_only=True,fitness=None)
            all_rows.append(row)
            if admission['allowed']:
                stat['allowed_entries']+=1
                if key in catalog:catalog[key]['source_aliases'].append(location)
                else:
                    catalog[key]=dict(template_sha256=key,**location,source_aliases=[],
                      wrapper_tokens=admission['metrics']['wrapper_tokens'],family=row['family'],
                      existing_native_score_path=grouped[key]['path'].as_posix()if key in grouped else None,
                      proposal_only=True,fitness=None)
            else:
                stat['rejection_reasons'].update(set(admission['reason_codes']))
                all_reasons.update(set(admission['reason_codes']))
        stat['rejection_reasons']=dict(stat['rejection_reasons']);bank_stats.append(stat)
        print(json.dumps(dict(stage='bank',name=bank.name,entries=len(records),allowed=stat['allowed_entries'])),flush=True)
    entries=sorted(catalog.values(),key=lambda r:(r['wrapper_tokens'],r['template_sha256']))
    cap_counts={str(cap):sum(r['wrapper_tokens']<=cap for r in entries)for cap in [128,256,512,1024,2048,4096,8192,16384]}
    report=dict(schema='finite-existing-template-catalog-index-v1',created_utc=datetime.now(timezone.utc).isoformat(),
                policy_sha256=policy_sha,policy_schema=SCHEMA,thresholds=THRESHOLDS,banks=bank_stats,
                original_entry_count=len(all_rows),unique_allowed_templates=len(entries),cap_counts=cap_counts,
                rejection_reasons=dict(all_reasons),entries=entries,
                new_candidate_generation=False,mutation_calls=0,api_calls=0,backbone_forward_calls=0,
                existing_text_references_only=True,prediction_or_fitness=False)
    write(args.out/'static_catalog_index.json',report)
    write(args.out/'static_catalog_admission_rows.json',dict(policy_sha256=policy_sha,rows=all_rows))
    compact_policy={r['template_sha256']:r['policy']for r in all_rows if r['policy']['allowed']}
    source_documents={}
    jsonl_rows=[]
    for entry in entries:
        path=entry['source_path']
        if path not in source_documents:source_documents[path]=json.loads(Path(path).read_text())
        value=source_documents[path]
        for part in entry['json_pointer'].strip('/').split('/'):
            value=value[int(part)]if isinstance(value,list)else value[part.replace('~1','/').replace('~0','~')]
        assert isinstance(value,str) and raw_sha(value)==entry['template_sha256']
        jsonl_rows.append(dict(template=value,raw_sha256=entry['template_sha256'],
             wrapper_tokens=entry['wrapper_tokens'],family=entry['family']or'historical_static',
             source_path=path,source_index=int(entry['json_pointer'].split('/')[2]),
             json_pointer=entry['json_pointer'],source_sha256=entry['source_sha256'],
             source_aliases=entry['source_aliases'],policy=compact_policy[entry['template_sha256']],
             policy_sha256=policy_sha,fitness=None,proposal_only=True))
    write_jsonl(args.out/'static_catalog.jsonl',jsonl_rows)
    print(json.dumps(dict(stage='complete',unique_allowed=len(entries),cap_counts=cap_counts,
                         policy_sha256=policy_sha,catalog_sha256=sha(args.out/'static_catalog_index.json'))),flush=True)


if __name__=='__main__':main()
