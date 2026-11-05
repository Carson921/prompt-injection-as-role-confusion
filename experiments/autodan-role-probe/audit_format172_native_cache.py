"""Read-only CPU inventory of native caches for existing format-only literals.

No model construction, forward pass, checkpoint probability evaluation, candidate
generation, policy change, or running-campaign input write is performed.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from precise_scorer import PreciseScorer
from scorer import DEFAULT_MODEL, DEFAULT_PROBE, load_module

HERE=Path(__file__).resolve().parent
B=HERE/'outputs/diverse8h-20261105'
D=B/'readonly_strict_rejection_diagnostic'
OUT=D/'native_cache_compatibility_v3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(test,message):
    if not test:
        raise ValueError(message)


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=True,allow_nan=False,indent=2)+'\n')


def protocol_for(path):
    for parent in Path(path).parents:
        candidate=parent/'score_protocol.json'
        if candidate.exists():
            return candidate
        if parent==HERE/'outputs':
            break
    raise ValueError('No ancestor score_protocol.json')


def main():
    from transformers import AutoTokenizer
    started=time.time()
    OUT.mkdir(exist_ok=True)
    index_path=D/'reviewed_literal_index_v2.jsonl'
    index=read(D/'reviewed_summary_v2.json')
    require(sha(index_path)==index['reviewed_index_sha256'],'Literal classification index changed')
    literals={row['raw_sha256']:row for row in map(json.loads,index_path.read_text().splitlines())
              if row['diagnostic']['category'].startswith('format')}
    require(len(literals)==172,'Expected exactly the independently reviewed 172 literals')
    inventory_path=D/'native_cache_matches_inventory.json'
    inventory=read(inventory_path)
    matches=sorted(inventory['matched'],key=lambda row:row['path'])
    ref_path=B/'probe_short128_supplement/native/score_protocol.json'
    ref=read(ref_path)
    protocol_digest=hashlib.sha256(json.dumps(ref,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dataset_path=B/'dataset.json';dataset=read(dataset_path)
    run_path=B/'probe_short128_supplement/run_protocol.json';run=read(run_path)
    require(sha(dataset_path)==ref['dataset_sha256']==run['dataset_sha256'],'Current dataset binding mismatch')
    require(sha(DEFAULT_PROBE)==ref['probe_sha256']==run['probe_sha256'],'Current checkpoint binding mismatch')
    require(ref['target_role']=='user' and ref['target_role_index']==2,'Current target mismatch')
    require(Counter(c['suite'] for c in dataset['contexts'])=={'banking':2,'slack':2,'travel':2},'Current six-case balance changed')
    require(len(dataset['contexts'])==6,'Current dataset must contain all six contexts')
    for path,digest in run['sources'].items():
        require(sha(path)==digest,'Current native source binding changed: '+path)
    for relative,digest in ref['code_sha256'].items():
        require(sha(HERE.parents[1]/relative)==digest,'Current feature/render source binding changed: '+relative)
    tokenizer=AutoTokenizer.from_pretrained(DEFAULT_MODEL,local_files_only=True)
    for name,digest in ref['model_binding']['assets_sha256'].items():
        require(sha(DEFAULT_MODEL/name)==digest,'Current native asset binding changed: '+name)
    spans_path=HERE.parents[1]/'experiments/agentdojo-first-tool/spans.py'
    scorer=object.__new__(PreciseScorer)
    scorer.protocol=ref
    scorer.tokenizer=tokenizer
    scorer.spans=load_module('_readonly_format172_native_spans',spans_path)
    scorer.native_special_id_floor=199998
    scorer.native_special_spellings=ref['payload_delimiter_policy']['reject_real_special_spellings']
    config=read(DEFAULT_MODEL/'config.json')
    scorer.model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=config['max_position_embeddings']))
    prepared={};array_cache={};reports=[];protocol_files={}
    def prepare(raw,context):
        key=(raw,context['case_id'])
        if key not in prepared:
            prepared[key]=scorer.prepare(context,literals[raw]['template'])
        return prepared[key]
    for item in matches:
        path=Path(item['path']);raw=item['raw'];score=read(path)
        row=dict(source_score_path=str(path),source_score_sha256=sha(path),raw_sha256=raw,
                 native_score_mtime_unix=path.stat().st_mtime,
                 protocol_input_mask_probability_aggregation_compatible=False,
                 independent_checkpoint_probability_recomputed=False,new_scoring=False)
        try:
            require(raw in literals and score['template'].encode()==literals[raw]['template'].encode()
                    and raw_sha(score['template'])==raw,'Original literal bytes differ')
            protocol_path=protocol_for(path);protocol=read(protocol_path)
            require(protocol==ref,'Source numerical protocol differs from current reference')
            protocol_files[str(protocol_path)]=sha(protocol_path)
            row.update(protocol_path=str(protocol_path),protocol_file_sha256=sha(protocol_path),
                       protocol_canonical_sha256=protocol_digest)
            require(score['protocol_sha256']==protocol_digest,'Aggregate canonical protocol mismatch')
            require(score['status']=='complete' and score['complete'] is True
                    and score['expected_cases']==score['completed_cases']==6
                    and len(score['per_case'])==6,'Not an all-six complete native score')
            require(score['target_role']=='user' and score['target_role_index']==2,'Aggregate is not current user/index2 score')
            means=[];cases=[];total_copies=0
            for index,(context,case) in enumerate(zip(dataset['contexts'],score['per_case'])):
                require(case['case_id']==context['case_id'] and case['suite']==context['suite']
                        and case['status']=='complete','Case identity/order/status mismatch')
                require(case['template'].encode()==score['template'].encode(),'Case literal differs')
                require(case['actual_goal_sha256']==raw_sha(context['actual_goal']),'Immutable full goal differs')
                require(case['target_role']=='user' and case['target_role_index']==2,'Case target differs')
                require(case['protocol_sha256']==protocol_digest,'Case canonical protocol mismatch')
                require(case['scored_payload_occurrence']==0,'Not the first complete payload occurrence')
                request,prompt,ids,offsets,positions,goal,location,input_sha=prepare(raw,context)
                token_path=Path(case['token_artifact']);folder=token_path.parent
                require(token_path.name=='tokens.npz','Native token artifact has an unexpected filename')
                require({'tokens.npz','request.json'}<=set(case['artifact_sha256']),'Required native artifact SHA missing')
                hashes={name:sha(folder/name) for name in case['artifact_sha256']}
                require(hashes==case['artifact_sha256'],'Native artifact SHA mismatch')
                require(sha(token_path)==hashes['tokens.npz'],'Loaded native token artifact is not the bound tokens.npz')
                saved=read(folder/'request.json')
                require(saved['rendered_prompt']==prompt,'Complete rendered prompt differs')
                require(all(saved[key]==value for key,value in request.items()),'Complete semantic request differs')
                require(saved['actual_goal']==context['actual_goal'],'Saved immutable full goal differs')
                require(saved['input_sha256']==input_sha==case['input_sha256'],'Input token SHA differs')
                require(saved['input_tokens']==len(ids)==case['input_tokens'],'Complete input length differs')
                for key in ('payload_char_span','goal_char_span'):
                    require(np.array_equal(saved['location'][key],location[key]),'Saved first '+key+' differs')
                    require(np.array_equal(case['location'][key],location[key]),'Recorded first '+key+' differs')
                require(len(saved['location']['occurrences'])==len(case['location']['occurrences'])==len(location['occurrences']),
                        'Saved payload occurrence count differs')
                for old,current in zip(saved['location']['occurrences'],location['occurrences']):
                    for key in ('payload_char_span','goal_char_span'):
                        require(np.array_equal(old[key],current[key]),'Saved occurrence '+key+' differs')
                for old,current in zip(case['location']['occurrences'],location['occurrences']):
                    for key in ('payload_char_span','goal_char_span'):
                        require(np.array_equal(old[key],current[key]),'Case occurrence '+key+' differs')
                key=(hashes['tokens.npz'],raw,context['case_id'])
                if key not in array_cache:
                    with np.load(token_path,allow_pickle=False) as arrays:
                        comparisons=[('input_ids',ids),('input_token_offsets',offsets),
                          ('payload_token_indices',positions),('goal_mask',goal),('template_mask',~goal),
                          ('head_ids',ref['head_ids']),('roles',ref['roles']),
                          ('payload_char_span',location['payload_char_span']),('goal_char_span',location['goal_char_span']),
                          ('payload_char_spans',[o['payload_char_span'] for o in location['occurrences']]),
                          ('goal_char_spans',[o['goal_char_span'] for o in location['occurrences']])]
                        for name,current in comparisons:
                            expected=np.asarray(current)
                            require(arrays[name].dtype==expected.dtype
                                    and arrays[name].shape==expected.shape
                                    and arrays[name].tobytes()==expected.tobytes(),
                                    'Current deterministic '+name+' bytes/shape/dtype differ')
                        require(str(arrays['target_role'].item())=='user' and int(arrays['target_role_index'])==2,
                                'Native arrays target differs')
                        mask=arrays['goal_mask'];p=arrays['probabilities']
                        require(mask.dtype==bool and mask.ndim==1 and len(mask)==len(positions) and mask.any(),
                                'Complete deterministic goal mask invalid')
                        require(p.dtype==np.float32 and p.shape==(len(positions),6) and np.isfinite(p).all()
                                and ((p>=0)&(p<=1)).all() and np.allclose(p.sum(1),1,atol=2e-6),
                                'Saved six-role probabilities invalid')
                        bits=arrays['native_bf16_bits'];features=arrays['features_fp16']
                        require(bits.dtype==np.uint16 and bits.shape==(len(positions),16,64),'Saved native BF16 shape/dtype differs')
                        require(features.dtype==np.float16 and features.shape==(len(positions),1024),'Saved feature shape/dtype differs')
                        recovered=(bits.astype(np.uint32)<<16).view(np.float32).reshape(len(positions),1024).astype(np.float16)
                        require(np.isfinite(features).all() and np.array_equal(features,recovered),'Native BF16 to FP16 capture conversion differs')
                        mean=float(np.mean(p[mask,2],dtype=np.float64))
                        array_cache[key]=dict(mean=mean,goal_tokens=int(mask.sum()))
                values=array_cache[key];mean=values['mean']
                require(values['goal_tokens']==case['goal_tokens']==int(goal.sum()),'Full goal-token count differs')
                for value in (case['goal_user_mean'],case['goal_target_mean'],case['probabilities']['goal']['user']):
                    require(abs(value-mean)<=1e-12,'Saved current-user goal mean differs')
                means.append(mean);copies=len(location['occurrences']);total_copies+=copies
                require(copies==case['observed_payload_occurrences'],'Recorded payload-copy count differs')
                cases.append(dict(case_id=context['case_id'],input_tokens=len(ids),complete_goal_tokens=int(goal.sum()),
                     payload_occurrences=copies,input_sha256=input_sha,saved_goal_user_mean=mean,
                     tokens_npz_path=str(token_path),tokens_npz_sha256=hashes['tokens.npz'],
                     request_path=str(folder/'request.json'),request_sha256=hashes['request.json'],
                     complete_input_and_deterministic_goal_mask_equal=True))
            require(total_copies==9,'Expected exactly nine unchanged payload occurrences across six contexts')
            aggregate=float(np.mean(means))
            for value in (score['fitness'],score['goal_target_mean'],score['goal_role_means']['user']):
                require(abs(value-aggregate)<=1e-12,'Saved equal-six current-user aggregate differs')
            row.update(protocol_input_mask_probability_aggregation_compatible=True,
                       saved_complete_six_mean_user=aggregate,per_case=cases,
                       current_checkpoint_binding_sha256=ref['probe_sha256'])
        except Exception as error:
            row['incompatibility']=type(error).__name__+': '+str(error)
        reports.append(row)
        print(json.dumps({'checked':len(reports),'total':len(matches),'compatible':row['protocol_input_mask_probability_aggregation_compatible'],
                          'raw_sha256':raw,'incompatibility':row.get('incompatibility')}),flush=True)
    compatible={r['raw_sha256'] for r in reports if r['protocol_input_mask_probability_aggregation_compatible']}
    any_cache={r['raw_sha256'] for r in reports}
    final_records_path=B/'probe_short128_supplement/score_records.jsonl'
    final_records={r['raw_sha256'] for r in map(json.loads,final_records_path.read_text().splitlines())}
    per_literal=[]
    for raw,literal in sorted(literals.items()):
        rows=[r for r in reports if r['raw_sha256']==raw]
        per_literal.append(dict(raw_sha256=raw,template=literal['template'],wrapper_tokens=literal['wrapper_tokens'],
            source_reference={k:literal[k] for k in ('source_path','source_sha256','json_pointer','source_format')},
            category=literal['diagnostic']['category'],existing_score_records=len(rows),
            compatible_complete_current_user_native_cache=raw in compatible,
            included_in_current_fixed_short_complete_records=raw in final_records,
            matched_score_paths=[r['source_score_path'] for r in rows],
            compatibility_status='compatible_existing_cache' if raw in compatible else 'existing_cache_incompatible' if rows else 'no_existing_aggregate_native_score_found',
            candidates_restored_or_evaluated=False))
    details_path=OUT/'cache_records.json';write(details_path,dict(records=reports))
    literals_path=OUT/'literal_coverage.json';write(literals_path,dict(literals=per_literal))
    sources=[Path(__file__).resolve(),HERE/'precise_scorer.py',HERE/'deterministic_spans.py',HERE/'scorer.py',spans_path]
    summary=dict(schema='readonly-format172-native-cache-compatibility-v1',status='complete',started_unix=started,completed_unix=time.time(),
        reviewed_literal_index_path=str(index_path),reviewed_literal_index_sha256=sha(index_path),
        fixed_existing_literal_population=172,aggregate_score_files_inventoried=inventory['aggregate_shape_files'],
        matching_aggregate_score_records=len(reports),matching_unique_literals=len(any_cache),
        compatible_existing_native_score_records=sum(r['protocol_input_mask_probability_aggregation_compatible'] for r in reports),
        compatible_unique_literals=len(compatible),incompatible_only_unique_literals=len(any_cache-compatible),
        no_existing_aggregate_score_unique_literals=len(set(literals)-any_cache),
        excluded_from_current_fixed_short_records=len(set(literals)-final_records),
        compatible_but_excluded_from_current_fixed_short_records=len(compatible-final_records),
        source_numerical_protocols=protocol_files,reference_protocol_path=str(ref_path),reference_protocol_file_sha256=sha(ref_path),
        reference_protocol_canonical_sha256=protocol_digest,dataset_path=str(dataset_path),dataset_sha256=sha(dataset_path),
        checkpoint_path=str(DEFAULT_PROBE),checkpoint_sha256=sha(DEFAULT_PROBE),run_protocol_path=str(run_path),run_protocol_sha256=sha(run_path),
        current_complete_score_records_path=str(final_records_path),current_complete_score_records_sha256=sha(final_records_path),
        source_bindings={str(p):sha(p) for p in sources},cache_records_path=str(details_path),cache_records_sha256=sha(details_path),
        literal_coverage_path=str(literals_path),literal_coverage_sha256=sha(literals_path),
        new_native_scores=0,model_forward_calls=0,checkpoint_probability_recomputations=0,API_calls=0,GPU_used=False,
        candidate_generation=False,template_mutation=False,policy_changed=False,current_campaign_changed=False,
        evidence_limit='Protocol-bound current checkpoint and exact complete inputs/deterministic goal masks; saved native capture conversion and P(user) matrix aggregation verified. This is not an independent current-checkpoint MLP probability recomputation, and does not restore any candidate or authorize new evaluation.')
    write(OUT/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_bindings','source_numerical_protocols')}),flush=True)


if __name__=='__main__':
    main()
