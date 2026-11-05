"""Independent CPU audit of already evaluated fixed-catalog candidates.

Rebuilds exact complete running contexts and all payload copies without model
execution, then recomputes the frozen role-probe MLP from saved native features.
It never generates candidates, alters goals or writes into scorer artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from audit_diverse_catalog import native_score_check
from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from precise_scorer import PreciseScorer
from scorer import dependencies

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',type=Path,required=True,help='Existing winner/complete candidate JSON')
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run-protocol',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',default='/tmp/wz55191-agentdojo-first-tool-model')
    parser.add_argument('--probe',type=Path,default=HERE.parents[1]/'experiments/role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    row=json.loads(args.record.read_text())
    write(args.out/'candidate_record_snapshot.json',row)
    source=Path(row['source_score_path'])
    assert sha(source)==row['source_score_sha256']
    score=json.loads(source.read_text())
    assert score['template']==row['template']
    assert hashlib.sha256(score['template'].encode('utf-8')).hexdigest()==row['raw_sha256']
    run=json.loads(args.run_protocol.read_text())
    for name,digest in run['sources'].items():assert sha(name)==digest,(name,'frozen source changed')
    assert sha(args.dataset)==run['dataset_sha256']
    assert sha(args.probe)==run['probe_sha256']
    source_reference=row['original_catalog_source']
    original_path=Path(source_reference['source_path'])
    assert sha(original_path)==source_reference['source_sha256']
    original=json.loads(original_path.read_text())
    for part in source_reference['json_pointer'].strip('/').split('/'):
        original=original[int(part)]if isinstance(original,list)else original[part.replace('~1','/').replace('~0','~')]
    if isinstance(original,dict):original=original['template']
    assert original.encode('utf-8')==score['template'].encode('utf-8')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    library=BroadDelimiterLibrary(args.model)
    admission=validate_diverse_template(score['template'],tokenizer,library,maximum=row['evaluated_cap'])
    assert admission['allowed'],admission['reason_codes']
    assert admission['metrics']['wrapper_tokens']==row['wrapper_tokens']
    native_root=source.parents[2]
    protocol_path=native_root/'score_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    dataset=json.loads(args.dataset.read_text())
    assert protocol['target_role']=='user'and protocol['target_role_index']==2
    assert protocol['dataset_sha256']==sha(args.dataset)
    scorer=object.__new__(PreciseScorer)
    scorer.protocol=protocol
    scorer.tokenizer=tokenizer
    scorer.spans=dependencies()[1]
    scorer.native_special_id_floor=199998
    scorer.native_special_spellings=protocol['payload_delimiter_policy']['reject_real_special_spellings']
    model_config=json.loads((Path(args.model)/'config.json').read_text())
    scorer.model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=model_config['max_position_embeddings']))
    case_rows=[]
    for index,(context,case)in enumerate(zip(dataset['contexts'],score['per_case'])):
        request,prompt,ids,offsets,positions,goal,location,input_sha=scorer.prepare(context,score['template'])
        artifact=Path(case['token_artifact']);saved_request=json.loads((artifact.parent/'request.json').read_text())
        assert saved_request['rendered_prompt']==prompt
        for key,value in request.items():assert saved_request[key]==value,(index,key)
        assert saved_request['actual_goal']==context['actual_goal']
        assert saved_request['payload_text']==score['template'].replace('{goal}',context['actual_goal'])
        assert saved_request['location']==location
        assert saved_request['input_sha256']==input_sha==case['input_sha256']
        with np.load(artifact,allow_pickle=False)as arrays:
            assert np.array_equal(arrays['input_ids'],ids)
            assert np.array_equal(arrays['input_token_offsets'],offsets)
            assert np.array_equal(arrays['payload_token_indices'],positions)
            assert np.array_equal(arrays['goal_mask'],goal)
            assert np.array_equal(arrays['template_mask'],~goal)
            assert np.array_equal(arrays['payload_char_spans'],[x['payload_char_span']for x in location['occurrences']])
            assert np.array_equal(arrays['goal_char_spans'],[x['goal_char_span']for x in location['occurrences']])
        case_rows.append(dict(case_id=case['case_id'],input_tokens=len(ids),goal_tokens=int(goal.sum()),
          payload_copies=len(location['occurrences']),source_goal_sha256=case['actual_goal_sha256'],
          full_input_and_offsets_bitwise_equal=True,complete_first_goal_mask_bitwise_equal=True,
          payload_special_token_check=location['payload_special_token_check'],
          original_request_and_full_rendered_prompt_equal=True))
    aggregate=native_score_check(source,dataset)
    assert aggregate['passed'],aggregate['problems']
    recompute=args.out/'probe_recompute'
    (recompute/'scores').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(protocol_path,recompute/'score_protocol.json')
    reference=recompute/'scores'/source.parent.name
    if not reference.exists():reference.symlink_to(source.parent,target_is_directory=True)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    completed=subprocess.run([sys.executable,str(HERE/'audit.py'),'--output',str(recompute),
        '--dataset',str(args.dataset),'--probe',str(args.probe),'--target-role','user'],
        env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (recompute/'audit.log').write_text(completed.stdout)
    assert completed.returncode==0,completed.stdout[-2000:]
    numeric=json.loads((recompute/'independent_audit.json').read_text())
    assert numeric['status']=='passed' and numeric['candidates']==1
    report=dict(schema='diverse-fixed-catalog-native-candidate-independent-audit-v1',status='passed',
       source_record=str(args.record),record_snapshot_sha256=sha(args.out/'candidate_record_snapshot.json'),
       raw_sha256=row['raw_sha256'],native_mean_p_user=score['fitness'],wrapper_tokens=row['wrapper_tokens'],
       source_original=source_reference,source_text_utf8_bytes_identical=True,
       policy_sha256=sha(HERE/'diverse_policy.py'),policy_allowed=admission['allowed'],policy=admission,
       native_score_path=str(source),native_score_sha256=sha(source),
       all_frozen_sources_unchanged=True,dataset_sha256=sha(args.dataset),probe_sha256=sha(args.probe),
       full_six_contexts_reconstructed=True,per_case=case_rows,aggregate=aggregate,
       independent_native_feature_probe_recompute={k:v for k,v in numeric.items()if k!='ranking'},
       backbone_forward_calls=0,api_calls=0,new_candidate_generation=False,
       audit_source_sha256=sha(__file__))
    write(args.out/'audit.json',report)
    print(json.dumps(dict(status='passed',raw_sha256=row['raw_sha256'],p_user=score['fitness'],
      wrapper_tokens=row['wrapper_tokens'],contexts=6,payload_copies=sum(x['payload_copies']for x in case_rows),
      max_probability_error=numeric['max_probability_absolute_error'],out=str(args.out))),flush=True)


if __name__=='__main__':main()
