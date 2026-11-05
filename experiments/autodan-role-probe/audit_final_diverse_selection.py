"""Audit a frozen ASR selection, reusing hash-identical native CPU evidence.

No candidate generation, backbone execution, mutation, API use or source edits.
Native checkpoint recomputation is reused only when the score and every native
token/request artifact are byte-identical to a previously verified candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from audit_diverse_catalog import native_score_check
from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from strict_header_policy import strict_header_metrics

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--probe-root',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',default='/tmp/wz55191-agentdojo-first-tool-model')
    parser.add_argument('--maximum-wrapper-tokens',type=int,default=128)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest_bytes=args.manifest.read_bytes()
    manifest=json.loads(manifest_bytes)
    (args.out/'asr_candidates_final_snapshot.json').write_bytes(manifest_bytes)
    entries=manifest['candidates']if isinstance(manifest,dict)else manifest
    assert isinstance(entries,list)and 3<=len(entries)<=6
    dataset=json.loads(args.dataset.read_text())
    run_protocol=args.probe_root/'run_protocol.json'
    run=json.loads(run_protocol.read_text())
    assert sha(args.dataset)==run['dataset_sha256']
    for path,digest in run['sources'].items():assert sha(path)==digest,path
    records={}
    for line in (args.probe_root/'score_records.jsonl').read_text().splitlines():
        if not line.strip():continue
        record=json.loads(line)
        records[record['raw_sha256']]=record
    prior_proofs=[]
    base=args.probe_root.parent
    for pattern in ('audit_best*/audit.json','audit_replacement*/audit.json','audit_intervals_snapshot/candidate_*/audit.json'):
        prior_proofs.extend(sorted(base.glob(pattern)))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    library=BroadDelimiterLibrary(args.model)
    rows=[];seen=set();intervals=[]
    for index,entry in enumerate(entries):
        text=entry['template'];raw=raw_sha(text)
        assert raw not in seen
        seen.add(raw)
        assert text not in dataset['conditions'].values()
        assert entry.get('raw_sha256',raw)==raw
        row=records[raw]
        assert row['template']==text
        native_path=Path(entry['score_path'])
        if not native_path.is_absolute():native_path=(args.manifest.parent/native_path).resolve()
        assert native_path==Path(row['source_score_path'])
        assert sha(native_path)==row['source_score_sha256']
        score=json.loads(native_path.read_text())
        assert score['template'].encode('utf-8')==text.encode('utf-8')
        admission=validate_diverse_template(text,tokenizer,library,maximum=args.maximum_wrapper_tokens)
        assert admission['allowed'],admission['reason_codes']
        short_headers=strict_header_metrics(text)
        assert short_headers['allowed'],short_headers
        lower,upper=entry['p_user_interval']
        assert 0<=lower<upper<=1
        assert lower<=score['fitness']<upper or upper==score['fitness']==1
        assert all(upper<=a or lower>=b for a,b in intervals)
        intervals.append((lower,upper))
        origin=row['original_catalog_source']
        assert sha(origin['source_path'])==origin['source_sha256']
        source_text=json.loads(Path(origin['source_path']).read_text())
        for part in origin['json_pointer'].strip('/').split('/'):
            source_text=source_text[int(part)]if isinstance(source_text,list)else source_text[part.replace('~1','/').replace('~0','~')]
        if isinstance(source_text,dict):source_text=source_text['template']
        assert source_text.encode('utf-8')==text.encode('utf-8')
        artifacts=[]
        for case in score['per_case']:
            token_path=Path(case['token_artifact'])
            for name,digest in case['artifact_sha256'].items():
                bound=token_path.parent/name
                assert sha(bound)==digest,(bound,'artifact changed')
                artifacts.append(dict(path=str(bound),sha256=digest))
        proof_path=None;proof=None
        for candidate in prior_proofs:
            candidate_proof=json.loads(candidate.read_text())
            if candidate_proof.get('status')=='passed' and candidate_proof.get('raw_sha256')==raw and candidate_proof.get('native_score_sha256')==sha(native_path):
                proof_path,proof=candidate,candidate_proof
                break
        reused=proof is not None
        if reused:
            assert proof['dataset_sha256']==sha(args.dataset)
            assert proof['probe_sha256']==run['probe_sha256']
            assert proof['policy_sha256']==sha(HERE/'diverse_policy.py')
            assert proof['audit_source_sha256']==sha(HERE/'audit_diverse_native.py')
            assert proof['source_text_utf8_bytes_identical']and proof['full_six_contexts_reconstructed']
            assert all(case['complete_first_goal_mask_bitwise_equal']and case['full_input_and_offsets_bitwise_equal']
                       and case['payload_special_token_check']['native_special_token_count']==0 for case in proof['per_case'])
            assert proof['independent_native_feature_probe_recompute']['status']=='passed'
        else:
            snapshot=args.out/f'candidate_{index:02d}_record.json'
            write(snapshot,row)
            child=args.out/f'candidate_{index:02d}_native_recompute'
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            result=subprocess.run([sys.executable,str(HERE/'audit_diverse_native.py'),'--record',str(snapshot),
               '--dataset',str(args.dataset),'--run-protocol',str(run_protocol),'--out',str(child),
               '--model',args.model],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (args.out/f'candidate_{index:02d}_native_recompute.log').write_text(result.stdout)
            assert result.returncode==0,result.stdout[-2000:]
            proof_path=child/'audit.json';proof=json.loads(proof_path.read_text())
        aggregate=native_score_check(native_path,dataset)
        assert aggregate['passed'],aggregate['problems']
        rows.append(dict(condition='candidate_'+raw[:16],raw_sha256=raw,p_user_interval=[lower,upper],
          native_mean_p_user=score['fitness'],wrapper_tokens=admission['metrics']['wrapper_tokens'],
          native_score_path=str(native_path),native_score_sha256=sha(native_path),source_original=origin,
          policy_allowed=True,source_text_utf8_bytes_identical=True,
          strict_header_selection=short_headers,
          artifact_hashes_rechecked=artifacts,aggregate_recomputed=aggregate,
          native_evidence_reused=reused,native_evidence_path=str(proof_path),native_evidence_sha256=sha(proof_path),
          native_checkpoint_recompute_max_error=proof['independent_native_feature_probe_recompute']['max_probability_absolute_error'],
          status='passed'))
        print(json.dumps(dict(raw_sha256=raw,p_user=score['fitness'],evidence_reused=reused,status='passed')),flush=True)
    report=dict(schema='diverse-final-asr-selection-independent-audit-v1',status='passed',
       manifest_path=str(args.manifest.resolve()),manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
       manifest_snapshot_path=str((args.out/'asr_candidates_final_snapshot.json').resolve()),
       dataset_sha256=sha(args.dataset),probe_checkpoint_sha256=run['probe_sha256'],
       candidate_count=len(rows),candidates=rows,all_frozen_scoring_sources_unchanged=True,
       no_plain_or_remember_candidates=True,complete_six_case_fitness_only=True,
       maximum_wrapper_tokens=args.maximum_wrapper_tokens,
       strict_header_selection_source_sha256=sha(HERE/'strict_header_policy.py'),
       policy_sha256=sha(HERE/'diverse_policy.py'),audit_source_sha256=sha(__file__),
       native_recompute_reuse='Only exact score SHA, every native/request artifact SHA, dataset, checkpoint, source helper and source text match permit reuse',
       new_candidate_generation=False,backbone_forward_calls=0,api_calls=0)
    steering=base/'user_steering_stop.json'
    if steering.exists():
        steering_data=json.loads(steering.read_text())
        checkpoint=args.probe_root/'checkpoint.json'
        checkpoint_data=json.loads(checkpoint.read_text())
        assert steering_data['original_protocol_sha256']==steering_data['final_original_protocol_sha256']==sha(run_protocol)
        assert steering_data['completed_score_records_sha256']==sha(args.probe_root/'score_records.jsonl')
        assert steering_data['maximum_final_wrapper_tokens']==args.maximum_wrapper_tokens
        assert checkpoint_data['status']==steering_data['checkpoint_status_will_remain_original']=='interrupted'
        report['probe_handoff']=dict(user_steering_stop_path=str(steering),user_steering_stop_sha256=sha(steering),
          checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),
          original_checkpoint_status=checkpoint_data['status'],
          explanation='Explicit user steering stopped the old expansion campaign; original interrupted checkpoint is preserved')
    write(args.out/'audit.json',report)
    print(json.dumps(dict(status='passed',candidates=len(rows),reused=sum(r['native_evidence_reused']for r in rows),
      manifest_sha256=report['manifest_sha256'],audit_path=str(args.out/'audit.json'))),flush=True)


if __name__=='__main__':main()
