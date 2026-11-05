"""CPU-only observation of the frozen existing-candidate short probe run."""
import hashlib
import json
from pathlib import Path
import time

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from evaluate_static_catalog import pointer_value,verify_complete
from strict_header_policy import strict_header_metrics

HERE=Path(__file__).resolve().parent
B=HERE/'outputs/diverse8h-20261105'
P=B/'probe_short128'
OUT=B/'audit_short128_supervision'
PID=3026664
START='548775130'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n')


def alive():
    stat=Path(f'/proc/{PID}/stat')
    try:
        fields=stat.read_text().split()
        return fields[21]==START and fields[2]!='Z'
    except FileNotFoundError:return False


def main():
    OUT.mkdir(exist_ok=True)
    run=json.loads((P/'run_protocol.json').read_text())
    short=json.loads((P/'short_constraint_protocol.json').read_text())
    bindings={**run['sources'],**short['sources'],**short['catalogs']}
    frozen={P/'run_protocol.json':sha(P/'run_protocol.json'),P/'short_constraint_protocol.json':sha(P/'short_constraint_protocol.json')}
    assert run['caps']==[128]and short['maximum_wrapper_tokens']==128
    assert all(short[key]is False for key in ('automatic_length_expansion','text_generation','template_mutation','API_generation'))
    assert short['strict_duplicate_headers_lines_paragraphs_excluded']and short['full_six_native_scores_only']
    dataset_path=B/'dataset.json';dataset=json.loads(dataset_path.read_text())
    assert sha(dataset_path)==short['dataset_sha256']==run['dataset_sha256']
    catalog={}
    for name,digest in short['catalogs'].items():
        assert sha(name)==digest
        for line in Path(name).read_text().splitlines():
            row=json.loads(line);catalog[row['raw_sha256']]=row
    assert len(catalog)==938
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/tmp/wz55191-agentdojo-first-tool-model',local_files_only=True)
    library=BroadDelimiterLibrary('/tmp/wz55191-agentdojo-first-tool-model')
    verified={};sources={};source_hashes={};observations=[]
    while True:
        for path,digest in bindings.items():assert sha(path)==digest,path
        for path,digest in frozen.items():assert sha(path)==digest,path
        all_bytes=(P/'score_records.jsonl').read_bytes();lines=all_bytes.decode().splitlines()
        records={}
        for index,line in enumerate(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:
                assert index==len(lines)-1 and alive(),'Malformed completed score record'
                continue
            raw=row['raw_sha256'];records[raw]=row
            if raw in verified:continue
            source=Path(row['source_score_path'])
            assert sha(source)==row['source_score_sha256']
            score=json.loads(source.read_text());text=score['template']
            assert hashlib.sha256(text.encode()).hexdigest()==raw and text==row['template']
            assert score.get('expected_cases')==score.get('completed_cases')==6
            assert all(case.get('status')=='complete'for case in score['per_case'])
            value=verify_complete(score,dataset['contexts'])
            assert abs(value-row['fitness'])<=1e-12
            policy=validate_diverse_template(text,tokenizer,library,maximum=128)
            assert policy['allowed']and strict_header_metrics(text)['allowed']
            assert policy['metrics']['wrapper_tokens']==row['wrapper_tokens']<=128
            if row['origin']=='fresh_fixed_catalog':
                assert raw in catalog and row['evaluated_cap']==128
                reference=row['original_catalog_source'];entry=catalog[raw]
                assert reference=={k:entry[k]for k in ('source_path','source_sha256','json_pointer')}
                original_path=Path(reference['source_path'])
                if original_path not in sources:
                    assert sha(original_path)==reference['source_sha256']
                    sources[original_path]=json.loads(original_path.read_text());source_hashes[str(original_path)]=sha(original_path)
                assert pointer_value(sources[original_path],reference['json_pointer']).encode()==text.encode()
            else:assert row['origin']=='verified_complete_legacy'
            verified[raw]=dict(source_score_path=str(source),source_score_sha256=sha(source),
                fitness=value,wrapper_tokens=row['wrapper_tokens'],origin=row['origin'],
                all_six_complete_actual_user_index2=True,strict_header_allowed=True)
        checkpoint=json.loads((P/'checkpoint.json').read_text())
        assert checkpoint['cap']==128 and checkpoint['phase_index']==0
        observation=dict(unix=time.time(),pid=PID,process_same_identity_alive=alive(),
          checkpoint_status=checkpoint['status'],complete_records=len(records),
          fresh_records=sum(r['origin']=='fresh_fixed_catalog'for r in records.values()),
          verified_records=len(verified),cap=128,all_frozen_sources_unchanged=True)
        observations.append(observation)
        with (OUT/'observations.jsonl').open('a')as handle:handle.write(json.dumps(observation)+'\n')
        print(json.dumps(observation),flush=True)
        if not observation['process_same_identity_alive']:
            assert checkpoint['status']!='scoring','Process missing before a terminal checkpoint'
            rejections={}
            rejection_path=P/'rejections.jsonl'
            if rejection_path.exists():
                for line in rejection_path.read_text().splitlines():
                    row=json.loads(line);rejections[row['raw_sha256']]=row
            remaining=sorted(set(catalog)-set(records)-set(rejections))
            assert all(sha(path)==digest for path,digest in source_hashes.items())
            report=dict(schema='independent-short128-static-native-supervision-v1',status='passed',
              pid=PID,process_start_ticks=START,terminal_checkpoint=checkpoint,
              frozen_source_bindings=bindings,frozen_run_protocols={str(k):v for k,v in frozen.items()},
              catalog_candidates=len(catalog),complete_records=len(records),
              catalog_complete=len(set(catalog)&set(records)),catalog_rejected=len(set(catalog)&set(rejections)),
              catalog_remaining=len(remaining),remaining_raw_sha256=remaining,
              fresh_complete=sum(r['origin']=='fresh_fixed_catalog'for r in records.values()),
              reused_complete=sum(r['origin']=='verified_complete_legacy'for r in records.values()),
              source_template_bindings=source_hashes,all_records=verified,
              maximum_wrapper_tokens=128,strict_headers_lines_paragraphs=True,
              full_six_actual_user_role_index2_only=True,no_generated_text=True,
              no_mutations=True,no_APIs=True,backbone_forwards_by_auditor=0,
              source_score_records_sha256=sha(P/'score_records.jsonl'),audit_source_sha256=sha(__file__),
              scope='Finite existing-candidate evaluation; not new mutation search or an eight-hour generation campaign')
            write(OUT/'audit.json',report)
            break
        time.sleep(60)


if __name__=='__main__':main()
