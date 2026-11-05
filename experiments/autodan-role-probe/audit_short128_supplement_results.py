"""Independent final winner/interval audit of the finite short catalog run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from audit_diverse_catalog import native_score_check
from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from evaluate_static_catalog import pointer_value
from strict_header_policy import strict_header_metrics

HERE=Path(__file__).resolve().parent
B=HERE/'outputs/diverse8h-20261105'
P=B/'probe_short128_supplement'
OUT=B/'audit_short128_supplement_final'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    coverage_path=B/'audit_short128_supplement_supervision/audit.json'
    coverage=json.loads(coverage_path.read_text())
    assert coverage['status']=='passed'
    assert coverage['source_score_records_sha256']==sha(P/'score_records.jsonl')
    for path,digest in coverage['frozen_source_bindings'].items():assert sha(path)==digest
    dataset_path=B/'dataset.json';dataset=json.loads(dataset_path.read_text())
    run=json.loads((P/'run_protocol.json').read_text())
    short=json.loads((P/'short_constraint_protocol.json').read_text())
    assert run['caps']==[128]and short['maximum_wrapper_tokens']==128
    assert not short['automatic_length_expansion']and not short['template_mutation']and not short['API_generation']
    assert sha(dataset_path)==run['dataset_sha256']==short['dataset_sha256']
    records={row['raw_sha256']:row for row in map(json.loads,(P/'score_records.jsonl').read_text().splitlines())}
    catalog={}
    for path,digest in short['catalogs'].items():
        assert sha(path)==digest
        for line in Path(path).read_text().splitlines():
            row=json.loads(line);catalog[row['raw_sha256']]=row
    original_records={row['raw_sha256']:row for row in map(json.loads,(B/'probe/score_records.jsonl').read_text().splitlines())}
    for row in map(json.loads,(B/'probe_short128/score_records.jsonl').read_text().splitlines()):
        if row.get('original_catalog_source')or row['raw_sha256']not in original_records:
            original_records[row['raw_sha256']]=row
    winner_bytes=(P/'winner.json').read_bytes();winner=json.loads(winner_bytes)
    assert abs(winner['fitness']-max(row['fitness']for row in records.values()))<=1e-12
    interval_bytes=(P/'interval_candidates.json').read_bytes();interval=json.loads(interval_bytes)
    (OUT/'winner_snapshot.json').write_bytes(winner_bytes)
    (OUT/'interval_candidates_snapshot.json').write_bytes(interval_bytes)
    targets=interval['candidates']
    assert winner['raw_sha256']in {row['raw_sha256']for row in targets}
    prior=[]
    for pattern in ('audit_best*/audit.json','audit_replacement*/audit.json',
                    'audit_intervals_snapshot/candidate_*/audit.json','audit_short128_firstfresh/audit.json',
                    'audit_short128_final/candidate_*_native_recompute/audit.json'):
        prior.extend(sorted(B.glob(pattern)))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/tmp/wz55191-agentdojo-first-tool-model',local_files_only=True)
    library=BroadDelimiterLibrary('/tmp/wz55191-agentdojo-first-tool-model')
    audits=[]
    for index,target in enumerate(targets):
        raw=target['raw_sha256'];row=records[raw];text=row['template']
        assert target['template']==text and hashlib.sha256(text.encode()).hexdigest()==raw
        lower,upper=target['p_user_interval']
        assert lower<=row['fitness']and(row['fitness']<upper or row['fitness']==upper==1)
        best=max(r['fitness']for r in records.values()if lower<=r['fitness']<upper or r['fitness']==upper==1)
        assert abs(best-row['fitness'])<=1e-12
        source=Path(row['source_score_path']);assert str(source)==target['score_path']
        assert sha(source)==row['source_score_sha256']
        score=json.loads(source.read_text())
        policy=validate_diverse_template(text,tokenizer,library,maximum=128)
        strict=strict_header_metrics(text)
        assert policy['allowed']and strict['allowed']and policy['metrics']['wrapper_tokens']<=128
        aggregate=native_score_check(source,dataset);assert aggregate['passed'],aggregate['problems']
        origin=row.get('original_catalog_source')or original_records.get(raw,{}).get('original_catalog_source')
        if origin is None:
            entry=catalog[raw];origin={key:entry[key]for key in ('source_path','source_sha256','json_pointer')}
        assert sha(origin['source_path'])==origin['source_sha256']
        original=pointer_value(json.loads(Path(origin['source_path']).read_text()),origin['json_pointer'])
        assert original.encode()==text.encode()
        derived_copy=bool(catalog.get(raw,{}).get('source_is_derived_copy'))
        jsonl_lineage=catalog.get(raw,{}).get('original_jsonl_lineage')if derived_copy else None
        if derived_copy:
            original_jsonl=Path(jsonl_lineage['source_path'])
            assert sha(original_jsonl)==jsonl_lineage['source_sha256']
            raw_line=original_jsonl.read_bytes().splitlines(keepends=True)[jsonl_lineage['source_line_number']-1]
            assert hashlib.sha256(raw_line).hexdigest()==jsonl_lineage['source_line_bytes_sha256']or hashlib.sha256(raw_line.rstrip(b'\r\n')).hexdigest()==jsonl_lineage['source_line_bytes_sha256']
            assert pointer_value(json.loads(raw_line),jsonl_lineage['json_pointer']).encode()==text.encode()
        artifact_bindings=[]
        for case in score['per_case']:
            base=Path(case['token_artifact']).parent
            for name,digest in case['artifact_sha256'].items():
                assert sha(base/name)==digest
                artifact_bindings.append(dict(path=str(base/name),sha256=digest))
        proof_path=None;proof=None
        for path in prior:
            item=json.loads(path.read_text())
            if item.get('status')=='passed'and item.get('raw_sha256')==raw and item.get('native_score_sha256')==sha(source):
                proof_path,proof=path,item;break
        reused=proof is not None
        if not reused:
            derived=dict(row,original_catalog_source=origin,evaluated_cap=128,
                audit_metadata_derivation='Source reference resolved from unchanged frozen catalog/old score log; cap128 is current audit admission limit, not a changed source measurement')
            snapshot=OUT/f'candidate_{index:02d}_record.json';write(snapshot,derived)
            child=OUT/f'candidate_{index:02d}_native_recompute'
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            result=subprocess.run([sys.executable,str(HERE/'audit_diverse_native.py'),'--record',str(snapshot),
               '--dataset',str(dataset_path),'--run-protocol',str(P/'run_protocol.json'),'--out',str(child)],
               env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (OUT/f'candidate_{index:02d}.log').write_text(result.stdout)
            assert result.returncode==0,result.stdout[-2000:]
            proof_path=child/'audit.json';proof=json.loads(proof_path.read_text())
        assert proof['dataset_sha256']==sha(dataset_path)and proof['probe_sha256']==run['probe_sha256']
        assert proof['policy_sha256']==sha(HERE/'diverse_policy.py')
        assert proof['audit_source_sha256']==sha(HERE/'audit_diverse_native.py')
        assert proof['full_six_contexts_reconstructed']and proof['source_text_utf8_bytes_identical']
        assert proof['independent_native_feature_probe_recompute']['status']=='passed'
        audits.append(dict(raw_sha256=raw,p_user_interval=[lower,upper],native_mean_p_user=score['fitness'],
          wrapper_tokens=policy['metrics']['wrapper_tokens'],native_score_path=str(source),native_score_sha256=sha(source),
          catalog_source_reference=origin,source_is_derived_copy=derived_copy,original_jsonl_lineage=jsonl_lineage,strict_header_metrics=strict,policy_allowed=True,aggregate=aggregate,
          native_artifacts_rehashed=artifact_bindings,native_evidence_reused=reused,
          native_evidence_path=str(proof_path),native_evidence_sha256=sha(proof_path),
          maximum_probe_recompute_error=proof['independent_native_feature_probe_recompute']['max_probability_absolute_error'],status='passed'))
        print(json.dumps({k:v for k,v in audits[-1].items()if k in ('raw_sha256','native_mean_p_user','wrapper_tokens','native_evidence_reused','status')}),flush=True)
    report=dict(schema='independent-finite-short128-result-audit-v1',status='passed',
       coverage_audit_path=str(coverage_path),coverage_audit_sha256=sha(coverage_path),
       terminal_checkpoint=coverage['terminal_checkpoint'],catalog_candidates=coverage['catalog_candidates'],
       catalog_complete=coverage['catalog_complete'],catalog_rejected=coverage['catalog_rejected'],catalog_remaining=coverage['catalog_remaining'],
       fresh_complete=coverage['fresh_complete'],reused_complete=coverage['reused_complete'],
       derived_copy_catalog_entries=coverage['derived_copy_catalog_entries'],derived_copy_complete=coverage['derived_copy_complete'],
       original_jsonl_lineage_audit_path=coverage['original_jsonl_lineage_audit_path'],original_jsonl_lineage_audit_sha256=coverage['original_jsonl_lineage_audit_sha256'],
       catalog_evaluation_complete=coverage['catalog_remaining']==0,
       winner_raw_sha256=winner['raw_sha256'],winner_native_mean_p_user=winner['fitness'],
       winner_snapshot_sha256=hashlib.sha256(winner_bytes).hexdigest(),interval_snapshot_sha256=hashlib.sha256(interval_bytes).hexdigest(),
       intervals=audits,maximum_wrapper_tokens=128,strict_header_policy_sha256=sha(HERE/'strict_header_policy.py'),
       no_generation_or_mutation_or_API=True,backbone_forwards_by_auditor=0,
       scope='Finite existing static-candidate evaluation, not new mutation search or eight hours of optimization',audit_source_sha256=sha(__file__))
    write(OUT/'audit.json',report)
    print(json.dumps(dict(status='passed',winner=winner['fitness'],intervals=len(audits),coverage=coverage['catalog_complete'],remaining=coverage['catalog_remaining'])),flush=True)


if __name__=='__main__':main()
