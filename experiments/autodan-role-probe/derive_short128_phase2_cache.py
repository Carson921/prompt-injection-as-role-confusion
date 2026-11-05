"""Derive the next stage's cache from the terminal, audited short run."""
import hashlib
import json
from pathlib import Path

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from evaluate_static_catalog import verify_complete
from strict_header_policy import strict_header_metrics

HERE=Path(__file__).resolve().parent
B=HERE/'outputs/diverse8h-20261105'
P=B/'probe_short128'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n')


def main():
    audit_path=B/'audit_short128_final/audit.json';audit=json.loads(audit_path.read_text())
    assert audit['status']=='passed'
    assert audit['catalog_evaluation_complete']
    coverage_path=B/'audit_short128_supervision/audit.json';coverage=json.loads(coverage_path.read_text())
    assert coverage['terminal_checkpoint']['status']=='audited_finite_catalog_exhausted'
    source_records=P/'score_records.jsonl';source_records_sha=sha(source_records)
    assert source_records_sha==coverage['source_score_records_sha256']
    rows={row['raw_sha256']:row for row in map(json.loads,source_records.read_text().splitlines())}
    assert len(rows)==coverage['complete_records']
    dataset_path=B/'dataset.json';dataset=json.loads(dataset_path.read_text())
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/tmp/wz55191-agentdojo-first-tool-model',local_files_only=True)
    library=BroadDelimiterLibrary('/tmp/wz55191-agentdojo-first-tool-model')
    entries=[];artifact_hashes={};score_hashes={}
    for raw,row in sorted(rows.items()):
        source=Path(row['source_score_path']);before=sha(source)
        assert before==row['source_score_sha256']
        score=json.loads(source.read_text());text=score['template']
        assert text.encode()==row['template'].encode()and hashlib.sha256(text.encode()).hexdigest()==raw
        mean=verify_complete(score,dataset['contexts'])
        assert score['status']=='complete'and score['complete']is True
        assert score['expected_cases']==score['completed_cases']==6
        assert all(case['status']=='complete'for case in score['per_case'])
        policy=validate_diverse_template(text,tokenizer,library,maximum=128)
        strict=strict_header_metrics(text)
        assert policy['allowed']and strict['allowed']and policy['metrics']['wrapper_tokens']<=128
        for case in score['per_case']:
            base=Path(case['token_artifact']).parent
            for name,digest in case['artifact_sha256'].items():
                path=base/name
                if str(path)not in artifact_hashes:
                    assert sha(path)==digest,(path,'Native artifact changed')
                    artifact_hashes[str(path)]=digest
                else:assert artifact_hashes[str(path)]==digest
        assert before==sha(source)
        score_hashes[str(source)]=before
        entries.append(dict(source_path=str(source),source_score_path=str(source),source_sha256=before,
          source_score_sha256=before,raw_sha256=raw,fitness=score['fitness'],
          verified_equal_six_mean=mean,wrapper_tokens=policy['metrics']['wrapper_tokens'],
          allowed=True,verified_complete_fixed_six=True,all_native_artifact_hashes_verified=True,
          source_text_unchanged=True,source_fitness_unchanged=True,strict_header_allowed=True,
          family=row.get('family','existing_complete')))
    assert sha(source_records)==source_records_sha
    assert all(sha(path)==digest for path,digest in score_hashes.items())
    out=B/'strict_complete_short128_phase2_cache_index.json'
    write(out,dict(schema='strict-terminal-short128-complete-native-cache-index-v1',maximum_wrapper_tokens=128,
      dataset_sha256=sha(dataset_path),entries=entries,source_score_records_path=str(source_records),
      source_score_records_sha256=source_records_sha,source_final_audit_path=str(audit_path),
      source_final_audit_sha256=sha(audit_path),strict_header_policy_sha256=sha(HERE/'strict_header_policy.py'),
      no_template_or_source_changes=True,no_candidate_generation=True))
    evidence=dict(schema='strict-short128-phase2-cache-derivation-audit-v1',status='passed',
      entries=len(entries),artifact_files_verified=len(artifact_hashes),source_score_files_verified=len(score_hashes),
      dataset_sha256=sha(dataset_path),source_score_records_sha256=source_records_sha,
      output_path=str(out),output_sha256=sha(out),native_artifact_hashes=artifact_hashes,
      source_score_hashes=score_hashes,verify_complete_source_sha256=sha(HERE/'evaluate_static_catalog.py'),
      source_final_audit_sha256=sha(audit_path),derive_source_sha256=sha(__file__),
      raw_templates_and_fitness_unchanged=True,read_only_sources=True,
      backbone_forward_calls=0,api_calls=0,new_candidate_generation=False)
    write(B/'strict_complete_short128_phase2_cache_derivation.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items()if k not in ('native_artifact_hashes','source_score_hashes')},indent=2),flush=True)


if __name__=='__main__':main()
