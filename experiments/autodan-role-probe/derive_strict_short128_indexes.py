"""Derive read-only short catalogs/caches from existing complete artifacts.

No generated/mutated text, model execution, API use or edits to source files.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from evaluate_static_catalog import pointer_value, verify_complete
from strict_header_policy import strict_header_metrics

HERE=Path(__file__).resolve().parent
B=HERE/'outputs/diverse8h-20261105'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n')


def main():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/tmp/wz55191-agentdojo-first-tool-model',local_files_only=True)
    library=BroadDelimiterLibrary('/tmp/wz55191-agentdojo-first-tool-model')
    dataset_path=B/'dataset.json';dataset=json.loads(dataset_path.read_text())
    complete_path=B/'probe/score_records.jsonl'
    complete_bytes=complete_path.read_bytes();records={}
    for line in complete_bytes.decode('utf-8').splitlines():
        if line.strip():
            row=json.loads(line);records[row['raw_sha256']]=row
    assert len(records)==1043,'Original completed snapshot changed; inspect before deriving'
    candidates={};catalog_hashes={};source_cache={};source_hashes={};counts=Counter();rejections=Counter()
    for catalog in (B/'static_catalog.jsonl',B/'static_catalog_extra.jsonl'):
        catalog_hashes[str(catalog)]=sha(catalog)
        for index,line in enumerate(catalog.read_text().splitlines()):
            if not line.strip():continue
            row=json.loads(line);text=row['template'];raw=raw_sha(text)
            assert raw==row['raw_sha256']
            if raw in candidates:
                counts['duplicate_eligible_catalog_occurrences']+=1
                continue
            counts['catalog_occurrences']+=1
            # Authoritative real tokenizer length; no truncation or rewriting.
            tokens=len(tokenizer.encode(text.replace('{goal}',''),add_special_tokens=False))
            if tokens>128:
                rejections['over128']+=1;continue
            strict=strict_header_metrics(text)
            if not strict['allowed']:
                rejections['repeated_header_line_or_paragraph']+=1;continue
            policy=validate_diverse_template(text,tokenizer,library,maximum=128)
            if not policy['allowed']:
                rejections.update(set(policy['reason_codes']));continue
            source=Path(row['source_path'])
            if source not in source_cache:
                assert sha(source)==row['source_sha256']
                source_cache[source]=json.loads(source.read_text());source_hashes[str(source)]=sha(source)
            else:assert source_hashes[str(source)]==row['source_sha256']
            original=pointer_value(source_cache[source],row['json_pointer'])
            assert original.encode('utf-8')==text.encode('utf-8')
            candidates[raw]=dict(row,wrapper_tokens=tokens,policy=policy,strict_header_policy=strict,
                readonly_derivation=dict(catalog_path=str(catalog),catalog_sha256=catalog_hashes[str(catalog)],source_row=index),
                text_modified=False,fitness=None,proposal_only=True)
    catalog_out=B/'strict_existing_short128_catalog.jsonl'
    with catalog_out.open('w')as handle:
        for raw,row in sorted(candidates.items()):handle.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+'\n')
    print(json.dumps(dict(stage='catalog_ready',path=str(catalog_out),eligible=len(candidates),
      not_in1043_complete=sum(raw not in records for raw in candidates),sha256=sha(catalog_out))),flush=True)
    caches=[];cache_rejections=Counter();complete_source_hashes={}
    for raw,row in sorted(records.items()):
        text=row['template'];assert raw_sha(text)==raw
        tokens=len(tokenizer.encode(text.replace('{goal}',''),add_special_tokens=False))
        if tokens>128:cache_rejections['over128']+=1;continue
        strict=strict_header_metrics(text)
        if not strict['allowed']:cache_rejections['repeated_header_line_or_paragraph']+=1;continue
        policy=validate_diverse_template(text,tokenizer,library,maximum=128)
        if not policy['allowed']:cache_rejections.update(set(policy['reason_codes']));continue
        source=Path(row['source_score_path']);before=sha(source)
        assert before==row['source_score_sha256']
        result=json.loads(source.read_text())
        assert result['template'].encode('utf-8')==text.encode('utf-8')
        fitness=verify_complete(result,dataset['contexts'])
        assert result.get('expected_cases')==result.get('completed_cases')==6
        assert all(case['status']=='complete'for case in result['per_case'])
        assert before==sha(source),'Source changed while auditing'
        complete_source_hashes[str(source)]=before
        caches.append(dict(source_path=str(source),source_score_path=str(source),source_sha256=before,
            source_score_sha256=before,raw_sha256=raw,fitness=fitness,wrapper_tokens=tokens,
            family=row.get('family','existing_complete'),allowed=True,policy=policy,strict_header_policy=strict,
            verified_complete_fixed_six=True,source_file_unchanged=True,
            native_artifact_hash_check='Unmodified runner verifies every NPZ/request against source score artifact_sha256 before reuse'))
    cache_out=B/'strict_complete_short128_cache_index.json'
    write(cache_out,dict(schema='strict-existing-complete-short128-native-cache-index-v1',maximum_wrapper_tokens=128,
        dataset_sha256=sha(dataset_path),strict_header_policy_sha256=sha(HERE/'strict_header_policy.py'),
        source_score_records_path=str(complete_path),source_score_records_sha256=hashlib.sha256(complete_bytes).hexdigest(),
        entries=caches,new_candidate_generation=False,text_modified=False))
    unscored=[row for raw,row in candidates.items()if raw not in records]
    unscored_out=B/'strict_unscored_short128_catalog.jsonl'
    with unscored_out.open('w')as handle:
        for row in sorted(unscored,key=lambda r:r['raw_sha256']):handle.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+'\n')
    assert sha(complete_path)==hashlib.sha256(complete_bytes).hexdigest()
    assert all(sha(path)==digest for path,digest in catalog_hashes.items())
    assert all(sha(path)==digest for path,digest in source_hashes.items())
    assert all(sha(path)==digest for path,digest in complete_source_hashes.items())
    sidecar=dict(schema='strict-short128-readonly-index-derivation-v1',status='passed',created_utc=datetime.now(timezone.utc).isoformat(),
       source_catalogs=catalog_hashes,source_score_records_sha256=hashlib.sha256(complete_bytes).hexdigest(),
       source_complete_scores=len(records),eligible_catalog_unique=len(candidates),
       eligible_complete_cache_entries=len(caches),unscored_strict_short_unique=len(unscored),
       maximum_wrapper_tokens=128,catalog_rejections=dict(rejections),cache_rejections=dict(cache_rejections),
       output_catalog_path=str(catalog_out),output_catalog_sha256=sha(catalog_out),
       output_cache_index_path=str(cache_out),output_cache_index_sha256=sha(cache_out),
       output_unscored_catalog_path=str(unscored_out),output_unscored_catalog_sha256=sha(unscored_out),
       source_template_files=source_hashes,source_complete_score_files=complete_source_hashes,
       strict_header_policy_sha256=sha(HERE/'strict_header_policy.py'),diverse_policy_sha256=sha(HERE/'diverse_policy.py'),
       verify_complete_source_sha256=sha(HERE/'evaluate_static_catalog.py'),
       readonly_source_files_unchanged=True,new_candidate_generation=False,backbone_forward_calls=0,api_calls=0)
    write(B/'strict_short128_index_derivation.json',sidecar)
    print(json.dumps({k:v for k,v in sidecar.items()if k not in ('source_template_files','source_complete_score_files')},indent=2),flush=True)


if __name__=='__main__':main()
