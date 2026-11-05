"""Reference-only addendum for already saved, frozen template seed banks."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template

HERE=Path(__file__).resolve().parent
OUTPUTS=HERE/'outputs'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(value,pointer='',family=None):
    if isinstance(value,dict):
        family=value.get('family',family)
        for key,child in value.items():
            child_pointer=pointer+'/'+key.replace('~','~0').replace('/','~1')
            if key=='template' and isinstance(child,str)and child.count('{goal}')==1:
                yield child,child_pointer,family
            elif isinstance(child,(dict,list)):
                yield from walk(child,child_pointer,family)
    elif isinstance(value,list):
        for index,child in enumerate(value):
            child_pointer=pointer+'/'+str(index)
            if isinstance(child,str)and child.count('{goal}')==1 and pointer.endswith(('/templates','/records')):
                yield child,child_pointer,family
            elif isinstance(child,(dict,list)):
                yield from walk(child,child_pointer,family)


def main():
    out=OUTPUTS/'diverse8h-20261105'
    previous=json.loads((out/'static_catalog_admission_rows.json').read_text())
    known={r['template_sha256']for r in previous['rows']}
    known.update(r['template_sha256']for r in json.loads((out/'legacy_score_admission.json').read_text())['rows'])
    paths=[]
    for name in ['user90-20261104','user90-broad-20261104','user90-precise-20261104',
                 'user90-adaptive-20261104','user90-content-20261105']:
        paths.extend(sorted((OUTPUTS/name).glob('seed_banks/*.json')))
    for name in ['near-delimiter-20261104','cot-near-delimiter-20261104','cot-near-delimiter-refined-20261104']:
        root=OUTPUTS/name
        paths.extend(sorted(root.glob('*seed*.json')))
        paths.extend(sorted(root.glob('*template*.json')))
        paths.extend(sorted(root.glob('seed_banks/*.json')))
        paths.extend(sorted(root.glob('templates/*.json')))
    paths=sorted(set(paths))
    candidates={};counts=Counter();sources=[]
    for path in paths:
        document=json.loads(path.read_text());digest=sha(path);count=0
        for text,pointer,family in walk(document):
            count+=1
            raw=hashlib.sha256(text.encode('utf-8')).hexdigest()
            if raw in known:
                counts['known_template_occurrences_skipped']+=1
                continue
            reference=dict(source_path=str(path),source_sha256=digest,json_pointer=pointer)
            if raw in candidates:
                candidates[raw]['source_aliases'].append(reference)
            else:
                candidates[raw]=dict(template=text,raw_sha256=raw,family=family or'historical_frozen_seed_bank',
                                     **reference,source_aliases=[])
        sources.append(dict(path=str(path),sha256=digest,template_occurrences=count))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/tmp/wz55191-agentdojo-first-tool-model',local_files_only=True)
    library=BroadDelimiterLibrary('/tmp/wz55191-agentdojo-first-tool-model')
    cap_counts=Counter();reasons=Counter();rows=[];audit=[]
    for index,(raw,row) in enumerate(sorted(candidates.items())):
        result=validate_diverse_template(row['template'],tokenizer,library,maximum=16384)
        audit.append(dict(raw_sha256=raw,source_path=row['source_path'],json_pointer=row['json_pointer'],
                          allowed=result['allowed'],reason_codes=result['reason_codes'],
                          wrapper_tokens=result['metrics']['wrapper_tokens']))
        if result['allowed']:
            row.update(wrapper_tokens=result['metrics']['wrapper_tokens'],policy=result,
                       policy_sha256=sha(HERE/'diverse_policy.py'),fitness=None,proposal_only=True)
            parts=row['json_pointer'].split('/')
            row['source_index']=next((int(x)for x in parts if x.isdigit()),None)
            rows.append(row)
            for cap in [128,256,512,1024,2048,4096,8192,16384]:
                if row['wrapper_tokens']<=cap:cap_counts[str(cap)]+=1
        else:reasons.update(set(result['reason_codes']))
    rows.sort(key=lambda r:(r['wrapper_tokens'],r['raw_sha256']))
    path=out/'static_catalog_extra.jsonl'
    with path.open('w')as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+'\n')
    evidence=dict(schema='existing-frozen-seed-bank-addendum-v1',created_utc=datetime.now(timezone.utc).isoformat(),
       policy_sha256=sha(HERE/'diverse_policy.py'),source_files=sources,known_raw_hashes=len(known),
       known_template_occurrences_skipped=counts['known_template_occurrences_skipped'],
       unique_unseen_templates=len(candidates),eligible_unique_templates=len(rows),cap_counts=dict(cap_counts),
       rejection_reasons=dict(reasons),rows=audit,catalog_path=str(path),catalog_sha256=sha(path),
       new_candidate_generation=False,mutation_calls=0,api_calls=0,backbone_forward_calls=0,
       candidate_text_unchanged=True,original_catalog_untouched=True)
    (out/'static_catalog_extra_evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in evidence.items()if k not in ('source_files','rows')},indent=2),flush=True)


if __name__=='__main__':main()
