"""Exact allowlist supplement for 60 already saved formatting-only literals.

Default execution is CPU preflight. Only --execute performs native scoring,
after the reviewed launch contract, empty stage lock and idle GPU checks.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
import evaluate_static_catalog as core
from precise_scorer import PreciseScorer
from scorer import load_module

HERE=Path(__file__).resolve().parent
B=HERE/'outputs/diverse8h-20261105'
P=B/'format60_preparation'
CONTRACT=P/'contract_v2.json'
DIAGNOSIS_SHA='cd0dc7a487fa86b3d40549e924e6e81f2f0098764a92c0e92e369e16638694e1'
COVERAGE_SHA='d2ab47fa3bf678e5795db323a5f1aa00b515bdb6973dac73f69d3929749c01c8'
CAP=128
COUNT=60


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value,message):
    if not value:
        raise ValueError(message)


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=True,allow_nan=False,indent=2)+'\n')


def preflight(contract_sha):
    from transformers import AutoTokenizer
    require(sha(CONTRACT)==contract_sha,'Frozen contract SHA mismatch')
    c=read(CONTRACT)
    require(c['schema']=='frozen-existing-format60-native-contract-v1','Unexpected contract schema')
    require(c['count']==COUNT and c['caps']==[CAP] and c['extra_catalog']==[], 'Frozen scope changed')
    require(c['output']==str(B/'probe_format60_short128'),'Unexpected independent output')
    require(c['diagnosis']['sha256']==DIAGNOSIS_SHA and c['coverage']['sha256']==COVERAGE_SHA,
            'Formatting-exclusion diagnosis/cache-coverage identity changed')
    for name in ('catalog','allowlist','cache_index','dataset','budget','probe','diagnosis','coverage','literal_coverage','diagnostic_index'):
        entry=c[name]
        require(sha(entry['path'])==entry['sha256'],'Frozen input changed: '+name)
    for path,digest in c['sources'].items():
        require(sha(path)==digest,'Frozen source changed: '+path)
    allowed=read(c['allowlist']['path'])['raw_sha256']
    require(len(allowed)==len(set(allowed))==COUNT and allowed==sorted(allowed),'Allowlist changed')
    allowed=set(allowed)
    coverage={r['raw_sha256']:r for r in read(c['literal_coverage']['path'])['literals']}
    expected={raw for raw,row in coverage.items() if not row['compatible_complete_current_user_native_cache']
              and row['existing_score_records']==0}
    require(allowed==expected and len(expected)==COUNT,'Allowlist is not exactly the unscored diagnosed 60')
    diagnostic={r['raw_sha256']:r for r in map(json.loads,Path(c['diagnostic_index']['path']).read_text().splitlines())}
    cache=read(c['cache_index']['path'])
    require(cache['entries']==[] and cache['imports_allowed'] is False,'Cache imports are not permitted')
    catalog_rows=[json.loads(line) for line in Path(c['catalog']['path']).read_text().splitlines() if line.strip()]
    require(len(catalog_rows)==COUNT and {r['raw_sha256']for r in catalog_rows}==allowed,'Reference catalog membership changed')
    require(all('template' not in r for r in catalog_rows),'Reference catalog must not contain reconstructed template strings')
    tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    library=BroadDelimiterLibrary(c['model'])
    core.CAPS=(CAP,)
    entries=core.load_catalog(Path(c['catalog']['path']),tokenizer,library)
    require(set(entries)==allowed,'Core silently dropped or added a reference-catalog literal')
    dataset=read(c['dataset']['path']);contexts=dataset['contexts']
    require(len(contexts)==6 and Counter(r['suite']for r in contexts)=={'banking':2,'slack':2,'travel':2},'Six-context dataset changed')
    ref_path=Path(c['reference_score_protocol']['path'])
    require(sha(ref_path)==c['reference_score_protocol']['sha256'],'Native numerical protocol changed')
    ref=read(ref_path)
    require(ref['dataset_sha256']==c['dataset']['sha256'] and ref['probe_sha256']==c['probe']['sha256'],
            'Dataset/checkpoint no longer match current native protocol')
    require(ref['target_role']=='user' and ref['target_role_index']==2,'Only current user/index2 is permitted')
    for relative,digest in ref['code_sha256'].items():
        require(sha(HERE.parents[1]/relative)==digest,'Native numerical/render source changed')
    for name,digest in ref['model_binding']['assets_sha256'].items():
        require(sha(Path(c['model'])/name)==digest,'Native tokenizer/model asset changed')
    scorer=object.__new__(PreciseScorer)
    scorer.protocol=ref;scorer.tokenizer=tokenizer
    scorer.spans=load_module('_format60_cpu_preflight_spans',HERE.parents[1]/'experiments/agentdojo-first-tool/spans.py')
    scorer.native_special_id_floor=199998
    scorer.native_special_spellings=ref['payload_delimiter_policy']['reject_real_special_spellings']
    config=read(Path(c['model'])/'config.json')
    scorer.model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=config['max_position_embeddings']))
    prepared=[];source_docs={}
    for raw,row in sorted(entries.items()):
        d=diagnostic[raw]
        require(d['diagnostic']['category']=='format_punctuation_only_repetition'
                and d['original_diverse_policy']['allowed'],'Literal was not independently confirmed format-only')
        source=Path(row['source_path'])
        require(source.suffix=='.json' and source.stat().st_mtime<=c['pre_round_cutoff_unix'],'Source is not pre-round ordinary JSON')
        if str(source) not in source_docs:
            require(sha(source)==row['source_sha256'],'Original whole-file source changed')
            source_docs[str(source)]=read(source)
        original=core.pointer_value(source_docs[str(source)],row['json_pointer'])
        require(original.encode()==row['template'].encode()==d['template'].encode()
                and raw_sha(original)==raw,'Exact original literal bytes differ')
        policy=validate_diverse_template(original,tokenizer,library,maximum=CAP)
        require(policy['allowed'] and policy['metrics']['wrapper_tokens']<=CAP,'Original diverse policy failed')
        cases=[]
        for context in contexts:
            request,prompt,ids,offsets,positions,goal,location,input_sha=scorer.prepare(context,original)
            require(request['actual_goal']==context['actual_goal'],'Immutable goal changed during CPU preparation')
            cases.append(dict(case_id=context['case_id'],input_sha256=input_sha,input_tokens=len(ids),
                complete_goal_tokens=int(goal.sum()),payload_occurrences=len(location['occurrences']),
                actual_goal_sha256=raw_sha(context['actual_goal']),all_payload_special_checks=location['payload_special_token_check']))
        require(len(cases)==6 and sum(r['payload_occurrences']for r in cases)==9,'Full six/nine CPU preparation incomplete')
        prepared.append(dict(raw_sha256=raw,wrapper_tokens=policy['metrics']['wrapper_tokens'],
            source_path=str(source),source_sha256=row['source_sha256'],json_pointer=row['json_pointer'],
            source_literal_utf8_unchanged=True,original_diverse_policy_allowed=True,per_case=cases))
    report=dict(schema='frozen-format60-cpu-input-preflight-v1',status='passed',observed_unix=time.time(),
       contract_sha256=contract_sha,catalog_sha256=c['catalog']['sha256'],exact_allowlist_count=COUNT,
       wrapper_cap=CAP,full_contexts_prepared=COUNT*6,payload_occurrences_prepared=COUNT*9,
       source_files=len(source_docs),maximum_observed_wrapper_tokens=max(r['wrapper_tokens']for r in prepared),
       raw_sha256=sorted(allowed),records=prepared,model_forward_calls=0,new_scores=0,API_calls=0,GPU_used=False,
       strict_exception_only_for_independently_confirmed_allowlist=True,run_output_created=Path(c['output']).exists())
    write(P/'cpu_preflight_v2.json',report)
    return c,entries,report


def run_native(c,entries,contract_sha):
    require(not Path(c['output']).exists() and not Path(c['output']).is_symlink(),'Independent output must be entirely new; no resume or cache')
    require(time.time()<read(c['budget']['path'])['probe_deadline_unix'],'Frozen deadline has elapsed')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    require(not apps and len(memory)==2 and all(int(v.strip())==0 for v in memory),'Both GPUs must be idle and free; no processes are stopped')
    allowed=set(entries);attempted=set();native_class=core.PreciseScorer;original_load=core.load_catalog
    def constrained_load(path,tokenizer,library):
        require(Path(path).resolve()==Path(c['catalog']['path']).resolve(),'Additional input catalog is forbidden')
        result=original_load(path,tokenizer,library)
        require(set(result)==allowed,'Runtime catalog membership changed')
        return result
    class ExactAllowlistScorer(native_class):
        def score(self,text):
            require(self.protocol==read(c['reference_score_protocol']['path']),
                    'Actual native numerical protocol differs from the frozen reference')
            raw=raw_sha(text)
            require(raw in allowed and text.encode()==entries[raw]['template'].encode(),'Non-allowlisted literal reached native score')
            require(raw not in attempted,'Repeated native score attempt forbidden')
            require(core.CAPS==(CAP,),'Length expansion forbidden')
            policy=validate_diverse_template(text,self.tokenizer,BroadDelimiterLibrary(c['model']),maximum=CAP)
            require(policy['allowed'],'Runtime original diverse policy failed')
            attempted.add(raw)
            result=super().score(text)
            require(result.get('cache_hit') is False,'Fresh isolated scoring must not import a native cache')
            return result
    core.CAPS=(CAP,);core.load_catalog=constrained_load;core.PreciseScorer=ExactAllowlistScorer
    output=Path(c['output']);output.mkdir()
    # The inherited constructor compares this complete reference before calling
    # collect.load_model; it is a protocol guard, with no token/features cache.
    (output/'native').mkdir()
    shutil.copyfile(c['reference_score_protocol']['path'],output/'native/score_protocol.json')
    write(output/'format60_constraint_protocol.json',dict(contract= str(CONTRACT),contract_sha256=contract_sha,
       sources=c['sources'],exact_allowlist=sorted(allowed),maximum_wrapper_tokens=CAP,cache_imports=0,
       extra_catalogs=[],generation=False,mutation=False,API=False,ASR=False,
       diagnosis_sha256=DIAGNOSIS_SHA,coverage_sha256=COVERAGE_SHA))
    args=SimpleNamespace(catalog=Path(c['catalog']['path']),cache_index=Path(c['cache_index']['path']),
       dataset=Path(c['dataset']['path']),budget=Path(c['budget']['path']),output=output,
       model=Path(c['model']),probe=Path(c['probe']['path']),seed=c['seed'],extra_catalog=[])
    status=core.run(args)
    records=core.read_lines(output/'score_records.jsonl');rejects=core.read_lines(output/'rejections.jsonl')
    contexts=read(c['dataset']['path'])['contexts'];checkpoint=read(output/'checkpoint.json')
    complete=(status=='audited_finite_catalog_exhausted' and len(records)==COUNT and not rejects
       and {r['raw_sha256']for r in records}==allowed and len(attempted)==COUNT
       and checkpoint['cap']==CAP and checkpoint['phase_index']==0
       and checkpoint['complete_scores']==checkpoint['fresh_scores']==COUNT)
    for row in records:
        core.verify_complete(row,contexts)
        require(row['origin']=='fresh_fixed_catalog' and row['evaluated_cap']==CAP,'Foreign or stale native score found')
    write(output/'format60_terminal_integrity.json',dict(status='passed' if complete else 'incomplete',
       native_status=status,exact60_complete=complete,complete_records=len(records),rejected_records=len(rejects),
       attempted=len(attempted),contract_sha256=contract_sha,new_text_generated=False,API_calls=0,ASR_runs=0))
    return 0 if complete else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--execute',action='store_true',help='Explicit reviewed native execution; omitted means CPU preflight only')
    args=parser.parse_args()
    c,entries,report=preflight(args.contract_sha256)
    if not args.execute:
        print(json.dumps({k:report[k]for k in ('status','exact_allowlist_count','full_contexts_prepared','payload_occurrences_prepared','maximum_observed_wrapper_tokens','model_forward_calls','GPU_used')}))
        return 0
    with (B/'asr_stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run_native(c,entries,args.contract_sha256)


if __name__=='__main__':
    raise SystemExit(main())
