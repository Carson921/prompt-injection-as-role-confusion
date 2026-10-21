"""Independent C4 builder source/tiny CPU audit; no actual modes or data."""
import ast,copy,hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
SOURCE=HERE/'systemness_complementary_confirmation_data.py'
PIN='c447596fa365f23e377e9068edf5ea8881ee541991777622cbf56b3abfcc35e3'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,s):
    if not bool(v):raise RuntimeError(s)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    require(digest(SOURCE)==PIN,'Source pin changed.');start=digest(SOURCE);sys.path.insert(0,str(HERE));m=load('independent_c4_builder_subject',SOURCE);old=load('independent_c4_prior_sequence',HERE/'systemness_aug_confirmation_data.py');folder=Path(tempfile.mkdtemp(prefix='long-data-complementary-c4-audit-'));checks=[]
    def check(name,v=True):require(v,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,ValueError,KeyError):checks.append(name);return
        raise RuntimeError('Fault accepted: '+name)
    class UnexpectedIO(Exception):pass
    class PathSpy:
        def __truediv__(self,k):raise UnexpectedIO('Formal path touched before unknown-pin gate.')
        def is_file(self):raise UnexpectedIO('Formal path stat before gate.')
    def blocked():raise RuntimeError('Synthetic unknown640 pin')
    def unexpected(*a,**k):raise UnexpectedIO('Formal read before gate')
    original=(m.closed.pin_gate,m.native.load_tokenizer,m.data.OUT,m.data.CACHE,old.OUT,old.CACHE)
    with patch.multiple(m,actual_gate=blocked,OUT=PathSpy(),SNAPSHOT=PathSpy(),read=unexpected):
        for name,fn in [('signature',m.source_signature),('declare',m.declare),('prepare',m.prepare),('load',m.load_verified_data),('verify',m.verify),('history-signature',m.natural_history_signature),('history-body',lambda:m.historical_natural_keys({},None))]:reject('unknownpins-before-IO-'+name,fn)
    check('exact_reserved_window_counts_seed_namespaces_and_snapshot_path',m.FIRST==12303 and m.LAST==20494 and m.GENERAL==400 and m.FULL==128 and m.PAIRED==64 and m.SELECT_SEED==20296231 and m.DOCUMENT_START==9000000 and m.SAMPLE_START==9000000000000 and m.GENERAL*6+m.PAIRED*5+m.FULL*5==3360 and m.SNAPSHOT==m.closed.OUT/'readout-snapshot.json')
    # Verify the real producer schema from its source, never its artifacts.
    def complete_keys(path):
        keys=set()
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='complete' for t in n.targets) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='dict':keys.update(k.arg for k in n.value.keywords if k.arg)
        return keys
    required={'cache_sha256','cache_path','prepared_prompts','content_rows','model_generation_run','gpu_work_run'}
    check('frozen_dev_and_test_producer_schema_supports_reader_fields',required<=complete_keys(HERE/'systemness_natural_transcripts.py') and required<=complete_keys(HERE/'systemness_natural_test.py'))
    check('dev320_quarantine_and_old128_pins_paths_and_scope',m.NATURAL_HISTORY['dev320']['cache']==m.data.CACHE/'natural-transcripts-dev' and m.NATURAL_HISTORY['test128']['cache']==m.data.CACHE/'natural-test' and m.NATURAL_HISTORY['dev320']['prompts']==320 and m.NATURAL_HISTORY['dev320']['rows']==80436 and m.NATURAL_HISTORY['test128']['prompts']==128 and m.NATURAL_HISTORY['test128']['rows']==33476)
    tree=ast.parse(SOURCE.read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    check('no_global_inventory_or_frozen_completed_verifier_calls',not any(n.func.attr in {'glob','rglob','verified_completion','frozen_snapshot','exclusion_inputs'} for n in calls))
    tok=m.SyntheticTokenizer();descriptors={};pins={};expected=set();expected_bodies=[]
    # Independent message positions come from serialization spans. Quarantine
    # False remains present; a one-character body proves no new short filter.
    specs={'dev320':[('fit',True,[('system','S'),('user','ＦＵＬＬ normalized historical body '+('blue notes '*20)),('assistant','Prepared clean reply '+('cedar diary '*20))]),('fit',False,[('system','Shared historical instruction.'),('user','The quarantined full historical body '+('tiny example '*20)),('assistant','q')]),('validation',False,[('system','Shared historical instruction.'),('user','Old validation query https://independent.invalid/shared'),('assistant','Validation reply '+('yellow wood '*20))])],
        'test128':[('test',False,[('system','Shared historical instruction.'),('user','Old test query '+('matrix column '*20)),('assistant','Old test reply '+('violet meadow '*20))])]}
    for name,entries in specs.items():
        cache=folder/name;cache.mkdir();inputs=[];rows=[];tokens=[]
        for prompt_ix,(split,eligible,messages) in enumerate(entries):
            text,spans=m.data.serialize(messages,bos=True,wrapper='native');enc=tok(text,return_offsets_mapping=True);tokens.append(enc['input_ids']);inputs.append(dict(prompt_ix=prompt_ix,length=len(enc['input_ids']),wrapper='native',split=split,eligible_fit=eligible))
            for mid,(role,body) in enumerate(messages):
                expected|=m.natural_sources.ContentKeys(tok)(body);expected_bodies.append(body)
                span=spans[mid]
                for token_ix,(a,b) in enumerate(enc['offset_mapping']):
                    if span['start']<=a and b<=span['end']:rows.append(dict(prompt_ix=prompt_ix,message_ix=mid,token_ix=token_ix,token_id=enc['input_ids'][token_ix],role=role))
        pd.DataFrame(inputs).to_parquet(cache/'inputs.parquet',index=False);pd.DataFrame(rows).to_parquet(cache/'rows.parquet',index=False);(cache/'token-ids.json').write_text(json.dumps(tokens));done=cache/'data-complete.json'
        value=dict(cache_sha256={f:digest(cache/f) for f in m.NATURAL_FILES},cache_path=str(cache),prepared_prompts=len(inputs),content_rows=len(rows),model_generation_run=False,gpu_work_run=False,classifier_probabilities_read=(name=='test128'))
        done.write_text(json.dumps(value));pins[name]=dict(data_complete=done,cache=cache,data_complete_sha256=digest(done),prompts=len(inputs),rows=len(rows));descriptors[name]=dict(data_complete_path=str(done),data_complete_sha256=digest(done),cache_path=str(cache),cache_sha256=value['cache_sha256'],prompts=len(inputs),rows=len(rows))
    with patch.multiple(m,NATURAL_HISTORY=pins,actual_gate=lambda:None):
        check('metadata_exact_threecache_hashes_accept_actual_test_verification_flag',m.natural_history_signature()==descriptors)
        file=folder/'dev320/token-ids.json';saved=file.read_bytes();file.write_text('[]');reject('history_cache_SHA_tamper_rejected_before_decode',m.natural_history_signature);file.write_bytes(saved)
    plan=dict(source_signature=dict(natural_history=descriptors));guard_calls=[]
    def guard(value):require(value is plan,'Wrong plan');guard_calls.append('declared')
    with patch.object(m,'guarded_source_access',guard):keys,audit=m.historical_natural_keys(plan,tok)
    check('independent_original_message_key_union_equals_rows_token_restore',keys==expected and audit['messages']==12 and guard_calls==['declared'])
    check('all_fit_quarantine_validation_test_and_one_character_body_included',('full',m.sha('q')) in keys and audit['sets']['dev320']['split_prompts']=={'fit':2,'validation':1} and audit['sets']['test128']['split_prompts']=={'test':1} and audit['all_prepared_splits_and_quarantine_included'])
    # If cache ownership hashes were consistently repinned, semantic token-ID
    # and frame-position guards still reject inconsistent prepared data.
    corrupt=copy.deepcopy(descriptors);file=folder/'dev320/rows.parquet';frame=pd.read_parquet(file);saved=file.read_bytes();frame.loc[0,'token_id']+=1;frame.to_parquet(file,index=False);corrupt['dev320']['cache_sha256']['rows.parquet']=digest(file)
    with patch.object(m,'guarded_source_access',lambda p:None):reject('rehashed_wrong_body_token_ID_rejected',lambda:m.historical_natural_keys(dict(source_signature=dict(natural_history=corrupt)),tok))
    file.write_bytes(saved);frame=pd.read_parquet(file);pd.concat([frame,frame.iloc[:1]],ignore_index=True).to_parquet(file,index=False);corrupt=copy.deepcopy(descriptors);corrupt['dev320']['cache_sha256']['rows.parquet']=digest(file);corrupt['dev320']['rows']+=1
    with patch.object(m,'guarded_source_access',lambda p:None):reject('rehashed_duplicate_content_position_rejected',lambda:m.historical_natural_keys(dict(source_signature=dict(natural_history=corrupt)),tok))
    file.write_bytes(saved)
    candidates=[]
    for i in range(16):
        item,reason=m.make_candidate(dict(text='Independent fabricated candidate '+str(i)+'. '+('lake silver conifer '*100),url='https://new.invalid/'+str(i)),m.FIRST+i,tok);require(reason is None,'Fixture eligible');candidates.append(item)
    # Source-level collision evidence covers normalization, URL, and the
    # original article beyond its retained1024 body, not only _body.
    article=dict(text='full NORMALIZED historical body '+('BLUE NOTES '*20),url='https://new.invalid/norm');item,reason=m.make_candidate(article,m.FIRST+100,tok)
    check('candidate_complete_original_normalization_keys_match_historical_message',reason is None and set(map(tuple,item['natural_content_keys']))&keys)
    urlitem,reason=m.make_candidate(dict(text='Unrelated complete original text '+('lake silver '*100),url='https://independent.invalid/shared'),m.FIRST+101,tok)
    check('candidate_sourceURL_matches_historical_message_URL',('url','https://independent.invalid/shared') in set(map(tuple,urlitem['natural_content_keys']))&keys)
    full=copy.deepcopy(min(candidates,key=lambda r:r['rank_sha256']));full['natural_content_keys']=[list(k) for k in keys];pool=[full]+[c for c in candidates if c['shard_line']!=full['shard_line']]
    selected,selection=m.select_candidates(pool,set(),set(),(3,2,1),natural_keys=keys);again,_=m.select_candidates(pool[::-1],set(),set(),(3,2,1),natural_keys=keys)
    check('crosscorpus_reject_then_fixed_greedy_permutation_independent',selected==again and full['shard_line'] not in {r['shard_line'] for r in selected} and selection['duplicate_rejections']['historical-natural-content-or-URL-duplicate']>=1)
    reject('insufficient_fixed_pool_fails_without_extension',lambda:m.select_candidates(pool,set(),set(),(99,2,1),natural_keys=keys))
    inputs,rows,tokens=m.render(selected,tok);oldinputs,oldrows,oldtokens=old.render(selected,tok)
    check('old_frozen_single_multi_long_pure_sequence_prompt_and_tokens_identical',inputs.prompt.tolist()==oldinputs.prompt.tolist() and tokens==oldtokens and rows[['token_ix','token_id','role','message_ix','body_offset']].equals(oldrows[['token_ix','token_id','role','message_ix','body_offset']]))
    # Own character-span frame parser is independent of the producer parser.
    import re
    frame_pattern=re.compile(r'<\|start\|>(.*?)<\|message\|>(.*?)(<\|end\|>|<\|return\|>|<\|call\|>)',re.S)
    for prompt_ix,inputrow in inputs.iterrows():
        enc=tok(inputrow.prompt,return_offsets_mapping=True);expected_roles={}
        for match in frame_pattern.finditer(inputrow.prompt):
            head=match.group(1)
            if 'to=functions.' in head:continue
            role='tool' if head.startswith('functions.') else 'cot' if head.startswith('assistant') and 'analysis' in head else 'assistant' if head.startswith('assistant') else head
            a,b=match.span(2)
            for ix,(l,r) in enumerate(enc['offset_mapping']):
                if a<=l and r<=b:expected_roles[ix]=role
        sub=rows[rows.prompt_ix==prompt_ix];require(dict(zip(sub.token_ix,sub.role))==expected_roles,'Independent normal-role/header exclusion differs.')
    check('independent_native_frame_parser_all_content_system_and_toolcall_exclusion',set(rows.role)==set(m.data.ROLES) and rows[rows.role=='system'].shape[0]>0)
    gallery=m.gallery_index([dict(cohort='general',cohort_rank=i) for i in range(400)]+[dict(cohort='full-length',cohort_rank=i) for i in range(128)])
    check('fixed_gallery_twelve_multi_four_full_and_summary_seventeen_pairs',len(gallery['general_multi'])==12 and [r['cohort_rank'] for r in gallery['full_system']]==[0,42,84,127])
    check('all_frozen_module_globals_unchanged',original==(m.closed.pin_gate,m.native.load_tokenizer,m.data.OUT,m.data.CACHE,old.OUT,old.CACHE));check('source_unchanged_start_end',digest(SOURCE)==start)
    report=dict(stage='independent_source_tiny_CPU_complementary_c4_builder_audit',audit_passed=True,source_sha256=start,checks=checks,checks_passed=len(checks),actual_cache_or_history_or_IDs_or_body_read=False,real_tokenizer_or_checkpoint_or_models_read=False,probabilities_or_feature_banks_read=False,gpu_used=False,production_code_modified=False)
    out=folder/'report.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(out),report_sha256=digest(out),source_sha256=start,checks=len(checks))))
if __name__=='__main__':main()
