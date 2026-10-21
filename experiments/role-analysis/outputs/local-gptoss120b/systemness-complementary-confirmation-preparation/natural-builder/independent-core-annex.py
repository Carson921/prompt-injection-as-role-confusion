"""Independent AST-only synthetic annex; no producer import or actual artifacts."""
import ast,copy,hashlib,json,re,tempfile,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis');SOURCE=HERE/'systemness_complementary_natural_data.py';EXPECTED='4cf654c5b7ad8f227369c2269f5eacb103da1fb164b16d0e6a3ff8e6fe4ea4bc'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def canonical(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')))
def require(v,s):
    if not bool(v):raise RuntimeError(s)
def read(p):return json.loads(Path(p).read_text())
folder=Path(tempfile.mkdtemp(prefix='confirmation-review-complementary-natural-builder-'));before=digest(SOURCE);require(before==EXPECTED,'Source changed.');text=SOURCE.read_text();nodes={n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};checks=[]
def check(name,v=True):require(v,name);checks.append(name)
def reject(name,fn):
    try:fn()
    except(RuntimeError,AssertionError):checks.append(name);return
    raise RuntimeError('Expected rejection: '+name)
def forbidden(*a,**k):raise LookupError('Real artifact/tokenizer/model access forbidden.')
SPECIAL=['<|start|>','<|message|>','<|end|>','<|return|>','<|call|>','<|channel|>','<|startoftext|>']
class Tok:
    def __init__(self):self.ids={s:i+1 for i,s in enumerate(SPECIAL)};self.inv={v:k for k,v in self.ids.items()}
    def convert_tokens_to_ids(self,s):return self.ids[s]
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        ids=[];offset=[];pos=0
        while pos<len(text):
            m=next((s for s in SPECIAL if text.startswith(s,pos)),None);end=pos+len(m) if m else pos+1;ids.append(self.ids[m] if m else ord(text[pos])+1000);offset.append((pos,end));pos=end
        return dict(input_ids=ids,offset_mapping=offset)
    def encode(self,t,add_special_tokens=False):return self(t)['input_ids']
    def decode(self,ids,skip_special_tokens=False):return ''.join(self.inv[i] if i in self.inv else chr(i-1000) for i in map(int,ids))
def parse(ids,tok):
    result={};head=[];active=None;header=False
    for i,n in enumerate(ids):
        if n==tok.ids['<|start|>']:require(active is None,'Nested body.');header=True;head=[]
        elif n==tok.ids['<|message|>']:
            require(header,'No header.');h=tok.decode(head);active='tool' if h.startswith('functions.') else 'cot' if h.startswith('assistant') and '<|channel|>analysis' in h else 'assistant' if h.startswith('assistant') else h;header=False
        elif n in [tok.ids[s] for s in ['<|end|>','<|return|>','<|call|>']]:require(active is not None,'No active role.');active=None
        elif header:head.append(n)
        elif active is not None:result[i]=active
    require(active is None and not header,'Unclosed native frame.');return result
class DSU:
    def __init__(self,nodes):self.parent={x:x for x in nodes}
    def find(self,x):
        if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def join(self,nodes):
        nodes=sorted(set(nodes))
        if nodes:
            for x in nodes[1:]:self.parent[self.find(x)]=self.find(nodes[0])
    def groups(self):
        d=defaultdict(list)
        for x in self.parent:d[self.find(x)].append(x)
        return sorted(sorted(v) for v in d.values())
def asskeys(raw,clean,body,tok):
    values=[raw,clean,body];norm=[' '.join(unicodedata.normalize('NFKC',v).split()).casefold() for v in values]
    return {('full',sha(v)) for v in values}|{('normalized',sha(v)) for v in norm}|{('prefix192',sha(np.asarray(tok.encode(v)[:192],dtype='<i8').tobytes())) for v in values+norm}
ns=dict(Path=Path,np=np,pd=pd,re=re,Counter=Counter,defaultdict=defaultdict,require=require,read=read,sha=sha,canonical_sha=canonical,ROLES=['system','user','assistant'],FILES=['inputs.parquet','rows.parquet','token-ids.json'],SAMPLE_START=10_000_000_000_000,SOURCE_SET='complementary-natural',
    data=SimpleNamespace(ROLES=['system','user','cot','assistant','tool'],digest=digest),native=SimpleNamespace(parse_native_labels=parse),closed=SimpleNamespace(pin_gate=lambda:None),OUT=folder/'fake-out',pure=SimpleNamespace(Components=DSU,assistant_keys=asskeys))
previous_source=(HERE/'systemness_natural_test.py').read_text();prevnode=next(n for n in ast.parse(previous_source).body if isinstance(n,ast.FunctionDef) and n.name=='duplicate_audit');prevns=dict(defaultdict=defaultdict,require=require,canonical_sha=canonical,dev=SimpleNamespace(Components=DSU));exec(compile(ast.Module(body=[prevnode],type_ignores=[]),'frozen-previous-duplicate','exec'),prevns)
ns['previous']=SimpleNamespace(duplicate_audit=prevns['duplicate_audit'])
for name in ['prepared_role_metadata','native_role_bodies','prepared_role_references','duplicate_audit','build_dataset','verify_cached_content','verify_summary']:
    exec(compile(ast.fix_missing_locations(ast.Module(body=[nodes[name]],type_ignores=[])),str(SOURCE),'exec'),ns)
# Metadata path/schema derived from actual producer sources, not fake manifests.
for filename in ['systemness_data.py','systemness_challenges.py','systemness_long_data.py','systemness_long_augmentation_data.py']:
    source=(HERE/filename).read_text();require('cache_sha256' in source and all(f in source for f in ns['FILES']),'Prepared producer no longer writes exact3cache schema: '+filename)
check('four_real_producer_source_schemas_have_inputs_rows_tokenids')
check('explicit_role_cache_paths_match_producers',all(s in text for s in ["data.OUT/'data-complete.json',data.CACHE","data.OUT/'supplement-data-complete.json',data.CACHE/'supplement'","native.OUT/'data-complete.json',native.CACHE","data.runtime.OUT/'systemness-long-augmentation/data-complete.json',data.CACHE/'long-augmentation'"]))
# Stub gate must reject before any metadata/digest/cache call.
def gate():raise RuntimeError('Unknown final pins.')
with patch.dict(ns,dict(closed=SimpleNamespace(pin_gate=gate),read=forbidden,data=SimpleNamespace(digest=forbidden))):
    reject('prepared_metadata_unknown_pin_beforeIO',ns['prepared_role_metadata']);reject('prepared_bodies_unknown_pin_beforeIO',lambda:ns['prepared_role_references']({},Tok()))
tok=Tok();middle='Distinct middle body, never the document prefix.';messages=[('system','fixed synthetic instruction.'),('user','Before the middle excerpt.'),('cot','Synthetic internal role body.'),('assistant',middle),('tool','External tool response.'),('tool_call','{"query":"synthetic"}')]
def render(messages):
    value='<|startoftext|>'
    for role,body in messages:
        head={'cot':'assistant<|channel|>analysis','assistant':'assistant<|channel|>final','tool':'functions.lookup to=assistant<|channel|>commentary','tool_call':'assistant to=functions.lookup<|channel|>commentary'}.get(role,role)
        value+='<|start|>'+head+'<|message|>'+body+('<|call|>' if role=='tool_call' else '<|return|>' if role=='assistant' else '<|end|>')
    return value
prompt=render(messages);frames=ns['native_role_bodies'](prompt,len(tok.encode(prompt)),tok)
check('complete_middle_and_all_native_roles_not_sparse_rows',[f['body'] for f in frames]==[b for r,b in messages] and [f['role'] for f in frames]==['system','user','cot','assistant','tool','assistant'] and frames[-1]['structural_tool_call'])
reject('length_corruption_rejected',lambda:ns['native_role_bodies'](prompt,len(tok.encode(prompt))+1,tok));reject('malformed_header_frame_rejected',lambda:ns['native_role_bodies']('<|start|>userunfinished',len(tok.encode('<|start|>userunfinished')),tok))
prepared={}
for i,name in enumerate(['main','supplement','long-full','long-augmentation']):
    cache=folder/name;cache.mkdir();pd.DataFrame([dict(prompt_ix=17+i,prompt=prompt,length=len(tok.encode(prompt)),wrapper='native',split='fit'),dict(prompt_ix=100+i,prompt='malformed nonnative frame ignored',length=0,wrapper='no-tags',split='test')]).to_parquet(cache/'inputs.parquet',index=False)
    pd.DataFrame([dict(token_ix=-99,token_id=-99)]).to_parquet(cache/'rows.parquet',index=False);(cache/'token-ids.json').write_text('[]');meta=cache/'metadata.json';meta.write_text(json.dumps(dict(cache_sha256={f:digest(cache/f) for f in ns['FILES']})));prepared[name]=(meta,cache)
ns['PREPARED_ROLE_INPUTS']=prepared;ns['OUT'].mkdir();(ns['OUT']/'source-freeze.json').write_text('synthetic');pin=ns['prepared_role_metadata']();plan=dict(source_signature=dict(prepared_role_inputs=pin))
ns['source_selection']=lambda p:require(p is plan,'Unfrozen source plan.')
refs,proof=ns['prepared_role_references'](plan,tok)
check('Arrow_native_filter_and_four_source_full_body_reference',len(refs)==24 and set(proof['counts'])==set(prepared) and all(v['native_prompts']==1 and v['body_segments']==6 for v in proof['counts'].values()) and any(asskeys(middle,middle,middle,tok)<=r['keys'] for r in refs))
with patch.dict(ns,dict(source_selection=lambda p:(_ for _ in ()).throw(RuntimeError('Not frozen.')))):
    reject('prepared_role_body_IO_requires128freeze',lambda:ns['prepared_role_references'](plan,tok))
fp=prepared['main'][1]/'token-ids.json';b=fp.read_bytes();fp.write_text('[1]');reject('prepared_tokenJSON_SHA_change_rejected',ns['prepared_role_metadata']);fp.write_bytes(b)
# External transitive graph: historical role -> public448 -> old generated.
KA=('full','role-text');B1=('normalized','public-bridge');B2=('full','generated-bridge')
role_refs=[dict(message_uid='prepared-role:main:0/message1',source_uid='prepared-role:main:0',split='historical-role-input',message_ix=1,role='assistant',keys={KA,B1})]
public=[dict(message_uid='public448/m2',source_uid='public448',split='fit',message_ix=2,keys={B1,B2})]
ledger=[dict(source_uid=s,family_id=f'family-{i}',status='prepared') for i,s in enumerate(['u','a','s','clean'])]
fresh=[dict(message_uid='u/m1',source_uid='u',message_ix=1,role='user',keys={KA}),dict(message_uid='a/m2',source_uid='a',message_ix=2,role='assistant',keys={B2}),dict(message_uid='s/m0',source_uid='s',message_ix=0,role='system',keys={KA}),dict(message_uid='clean/m2',source_uid='clean',message_ix=2,role='assistant',keys={('full','unique')})]
for r in fresh:r['split']='test'
result=ns['duplicate_audit'](ledger,[r for r in fresh if r['role']=='assistant'],fresh,public,{B2:{'legacy-generated'}},{'fixture_only':True},role_refs,{'fixture_only':True})
by={r['source_uid']:r for r in ledger}
check('allSUA_role_collisions_change_clean_only',not by['u']['eligible_clean_test'] and not by['s']['eligible_clean_test'] and by['clean']['eligible_clean_test'] and len(ledger)==4)
check('no_false_oldgenerated_or_public_flag_for_role_only_S_U',not by['u']['old_generated_collision'] and not by['s']['old_generated_collision'] and not by['u']['public448_assistant_collision'] and by['a']['old_generated_collision'] and by['a']['public448_assistant_collision'])
check('transitive_external_role_public_generated_bridge_one_component',by['u']['dependency_component_id']==by['a']['dependency_component_id']==by['s']['dependency_component_id'] and by['clean']['dependency_component_id']!=by['u']['dependency_component_id'])
check('flags_and_clean_component_counts_agree',result['eligibility_counts']['eligible_clean_test']==1 and result['dependency_components']==2 and result['clean_dependency_components']==1)
# All failed128: no tokenizer call or model, empty observation cache stays valid.
ns['previous'].processed_messages=lambda r,t:([],[],r['failure'])
records=[dict(source_uid='fake:'+str(i),split='test',family_id='family-'+str(i%32),failure='quality-failed-fixture',messages=[]) for i in range(128)]
empty=ns['build_dataset'](records,forbidden,[],{},dict(fixture_only=True));ns['verify_cached_content'](*empty[:3],empty[3],forbidden);ns['verify_summary'](empty[0],empty[1],empty[3],empty[4],empty[5])
check('all128failed_empty_cache_ledger32families_zero_support',len(empty[0])==len(empty[1])==0 and len(empty[3])==128 and empty[4]['dependency_components']==0 and empty[5]['no_observations'] and all(v['recall'] is None for v in empty[5]['roles_with_no_observations'].values()))
check('source_unchanged',digest(SOURCE)==before)
report=dict(stage='independent_synthetic_complementary_natural_builder_audit',audit_passed=True,source_sha256=before,checks=checks,check_count=len(checks),production_module_imported=False,actual_metadata_or_IDs_or_bodies_read=False,real_cache_probability_or_feature_read=False,real_tokenizer_checkpoint_or_model_read=False,gpu_used=False,actual_run=False)
p=folder/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(p),report_sha256=digest(p),script_sha256=digest(__file__),source_sha256=before,checks=len(checks))))
