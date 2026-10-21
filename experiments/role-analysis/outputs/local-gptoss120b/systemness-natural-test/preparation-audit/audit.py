"""Read-only independent audit of completed natural128 preparation.

Default help; --preflight uses toy data only. --actual requires the completed
builder cache and reads no C4/natural-test probability or metric. The required
builder verifier does inspect pre-existing development validation probabilities.
No source preparation, source reply replay, inference or GPU is performed.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT=Path('/root/prompt-injection-as-role-confusion')
MODULES=ROOT/'experiments/role-analysis'
PRIMARY_SHA='27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
PRIMARY_THRESHOLD=.620969696969697
EXPECTED_COMPLETE_SHA='a3ba614a83feda6687a49b2fb10aca2ab3e9af8d3b1b24935e98ce54cf37175a'
NAMES=['residual','base-attention','dense128','head-selected']
SUA=['system','user','assistant']
ALL_ROLES=['system','user','cot','assistant','tool']
RESERVED=['<|start|>','<|message|>','<|end|>','<|return|>','<|call|>','<|channel|>','<|startoftext|>']

def need(condition,message):
    if not condition:raise RuntimeError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def canonical(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':')))
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path).resolve();need(path.is_relative_to(Path('/tmp')),'Only /tmp audit outputs allowed.')
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

class Union:
    def __init__(self,nodes):self.p={node:node for node in nodes}
    def find(self,node):
        while self.p[node]!=node:self.p[node]=self.p[self.p[node]];node=self.p[node]
        return node
    def join(self,nodes):
        nodes=list(nodes)
        for node in nodes[1:]:self.p[self.find(node)]=self.find(nodes[0])
    def groups(self):
        groups=defaultdict(list)
        for node in self.p:groups[self.find(node)].append(node)
        return sorted(sorted(nodes) for nodes in groups.values())


def parse_native(ids,tok):
    """Own token-state parser; never calls the builder's role parser."""
    codes={text:tok.convert_tokens_to_ids(text) for text in RESERVED}
    need(len(set(codes.values()))==len(codes),'Reserved token IDs collide.')
    need(bool(ids) and ids[0]==codes['<|startoftext|>'],'Native input requires exactly one leading BOS.')
    result=[];cursor=1;message=0;roles=[]
    while cursor<len(ids):
        need(ids[cursor]==codes['<|start|>'],'Message must start at a native start token.')
        cursor+=1;header=[]
        while cursor<len(ids) and ids[cursor]!=codes['<|message|>']:
            need(ids[cursor] not in {codes[x] for x in ['<|start|>','<|end|>','<|return|>','<|call|>','<|startoftext|>']},'Control token inside native header.')
            header.append(ids[cursor]);cursor+=1
        need(cursor<len(ids),'Unterminated native header.')
        header=tok.decode(header,skip_special_tokens=False)
        need(header in ['system','user','assistant<|channel|>final'],'Invented/invalid role or assistant channel.')
        role='assistant' if header.startswith('assistant') else header
        roles.append(role);cursor+=1;begin=cursor
        while cursor<len(ids) and ids[cursor] not in {codes['<|end|>'],codes['<|return|>']}:
            need(ids[cursor] not in set(codes.values()),'Reserved control token in source body.')
            result.append((cursor,role,message,ids[cursor]));cursor+=1
        need(cursor>begin and cursor<len(ids),'Empty or unterminated native message.')
        need(ids[cursor]==codes['<|return|>' if role=='assistant' else '<|end|>'],'Wrong native ending for formal role.')
        cursor+=1;message+=1
    need(roles in [['system','user','assistant'],['system','user','assistant','user','assistant']],'Not an authentic complete S/U/A path.')
    return result


def audit_prompt(entry,rows,ids,ledger,tok):
    encoded=tok(entry.prompt,add_special_tokens=False,return_offsets_mapping=True)
    need(encoded['input_ids']==ids and len(ids)==entry.length,'Cached prompt token IDs/length differ.')
    parsed=parse_native(ids,tok)
    actual=[(int(row.token_ix),row.role,int(row.message_ix),int(row.token_id)) for row in rows.itertuples()]
    need(actual==parsed,'Complete ordered native content positions/roles/message IDs differ.')
    need(rows.semantic_role.eq(rows.role).all(),'Formal and semantic S/U/A labels disagree.')
    need(not rows.duplicated(['prompt_ix','token_ix']).any(),'Repeated content position.')
    offset=encoded['offset_mapping']
    for row in rows.itertuples():
        left,right=offset[row.token_ix]
        need(entry.prompt[left:right]==row.token,'Token text differs from independent character offsets.')
    pattern=re.compile(r'<\|start\|>(system|user|assistant<\|channel\|>final)<\|message\|>(.*?)(<\|end\|>|<\|return\|>)',re.S)
    matches=list(pattern.finditer(entry.prompt))
    need('<|startoftext|>'+''.join(match.group(0) for match in matches)==entry.prompt,'Unparsed characters/native boundaries.')
    need(len(matches)==len(ledger['messages'])==entry.message_count,'Message count/provenance differs.')
    for index,match in enumerate(matches):
        header,body,ending=match.groups();meta=ledger['messages'][index]
        role='assistant' if header.startswith('assistant') else header
        need(meta['role']==role and ending==('<|return|>' if role=='assistant' else '<|end|>'),'Message formal role/ending differs.')
        body_ids=tok.encode(body,add_special_tokens=False)
        need(0<len(body_ids)<=192 and len(body_ids)==meta['body_tokens'] and sha(body)==meta['serialized_body_sha256']==meta['capped_sha256'],'Serialized body hash/token cap differs.')
        selected=rows[rows.message_ix==index]
        need(len(selected)==len(body_ids),'Per-message complete content count differs.')
        need(all(match.start(2)<=offset[p][0]<offset[p][1]<=match.end(2) for p in selected.token_ix),'Token crosses body/header boundary.')
    return dict(tokens=len(ids),content_tokens=len(rows),messages=len(matches),turns=sum(m.group(1)=='user' for m in matches))


def independent_structure(ledger,audit):
    mapping={row['source_uid']:row for row in ledger}
    need(len(mapping)==len(ledger) and all(row['split']=='test' for row in ledger),'Repeated/non-test ledger source.')
    prepared={uid:row for uid,row in mapping.items() if row['status']=='prepared'}
    need(all(row['status'] in ['prepared','quality-failed'] for row in ledger),'Unexpected source status.')
    union=Union(prepared);families=defaultdict(list)
    for uid,row in prepared.items():families[row['family_id']].append(uid)
    for nodes in families.values():union.join(nodes)
    flags=defaultdict(list)
    for row in audit['message_collision_flags']:
        need(row['source_uid'] in prepared,'Collision evidence belongs to a failed/unknown source.')
        need(row['message_uid'].startswith(row['source_uid']+'/message'),'Collision message identity escaped source.')
        flags[row['source_uid']].append(row)
    cluster_members=defaultdict(set)
    for cluster in audit['duplicate_clusters']:
        nodes=cluster['test_source_uids']
        need(nodes==sorted(set(nodes)) and set(nodes)<=set(prepared),'Duplicate cluster membership invalid.')
        test_messages={row['source_uid'] for row in cluster['messages'] if row['split']=='test'}
        need(test_messages==set(nodes),'Duplicate cluster source/member evidence differs.')
        union.join(nodes)
        for uid in nodes:cluster_members[uid].add(cluster['cluster_id'])
    for uid,row in mapping.items():
        evidence=flags[uid]
        need(len(evidence)==sum(m['role']=='assistant' for m in row['messages']) if row['status']=='prepared' else not evidence,'Assistant collision evidence count differs.')
        old=any(item['old_key_types'] for item in evidence);dev=any(item['dev_key_types'] for item in evidence)
        internal=any(item['within_test_duplicate'] for item in evidence)
        need(row['old_generated_collision']==old and row['dev_assistant_collision']==dev and row['within_test_assistant_duplicate']==internal,'Collision flags disagree with message evidence.')
        need(bool(row['old_generated_references'])==old and bool(row['dev_assistant_references'])==dev,'Collision reference flags disagree.')
        need(set(row['assistant_duplicate_cluster_ids'])==cluster_members[uid],'Source duplicate cluster IDs differ.')
        clean=row['status']=='prepared' and not old and not dev
        need(row['eligible_clean_test']==clean,'Clean eligibility depends on something beyond prepared/direct old-dev collision.')
        expected=([] if row['status']=='prepared' else ['quality-failed'])+(['assistant-collision-with-old-generated-content'] if old else [])+(['assistant-collision-with-fixed-development-content'] if dev else [])
        need(row['eligibility_reasons']==expected,'Eligibility reason ledger differs.')
    groups=union.groups();expected=[]
    for nodes in groups:
        cid='test-dependency-'+canonical(nodes)[:16]
        need(all(mapping[uid]['dependency_component_id']==cid for uid in nodes),'Independent family/duplicate component ID differs.')
        expected.append(dict(component_id=cid,source_uids=nodes,clean_source_uids=[uid for uid in nodes if mapping[uid]['eligible_clean_test']],system_family_ids=sorted({mapping[uid]['family_id'] for uid in nodes})))
    need(audit['components']==expected,'Component membership/ordering differs from independent union.')
    family_counts=dict(fixed=32,all_prepared=len(families),clean_prepared=len({row['family_id'] for row in prepared.values() if row['eligible_clean_test']}))
    eligibility=dict(all_fixed_ids=len(ledger),prepared=len(prepared),quality_failed=len(ledger)-len(prepared),eligible_clean_test=sum(row['eligible_clean_test'] for row in ledger),old_collision=sum(row['old_generated_collision'] for row in ledger),dev_collision=sum(row['dev_assistant_collision'] for row in ledger),internal_repeat=sum(row['within_test_assistant_duplicate'] for row in ledger))
    need(audit['system_family_counts']==family_counts and audit['eligibility_counts']==eligibility,'Published family/eligibility counts differ.')
    need(audit['dependency_components']==len(expected) and audit['clean_dependency_components']==sum(bool(row['clean_source_uids']) for row in expected),'Published all/clean component counts differ.')
    return dict(eligibility=eligibility,system_family_counts=family_counts,components=expected)


def actual(output):
    sys.path.insert(0,str(MODULES))
    import systemness_natural_test as b
    complete_path=b.OUT/'data-complete.json'
    need(complete_path.is_file() and digest(complete_path)==EXPECTED_COMPLETE_SHA,'Completed preparation SHA/marker not yet present or changed; never read partial cache.')
    complete,inputs,rows,tokens,ledger=b.verified_data()  # Mandatory metadata/cache verifier only.
    protocol=read(b.OUT/'protocol.json');declaration=read(b.OUT/'frozen-readouts.json');frozen=complete['frozen']
    provenance={}
    def bind(path,expected=None):
        path=Path(path).resolve();value=digest(path)
        if expected is not None:need(value==expected,'Bound SHA changed: '+str(path))
        provenance[str(path)]=value;return value
    bind(__file__);bind(complete_path,EXPECTED_COMPLETE_SHA);bind(b.OUT/'protocol.json',complete['protocol_sha256']);bind(b.OUT/'frozen-readouts.json',complete['frozen_readouts_sha256'])
    need(protocol['frozen']==declaration['frozen']==frozen and declaration['protocol_sha256']==canonical(protocol),'Pre-read model/source declaration differs.')
    need(frozen['primary_candidate']=='joint-mlp128' and frozen['models']['head-selected']==PRIMARY_SHA and frozen['thresholds']['head-selected']==PRIMARY_THRESHOLD,'Independent exact primary pin differs.')
    need(set(frozen['models'])==set(frozen['model_paths'])==set(frozen['thresholds'])==set(NAMES),'Four frozen models/threshold coverage differs.')
    for name,sha256 in protocol['dependencies_sha256'].items():bind(MODULES/name,sha256)
    for name in NAMES:
        bind(frozen['model_paths'][name],frozen['models'][name])
        need(np.isfinite(frozen['thresholds'][name]) and 0<=frozen['thresholds'][name]<=1.00001,'Invalid frozen threshold.')
    source_selection={'residual':b.data.OUT/'selection.json','base-attention':b.head_confirm.attention.OUT/'selection.json','dense128':b.head_confirm.long_data.OUT/'candidates/dense-mlp128.json','head-selected':b.head_probes.OUT/'selection.json'}
    for name,path in source_selection.items():
        bind(path,frozen['selection_sha256'][name]);value=read(path)
        threshold=value['validation']['system_threshold'] if name=='dense128' else value['system_threshold']
        need(threshold==frozen['thresholds'][name],'Selection frozen threshold differs: '+name)
    for name,value in complete['cache_sha256'].items():bind(b.CACHE/name,value)
    for name,value in complete['metadata_sha256'].items():bind(b.OUT/name,value)
    bind(b.CACHE/'initialized.json')
    audit=read(b.OUT/'duplicate-audit.json');coverage=read(b.OUT/'coverage.json');read_audit=read(b.OUT/'source-read-audit.json')
    need(len(ledger)==128 and Counter(row['dataset'] for row in ledger)=={'oasst':64,'toxicchat0124':64} and len({row['family_id'] for row in ledger})==32 and set(Counter(row['family_id'] for row in ledger).values())=={4},'Fixed128/32 or dataset assignment differs.')
    structure=independent_structure(ledger,audit)
    need(complete['components']==structure['components'] and complete['system_family_counts']==structure['system_family_counts'],'Completion structure differs from independent audit.')
    prepared={row['source_uid']:row for row in ledger if row['status']=='prepared'}
    need(len(inputs)==len(tokens)==len(prepared) and len(rows)==complete['content_rows'] and len(inputs)==complete['prepared_prompts'],'Prepared input/content counts differ.')
    need(inputs.prompt_ix.tolist()==list(range(len(inputs))) and np.array_equal(rows.cache_ix,np.arange(len(rows))) and np.array_equal(rows.sample_ix,5_000_000_000_000+np.arange(len(rows))),'Continuous native row identity differs.')
    need(set(inputs.source_uid)==set(rows.source_uid)==set(prepared),'Native input/row source coverage differs.')
    tok=b.long_data.load_tokenizer();totals=Counter();prompt_audits=[]
    for entry in inputs.itertuples():
        native_rows=rows[rows.prompt_ix==entry.prompt_ix]
        source=prepared[entry.source_uid]
        for field in ['source_uid','dataset','native_id','split','template_id','family_id','dependency_component_id','eligible_clean_test','old_generated_collision','dev_assistant_collision','within_test_assistant_duplicate']:
            need(getattr(entry,field)==source[field] and native_rows[field].eq(source[field]).all(),'Native input/row ownership differs: '+field)
        need(entry.wrapper=='native' and native_rows.wrapper.eq('native').all() and entry.group_id==entry.source_uid and native_rows.group_id.eq(entry.source_uid).all(),'Formal native wrapper/group mapping differs.')
        result=audit_prompt(entry,native_rows,tokens[entry.prompt_ix],source,tok);totals.update(result);prompt_audits.append(dict(source_uid=entry.source_uid,**result))
    role_counts={role:int(rows.role.eq(role).sum()) for role in SUA}
    need(coverage['role_counts']==role_counts and coverage['input_prompts']==len(inputs) and coverage['content_rows']==len(rows) and coverage['no_observations']==(not len(inputs)),'Independent coverage counts differ.')
    need(not rows.role.isin(['cot','tool']).any() and all(coverage['roles_with_no_observations'][role]['support']==0 and coverage['roles_with_no_observations'][role]['recall'] is None for role in ['cot','tool']),'Invented CoT/tool support.')
    counts={}
    for stratum,selected in [('all-prepared',rows),('clean-test',rows[rows.eligible_clean_test.eq(True)])]:
        counts[stratum]={}
        for dataset in ['all','oasst','toxicchat0124']:
            subset=selected if dataset=='all' else selected[selected.dataset.eq(dataset)]
            counts[stratum][dataset]=dict(source_roots=int(subset.source_uid.nunique()),content_tokens=len(subset),system_families=int(subset.family_id.nunique()),dependency_components=int(subset.dependency_component_id.nunique()),role_support={role:int(subset.role.eq(role).sum()) for role in ALL_ROLES})
    gallery=[sorted(prepared.keys()|{row['source_uid'] for row in ledger if row['status']!='prepared'})[i] for i in np.linspace(0,127,12,dtype=int)]
    need(gallery==protocol['gallery_source_uids']==complete['gallery_source_uids'],'Fixed gallery IDs differ.')
    need(read_audit['test']['test_assistant_materialized'] is True and read_audit['reference_dev']['test_assistant_materialized'] is False and read_audit['reference_dev']['reference_dev_assistant_materialized'] is True,'Read-subset audit scope differs.')
    for part in ['test','reference_dev']:
        scope=read_audit[part]
        need(scope['protocol_sha256_before_text_materialization']==complete['protocol_sha256'] and scope['oasst_nonselected_text_rows_materialized']==0 and scope['development_reader_called'] is False,'Frozen-before-read/logical source subset proof differs.')
        need(scope['toxicchat']['nonselected_assistant_fields_decoded']==0 and scope['toxicchat']['nonselected_assistant_fields_buffered']==0,'Nonselected ToxicChat assistant was materialized.')
    for path,sha256 in provenance.items():need(digest(path)==sha256,'Bound artifact changed during audit: '+path)
    result=dict(stage='actual_natural128_preparation_audit',source_sha256=digest(__file__),data_complete_sha256=EXPECTED_COMPLETE_SHA,verified_builder_cache_first=True,
        fixed_roots=128,fixed_system_families=32,prepared_prompts=len(inputs),quality_failures=128-len(inputs),content_rows=len(rows),input_tokens=sum(map(len,tokens)),messages=totals['messages'],
        independent_counts=counts,independent_structure=structure,independent_native_token_replay=True,prompt_replay_count=len(prompt_audits),gallery_source_uids=gallery,
        primary_model_sha256=PRIMARY_SHA,primary_system_threshold=PRIMARY_THRESHOLD,model_hashes=frozen['models'],thresholds=frozen['thresholds'],provenance_sha256=provenance,
        test_probability_files_opened=False,test_metric_files_opened=False,stored_validation_probabilities_read_for_completion_verification=True,source_reply_replay_performed=False,
        classifier_predictions_generated=False,model_or_threshold_selection=False,formal_preparation_run=False,gpu_used=False,formal_outputs_written=False,
        limitation='Collision extraction from raw source replies is not replayed here. Published message collision evidence, eligibility and union component membership are independently crosschecked; the root separately performs frozen source replay. All serialized content after the declared192-token per-message cap is audited.')
    write(output,result);print(json.dumps(dict(output=str(output),prepared_prompts=len(inputs),content_rows=len(rows),input_tokens=sum(map(len,tokens)),components=len(structure['components']),clean=structure['eligibility']['eligible_clean_test']),sort_keys=True));return result


def preflight(output):
    from types import SimpleNamespace
    class Toy:
        def __init__(self):self.special={text:i+1 for i,text in enumerate(RESERVED)};self.reverse={i:text for text,i in self.special.items()}
        def convert_tokens_to_ids(self,text):return self.special[text]
        def __call__(self,text,**kwargs):
            units=list(re.finditer(r'<\|[^>]+\|>|.',text,re.S))
            return dict(input_ids=[self.special[match.group()] if match.group() in self.special else 100+ord(match.group()) for match in units],offset_mapping=[match.span() for match in units])
        def encode(self,text,**kwargs):return self(text)['input_ids']
        def decode(self,ids,**kwargs):return ''.join(self.reverse[i] if i in self.reverse else chr(i-100) for i in ids)
    tok=Toy();prompt='<|startoftext|>';messages=[]
    for role,body in [('system','Normal system'),('user','Public user'),('assistant','Recorded reply')]:
        prompt+='<|start|>'+('assistant<|channel|>final' if role=='assistant' else role)+'<|message|>'+body+('<|return|>' if role=='assistant' else '<|end|>')
        messages.append(dict(role=role,body_tokens=len(tok.encode(body)),serialized_body_sha256=sha(body),capped_sha256=sha(body)))
    ids=tok.encode(prompt);encoded=tok(prompt);rows=[]
    for position,role,mi,tid in parse_native(ids,tok):
        left,right=encoded['offset_mapping'][position];rows.append(dict(token_ix=position,role=role,semantic_role=role,message_ix=mi,token_id=tid,token=prompt[left:right],prompt_ix=0))
    rows=pd.DataFrame(rows);entry=SimpleNamespace(prompt=prompt,length=len(ids),message_count=3);ledger=dict(messages=messages)
    audit_prompt(entry,rows,ids,ledger,tok);checks=['independent token-state/native-character-span parser agrees on completeSUA']
    def reject(name,fn):
        try:fn()
        except (RuntimeError,KeyError,ValueError):checks.append(name)
        else:raise RuntimeError('Corruption was accepted: '+name)
    bad=rows.copy();bad.loc[0,'role']='tool';reject('corrupt formal role rejected',lambda:audit_prompt(entry,bad,ids,ledger,tok))
    reject('missing content position rejected',lambda:audit_prompt(entry,rows.iloc[1:],ids,ledger,tok))
    bad=pd.concat([rows,rows.iloc[:1]],ignore_index=True);reject('duplicate content position rejected',lambda:audit_prompt(entry,bad,ids,ledger,tok))
    bad=rows.copy();bad.loc[0,'token_id']+=1;reject('corrupt content token ID rejected',lambda:audit_prompt(entry,bad,ids,ledger,tok))
    bad=rows.copy();bad.loc[0,'token']='changed';reject('corrupt token text rejected',lambda:audit_prompt(entry,bad,ids,ledger,tok))
    bad_ids=ids.copy();bad_ids[-1]=tok.convert_tokens_to_ids('<|end|>');reject('assistant return boundary corruption rejected',lambda:parse_native(bad_ids,tok))
    bad_ledger=deepcopy(ledger);bad_ledger['messages'][2]['capped_sha256']='wrong';reject('body provenance hash corruption rejected',lambda:audit_prompt(entry,rows,ids,bad_ledger,tok))
    check=Union(['a','b','c','d']);check.join(['a','b']);check.join(['b','c']);need(check.groups()==[['a','b','c'],['d']],'Independent transitive union failed.');checks.append('independent transitive component union')
    result=dict(stage='synthetic_natural_preparation_audit_preflight',checks=checks,checks_passed=len(checks),source_sha256=digest(__file__),test_cache_or_replies_read=False,test_probabilities_or_metrics_read=False,gpu_used=False)
    write(output,result);print(json.dumps(dict(output=str(output),checks_passed=len(checks))));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--actual',action='store_true');group.add_argument('--preflight',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.actual:actual(args.output or Path('/tmp/systemness-natural-preparation-audit.json'))
    elif args.preflight:preflight(args.output or Path('/tmp/systemness-natural-preparation-audit-preflight.json'))
    else:parser.print_help()
if __name__=='__main__':main()
