"""Root-authorized data-only audit. No producer imports or model/probability reads."""
import ast,hashlib,json,re,unicodedata,urllib.parse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
OUT=HERE/'outputs/local-gptoss120b/systemness-complementary-natural-confirmation'
COMPLETE_SHA='4b46e0c1e1e460cbdfb35058e68ebb7e342e27e11c17c09e71c7aaaa38f7d5fe'
SNAPSHOT_SHA='58f95bd32c0866881f940fc30075487770abe04a28eb7ee383b44e2718b8d6e6'
SOURCE_SHA='cc9c8f3ad46c214747e3a49ab18f81701b88f09e8805b229ae56c55127944fa4'
HELPER_SHA='12b2078ffdb358801855ef9c31a31816e96724acd813f340f0caf49683e834ae'
SPECIAL=['<|start|>','<|message|>','<|end|>','<|return|>','<|call|>','<|channel|>','<|startoftext|>']
CHECKS=[];PROVENANCE={}
def check(name,value=True):
    if not bool(value):raise RuntimeError(name)
    CHECKS.append(name)
def sha_bytes(v):return hashlib.sha256(v).hexdigest()
def sha_text(v):return sha_bytes(v.encode())
def read(path):return json.loads(Path(path).read_text())
def digest(path,want=None):
    path=Path(path);check_path=str(path.resolve());assert not check_path.startswith('/workspace') and '/inference/' not in check_path and path.suffix not in ['.npz','.npy','.f16','.f32']
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*2**20),b''):h.update(b)
    value=h.hexdigest();assert want is None or value==want,'SHA differs: '+str(path);PROVENANCE[str(path)]=value;return value
def canon(value):return sha_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))
def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).split()).casefold()
class Tok:
    def __init__(self,path):self.t=Tokenizer.from_file(str(path));self.ids={s:self.t.token_to_id(s) for s in SPECIAL};assert all(type(i)is int for i in self.ids.values())
    def encode(self,text):return self.t.encode(text,add_special_tokens=False).ids
    def decode(self,ids):return self.t.decode(list(map(int,ids)),skip_special_tokens=False)
def keys(text,tok):
    cleaned=text
    for marker in SPECIAL:cleaned=cleaned.replace(marker,'['+marker[2:-2]+']')
    forms=[text,norm(text),cleaned,norm(cleaned)];result={('full',sha_text(s)) for s in forms}
    result.update(('prefix192',sha_bytes(np.asarray(tok.encode(s)[:192],np.int64).tobytes())) for s in forms)
    for raw in re.findall(r'https?://[^\s<>"\'\)\]]+',text):
        p=urllib.parse.urlsplit(raw.rstrip('.,;:!?'));q=urllib.parse.urlencode(sorted((k,v) for k,v in urllib.parse.parse_qsl(p.query,keep_blank_values=True) if not k.lower().startswith('utm_')))
        result.add(('url',urllib.parse.urlunsplit((p.scheme.lower(),p.netloc.lower(),p.path.rstrip('/'),q,''))))
    return result
def parse_native(ids,tok):
    """Independent sequential Harmony parser; headers do not come from cached roles."""
    assert ids and ids[0]==tok.ids['<|startoftext|>'];pos=1;messages=[];rows=[]
    while pos<len(ids):
        assert ids[pos]==tok.ids['<|start|>'];pos+=1;head=[]
        while pos<len(ids) and ids[pos]!=tok.ids['<|message|>']:head.append(ids[pos]);pos+=1
        assert pos<len(ids);header=tok.decode(head);role={'system':'system','user':'user','assistant<|channel|>final':'assistant'}.get(header);assert role is not None,'Non-SUA or abnormal native header'
        pos+=1;start=pos
        while pos<len(ids) and ids[pos] not in [tok.ids[x] for x in ['<|end|>','<|return|>','<|call|>']]:
            assert ids[pos] not in [tok.ids[x] for x in ['<|start|>','<|message|>','<|startoftext|>']];pos+=1
        assert pos<len(ids) and pos>start and ids[pos]==tok.ids['<|return|>' if role=='assistant' else '<|end|>']
        message_ix=len(messages);body=ids[start:pos];messages.append(dict(role=role,header=header,body=tok.decode(body),tokens=body))
        rows.extend((i,int(ids[i]),role,message_ix,i-start) for i in range(start,pos));pos+=1
    assert [m['role'] for m in messages] in [['system','user','assistant'],['system','user','assistant','user','assistant']]
    return rows,messages
class DSU:
    def __init__(self,items):self.p={s:s for s in items}
    def find(self,x):
        while x!=self.p[x]:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def join(self,a,b):self.p[self.find(a)]=self.find(b)
    def groups(self):
        result=defaultdict(set)
        for x in self.p:result[self.find(x)].add(x)
        return {frozenset(v) for v in result.values()}
def main():
    digest(OUT/'data-complete.json',COMPLETE_SHA);meta=read(OUT/'data-complete.json');plan=read(OUT/'plan.json');digest(OUT/'plan.json',meta['plan_sha256'])
    check('Frozen data source/helper code pins',meta['source_sha256']==SOURCE_SHA and meta['dependencies_sha256']['systemness_complementary_guard_confirm_snapshot.py']==HELPER_SHA)
    for name,want in meta['dependencies_sha256'].items():digest(HERE/name,want)
    snapshot=read(meta['snapshot_path']);digest(meta['snapshot_path'],SNAPSHOT_SHA);bare=dict(snapshot);claimed=bare.pop('snapshot_sha256')
    check('58f snapshot canonical/source/model-threshold statement bound',claimed==canon(bare)==meta['frozen_sha256'] and snapshot==meta['frozen']==plan['frozen'] and snapshot['helper_source_sha256']==HELPER_SHA and snapshot['calibrated_decision']['eligible'] is True and snapshot['original_failed_trial']['promoted'] is False)
    for path,want in snapshot['runtime_files_sha256'].items():
        if path.endswith('.py'):digest(path,want)
    for name,want in meta['metadata_sha256'].items():digest(OUT/name,want)
    cache=Path(meta['cache_path'])
    for name,want in meta['cache_sha256'].items():digest(cache/name,want)
    freeze=read(OUT/'source-freeze.json');digest(OUT/'source-freeze.json',meta['source_freeze_sha256'])
    check('IDs/templates/gallery freeze predates assistant materialization',freeze['assistant_materialized'] is False and freeze['fixed_roots']==128 and freeze['fixed_system_families']==32 and freeze['plan_sha256']==meta['plan_sha256'] and freeze['source_sha256']==SOURCE_SHA and freeze['frozen_sha256']==meta['frozen_sha256'])
    for name,want in freeze['artifacts_sha256'].items():digest(OUT/name,want)
    inputs=pd.read_parquet(cache/'inputs.parquet');rows=pd.read_parquet(cache/'rows.parquet');tokens=read(cache/'token-ids.json');ledgerobj=read(meta['ledger_path']);ledger=ledgerobj['records'];duplicates=read(meta['duplicate_audit_path']);sourcekeys=read(OUT/'natural-source-keys.json')['source_keys'];assignments=read(OUT/'template-assignments.json')['assignments'];templates=read(OUT/'system-templates.json')['templates'];gallery=read(OUT/'gallery-index.json');selection=read(OUT/'source-selection-audit.json')
    check('128 sources/64OA64TC all prepared clean and no replacements',len(inputs)==len(tokens)==len(ledger)==len(sourcekeys)==128 and Counter(k['dataset'] for k in sourcekeys)=={'oasst':64,'toxicchat0124':64} and len({k['source_uid'] for k in ledger})==128 and all(k['status']=='prepared' and k['eligible_clean_test'] for k in ledger) and ledgerobj['all_fixed_ids_retained'] is True and ledgerobj['no_replacements'] is True)
    uid=lambda k:'complementary-natural:'+k['dataset']+':'+k['native_id'];keymap={uid(k):k for k in sourcekeys};lmap={x['source_uid']:x for x in ledger}
    check('Every frozen root retained without reselection',set(keymap)==set(lmap) and [r['source_index'] for r in sourcekeys]==list(range(128)) and [r['document_id'] for r in sourcekeys]==list(range(10000000,10000128)) and all(lmap[u]['native_id']==k['native_id'] and lmap[u]['root_raw_sha256']==k['root_raw_sha256'] and lmap[u]['revision']==k['revision'] for u,k in keymap.items()))
    oldpath=next(Path(p) for p in meta['source_signature']['history_files_sha256'] if '/systemness-natural-sources/' in p and p.endswith('/natural-source-keys.json'));digest(oldpath,meta['source_signature']['old448_source_keys_sha256']);oldkeys=read(oldpath)['source_keys']
    check('New roots/tree IDs/raw/normalized root fingerprints disjoint from old448',len(oldkeys)==448 and not {(k['dataset'],k['native_id']) for k in oldkeys}&{(k['dataset'],k['native_id']) for k in sourcekeys} and not {k.get('root_message_id') for k in oldkeys if k.get('root_message_id')}&{k.get('root_message_id') for k in sourcekeys if k.get('root_message_id')} and not {k['root_raw_sha256'] for k in oldkeys}&{k['root_raw_sha256'] for k in sourcekeys} and not {k['root_normalized_sha256'] for k in oldkeys}&{k['root_normalized_sha256'] for k in sourcekeys})
    tokpath=Path(meta['source_signature']['tokenizer_path']);assert str(tokpath)=='/dev/shm/role-analysis-gptoss120b'
    for name,want in meta['source_signature']['tokenizer_sha256'].items():digest(tokpath/name,want)
    tok=Tok(tokpath/'tokenizer.json')
    oldtemplatepath=oldpath.parent/'system-templates.json';digest(oldtemplatepath,meta['source_signature']['history_files_sha256'][str(oldtemplatepath)]);oldtemplates=read(oldtemplatepath)['templates']
    for name in ['systemness_data.py','systemness_challenges.py']:
        for node in ast.walk(ast.parse((HERE/name).read_text())):
            if not isinstance(node,ast.Assign) or not any(isinstance(t,ast.Name) and t.id in ['systems','instructions'] for t in node.targets):continue
            try:values=ast.literal_eval(node.value)
            except (ValueError,TypeError,SyntaxError):continue
            if isinstance(values,list) and all(isinstance(v,str) for v in values):oldtemplates.extend(dict(source=name,catalog=node.targets[0].id,text=v) for v in values)
    check('Old124 template inventory independently recovered and bound',len(oldtemplates)==124 and canon(oldtemplates)==plan['template_audit']['old_catalog_sha256'])
    old_template_keys=set().union(*(keys(t['text'],tok) for t in oldtemplates));new_used=set()
    for t in templates:
        k=keys(t['text'],tok);assert not k&old_template_keys and not k&new_used;new_used|=k;assert '<|' not in t['text'] and sha_text(t['text'])==t['text_sha256'] and sha_text(norm(t['text']))==t['normalized_sha256'] and len(tok.encode(t['text']))==t['token_count']<=192
    check('32 new intact normal system wordings disjoint from old124/all new families',len(templates)==32 and len({t['family_id'] for t in templates})==32 and templates==plan['templates'])
    expected_assign=[]
    for dataset in ['oasst','toxicchat0124']:
        part=[k for k in sourcekeys if k['dataset']==dataset]
        for i,k in enumerate(part):expected_assign.append(dict(dataset=dataset,native_id=k['native_id'],source_uid=uid(k),template_id=templates[i%32]['template_id'],family_id=templates[i%32]['family_id'],split='test'))
    check('Deterministic 2OA+2TC assignment per system family',assignments==expected_assign and Counter((x['family_id'],x['dataset']) for x in assignments)=={(t['family_id'],d):2 for t in templates for d in ['oasst','toxicchat0124']})
    tmap={t['template_id']:t for t in templates};amap={a['source_uid']:a for a in assignments}
    check('All-content sequential namespace and SUA-only support',np.array_equal(rows.cache_ix,np.arange(len(rows))) and np.array_equal(rows.sample_ix,10000000000000+np.arange(len(rows))) and inputs.prompt_ix.tolist()==list(range(128)) and set(rows.role)=={'system','user','assistant'} and rows.semantic_role.eq(rows.role).all() and rows.role.value_counts().to_dict()=={'system':4124,'user':6121,'assistant':19751} and len(rows)==29996 and rows.split.eq('test').all() and rows.wrapper.eq('native').all())
    # Explicit historical user-query inventory only; no old replies/models/probabilities.
    legacy_path=next(Path(p) for p in meta['source_signature']['history_files_sha256'] if p.endswith('/user_queries.csv'));digest(legacy_path,meta['source_signature']['history_files_sha256'][str(legacy_path)]);legacy=pd.read_csv(legacy_path,keep_default_na=False,usecols=['dataset','user_query'])
    legacy_keys=set().union(*(keys(s,tok) for s in legacy.user_query));legacy_raw={sha_text(s) for s in legacy.user_query};legacy_norm={sha_text(norm(s)) for s in legacy.user_query}
    check('225 legacy requests excluded by root raw/normalized fingerprints',len(legacy)==225 and selection['legacy_requested_roots']==200 and selection['legacy_user_turns']==225 and not legacy_raw&{k['root_raw_sha256'] for k in sourcekeys} and not legacy_norm&{k['root_normalized_sha256'] for k in sourcekeys})
    decoded={};assistant_keys=[];parser_count=0;messages_count=0
    for entry in inputs.itertuples():
        ids=tok.encode(entry.prompt);assert ids==tokens[entry.prompt_ix] and len(ids)==entry.length and tok.decode(ids)==entry.prompt
        parsed,messages=parse_native(ids,tok);part=rows[rows.prompt_ix==entry.prompt_ix];record=lmap[entry.source_uid];assignment=amap[entry.source_uid]
        actual=list(part[['token_ix','token_id','role','message_ix','body_offset']].itertuples(index=False,name=None));assert parsed==actual and len(messages)==entry.message_count==len(record['messages'])
        assert entry.family_id==assignment['family_id']==record['family_id'] and entry.template_id==assignment['template_id']==record['template_id'];assert messages[0]['body']==tmap[entry.template_id]['text']
        for mi,msg in enumerate(messages):
            evidence=record['messages'][mi];assert msg['role']==evidence['role'] and sha_text(msg['body'])==evidence['serialized_body_sha256'] and len(msg['tokens'])==evidence['body_tokens']<=192 and tok.encode(msg['body'])==msg['tokens'];decoded[(entry.source_uid,mi)]=msg
            if msg['role']=='user':assert not keys(msg['body'],tok)&legacy_keys
            if msg['role']=='assistant':assistant_keys.append((entry.source_uid,keys(msg['body'],tok)))
        for field in ['source_uid','dataset','native_id','split','template_id','family_id','document_id','dependency_component_id','eligible_clean_test','old_generated_collision','dev_assistant_collision','within_test_assistant_duplicate','historical_role_input_collision','public448_assistant_collision']:
            assert getattr(entry,field)==record[field] and part[field].eq(record[field]).all()
        assert entry.source_set=='complementary-natural' and entry.uid==f'complementary-natural:{entry.prompt_ix}' and entry.group_id==entry.source_uid
        parser_count+=len(parsed);messages_count+=len(messages)
    check('Independent Harmony parser labels and full row alignment on every token',parser_count==len(rows) and inputs.length.sum()==meta['input_tokens'])
    check('Every native normal system header and contiguous capped body proven; no sampling/gap fill',all(t['body_tokens']>0 for r in ledger for t in r['messages']) and parser_count==meta['coverage']['content_rows'] and meta['coverage']['no_sparse_sampling'] is True)
    check('Prepared user bodies also disjoint from legacy225 full/norm/sanitized/prefix/URL keys')
    dsu=DSU(lmap);families=defaultdict(list)
    for r in ledger:families[r['family_id']].append(r['source_uid'])
    for members in families.values():
        for s in members[1:]:dsu.join(members[0],s)
    owners=defaultdict(set)
    for source,ks in assistant_keys:
        for k in ks:owners[k].add(source)
    repeats=[members for members in owners.values() if len(members)>1]
    for members in repeats:
        members=sorted(members)
        for s in members[1:]:dsu.join(members[0],s)
    check('No new capped assistant repetition or published external bridge',not repeats and duplicates['duplicate_clusters']==[] and duplicates['external_reference_bridges']==[])
    published={frozenset(c['source_uids']) for c in duplicates['components']};check('Independent family/repeat union equals32 full graph components',dsu.groups()==published and len(published)==32 and all(len(g)==4 for g in published))
    for component in duplicates['components']:
        assert component['source_uids']==component['clean_source_uids'] and component['system_family_ids']==sorted({lmap[s]['family_id'] for s in component['source_uids']})
        assert all(lmap[s]['dependency_component_id']==component['component_id'] for s in component['source_uids'])
    strata={}
    for clean in [False,True]:
        for dataset in ['all','oasst','toxicchat0124']:
            selected=[r for r in ledger if (not clean or r['eligible_clean_test']) and (dataset=='all' or r['dataset']==dataset)];ids={r['source_uid'] for r in selected};parts=rows[rows.source_uid.isin(ids)]
            strata[('clean' if clean else 'all')+'/'+dataset]=dict(roots=len(ids),families=len({r['family_id'] for r in selected}),components=len({r['dependency_component_id'] for r in selected}),tokens=len(parts),role_counts=parts.role.value_counts().to_dict())
    check('Original32 component IDs retained in all/clean/source strata',all(s['components']==32 and s['families']==32 for s in strata.values()) and all(strata['all/'+d]==strata['clean/'+d] for d in ['all','oasst','toxicchat0124']))
    expected_gallery=[]
    for dataset in ['oasst','toxicchat0124']:
        group=[k for k in sourcekeys if k['dataset']==dataset];expected_gallery += [uid(group[i]) for i in np.linspace(0,63,6,dtype=int)]
    check('Fixed6OA6TC gallery before assistant/probabilities, no score-based selection',gallery['source_uids']==expected_gallery==meta['gallery_source_uids'] and gallery['selected_before_assistant_materialization'] is True and gallery['selected_before_probabilities'] is True and gallery['score_based_selection'] is False and gallery['quality_failed_keep_placeholders'] is True)
    check('Clean flags/components and zero CoT/tool support declared consistently',all(not r['old_generated_collision'] and not r['dev_assistant_collision'] and not r['within_test_assistant_duplicate'] and not r['historical_role_input_collision'] and not r['public448_assistant_collision'] and not r['eligibility_reasons'] for r in ledger) and meta['components']==duplicates['components'] and meta['dependency_components']==meta['clean_dependency_components']==32 and meta['coverage']['roles_with_no_observations']=={'cot':{'support':0,'recall':None,'status':'N/A'},'tool':{'support':0,'recall':None,'status':'N/A'}})
    check('Data-only no generation/classifier/probability/GPU/primary actions',all(meta[k] is False for k in ['model_generation_run','gpu_work_run','classifier_probabilities_read','stored_validation_probabilities_read_for_completion_verification','test_classifier_probabilities_read','test_metrics_read','primary_changed']))
    for path,want in list(PROVENANCE.items()):digest(path,want)
    report=dict(stage='actual_completed_complementary_natural_data_only_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=digest(__file__),data_complete_sha256=COMPLETE_SHA,builder_source_sha256=SOURCE_SHA,helper_source_sha256=HELPER_SHA,snapshot_file_sha256=SNAPSHOT_SHA,checks=CHECKS,checks_passed=len(CHECKS),source_roots=128,prepared=128,clean=128,source_counts=dict(Counter(k['dataset'] for k in sourcekeys)),families=32,dependency_components=32,content_rows=len(rows),messages=messages_count,role_support=rows.role.value_counts().to_dict(),strata=strata,fixed_gallery_source_uids=expected_gallery,provenance_sha256=PROVENANCE,tokenizer_loaded_without_model=True,producer_prepare_or_verify_called=False,model_weights_loaded=False,gpu_used=False,classifier_probabilities_or_metrics_read=False,limitations='No independent raw public-source/history full-tree replay: root separate source verify covers that. Independently checked frozen IDs/old448 trees+roots, legacy225 query keys against selected roots and all prepared capped user bodies, old124 system fingerprints, full cached native labeling, and family/internal-repeat graph. Historical prepared-role/uncapped assistant collisions are bound published source-audit evidence, not recomputed here.')
    output=Path('/tmp/confirmation_review_complementary_natural_data_actual_audit.json');output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report_path=str(output),report_sha256=sha_bytes(output.read_bytes()),checks_passed=len(CHECKS),source_roots=128,content_rows=len(rows),role_support=report['role_support'])),flush=True)
if __name__=='__main__':main()
