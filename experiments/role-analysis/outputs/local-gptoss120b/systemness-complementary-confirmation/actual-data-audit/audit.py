"""Authorized read-only independent C4 DATA audit; no predictor/model/GPU/p.
No production prepare/verify/helper is imported; no C4 archive rescan.
"""
import hashlib,json,re,tempfile,time
from pathlib import Path
from collections import Counter
from urllib.parse import urlsplit,urlunsplit
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
BASE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/local-gptoss120b')
OUT=BASE/'systemness-complementary-confirmation';COMPLETE_SHA='85629f54d51cde4d5422167e2742876dd6e4cca1efe3308ce0ed2d108203e8a6'
SNAPSHOT_SHA='58f95bd32c0866881f940fc30075487770abe04a28eb7ee383b44e2718b8d6e6';CANONICAL='60e683376f7fd4535c1cc307f028651d1c0c23a0d1449b988438a90d98808f46'
ROLES=['system','user','cot','assistant','tool'];HASH_FIELDS=['body_sha256','full_body_sha256','truncated_body_sha256','short_body_sha256','prefix192_sha256','prefix1024_sha256','full_text_sha256']
checks=[];bindings={};start=time.time()
def check(name,value):assert bool(value),name;checks.append(name)
def digest(p):
 p=Path(p);assert str(p)!='/workspace' and not str(p).startswith('/workspace/')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
def bind(p,want):assert digest(p)==want,str(p);bindings[str(p)]=want
def read(p):return json.loads(Path(p).read_text())
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def canonical_url(u):
 if not u:return None
 try:
  p=urlsplit(u)
  if not p.scheme or not p.hostname:return u
  host=p.hostname.lower();host='['+host+']' if ':' in host else host
  if p.port is not None and not((p.scheme.lower()=='http' and p.port==80)or(p.scheme.lower()=='https' and p.port==443)):host+=':'+str(p.port)
  return urlunsplit((p.scheme.lower(),(p.netloc.rsplit('@',1)[0]+'@' if '@' in p.netloc else '')+host,p.path,p.query,''))
 except ValueError:return u
bind(OUT/'data-complete.json',COMPLETE_SHA);c=read(OUT/'data-complete.json');cache=Path(c['cache_path'])
for name,want in c['cache_sha256'].items():bind(cache/name,want)
for name,want in c['metadata_sha256'].items():bind(OUT/name,want)
bind(OUT/'plan.json',c['plan_sha256']);plan=read(OUT/'plan.json');sig=plan['source_signature'];snapshot=Path(c['snapshot_path']);bind(snapshot,SNAPSHOT_SHA)
check('sealed-snapshot-hash-chain',c['snapshot_file_sha256']==sig['snapshot_file_sha256']==SNAPSHOT_SHA and c['frozen_sha256']==sig['frozen_sha256']==CANONICAL and read(snapshot)==c['frozen']==plan['frozen'])
check('source-plan-and-dependency-bytes',c['source_sha256']==digest(BASE.parent.parent/'systemness_complementary_confirmation_data.py') and all(digest(BASE.parent.parent/name)==want for name,want in c['dependencies_sha256'].items()))
check('snapshot-plan-before-source-ID-files-mtime',snapshot.stat().st_mtime_ns<=(OUT/'plan.json').stat().st_mtime_ns<=(OUT/'document-keys.json').stat().st_mtime_ns)
check('CPU-data-only-producer-flags',all(c[k] is False for k in ['test_probabilities_read','classifier_probabilities_read','test_metrics_read','gpu_work_run','model_generation_run','downloads_run']))
inputs=pd.read_parquet(cache/'inputs.parquet');rows=pd.read_parquet(cache/'rows.parquet');tokens=read(cache/'token-ids.json');bodies=read(cache/'selected-bodies.json');keys=read(OUT/'document-keys.json')
check('counts-and-contiguous-native-test-identity',len(inputs)==len(tokens)==3360 and len(rows)==1290942 and len(keys)==len(bodies)==528 and np.array_equal(inputs.prompt_ix,np.arange(3360)) and np.array_equal(rows.cache_ix,np.arange(len(rows))) and np.array_equal(rows.sample_ix,9_000_000_000_000+np.arange(len(rows))) and rows.wrapper.eq('native').all() and rows.split.eq('test').all() and rows.source_set.eq('complementary-confirm-c4').all() and rows.semantic_role.eq(rows.role).all())
check('unique-source-namespace',len({k['shard_line'] for k in keys})==len({k['group_id'] for k in keys})==528 and [k['document_id'] for k in keys]==list(range(9_000_000,9_000_528)) and [k['source_index'] for k in keys]==list(range(528)) and [k['source_rank'] for k in keys]==list(range(528)))
check('400general-128full-64paired-not-new-independent-sources',[k['cohort'] for k in keys]==['general']*400+['full-length']*128 and [k['up_to1024_variant'] for k in keys]==[True]*64+[False]*464 and c['independent_source_documents']==528 and rows.groupby('profile').group_id.nunique().to_dict()=={'single':400,'multi':400,'long-single':64,'long-full':128})
check('3360-fixed-profile-inputs',inputs.groupby('profile').size().to_dict()=={'single':2000,'multi':400,'long-single':320,'long-full':640})
# Explicit six historical manifests: actual line, stored full/body/prefix, raw
# and conservative canonical URL disjointness; do not claim an archive replay.
hist_lines=set();hist_hash=set();hist_urls=set();hist_counts={}
for p,entry in sig['manifests'].items():
 bind(p,entry['sha256']);old=read(p);assert len(old)==entry['records'];hist_counts[p]=len(old)
 for k in old:
  hist_lines.add(k['shard_line']);hist_hash.update(k[f] for f in HASH_FIELDS if k.get(f))
  if k.get('source_url'):hist_urls.update([k['source_url'],canonical_url(k['source_url'])])
check('all-six-explicit-manifests-bound',len(hist_counts)==6 and len(hist_lines)==3180 and max(hist_lines)<12303 and c['exclusion_input_sha256']=={p:e['sha256'] for p,e in sig['manifests'].items()})
seen_h=set();seen_u=set()
for k in keys:
 h={k[f] for f in HASH_FIELDS if k.get(f)};u={v for v in [k.get('source_url'),k.get('canonical_source_url')] if v}
 assert k['shard_line'] not in hist_lines and not h&hist_hash and not u&hist_urls and not h&seen_h and not u&seen_u
 assert 12303<=k['shard_line']<=20494 and k['source_revision']=='1588ec454efa1a09f29cd18ddd04fe05fc8653a2' and k['source_shard']=='en/c4-validation.00000-of-00008.json.gz'
 assert k['rank_sha256']==sha('\0'.join(map(str,[20296231,k['source_revision'],k['source_shard'],k['shard_line'],k['full_body_sha256']]))) and k['canonical_source_url']==canonical_url(k['source_url'])
 seen_h.update(h);seen_u.update(u)
check('selected-lines-stored-fingerprints-and-URLs-disjoint',True)
for p,want in sig['original249_sha256'].items():bind(p,want)
check('original249-two-inventory-bytes-bound',len(sig['original249_sha256'])==2)
scan=read(OUT/'scan-audit.json');check('fixed-window-scan-not-score-based',scan['window']==[12303,20494] and scan['window_records_scanned']==8192 and scan['selected']==528 and scan['no_window_extension'] is True and scan['no_model_or_role_scores_used'] is True and scan['classifier_probabilities_read'] is False and scan['test_metrics_read'] is False)
# Use only the pinned tokenizer backend, with no transformer/model import.
tokpath=Path(sig['tokenizer_path']);assert str(tokpath)=='/dev/shm/role-analysis-gptoss120b'
for name,want in sig['tokenizer_sha256'].items():bind(tokpath/name,want)
tok=Tokenizer.from_file(str(tokpath/'tokenizer.json'));special={s:tok.token_to_id(s) for s in ['<|start|>','<|message|>','<|end|>','<|return|>','<|call|>','<|startoftext|>']};assert all(v is not None for v in special.values())
heads={'system':'system','user':'user','assistant<|channel|>analysis':'cot','assistant<|channel|>final':'assistant','functions.lookup to=assistant<|channel|>commentary':'tool','assistant to=functions.lookup<|channel|>commentary':'tool_call'}
def parse(ids):
 out={};frames=[];i=0
 if ids and ids[0]==special['<|startoftext|>']:i=1
 while i<len(ids):
  assert ids[i]==special['<|start|>'];j=i+1
  while j<len(ids) and ids[j]!=special['<|message|>']:j+=1
  assert j<len(ids);header=tok.decode(ids[i+1:j],skip_special_tokens=False);assert header in heads;role=heads[header];a=j+1;b=a
  while b<len(ids) and ids[b] not in [special['<|end|>'],special['<|return|>'],special['<|call|>']]:b+=1
  assert b<len(ids) and b>a and ((ids[b]==special['<|call|>'])==(role=='tool_call'))
  mid=len(frames);frames.append((role,a,b,header))
  if role!='tool_call':out.update({p:(role,mid,p-a) for p in range(a,b)})
  i=b+1
 return out,frames
body_ids=[]
for k,body in zip(keys,bodies):
 ids=tok.encode(body,add_special_tokens=False).ids;assert 96<=len(ids)<=1024 and len(ids)==k['body_tokens']==min(1024,k['original_tokens']) and tok.decode(ids,skip_special_tokens=False)==body and '<|' not in body
 assert sha(body)==k['body_sha256']==k['prefix1024_sha256'] and sha(tok.decode(ids[:192],skip_special_tokens=False))==k['prefix192_sha256']==k['short_body_sha256']
 if k['cohort']=='full-length':assert len(ids)==1024
 body_ids.append(ids)
check('all528-source-body-and-prefix-roundtrip-SHA',True)
structural=0;sys_messages=0;lookup={int(p):g for p,g in rows.groupby('prompt_ix',sort=False)};full_counts=Counter()
for entry in inputs.itertuples():
 ids=tokens[entry.prompt_ix];encoded=tok.encode(entry.prompt,add_special_tokens=False);assert encoded.ids==ids and len(ids)==entry.length
 labels,frames=parse(ids);g=lookup[entry.prompt_ix];pos=g.token_ix.to_numpy(np.int64)
 assert np.array_equal(pos,np.asarray(sorted(labels),np.int64)) and np.array_equal(g.token_id,np.asarray(ids,np.int64)[pos]) and g.cache_ix.iloc[0]==entry.row_start and g.cache_ix.iloc[-1]+1==entry.row_stop
 assert np.array_equal(g.role,[labels[int(p)][0] for p in pos]) and np.array_equal(g.message_ix,[labels[int(p)][1] for p in pos]) and np.array_equal(g.body_offset,[labels[int(p)][2] for p in pos])
 assert g.uid.eq('complementary-confirm-c4:'+str(entry.prompt_ix)).all() and g.group_id.eq(keys[entry.source_index]['group_id']).all() and g.document_id.eq(9_000_000+entry.source_index).all()
 for r in g.itertuples():
  left,right=encoded.offsets[r.token_ix];assert r.token==entry.prompt[left:right]
 sys_messages+=sum(r=='system' and h=='system' for r,a,b,h in frames);structural+=sum(b-a for r,a,b,h in frames if r=='tool_call')
 if entry.profile=='multi':assert [r for r,_,_,_ in frames]==['system','user','cot','assistant','user','tool_call','tool','cot','assistant'] and entry.bos is True
 else:
  assert len(frames)==1 and frames[0][0]==entry.role and entry.bos==bool(entry.source_index%2)
  if entry.profile in ['long-full','long-single']:
   assert ids[frames[0][1]:frames[0][2]]==body_ids[entry.source_index]
  if entry.profile=='long-full':assert len(g)==1024 and np.array_equal(g.body_offset,np.arange(1024));full_counts[(entry.source_index,entry.role)]+=1
check('all3360-tokenization-independent-native-labels-and-row-offsets',True)
check('normal-system-tags-retained',sys_messages==400+400+64+128 and len(rows[rows.role.eq('system')])>0)
check('all128-full-same1024-in-five-roles',len(full_counts)==128*5 and all(v==1 for v in full_counts.values()) and set(r for _,r in full_counts)==set(ROLES))
check('no-header-gap-or-toolcall-content-scored',sum(map(len,tokens))==1336374 and len(rows)==1290942 and structural>0)
roles=rows.groupby(['profile','role']).size().unstack(fill_value=0).to_dict();check('producer-role-count-proof-matches',roles==c['coverage']['role_counts'] and int((rows.profile.eq('long-full')&rows.role.eq('system')&rows.body_offset.ge(512)).sum())==65536)
gallery=read(OUT/'gallery-index.json');ranks=np.linspace(0,399,12,dtype=int).tolist();check('predeclared12general4full-and-positioncurve17plots',[e['cohort_rank'] for e in gallery['general_multi']]==ranks and [e['document_id'] for e in gallery['general_multi']]==[keys[i]['document_id'] for i in ranks] and [e['cohort_rank'] for e in gallery['full_system']]==[0,42,84,127] and [e['document_id'] for e in gallery['full_system']]==[keys[400+i]['document_id'] for i in [0,42,84,127]] and gallery['full_position_curve']['documents']==128 and gallery['selected_before_probabilities'] is True and gallery['score_based_selection'] is False)
for p,want in bindings.items():assert digest(p)==want,p
report=dict(stage='independent_completed_complementary_C4_data_only_audit',audit_passed=True,checks=len(checks),check_names=checks,data_complete_sha256=COMPLETE_SHA,snapshot_file_sha256=SNAPSHOT_SHA,snapshot_canonical_sha256=CANONICAL,inputs=3360,content_rows=1290942,input_tokens=1336374,independent_sources=528,general_sources=400,full_true1024_sources=128,general_paired_long_views=64,full_system_tail_rows=65536,role_counts=roles,unscored_structural_toolcall_body_tokens=structural,explicit_historical_manifest_counts=hist_counts,known_prior_unique_lines=len(hist_lines),historical_stored_fingerprints_checked=len(hist_hash),all_normal_content_token_labels_independently_checked=True,normal_system_tag_retained=True,header_gap_fill_or_structural_toolcall_scored=False,raw_cache_sha256=c['cache_sha256'],provenance_sha256=bindings,limitations='No C4 archive/window reselection or historical raw-source fingerprint recovery replay; line/stored fingerprint/URL exclusions and fixed byte manifests independently checked. Root separate source verify covers full pinned archive recovery. Mtime ordering supports declared-before-ID chronology; frozen hash/source chain is also bound. No semantic/domain independence claim.',test_probabilities_or_metrics_read=False,model_weights_or_prediction_read=False,gpu_used=False,production_prepare_or_verify_called=False,seconds=time.time()-start)
folder=Path(tempfile.mkdtemp(prefix='long-data-complementary-c4-actual-data-',dir='/tmp'));p=folder/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(p),report_sha256=digest(p),script_sha256=digest(__file__),checks=len(checks),inputs=3360,rows=1290942)))
