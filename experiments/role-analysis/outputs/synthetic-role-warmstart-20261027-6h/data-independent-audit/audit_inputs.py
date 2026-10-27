import argparse,collections,hashlib,json,time,re,unicodedata
from urllib.parse import urlsplit,urlunsplit
from pathlib import Path
from transformers import AutoTokenizer

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def require(x,m):
 if not x:raise AssertionError(m)

p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--tokenizer-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-prior-context-cap',action='store_true');p.add_argument('--old-data-dir',type=Path);p.add_argument('--historical-paths',type=Path);p.add_argument('--conservative',action='store_true');a=p.parse_args()
t0=time.time();tok=AutoTokenizer.from_pretrained(a.tokenizer_dir,local_files_only=True)
controls={s:tok.convert_tokens_to_ids(s) for s in ['<|start|>','<|message|>','<|end|>','<|return|>','<|call|>']}
require(len(set(controls.values()))==5,'control ID collision')
roles=['system','developer','user','cot','assistant','tool']
header_roles={'system':'system','developer':'developer','user':'user','assistant<|channel|>analysis':'cot','assistant<|channel|>final':'assistant','functions.lookup':'tool','assistant to=functions.lookup<|channel|>commentary':None}
source_path=a.data_dir/('sources-conservative.jsonl' if a.conservative else 'sources.jsonl')
docs={d['source_id']:d for d in map(json.loads,source_path.read_text().splitlines())}
article_ids={sid:tok.encode(d['text'],add_special_tokens=False) for sid,d in docs.items()}
clusters={};paired=collections.defaultdict(list);counts=collections.Counter();lengths=collections.Counter();max_input={};records=0;raw_targets=0;cap_violations=[];digests={};templates=collections.Counter();wrapper=collections.Counter();boundaries=0
for split in ['fit','dev','test']:
 path=a.data_dir/('inputs-fit-conservative.jsonl' if a.conservative and split=='fit' else f'inputs-{split}.jsonl')
 if not path.exists():continue
 digests[path.name]=sha(path);max_input[split]=0
 for line in path.open():
  r=json.loads(line);records+=1;sid=r['source_id'];doc=docs[sid]
  require(r['split']==split==doc['split'],'source split mismatch')
  require(r['cluster_id']==doc['cluster_id'],'cluster ID differs')
  require(r['cluster_id'] not in clusters or clusters[r['cluster_id']]==split,'cluster crosses splits')
  clusters[r['cluster_id']]=split
  enc=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];off=enc['offset_mapping']
  require(ids==r['input_ids'],'saved input IDs differ')
  require(0<len(ids)<=4096,'input cap violation')
  max_input[split]=max(max_input[split],len(ids))
  frames=[];i=0
  while i<len(ids):
   require(ids[i]==controls['<|start|>'],'non-native prefix')
   j=i+1
   while j<len(ids) and ids[j]!=controls['<|message|>']:j+=1
   require(j<len(ids),'missing message boundary')
   header=tok.decode(ids[i+1:j],skip_special_tokens=False)
   require(header in header_roles,'unknown native header')
   k=j+1
   while k<len(ids) and ids[k] not in [controls['<|end|>'],controls['<|return|>'],controls['<|call|>']]:k+=1
   require(k<len(ids),'unterminated frame')
   frames.append((j+1,k,header_roles[header],header))
   i=k+1
  if r['profile']=='context':
   require([x[2] for x in frames[:2]]==['system','developer'],'illegal SD prefix')
   require(sum(x[2]=='system' for x in frames)==1 and sum(x[2]=='developer' for x in frames)==1,'duplicated SD')
  spans={s['event_id']:s for s in r['message_spans'] if s['supervised']}
  for span in spans.values():
   prov=span['article_provenance'];start=prov['article_token_start'];n=prov['article_token_length']
   body=r['text'][span['body_start']:span['body_end']]
   require(body==tok.decode(article_ids[sid][start:start+n],skip_special_tokens=False),'article provenance mismatch')
   require(hashlib.sha256(body.encode()).hexdigest()==prov['body_sha256'],'article span hash differs')
  pure_indices={ev:[z for z,(l,h) in enumerate(off) if h>l and sp['body_start']<=l<h<=sp['body_end']] for ev,sp in spans.items()}
  pure_positions={ev:{ix:i for i,ix in enumerate(val)} for ev,val in pure_indices.items()}
  seen=set();by_role=collections.Counter();by_event=collections.Counter();positions=[]
  for t in r['targets']:
   ix=t['token_ix'];require(ix not in seen,'duplicate target');seen.add(ix)
   require(0<=ix<len(ids),'target out of bounds')
   frame=[f for f in frames if f[0]<=ix<f[1]];require(len(frame)==1,'not native body token')
   require(frame[0][2]==t['role']==roles[t['label']],'role label differs from token ID header')
   s=spans[t['event_id']];left,right=off[ix]
   require(s['body_start']<=left<right<=s['body_end'],'non-article target')
   require(t['body_token_id']==ids[ix],'target token differs')
   require(t['body_char_start']==left-s['body_start'] and t['body_char_end']==right-s['body_start'],'target offset differs')
   pos=pure_positions[t['event_id']][ix];n=len(pure_indices[t['event_id']])
   require(t['body_position']==pos and t['body_tokens']==n,'body position differs')
   require(t['body_quartile']==min(3,pos*4//n),'body quartile differs')
   require(t['profile']==r['profile'] and t['wrapper_type']==r['wrapper_type'],'group metadata differs')
   by_role[t['role']]+=1;by_event[t['event_id']]+=1;raw_targets+=1
   counts[(split,r['profile'],t['role'])]+=1
   positions.append((t['body_char_start'],t['body_char_end'],t['body_token_id'],t['paired_alignment_key']))
  require(all(n<=96 for n in by_event.values()),'per-message cap violation')
  if any(n>96 for n in by_role.values()):cap_violations.append(r['id'])
  if r['profile']=='paired':paired[sid].append((r['template_id'],r['text'][r['message_spans'][0]['body_start']:r['message_spans'][0]['body_end']],set(positions)))
  templates[(split,r['profile'],r['template_id'])]+=1;wrapper[(split,r['wrapper_type'])]+=1
  lengths[(split,doc['source'],doc['paired_length'])]+=int(r['profile']=='paired')
  boundaries+=len(r['boundary_tokens'])
for sid,ps in paired.items():
 require(len(ps)==6 and {z[0] for z in ps}==set('SDUCAT'),'incomplete paired family')
 require(len({z[1] for z in ps})==1,'paired content differs')
 require(all(z[2]==ps[0][2] for z in ps),'paired target positions differ')
if not a.allow_prior_context_cap:require(not cap_violations,'per-role cap violation')
def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def canonical_url(url):
 u=urlsplit(url);require(u.scheme in ('http','https') and u.netloc,'invalid source URL')
 return urlunsplit(('https',u.netloc.lower().removeprefix('www.'),re.sub('/+','/',u.path).rstrip('/'),'',''))
def fingerprint(text):
 words=re.findall('[a-z0-9]+',norm(text));frequencies=collections.Counter(' '.join(words[i:i+5]) for i in range(max(1,len(words)-4)))
 totals=[0]*64
 for shingle,count in frequencies.items():
  n=int.from_bytes(hashlib.blake2b(shingle.encode(),digest_size=8).digest(),'little')
  for bit in range(64):totals[bit]+=count if (n>>bit)&1 else -count
 return sum((n>=0)<<bit for bit,n in enumerate(totals))
source_hashes={};source_urls={};source_prints=[]
for d in docs.values():
 raw=hashlib.sha256(d['text'].encode()).hexdigest();normalized=hashlib.sha256(norm(d['text']).encode()).hexdigest();url=canonical_url(d['normalized_url']);fp=fingerprint(d['text'])
 require(raw==d['text_sha256'],'full source raw SHA differs')
 require(normalized==d['normalized_sha256']==d['cluster_id'],'source normalized SHA or cluster differs')
 require(fp==int(d['simhash64'],16),'source SimHash differs')
 require(normalized not in source_hashes and url not in source_urls,'duplicate normalized text or canonical URL')
 source_hashes[normalized]=d['source_id'];source_urls[url]=d['source_id'];source_prints.append((d['source_id'],fp))
for i,(sid,fp) in enumerate(source_prints):
 for other,op in source_prints[:i]:require((fp^op).bit_count()>3,'selected source SimHash radius3 collision')
history=[];history_files={}
if a.old_data_dir:
 path=a.old_data_dir/'sources.jsonl';history.extend(map(json.loads,path.open()));history_files[str(path)]={'sha256':sha(path),'records':len(history)}
if a.historical_paths:
 for filename in json.loads(a.historical_paths.read_text()):
  path=Path(filename);value=json.loads(path.read_text())
  if isinstance(value,dict):value=value.get('sources',value.get('documents',value.get('keys',[])))
  require(isinstance(value,list),'unsupported historical source keys')
  history.extend(value);history_files[str(path)]={'sha256':sha(path),'records':len(value)}
history_hashes=set();history_urls=set();history_prints=[]
for d in history:
 for key in ['text_sha256','full_text_sha256','full_body_sha256','body_sha256','short_body_sha256','prefix192_sha256','prefix1024_sha256','truncated_body_sha256','normalized_sha256']:
  if d.get(key):history_hashes.add(d[key])
 for key in ['normalized_url','canonical_source_url','source_url','url']:
  if d.get(key):history_urls.add(canonical_url(d[key]))
 if d.get('simhash64'):history_prints.append(int(d['simhash64'],16))
fresh_test=[d for d in docs.values() if d['split']=='test']
for d in fresh_test:
 require(d['text_sha256'] not in history_hashes and d['normalized_sha256'] not in history_hashes and hashlib.sha256(d['text'].strip().encode()).hexdigest() not in history_hashes,'TEST exact source hash overlaps history')
 require(canonical_url(d['normalized_url']) not in history_urls,'TEST source URL overlaps history')
 require(all((int(d['simhash64'],16)^n).bit_count()>3 for n in history_prints),'TEST near-duplicate overlaps history with stored SimHash')
source_audit={'status':'passed','sources':len(docs),'full_text_sha_verified':len(docs),'normalized_text_and_cluster_sha_verified':len(docs),'simhash_recomputed':len(docs),'unique_canonical_urls':len(source_urls),'radius3_pairs_checked':len(docs)*(len(docs)-1)//2,'fresh_test_sources':len(fresh_test),'historical_source_records':len(history),'historical_unique_stored_hashes':len(history_hashes),'historical_unique_urls':len(history_urls),'historical_stored_simhashes':len(history_prints),'historical_key_files':history_files,'limitation':'Radius3 historical check covers only records carrying a stored SimHash; other historical keys permit exact raw/normalized hashes and URL checks only. No semantic paraphrase exclusion claim.'}
report=dict(status='passed',data_dir=str(a.data_dir),records=records,targets=raw_targets,paired_sources=len(paired),source_clusters=len(clusters),max_input_tokens=max_input,targets_by_split_profile_role={'/'.join(k):v for k,v in sorted(counts.items())},records_by_template={'/'.join(k):v for k,v in sorted(templates.items())},records_by_wrapper={'/'.join(k):v for k,v in sorted(wrapper.items())},paired_records_by_corpus_allocated_length={'/'.join(map(str,k)):v for k,v in sorted(lengths.items())},boundary_tokens_excluded=boundaries,prior_context_role_cap_violations=len(cap_violations),prior_context_cap_exception=a.allow_prior_context_cap,input_sha256=digests,script_sha256=sha(__file__),source_partition_audit=source_audit,conservative=a.conservative,elapsed_seconds=time.time()-t0,checks=['official tokenizer retokenization','independent token-ID native-header role labels','article source token-slice equality','whole-token article-only masks','relative body positions and quartiles','six-role exact paired content and positions','cluster partition disjointness','complete input cap4096'])
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['targets_by_split_profile_role','records_by_template','records_by_wrapper','paired_records_by_corpus_allocated_length','input_sha256','checks']}))
