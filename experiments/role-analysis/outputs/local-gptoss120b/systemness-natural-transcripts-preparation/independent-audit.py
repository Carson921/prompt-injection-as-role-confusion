import csv,hashlib,io,json,random,sys,tempfile
from pathlib import Path
import numpy as np
root=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
sys.path.insert(0,str(root))
import systemness_natural_transcripts as m
source_before=m.data.digest(m.__file__)
assert source_before=='e2337005e9ffb3fc9bf400df6cbf4357621feeb5dab3b7b20cc397301d4775cc'
assert not m.OUT.exists() and not m.CACHE.exists()
folder=Path(tempfile.mkdtemp(prefix='root-natural-transcripts-audit-',dir='/tmp'))
checks=[]
rng=random.Random(617521)
records=[]
for i in range(320):
    body=''.join(rng.choices(['a','b','中','😀',',','"','\n','\r',' '],k=80+(i%7)))
    if i in [7,99,121]: body += '\n"边界"😀,'*22000
    records.append([f'root-{i}',f'Question {i}, "quoted"\r\nline',body,'extra\nfield'])
wanted={f'root-{i}' for i in range(0,320,17)} | {'root-7','root-99'}
records[121][2]='NONSELECTED_INVALID_UTF8_SENTINEL'
s=io.StringIO(newline='');w=csv.writer(s,lineterminator='\r\n')
w.writerow(['conv_id','user_input','model_output','other']);w.writerows(records)
raw=s.getvalue().encode().replace(b'NONSELECTED_INVALID_UTF8_SENTINEL',b'\xff\xfe\x80')
p=folder/'quoted.csv';p.write_bytes(raw)
selected,audit=m.selected_csv(p,wanted)
expected={r[0]:dict(conv_id=r[0],user_input=r[1],model_output=r[2]) for r in records if r[0] in wanted}
assert selected==expected and audit['source_records_framed']==320
assert audit['assistant_fields_decoded']==len(wanted) and audit['nonselected_assistant_fields_buffered']==0
checks.append('independent_csv_writer320_unicode_quoted_CRLF_and64KiB_boundary_records_match_exactly')
try:m.selected_csv(p,{'root-121'})
except UnicodeDecodeError:pass
else:raise AssertionError('Selected invalid UTF8 accepted')
checks.append('nonselected_invalid_UTF8_isolation_positive_and_negative_controls')
def node(i,parent,role,text='Source content.',**kw):
    return dict(message_id=i,parent_id=parent,role=role,text=text,lang=kw.get('lang','en'),deleted=kw.get('deleted',False),synthetic=kw.get('synthetic',False),model_name=None)
key=lambda p,c:hashlib.sha256((p+'\0'+c).encode()).hexdigest()
a,b=sorted(['reply-a','reply-b'],key=lambda c:key('root',c))
u,v=sorted(['follow-u','follow-v'],key=lambda c:key(a,c))
tree=[node('root',None,'prompter'),node(a,'root','assistant'),node(b,'root','assistant'),node(u,a,'prompter'),node(v,a,'prompter'),node('end-u',u,'assistant'),node('end-v',v,'assistant'),node('bad',a,'prompter',deleted=True),node('bad-reply','bad','assistant')]
for _ in range(12):
    rng.shuffle(tree);chosen,failure=m.choose_oa_path(tree,'root')
    assert failure is None and [x['message_id'] for x in chosen]==['root',a,u,'end-u']
checks.append('independent_SHA_path_expected_four_nodes_invariant_to12_row_permutations')
tok=m.long_data.load_tokenizer()
text=('中文😀 exact body ' + ' '.join(m.long_data.RESERVED))*100
body,meta,clean=m.capped_body(text,tok)
assert meta['raw_sha256']==hashlib.sha256(text.encode()).hexdigest() and meta['body_tokens']<=192
assert tok.decode(tok.encode(body,add_special_tokens=False),skip_special_tokens=False)==body
assert not any(marker in body for marker in m.long_data.RESERVED)
prompt,spans=m.render_native([('system','Ordinary guidance.'),('user',body),('assistant','Actual recorded reply.')])
ids,rows=m.content_rows(prompt,spans,tok)
assert len(rows)==sum(len(tok.encode(b,add_special_tokens=False)) for r,b in [('system','Ordinary guidance.'),('user',body),('assistant','Actual recorded reply.')])
assert set(r['role'] for r in rows)=={'system','user','assistant'} and '<|channel|>analysis' not in prompt
checks.append('Unicode_sanitized192token_roundtrip_and_complete_normal_SUA_coverage')
def record(uid,split,reply,family='family',failed=False):
    return dict(source_uid=uid,dataset='synthetic',native_id=uid,split=split,template_id=family,family_id=family,failure='fixed-quality-failure' if failed else None,messages=[dict(role='system',text='Normal system guidance.'),dict(role='user',text='Actual user input '+uid),dict(role='assistant',text=reply)])
recs=[record('fit-dup','fit','Repeated recorded reply.'),record('fit-clean','fit','Unique new reply.'),record('fit-old','fit','An old generated body.'),record('fit-failed','fit','',failed=True),record('val-a','validation','Repeated recorded reply.','family-a'),record('val-b','validation','Repeated recorded reply.','family-b')]
old={('full',hashlib.sha256(b'An old generated body.').hexdigest()):{'old-reference'}}
inputs,rows,tokens,ledger,dups,coverage=m.build_dataset(recs,tok,old,{'synthetic_only':True})
byid={r['source_uid']:r for r in ledger}
assert len(ledger)==6 and len(inputs)==5 and byid['fit-clean']['eligible_fit']
assert not byid['fit-dup']['eligible_fit'] and byid['fit-dup']['cross_split_assistant_collision']
assert not byid['fit-old']['eligible_fit'] and byid['fit-old']['old_generated_collision']
assert not byid['fit-failed']['eligible_fit'] and byid['fit-failed']['status']=='quality-failed'
assert dups['conversation_dependency_components']['validation']['dependency_components']==1
assert dups['conversation_dependency_components']['validation']['distinct_system_families']==2
assert coverage['roles_with_no_observations']['cot']['recall'] is None and coverage['roles_with_no_observations']['tool']['recall'] is None
m.verify_cached_content(inputs,rows,tokens,ledger,tok)
checks.append('independent_expected_duplicate_eligibility_fixed_failure_ledger_and_connected_validation_clusters')
source_after=m.data.digest(m.__file__);assert source_after==source_before
assert not m.OUT.exists() and not m.CACHE.exists()
result=dict(source_sha256_at_start=source_before,source_sha256_at_completion=source_after,checks=checks,synthetic_only=True,source_reply_text_read=False,validation_reply_text_read=False,test_reply_text_read=False,formal_OUT_created=False,gpu_run=False,downloads_run=False,synthetic_CSV_bytes=len(raw),selected_CSV_records=len(wanted),synthetic_transcripts=len(inputs))
(folder/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(report=str(folder/'report.json'),**result)))
