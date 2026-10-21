"""Independent CPU boundary annex; no real unseen user/reply catalog."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, '/root/prompt-injection-as-role-confusion/experiments/role-analysis')
import systemness_aug_natural_data as m

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source=Path(m.__file__); before=digest(source)
checks=[]
def check(name, condition):
    assert condition, name
    checks.append(name)
def rejects(name, call):
    try: call()
    except (AssertionError, ValueError, RuntimeError, FileNotFoundError): checks.append(name)
    else: raise AssertionError(name)

folder=Path(tempfile.mkdtemp(prefix='root-aug-natural-builder-annex-'))
raw=folder/'users.csv'
raw.write_bytes(b'conv_id,user_input,model_output,extra\r\n"id,1","user\r\nquestion with ""quotes""",' +
    b'"\xff\xfe never decode assistant\n""quoted""",ok\r\nsecond,question2,\xff,ok\r\n')
users,audit=m.user_csv(raw)
check('Quoted framing ignores invalid UTF8 assistant fields', users=={
    'id,1': {'conv_id':'id,1','user_input':'user\r\nquestion with "quotes"'},
    'second': {'conv_id':'second','user_input':'question2'}} and
    audit['assistant_fields_decoded']==audit['assistant_fields_buffered']==0)
raw.write_bytes(b'conv_id,user_input,model_output\nsame,u,a\nsame,v,b\n')
rejects('Duplicate source IDs fail', lambda:m.user_csv(raw))
with m.patched(OUT=folder/'undeclared-out', CACHE=folder/'undeclared-cache'):
    rejects('Prepare requires explicit earlier declaration',m.prepare)
    rejects('User catalog cannot run before source declaration',lambda:m.guarded_catalog({}))

tokenizer=m.native.load_tokenizer()
templates,proof=m.build_templates(tokenizer,m.old_template_catalog())
check('New wordings retain native system role and independent wording count32',
    len(templates)==32 and all(t['role']=='system' and 0<t['token_count']<=192 for t in templates)
    and proof['exact_normalized_sanitized_prefix_url_overlap']==0)
records=[]
for i in range(128):
    template=templates[i%32]
    reply='Synthetic independent boundary reply '+str(i)+' with distinct content.'
    if i in [0,1]:reply='A deliberate shared reply links two instruction families.'
    records.append(dict(source_uid='annex/'+str(i),dataset='oasst' if i<64 else 'toxicchat0124',
        native_id='annex-'+str(i),split='test',revision='synthetic',root_raw_sha256='synthetic',
        template_id=template['template_id'],family_id=template['family_id'],document_id=m.DOCUMENT_START+i,
        failure='annex-quality-failure' if i==127 else None,messages=[
            dict(role='system',text=template['text'],source_message_id='s',origin='fixture',source_model=None),
            dict(role='user',text='Independent fabricated question '+str(i),source_message_id='u',origin='fixture',source_model=None),
            dict(role='assistant',text=reply,source_message_id='a',origin='fixture',source_model='synthetic')]))
def keys(text):
    body,_,clean=m.pure.capped_body(text,tokenizer)
    return m.pure.assistant_keys(text,clean,body,tokenizer)
old={key:{'legacy-annex'} for key in keys(records[2]['messages'][2]['text'])}
reference=[dict(message_uid='historical-annex/a',source_uid='historical-annex',split='fit',message_ix=2,
    keys=keys(records[3]['messages'][2]['text']))]
inputs,rows,tokens,ledger,duplicates,coverage=m.build_dataset(records,tokenizer,reference,old,{'synthetic':True})
m.verify_cached_content(inputs,rows,tokens,ledger,tokenizer)
m.verify_summary(inputs,rows,ledger,duplicates,coverage)
check('Quality and contamination never replace frozen sources',len(ledger)==128 and len(inputs)==127 and
    not ledger[2]['eligible_clean_test'] and not ledger[3]['eligible_clean_test'] and
    sum(r['eligible_clean_test'] for r in ledger)==125 and ledger[127]['status']=='quality-failed')
check('Family and repeated-assistant bridges precede clean subset filtering',
    duplicates['dependency_components']==31 and
    ledger[0]['dependency_component_id']==ledger[1]['dependency_component_id'] and
    ledger[2]['dependency_component_id']==ledger[34]['dependency_component_id'] and
    duplicates['system_family_counts']==dict(fixed=32,all_prepared=32,clean_prepared=32))
check('Every scored position is native S/U/A content',set(rows.role)=={'system','user','assistant'} and
    all(len({r.token_ix for r in rows[rows.prompt_ix==i].itertuples()})==
        sum(message['body_tokens'] for message in ledger[i]['messages']) for i in range(127)) and
    coverage['roles_with_no_observations']['cot']['recall'] is None)
failed=copy.deepcopy(records)
for record in failed:record['failure']='all-failed-boundary'
empty=m.build_dataset(failed,tokenizer,[],{}, {'synthetic':True})
m.verify_cached_content(*empty[:3],empty[3],tokenizer)
m.verify_summary(empty[0],empty[1],empty[3],empty[4],empty[5])
check('All-failed cohort preserves128 ledger and zero observations/components',
    len(empty[3])==128 and not len(empty[0]) and not len(empty[1]) and
    empty[4]['dependency_components']==0 and empty[5]['no_observations'])
check('Reviewed source unchanged during annex',digest(source)==before)
report=dict(stage='independent_synthetic_natural_builder_annex',source_sha256=before,
    checks=checks,checks_passed=len(checks),source_rows_read=False,real_new_ids_selected=False,
    real_unseen_user_or_reply_text_read=False,gpu_work_run=False,formal_declaration_run=False,
    actual_test_results_audited=False,fixture_directory=str(folder))
path=Path('/tmp/root_aug_natural_builder_annex.json')
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report_path':str(path),'report_sha256':digest(path),'checks_passed':len(checks)}))
