"""Read-only actual training membership and bounded native-system-tag audit."""
import hashlib,json,re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis');BASE=HERE/'outputs/local-gptoss120b';TRAIN=BASE/'systemness-complementary-head-readout'
ROLES=['system','user','cot','assistant','tool'];CHECKS=[];PROVENANCE={}
def read(p):return json.loads(Path(p).read_text())
def sha(p,want=None):
    p=Path(p);assert not str(p).startswith('/workspace') and p.suffix not in ['.npz','.npy','.f16','.f32'] and '/inference/' not in str(p)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    got=h.hexdigest();assert want is None or got==want,str(p);PROVENANCE[str(p)]=got;return got
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def check(name,x=True):assert x,name;CHECKS.append(name)
def main():
    decl=read(TRAIN/'protocol.json');static=decl['static'];audit=read(TRAIN/'actual-audit/actual-development-audit.json');done=read(TRAIN/'complete.json')
    sha(TRAIN/'actual-audit/actual-development-audit.json','7bac44572d8ea37fc4cae87d447061228496a218eadaa92583a4a96bb3c4989c');sha(TRAIN/'protocol.json',audit['trainer_protocol_sha256']);sha(TRAIN/'complete.json',audit['trainer_complete_sha256']);sha(HERE/'systemness_complementary_head_readout.py',audit['trainer_source_sha256'])
    check('Actual ef92 new640 model and completed no-test training bound',audit['audit_passed'] is True and audit['model_sha256']['new']=='ef92b421007b7d0c7a11f13e91008e6d2e39b34b730c6ca7d6ecbf03590236ff' and audit['test_data_or_results_read'] is False and done['tests_read'] is False)
    op=Path('/dev/shm/role-analysis-top8-heads/rows.parquet');npth=Path('/dev/shm/role-analysis-natural-dev-heads/rows.parquet');ep=Path('/dev/shm/role-analysis-complementary-heads/rows.parquet')
    for p in [op,npth,ep]:sha(p,audit['provenance_sha256'][str(p)])
    columns=['cache_ix','sample_ix','prompt_ix','token_ix','message_ix','role','split','wrapper','profile','source_set','document_id','group_id']
    old=pd.read_parquet(op,columns=columns);raw=old[old.split.eq('fit')];full_docs=set(raw.loc[raw.profile.eq('long-full'),'document_id']);removed=raw[raw.profile.eq('long-single')&raw.document_id.isin(full_docs)];fit=raw[~raw.cache_ix.isin(removed.cache_ix)]
    labels=fit.role.map({r:i for i,r in enumerate(ROLES)}).to_numpy(np.int64)
    check('Exact582250 raw fit minus4160 sparse rows =578090 native fit',len(raw)==582250 and len(fit)==578090 and len(removed)==4160 and removed.document_id.nunique()==26 and len(full_docs)==132 and fit.wrapper.eq('native').all() and fit.split.eq('fit').all())
    check('Old fit cache/sample/label hashes match actually declared training',array_sha(fit.cache_ix.to_numpy())==static['original_partitions']['fit_cache_ix_sha256'] and array_sha(fit.sample_ix.to_numpy())==static['sample_ix_sha256']['fit'] and array_sha(labels)==static['original_partitions']['fit_labels_sha256'])
    natural=pd.read_parquet(npth);nf=natural[natural.split.eq('fit')].copy();check('255clean natural63403 actual-fit membership',len(nf)==63403 and nf.source_uid.nunique()==255 and nf.wrapper.eq('native').all() and nf.eligible_fit.eq(True).all() and not nf.old_generated_collision.any() and not nf.cross_split_assistant_collision.any() and array_sha(nf.sample_ix.to_numpy())==static['sample_ix_sha256']['natural-fit'])
    extra=pd.read_parquet(ep,columns=['cache_ix','parent_bank','parent_cache_ix','sample_ix','role','split'])
    for name,frame in [('old',fit),('natural',nf)]:
        mapping=extra[extra.parent_bank.eq(name)].set_index('parent_cache_ix');part=mapping.loc[frame.cache_ix.to_numpy()]
        assert np.array_equal(part.sample_ix.to_numpy(),frame.sample_ix.to_numpy()) and np.array_equal(part.role.to_numpy(),frame.role.to_numpy()) and part.split.eq('fit').all()
    ec=read(BASE/'systemness-complementary-head-features/extraction-complete.json');check('Original8+new2 feature rows map to identical actualfit labels',ec['added_head_ids']==[247,2248] and len(ec['original_head_ids'])==8 and ec['all_rows_written'] is True and ec['all_finite'] is True and not extra[extra.parent_bank.eq('c4')].split.eq('fit').any())
    sources={
      'main':(BASE/'systemness-expansion/data-complete.json',BASE/'systemness-expansion/cache'),
      'supplement':(BASE/'systemness-expansion/supplement-data-complete.json',BASE/'systemness-expansion/cache/supplement'),
      'long-full':(BASE/'systemness-long-full/data-complete.json',BASE/'systemness-expansion/cache/long-full'),
      'natural-clean-fit':(BASE/'systemness-natural-transcripts/data-complete.json',BASE/'systemness-expansion/cache/natural-transcripts-dev')}
    tokenizer_path=Path('/dev/shm/role-analysis-gptoss120b/tokenizer.json');t=Tokenizer.from_file(str(tokenizer_path));sha(tokenizer_path);sid={s:t.token_to_id(s) for s in ['<|start|>','<|message|>','<|end|>','<|return|>','<|call|>','<|startoftext|>']}
    pattern=re.compile(r'<\|start\|>(.*?)<\|message\|>(.*?)(<\|end\|>|<\|return\|>|<\|call\|>)',re.S)
    cohort={};bounded=0;all_prompt_count=0;all_system_rows=0
    for name,(metapath,cache) in sources.items():
        meta=read(metapath);sha(metapath);sha(cache/'inputs.parquet',meta['cache_sha256']['inputs.parquet'])
        frame=nf.copy() if name=='natural-clean-fit' else fit[fit.source_set.eq(name)].copy()
        if name=='natural-clean-fit':frame['source_prompt_ix']=frame.original_prompt_ix
        else:frame['source_prompt_ix']=frame.prompt_ix
        positive=frame[frame.role.eq('system')];prompt_ids=sorted(positive.source_prompt_ix.unique().tolist())
        table=pd.read_parquet(cache/'inputs.parquet',columns=['prompt_ix','prompt','length','wrapper','split'],filters=[('prompt_ix','in',prompt_ids)])
        assert len(table)==len(prompt_ids) and table.prompt_ix.is_unique and table.wrapper.eq('native').all() and table.split.eq('fit').all();byid={int(r.prompt_ix):r for r in table.itertuples()}
        sampled=set(prompt_ids[i] for i in np.linspace(0,len(prompt_ids)-1,min(16,len(prompt_ids)),dtype=int));covered=0
        for pi,rows in positive.groupby('source_prompt_ix'):
            entry=byid[int(pi)];frames=pattern.findall(entry.prompt);assert frames and entry.prompt.startswith(('<|start|>','<|startoftext|><|start|>'))
            for mi,g in rows.groupby('message_ix'):
                header,body,ending=frames[int(mi)];assert header=='system' and ending=='<|end|>' and body and '<|start|>system<|message|>' in entry.prompt;covered+=len(g)
            if int(pi) in sampled:
                ids=t.encode(entry.prompt,add_special_tokens=False).ids;assert len(ids)==entry.length;pos=0;mi=-1;system_positions=set()
                while pos<len(ids):
                    if ids[pos]!=sid['<|start|>']:pos+=1;continue
                    mi+=1;pos+=1;header=[]
                    while ids[pos]!=sid['<|message|>']:header.append(ids[pos]);pos+=1
                    role=t.decode(header,skip_special_tokens=False);pos+=1
                    while ids[pos] not in [sid['<|end|>'],sid['<|return|>'],sid['<|call|>']]:
                        if role=='system':system_positions.add((pos,mi))
                        pos+=1
                assert all((int(r.token_ix),int(r.message_ix)) in system_positions for r in rows.itertuples());bounded+=1
        assert covered==len(positive)
        cohort[name]=dict(training_rows=len(frame),system_positive_rows=len(positive),system_bearing_fit_inputs=len(prompt_ids),all_positive_input_headers_checked=True,bounded_tokenizer_reencoded_fit_inputs=len(sampled),with_BOS=sum(v.prompt.startswith('<|startoftext|>') for v in byid.values()),without_BOS=sum(not v.prompt.startswith('<|startoftext|>') for v in byid.values()),role_support=frame.role.value_counts().to_dict(),input_cache_path=str(cache/'inputs.parquet'))
        all_prompt_count+=len(prompt_ids);all_system_rows+=len(positive)
    check('All2961 actualFIT system-bearing prompts use normal Harmony system header',all_prompt_count==2961 and all_system_rows==122740)
    check('64 bounded FIT prompt tokenizers confirm positive token/message alignment',bounded==64)
    budget=audit['training_budgets']['new'];check('Actual training641493 instances157updates last2517, not newC4fit',len(fit)+len(nf)==budget['instances_per_epoch']==641493 and budget['steps_per_epoch']==157 and budget['last_batch_instances']==2517 and decl['protocol']['arms']==['new','matched'])
    for path,want in list(PROVENANCE.items()):sha(path,want)
    report=dict(stage='actual_training_native_system_tag_data_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=sha(__file__),checks=CHECKS,checks_passed=len(CHECKS),model_sha256=audit['model_sha256']['new'],old_fit_rows=578090,clean_natural_fit_rows=63403,natural_fit_sources=255,instances_per_epoch=641493,removed_sparse_rows=4160,removed_sparse_documents=26,system_positive_training_rows=all_system_rows,normal_system_bearing_fit_prompts=all_prompt_count,bounded_tokenizer_reencoded_fit_prompts=bounded,cohorts=cohort,actual_training_budget=budget,provenance_sha256=PROVENANCE,producer_called=False,model_weights_or_features_loaded=False,probability_files_or_fresh_results_read=False,gpu_used=False,scope='All actual-fit system-positive source prompt strings checked for normal native system headers; only64 bounded FIT prompts fully re-encoded, no full LM replay.')
    output=Path('/tmp/confirmation_review_training_system_tags.json');output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report_path=str(output),report_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),script_sha256=report['auditor_source_sha256'],system_positive_rows=all_system_rows,system_bearing_fit_prompts=all_prompt_count,bounded_reencoded=bounded)),flush=True)
if __name__=='__main__':main()
