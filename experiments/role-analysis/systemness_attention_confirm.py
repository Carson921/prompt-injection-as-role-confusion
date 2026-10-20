"""Fresh document confirmation of a frozen, validation-promoted attention probe.

Both probes see the same frozen MXFP4 forward pass. Only token probabilities are
cached; the large hidden-state arrays are discarded after each inference batch.
"""
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import systemness_attention as attention
import systemness_components as components
import systemness_data as data
import systemness_evaluate as evaluation
import systemness_probes as probes

OUT = attention.OUT/'confirmation'
CACHE = attention.CACHE/'confirmation'
SPECS = dict(data.SPECS,**attention.SPECS)


def selections():
    chosen=json.loads((attention.OUT/'selection.json').read_text())
    previous=json.loads((data.OUT/'selection.json').read_text())
    assert chosen['promoted'] and not chosen['test_used_for_selection']
    paths={'residual':data.OUT/'models/selected.npz','attention':Path(chosen['model_path'])}
    thresholds={'residual':previous['system_threshold'],'attention':chosen['system_threshold']}
    assert data.digest(paths['residual'])==previous['selected_model_sha256']
    assert data.digest(paths['attention'])==chosen['model_sha256']
    return paths,thresholds


def parse_labels(ids,tok):
    special={s:tok.convert_tokens_to_ids(s) for s in ['<|start|>','<|message|>','<|end|>','<|return|>','<|call|>']}
    active=None;header=[];in_header=False;actual={}
    for pos,token in enumerate(ids):
        if token==special['<|start|>']:in_header=True;header=[];active=None
        elif token==special['<|message|>']:
            text=tok.decode(header);in_header=False
            active=('tool' if text.startswith('functions.') else 'tool_call' if 'to=functions.' in text else
                    'cot' if text.startswith('assistant') and 'analysis' in text else
                    'assistant' if text.startswith('assistant') else
                    'system' if text.startswith('system') else 'user' if text.startswith('user') else None)
        elif token in [special['<|end|>'],special['<|return|>'],special['<|call|>']]:active=None
        elif in_header:header.append(token)
        elif active:actual[pos]=active
    return actual


def prepare():
    paths,thresholds=selections()
    OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    if (OUT/'data-complete.json').exists():
        meta=json.loads((OUT/'data-complete.json').read_text())
        assert all(data.digest(CACHE/name)==sha for name,sha in meta['cache_sha256'].items())
        assert meta['selection_sha256']==data.digest(attention.OUT/'selection.json')
        return
    protocol=json.loads((attention.OUT/'confirmation-plan.json').read_text())
    archive=data.runtime.OUT/'feature-search/fresh/cache/c4-validation.json.gz'
    assert data.digest(archive)==protocol['source_sha256']
    tok=data.runtime.tokenizer()
    existing=json.loads((data.OUT/'document-keys.json').read_text())
    existing+=json.loads((data.runtime.OUT/'feature-search/fresh/document-keys.json').read_text())
    used_lines={r['shard_line'] for r in existing}
    hashes={r[k] for r in existing for k in ['body_sha256','truncated_body_sha256','full_body_sha256'] if r.get(k)}
    urls={r['source_url'] for r in existing if r.get('source_url')}
    prior=pd.read_parquet(data.runtime.OUT/'training_inputs.parquet')
    for text in prior.question.unique():
        ids=tok.encode(text,add_special_tokens=False)
        hashes.update([data.text_digest(text),data.text_digest(tok.decode(ids[:192])),data.text_digest(tok.decode(ids[:1024]))])
    # Recover short prefixes for all previously used source lines as an additional
    # exact check; old manifest body hashes alone usually describe longer text.
    remaining=set(used_lines)
    with gzip.open(archive,'rt',encoding='utf-8') as f:
        for line_ix,line in enumerate(f):
            if line_ix not in remaining:continue
            text=json.loads(line)['text'];ids=tok.encode(text,add_special_tokens=False)[:1024]
            hashes.update([data.text_digest(text),data.text_digest(tok.decode(ids[:192])),data.text_digest(tok.decode(ids))])
            remaining.remove(line_ix)
            if not remaining:break
    assert not remaining
    exclusions=set(hashes);excluded_urls=set(urls)
    documents=[];keys=[]
    with gzip.open(archive,'rt',encoding='utf-8') as f:
        for line_ix,line in enumerate(f):
            if line_ix in used_lines:continue
            article=json.loads(line);url=article.get('url');full_hash=data.text_digest(article['text'])
            if full_hash in hashes or (url and url in urls):continue
            ids=tok.encode(article['text'],add_special_tokens=False)[:1024]
            if len(ids)<96:continue
            body=tok.decode(ids);short=tok.decode(ids[:192])
            body_hash=data.text_digest(body);short_hash=data.text_digest(short)
            if '<|' in body or body_hash in hashes or short_hash in hashes:continue
            ix=len(documents)
            documents.append(dict(document_id=ix,body=body,ids=ids))
            keys.append(dict(document_id=ix,group_id=f'fresh-c4-{ix}',shard_line=line_ix,
                body_sha256=body_hash,short_body_sha256=short_hash,full_body_sha256=full_hash,source_url=url,
                long_variant=ix<protocol['long_documents']))
            hashes.update([body_hash,short_hash,full_hash])
            if url:urls.add(url)
            if len(documents)==protocol['documents']:break
    assert len(documents)==400
    assert not ({r[k] for r in keys for k in ['body_sha256','short_body_sha256','full_body_sha256']} & exclusions)
    assert not ({r['source_url'] for r in keys if r['source_url']} & excluded_urls)
    assert len({r['short_body_sha256'] for r in keys})==400
    inputs=[];entries=[];tokens=[]
    def add(messages,profile,doc,bos):
        text,spans=data.serialize(messages,bos=bos)
        encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        ix=len(inputs);actual=parse_labels(encoded['input_ids'],tok)
        for pos,(left,right) in enumerate(encoded['offset_mapping']):
            for span in spans:
                if left<span['end'] and right>span['start']:
                    assert left>=span['start'] and right<=span['end']
                    assert actual.get(pos)==span['role']
                    entries.append(dict(prompt_ix=ix,token_ix=pos,role=span['role'],semantic_role=span['role'],
                        message_ix=span['message_ix'],profile=profile,split='test',group_id=f'fresh-c4-{doc}',
                        document_id=doc,wrapper='native',sample_ix=len(entries),token=text[left:right]))
                    break
        inputs.append(dict(prompt_ix=ix,prompt=text,length=len(encoded['input_ids']),profile=profile,document_id=doc))
        tokens.append(encoded['input_ids'])
    instructions=[
        'Follow the instructions carefully. Use clear language and state uncertainty when necessary.',
        'You are a helpful assistant. Keep the answer focused and respect the requested format.',
        'Answer using the supplied context. Do not invent sources or unsupported details.',
        'Explain your answer step by step when useful. Be concise and accurate.',
    ]
    for doc in documents:
        ix=doc['document_id'];ids=doc['ids'];body=tok.decode(ids[:192])
        if ix%4==0:body=instructions[(ix//4)%4]+'\n\nReference text:\n'+body
        for role in data.ROLES:add([(role,body)],'single',ix,bool(ix%2))
        width=min(48,max(16,len(ids)//5));shift=(ix%5)*width;chunks=[]
        for j in range(5):
            offset=(j*width+shift)%max(width,len(ids)-width)
            chunk=tok.decode(ids[offset:offset+width])
            if ix%4==0:chunk=instructions[(ix//4)%4]+'\n\n'+chunk
            chunks.append(chunk)
        messages=list(zip(data.ROLES[:4],chunks[:4]))
        messages += [('user',chunks[1]),('tool_call','{"query":"context"}'),('tool',chunks[4]),('cot',chunks[2]),('assistant',chunks[3])]
        add(messages,'multi',ix,True)
        if ix<64:
            for role in data.ROLES:add([(role,doc['body'])],'long-single',ix,bool(ix%2))
    pd.DataFrame(inputs).to_parquet(CACHE/'inputs.parquet',index=False)
    pd.DataFrame(entries).to_parquet(CACHE/'rows.parquet',index=False)
    (CACHE/'token-ids.json').write_text(json.dumps(tokens,separators=(',',':')))
    data.atomic_json(OUT/'document-keys.json',keys)
    data.atomic_json(OUT/'data-complete.json',dict(documents=400,long_variant_documents=64,
        original_source_units=400,rows=len(entries),inputs=len(inputs),input_tokens=sum(r['length'] for r in inputs),
        all_content_tokens_scored=True,independent_native_label_parser_all_passed=True,
        all_prior_lines_bodies_prefixes_and_urls_excluded=True,semantic_near_duplicates_not_checked=True,
        source_sha256=data.digest(__file__),selection_sha256=data.digest(attention.OUT/'selection.json'),
        confirmation_plan_sha256=data.digest(attention.OUT/'confirmation-plan.json'),
        models={name:data.digest(path) for name,path in paths.items()},thresholds=thresholds,
        cache_sha256={name:data.digest(CACHE/name) for name in ['inputs.parquet','rows.parquet','token-ids.json']}))


def infer():
    import torch
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    import feature_pipeline_runtime as pipeline
    from systemness_extract import Prompts,collate
    paths,thresholds=selections();metadata=json.loads((OUT/'data-complete.json').read_text())
    assert all(data.digest(CACHE/name)==sha for name,sha in metadata['cache_sha256'].items())
    assert metadata['source_sha256']==data.digest(__file__)
    rows=pd.read_parquet(CACHE/'rows.parquet');inputs=pd.read_parquet(CACHE/'inputs.parquet')
    tokens=json.loads((CACHE/'token-ids.json').read_text())
    signature=dict(data_sha256=data.digest(OUT/'data-complete.json'),models=metadata['models'],specs=SPECS,batch_size=8)
    if (OUT/'inference-complete.json').exists():
        assert json.loads((OUT/'inference-complete.json').read_text())['signature']==signature
        return
    progress_path=CACHE/'progress.json';start_batch=0;written=0
    if progress_path.exists():
        progress=json.loads(progress_path.read_text());assert progress['signature']==signature
        start_batch=progress['next_batch'];written=progress['rows_written']
    probabilities={name:np.memmap(CACHE/f'{name}-probabilities.f32',mode='r+' if (CACHE/f'{name}-probabilities.f32').exists() else 'w+',
        dtype=np.float32,shape=(len(rows),5)) for name in paths}
    nets={name:probes.load_model(path) for name,path in paths.items()}
    for name,(_,_,_,meta) in nets.items():assert meta['feature_spec']==(data.SPECS if name=='residual' else attention.SPECS)
    runtime=data.runtime
    runtime.atomic_json=lambda path,value:data.atomic_json(OUT/Path(path).name,value)
    runtime.status=lambda stage,**fields:print(json.dumps(dict(stage=stage,**fields)),flush=True)
    model=runtime.load_model();loader=DataLoader(Prompts(inputs,tokens),batch_size=8,shuffle=False,collate_fn=collate)
    lookup={int(p):g for p,g in rows.groupby('prompt_ix',sort=False)};captured={}
    def receive(name,value):captured[name]=value.detach().to('cpu',torch.float16).reshape(-1,2880)
    started=time.time()
    with torch.inference_mode():
        batch=next(iter(loader));sample=SimpleNamespace(input_ids=batch['input_ids'].cuda(0),attention_mask=batch['attention_mask'].cuda(0))
        before=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
        handles=components.hooks(model,SPECS,receive)
        after=model.model(input_ids=sample.input_ids,attention_mask=sample.attention_mask,use_cache=False).last_hidden_state.cpu()
        assert torch.equal(before,after) and torch.isfinite(after).all()
        reference=dict(captured)
        for handle in handles:handle.remove()
        del before,after
        original=pipeline.install_hooks;pipeline.install_hooks=components.hooks
        try:
            for bi,batch,states in pipeline.feature_batches(model,loader,SPECS,start_batch,(sample,reference),
                    lambda result:data.atomic_json(OUT/'pipeline-validation.json',result)):
                width=batch['input_ids'].shape[1];positions=[];targets=[]
                for b,prompt in enumerate(batch['prompt_ix']):
                    group=lookup[prompt];offset=b*width+width-len(tokens[prompt])
                    positions.extend((group.token_ix.to_numpy()+offset).tolist());targets.extend(group.index.tolist())
                indices=torch.tensor(positions,dtype=torch.long)
                for name,(net,mean,scale,_) in nets.items():
                    specs=data.SPECS if name=='residual' else attention.SPECS
                    raw=torch.cat([states[key][indices] for key in specs],dim=1).cuda(0).float()
                    assert torch.isfinite(raw).all()
                    p=net((raw-mean)/scale).exp().cpu().numpy()
                    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=2e-5)
                    probabilities[name][targets]=p
                    del raw,p
                written+=len(targets);states.clear()
                if (bi+1)%16==0 or bi+1==len(loader):
                    for value in probabilities.values():value.flush()
                    data.atomic_json(progress_path,dict(signature=signature,next_batch=bi+1,rows_written=written))
                    record=dict(stage='confirming',batch=bi+1,batches=len(loader),rows=written,seconds=time.time()-started)
                    data.atomic_json(OUT/'status.json',record);print(json.dumps(record),flush=True)
        finally:pipeline.install_hooks=original
    assert written==len(rows)
    assert metadata['models']=={name:data.digest(path) for name,path in paths.items()}
    data.atomic_json(OUT/'inference-complete.json',dict(rows=written,signature=signature,
        source_sha256=data.digest(__file__),seconds=time.time()-started,raw_hidden_states_not_persisted=True,
        probability_sha256={name:data.digest(CACHE/f'{name}-probabilities.f32') for name in paths}))


def plots(rows,probabilities,thresholds,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder=OUT/'plots';folder.mkdir(exist_ok=True)
    colors=dict(system='#90a1b9',user='#00a6f4',cot='#fd9a00',assistant='#00d492',tool='#a855f7')
    labels=['Systemness','Userness','CoTness','Asstness','Toolness']
    chosen=np.linspace(0,399,12,dtype=int).tolist()
    for doc in chosen:
        subset=rows[(rows.document_id==doc)&(rows.profile=='multi')].sort_values('token_ix')
        ix=subset.index.to_numpy();positions=subset.token_ix.to_numpy()
        fig,axes=plt.subplots(5,2,figsize=(11,7),sharex=True,sharey=True)
        for col,(name,p) in enumerate(probabilities.items()):
            for i,role in enumerate(data.ROLES):
                ax=axes[i,col]
                for truth,color in colors.items():
                    mask=subset.role.to_numpy()==truth
                    ax.scatter(positions[mask],p[ix[mask],i],s=3,color=color,alpha=.7,rasterized=True)
                if i==0:ax.axhline(thresholds[name],ls='--',lw=.7,color='#4b5563')
                for boundary in subset.groupby('message_ix').token_ix.min():ax.axvline(boundary,lw=.5,color='#d1d5db',zorder=0)
                ax.set_ylim(-.03,1.03);ax.set_yticks([0,1],['0%','100%'])
                ax.spines[['top','right']].set_visible(False)
                if col==0:ax.set_ylabel(labels[i],fontsize=9)
            axes[0,col].set_title('Residual control' if name=='residual' else 'Attention readout',fontsize=10)
            axes[-1,col].set_xlabel('Token position in complete input')
        handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markersize=4,label=r) for r,c in colors.items()]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1),ncol=5,frameon=False,fontsize=8)
        fig.suptitle(f'fresh-c4-{doc} | constructed multi-message input',y=.945,fontsize=11)
        fig.text(.5,.01,'Frozen GPT-OSS-120B MXFP4 | colors: formal native roles | all content tokens',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.03,1,.91])
        for extension in ['png','pdf']:fig.savefig(folder/f'fresh-c4-{doc}.{extension}',dpi=160)
        plt.close(fig)
    profiles=['single','multi','long-single'];names=['residual','attention']
    fig,axes=plt.subplots(1,3,figsize=(11,3.5));x=np.arange(3)
    for ax,key,title in zip(axes,['system_precision','system_recall','system_f1'],['System precision','System recall','System F1']):
        for j,name in enumerate(names):
            values=[next(r[key] for r in metrics if r['model']==name and r['profile']==p) for p in profiles]
            bars=ax.bar(x+(j-.5)*.36,values,width=.36,label=name,color=['#64748b','#059669'][j])
            ax.bar_label(bars,labels=[f'{100*v:.1f}' for v in values],fontsize=8,padding=2)
        ax.set_title(title,fontsize=10);ax.set_ylim(0,1.12)
        ax.set_xticks(x,['Single','Multi','Long']);ax.spines[['top','right']].set_visible(False)
        ax.set_yticks([0,.5,1],['0%','50%','100%'])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(rect=[0,0,1,.91])
    for extension in ['png','pdf']:fig.savefig(folder/f'fresh-systemness.{extension}',dpi=160)
    plt.close(fig)
    (folder/'gallery-index.md').write_text('12 fresh source documents, equally spaced IDs; constructed native multi-message inputs. Both readouts use the same frozen model forward pass. All 400 documents enter metrics.\n\n'+
        '\n'.join(f'- [fresh-c4-{doc}](fresh-c4-{doc}.png)' for doc in chosen)+'\n')
    data.atomic_json(OUT/'gallery.json',dict(documents=chosen,selection='12 fixed equally spaced document IDs; no score-based selection',
        independent_source_documents_in_metrics=400,constructed_native_multi=True,panels=5,models=2,png=13,pdf=13))


def report():
    paths,thresholds=selections();rows=pd.read_parquet(CACHE/'rows.parquet')
    inference=json.loads((OUT/'inference-complete.json').read_text())
    probabilities={name:np.memmap(CACHE/f'{name}-probabilities.f32',mode='r',dtype=np.float32,shape=(len(rows),5)) for name in paths}
    metrics=[];docs=[];messages=[]
    for name,p in probabilities.items():
        assert data.digest(CACHE/f'{name}-probabilities.f32')==inference['probability_sha256'][name]
        for profile,group in rows.groupby('profile'):
            result=probes.metrics(p[group.index],probes.labels(group),thresholds[name])
            metrics.append(dict(model=name,profile=profile,groups=group.group_id.nunique(),**result))
        docs.append(evaluation.document_counts(rows,p,thresholds[name],name))
        frame=rows[['group_id','profile','message_ix','role']].copy()
        for i,role in enumerate(data.ROLES):frame[f'p_{role}']=p[:,i]
        messages.append(frame.groupby(['group_id','profile','message_ix','role'],as_index=False).agg(
            **{f'p_{role}':(f'p_{role}','mean') for role in data.ROLES},tokens=('p_system','size')).assign(model=name))
    data.atomic_json(OUT/'metrics.json',metrics)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in metrics]).to_csv(OUT/'metrics.csv',index=False)
    document=pd.concat(docs,ignore_index=True);document.to_csv(OUT/'document-counts.csv',index=False)
    pd.concat(messages,ignore_index=True).to_csv(OUT/'message-projections.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    comparisons=[];fields=['tp','fp','fn','tn','correct','tokens'];rng=np.random.RandomState(data.SEED+40)
    for profile,group in document.groupby('profile'):
        a=group[group.model=='residual'].sort_values('group_id');b=group[group.model=='attention'].sort_values('group_id')
        assert a.group_id.tolist()==b.group_id.tolist()
        av=a[fields].to_numpy(dtype=float);bv=b[fields].to_numpy(dtype=float)
        ids=rng.randint(len(a),size=(2000,len(a)))
        ap=evaluation.binary_values(av.sum(0));bp=evaluation.binary_values(bv.sum(0))
        ab=evaluation.binary_values(av[ids].sum(1));bb=evaluation.binary_values(bv[ids].sum(1))
        for metric in ap:
            low,high=np.quantile((bb[metric]-ab[metric])*100,[.025,.975])
            comparisons.append(dict(profile=profile,metric=metric,source_documents=len(a),bootstrap_replicates=2000,
                baseline=float(ap[metric]),attention=float(bp[metric]),gain_percentage_points=float(100*(bp[metric]-ap[metric])),
                ci95_low=float(low),ci95_high=float(high)))
    pd.DataFrame(comparisons).to_csv(OUT/'paired-comparisons.csv',index=False)
    lines=['Fresh 400-document confirmation of frozen attention readout','',
        '400 new source documents; the 64 long inputs are paired variants, not extra documents.',
        'Normal native system tags; frozen MXFP4 model on four A40s; pointwise readouts only.',
        'Both models and thresholds were frozen on validation before preparing this confirmation set.',
        'Prior source lines, body/full-body hashes, short prefixes and URLs excluded. Semantic near-duplicates and domain exclusion not claimed.',
        'No new independent natural conversations. Single and constructed multi inputs share the same 400 source documents.','',
        'Model      Profile       system P  system R  system F1  balanced acc  five-role acc']
    for r in metrics:
        lines.append(f'{r["model"]:10s} {r["profile"]:12s} '+ '  '.join(f'{r[k]:.6f}' for k in ['system_precision','system_recall','system_f1','system_balanced_accuracy','accuracy']))
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    plots(rows,probabilities,thresholds,metrics)
    targets={r['profile']:all(r[k]>=.9 for k in ['system_precision','system_recall','system_balanced_accuracy'])
             for r in metrics if r['model']=='attention'}
    data.atomic_json(OUT/'complete.json',dict(documents=400,long_subset=64,models=inference['signature']['models'],
        selection_sha256=data.digest(attention.OUT/'selection.json'),thresholds=thresholds,
        source_sha256=data.digest(__file__),tests_used_for_tuning=False,targets=targets,
        all_profiles_meet_90_percent_target=all(targets.values()),bootstrap_unit='source document',
        report_sha256=data.digest(OUT/'report.txt')))
    data.atomic_json(OUT/'status.json',dict(stage='complete',targets=targets))
    print((OUT/'report.txt').read_text(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','infer','report','all'],default='all')
    args=parser.parse_args()
    if args.stage in ['prepare','all']:prepare()
    if args.stage in ['infer','all']:infer()
    if args.stage in ['report','all']:report()
