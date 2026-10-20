"""Evaluate frozen probes on hundreds of document/conversation groups.

Formal role accuracy is reported only where a structural label exists. Untagged
projections retain original message functions for visualization, not ground truth.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_probes as probes


def binary_values(counts):
    # Last axis: tp, fp, fn, tn, multiclass_correct, tokens.
    tp,fp,fn,tn,correct,total=np.moveaxis(counts,-1,0)
    div=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    recall=div(tp,tp+fn); specificity=div(tn,tn+fp)
    return dict(system_precision=div(tp,tp+fp),system_recall=recall,
                system_f1=div(2*tp,2*tp+fp+fn),system_balanced_accuracy=(recall+specificity)/2,
                system_false_positive_rate=div(fp,fp+tn),accuracy=div(correct,total))


def document_counts(rows,p,threshold,model):
    y=probes.labels(rows)
    keep=y>=0
    frame=rows.loc[keep,['group_id','profile','wrapper']].copy()
    y=y[keep]; p=p[keep]; detected=p[:,0]>=threshold; truth=y==0
    frame['tp']=truth&detected; frame['fp']=~truth&detected
    frame['fn']=truth&~detected; frame['tn']=~truth&~detected
    frame['correct']=p.argmax(axis=1)==y; frame['tokens']=1
    return frame.groupby(['profile','wrapper','group_id'],as_index=False).sum(numeric_only=True).assign(model=model)


def evaluate(part):
    frozen=json.loads((data.OUT/'selection.json').read_text())
    assert frozen['test_used_for_selection'] is False
    assert data.digest(data.OUT/'models/selected.npz')==frozen['selected_model_sha256']
    assert all(data.digest(data.OUT/'models'/name)==sha for name,sha in frozen['model_hashes'].items())
    folder=data.FEATURE_CACHE/part
    meta=json.loads((data.OUT/f'extraction-{part}.json').read_text())
    rows=pd.read_parquet(folder/'rows.parquet')
    assert set(rows.split)=={'test'} and data.digest(folder/'rows.parquet')==meta['rows_sha256']
    array=np.memmap(folder/'features.f16',mode='r',dtype=np.float16,shape=(len(rows),8640))
    output=data.OUT/'evaluation'/part
    output.mkdir(parents=True,exist_ok=True)
    names=['previous-selected','selected']+probes.CASES+[f'fixed-{case}' for case in probes.CASES]
    model_hashes={name:data.digest(data.OUT/f'models/{name}.npz') for name in names}
    cache={}; all_metrics=[]; all_docs=[]; projections=[]
    for name in names:
        path=data.OUT/f'models/{name}.npz'
        key=model_hashes[name]
        if key in cache:
            p=cache[key]
        else:
            p=probes.predict_raw(path,array,np.arange(len(rows)))
            cache[key]=p
        if name=='previous-selected':
            threshold=frozen['baseline_validation']['system_threshold']
        elif name=='selected':
            threshold=frozen['system_threshold']
        else:
            with np.load(path,allow_pickle=False) as archive:
                candidate_meta=json.loads(str(archive['metadata']))
            candidate_name=next(r['candidate'] for r in [json.loads(x.read_text()) for x in (data.OUT/'candidates').glob('*.json')]
                if r['case']==candidate_meta['case'] and r['model_sha256']==key)
            threshold=json.loads((data.OUT/'candidates'/f'{candidate_name}.json').read_text())['validation']['system_threshold']
        for (profile,wrapper),group in rows.groupby(['profile','wrapper'],sort=False):
            ix=group.index.to_numpy()
            labels=probes.labels(group)
            if (labels>=0).all():
                result=probes.metrics(p[ix],labels,threshold)
                result.update(model=name,profile=profile,wrapper=wrapper,groups=int(group.group_id.nunique()))
                all_metrics.append(result)
        all_docs.append(document_counts(rows,p,threshold,name))
        if name in ['previous-selected','selected']:
            # Message-level values retain every evaluated conversation, not just gallery examples.
            frame=rows[['group_id','profile','wrapper','message_ix','semantic_role']].copy()
            for role,values in zip(data.ROLES,p.T):
                frame[f'p_{role}']=values
            aggregate=frame.groupby(['group_id','profile','wrapper','message_ix','semantic_role'],as_index=False).agg(
                **{f'p_{role}':(f'p_{role}','mean') for role in data.ROLES},tokens=('p_system','size'))
            projections.append(aggregate.assign(model=name))
            # Local-only full token probabilities enable any example to be plotted later.
            np.savez_compressed(data.CACHE/f'{part}-{name}-probabilities.npz',p=p,sample_ix=rows.sample_ix.to_numpy())
        print(json.dumps(dict(stage='test_evaluated',part=part,model=name,rows=len(rows))),flush=True)
    assert model_hashes=={name:data.digest(data.OUT/f'models/{name}.npz') for name in names}
    data.atomic_json(output/'metrics.json',all_metrics)
    flat=[]
    for r in all_metrics:
        flat.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))}|
                    {f'role_recall_{k}':v for k,v in r['recall'].items()}|
                    {f'system_detection_rate_{k}':v for k,v in r['system_detection_by_role'].items()})
    pd.DataFrame(flat).to_csv(output/'metrics.csv',index=False)
    docs=pd.concat(all_docs,ignore_index=True)
    docs.to_csv(output/'document-counts.csv',index=False)
    pd.concat(projections,ignore_index=True).to_csv(output/'message-projections.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    comparisons=[]
    fields=['tp','fp','fn','tn','correct','tokens']
    rng=np.random.RandomState(data.SEED)
    natural_templates={r['group_id']:r['system_template'] for r in json.loads((data.OUT/'natural-conversation-keys.json').read_text())}
    supplement_inputs=pd.read_parquet(data.CACHE/'supplement/inputs.parquet')
    unseen=supplement_inputs[(supplement_inputs.profile=='natural-unseen-system')&(supplement_inputs.wrapper=='native')]
    template_count=len(json.loads((data.OUT/'supplement-data-complete.json').read_text())['unseen_system_templates'])
    unseen_templates={name:i%template_count for i,name in enumerate(unseen.group_id)}
    template_summaries=[]
    for (profile,wrapper),group in docs[docs.model.isin(['previous-selected','selected'])].groupby(['profile','wrapper']):
        conversation_count=group.group_id.nunique()
        resampling_unit='source-document'
        if profile in ['natural','natural-unseen-system']:
            mapping=natural_templates if profile=='natural' else unseen_templates
            group=group.copy()
            group['template']=group.group_id.map(mapping)
            assert group.template.notna().all()
            # The system message precedes dialogue content, so repeats of its
            # wording do not create independent system-state measurements.
            group=group.groupby(['model','template'],as_index=False)[fields].sum()
            for record in group.to_dict('records'):
                template_summaries.append(record|dict(profile=profile,wrapper=wrapper))
            group['group_id']=group.template.astype(str)
            resampling_unit='system-template'
        before=group[group.model=='previous-selected'].sort_values('group_id')
        after=group[group.model=='selected'].sort_values('group_id')
        assert before.group_id.tolist()==after.group_id.tolist()
        ix=rng.randint(len(before),size=(2000,len(before)))
        a=before[fields].to_numpy(dtype=float); b=after[fields].to_numpy(dtype=float)
        av=binary_values(a.sum(axis=0)); bv=binary_values(b.sum(axis=0))
        ab=binary_values(a[ix].sum(axis=1)); bb=binary_values(b[ix].sum(axis=1))
        for metric in av:
            if wrapper=='all-user' and metric in ['system_precision','system_recall','system_f1','system_balanced_accuracy']:
                continue
            lo,hi=np.quantile((bb[metric]-ab[metric])*100,[.025,.975])
            comparisons.append(dict(profile=profile,wrapper=wrapper,metric=metric,groups=int(conversation_count),
                                    bootstrap_units=len(before),resampling_unit=resampling_unit,
                                    baseline=float(av[metric]),selected=float(bv[metric]),
                                    gain_percentage_points=float((bv[metric]-av[metric])*100),ci95_low=float(lo),ci95_high=float(hi)))
    pd.DataFrame(comparisons).to_csv(output/'paired-comparisons.csv',index=False)
    pd.DataFrame(template_summaries).to_csv(output/'system-template-counts.csv',index=False)
    record=dict(part=part,models=model_hashes,selection_sha256=data.digest(data.OUT/'selection.json'),
                rows=len(rows),groups=int(rows.group_id.nunique()),all_finite=True,probability_sums_checked=True,
                thresholds_frozen_on_validation=True,test_weights_unchanged=True,paired_group_bootstrap_replicates=2000,
                formal_and_semantic_labels_separate=True,system_prompt_repetitions_clustered=True)

    # Inspect how far the system signal persists from its header on full test text.
    position_records=[]
    boundaries=np.array([0,32,64,128,256,512,1024,2048])
    system_rows=rows[(rows.wrapper=='native')&(rows.role=='system')].copy()
    system_rows['message_position']=system_rows.token_ix-system_rows.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
    for name in ['previous-selected','selected']:
        cached=np.load(data.CACHE/f'{part}-{name}-probabilities.npz',allow_pickle=False)
        pp=cached['p'][:,0]
        threshold=(frozen['baseline_validation']['system_threshold'] if name=='previous-selected' else frozen['system_threshold'])
        for profile,group in system_rows.groupby('profile'):
            bins=np.digitize(group.message_position,boundaries,right=False)-1
            for b in np.unique(bins):
                chosen=group[bins==b]
                indices=chosen.index.to_numpy()
                position_records.append(dict(model=name,profile=profile,start=int(boundaries[b]),end=int(boundaries[b+1])-1,
                    tokens=len(chosen),system_recall=float((pp[indices]>=threshold).mean()),mean_system_probability=float(pp[indices].mean())))
    pd.DataFrame(position_records).to_csv(output/'system-by-position.csv',index=False)
    data.atomic_json(output/'complete.json',record)


def plot_gallery():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows=pd.read_parquet(data.FEATURE_CACHE/'test/rows.parquet')
    archive=np.load(data.CACHE/'test-selected-probabilities.npz',allow_pickle=False)
    assert np.array_equal(archive['sample_ix'],rows.sample_ix.to_numpy())
    p=archive['p']
    frozen=json.loads((data.OUT/'selection.json').read_text())
    groups=sorted(rows[rows.profile=='natural'].group_id.unique())
    chosen=[groups[i] for i in np.linspace(0,len(groups)-1,12,dtype=int)]
    out=data.OUT/'plots/gallery'
    out.mkdir(parents=True,exist_ok=True)
    colors=dict(system='#90a1b9',user='#00a6f4',cot='#fd9a00',assistant='#00d492',tool='#a855f7')
    names=dict(system='Systemness',user='Userness',cot='CoTness',assistant='Asstness')
    for group_id in chosen:
        for wrapper,title in [('native','Correct tags'),('all-user','Everything in user tags'),('no-tags','No tags')]:
            subset=rows[(rows.group_id==group_id)&(rows.wrapper==wrapper)].sort_values('token_ix')
            ix=subset.index.to_numpy()
            fig,axes=plt.subplots(4,1,figsize=(9,5),sharex=True)
            for i,role in enumerate(names):
                ax=axes[i]
                for source_role in colors:
                    mask=subset.semantic_role.to_numpy()==source_role
                    if mask.any():
                        ax.scatter(subset.token_ix.to_numpy()[mask],p[ix[mask],i],s=3,color=colors[source_role],alpha=.7,rasterized=True)
                ax.set_ylim(-.03,1.03);ax.set_yticks([0,1],['0%','100%'])
                ax.set_ylabel(names[role],fontsize=9)
                if i==0:
                    ax.axhline(frozen['system_threshold'],color='#4b5563',lw=.7,ls='--')
                for pos in subset.groupby('message_ix').token_ix.min():
                    ax.axvline(pos,color='#d1d5db',lw=.5,zorder=0)
                ax.spines[['top','right']].set_visible(False)
            axes[0].set_title(f'{group_id} | {title}',fontsize=11)
            axes[-1].set_xlabel('Token position in full input')
            handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=colors[r],markersize=4,label=r) for r in ['system','user','cot','assistant']]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.99),ncol=4,frameon=False,fontsize=8)
            fig.text(.5,.005,'GPT-OSS-120B MXFP4 | five-role probe | colors show original message function',ha='center',fontsize=8)
            fig.tight_layout(rect=[0,.02,1,.94])
            for suffix in ['png','pdf']:
                fig.savefig(out/f'{group_id}-{wrapper}.{suffix}',dpi=160)
            plt.close(fig)
    data.atomic_json(data.OUT/'gallery.json',dict(conversations=chosen,formats=['native','all-user','no-tags'],
            selection='12 equally spaced conversation IDs; no accuracy-based example selection',png=36,pdf=36,
            all_144_conversations_in_metrics=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',choices=['test','challenge'])
    parser.add_argument('--gallery',action='store_true')
    args=parser.parse_args()
    if args.gallery:plot_gallery()
    else:evaluate(args.part)
