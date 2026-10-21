"""Independent completed-result audit. Default help; preflight is toy CPU only.

--audit-result reads one terminal dataset only when the root agent authorizes it.
No producer import, model/scaler/tokenizer load, source body decode or LM replay.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
BASE=ROOT/'outputs/local-gptoss120b'
SNAPSHOT=BASE/'systemness-complementary-confirmation-preparation/readout-snapshot.json'
SNAPSHOT_FILE_SHA='58f95bd32c0866881f940fc30075487770abe04a28eb7ee383b44e2718b8d6e6'
SNAPSHOT_CONTENT_SHA='60e683376f7fd4535c1cc307f028651d1c0c23a0d1449b988438a90d98808f46'
SOURCE_PINS={
 'systemness_complementary_guard_confirm_snapshot.py':'12b2078ffdb358801855ef9c31a31816e96724acd813f340f0caf49683e834ae',
 'systemness_complementary_confirmation_data.py':'639b7b21163a202b8e0caa311958609103b4a5d3c2c5376610351638f2a25a82',
 'systemness_complementary_natural_data.py':'cc9c8f3ad46c214747e3a49ab18f81701b88f09e8805b229ae56c55127944fa4',
 'systemness_complementary_confirmation_infer.py':'4aabd2fbb8139223c16273f451087739b33a92b0a84b2209130d78b0494f93cf',
 'systemness_complementary_confirmation_report.py':'cfb26eaafb02f313f5cce086fcbddbd6008f0445af706aea2dbd5774424e63be',
 'systemness_complementary_natural_report.py':'d82be0edb5cf7a111a2b22cb7e78193a892e48fa446ad7e86d825e58cd138054'}
ROLES=['system','user','cot','assistant','tool'];OBS=['system','user','assistant']
RECORDS=['new-guard-max33','matched-max33','oldhead-max33','oldhead-original']
MAP=dict(zip(RECORDS,['new','matched','old-head','old-head']));PHYSICAL=['new','matched','old-head']
PROFILES=['single','multi','long-single','long-full','long-full-tail']
BINS=[(0,31),(32,63),(64,127),(128,255),(256,511),(512,1023)]
BIN_NAMES=[f'long-full-bin-{a:04d}-{b:04d}' for a,b in BINS];GROUPS=PROFILES+BIN_NAMES
STRATA=list(itertools.product(['all-prepared','clean-test'],['all','oasst','toxicchat0124']))
SIX=['system_precision','system_recall','system_f1','system_balanced_accuracy','system_false_positive_rate','accuracy']
NINE=SIX+['role_recall_'+r for r in OBS]
COUNT=['tp','fp','fn','tn','correct','tokens'];NCOUNT=COUNT+sum(([r+'_correct',r+'_tokens'] for r in OBS),[])
EXPECTED_DOCS=dict(zip(GROUPS,[400,400,64,128,128]+[128]*6))
NAMES={'c4':'systemness-complementary-confirmation','natural':'systemness-complementary-natural-confirmation'}
SEED=20261021;REPLICATES=2000
PROVENANCE={}


def require(value,message):
    if not bool(value):raise RuntimeError(message)
def clean(p):
    p=Path(os.path.abspath(str(p)));require(p!=Path('/workspace') and Path('/workspace') not in p.parents,'Forbidden workspace mount.');return p
def sha(p):
    p=clean(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def bind(p,want=None):
    p=clean(p);got=sha(p);require(want is None or got==want,'SHA mismatch: '+str(p));PROVENANCE[str(p)]=got;return got
def read(p):return json.loads(clean(p).read_text())
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def finite(x):return float(x) if np.isfinite(x) else None
def same(a,b,label='value',tol=1e-12):
    if isinstance(b,dict):
        require(isinstance(a,dict) and set(b)<=set(a),label+' keys')
        for k,v in b.items():same(a[k],v,label+'/'+k,tol)
    elif isinstance(b,list):
        require(isinstance(a,list) and len(a)==len(b),label+' length')
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,label+'/'+str(i),tol)
    elif b is None:require(a is None or isinstance(a,float) and np.isnan(a),label+' should be undefined')
    elif isinstance(b,(bool,str)):require(type(a)==type(b) and a==b,label+' exact')
    else:require(a is not None and np.isfinite(a) and abs(float(a)-float(b))<=tol,label+' numeric')
def table(path,expected,keys,tol=1e-12):
    actual=pd.read_csv(path);require(len(actual)==len(expected),'CSV row count '+str(path))
    ids=lambda x:tuple(x[k] for k in keys)
    lookup={ids(x):x for x in actual.to_dict('records')};require(len(lookup)==len(actual),'Duplicate CSV key.')
    for x in expected:
        require(ids(x) in lookup,'CSV key missing.');same(lookup[ids(x)],x,str(path),tol)
def metric_csv(path,items,keys):
    flat=[{k:v for k,v in x.items() if not isinstance(v,(dict,list))}|{'role_support_'+r:v for r,v in x['role_support'].items()}|{'role_recall_'+r:v for r,v in x['recall'].items()} for x in items]
    table(path,flat,keys)
def ratio(a,b):return np.divide(a,b,out=np.full(np.shape(a),np.nan,dtype=float),where=np.asarray(b)>0)


def values(c,natural=False):
    c=np.asarray(c,dtype=float);tp,fp,fn,tn,correct,total=np.moveaxis(c[...,:6],-1,0)
    r=ratio(tp,tp+fn);out=dict(system_precision=ratio(tp,tp+fp),system_recall=r,system_f1=ratio(2*tp,2*tp+fp+fn),
        system_balanced_accuracy=(r+ratio(tn,tn+fp))/2,system_false_positive_rate=ratio(fp,fp+tn),accuracy=ratio(correct,total))
    if natural:
        for i,r in enumerate(OBS):out['role_recall_'+r]=ratio(c[...,6+2*i],c[...,7+2*i])
    return out


def metric(p,y,tau,natural=False):
    """Own integer confusion/counts and tied-ranking formulas; original F32 consumer."""
    y=np.asarray(y,dtype=np.int64);pred=p.argmax(1);yes=y==0;det=p[:,0]>=np.float32(tau)
    matrix=np.bincount(y*5+pred,minlength=25).reshape(5,5);support=matrix.sum(1)
    c=np.array([np.sum(yes&det),np.sum(~yes&det),np.sum(yes&~det),np.sum(~yes&~det),np.sum(pred==y),len(y)]+
        ([z for r in [0,1,3] for z in [np.sum((y==r)&(pred==r)),np.sum(y==r)]] if natural else []),np.int64)
    v={k:finite(z) for k,z in values(c,natural).items()};den=support+matrix.sum(0)
    f=np.divide(2*matrix.diagonal(),den,out=np.zeros(5),where=den>0)
    auc=ap=None;npos=int(yes.sum());nneg=len(y)-npos
    if npos:
        order=np.argsort(-p[:,0],kind='stable');s=p[order,0];t=yes[order]
        end=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];tp=np.cumsum(t)[end]
        ap=float(np.sum(np.diff(np.r_[0,tp])/npos*tp/(end+1)))
        if nneg:
            from scipy.stats import rankdata
            ranks=rankdata(p[:,0],method='average');auc=float((ranks[yes].sum()-npos*(npos+1)/2)/(npos*nneg))
    v.update(tokens=len(y),system_threshold=float(tau),macro_f1=float(f[support>0].mean()) if len(y) else None,
        nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()) if len(y) else None,
        recall={r:int(matrix[i,i])/int(support[i]) if support[i] else None for i,r in enumerate(ROLES)},
        confusion=matrix.tolist(),role_support={r:int(support[i]) for i,r in enumerate(ROLES)},system_auroc=auc,system_average_precision=ap,
        system_confusion=dict(zip(COUNT[:4],map(int,c[:4]))),system_detection_by_role={r:float(det[y==i].mean()) if support[i] else None for i,r in enumerate(ROLES)})
    if natural:v.update(five_class_argmax_accuracy=v['accuracy'],status='observed-SUA-only' if len(y) else 'no-observations',
        observed_roles=[r for i,r in enumerate(ROLES) if support[i]],cot_tool_recall_status='N/A: zero observed truth support')
    else:v['system_accuracy']=float((yes==det).mean())
    return v,c,pred,det


def source_counts(rows,y,pred,det,natural=False):
    truth=y==0;cols=NCOUNT if natural else COUNT
    ids=['source_uid','family_id','dependency_component_id'] if natural else ['group_id','document_id']
    frame=rows[ids].copy()
    for k,v in zip(COUNT,[truth&det,~truth&det,truth&~det,~truth&~det,pred==y,np.ones(len(y),np.int64)]):frame[k]=v
    if natural:
        for r in OBS:
            mask=y==ROLES.index(r);frame[r+'_correct']=mask&(pred==y);frame[r+'_tokens']=mask
    return frame.groupby(ids,as_index=False)[cols].sum()


def intervals(arrays,seed,natural=False):
    """Paired original documents or already-unioned dependency components."""
    n=len(arrays[RECORDS[0]]);draw=np.random.RandomState(seed).randint(n,size=(REPLICATES,n)) if n else np.empty((REPLICATES,0),np.int64)
    points={r:values(a.sum(0),natural) for r,a in arrays.items()};sample={r:values(a[draw].sum(1),natural) for r,a in arrays.items()}
    fields=NINE if natural else SIX;cis=[];pairs=[]
    def ci(x):
        ok=np.isfinite(x);num=int(ok.sum());reason=('fewer_than_two_dependency_components' if n<2 else 'no_defined_bootstrap_replicates' if not num else None) if natural else (None if num and n>=2 else 'undefined_ratio_or_fewer_than_two_sources')
        low,high=np.quantile(x[ok],[.025,.975]) if reason is None else [np.nan,np.nan]
        return dict(ci95_low=finite(low),ci95_high=finite(high),**(dict(defined_bootstrap_replicates=num,undefined_bootstrap_replicates=REPLICATES-num,ci_available=reason is None,ci_unavailable_reason=reason) if natural else dict(valid_bootstrap_draws=num,ci_unavailable_reason=reason)))
    for r in RECORDS:
        for k in fields:cis.append(dict(model=r,metric=k,value=finite(points[r][k]),**ci(sample[r][k])))
    for a,b in itertools.combinations(RECORDS,2):
        for k in fields:
            out=dict(baseline_model=a,comparison_model=b,metric=k,baseline=finite(points[a][k]),comparison=finite(points[b][k]),gain_percentage_points=finite((points[b][k]-points[a][k])*100),**ci((sample[b][k]-sample[a][k])*100))
            if not natural:out['valid_paired_bootstrap_draws']=out.pop('valid_bootstrap_draws')
            pairs.append(out)
    return cis,pairs


def terminal_gate(dataset):
    """Same-dataset producers must stop; the companion may independently infer."""
    report_file='systemness_complementary_'+('confirmation_report.py' if dataset=='c4' else 'natural_report.py')
    builder_file='systemness_complementary_'+('confirmation_data.py' if dataset=='c4' else 'natural_data.py')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0');args=[a.decode(errors='replace') for a in args if a]
        except(OSError,ProcessLookupError):continue
        names={Path(a).name for a in args}
        same_infer='systemness_complementary_confirmation_infer.py' in names and (('--dataset' in args and args[args.index('--dataset')+1]==dataset) or '--dataset='+dataset in args)
        require(not (report_file in names or builder_file in names or same_infer),'Same-dataset producer still alive: '+entry.name)
    out=BASE/NAMES[dataset];report=out/'report'
    require((out/'inference/inference-complete.json').is_file() and read(out/'inference/status.json')['stage']=='inference_complete','Inference not terminal.')
    require((report/'complete.json').is_file(),'Report not complete.')
    if dataset=='c4':require(read(report/'status.json')['stage']=='complete','C4 report still partial.')
    return out,report


def load(dataset):
    out,report=terminal_gate(dataset)  # No p/cache/snapshot reads precede this gate.
    for name,want in SOURCE_PINS.items():bind(ROOT/name,want)
    bind(__file__);bind(SNAPSHOT,SNAPSHOT_FILE_SHA);snap=read(SNAPSHOT);content=dict(snap);content.pop('snapshot_sha256')
    require(canonical(content)==snap['snapshot_sha256']==SNAPSHOT_CONTENT_SHA and snap['record_to_physical']==MAP,'Snapshot hash/mapping differs.')
    for name,want in SOURCE_PINS.items():require(snap['runtime_files_sha256'].get(str(ROOT/name))==want,'Source not prospectively frozen.')
    thresholds=snap['record_thresholds'];require(set(thresholds)==set(RECORDS) and all(np.isfinite(x) and 0<=x<=1.00001 for x in thresholds.values()),'Frozen thresholds differ.')
    bind(out/'data-complete.json');meta=read(out/'data-complete.json');require(meta['frozen']==snap and meta['frozen_sha256']==SNAPSHOT_CONTENT_SHA and meta['snapshot_file_sha256']==SNAPSHOT_FILE_SHA,'Prepared data froze another readout.')
    require(meta['coverage']['all_content_tokens_scored'] is True and meta['coverage']['independent_native_labels_match'] is True,'Native label/all-content proof missing.')
    require(meta['coverage']['normal_system_tag_retained' if dataset=='c4' else 'normal_system_tag'] is True,'Normal system tag proof absent.')
    for name,want in meta['metadata_sha256'].items():bind(out/name,want)
    bind(out/'plan.json',meta['plan_sha256']);cache=clean(meta['cache_path'])
    require(cache==BASE/'systemness-expansion/cache'/('complementary-confirmation' if dataset=='c4' else 'complementary-natural-confirmation'),'Wrong cache namespace.')
    for name,want in meta['cache_sha256'].items():bind(cache/name,want) # Hash body/token payloads without parsing them.
    cols=['cache_ix','sample_ix','prompt_ix','token_ix','token_id','body_offset','role','group_id','document_id','split','wrapper']
    if dataset=='c4':cols+=['profile']
    else:cols+=['source_uid','dataset','family_id','dependency_component_id','eligible_clean_test','message_ix']
    rows=pd.read_parquet(cache/'rows.parquet',columns=cols);inputs=pd.read_parquet(cache/'inputs.parquet',columns=['prompt_ix','length'])
    n=len(rows);require(rows.index.to_list()==list(range(n)) and np.array_equal(rows.cache_ix,np.arange(n)) and rows.sample_ix.is_unique and rows.split.eq('test').all() and rows.wrapper.eq('native').all(),'Row ownership differs.')
    require(np.array_equal(inputs.prompt_ix,np.arange(len(inputs))) and n==meta['rows'] and len(inputs)==meta['inputs'] and int(inputs.length.sum())==meta['input_tokens'],'Input/source counts differ.')
    bind(out/'inference/inference-complete.json');done=read(out/'inference/inference-complete.json');sig=done['signature'];pcache=cache/'inference'
    require(sig['dataset']==dataset and sig['source_sha256']==SOURCE_PINS['systemness_complementary_confirmation_infer.py'] and sig['data_complete_sha256']==sha(out/'data-complete.json') and sig['snapshot_sha256']==SNAPSHOT_CONTENT_SHA and sig['snapshot_file_sha256']==SNAPSHOT_FILE_SHA,'Inference signature/source differs.')
    require(sig['record_thresholds']==thresholds and sig['model_sha256']==snap['physical_model_sha256'] and sig['rows']==n and sig['inputs']==len(inputs) and sig['sample_ix_sha256']==array_sha(rows.sample_ix.to_numpy()) and sig['cache_ix_sha256']==array_sha(rows.cache_ix.to_numpy()),'Inference rows/model/threshold ownership differs.')
    require(sig['cache_sha256']==meta['cache_sha256'] and sig['batch_prompt_order_sha256']==canonical(inputs.sort_values(['length','prompt_ix']).prompt_ix.to_list()),'Prepared cache/order differs.')
    require(done['rows']==n and done['inputs']==len(inputs) and done['all_finite'] is True and done['all_normalized'] is True and done['no_features_persisted'] is True,'Completed inference coverage missing.')
    for key in ['probability_sha256','argmax_sha256']:require(set(done[key])==set(PHYSICAL),'Three physical caches required.')
    require(done['record_to_physical']==MAP and done['old_head_two_records_share_probability_and_argmax'] is True,'Record aliases differ.')
    for key,rkey in [('probability_sha256','record_probability_sha256'),('argmax_sha256','record_argmax_sha256')]:require(done[rkey]=={r:done[key][MAP[r]] for r in RECORDS},'Old probability/argmax alias differs.')
    p={};evidence={}
    if n:
        require(done['no_observations'] is False and done['gpu_run'] is True and done['model_forward_run'] is True,'Observed rows lack GPU evidence.')
        owner=read(pcache/'initialized.json');require(owner==dict(signature=sig,models=PHYSICAL,rows=n,classes=5,dtype='float32'),'Probability owner differs.');bind(pcache/'initialized.json')
        bind(pcache/'progress.json',done['progress_sha256']);prog=read(pcache/'progress.json')
        require(prog['signature']==sig and prog['next_batch']==math.ceil(len(inputs)/4) and prog['rows_written']==n and prog['sample_ix_sha256']==sig['sample_ix_sha256'] and prog['cache_ix_sha256']==sig['cache_ix_sha256'] and prog['completed_probability_sha256']==done['probability_sha256'],'Final committed prefix differs.')
        for name,key in [('portable-validation.json','portable_validation_sha256'),('hook-validation.json','hook_validation_sha256'),('model-backend.json','model_backend_sha256')]:bind(out/'inference'/name,done[key]);evidence[name]=read(out/'inference'/name)
        backend=evidence['model-backend.json'];hooks=evidence['hook-validation.json'];portable=evidence['portable-validation.json']
        require(all(e['signature']==sig for e in evidence.values()),'Evidence signed another inference.')
        require(backend['native_mxfp4'] is True and backend['parameters_frozen'] is True and backend['expert_backends']==['Mxfp4GptOssExperts'] and backend['checkpoint_path']=='/dev/shm/role-analysis-gptoss120b' and len(backend['gpus'])==4 and all('A40' in s for s in backend['gpus']) and backend['compute_capabilities']==[[8,6]]*4,'Native4A40 MXFP4 proof differs.')
        device_map={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':3,'lm_head':3,**{f'model.layers.{i}':i//9 for i in range(36)}}
        require(backend['device_map']==device_map and backend['gpu_transfer']=='Synchronous CPU staging between different CUDA devices','36layers9/card backend differs.')
        require(hooks['hidden_states_bitexact'] is True and hooks['all_capture_writes_disabled'] is True and hooks['only_pre_o_proj_hooks'] is True and hooks['legacy_capture'] is False and hooks['dimensions']==640 and hooks['selected_head_ids']==sig['selected_head_ids'],'Disabled-hook proof differs.')
        require(portable['tolerance']==5e-4 and portable['same_vectors'] is True and portable['CPU_outputs_are_saved'] is True and portable['first_original_batch'] is True and set(portable['max_probability_difference'])==set(PHYSICAL) and all(0<=x<5e-4 for x in portable['max_probability_difference'].values()),'Same-vector portable parity proof differs.')
        first=inputs.sort_values(['length','prompt_ix']).prompt_ix.to_list()[:4];targets=np.concatenate([rows.index[rows.prompt_ix.eq(prompt)].to_numpy(np.int64) for prompt in first]);count=min(32,len(targets))
        require(portable['sampled_tokens']==count and portable['target_cache_ix_sha256']==array_sha(targets[:count]) and portable['sample_ix_sha256']==array_sha(rows.iloc[targets[:count]].sample_ix.to_numpy()) and portable['captured_dtype']=='float16' and portable['selected_head_ids']==sig['selected_head_ids'] and portable['physical_feature_views']==snap['physical_feature_views'],'Portable parity first-batch/sample/view ownership differs.')
        require(hooks['sample_rows']==len(targets) and hooks['sample_ix_sha256']==array_sha(rows.iloc[targets].sample_ix.to_numpy()),'Disabled-hook first-batch sample ownership differs.')
        for physical in PHYSICAL:
            path=pcache/(physical+'-probabilities.f32');require(path.stat().st_size==n*20,'Truncated probability cache.');bind(path,done['probability_sha256'][physical]);p[physical]=np.memmap(path,mode='r',dtype=np.float32,shape=(n,5));h=hashlib.sha256()
            for start in range(0,n,65536):
                block=p[physical][start:start+65536];require(np.isfinite(block).all() and (block>=0).all() and (block<=1.00001).all() and np.allclose(block.sum(1),1,atol=2e-5,rtol=0),'Nonfinite or unnormalized p.');h.update(block.argmax(1).astype(np.int64).tobytes())
            require(h.hexdigest()==done['argmax_sha256'][physical],'Argmax hash mismatch.')
    else:
        require(dataset=='natural' and done['no_observations'] is True and done['gpu_run'] is False and done['model_forward_run'] is False,'Empty inference invented observations.')
        require(all(v is None for k in ['probability_sha256','argmax_sha256'] for v in done[k].values()),'Empty probabilities have SHA.');p={r:np.empty((0,5),np.float32) for r in PHYSICAL}
    bind(report/'complete.json');rc=read(report/'complete.json')
    for name,want in rc['artifact_sha256'].items():bind(report/name,want)
    identity=rc['signature'] if dataset=='c4' else rc['source_identity']
    require(identity['inference_complete_sha256']==sha(out/'inference/inference-complete.json') and identity['data_complete_sha256']==sha(out/'data-complete.json'),'Report source/completion differs.')
    report_source='systemness_complementary_'+('confirmation_report.py' if dataset=='c4' else 'natural_report.py')
    require(rc['source_sha256']==SOURCE_PINS[report_source] and rc['record_to_physical']==MAP and rc['record_thresholds' if dataset=='c4' else 'system_thresholds']==thresholds,'Report source/model/decision ownership differs.')
    require(identity['frozen_snapshot_sha256' if dataset=='c4' else 'snapshot_sha256']==SNAPSHOT_CONTENT_SHA and identity['model_sha256']==snap['physical_model_sha256'],'Report snapshot/model differs.')
    for key,ck in [('physical_probability_sha256' if dataset=='c4' else 'probability_sha256','probability_sha256'),('physical_argmax_sha256' if dataset=='c4' else 'argmax_sha256','argmax_sha256'),('record_probability_sha256','record_probability_sha256'),('record_argmax_sha256','record_argmax_sha256')]:require(identity[key]==done[ck],'Report probability/argmax chain differs.')
    return out,report,meta,rows,inputs,p,thresholds,done,rc


def audit_c4(data):
    out,report,meta,rows,inputs,p,tau,done,rc=data;require(rows.profile.isin(PROFILES[:-1]).all() and rows.group_id.nunique()==528 and len(inputs)==3360,'C4 coverage differs.')
    full=rows.profile.eq('long-full');groups={g:np.flatnonzero(rows.profile.eq(g)) for g in PROFILES[:-1]};groups['long-full-tail']=np.flatnonzero(full&rows.body_offset.ge(512))
    groups.update({g:np.flatnonzero(full&rows.body_offset.between(a,b)) for g,(a,b) in zip(BIN_NAMES,BINS)})
    f=rows[full];require(len(f)==128*5*1024 and not set(f.group_id)&set(rows.loc[~full,'group_id']),'Full cohort overlap/coverage.')
    for _,g in f.groupby(['group_id','role']):require(g.body_offset.tolist()==list(range(1024)),'Full role body truncated.')
    saved=read(report/'metrics.json')+read(report/'system-by-position.json');require(len(saved)==44,'C4 metrics count.');lookup={(v['record'],v['profile']):v for v in saved};require(len(lookup)==44,'Duplicate metric key.')
    docs=[];ci=[];pairs=[];target={};source={}
    for gi,(name,ix) in enumerate(groups.items()):
        r=rows.iloc[ix];require(r.group_id.nunique()==EXPECTED_DOCS[name],'C4 independent source count.');arrays={};ids=None
        source[name]=dict(source_documents=int(r.group_id.nunique()),tokens=len(r),role_support={k:int(r.role.eq(k).sum()) for k in ROLES})
        for rec in RECORDS:
            pp=p[MAP[rec]][ix];v,_,pred,det=metric(pp,r.role.map(dict(zip(ROLES,range(5)))).to_numpy(),tau[rec]);v.update(record=rec,physical_model=MAP[rec],profile=name,source_documents=EXPECTED_DOCS[name]);same(lookup[rec,name],v,'C4 metric '+rec+'/'+name)
            if name in BIN_NAMES:
                a,b=BINS[BIN_NAMES.index(name)];same(lookup[rec,name],dict(start=a,end=b,system_tokens=int(r.role.eq('system').sum()),system_source_documents=128,mean_system_probability=float(pp[r.role.eq('system').to_numpy(),0].mean())))
            c=source_counts(r,r.role.map(dict(zip(ROLES,range(5)))).to_numpy(),pred,det);c=c.sort_values('group_id');require(ids is None or c.group_id.tolist()==ids,'Unpaired document units.');ids=c.group_id.tolist();arrays[rec]=c[COUNT].to_numpy(float);docs.extend(c.assign(record=rec,physical_model=MAP[rec],profile=name).to_dict('records'))
            if rec==RECORDS[0]:target[name]=all(v[k] is not None and v[k]>=.9 for k in ['system_precision','system_recall','system_balanced_accuracy'])
        ic,pc=intervals(arrays,SEED+528123+gi)
        for x in ic:x['record']=x.pop('model');x.update(profile=name,source_documents=len(ids),bootstrap_unit='original-source-document',bootstrap_replicates=2000)
        for x in pc:x['baseline_record']=x.pop('baseline_model');x['comparison_record']=x.pop('comparison_model');x.update(profile=name,source_documents=len(ids),bootstrap_unit='original-source-document',bootstrap_replicates=2000)
        ci+=ic;pairs+=pc
    table(report/'document-counts.csv',docs,['record','profile','group_id']);table(report/'bootstrap-metrics.csv',ci,['profile','record','metric']);table(report/'paired-comparisons.csv',pairs,['profile','baseline_record','comparison_record','metric'])
    metric_csv(report/'metrics.csv',saved[:20],['record','profile']);metric_csv(report/'system-by-position.csv',saved[20:],['record','profile'])
    same(read(report/'source-counts.json')['groups'],source);require(rc['primary_record']==RECORDS[0] and rc['working_target_pass']==target and rc['all_primary_groups_meet_90_percent']==all(target.values()) and rc['thresholds_retuned'] is False and rc['test_results_used_for_selection'] is False and rc['primary_changed'] is False,'C4 reported target/selection differs.')
    gallery=read(report/'gallery.json');fixed=read(out/'gallery-index.json');require(len(gallery['entries'])==16 and gallery['png_pdf_pairs']==17 and [x['cohort_rank'] for x in fixed['full_system']]==[0,42,84,127],'C4 plot count/fixed selection.')
    require([x['group_id'] for x in gallery['entries']]==[x['group_id'] for x in fixed['general_multi']+fixed['full_system']] and gallery['fixed_before_probabilities'] is True and gallery['score_based_selection'] is False and gallery['smoothing'] is False,'C4 gallery reselected.')
    require([x['cohort_rank'] for x in fixed['general_multi']]==np.linspace(0,399,12,dtype=int).tolist(),'C4 general gallery rank.')
    for x in gallery['entries']:require(x['axes']==15 and (not x['system_only'] or (x['x_min'],x['x_max'],x['content_tokens'],x['truth_roles'])==(0,1023,1024,['system'])),'C4 gallery panel/body geometry.')
    return dict(metrics=44,intervals=len(ci),paired_comparisons=len(pairs),source_counts=source,primary_pass_by_group=target,all_primary_groups_pass=all(target.values()),png_pdf_pairs=17)


def audit_natural(data):
    out,report,meta,rows,inputs,p,tau,done,rc=data;require(set(rows.role)<=set(OBS) and meta['fixed_roots']==128 and meta['fixed_system_families']==32,'Natural truth/root scope.')
    saved=read(report/'metrics.json');lookup={(v['model'],v['stratum'],v['dataset']):v for v in saved};require(len(saved)==len(lookup)==24,'Natural metrics coverage.')
    counts=[];component=[];ci=[];pairs=[];goals=[];sources={}
    for si,(stratum,dataset) in enumerate(STRATA):
        subset=rows if stratum=='all-prepared' else rows.loc[rows.eligible_clean_test.eq(True)];r=subset if dataset=='all' else subset.loc[subset.dataset.eq(dataset)];arrays={};ids=None
        sources[stratum+'/'+dataset]=dict(roots=int(r.source_uid.nunique()),families=int(r.family_id.nunique()),components=int(r.dependency_component_id.nunique()),tokens=len(r))
        for rec in RECORDS:
            v,_,pred,det=metric(p[MAP[rec]][r.index.to_numpy()],r.role.map(dict(zip(ROLES,range(5)))).to_numpy(),tau[rec],True);v.update(stratum=stratum,dataset=dataset,model=rec,physical_model=MAP[rec],source_roots=int(r.source_uid.nunique()),system_families=int(r.family_id.nunique()),dependency_components=int(r.dependency_component_id.nunique()));same(lookup[rec,stratum,dataset],v,'Natural '+rec+'/'+stratum+'/'+dataset)
            c=source_counts(r,r.role.map(dict(zip(ROLES,range(5)))).to_numpy(),pred,det,True);counts+=c.assign(stratum=stratum,dataset=dataset,model=rec).to_dict('records');aggregate=c.groupby('dependency_component_id',as_index=False)[NCOUNT].sum().sort_values('dependency_component_id');require(ids is None or aggregate.dependency_component_id.tolist()==ids,'Unpaired natural components.');ids=aggregate.dependency_component_id.tolist();arrays[rec]=aggregate[NCOUNT].to_numpy(float);component+=aggregate.assign(stratum=stratum,dataset=dataset,model=rec).to_dict('records')
            goals.append(dict(model=rec,stratum=stratum,dataset=dataset,supported=len(r)>0,system_P_R_BAcc_ge90=all(v[k] is not None and v[k]>=.9 for k in ['system_precision','system_recall','system_balanced_accuracy']) if len(r) else None))
        ic,pc=intervals(arrays,SEED+128432+si,True);common=dict(stratum=stratum,dataset=dataset,source_roots=int(r.source_uid.nunique()),dependency_components=len(ids),system_families=int(r.family_id.nunique()),bootstrap_replicates=2000,resampling_unit='shared-family-and-assistant-repeat connected component')
        ci += [dict(x,**common) for x in ic];pairs += [dict(x,**common) for x in pc]
    table(report/'source-counts.csv',counts,['model','stratum','dataset','source_uid']);table(report/'component-counts.csv',component,['model','stratum','dataset','dependency_component_id']);table(report/'bootstrap-metrics.csv',ci,['model','stratum','dataset','metric']);table(report/'paired-comparisons.csv',pairs,['baseline_model','comparison_model','stratum','dataset','metric']);metric_csv(report/'metrics.csv',saved,['model','stratum','dataset'])
    same(rc['goal_flags'],goals);require(rc['system_thresholds']==tau and rc['source_replies_generated_by_120b'] is False and rc['source_replies_generated_under_assigned_system'] is False and rc['no_instruction_following_claim'] is True and rc['thresholds_retuned'] is False and rc['models_reselected'] is False and rc['primary_changed'] is False and rc['test_used_for_selection'] is False,'Natural semantic scope changed.')
    status=read(report/'source-status.json');records=status['records'];require(len(records)==128 and len({x['source_uid'] for x in records})==128,'Fixed128 source statuses missing.');prepared=[x for x in records if x['status']=='prepared'];require({x['source_uid'] for x in prepared}==set(rows.source_uid),'Status/observed roots differ.')
    for x in prepared:
        r=rows.loc[rows.source_uid.eq(x['source_uid'])];require(r.family_id.eq(x['family_id']).all() and r.dependency_component_id.eq(x['dependency_component_id']).all() and r.eligible_clean_test.eq(x['eligible_clean_test']).all(),'Prepared clean/component status differs.')
    comp=meta['components'];mapping={u:c['component_id'] for c in comp for u in c['source_uids']}
    require(mapping==dict(rows[['source_uid','dependency_component_id']].drop_duplicates().to_numpy()),'Prepared union components changed.')
    gallery=read(report/'gallery.json');require(gallery['source_uids']==meta['gallery_source_uids'] and len(gallery['source_uids'])==12 and gallery['png']==gallery['pdf']==13 and gallery['failed_ids_replaced'] is False and gallery['unsmoothed'] is True,'Natural fixed gallery differs.')
    index={x['source_uid']:x for x in records};require([index[u]['dataset'] for u in gallery['source_uids']].count('oasst')==6 and [index[u]['dataset'] for u in gallery['source_uids']].count('toxicchat0124')==6,'Natural6OA6TC gallery differs.')
    primary={s+'/'+d:next(x['system_P_R_BAcc_ge90'] for x in goals if x['model']==RECORDS[0] and x['stratum']==s and x['dataset']==d) for s,d in STRATA}
    return dict(metrics=24,intervals=len(ci),paired_comparisons=len(pairs),source_counts=sources,fixed_source_statuses=128,primary_pass_by_stratum=primary,all_primary_strata_pass=all(v is True for v in primary.values()),png_pdf_pairs=13)


def actual(dataset,output):
    start=sha(__file__);data=load(dataset);result=audit_c4(data) if dataset=='c4' else audit_natural(data)
    out,report,meta,rows,inputs,p,tau,done,rc=data;aliases={r:p[MAP[r]] for r in RECORDS};require(aliases['oldhead-max33'] is aliases['oldhead-original'],'Old alias object differs.')
    swaps={}
    for arm in ['new','matched']:
        old=p['old-head'].argmax(1);new=p[arm].argmax(1);swaps[arm]=int(np.sum((old!=0)&(new!=0)&(old!=new)))
    # Architecture/consumer invariance diagnostic; never replaces the frozen test target.
    plots=[n for n in rc['artifact_sha256'] if n.startswith('plots/')];expected=17 if dataset=='c4' else 13
    require(sum(n.endswith('.png') for n in plots)==sum(n.endswith('.pdf') for n in plots)==expected,'Plot inventory count differs.')
    for n in plots:
        with (report/n).open('rb') as f:header=f.read(8)
        require(header.startswith(b'\x89PNG') if n.endswith('.png') else header.startswith(b'%PDF-'),'Plot format missing.')
    require(start==sha(__file__),'Auditor changed during run.')
    value=dict(stage='actual_completed_fresh_complementary_confirmation_result_audit',dataset=dataset,audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=start,
        snapshot_file_sha256=SNAPSHOT_FILE_SHA,snapshot_content_sha256=SNAPSHOT_CONTENT_SHA,primary_record=RECORDS[0],system_thresholds=tau,
        row_count=len(rows),input_count=len(inputs),input_tokens=int(inputs.length.sum()),input_length_range=[int(inputs.length.min()),int(inputs.length.max())] if len(inputs) else None,all_probability_rows_finite_normalized=True,physical_models=3,records=4,oldhead_two_thresholds_one_probability=True,
        observed_non_system_to_different_non_system_swaps=swaps,**result,provenance_sha256=PROVENANCE,
        test_data_or_results_read=True,test_probability_arrays_read=True,test_results_used_for_selection=False,model_or_threshold_changed=False,gpu_used=False,GPU_or_model_inference_run=False,source_bodies_or_tokenizer_read=False,
        independent_LM_or_raw_activation_replay=False,labels_and_source_exclusions='Frozen byte-bound prepared rows and independent-builder proof; no new source/parser/LM replay.',
        gallery_audit='Fixed metadata, PNG/PDF inventory/header and exact artifact hashes; visual series rendering is a separate review.',
        joint_goal='Both separately audited fresh C4 and natural targets required; this one-dataset audit never claims joint success or behavioral compliance.')
    output=clean(output);require(output.is_relative_to(Path('/tmp')),'Only /tmp audit output allowed.');output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(output),sha256=sha(output),dataset=dataset,passed=True)))


def preflight():
    """Few independent hand-counted/edge fixtures. No producer/data IO."""
    checks=[];p=np.array([[.5,.2,.1,.1,.1],[.5,.1,.2,.1,.1],[.1,.6,.1,.1,.1],[.2,.1,.1,.5,.1]],np.float32);y=np.array([0,0,1,3])
    v,c,pred,det=metric(p,y,.5,True);require(c.tolist()==[2,0,0,2,4,4,2,2,1,1,1,1] and v['system_precision']==v['system_recall']==v['accuracy']==1,'Hand counts');checks.append('hand_counted_SUA_and_fiveway')
    edge=float(.5)+.1*np.spacing(np.float32(.5));require(metric(p,y,edge)[0]['system_recall']==1,'F32 effective threshold');checks.append('F32_threshold_boundary_matches_consumer')
    v,*_=metric(p,y,1.00001,True);require(v['system_precision'] is None and v['system_recall']==0 and v['recall']['cot'] is None and v['recall']['tool'] is None,'Undefined ratios');checks.append('undefined_precision_and_zero_support_NA')
    empty=np.empty((0,5),np.float32);v,*_=metric(empty,np.array([],np.int64),.5,True);require(v['tokens']==0 and v['accuracy'] is None,'Empty');checks.append('empty_natural_keeps_NA')
    arrays={r:np.tile(c,(3,1)) for r in RECORDS};ic,pc=intervals(arrays,314,True);require(len(ic)==36 and len(pc)==54 and all(x['gain_percentage_points']==0 for x in pc),'Paired aliases');checks.append('paired_common_component_draws_zero_differences')
    arrays[RECORDS[0]]=np.array([[1,1,1,1,2,4,1,2,1,1,0,1],[3,1,0,2,5,6,3,3,1,1,1,2],[1,0,2,3,4,6,1,3,1,1,2,2]],float)
    ic,pc=intervals(arrays,315,True);draw=np.random.RandomState(315).randint(3,size=(2000,3));manual=[]
    for sample in draw:
        a=arrays[RECORDS[0]][sample].sum(0);b=arrays[RECORDS[1]][sample].sum(0);manual.append((b[0]/(b[0]+b[2])-a[0]/(a[0]+a[2]))*100)
    item=next(x for x in pc if x['baseline_model']==RECORDS[0] and x['comparison_model']==RECORDS[1] and x['metric']=='system_recall');same(item,dict(zip(['ci95_low','ci95_high'],np.quantile(manual,[.025,.975]).tolist())));checks.append('nonconstant_paired_CI_matches_explicit_2000draw_loop')
    ic,_=intervals({r:c[None,:] for r in RECORDS},314,True);require(all(x['ci_available'] is False and x['ci_unavailable_reason']=='fewer_than_two_dependency_components' for x in ic),'Single component');checks.append('single_component_no_pseudocertainty')
    from sklearn.metrics import roc_auc_score,average_precision_score
    v,*_=metric(p,y,.5);require(abs(v['system_auroc']-roc_auc_score(y==0,p[:,0]))<1e-15 and abs(v['system_average_precision']-average_precision_score(y==0,p[:,0]))<1e-15,'Tied ranks');checks.append('tied_AUC_AP_independent_sklearn_parity')
    import ast
    tree=ast.parse(Path(__file__).read_text());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='load');require(ast.unparse(f.body[0]).endswith('terminal_gate(dataset)'),'Gate order');checks.append('actual_load_terminal_gate_first')
    folder=Path(tempfile.mkdtemp(prefix='feature-review-fresh-complementary-audit-',dir='/tmp'));output=folder/'report.json';value=dict(stage='synthetic_only_independent_completed_result_auditor_preflight',checks=checks,checks_passed=len(checks),auditor_source_sha256=sha(__file__),actual_fresh_data_or_probabilities_read=False,models_tokenizer_or_GPU_used=False,production_modules_imported=False)
    output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(report=str(output),sha256=sha(output),checks=len(checks))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group();action.add_argument('--preflight',action='store_true');action.add_argument('--audit-result',action='store_true');parser.add_argument('--dataset',choices=['c4','natural']);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.preflight:preflight()
    elif args.audit_result:
        require(args.dataset is not None and args.output is not None,'Actual audit requires --dataset and --output.');actual(args.dataset,args.output)
    else:parser.print_help()
