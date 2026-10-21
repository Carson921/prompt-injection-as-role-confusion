"""Independent fit-only complementary-head result auditor. Default help.

--preflight is synthetic CPU/tmp. --actual may run only after root confirms the
scanner terminal0. It never imports production code, loads an LM, uses a GPU,
or converts the full F32 discovery matrix to F64. Native activations were not
persisted: their producer hook/parity evidence is bound, not independently replayed.
"""
import argparse,copy,hashlib,itertools,json,os,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis');BASE=HERE/'outputs/local-gptoss120b'
SOURCE=HERE/'systemness_complementary_head_scan.py';SOURCE_SHA='70db2015b90a496dc465c63631911da81d641614fcf0b1d48b04313a078ce156'
OUT=BASE/'systemness-complementary-head-scan';CACHE=Path('/dev/shm/role-analysis-complementary-head-scan')
PLAN_SHA='0db7b1a351196b5894389fe405f0ca1c93a37d218873b9a0ad119dc603575f70'
HOLD_SHA='585cc7f77029caa6e9993e000b2d6e14a6935f2915d702b777d151d839e89d8e'
FULL_CACHE=BASE/'systemness-expansion/cache/long-full';NAT_CACHE=BASE/'systemness-expansion/cache/natural-transcripts-dev';NAT_FEATURE=Path('/dev/shm/role-analysis-natural-dev-heads')
ORIGINAL=[458,462,2045,460,459,2047,456,997];DOMAINS=['C4-all','C4-tail','public-oasst','public-toxicchat0124'];ROLES=['system','user','cot','assistant','tool']
SALT='role-analysis-complementary-heads-natural-components-v1|20261021|';DIFFERENCE=0.
def require(v,s):
    if not bool(v):raise RuntimeError(s)
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def frame_sha(f,cols):return hashlib.sha256(f[cols].to_csv(index=False).encode()).hexdigest()
def labels(f):
    y=f.role.map(dict(zip(ROLES,range(5))));require(y.notna().all(),'Unknown native role.');return y.to_numpy(np.int64)
def close(a,b,label='',tol=2e-11):
    global DIFFERENCE
    if isinstance(b,dict):
        require(isinstance(a,dict) and set(a)==set(b),'Mapping mismatch: '+label)
        for k,v in b.items():close(a[k],v,label+'/'+str(k),tol)
    elif isinstance(b,(list,tuple)):
        require(isinstance(a,(list,tuple)) and len(a)==len(b),'Sequence mismatch: '+label)
        for i,v in enumerate(b):close(a[i],v,label+'/'+str(i),tol)
    elif b is None or isinstance(b,(bool,int,str)):require(a==b,'Discrete mismatch: '+label)
    else:
        require(a is not None and np.isfinite(a) and np.isfinite(b),'Nonfinite metric: '+label);d=abs(float(a)-float(b));DIFFERENCE=max(DIFFERENCE,d);require(d<=tol,'Numeric mismatch: '+label+' '+str(d))
def processes():
    active=[];names={'systemness_complementary_head_scan.py'}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(x.decode(errors='replace')).name in names for x in args if x):active.append(int(p.name))
    return active
def terminal_gate():
    require(not processes(),'Scanner live: no partial artifact or probability reads.')
    require((OUT/'scan-complete.json').is_file() and (OUT/'status.json').is_file(),'Completion/status absent.')
    require(read(OUT/'status.json')['stage']=='scan_complete','Scanner not terminal complete.')

def binary_metrics(score,truth):
    """Complete score groups, independent clipped ROC polygon integration."""
    score=np.asarray(score);truth=np.asarray(truth,bool);pos=int(truth.sum());neg=len(truth)-pos
    require(score.shape==truth.shape and pos and neg and np.isfinite(score).all(),'Invalid binary unit.')
    order=np.argsort(-score,kind='stable');s=score[order];yy=truth[order];starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    group_p=np.add.reduceat(yy.astype(np.int64),starts);group_n=np.diff(np.r_[starts,len(s)])-group_p
    fp=np.r_[0,np.cumsum(group_n)];tp=np.r_[0,np.cumsum(group_p)];x=fp/neg;t=tp/pos;area=0.
    for p,n,left,begin in zip(group_p,group_n,x[:-1],t[:-1]):
        if n and left<.05:
            width=min(.05-left,n/neg);area+=begin*width+.5*(p/pos)/(n/neg)*width*width
    p=np.clip(score.astype(np.float64),1e-15,1-1e-15)
    return dict(partial_auroc=float(area/.05),auroc=float(np.trapezoid(t,x)),balanced_nll=float(-.5*(np.log(p[truth]).mean()+np.log1p(-p[~truth]).mean())))
def units(rows):
    out=[]
    for source,g in rows.groupby('source_uid',sort=True):
        ix=g.index.to_numpy();y=g.role.eq('system').to_numpy();require(y.any() and (~y).any(),'Source lacks system/other support.')
        if g.domain.iloc[0]=='C4':
            out.append(('C4-all',str(source),ix,y));mask=g.body_offset.ge(512).to_numpy();out.append(('C4-tail',str(source),ix[mask],y[mask]))
        else:out.append(('public-'+g.dataset.iloc[0],str(g.component_id.iloc[0]),ix,y))
    return out
def evaluate(score,groups):
    pieces={d:{} for d in DOMAINS}
    for domain,key,ix,y in groups:pieces[domain].setdefault(key,[]).append(binary_metrics(score[ix],y))
    values={d:{key:{k:float(np.mean([v[k] for v in group])) for k in ['partial_auroc','auroc','balanced_nll']} for key,group in pieces[d].items()} for d in DOMAINS}
    require(all(values.values()),'Missing domain.');return values,{d:{k:float(np.mean([v[k] for v in values[d].values()])) for k in ['partial_auroc','auroc','balanced_nll']} for d in DOMAINS}
def total_score(p,heads):
    total=np.zeros(p.shape[0],np.float64)
    for h in heads:total+=np.asarray(p[:,h],np.float64)
    return (total/len(heads)).astype(np.float32)
def choose(p,groups,original=ORIGINAL,screen=32):
    original_sum=np.zeros(p.shape[0],np.float64)
    for h in original:original_sum+=np.asarray(p[:,h],np.float64)
    baseline_units,baseline=evaluate((original_sum/len(original)).astype(np.float32),groups);marginal=[]
    for h in range(p.shape[1]):
        if h in original:continue
        _,v=evaluate(((original_sum+np.asarray(p[:,h],np.float64))/(len(original)+1)).astype(np.float32),groups);gain={d:v[d]['partial_auroc']-baseline[d]['partial_auroc'] for d in DOMAINS}
        marginal.append(dict(head_id=h,gains=gain,worst_gain=min(gain.values()),tail_gain=gain['C4-tail'],mean_gain=float(np.mean(list(gain.values()))),metrics=v))
    rank=sorted(marginal,key=lambda v:(-v['worst_gain'],-v['tail_gain'],-v['mean_gain'],v['head_id']));require(len(rank)>=screen,'Too few screen candidates.');heads=sorted(v['head_id'] for v in rank[:screen]);pairs=[]
    for a,b in itertools.combinations(heads,2):
        _,v=evaluate(((original_sum+np.asarray(p[:,a],np.float64)+np.asarray(p[:,b],np.float64))/(len(original)+2)).astype(np.float32),groups);gain={d:v[d]['partial_auroc']-baseline[d]['partial_auroc'] for d in DOMAINS}
        pairs.append(dict(head_ids=[a,b],gains=gain,tail_gain=gain['C4-tail'],worst_gain=min(gain.values()),mean_full_auroc_gain=float(np.mean([v[d]['auroc']-baseline[d]['auroc'] for d in DOMAINS])),metrics=v,
            admissible=gain['C4-tail']>=.005-1e-12 and all(gain[d]>=-.001-1e-12 for d in DOMAINS if d!='C4-tail')))
    eligible=sorted([v for v in pairs if v['admissible']],key=lambda v:(-v['tail_gain'],-v['worst_gain'],-v['mean_full_auroc_gain'],v['head_ids']));best=eligible[0] if eligible else None
    result=dict(original_head_ids=list(original),baseline=baseline,screened_head_ids=heads,screened_count=screen,pairs_evaluated=len(pairs),admissible_pairs=len(eligible),selected_pair=best,selected_head_ids=list(original)+best['head_ids'] if best else None,stop_without_downstream_extraction=best is None)
    return result,marginal,pairs,baseline_units

def description(v):
    i,r=v['inputs'],v['rows'];return dict(prompts=len(i),rows=len(r),input_tokens=int(i.length.sum()),inputs_sha256=frame_sha(i,['uid','source_prompt_ix','length','domain']),rows_sha256=frame_sha(r,['sample_ix','uid','token_ix','token_id','role','document_id','component_id']),sample_ix_sha256=arrsha(r.sample_ix.to_numpy()),class_counts=np.bincount(labels(r),minlength=5).tolist())
def make_views(ci,cr,ni,nr,held):
    """Pure fit metadata construction; no filesystem or production helpers."""
    cr=cr.copy();nr=nr.copy();ci=ci.copy();ni=ni.copy()
    cr['body_offset']=cr.token_ix-cr.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
    cr['uid']='long-full:'+cr.prompt_ix.astype(str);cr['domain']='C4';cr['component_id']='c4-doc:'+cr.document_id.astype(str);cr['source_uid']=cr.component_id;cr['source_prompt_ix']=cr.prompt_ix;cr['dataset']='C4'
    nr['uid']='public:'+nr.source_uid;nr['domain']='public';nr['component_id']=nr.dependency_component_id;nr['document_id']=-1;nr['source_prompt_ix']=nr.original_prompt_ix
    ci['uid']='long-full:'+ci.prompt_ix.astype(str);ci['source_prompt_ix']=ci.prompt_ix;ci['domain']='C4'
    ni['uid']='public:'+ni.source_uid;ni['source_prompt_ix']=ni.prompt_ix;ni['domain']='public';views={}
    for phase,isheld,cap in [('statistics',False,128),('holdout',True,256)]:
        a=cr[(cr.document_id%5==0)==isheld].copy().sort_values(['prompt_ix','message_ix','token_ix']);b=nr[nr.component_id.isin(held)==isheld]
        b=pd.concat([g.sort_values('token_ix').iloc[np.linspace(0,len(g)-1,min(len(g),cap),dtype=int)] for _,g in b.groupby(['original_prompt_ix','message_ix'],sort=True)],ignore_index=True)
        r=pd.concat([a,b],ignore_index=True);r['target_ix']=np.arange(len(r),dtype=np.int64);i=pd.concat([ci[ci.uid.isin(a.uid)],ni[ni.uid.isin(b.uid)]],ignore_index=True)
        require(i.uid.is_unique and set(i.uid)==set(r.uid),'Prompt/source membership changed.');views[phase]=dict(inputs=i,rows=r)
    return views

def reconstruct(plan):
    """Independent metadata-only fit filters; no prompt/body/tokenJSON decoding."""
    ci=pd.read_parquet(FULL_CACHE/'inputs.parquet',filters=[('split','=','fit')],columns=['prompt_ix','length','split'])
    cr=pd.read_parquet(FULL_CACHE/'rows.parquet',filters=[('split','=','fit')],columns=['sample_ix','prompt_ix','token_ix','token_id','role','document_id','message_ix','profile','wrapper','split'])
    ni=pd.read_parquet(NAT_CACHE/'inputs.parquet',filters=[('split','=','fit'),('eligible_fit','=',True)],columns=['prompt_ix','length','split','source_uid','eligible_fit'])
    nr=pd.read_parquet(NAT_FEATURE/'rows.parquet',filters=[('split','=','fit')],columns=['sample_ix','original_prompt_ix','token_ix','token_id','role','source_uid','dependency_component_id','family_id','dataset','message_ix','split','eligible_fit','old_generated_collision','cross_split_assistant_collision'])
    require(len(cr)==168960 and cr.document_id.nunique()==132 and len(nr)==63403 and len(ni)==255 and cr.wrapper.eq('native').all() and cr.profile.eq('long-full').all(),'Original132/clean255 fit scope changed.')
    require(nr.eligible_fit.eq(True).all() and not nr.old_generated_collision.any() and not nr.cross_split_assistant_collision.any() and set(nr.role)=={'system','user','assistant'},'Quarantined/foreign natural source entered.')
    comps=sorted(nr.dependency_component_id.unique(),key=lambda c:hashlib.sha256((SALT+c).encode()).hexdigest());held=set(comps[:8]);require(len(comps)==32 and held==set(plan['partition']['holdout_components']),'Fixed whole-component split changed.')
    for k in ['source_uid','family_id']:require(nr.groupby(k).dependency_component_id.nunique().eq(1).all(),'Source/family crosses components.')
    for _,g in cr.groupby(['document_id','role']):require(np.array_equal(np.sort(g.token_ix-g.token_ix.min()),np.linspace(0,1023,256,dtype=int)),'Original uniform256 body grid changed.')
    views=make_views(ci,cr,ni,nr,held)
    for phase in views:close(description(views[phase]),plan['data'][phase],'independent '+phase+' descriptor',tol=0)
    for k in ['uid','component_id','source_uid']:require(not set(views['statistics']['rows'][k])&set(views['holdout']['rows'][k]),'Fit statistics/holdout bridge.')
    require(len(views['holdout']['rows'])==55859 and arrsha(views['holdout']['rows'].sample_ix.to_numpy())==HOLD_SHA,'Exact holdout row/order pin failed.');return views

def moments(path):
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={'n','sum','xx','metadata'},'Unexpected moment cache keys.');a={k:z[k].copy() for k in ['n','sum','xx']};m=json.loads(str(z['metadata']))
    require(a['n'].shape==(2304,5) and a['n'].dtype==np.int64 and a['sum'].shape==(2304,5,64) and a['xx'].shape==(2304,64,64) and a['sum'].dtype==a['xx'].dtype==np.float64
        and (a['n']>=0).all() and np.isfinite(a['sum']).all() and np.isfinite(a['xx']).all(),'Corrupt F64 moments.');return a,m
def refit_lda(a):
    n,s,xx=a['n'],a['sum'],a['xx'];heads,classes,dimensions=s.shape;require((n>0).all(),'LDA missing role.');N=n.sum(1);mean=s/n[:,:,None]
    cov=(xx-np.einsum('hkd,hke,hk->hde',s,s,1./n,optimize=True))/(N-classes)[:,None,None];cov=(cov+cov.transpose(0,2,1))/2
    total=(xx-np.einsum('hd,he,h->hde',s.sum(1),s.sum(1),1./N))/(N-1)[:,None,None];tau=np.trace(cov,axis1=1,axis2=2)/dimensions;tt=np.trace(total,axis1=1,axis2=2)/dimensions;ridge=np.maximum.reduce([tau,1e-6*tt,np.full(heads,1e-12)])
    cov=.9*cov+.1*ridge[:,None,None]*np.eye(dimensions);w=np.linalg.solve(cov,mean.transpose(0,2,1)).transpose(0,2,1);b=-.5*np.einsum('hkd,hkd->hk',mean,w)+np.log(n/N[:,None]);return w,b,ridge,tau,tt

def actual(output):
    terminal_gate();require(Path(output).resolve().is_relative_to(Path('/tmp')),'Auditor writes /tmp only.');bindings={}
    def bind(p,want=None):
        p=Path(p);got=digest(p);require(want is None or got==want,'Artifact SHA differs: '+str(p));bindings[str(p.resolve())]=got;return got
    bind(__file__);bind(SOURCE,SOURCE_SHA);bind(OUT/'plan.json',PLAN_SHA);plan=read(OUT/'plan.json');done=read(OUT/'scan-complete.json');complete_sha=bind(OUT/'scan-complete.json');bind(OUT/'status.json')
    require(done['source_sha256']==SOURCE_SHA and done['plan_sha256']==PLAN_SHA and done['tests_read'] is False and done['primary_changed'] is False and done['final_bank_created'] is False and done['validation_pending'] is True,'Terminal scan identity/scope changed.')
    for name,want in done['artifact_sha256'].items():bind(OUT/name,want)
    base_artifacts={'plan.json','prestream-source-audit.json','statistics-complete.json','lda-diagnostics.json','discovery-complete.json','models/mixed-statistics-lda.npz','model-backend.json','head-selection.json','marginal-head-scores.csv','pair-scores.csv','pair-details.json','hook-validation-statistics.json','hook-validation-holdout.json'}
    optional={'selected-pair-gains-bootstrap.json','selected-pair-unit-metrics.json','C4-position-gains.csv','models/selected10-proxy-lda.npz'}
    require(set(done['artifact_sha256'])==base_artifacts|(optional if done['admissible'] else set()),'Complete artifact coverage changed.')
    sig=plan['signature'];require(sig['source_sha256']==SOURCE_SHA and plan['protocol']['original_head_ids']==ORIGINAL and plan['protocol']['screen']['count']==32 and plan['protocol']['pairs']['count']==496
        and sig['tokenizer_loaded'] is False and sig['source_text_materialized'] is False and sig['validation_inputs_loaded'] is False and sig['tests_read'] is False,'Frozen scanner protocol differs.')
    for n,want in sig['dependencies_sha256'].items():bind(HERE/n,want)
    for p,want in sig['closed_files_sha256'].items():require(not str(p).startswith('/workspace'),'Forbidden failed mount.');bind(p,want)
    require(read(Path(sig['checkpoint_path'])/'config.json')['quantization_config']['quant_method']=='mxfp4','Native checkpoint config changed.')
    views=reconstruct(plan);rows=views['holdout']['rows'];groups=units(rows)
    require(sum(1 for d,_,_,_ in groups if d=='C4-all')==31 and sum(1 for d,_,_,_ in groups if d=='public-oasst')==32 and sum(1 for d,_,_,_ in groups if d=='public-toxicchat0124')==32,'Holdout source counts changed.')
    proof=read(OUT/'prestream-source-audit.json');require(proof['cross_partition_bridges']==0 and proof['fit_only'] is True and proof['validation_inputs_loaded'] is False and proof['tests_read'] is False,'Prestream source guard failed.')
    backend=read(OUT/'model-backend.json');require(backend['native_mxfp4'] is True and backend['parameters_frozen'] is True and backend['expert_backends']==['Mxfp4GptOssExperts'] and len(backend['gpus'])==4 and all('A40' in g for g in backend['gpus']) and backend['checkpoint_path']==sig['checkpoint_path'],'Native frozen4A40 evidence missing.')
    expected_map={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':3,'lm_head':3,**{f'model.layers.{i}':i//9 for i in range(36)}};require(backend['device_map']==expected_map,'Four-device layer map changed.')
    for phase in ['statistics','holdout']:
        h=read(OUT/f'hook-validation-{phase}.json');require(h['hidden_states_bitexact'] is True and h['statistics_disabled'] is True and h['probability_writes_disabled'] is True and h['layers']==list(range(36)) and h['sample_rows']>0,'Producer disabled-hook evidence invalid.')
    stat=read(OUT/'statistics-complete.json');require(stat['plan_sha256']==PLAN_SHA and stat['fit_only'] is True and stat['tests_read'] is False and stat['validation_inputs_loaded'] is False,'Statistics scope changed.')
    bind(CACHE/'statistics-moments.npz',stat['moments_sha256']);mixed,meta=moments(CACHE/'statistics-moments.npz');close(meta,stat['metadata'],'complete moment metadata');close(meta['data'],plan['data']['statistics'],'moment fit rows')
    require(meta['plan_sha256']==PLAN_SHA and meta['class_counts']==plan['data']['statistics']['class_counts'] and meta['old_C4_moment_reuse']==sig['old_C4_moment_reuse'] and np.array_equal(mixed['n'],np.broadcast_to(meta['class_counts'],(2304,5))),'Mixed class-count matrix changed.')
    bind(CACHE/'additional-moments.npz',meta['additional_checkpoint_sha256']);extra,extra_meta=moments(CACHE/'additional-moments.npz')
    addrows=views['statistics']['rows'];addinputs=views['statistics']['inputs']
    if sig['old_C4_moment_reuse']:addrows=addrows[addrows.domain.eq('public')].copy().reset_index(drop=True);addinputs=addinputs[addinputs.domain.eq('public')].copy().reset_index(drop=True)
    order=addinputs.sort_values(['length','uid']).uid.tolist();counts=np.bincount(labels(addrows),minlength=5).tolist();batches=(len(order)+3)//4
    require(extra_meta['plan_sha256']==PLAN_SHA and extra_meta['phase']=='additional-statistics' and extra_meta['next_batch']==extra_meta['total_batches']==batches and extra_meta['rows']==len(addrows)
        and extra_meta['class_counts']==counts and extra_meta['sample_ix_sha256']==arrsha(addrows.sample_ix.to_numpy()) and extra_meta['complete_batch_order_sha256']==canonical(order)
        and np.array_equal(extra['n'],np.broadcast_to(counts,(2304,5))),'Complete additional-moment prefix changed.')
    if sig['old_C4_moment_reuse']:
        bind(sig['old_moment_path'],sig['old_moment_sha256']);previous,oldmeta=moments(sig['old_moment_path'])
        require(oldmeta['plan_sha256']=='94edda02b6752d4a584d2b5155fd34b4cf3e1b599593a681f5651ed8c0912f5c' and oldmeta['rows_written']==129280 and oldmeta['class_counts']==[25856]*5,'Old101 moment source changed.')
        for k in mixed:require(np.array_equal(extra[k]+previous[k],mixed[k]),'Mixed moment additivity failed: '+k)
        del previous
    else:
        for k in mixed:require(np.array_equal(extra[k],mixed[k]),'Full-recompute mixed moment mismatch.')
    del extra
    modelpath=OUT/'models/mixed-statistics-lda.npz';model_sha=bind(modelpath)
    with np.load(modelpath,allow_pickle=False) as z:w=z['weight'].copy();b=z['bias'].copy();mm=json.loads(str(z['metadata']));diag=json.loads(str(z['diagnostics']))
    require(w.shape==(2304,5,64) and b.shape==(2304,5) and w.dtype==b.dtype==np.float64 and np.isfinite(w).all() and np.isfinite(b).all(),'Portable mixed LDA malformed.')
    require(mm==dict(plan_sha256=PLAN_SHA,statistics_complete_sha256=digest(OUT/'statistics-complete.json'),roles=ROLES,heads=2304,head_dim=64,statistics_only=True),'Mixed LDA not frozen on statistics only.')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):rw,rb,ridge,tau,tt=refit_lda(mixed)
    model_error=dict(weight_maximum_absolute_difference=float(np.abs(rw-w).max()),bias_maximum_absolute_difference=float(np.abs(rb-b).max()))
    require(np.allclose(rw,w,rtol=2e-10,atol=2e-8) and np.allclose(rb,b,rtol=2e-10,atol=2e-8),'Independent mixed-moment LDA refit differs.')
    close(diag,read(OUT/'lda-diagnostics.json'),'portable LDA diagnostics');require(diag['zero_within_heads']==int((tau<=0).sum()) and diag['zero_total_heads']==int((tt<=0).sum()) and diag['floor_applied_heads']==int((ridge>tau).sum()),'Degenerate-head diagnostics changed.')
    del mixed,rw,rb
    discovery=read(OUT/'discovery-complete.json');require(discovery['plan_sha256']==PLAN_SHA and discovery['model_sha256']==model_sha and discovery['data']==plan['data']['holdout'] and discovery['actual_probabilities_streamed'] is True
        and discovery['partial_resume'] is False and discovery['validation_inputs_loaded'] is False and discovery['tests_read'] is False and np.isfinite(discovery['cpu_probability_max_difference']) and 0<=discovery['cpu_probability_max_difference']<1e-8,'Actual discovery/parity evidence differs.')
    require(read(CACHE/'initialized.json')==dict(plan_sha256=PLAN_SHA,source_sha256=SOURCE_SHA),'Probability cache owner changed.');bind(CACHE/'initialized.json')
    pp=CACHE/'discovery-system-probabilities.npy';p_sha=bind(pp,discovery['probabilities_sha256']);p=np.load(pp,mmap_mode='r',allow_pickle=False)
    require(p.shape==(55859,2304) and p.dtype==np.float32 and p.flags.c_contiguous and pp.stat().st_size==p.nbytes+p.offset,'F32 probability matrix geometry/file size changed.')
    for start in range(0,len(p),1024):block=p[start:start+1024];require(np.isfinite(block).all() and ((block>=0)&(block<=1)).all(),'Unwritten/invalid discovery score.')
    result,marginal,pairs,base_units=choose(p,groups);saved=read(OUT/'head-selection.json')
    for k,v in result.items():close(saved[k],v,'selection/'+k)
    require(saved['source_sha256']==SOURCE_SHA and saved['plan_sha256']==PLAN_SHA and saved['discovery_sha256']==digest(OUT/'discovery-complete.json') and saved['statistics_model_sha256']==model_sha and saved['head_selection_fit_only'] is True and saved['validation_pending'] is True and saved['tests_read'] is False and saved['primary_changed'] is False,'Selection source/fit scope changed.')
    require(len(marginal)==2296 and len(pairs)==496 and done['admissible']==(result['selected_pair'] is not None) and done['selected_head_ids']==result['selected_head_ids'],'2296/32/496/final completion differs.')
    csv=pd.read_csv(OUT/'marginal-head-scores.csv');mi={v['head_id']:v for v in marginal};require(len(csv)==2296 and csv.head_id.is_unique and set(csv.head_id)==set(mi),'Marginal CSV coverage.')
    for r in csv.to_dict('records'):
        h=int(r['head_id']);v=mi[h];require(r['layer']==h//64 and r['head']==h%64,'Marginal head coordinates changed.')
        for k in ['worst_gain','tail_gain','mean_gain']:close(r[k],v[k],'marginal/'+k)
        for d in DOMAINS:close(r[d+'_partial_auroc_gain'],v['gains'][d],'marginal/'+d)
    close(read(OUT/'pair-details.json'),pairs,'all496pair-details');csv=pd.read_csv(OUT/'pair-scores.csv');pi={tuple(v['head_ids']):v for v in pairs}
    require(len(csv)==496 and not csv.duplicated(['head_a','head_b']).any() and set(zip(csv.head_a,csv.head_b))==set(pi),'Pair CSV coverage.')
    for r in csv.to_dict('records'):
        v=pi[(r['head_a'],r['head_b'])];require(bool(r['admissible'])==v['admissible'],'Pair admissibility CSV differs.')
        for k in ['tail_gain','worst_gain','mean_full_auroc_gain']:close(r[k],v[k],'pairCSV/'+k)
        for d in DOMAINS:close(r[d+'_partial_auroc_gain'],v['gains'][d],'pairCSV/'+d)
    if result['selected_head_ids']:
        selected=result['selected_head_ids'];selected_units,_=evaluate(total_score(p,selected),groups);close(read(OUT/'selected-pair-unit-metrics.json'),dict(baseline=base_units,selected=selected_units),'selected per-unit macros')
        proxy=OUT/'models/selected10-proxy-lda.npz';require(saved['proxy_model_path']==str(proxy) and saved['proxy_model_sha256']==digest(proxy),'Selected proxy model binding differs.')
        with np.load(proxy,allow_pickle=False) as z:
            require(json.loads(str(z['metadata']))==dict(plan_sha256=PLAN_SHA,statistics_only=True,proxy_not_readout=True,roles=ROLES,head_dim=64),'Selected10 proxy metadata changed.')
            require(np.array_equal(z['selected_head_ids'],selected) and np.array_equal(z['weight'],w[selected]) and np.array_equal(z['bias'],b[selected]),'Selected10 portable is not exact mixed-model subset.')
    else:require(saved['proxy_model_path'] is None and saved['proxy_model_sha256'] is None,'No-admissible branch exposed model.')
    terminal_gate()
    for path,want in bindings.items():require(digest(path)==want,'Artifact changed during independent audit: '+path)
    report=dict(stage='actual_completed_complementary_head_scan_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=digest(__file__),scanner_source_sha256=SOURCE_SHA,scanner_plan_sha256=PLAN_SHA,
        scanner_complete_sha256=complete_sha,head_selection_sha256=digest(OUT/'head-selection.json'),probabilities_sha256=p_sha,statistics_moments_sha256=stat['moments_sha256'],mixed_lda_model_sha256=model_sha,holdout_sample_ix_sha256=HOLD_SHA,
        selected_head_ids=result['selected_head_ids'],admissible=result['selected_pair'] is not None,selection=result,marginals_recomputed=2296,pairs_recomputed=496,maximum_metric_difference=DIFFERENCE,independent_LDA_refit=model_error,
        macro_unit_counts={d:len(base_units[d]) for d in DOMAINS},discovery_source_counts={'C4':31,'public-oasst':32,'public-toxicchat0124':32},public_component_counts={'public-oasst':8,'public-toxicchat0124':8},provenance_sha256=bindings,producer_native_MXFP4_4A40_hook_evidence_bound=True,producer_sampled_layer_CPU_GPU_parity_max_error=discovery['cpu_probability_max_difference'],
        native_LM_independently_replayed=False,limitation='Raw sampled LM activations were not persisted. This audit verifies arrays/fit-only metadata/statistical refit/proxy selection and binds producer native hook+sampled parity evidence; it does not independently reproduce LM outputs. Postselection bootstrap and position-bin artifacts are hash-bound only here.',
        probability_matrix_dtype='float32',all_matrix_F64_conversion=False,test_data_or_results_read=False,source_text_materialized=False,LM_loaded=False,gpu_used=False,formal_outputs_written=False)
    output=Path(output);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(output),report_sha256=digest(output),source_sha256=report['auditor_source_sha256'],selected_head_ids=result['selected_head_ids'],admissible=report['admissible'],maximum_metric_difference=DIFFERENCE)))


def preflight():
    """Small synthetic checks; no production artifacts or probability reads."""
    root=Path(tempfile.mkdtemp(prefix='confirmation-review-complementary-result-preflight-'));start_sha=digest(__file__);checks=[]
    def check(name,value):require(value,'Preflight failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except RuntimeError:checks.append(name);return
        raise RuntimeError('Expected refusal: '+name)
    def forbidden(*a,**k):raise LookupError('Synthetic guard forbids artifact/probability access.')
    with patch.dict(globals(),dict(processes=lambda:[123],read=forbidden,digest=forbidden)),patch.object(np,'load',forbidden),patch.object(pd,'read_parquet',forbidden):
        reject('live_scanner_before_any_artifact_or_probability_read',lambda:actual(root/'forbidden.json'))
    fake=root/'not-terminal';fake.mkdir();(fake/'scan-complete.json').write_text('{}');(fake/'status.json').write_text('{"stage":"selecting"}')
    with patch.dict(globals(),dict(OUT=fake,processes=lambda:[])),patch.object(np,'load',forbidden),patch.object(pd,'read_parquet',forbidden):
        reject('nonterminal_status_before_data_or_probability_access',lambda:actual(root/'forbidden.json'))
    for name,s,y,want in [('perfect',[.9,.8,.2,.1],[1,1,0,0],1.),('reverse',[.1,.2,.8,.9],[1,1,0,0],0.),('one_complete_tie',[.5]*4,[1,0,1,0],.025),('leading_vertical',[.9,.5,.5,.5],[1,1,0,0],.5125)]:
        check('hand_pAUC_'+name,abs(binary_metrics(np.asarray(s,np.float32),y)['partial_auroc']-want)<1e-14)
    rows=[]
    for domain,dataset,source,component in [('C4','C4','c4','c4'),('public','oasst','oa','component'),('public','toxicchat0124','tc','component')]:
        for t,role in enumerate(['system','user','system','assistant']):rows.append(dict(domain=domain,dataset=dataset,source_uid=source,component_id=component,role=role,body_offset=512 if t>=2 else 0))
    rows=pd.DataFrame(rows);p=np.full((len(rows),40),.5,np.float32)
    class ColumnOnly:
        shape=p.shape
        def __getitem__(self,key):require(isinstance(key,tuple) and key[0]==slice(None) and isinstance(key[1],(int,np.integer)),'Whole matrix conversion requested.');return p[key]
        def __array__(self,*args,**kwargs):raise RuntimeError('Whole matrix conversion prohibited.')
    result,marginal,pairs,baseline_units=choose(ColumnOnly(),units(rows),original=list(range(8)))
    check('F32_column_only_32_screen_496_pairs_and_no_fallback',len(marginal)==32 and len(pairs)==496 and result['screened_head_ids']==list(range(8,40)) and result['selected_pair'] is None and result['stop_without_downstream_extraction'])
    check('fixed_order_F64_sum_then_F32_average',np.array_equal(total_score(ColumnOnly(),list(range(8))),np.full(len(rows),.5,np.float32)))
    ci=pd.DataFrame([dict(prompt_ix=j,length=30,split='fit') for j in [1,0]]);cr=pd.DataFrame([dict(prompt_ix=d,message_ix=0,token_ix=t,token_id=100+t,document_id=d,role=role,sample_ix=100*d+10*r+k) for d in [1,0] for r,role in enumerate(ROLES) for k,t in enumerate([7,519])])
    # Each role has a separate native prompt in production; use only one role
    # per tiny prompt here to exercise mapping/sort rather than native parsing.
    cr=cr[cr.role.eq('system')].copy()
    ni=pd.DataFrame([dict(prompt_ix=j,length=50,split='fit',source_uid='n'+str(j)) for j in [3,2]])
    nr=pd.DataFrame([dict(original_prompt_ix=j,message_ix=m,token_ix=t,token_id=t,role=role,sample_ix=1000+100*j+10*m+t,source_uid='n'+str(j),dependency_component_id='h' if j==2 else 's',dataset='oasst',family_id='f'+str(j)) for j in [3,2] for m,role in enumerate(['system','user']) for t in [5,2,4]])
    views=make_views(ci,cr,ni,nr,{'h'})
    check('pure_metadata_row_order_and_component_split',all(np.array_equal(v['rows'].target_ix,np.arange(len(v['rows']))) and v['rows'].domain.iloc[0]=='C4' for v in views.values()) and not set(views['holdout']['rows'].source_uid)&set(views['statistics']['rows'].source_uid) and views['holdout']['rows'].token_ix.tolist()==[7,519,2,4,5,2,4,5])
    wrong=copy.deepcopy(description(views['holdout']));wrong['sample_ix_sha256']='0'*64
    reject('sample_order_descriptor_tamper_rejected',lambda:close(wrong,description(views['holdout'])))
    rng=np.random.RandomState(71);x=rng.normal(size=(25,3,4));x[:,0]=0;x[:,1]=np.array([.5,1.,-1.,2.]);y=np.repeat(np.arange(5),5)
    a=dict(n=np.full((3,5),5,np.int64),sum=np.stack([x[y==c].sum(0) for c in range(5)],axis=1),xx=np.einsum('nhd,nhe->hde',x,x))
    w,b,ridge,tau,tt=refit_lda(a);means=np.stack([x[y==c,2].mean(0) for c in range(5)]);centered=x[:,2]-means[y];within=centered.T@centered/20;cov=.9*within+.1*np.trace(within)/4*np.eye(4);expected=np.linalg.solve(cov,means.T).T
    check('independent_centered_covariance_LDA_and_degenerate_heads',np.allclose(w[2],expected,atol=1e-12,rtol=1e-12) and np.isfinite(w).all() and np.isfinite(b).all() and np.all(ridge[:2]==1e-12))
    check('source_unchanged',digest(__file__)==start_sha)
    report=dict(stage='synthetic_CPU_complementary_result_auditor_preflight',audit_passed=True,auditor_source_sha256=start_sha,checks=checks,check_count=len(checks),production_artifacts_read=False,actual_probability_files_read=False,production_module_imported=False,gpu_used=False,LM_loaded=False,actual_run=False)
    path=root/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(path),report_sha256=digest(path),source_sha256=start_sha,checks=len(checks))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--preflight',action='store_true');group.add_argument('--actual',action='store_true');parser.add_argument('--output',default='/tmp/confirmation_review_complementary_head_results_audit.json');args=parser.parse_args()
    if args.preflight:preflight()
    elif args.actual:actual(args.output)
    else:parser.print_help()
if __name__=='__main__':main()
