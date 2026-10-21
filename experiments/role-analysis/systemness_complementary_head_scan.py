"""Fit-only mixed-domain complementary-head discovery, frozen native MXFP4.

Default help. --preflight is synthetic CPU/tmp. --stage declare freezes metadata
without loading text/model; --stage scan requires that plan and runs four A40s.
Only original132 true1024 FIT and255 clean natural FIT sources are eligible.
No validation/test inference, final extra-head feature bank or readout training.
"""
import argparse
from contextlib import contextmanager,ExitStack
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unicodedata

import numpy as np
import pandas as pd
import systemness_data as data
import systemness_long_data as native
import systemness_head_scan as old

OUT=data.runtime.OUT/'systemness-complementary-head-scan'
CACHE=Path('/dev/shm/role-analysis-complementary-head-scan')
PROPOSAL=data.runtime.OUT/'systemness-method-proposals/complementary-two-head-formal-proposal.json'
COUNTS=data.runtime.OUT/'systemness-method-proposals/complementary-heads-metadata-counts.json'
PROPOSAL_SHA='287f51d328da1bba3281313da750adb93af818bb37c9af26cad44d9159aab534'
COUNTS_SHA='6036720e6ebcc63d7ead58cfb8ccb305d6c5592d5da65fe502f06f7f5000a0ce'
OLD_SOURCE_SHA='6095e77d8ecf9cc22fb51169bdd126ec1f69efe9d3b59575b9e4b9094ba29ecd'
OLD_PLAN_SHA='94edda02b6752d4a584d2b5155fd34b4cf3e1b599593a681f5651ed8c0912f5c'
OLD_FIT_SHA='9da4a4f535f32b8f5b88d4f4f5e40f936a5366b09a2a7969a1f9de7fd595017c'
OLD_MOMENTS_SHA='28e30a60f04f1ca2f1aa32f924357b1add2205a2c6eeb7cfc01123ffe5a9ee98'
NAT_OUT=data.runtime.OUT/'systemness-natural-transcripts'
NAT_CACHE=data.CACHE/'natural-transcripts-dev'
NAT_FEATURE_OUT=data.runtime.OUT/'systemness-natural-dev-features'
NAT_FEATURE_CACHE=Path('/dev/shm/role-analysis-natural-dev-heads')
NAT_COMPLETE_SHA='b6c411c920f583a128401c0d787869e2eba17b632136a5cec9613fd227bb9dfd'
NAT_FEATURE_SHA='4f020979dc1e337924d6f8ef6b7a576fd36a7f629c72a85431b8caf68d01f9f4'
ORIGINAL=[458,462,2045,460,459,2047,456,997]
ROLES=['system','user','cot','assistant','tool']
DOMAINS=['C4-all','C4-tail','public-oasst','public-toxicchat0124']
SALT='role-analysis-complementary-heads-natural-components-v1|20261021|'
HEADS,DIM,CLASSES,LAYERS,BATCH=2304,64,5,36,4
CHECKPOINT_INTERVAL=16
DEPENDENCIES=['systemness_complementary_head_scan.py','systemness_head_scan.py','systemness_data.py','systemness_long_data.py','systemness_long_probes.py','systemness_probes.py','run_local_gptoss120b.py']
require=native.require


def read(path):return json.loads(Path(path).read_text())
def sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def frame_sha(frame,columns):return hashlib.sha256(frame[columns].to_csv(index=False).encode()).hexdigest()
def labels(rows):return rows.role.map(dict(zip(ROLES,range(5)))).to_numpy(dtype=np.int64)
def uniform(n,k):return np.linspace(0,n-1,min(n,k),dtype=int)


def immutable(path,value):
    if path.exists():require(read(path)==value,'Immutable artifact differs: '+str(path))
    else:data.atomic_json(path,value)


def protocol():
    return dict(version=1,experiment='fit-only-mixed-domain-complementary-two-heads',original_head_ids=ORIGINAL,
        capture='36x64x64 pre-o_proj, sequential4A40 nativeMXFP4; sampled content only, no raw sequence bank.',
        statistics='Original101 C4 docid%5!=0 at256/role +191 natural sources in24 components at min128/message.',
        holdout='Original31 C4 docid%5==0 at256/role +64 natural sources in8 components, all capped192/message.',
        scope='Only original132 C4 fit and255 clean natural fit; no new256 C4fit, validation or tests.',
        lda=dict(alpha=.1,within_denominator='N-5',priors='Empirical mixed-statistics class counts',
            ridge='.9W+.1max(trace(W)/64,1e-6trace(totalCov)/64,1e-12)I',total_cov_denominator='N-1'),
        domains=DOMAINS,metric='Normalized integral TPR(FPR) over[0,.05], complete ties with linear interpolation. Not sklearn standardized partial AUC.',
        aggregation='Original8 fixed-order F64sum; additions ascending ID; divide8/9/10 then castF32.',
        screen=dict(count=32,order=['maximum worst-domain partialAUROC gain','tail gain','mean gain','smaller headID']),
        pairs=dict(count=496,tail_gain_min=.005,other_domain_loss_max=.001,tolerance=1e-12,
            order=['tail partialAUROC gain','worst-domain gain','mean fullAUROC gain','lexicographically smaller pair'],exact_two=True,no_admissible='Stop; no bank/readout/relaxation.'),
        bootstrap=dict(draws=2000,seed=20261021,C4='Same31 document draws for all/tail',public='Same8 component draws for OA/TC',post_selection='Descriptive optimistic development intervals, not independent significance.'),
        restart=dict(statistics='Atomic full F64 moments + exact complete-batch prefix; no status-only restore.',
            discovery='No partial resume; explicit --restart-discovery overwrites only this owned incomplete matrix after logging prior hash. Completed files immutable.'),
        checks='Disabled hooks bitexact hidden states and no statistics/p writes; GPU/NumPy head-LDA probability parity.',
        source_guard='Whole source/family/repeat components; metadata full/raw/normalized/sourceURL first, then selected native full/normalized/prefix192 beforeLM; cross-partition bridge halts without moving/dropping sources.',
        resources=dict(moments_MiB=77.712890625,discovery_shape=[55859,2304],discovery_bytes=514796544,
            shm_min_free_bytes=int(1.5*2**30),reserve_bytes=512*2**20,all_F64_probability_matrix_forbidden=True),
        model='Frozen original nativeMXFP4,36layers64heads64dim,4A40; no training or changed weights.',
        limitation='One fit-development partition;32screen+496pair selection can overfit31documents/8components. Proxy gains are not validation F1 or downstream model gains.',
        logical_reader_scope='Only selected fit rows/text materialize. Whole-container byte hashes and metadata may cover stored validation rows; Parquet physical blocks may contain unselected validation bytes. Mixed token JSON never decoded.',
        validation_inputs_loaded=False,tests_read=False,final_feature_bank_created=False,primary_changed=False)


def sample_natural(rows,cap):
    pieces=[]
    for _,group in rows.groupby(['original_prompt_ix','message_ix'],sort=True):
        group=group.sort_values('token_ix');pieces.append(group.iloc[uniform(len(group),cap)])
    return pd.concat(pieces).reset_index(drop=True)


def partition_metadata(c4_inputs,c4_rows,nat_inputs,nat_rows,*,expected=None):
    """All arguments are projected metadata, without prompt/content strings."""
    require(c4_inputs.split.eq('fit').all() and c4_rows.split.eq('fit').all() and nat_inputs.split.eq('fit').all() and nat_rows.split.eq('fit').all(),'Discovery source escaped original fit.')
    require(c4_rows.profile.eq('long-full').all() and c4_rows.wrapper.eq('native').all() and set(c4_rows.role)==set(ROLES),'C4 native ownership changed.')
    require(nat_rows.eligible_fit.eq(True).all() and nat_inputs.eligible_fit.eq(True).all() and not nat_rows.old_generated_collision.any()
        and not nat_rows.cross_split_assistant_collision.any() and set(nat_rows.role)=={'system','user','assistant'},'Natural clean fit/quarantine ownership changed.')
    comps=sorted(nat_rows.dependency_component_id.unique(),key=lambda x:hashlib.sha256((SALT+x).encode()).hexdigest());held=set(comps[:8]);require(len(comps)==32,'Expected32 clean fit dependency components.')
    for key in ['source_uid','family_id']:
        require(nat_rows.groupby(key).dependency_component_id.nunique().eq(1).all(),'Existing source/family bridges components: '+key)
    c4=c4_rows.copy();c4['body_offset']=c4.token_ix-c4.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
    grid=uniform(1024,256)
    for _,g in c4.groupby(['document_id','role']):require(np.array_equal(np.sort(g.body_offset),grid),'Original256 C4 grid changed.')
    c4['uid']='long-full:'+c4.prompt_ix.astype(str);c4['domain']='C4';c4['component_id']='c4-doc:'+c4.document_id.astype(str);c4['source_uid']=c4.component_id
    c4['source_prompt_ix']=c4.prompt_ix;c4['dataset']='C4'
    public=nat_rows.copy();public['uid']='public:'+public.source_uid;public['domain']='public';public['component_id']=public.dependency_component_id;public['document_id']=-1;public['source_prompt_ix']=public.original_prompt_ix
    c4_inputs=c4_inputs.copy();c4_inputs['uid']='long-full:'+c4_inputs.prompt_ix.astype(str);c4_inputs['source_prompt_ix']=c4_inputs.prompt_ix;c4_inputs['domain']='C4'
    nat_inputs=nat_inputs.copy();nat_inputs['uid']='public:'+nat_inputs.source_uid;nat_inputs['source_prompt_ix']=nat_inputs.prompt_ix;nat_inputs['domain']='public'
    frames={}
    for phase,discovery,cap in [('statistics',False,128),('holdout',True,256)]:
        a=c4[(c4.document_id%5==0)==discovery].copy();b=public[public.component_id.isin(held)==discovery].copy();b=sample_natural(b,cap)
        a=a.sort_values(['prompt_ix','message_ix','token_ix']).reset_index(drop=True);rows=pd.concat([a,b],ignore_index=True);rows['target_ix']=np.arange(len(rows),dtype=np.int64)
        inputs=pd.concat([c4_inputs[c4_inputs.uid.isin(a.uid)],nat_inputs[nat_inputs.uid.isin(b.uid)]],ignore_index=True)
        require(inputs.uid.is_unique and set(inputs.uid)==set(rows.uid),'Metadata prompt/source membership mismatch.')
        frames[phase]=dict(inputs=inputs,rows=rows)
    for key in ['uid','component_id','source_uid']:
        require(not set(frames['statistics']['rows'][key])&set(frames['holdout']['rows'][key]),'Statistics/holdout bridge: '+key)
    if expected:
        require(held==set(expected['algorithm']['natural_head_holdout_components']),'Fixed8 held-out components changed.')
        for phase,key in [('statistics','statistics_fit'),('holdout','head_holdout')]:
            for domain,label in [('C4','C4'),('public','public')]:
                f=frames[phase];rows=f['rows'][f['rows'].domain.eq(domain)];inputs=f['inputs'][f['inputs'].domain.eq(domain)];want=expected['domains'][label][key]
                require(len(rows)==want['sampled_rows'] and rows.source_uid.nunique()==want['source_units'] and len(inputs)==want['prompts']
                    and int(inputs.length.sum())==want['full_nonpadding_tokens'] and sha(rows.sample_ix.to_numpy())==want['sample_ix_sha256']
                    and {r:int(rows.role.eq(r).sum()) for r in ROLES}==want['role_rows'],'Published partition/density/count/sample identity changed: '+phase+'/'+domain)
    return frames,dict(holdout_components=sorted(held),statistics_components=sorted(set(comps)-held),component_split_fit_only=True)


def metadata_views():
    c4_inputs=pd.read_parquet(native.CACHE/'inputs.parquet',filters=[('split','=','fit')],columns=['prompt_ix','length','split'])
    c4_rows=pd.read_parquet(native.CACHE/'rows.parquet',filters=[('split','=','fit')])
    columns=['prompt_ix','length','split','source_uid','eligible_fit']
    ni=pd.read_parquet(NAT_CACHE/'inputs.parquet',filters=[('split','=','fit'),('eligible_fit','=',True)],columns=columns)
    nr=pd.read_parquet(NAT_FEATURE_CACHE/'rows.parquet',filters=[('split','=','fit')])
    require(len(c4_rows)==168960 and c4_rows.document_id.nunique()==132 and len(ni)==255 and len(nr)==63403,'Fixed original132/clean255 fit scope changed.')
    return partition_metadata(c4_inputs,c4_rows,ni,nr,expected=read(COUNTS))


def descriptor(view):
    inputs,rows=view['inputs'],view['rows']
    return dict(prompts=len(inputs),rows=len(rows),input_tokens=int(inputs.length.sum()),
        inputs_sha256=frame_sha(inputs,['uid','source_prompt_ix','length','domain']),
        rows_sha256=frame_sha(rows,['sample_ix','uid','token_ix','token_id','role','document_id','component_id']),
        sample_ix_sha256=sha(rows.sample_ix.to_numpy()),class_counts=np.bincount(labels(rows),minlength=5).tolist())


def signature():
    """Explicit metadata/cache hashes only; no global verifier or text decode."""
    files={};
    def bind(path,want=None):
        got=data.digest(path);require(want is None or got==want,'Closed metadata SHA changed: '+str(path));files[str(path)]=got;return read(path) if Path(path).suffix=='.json' else got
    bind(PROPOSAL,PROPOSAL_SHA);bind(COUNTS,COUNTS_SHA);bind(Path(old.__file__),OLD_SOURCE_SHA)
    oldplan=bind(old.OUT/'plan.json',OLD_PLAN_SHA);fitdone=bind(old.OUT/'fit-complete.json',OLD_FIT_SHA)
    require(fitdone['moments_sha256']==OLD_MOMENTS_SHA and fitdone['tests_read'] is False and fitdone['rows']==129280,'Old101 C4 moments provenance changed.')
    for name in ['plan.json','data-complete.json','document-keys.json']:bind(native.OUT/name)
    longdone=read(native.OUT/'data-complete.json')
    for name in ['inputs.parquet','rows.parquet']:bind(native.CACHE/name,longdone['cache_sha256'][name])
    dev=bind(NAT_OUT/'data-complete.json',NAT_COMPLETE_SHA);nf=bind(NAT_FEATURE_OUT/'extraction-complete.json',NAT_FEATURE_SHA)
    require(dev['test_assistant_decoded'] is False and dev['classifier_probabilities_read'] is False and dev['gpu_work_run'] is False
        and dev['coverage']['independent_native_labels_match'] is True and nf['all_rows_written'] is True and nf['all_finite'] is True and nf['rows']==80116,'Prepared natural dev-only provenance changed.')
    for name in ['inputs.parquet','rows.parquet']:bind(NAT_CACHE/name,dev['cache_sha256'][name])
    for name in ['dev-ledger.json','duplicate-audit.json','coverage.json']:bind(NAT_OUT/name,dev['metadata_sha256'][name])
    bind(NAT_FEATURE_CACHE/'rows.parquet',nf['rows_sha256'])
    checkpoint=Path(oldplan['model_path']);require_checkpoint_path(checkpoint)
    for name,want in [('config.json',oldplan['config_sha256']),('verified-weights.json',oldplan['verified_weights_sha256'])]:bind(checkpoint/name,want)
    config=read(checkpoint/'config.json');require(config['quantization_config']['quant_method']=='mxfp4' and (config['num_hidden_layers'],config['num_attention_heads'],config['head_dim'])==(36,64,64),'Native MXFP4 head layout changed.')
    for name,want in oldplan['tokenizer_sha256'].items():bind(checkpoint/name,want)
    bind(data.runtime.OUT/'checkpoint-cache-verification.json')
    reuse=(old.CACHE/'fit-moments.npz').exists()
    if reuse:bind(old.CACHE/'fit-moments.npz',OLD_MOMENTS_SHA)
    return dict(source_sha256=data.digest(__file__),dependencies_sha256={n:data.digest(data.HERE/n) for n in DEPENDENCIES},closed_files_sha256=files,
        checkpoint_path=str(checkpoint),old_C4_moment_reuse=reuse,old_moment_path=str(old.CACHE/'fit-moments.npz'),
        old_moment_sha256=OLD_MOMENTS_SHA,old_fit_complete=fitdone,kernel_revision=oldplan['kernel_revision'],
        packages={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','numpy','scipy','pandas','pyarrow']},
        tokenizer_loaded=False,source_text_materialized=False,validation_inputs_loaded=False,tests_read=False)


def fingerprint_audit(entries):
    index={}
    for kind,key,phase,unit in entries:index.setdefault((kind,key),set()).add((phase,unit))
    bridges=[(kind,key,sorted(units)) for (kind,key),units in index.items() if len({p for p,u in units})>1]
    require(not bridges,'Existing metadata duplicate bridge across statistics/head-holdout; stop without changing fixed sources: '+str(bridges[:3]))
    return dict(comparable_fingerprints=len(index),cross_partition_bridges=0,index_sha256=canonical([(k,d,sorted(v)) for (k,d),v in sorted(index.items())]),
        key_types=sorted({k for k,d in index}),raw_text_materialized=False,semantic_dedupe=False)


def metadata_duplicate_audit(views):
    units={};documents={}
    for phase,view in views.items():
        for row in view['rows'][view['rows'].domain.eq('public')].drop_duplicates('source_uid').itertuples():units[row.source_uid]=(phase,row.component_id)
        for row in view['rows'][view['rows'].domain.eq('C4')].drop_duplicates('document_id').itertuples():documents[int(row.document_id)]=(phase,row.component_id)
    entries=[]
    for row in read(native.OUT/'document-keys.json'):
        if row['split']!='fit' or row['document_id'] not in documents:continue
        phase,unit=documents[row['document_id']]
        for key in ['body_sha256','full_body_sha256']:entries.append(('full',row[key],phase,unit))
        entries.append(('sourceURL',row['source_url'],phase,unit))
    # The ledger stores small hashes/provenance, never response strings. Only
    # fixed clean FIT source records are consumed logically from this container.
    for row in read(NAT_OUT/'dev-ledger.json')['records']:
        if row['split']!='fit' or row['source_uid'] not in units:continue
        require(row['eligible_fit'] is True,'Non-clean natural source entered discovery metadata.')
        phase,unit=units[row['source_uid']]
        for message in row['messages']:
            for key in ['raw_sha256','clean_sha256','capped_sha256','serialized_body_sha256']:entries.append(('full',message[key],phase,unit))
            entries.append(('normalized',message['normalized_raw_sha256'],phase,unit))
    audit=fingerprint_audit(entries);audit['public_source_URL_comparison']='No comparable public document sourceURL is stored; no URL extraction from response text.'
    return audit


def plan_value():
    sig=signature();views,partition=metadata_views()
    return dict(protocol=protocol(),signature=sig,data={k:descriptor(v) for k,v in views.items()},partition=partition,metadata_duplicate_audit=metadata_duplicate_audit(views))


def declare():
    value=plan_value()
    if not (OUT/'plan.json').exists():require(not CACHE.exists(),'New scanner cache predates metadata declaration; inspect explicitly.')
    OUT.mkdir(parents=True,exist_ok=True);immutable(OUT/'plan.json',value)
    return value


def verify_plan(plan):
    require(plan['protocol']==protocol() and plan['signature']==signature(),'Frozen scanner source/checkpoint/metadata changed.')
    views,partition=metadata_views();require(plan['data']=={k:descriptor(v) for k,v in views.items()} and plan['partition']==partition and plan['metadata_duplicate_audit']==metadata_duplicate_audit(views),'Fixed fit source partitions or comparable duplicate keys changed.')
    return views


def require_checkpoint_path(checkpoint):
    require(str(checkpoint)=='/dev/shm/role-analysis-gptoss120b','Only fixed verified shared-memory checkpoint permitted before any path access.')


def materialize(views,checkpoint):
    """Arrow filters selected original FIT prompts before Python text objects."""
    require_checkpoint_path(checkpoint)
    import pyarrow.dataset as ds
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,add_eos_token=False,add_bos_token=False,padding_side='left')
    bundles={};fingerprints={};urls={int(r['document_id']):r['source_url'] for r in read(native.OUT/'document-keys.json') if r['split']=='fit'}
    for phase,view in views.items():
        inputs=view['inputs'].copy();texts={}
        for domain,cache in [('C4',native.CACHE),('public',NAT_CACHE)]:
            wanted=inputs[inputs.domain.eq(domain)]
            predicate=ds.field('prompt_ix').isin(wanted.source_prompt_ix.tolist()) & (ds.field('split')=='fit')
            if domain=='public':predicate=predicate & (ds.field('eligible_fit')==True)
            selected=ds.dataset(cache/'inputs.parquet',format='parquet').to_table(filter=predicate,columns=['prompt_ix','prompt']).to_pandas().set_index('prompt_ix')
            require(selected.index.is_unique and set(selected.index)==set(wanted.source_prompt_ix),'Selected fit-only prompt filter changed.')
            for r in wanted.itertuples():texts[r.uid]=selected.loc[r.source_prompt_ix,'prompt']
        inputs['prompt']=inputs.uid.map(texts);bundle=old.Bundle(inputs,view['rows'],tokenizer)
        # Full/native bodies are used only for pre-stream cross-partition dedupe.
        for entry in inputs.itertuples():
            ids=bundle.tokens[entry.uid];roles=native.parse_native_labels(ids,tokenizer);positions=sorted(roles);pieces=[];part=[];last=None
            for pos in positions:
                if part and (pos!=last+1 or roles[pos]!=roles[last]):pieces.append(part);part=[]
                part.append(pos);last=pos
            if part:pieces.append(part)
            member=view['rows'][view['rows'].uid.eq(entry.uid)].iloc[0];unit=member.component_id
            for indices in pieces:
                body=tokenizer.decode([ids[i] for i in indices],skip_special_tokens=False);normal=' '.join(unicodedata.normalize('NFKC',body).casefold().split())
                prefix=tokenizer.decode([ids[i] for i in indices[:192]],skip_special_tokens=False)
                for kind,value in [('full',body),('normalized',normal),('prefix192',prefix)]:fingerprints.setdefault((kind,data.text_digest(value)),set()).add((phase,unit))
            if member.domain=='C4':fingerprints.setdefault(('sourceURL',urls[int(member.document_id)]),set()).add((phase,unit))
        bundles[phase]=bundle
    bridges=[(kind,key,sorted(units)) for (kind,key),units in fingerprints.items() if len({p for p,u in units})>1]
    require(not bridges,'Existing fit cross-partition duplicate bridge; stop without moving/dropping/replacing sources: '+str(bridges[:3]))
    proof=dict(comparable_key_types=['full','normalized','prefix192','C4-sourceURL'],fingerprints=len(fingerprints),cross_partition_bridges=0,
        fingerprints_sha256=canonical([(k,d,sorted(v)) for (k,d),v in sorted(fingerprints.items())]),
        source_URL_limit='C4 exact sourceURL only; public transcript source metadata has no comparable document URL. No semantic dedupe.',
        fit_only=True,validation_inputs_loaded=False,tests_read=False)
    return bundles,proof


def binary_scores(score,truth,alpha=.05):
    """Own complete-tie ROC integral, plus class-balanced binary NLL."""
    score=np.asarray(score);truth=np.asarray(truth,bool);require(score.ndim==1 and len(score)==len(truth) and len(score)>0 and np.isfinite(score).all() and ((score>=0)&(score<=1)).all(),'Invalid system probability vector.')
    pos=int(truth.sum());neg=int((~truth).sum());require(pos>0 and neg>0,'Source unit lacks binary support.')
    order=np.argsort(-score,kind='stable');s=score[order];y=truth[order];ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y,dtype=np.int64)[ends];fp=ends+1-tp;x=np.r_[0.,fp/neg];t=np.r_[0.,tp/pos]
    area=0.
    for i in range(1,len(x)):
        if x[i-1]>=alpha:break
        stop=min(float(x[i]),alpha)
        if stop>x[i-1]:
            end=t[i-1]+(t[i]-t[i-1])*(stop-x[i-1])/(x[i]-x[i-1]);area+=(stop-x[i-1])*(t[i-1]+end)/2
    p=np.clip(score.astype(np.float64),1e-15,1-1e-15)
    return dict(partial_auroc=float(area/alpha),auroc=float(np.trapezoid(t,x)),balanced_nll=float(-.5*(np.log(p[truth]).mean()+np.log1p(-p[~truth]).mean())))


def unit_scores(rows,score):
    values={d:{} for d in DOMAINS}
    for source,g in rows.groupby('source_uid',sort=True):
        ix=g.index.to_numpy();truth=g.role.eq('system').to_numpy()
        if g.domain.iloc[0]=='C4':
            values['C4-all'][str(source)]=binary_scores(score[ix],truth);tail=g.body_offset.ge(512).to_numpy();values['C4-tail'][str(source)]=binary_scores(score[ix[tail]],truth[tail])
        else:
            domain='public-'+g.dataset.iloc[0];component=g.component_id.iloc[0];values[domain].setdefault(component,[]).append(binary_scores(score[ix],truth))
    for domain in DOMAINS[2:]:values[domain]={c:{k:float(np.mean([v[k] for v in items])) for k in ['partial_auroc','auroc','balanced_nll']} for c,items in values[domain].items()}
    require(all(values[d] for d in DOMAINS),'Missing fixed discovery domain.')
    return values,{d:{k:float(np.mean([v[k] for v in values[d].values()])) for k in ['partial_auroc','auroc','balanced_nll']} for d in DOMAINS}


def aggregate(probability,heads):
    total=np.zeros(len(probability),np.float64)
    for head in heads:total+=np.asarray(probability[:,head],np.float64)
    return (total/len(heads)).astype(np.float32)


def choose_pair(rows,probability,*,original=ORIGINAL,screen=32):
    require(probability.shape[0]==len(rows) and len(set(original))==len(original) and probability.dtype==np.float32,'Discovery row/head matrix changed.')
    base_score=aggregate(probability,original);base_units,baseline=unit_scores(rows,base_score);oldsum=np.zeros(len(rows),np.float64)
    for head in original:oldsum+=np.asarray(probability[:,head],np.float64)
    marginal=[]
    for head in range(probability.shape[1]):
        if head in original:continue
        score=((oldsum+np.asarray(probability[:,head],np.float64))/(len(original)+1)).astype(np.float32);_,value=unit_scores(rows,score)
        gains={d:value[d]['partial_auroc']-baseline[d]['partial_auroc'] for d in DOMAINS}
        marginal.append(dict(head_id=head,gains=gains,worst_gain=min(gains.values()),tail_gain=gains['C4-tail'],mean_gain=float(np.mean(list(gains.values()))),metrics=value))
    require(len(marginal)>=screen,'Insufficient complement heads for fixed screen.')
    ranked=sorted(marginal,key=lambda v:(-v['worst_gain'],-v['tail_gain'],-v['mean_gain'],v['head_id']));selected=sorted(v['head_id'] for v in ranked[:screen]);pairs=[]
    for a,b in itertools.combinations(selected,2):
        score=((oldsum+np.asarray(probability[:,a],np.float64)+np.asarray(probability[:,b],np.float64))/(len(original)+2)).astype(np.float32);_,value=unit_scores(rows,score)
        gains={d:value[d]['partial_auroc']-baseline[d]['partial_auroc'] for d in DOMAINS};tail=gains['C4-tail'];other=[gains[d] for d in DOMAINS if d!='C4-tail']
        pairs.append(dict(head_ids=[a,b],gains=gains,tail_gain=tail,worst_gain=min(gains.values()),mean_full_auroc_gain=float(np.mean([value[d]['auroc']-baseline[d]['auroc'] for d in DOMAINS])),metrics=value,
            admissible=tail>=.005-1e-12 and min(other)>=-.001-1e-12))
    eligible=sorted([v for v in pairs if v['admissible']],key=lambda v:(-v['tail_gain'],-v['worst_gain'],-v['mean_full_auroc_gain'],v['head_ids']))
    chosen=eligible[0] if eligible else None
    return dict(original_head_ids=list(original),baseline=baseline,screened_head_ids=selected,screened_count=screen,pairs_evaluated=len(pairs),
        admissible_pairs=len(eligible),selected_pair=chosen,selected_head_ids=list(original)+chosen['head_ids'] if chosen else None,
        stop_without_downstream_extraction=chosen is None),marginal,pairs,base_units


def descriptive_bootstrap(base,chosen,draws=2000,seed=20261021):
    require(all(set(base[d])==set(chosen[d]) for d in DOMAINS),'Paired source units changed.')
    require(set(base['C4-all'])==set(base['C4-tail']) and set(base['public-oasst'])==set(base['public-toxicchat0124']),'Shared document/component draw IDs differ.')
    rng=np.random.RandomState(seed);out={}
    for domains in [DOMAINS[:2],DOMAINS[2:]]:
        ids=sorted(base[domains[0]]);indices=rng.randint(0,len(ids),(draws,len(ids)))
        for domain in domains:
            out[domain]={}
            for key in ['partial_auroc','auroc','balanced_nll']:
                differences=np.asarray([chosen[domain][i][key]-base[domain][i][key] for i in ids]);values=differences[indices].mean(1)
                out[domain][key]=dict(mean_gain=float(differences.mean()),percentile95=[float(v) for v in np.quantile(values,[.025,.975])],paired_units=len(ids),draws=draws,positive_is_better=key!='balanced_nll')
    return dict(intervals=out,post_selection_optimistic=True,independent_significance=False,token_bootstrap=False)


def resource_guard():
    require(CACHE.parent==Path('/dev/shm'),'Scanner scratch must stay in ignored shared memory.')
    require(shutil.disk_usage(CACHE.parent).free>=int(1.5*2**30),'Need1.5GiB free shared memory including512MiB reserve; existing banks remain intact.')


def no_competing_jobs():
    names={Path(__file__).name,'systemness_head_scan.py','systemness_head_extract.py','systemness_natural_dev_extract.py','systemness_long_mix_residual.py','systemness_binary_residual.py','systemness_natural_fit_probes.py','systemness_aug_confirmation_infer.py'};active=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(a.decode(errors='replace')).name in names for a in args if a):active.append(int(entry.name))
    require(not active,'Concurrent model/readout job live: '+str(active))


@contextmanager
def run_lock():
    import fcntl
    with (OUT/'.run.lock').open('a') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Another scanner owns this output.') from e
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


@contextmanager
def model_context(plan):
    """Own loader: no writes or reassignment to any frozen runner globals."""
    import torch
    from transformers import AutoModelForCausalLM,Mxfp4Config
    from transformers.integrations import hub_kernels
    from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
    import accelerate.hooks
    import accelerate.utils.operations as operations
    checkpoint=plan['signature']['checkpoint_path']
    require_checkpoint_path(checkpoint)
    require(torch.cuda.device_count()==4 and all('A40' in torch.cuda.get_device_name(i) for i in range(4)),'Exactly4 visible A40s required.')
    require(all(torch.cuda.mem_get_info(i)[0]>=22*2**30 for i in range(4)),'Insufficient free assigned GPU memory; no process is killed/reset.')
    original_get=hub_kernels.get_kernel;original_validate=Mxfp4HfQuantizer.validate_environment;original_send=operations.send_to_device;original_hook_send=accelerate.hooks.send_to_device
    def pinned(name,*args,**kwargs):
        if name=='kernels-community/gpt-oss-triton-kernels':kwargs['revision']=plan['signature']['kernel_revision']
        return original_get(name,*args,**kwargs)
    def strict(self,*args,**kwargs):
        original_validate(self,*args,**kwargs);require(not self.quantization_config.dequantize,'Native MXFP4 required; no BF16 dequantization.')
    def staged(tensor,device,non_blocking=False,skip_keys=None):
        if isinstance(tensor,torch.Tensor) and tensor.device.type=='cuda':
            target=torch.device(f'cuda:{device}' if isinstance(device,int) else device)
            if target.type=='cuda' and tensor.device!=target:return tensor.cpu().to(target,non_blocking=False)
        return original_send(tensor,device,non_blocking=non_blocking,skip_keys=skip_keys)
    device_map={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':3,'lm_head':3};device_map.update({f'model.layers.{i}':i//9 for i in range(36)})
    model=None
    try:
        hub_kernels.get_kernel=pinned;Mxfp4HfQuantizer.validate_environment=strict;operations.send_to_device=staged;accelerate.hooks.send_to_device=staged
        model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,dtype=torch.bfloat16,quantization_config=Mxfp4Config(dequantize=False),device_map=device_map,attn_implementation='eager',use_kernels=False).eval().requires_grad_(False)
        backends={type(layer.mlp.experts).__name__ for layer in model.model.layers}
        require(backends=={'Mxfp4GptOssExperts'} and not model.hf_quantizer.quantization_config.dequantize and not any(p.requires_grad for p in model.parameters()),'Model backend/quantization/freeze differs.')
        immutable(OUT/'model-backend.json',dict(expert_backends=sorted(backends),device_map=model.hf_device_map,checkpoint_path=checkpoint,native_mxfp4=True,parameters_frozen=True,
            gpu_transfer='Synchronous CPU staging between different CUDA devices',gpus=[torch.cuda.get_device_name(i) for i in range(4)],compute_capabilities=[list(torch.cuda.get_device_capability(i)) for i in range(4)]))
        yield model
    finally:
        hub_kernels.get_kernel=original_get;Mxfp4HfQuantizer.validate_environment=original_validate;operations.send_to_device=original_send;accelerate.hooks.send_to_device=original_hook_send
        if model is not None:del model
        gc.collect()


def forward(model,collector,batch,layers,device='cuda:0'):
    collector.begin(batch)
    output=model.model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device),use_cache=False).last_hidden_state
    require(collector.seen==set(layers),'Sequential hook missed or repeated a layer.')
    return output


@contextmanager
def checked_hooks(model,collector,bundle,name,device='cuda:0'):
    import torch
    sample=bundle.collate(bundle.batches()[0]);handles=[];layers=list(range(36))
    with torch.inference_mode():
        before=model.model(input_ids=sample['input_ids'].to(device),attention_mask=sample['attention_mask'].to(device),use_cache=False).last_hidden_state.cpu()
        try:
            for layer in layers:handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(collector.hook(layer)))
            collector.enabled=False;after=forward(model,collector,sample,layers,device).cpu()
            require(torch.equal(before,after) and bool(torch.isfinite(after).all()),'Disabled hook changed final hidden states.')
            require(collector.moments is None or not collector.moments.blocks,'Disabled hook accumulated statistics.')
            if collector.probabilities is not None:require(np.isnan(collector.probabilities[sample['targets']]).all(),'Disabled hook wrote discovery probabilities.')
            immutable(OUT/f'hook-validation-{name}.json',dict(hidden_states_bitexact=True,statistics_disabled=True,probability_writes_disabled=True,layers=layers,sample_rows=len(sample['targets'])))
            del before,after;collector.enabled=True;yield layers
        finally:
            for handle in handles:handle.remove()


def sub_bundle(bundle,domain):
    result=object.__new__(old.Bundle);result.inputs=bundle.inputs[bundle.inputs.domain.eq(domain)].copy().reset_index(drop=True)
    result.rows=bundle.rows[bundle.rows.domain.eq(domain)].copy().reset_index(drop=True);result.rows['target_ix']=np.arange(len(result.rows),dtype=np.int64)
    result.tokens={u:bundle.tokens[u] for u in result.inputs.uid};result.lookup={u:g for u,g in result.rows.groupby('uid',sort=False)}
    return result


def prefix(bundle,next_batch,phase,plan_sha=None):
    batches=bundle.batches();require(0<=next_batch<=len(batches),'Out-of-range moment prefix.')
    uids=[u for part in batches[:next_batch] for u in part.uid];rows=bundle.rows[bundle.rows.uid.isin(uids)]
    return dict(plan_sha256=plan_sha or data.digest(OUT/'plan.json'),phase=phase,next_batch=next_batch,total_batches=len(batches),rows=len(rows),
        class_counts=np.bincount(labels(rows),minlength=5).tolist(),sample_ix_sha256=sha(rows.sample_ix.to_numpy()),
        complete_batch_order_sha256=canonical(uids),bundle=bundle.descriptor())


def reused_moments(bundle,plan):
    require(plan['signature']['old_C4_moment_reuse'],'Old moment reuse not declared.')
    path=Path(plan['signature']['old_moment_path']);require(data.digest(path)==OLD_MOMENTS_SHA,'Old101 cached moment bytes changed.')
    moments,metadata=old.load_moments(path);done=plan['signature']['old_fit_complete'];desc=bundle.descriptor()
    expected=dict(plan_sha256=OLD_PLAN_SHA,phase='fit',next_batch=len(bundle.batches()),rows_written=len(bundle.rows),class_counts=[101*256]*5,data=desc)
    require(done['data']==desc and metadata==expected and np.array_equal(moments.n,np.full((2304,5),25856,np.int64)),'Old101 moment metadata/count/density/source replay differs.')
    return moments


def statistics(get_model,bundle,plan,previous=None):
    done_path=OUT/'statistics-complete.json';path=CACHE/'statistics-moments.npz';checkpoint=CACHE/'additional-moments.npz'
    c4=sub_bundle(bundle,'C4');reuse=plan['signature']['old_C4_moment_reuse'];additional=sub_bundle(bundle,'public') if reuse else bundle
    if done_path.exists():
        done=read(done_path);require(done['plan_sha256']==data.digest(OUT/'plan.json') and data.digest(path)==done['moments_sha256'],'Completed mixed moments changed.')
        moments,meta=old.load_moments(path);require(meta==done['metadata'] and meta['plan_sha256']==data.digest(OUT/'plan.json') and meta['data']==plan['data']['statistics'] and meta['old_C4_moment_reuse']==reuse
            and meta['additional_checkpoint_sha256']==data.digest(checkpoint) and np.array_equal(moments.n,np.broadcast_to(plan['data']['statistics']['class_counts'],moments.n.shape)),'Mixed class counts or completed checkpoint differ.')
        return moments
    initial=None;start=0
    if checkpoint.exists():
        initial,meta=old.load_moments(checkpoint);start=meta['next_batch'];require(meta==prefix(additional,start,'additional-statistics')
            and np.array_equal(initial.n,np.broadcast_to(meta['class_counts'],initial.n.shape)),'Moment checkpoint is not exact complete-batch prefix.')
    if start<len(additional.batches()):
        import torch
        gpu=old.TorchMoments(initial);collector=old.Collector(moments=gpu)
        with checked_hooks(get_model(),collector,additional,'statistics') as layers,torch.inference_mode():
            for i,part in enumerate(additional.batches()):
                if i<start:continue
                output=forward(get_model(),collector,additional.collate(part),layers);del output
                if (i+1)%CHECKPOINT_INTERVAL==0 or i+1==len(additional.batches()):
                    snapshot=gpu.export();meta=prefix(additional,i+1,'additional-statistics');require(np.array_equal(snapshot.n,np.broadcast_to(meta['class_counts'],snapshot.n.shape)),'Online class counts differ.')
                    old.atomic_npz(checkpoint,metadata=np.array(json.dumps(meta)),**snapshot.arrays());data.atomic_json(OUT/'status.json',dict(stage='statistics',batch=i+1,batches=len(additional.batches()),rows=meta['rows']))
            initial=gpu.export()
        del gpu,collector;gc.collect()
    require(initial is not None,'Missing completed statistics.')
    if reuse:
        previous=reused_moments(c4,plan) if previous is None else previous
        moments=old.HeadMoments(arrays={n:initial.arrays()[n]+previous.arrays()[n] for n in ['n','sum','xx']});del previous
    else:moments=initial
    expected=plan['data']['statistics']['class_counts'];require(np.array_equal(moments.n,np.broadcast_to(expected,moments.n.shape)),'Combined mixed-statistics class counts changed.')
    metadata=dict(plan_sha256=data.digest(OUT/'plan.json'),data=plan['data']['statistics'],class_counts=expected,old_C4_moment_reuse=reuse,additional_checkpoint_sha256=data.digest(checkpoint))
    old.atomic_npz(path,metadata=np.array(json.dumps(metadata)),**moments.arrays());immutable(done_path,dict(plan_sha256=data.digest(OUT/'plan.json'),moments_sha256=data.digest(path),metadata=metadata,fit_only=True,validation_inputs_loaded=False,tests_read=False))
    return moments


def lda_model(moments,plan):
    path=OUT/'models/mixed-statistics-lda.npz'
    metadata=dict(plan_sha256=data.digest(OUT/'plan.json'),statistics_complete_sha256=data.digest(OUT/'statistics-complete.json'),roles=ROLES,heads=2304,head_dim=64,statistics_only=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            require(json.loads(str(z['metadata']))==metadata,'Mixed statistics classifier belongs to another protocol.');weight,bias=z['weight'].copy(),z['bias'].copy();diagnostics=json.loads(str(z['diagnostics']))
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):weight,bias,diagnostics=moments.fit(return_diagnostics=True)
        old.atomic_npz(path,weight=weight,bias=bias,metadata=np.array(json.dumps(metadata)),diagnostics=np.array(json.dumps(diagnostics)))
    require(weight.shape==(2304,5,64) and bias.shape==(2304,5) and weight.dtype==bias.dtype==np.float64 and np.isfinite(weight).all() and np.isfinite(bias).all(),'LDA portable arrays invalid.')
    immutable(OUT/'lda-diagnostics.json',diagnostics);return weight,bias


def valid_probability(array,rows):
    require(array.shape==(rows,2304) and array.dtype==np.float32,'Discovery probability shape/dtype changed.')
    for start in range(0,rows,4096):
        x=array[start:start+4096];require(np.isfinite(x).all() and ((x>=0)&(x<=1)).all(),'Unwritten/invalid discovery probabilities.')


def discovery(get_model,bundle,weight,bias,plan,restart=False):
    path=CACHE/'discovery-system-probabilities.npy';done_path=OUT/'discovery-complete.json'
    if done_path.exists():
        done=read(done_path);require(done['plan_sha256']==data.digest(OUT/'plan.json') and done['probabilities_sha256']==data.digest(path)
            and done['model_sha256']==data.digest(OUT/'models/mixed-statistics-lda.npz') and done['data']==plan['data']['holdout'] and done['actual_probabilities_streamed'] is True
            and 0<=done['cpu_probability_max_difference']<1e-8 and done['partial_resume'] is False and done['validation_inputs_loaded'] is False and done['tests_read'] is False,'Completed discovery provenance changed.')
        p=np.load(path,mmap_mode='r');valid_probability(p,len(bundle.rows));return p
    if path.exists():
        require(restart,'Incomplete owned discovery requires explicit --restart-discovery; no cursor-only recovery.')
        data.atomic_json(OUT/f'discovery-restart-{time.time_ns()}.json',dict(plan_sha256=data.digest(OUT/'plan.json'),prior_incomplete_sha256=data.digest(path),full_restart=True,no_protocol_change=True))
    require(shutil.disk_usage(CACHE).free>=514796544+512*2**20,'Insufficient discovery bytes plus512MiB reserve.')
    p=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(bundle.rows),2304))
    for start in range(0,len(p),4096):p[start:start+4096]=np.nan
    p.flush();collector=old.Collector(weight=weight,bias=bias,probabilities=p)
    import torch
    with checked_hooks(get_model(),collector,bundle,'holdout') as layers,torch.inference_mode():
        for i,part in enumerate(bundle.batches()):
            output=forward(get_model(),collector,bundle.collate(part),layers);del output
            if (i+1)%8==0 or i+1==len(bundle.batches()):data.atomic_json(OUT/'status.json',dict(stage='fit-holdout-probabilities',batch=i+1,batches=len(bundle.batches()),partial_resume=False))
    p.flush();valid_probability(p,len(bundle.rows));require(collector.cpu_probability_difference<1e-8 and collector.checked_layers==set(range(36)),'All-layer NumPy/GPU LDA parity failed.')
    immutable(done_path,dict(plan_sha256=data.digest(OUT/'plan.json'),probabilities_sha256=data.digest(path),model_sha256=data.digest(OUT/'models/mixed-statistics-lda.npz'),
        data=plan['data']['holdout'],actual_probabilities_streamed=True,cpu_probability_max_difference=collector.cpu_probability_difference,partial_resume=False,
        validation_inputs_loaded=False,tests_read=False))
    del collector;gc.collect();return p


def publish_selection(rows,p,weight,bias,plan):
    result,marginals,pairs,baseline_units=choose_pair(rows,p);result.update(plan_sha256=data.digest(OUT/'plan.json'),discovery_sha256=data.digest(OUT/'discovery-complete.json'),
        source_sha256=data.digest(__file__),statistics_model_sha256=data.digest(OUT/'models/mixed-statistics-lda.npz'),head_selection_fit_only=True,validation_pending=True,tests_read=False,primary_changed=False,
        limitation=protocol()['limitation'])
    pd.DataFrame([dict(head_id=v['head_id'],layer=v['head_id']//64,head=v['head_id']%64,worst_gain=v['worst_gain'],tail_gain=v['tail_gain'],mean_gain=v['mean_gain'],**{d+'_partial_auroc_gain':g for d,g in v['gains'].items()}) for v in marginals]).to_csv(OUT/'marginal-head-scores.csv',index=False)
    pd.DataFrame([dict(head_a=v['head_ids'][0],head_b=v['head_ids'][1],tail_gain=v['tail_gain'],worst_gain=v['worst_gain'],mean_full_auroc_gain=v['mean_full_auroc_gain'],admissible=v['admissible'],**{d+'_partial_auroc_gain':g for d,g in v['gains'].items()}) for v in pairs]).to_csv(OUT/'pair-scores.csv',index=False)
    data.atomic_json(OUT/'pair-details.json',pairs)
    if result['selected_pair'] is not None:
        selected=result['selected_head_ids'];score=aggregate(p,selected);units,_=unit_scores(rows,score)
        immutable(OUT/'selected-pair-gains-bootstrap.json',descriptive_bootstrap(baseline_units,units));immutable(OUT/'selected-pair-unit-metrics.json',dict(baseline=baseline_units,selected=units))
        position=[];oldscore=aggregate(p,ORIGINAL)
        for low in [0,256,512,768]:
            bins=rows[rows.domain.eq('C4')&rows.body_offset.between(low,low+255)]
            for source,g in bins.groupby('source_uid',sort=True):
                ix=g.index.to_numpy();truth=g.role.eq('system').to_numpy();a=binary_scores(oldscore[ix],truth);b=binary_scores(score[ix],truth)
                position.append(dict(source_uid=source,body_start=low,body_end=low+255,**{k+'_gain':b[k]-a[k] for k in a}))
        pd.DataFrame(position).to_csv(OUT/'C4-position-gains.csv',index=False)
        model=OUT/'models/selected10-proxy-lda.npz';old.atomic_npz(model,weight=weight[selected],bias=bias[selected],selected_head_ids=np.asarray(selected,np.int64),metadata=np.array(json.dumps(dict(plan_sha256=data.digest(OUT/'plan.json'),statistics_only=True,proxy_not_readout=True,roles=ROLES,head_dim=64))))
        result.update(proxy_model_path=str(model),proxy_model_sha256=data.digest(model))
    else:result.update(proxy_model_path=None,proxy_model_sha256=None)
    immutable(OUT/'head-selection.json',result)
    return result


def verified_completion():
    plan=read(OUT/'plan.json');verify_plan(plan);complete=read(OUT/'scan-complete.json')
    require(complete['plan_sha256']==data.digest(OUT/'plan.json') and complete['source_sha256']==data.digest(__file__) and complete['tests_read'] is False,'Scan completion identity changed.')
    for name,want in complete['artifact_sha256'].items():require(data.digest(OUT/name)==want,'Scan artifact changed: '+name)
    done=read(OUT/'discovery-complete.json');require(data.digest(CACHE/'discovery-system-probabilities.npy')==done['probabilities_sha256'],'Discovery matrix changed.')
    valid_probability(np.load(CACHE/'discovery-system-probabilities.npy',mmap_mode='r'),55859)
    require(read(CACHE/'initialized.json')==dict(plan_sha256=data.digest(OUT/'plan.json'),source_sha256=data.digest(__file__)),'Cache owner changed.')
    return complete


def scan(restart_discovery=False):
    require((OUT/'plan.json').is_file(),'Declare and publish immutable metadata protocol before any scan/text/model loading.')
    no_competing_jobs();plan=read(OUT/'plan.json');views=verify_plan(plan)
    with run_lock():
        if (OUT/'scan-complete.json').exists():return verified_completion()
        resource_guard();owner=dict(plan_sha256=data.digest(OUT/'plan.json'),source_sha256=data.digest(__file__))
        if CACHE.exists():require((CACHE/'initialized.json').is_file() and read(CACHE/'initialized.json')==owner,'Unowned scanner cache; no silent reconstruction.')
        else:CACHE.mkdir();data.atomic_json(CACHE/'initialized.json',owner)
        (OUT/'models').mkdir(exist_ok=True)
        bundles,proof=materialize(views,plan['signature']['checkpoint_path']);immutable(OUT/'prestream-source-audit.json',proof)
        # Fail before first model load if the reused101-document snapshot does
        # not exactly match these retokenized native256-position C4 inputs.
        previous=reused_moments(sub_bundle(bundles['statistics'],'C4'),plan) if plan['signature']['old_C4_moment_reuse'] else None
        with ExitStack() as stack:
            loaded=[]
            def get_model():
                if not loaded:loaded.append(stack.enter_context(model_context(plan)))
                return loaded[0]
            fitted=statistics(get_model,bundles['statistics'],plan,previous);del previous
            weight,bias=lda_model(fitted,plan);del fitted;gc.collect()
            p=discovery(get_model,bundles['holdout'],weight,bias,plan,restart_discovery)
        result=publish_selection(bundles['holdout'].rows,p,weight,bias,plan);verify_plan(plan)
        names=['plan.json','prestream-source-audit.json','statistics-complete.json','lda-diagnostics.json','discovery-complete.json','models/mixed-statistics-lda.npz','model-backend.json','head-selection.json','marginal-head-scores.csv','pair-scores.csv','pair-details.json']
        names += ['selected-pair-gains-bootstrap.json','selected-pair-unit-metrics.json','C4-position-gains.csv','models/selected10-proxy-lda.npz'] if result['selected_pair'] else []
        for phase in ['statistics','holdout']:
            if (OUT/f'hook-validation-{phase}.json').exists():names.append(f'hook-validation-{phase}.json')
        complete=dict(plan_sha256=data.digest(OUT/'plan.json'),source_sha256=data.digest(__file__),artifact_sha256={n:data.digest(OUT/n) for n in names},
            selected_head_ids=result['selected_head_ids'],admissible=result['selected_pair'] is not None,validation_pending=True,tests_read=False,primary_changed=False,final_bank_created=False)
        immutable(OUT/'scan-complete.json',complete);data.atomic_json(OUT/'status.json',dict(stage='scan_complete',admissible=complete['admissible'],validation_pending=True));return complete


def preflight():
    import torch
    from scipy.stats import rankdata
    folder=Path(tempfile.mkdtemp(prefix='systemness-complementary-head-scan-preflight-',dir='/tmp'));before=data.digest(__file__);checks=[]
    globals_before=(old.OUT,old.CACHE,old.SHRINKAGE,old.TOP_K,data.runtime.atomic_json,data.runtime.status)
    def check(name,value=True):require(bool(value),'CPU fixture failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,ValueError,AssertionError,KeyError):checks.append(name);return
        raise RuntimeError('Expected rejection: '+name)
    truth=np.array([True,True,False,False]);a=binary_scores(np.array([.9,.8,.7,.1],np.float32),truth)
    check('perfect_lowFPR_partial_AUC_is1',a['partial_auroc']==a['auroc']==1.)
    a=binary_scores(np.full(4,.5,np.float32),truth);check('constant_tie_normalized_partial_AUC_is_alpha_over2',abs(a['partial_auroc']-.025)<1e-12 and a['auroc']==.5)
    a=binary_scores(np.array([.9,.7,.8,.1],np.float32),truth);check('FPRzero_vertical_segment_retained',abs(a['partial_auroc']-.5)<1e-12)
    a=binary_scores(np.array([.9,.9,.2,.1],np.float32),np.array([True,False,True,False]));check('complete_tie_groups_and_alpha_interpolation',abs(a['partial_auroc']-.025)<1e-12)
    reject('single_truth_class_NA_not_fabricated',lambda:binary_scores(np.ones(3),np.ones(3,bool)))
    reject('probability_range_guard',lambda:binary_scores(np.array([1.1,.2]),np.array([True,False])))
    rng=np.random.RandomState(7)
    for i in range(6):
        score=rng.choice(np.array([.1,.2,.5,.9],np.float32),40);truth=np.array([True]*13+[False]*27);a=binary_scores(score,truth)
        ranks=rankdata(score,method='average');expected=(ranks[truth].sum()-13*14/2)/(13*27);require(abs(a['auroc']-expected)<1e-12,'Independent tied rank AUC mismatch.')
    check('six_independent_rank_AUROC_tie_fixtures')
    rows=[]
    for doc in range(2):
        for role in ROLES:
            for offset in [0,600]:rows.append(dict(source_uid='c4-doc:'+str(doc),role=role,domain='C4',body_offset=offset,component_id='c4-doc:'+str(doc),dataset='C4'))
    for component in range(8):
        for dataset in ['oasst','toxicchat0124']:
            for role in ['system','user','assistant']:rows.append(dict(source_uid=f'{dataset}:{component}',role=role,domain='public',body_offset=0,component_id='public:'+str(component),dataset=dataset))
    rows=pd.DataFrame(rows);y=rows.role.eq('system').to_numpy();p=np.full((len(rows),40),.2,np.float32);p[:,8]=np.where(y,.9,.01);p[:,9]=np.where(y,.8,.02)
    expected=np.zeros(len(p),np.float64)
    for head in [0,8,1]:expected+=np.asarray(p[:,head],np.float64)
    check('fixed_order_F64_sum_then_F32_aggregation',np.array_equal(aggregate(p,[0,8,1]),(expected/3).astype(np.float32)))
    units,means=unit_scores(rows,aggregate(p,[8]));check('source_then_component_then_domain_macro',len(units['C4-all'])==len(units['C4-tail'])==2 and len(units['public-oasst'])==len(units['public-toxicchat0124'])==8 and all(v['partial_auroc']==1 for v in means.values()))
    result,marginal,pairs,baseline=choose_pair(rows,p,original=list(range(8)),screen=32)
    check('fixed32screen496joint_pairs',len(marginal)==32 and result['screened_count']==32 and len(pairs)==496)
    check('exact_two_new_heads_keep_original_order',result['selected_head_ids']==list(range(10)) and result['selected_pair']['head_ids']==[8,9])
    allconstant=np.full_like(p,.2);empty,_,pairs,_=choose_pair(rows,allconstant,original=list(range(8)),screen=32);check('inadmissible_stops_without_relaxation',empty['selected_pair'] is None and empty['selected_head_ids'] is None and len(pairs)==496)
    chosen_units,_=unit_scores(rows,aggregate(p,result['selected_head_ids']));ci=descriptive_bootstrap(baseline,chosen_units)
    check('bootstrap_sameC4doc_and_public_component_draws',ci['intervals']['C4-all']['partial_auroc']['percentile95']==ci['intervals']['C4-tail']['partial_auroc']['percentile95'] and ci['intervals']['public-oasst']['partial_auroc']['percentile95']==ci['intervals']['public-toxicchat0124']['partial_auroc']['percentile95'])
    check('postselection_intervals_explicitly_descriptive',ci['post_selection_optimistic'] and not ci['independent_significance'] and not ci['token_bootstrap'])
    require_checkpoint_path('/dev/shm/role-analysis-gptoss120b');check('fixed_checkpoint_string_guard_accepts_only_verified_path')
    reject('workspace_rejected_before_any_path_access',lambda:require_checkpoint_path('/workspace'))
    reject('other_checkpoint_namespace_rejected',lambda:require_checkpoint_path('/tmp/model'))
    crows=[];cinputs=[]
    for doc in [0,1]:
        for ri,role in enumerate(ROLES):
            prompt=doc*5+ri;cinputs.append(dict(prompt_ix=prompt,length=1032,split='fit'))
            for offset in uniform(1024,256):crows.append(dict(prompt_ix=prompt,message_ix=0,token_ix=23+offset,token_id=offset,role=role,document_id=doc,split='fit',profile='long-full',wrapper='native',sample_ix=2_000_000_000+len(crows)))
    nrows=[];ninputs=[]
    for comp in range(32):
        for dataset in ['oasst','toxicchat0124']:
            prompt=len(ninputs);uid=f'{dataset}:{comp}';ninputs.append(dict(prompt_ix=prompt,length=350,split='fit',source_uid=uid,eligible_fit=True))
            for mi,(role,count) in enumerate([('system',8),('user',140),('assistant',160)]):
                for offset in range(count):nrows.append(dict(original_prompt_ix=prompt,prompt_ix=prompt,message_ix=mi,token_ix=mi*200+offset,token_id=offset,role=role,split='fit',source_uid=uid,
                    dependency_component_id='component:'+str(comp),family_id='family:'+str(comp),dataset=dataset,eligible_fit=True,old_generated_collision=False,cross_split_assistant_collision=False,sample_ix=4_000_000_000_000+len(nrows)))
    ci,cr,ni,nr=map(pd.DataFrame,[cinputs,crows,ninputs,nrows]);views,split=partition_metadata(ci,cr,ni,nr)
    check('32natural_components_partition24plus8_without_text',len(split['holdout_components'])==8 and len(split['statistics_components'])==24)
    check('fit_sources_components_disjoint_and_SUA_only',not set(views['statistics']['rows'].uid)&set(views['holdout']['rows'].uid) and set(views['statistics']['rows'].query("domain=='public'").role)=={'system','user','assistant'})
    check('natural_statistics128_messagecap_holdout_all192',views['statistics']['rows'].query("domain=='public'").groupby(['original_prompt_ix','message_ix']).size().max()==128 and views['holdout']['rows'].query("domain=='public'").groupby(['original_prompt_ix','message_ix']).size().max()==160)
    bad=nr.copy();bad.loc[0,'split']='validation';reject('validation_rows_fail_before_partition_or_tokenizer',lambda:partition_metadata(ci,cr,ni,bad))
    bad=nr.copy();bad.loc[bad.family_id.eq('family:0'),'dependency_component_id']='component:1';reject('component_count_bridge_rejected',lambda:partition_metadata(ci,cr,ni,bad))
    audit=fingerprint_audit([('full','a','statistics','one'),('full','a','statistics','two'),('normalized','b','holdout','three')]);check('within_partition_duplicate_is_not_silent_repartition',audit['cross_partition_bridges']==0)
    reject('cross_partition_stored_hash_bridge_stops',lambda:fingerprint_audit([('full','a','statistics','one'),('full','a','holdout','two')]))
    # Additive mixed moments, including legitimate class-constant/zero heads.
    raw=rng.normal(size=(50,3,4));lab=np.tile(np.arange(5),10);raw[:,0]=0;raw[:,1]=lab[:,None]
    public=rng.normal(size=(18,3,4));plab=np.tile([0,1,3],6);public[:,0]=0;public[:,1]=plab[:,None]
    a=old.HeadMoments(3,4,5);a.update(raw,lab);b=old.HeadMoments(3,4,5);b.update(public,plab)
    mixed=old.HeadMoments(3,4,5,arrays={k:a.arrays()[k]+b.arrays()[k] for k in ['n','sum','xx']});direct=old.HeadMoments(3,4,5);direct.update(np.r_[raw,public],np.r_[lab,plab])
    check('old101_plus_public_moments_additive',all(np.allclose(mixed.arrays()[k],direct.arrays()[k],atol=1e-12,rtol=1e-12) for k in ['n','sum','xx']))
    wa,ba,diag=mixed.fit(return_diagnostics=True);wb,bb=direct.fit();check('shrinkage_LDA_stream_raw_match_and_degenerate_heads',np.allclose(wa,wb,atol=1e-8,rtol=1e-8) and np.allclose(ba,bb,atol=1e-8,rtol=1e-8) and diag['zero_within_heads']>=2)
    prob=old.predict_heads(np.r_[raw,public],wa,ba);check('headwise5class_probability_finite_normalized',np.isfinite(prob).all() and np.allclose(prob.sum(2),1))
    # Shared collate is exercised without real tokenizer/prompts/checkpoint.
    bundle=object.__new__(old.Bundle);bundle.inputs=pd.DataFrame(dict(uid=['a','b'],length=[3,5]));bundle.tokens={'a':[11,12,13],'b':[21,22,23,24,25]}
    bundle.rows=pd.DataFrame(dict(uid=['a','a','b','b'],token_ix=[0,2,0,4],token_id=[11,13,21,25],role=['system','user','assistant','system'],target_ix=[0,1,2,3],sample_ix=np.arange(4),document_id=[0,0,1,1]))
    bundle.lookup={u:g for u,g in bundle.rows.groupby('uid',sort=False)};batch=bundle.collate(bundle.inputs)
    check('left_padding_exact_targets_and_tokenIDs',batch['positions'].tolist()==[2,4,5,9] and batch['targets'].tolist()==[0,1,2,3] and batch['input_ids'].reshape(-1)[torch.from_numpy(batch['positions'])].tolist()==[11,13,21,25])
    # Direct disabled pre-hook: no moments or probability writes, input preserved.
    class Capture:
        def __init__(self):self.blocks={};self.calls=0
        def update(self,layer,values,lab):self.calls+=1
    capture=Capture();collector=old.Collector(moments=capture);collector.batch={'positions':np.array([0,2]),'labels':np.array([0,1]),'targets':np.array([0,1])};collector.enabled=False
    layer=torch.nn.Identity();handle=layer.register_forward_pre_hook(collector.hook(0));x=torch.randn(1,3,4096);before_x=x.clone();out=layer(x);handle.remove()
    check('disabled_pre_o_proj_hook_bitexact_and_no_statistics',torch.equal(out,before_x) and capture.calls==0 and collector.seen=={0})
    matrix=np.full((5,2304),np.nan,np.float32);reject('NaN_unwritten_probability_matrix_rejected',lambda:valid_probability(matrix,5))
    matrix.fill(.2);valid_probability(matrix,5);check('finite_F32_probability_matrix_accepted')
    check('old_module_globals_unchanged',globals_before==(old.OUT,old.CACHE,old.SHRINKAGE,old.TOP_K,data.runtime.atomic_json,data.runtime.status))
    check('preflight_source_unchanged',before==data.digest(__file__))
    record=dict(stage='synthetic_fit_only_complementary_head_scan',source_sha256=before,checks=checks,check_count=len(checks),proposal_sha256=PROPOSAL_SHA,
        cache_or_OUT_created=False,real_source_rows_read=False,prompt_text_or_tokenizer_read=False,old_cached_moment_values_read=False,
        real_head_or_validation_probabilities_read=False,real_head_selection_run=False,model_loaded=False,gpu_used=False,tests_read=False)
    path=folder/'report.json';path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(path),report_sha256=data.digest(path),source_sha256=before,checks=len(checks))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--preflight',action='store_true');group.add_argument('--stage',choices=['declare','scan','verify']);parser.add_argument('--restart-discovery',action='store_true');args=parser.parse_args()
    require(not args.restart_discovery or args.stage=='scan','Discovery restart is explicit scan-only technical retry.')
    if args.preflight:preflight()
    elif args.stage=='declare':declare()
    elif args.stage=='scan':scan(args.restart_discovery)
    elif args.stage=='verify':verified_completion()
    else:parser.print_help()

if __name__=='__main__':main()
