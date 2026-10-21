"""Independent source/AST-only synthetic review. Never imports production modules."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
C4=HERE/'systemness_complementary_confirmation_report.py'
NAT=HERE/'systemness_complementary_natural_report.py'
PINS={C4:'d3c0881bb271b2f4b29829054117cdac82ebf4864d526ecd19cde6eef853bb62',NAT:'d714a1b9ae0fc03edecc04e43ef288f1bc11e59d6d4c2e7e2d12e9c181d50e3d'}
ROLES=['system','user','cot','assistant','tool']
RECORDS=['new-max33','matched-max33','oldhead-max33','oldhead-original']
MAPPING=dict(zip(RECORDS,['new','matched','old-head','old-head']))
PHYSICAL=['new','matched','old-head']
CHECKS=[]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def check(name,ok=True):require(ok,name);CHECKS.append(name)
def reject(name,fn):
    try:fn()
    except RuntimeError:CHECKS.append(name)
    else:raise RuntimeError('Accepted '+name)
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n')
def extract(path,names,env):
    nodes=[node for node in ast.parse(path.read_text()).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
    require({n.name for n in nodes}==set(names),'Function extraction incomplete')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env
def norm_ast(path,name):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    text=ast.dump(node,include_attributes=False)
    for a,b in [('new-maximin','new-max33'),('matched-maximin','matched-max33'),('old-head-maximin','oldhead-max33'),('old-head-original','oldhead-original')]:text=text.replace(a,b)
    return text
def ratio(a,b):return float(a/b) if b else None
def ref_stats(v,natural=False):
    tp,fp,fn,tn,correct,total=map(float,v[:6]);r=ratio(tp,tp+fn);s=ratio(tn,tn+fp)
    result=dict(system_precision=ratio(tp,tp+fp),system_recall=r,system_f1=ratio(2*tp,2*tp+fp+fn),system_balanced_accuracy=None if r is None or s is None else (r+s)/2,system_false_positive_rate=ratio(fp,fp+tn),accuracy=ratio(correct,total))
    if natural:
        for i,role in enumerate(['system','user','assistant']):result['role_recall_'+role]=ratio(v[6+2*i],v[7+2*i])
    return result
def eq(a,b):
    if a is None or b is None:return a is None and b is None
    return np.isclose(a,b,rtol=0,atol=1e-12)
def ref_bootstrap(frame,records,fields,groups,seed,natural):
    maxdiff=0.
    for gi,key in enumerate(groups):
        group=frame[(frame.stratum==key[0])&(frame.dataset==key[1])] if natural else frame[frame.profile==key]
        arrays={}
        for record in records:
            part=group[group.model==record] if natural else group[group.record==record]
            if natural:part=part.groupby('dependency_component_id')[fields].sum().sort_index()
            else:part=part.sort_values('group_id')
            arrays[record]=part[fields].to_numpy(np.float64)
        n=len(next(iter(arrays.values())))
        draws=np.random.RandomState(seed+gi).randint(n,size=(2000,n)) if n else np.empty((2000,0),np.int64)
        points={r:ref_stats(a.sum(0),natural) for r,a in arrays.items()}
        sample={r:[ref_stats(a[d].sum(0),natural) for d in draws] for r,a in arrays.items()}
        yield key,n,points,sample

def main():
    start={str(path):digest(path) for path in PINS}
    check('Exact two candidate source pins',all(start[str(p)]==s for p,s in PINS.items()))
    # The copied statistics, including undefined policies, must retain the frozen semantics.
    for new,old,names in [(C4,HERE/'systemness_aug_confirmation_report.py',['binary_values','bootstrap','group_indices','targets']),
            (NAT,HERE/'systemness_aug_natural_report.py',['finite','count_values','token_metrics','summarize','bootstrap'])]:
        for name in names:check('Frozen-statistics AST parity '+new.stem+'/'+name,norm_ast(new,name)==norm_ast(old,name))
    folder=Path(tempfile.mkdtemp(prefix='confirmation-review-complementary-reports-',dir='/tmp'))
    closed=SimpleNamespace(RECORD_TO_MODEL=MAPPING,require=require,read=read,digest=digest)
    c=dict(np=np,pd=pd,itertools=itertools,Path=Path,json=json,__file__=str(C4),require=require,read=read,digest=digest,
        closed=closed,RECORDS=RECORDS,PHYSICAL=PHYSICAL,RECORD_TO_PHYSICAL=MAPPING,PRIMARY='new-max33',BOOTSTRAPS=2000,
        BOOTSTRAP_SEED=20261021+528123,FIELDS=['tp','fp','fn','tn','correct','tokens'])
    c['PROFILES']=['single','multi','long-single','long-full','long-full-tail'];c['BIN_NAMES']=[f'long-full-bin-{a:04d}-{b:04d}' for a,b in [(0,31),(32,63),(64,127),(128,255),(256,511),(512,1023)]]
    c['GROUPS']=c['PROFILES']+c['BIN_NAMES'];c['EXPECTED_DOCUMENTS']={g:3 for g in c['GROUPS']};c['METRIC_FIELDS']=list(ref_stats([1,1,1,1,2,4]))
    extract(C4,['record_probabilities','binary_values','scalar','bootstrap','targets','load_completed','fixed_gallery','protocol_spec'],c)
    p={name:np.array([[.4,.5,.02,.04,.04]],np.float32) for name in PHYSICAL}
    aliases=c['record_probabilities'](p)
    check('Old-head records alias physical array and argmax',aliases[RECORDS[2]] is aliases[RECORDS[3]] and aliases[RECORDS[2]].argmax(1).tolist()==[1])
    reject('A fourth physical probability array rejected',lambda:c['record_probabilities'](p|{'other':p['new']}))
    docs=[]
    for gi,g in enumerate(c['GROUPS']):
        for ri,r in enumerate(RECORDS):
            arm=min(ri,2)
            for k in range(3):
                v=[1+arm+k,arm,2+gi%2,4+k,4+arm,10+arm+k+gi%2]
                docs.append(dict(profile=g,record=r,group_id=f'g{k}',document_id=k,**dict(zip(c['FIELDS'],v))))
    docs=pd.DataFrame(docs);cp,ci=c['bootstrap'](docs)
    check('C4 11groups/4records/6statistics -> 264CI/396pairs',len(ci)==264 and len(cp)==396)
    max_c=0.
    for g,n,point,sample in ref_bootstrap(docs,RECORDS,c['FIELDS'],c['GROUPS'],c['BOOTSTRAP_SEED'],False):
        for out in [x for x in ci if x['profile']==g]:
            values=[x[out['metric']] for x in sample[out['record']] if x[out['metric']] is not None]
            bounds=np.quantile(values,[.025,.975])
            require(eq(out['value'],point[out['record']][out['metric']]),'C4 point differs')
            max_c=max(max_c,abs(out['ci95_low']-bounds[0]),abs(out['ci95_high']-bounds[1]))
        for out in [x for x in cp if x['profile']==g]:
            before,after=out['baseline_record'],out['comparison_record'];m=out['metric']
            values=[100*(b[m]-a[m]) for a,b in zip(sample[before],sample[after]) if a[m] is not None and b[m] is not None]
            bounds=np.quantile(values,[.025,.975]);max_c=max(max_c,abs(out['ci95_low']-bounds[0]),abs(out['ci95_high']-bounds[1]))
    check('Independent all C4 paired-bootstrap numeric parity',max_c<1e-12)
    check('Old-head C4 fiveaccuracy paired differences exact zero',all(x['gain_percentage_points']==x['ci95_low']==x['ci95_high']==0 for x in cp if x['baseline_record']==RECORDS[2] and x['comparison_record']==RECORDS[3] and x['metric']=='accuracy'))
    undef=docs.copy();undef[['tp','fp']]=0;_,ui=c['bootstrap'](undef)
    check('C4 undefined precision N/A and zero valid draws',all(x['value'] is None and x['ci95_low'] is None and x['valid_bootstrap_draws']==0 for x in ui if x['metric']=='system_precision'))
    dup=pd.concat([docs,docs.iloc[:1]],ignore_index=True);reject('C4 duplicate paired source rejected',lambda:c['bootstrap'](dup))
    target=[dict(record='new-max33',profile=g,system_precision=.9,system_recall=.9,system_balanced_accuracy=.9) for g in c['GROUPS']]
    check('C4 90 target requires every position bin',all(c['targets'](target[:5],target[5:]).values()))
    target[-1]['system_recall']=.899;check('C4 tail bin failure not hidden',not all(c['targets'](target[:5],target[5:]).values()))
    n=dict(np=np,pd=pd,itertools=itertools,Path=Path,json=json,__file__=str(NAT),require=require,read=read,digest=digest,
        data=SimpleNamespace(ROLES=ROLES,atomic_json=atomic),closed=closed,RECORDS=RECORDS,PHYSICAL=PHYSICAL,
        OBSERVED=['system','user','assistant'],STRATA=[(s,d) for s in ['all-prepared','clean-test'] for d in ['all','oasst','toxicchat0124']],
        BOOTSTRAPS=2000,BOOTSTRAP_SEED=20261021+128432,COUNT_FIELDS=['tp','fp','fn','tn','correct','tokens','system_correct','system_tokens','user_correct','user_tokens','assistant_correct','assistant_tokens'],LIMITATION='synthetic-source-only')
    n['STATISTICS']=list(ref_stats(np.ones(12),True))
    extract(NAT,['finite','count_values','token_metrics','summarize','bootstrap','write','write_artifacts','protocol_spec'],n)
    rows=[]
    for si in range(4):
        # Same external-reference component bridges OA/TC and two distinct families; clean drops its middle source only.
        for mi,role in enumerate(n['OBSERVED']):
            for off in range(2):rows.append(dict(cache_ix=len(rows),sample_ix=10**13+len(rows),source_uid='s'+str(si),dataset='oasst' if si<2 else 'toxicchat0124',family_id='f'+str(si),dependency_component_id='bridge' if si<3 else 'single',eligible_clean_test=si!=1,message_ix=mi,token_ix=mi*7+off,role=role))
    rows=pd.DataFrame(rows);rng=np.random.RandomState(71)
    physical={}
    for name in PHYSICAL:
        raw=rng.gamma(2,1,size=(len(rows),5));physical[name]=(raw/raw.sum(1,keepdims=True)).astype(np.float32)
    thresholds=dict(zip(RECORDS,[.15,.25,.35,.65]));nm,ncounts,msg=n['summarize'](rows,physical,thresholds);npairs,nis,components=n['bootstrap'](ncounts)
    check('Natural 24metrics/216CI/324pairs and SUA support',len(nm)==24 and len(nis)==216 and len(npairs)==324 and all(x['recall']['cot'] is x['recall']['tool'] is None for x in nm))
    check('External bridge retained across clean/source filters',set(ncounts.loc[ncounts.source_uid.isin(['s0','s2']),'dependency_component_id'])=={'bridge'} and all(x['dependency_components']==1 for x in nis if x['dataset']=='oasst'))
    max_n=0.
    for key,num,point,sample in ref_bootstrap(ncounts,RECORDS,n['COUNT_FIELDS'],n['STRATA'],n['BOOTSTRAP_SEED'],True):
        for out in [x for x in nis if (x['stratum'],x['dataset'])==key]:
            m=out['metric'];values=[s[m] for s in sample[out['model']] if s[m] is not None]
            require(eq(out['value'],point[out['model']][m]),'Natural point mismatch')
            require(out['defined_bootstrap_replicates']==len(values),'Natural defined draw mismatch')
            if num<2:require(out['ci95_low'] is None and not out['ci_available'],'Single component CI invented')
            elif values:
                bounds=np.quantile(values,[.025,.975]);max_n=max(max_n,abs(out['ci95_low']-bounds[0]),abs(out['ci95_high']-bounds[1]))
        for out in [x for x in npairs if (x['stratum'],x['dataset'])==key]:
            m=out['metric'];a,b=out['baseline_model'],out['comparison_model'];values=[100*(q[m]-p[m]) for p,q in zip(sample[a],sample[b]) if p[m] is not None and q[m] is not None]
            require(out['defined_bootstrap_replicates']==len(values),'Natural paired defined draw mismatch')
            if num>=2 and values:
                bounds=np.quantile(values,[.025,.975]);max_n=max(max_n,abs(out['ci95_low']-bounds[0]),abs(out['ci95_high']-bounds[1]))
    check('Independent all nine natural statistics / six paired component differences',max_n<1e-12)
    check('Natural old aliases exact accuracy/recall paired zero',all(x['gain_percentage_points']==0 for x in npairs if x['baseline_model']==RECORDS[2] and x['comparison_model']==RECORDS[3] and x['metric'] in ['accuracy','role_recall_system','role_recall_user','role_recall_assistant']))
    p=np.array([[.4,.5,0,.1,0],[.1,.6,.2,.1,0],[.1,.1,.5,.3,0]],np.float32)
    v,_,_,_=n['token_metrics'](p,['system','user','assistant'],.3)
    check('Threshold system recall remains distinct from argmax system recall',v['system_recall']==1 and v['role_recall_system']==0 and v['recall']['cot'] is None and v['role_support']['cot']==0)
    v2=n['token_metrics'](p,['system','user','assistant'],.9)[0]
    check('Old aliases probability ranking metrics unchanged by tau',v['system_auroc']==v2['system_auroc'] and v['system_average_precision']==v2['system_average_precision'] and v['accuracy']==v2['accuracy'])
    check('Truth-supported macro handles false positive CoT without inventing CoT truth',v['macro_f1']<1 and v['role_support']['cot']==0)
    # Report status payload exercises actual write helper, replacing only figure rendering in our isolated AST namespace.
    n['plots']=lambda *a,**k:[]
    ledger=[dict(source_uid='s0',dataset='oasst',native_id='fake',status='prepared',eligible_clean_test=True,family_id='f0',dependency_component_id='bridge',historical_role_input_collision=True,historical_role_input_collision_roles=['user'],historical_role_input_references=['prepared-role:fake'],public448_assistant_collision=False)]
    meta=dict(prepared_prompts=4,system_family_counts={'fixed':32,'all_prepared':4,'clean_prepared':3},dependency_components=2,clean_dependency_components=2,components=[{'component_id':'bridge','source_uids':['s0','s1','s2']}],eligibility_counts={'synthetic':True},gallery_source_uids=[])
    result=n['write_artifacts'](folder/'normal',meta,rows,ledger,physical,thresholds,{'synthetic_only':True})
    status=read(folder/'normal/source-status.json')
    check('Crosscorpus flags and reference components pass through source-status',status['records'][0]['historical_role_input_references']==['prepared-role:fake'] and status['records'][0]['historical_role_input_collision_roles']==['user'] and status['prepared_components']==meta['components'])
    empty=rows.iloc[:0].copy();ep={k:np.empty((0,5),np.float32) for k in PHYSICAL}
    em,ec,_=n['summarize'](empty,ep,thresholds);epr,eis,_=n['bootstrap'](ec)
    check('All-failed empty observations retain24/216/324 N/A and noCI',len(em)==24 and len(epr)==324 and len(eis)==216 and all(x['value'] is None and x['defined_bootstrap_replicates']==0 and not x['ci_available'] for x in eis))
    # Every actual-facing entry rejects unknown pins before even importing a builder.
    def no_pin():raise RuntimeError('unknown pins')
    closed.pin_gate=no_pin
    reject('C4 unknown pins before imports/cache/p',c['load_completed'])
    reject('Natural unknown pins before imports/cache/p',n['write'])
    # Fake module/terminal/source checks: any probability access is a sentinel, never actual IO.
    fake_builder=ModuleType('systemness_complementary_natural_data');fake_builder.OUT=folder/'gates'
    fake_c4=SimpleNamespace(OUT=fake_builder.OUT);iout=fake_builder.OUT/'inference';iout.mkdir(parents=True)
    runner=ModuleType('systemness_complementary_confirmation_infer');events=[]
    ctx=SimpleNamespace(snapshot={'runtime_files_sha256':{}},metadata={})
    runner.load_context=lambda *a,**k:(events.append('context') or ctx)
    def forbidden_p(*a,**k):events.append('probabilities');raise LookupError('Synthetic probability boundary reached')
    runner.completed_probabilities=forbidden_p
    closed.pin_gate=lambda:None
    c['importlib']=SimpleNamespace(import_module=lambda name:fake_c4 if name.endswith('_data') else runner)
    c['infer_module']=lambda:runner
    with patch.dict(sys.modules,{'systemness_complementary_natural_data':fake_builder,'systemness_complementary_confirmation_infer':runner}):
        for label,fn in [('C4',c['load_completed']),('Natural',n['write'])]:
            events.clear();reject(label+' missing terminal marker before context/p',fn);require(not events,'Early IO')
        atomic(iout/'inference-complete.json',{});atomic(iout/'status.json',{'stage':'infer'})
        for label,fn in [('C4',c['load_completed']),('Natural',n['write'])]:
            events.clear();reject(label+' wrong terminal status before context/p',fn);require(not events,'Early IO')
        atomic(iout/'status.json',{'stage':'inference_complete'})
        for label,fn in [('C4',c['load_completed']),('Natural',n['write'])]:
            events.clear();reject(label+' frozen report sourceSHA before p',fn);require(events==['context'],'Probability accessed before source pin')
    check('No production modules imported by annex',not any(name in sys.modules for name in ['systemness_data','systemness_probes','systemness_complementary_confirm_snapshot']))
    end={str(p):digest(p) for p in PINS};check('Both production source SHA unchanged',start==end)
    report=dict(stage='synthetic_complementary_reports_source_audit',audit_passed=True,source_sha256=start,checks=CHECKS,checks_passed=len(CHECKS),maximum_C4_bootstrap_difference=max_c,maximum_natural_bootstrap_difference=max_n,production_modules_imported=False,actual_artifacts_or_probabilities_read=False,actual_source_ID_body_cache_bank_tokenizer_model_read=False,gpu_used=False,figures_rerendered=False,notes='AST-only pure functions, tiny three-source C4 bootstrap geometry and four-source natural graph. Real figure counts/layout additionally reviewed in source and owner synthetic reports; no actual report run.')
    atomic(folder/'report.json',report)
    print(json.dumps({'report_path':str(folder/'report.json'),'report_sha256':digest(folder/'report.json'),'script_sha256':digest(__file__),**report}),flush=True)

if __name__=='__main__':main()
