#!/usr/bin/env python3
"""Independent aggregate-only CPU verification of one committed TEST result."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np

BASE=Path('/workspace/role-retraining-20261026')
RUN=BASE/'runs-v7/formal'
FEATURE=RUN/'features/formal'
CODE=Path('/root/role-retraining-20261026/code-v7')
ROLES=('system','developer','user','cot','assistant','tool')
HARD=datetime(2026,10,27,13,19,17,tzinfo=timezone.utc).timestamp()
RATIO_TOL=1e-12
NLL_TOL=1e-6

def guard(condition,code):
    if not condition:raise ValueError(code)

def read(path):return json.loads(path.read_text())

def sha(path):
    guard(path.name!='x.npy','x_array_access_forbidden')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def source(path):return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def budget():
    ends=[HARD]
    for phase in ('pilot','formal'):
        c=read(BASE/'runs-v7'/phase/'config.json');b=c['budget']
        start=datetime.fromisoformat(b['started_utc'].replace('Z','+00:00'))
        end=datetime.fromisoformat(b['deadline_utc'].replace('Z','+00:00'));hours=float(b['authorized_hours'])
        guard(c['phase']==phase and start.tzinfo is not None and end.tzinfo is not None and 0<hours<=24 and 0<(end-start).total_seconds()<=hours*3600,'budget_invalid')
        ends.append(end.timestamp())
    end=min(ends);guard(end>time.time(),'deadline_exceeded');return end

def stats(matrix,nll=None):
    matrix=np.asarray(matrix,dtype=np.int64)
    support=matrix.sum(1);predicted=matrix.sum(0);tp=np.diag(matrix);total=int(matrix.sum())
    f1=[2*int(tp[i])/int(support[i]+predicted[i]) for i in range(6) if support[i]]
    out={'tokens':total,'accuracy':int(tp.sum())/total if total else 0.0,'macro_f1':float(sum(f1)/len(f1)) if f1 else None,'confusion':matrix.tolist(),'roles':{}}
    if nll is not None:out['nll']=float(nll)
    for i,role in enumerate(ROLES):
        s,p,t=int(support[i]),int(predicted[i]),int(tp[i]);fp=p-t
        out['roles'][role]={'support':s,'predicted':p,'precision':t/p if p else None,'recall':t/s if s else None,
                            'f1':2*t/(s+p) if s else None,'false_positives':fp,
                            'false_positive_rate':fp/(total-s) if total>s else None}
    return out

def grouped(values,y,pred,losses):
    unique,inverse=np.unique(values,return_inverse=True)
    matrices=np.zeros((len(unique),6,6),np.int64)
    np.add.at(matrices,(inverse,y,pred),1)
    sums=np.bincount(inverse,weights=losses,minlength=len(unique))
    totals=matrices.sum((1,2))
    return {str(k):stats(matrices[i],sums[i]/totals[i]) for i,k in enumerate(unique)}

def compare(expected,observed,path,errors):
    if isinstance(expected,dict):
        guard(isinstance(observed,dict) and set(expected)==set(observed),'field_set_mismatch_'+path)
        for key in expected:compare(expected[key],observed[key],path+'/'+key,errors)
    elif isinstance(expected,list):
        guard(isinstance(observed,list) and len(expected)==len(observed),'list_shape_'+path)
        for i,(a,b) in enumerate(zip(expected,observed)):compare(a,b,path+'/'+str(i),errors)
    elif expected is None or isinstance(expected,(bool,str,int)):
        guard(expected==observed,'exact_value_mismatch_'+path)
    else:
        guard(isinstance(observed,(int,float)) and math.isfinite(expected) and math.isfinite(observed),'nonfinite_metric_'+path)
        error=abs(float(expected)-float(observed))
        kind='nll' if '/nll/' in path or path.endswith('/nll') else 'ratios'
        if error>errors[kind]['maximum_absolute_error']:errors[kind]={'maximum_absolute_error':error,'path':path}
        guard(error<=(NLL_TOL if kind=='nll' else RATIO_TOL),'metric_tolerance_'+path)

def cluster_ci(sources,y,pred,losses):
    unique,inverse=np.unique(sources,return_inverse=True);count=len(unique)
    guard(count==1024,'actual_source_cluster_count')
    matrices=np.zeros((count,6,6),np.int64);np.add.at(matrices,(inverse,y,pred),1)
    loss_sum=np.bincount(inverse,weights=losses,minlength=count);totals=matrices.sum((1,2))
    samples={key:[] for key in ('accuracy','macro_f1','nll')}
    samples.update({role+'.'+m:[] for role in ROLES for m in ('precision','recall','f1')})
    rng=np.random.default_rng(20261026)
    for _ in range(2000):
        draw=rng.integers(count,size=count)
        result=stats(matrices[draw].sum(0))
        samples['accuracy'].append(result['accuracy']);samples['macro_f1'].append(result['macro_f1'])
        samples['nll'].append(float(loss_sum[draw].sum()/totals[draw].sum()))
        for role in ROLES:
            for metric in ('precision','recall','f1'):
                value=result['roles'][role][metric]
                if value is not None:samples[role+'.'+metric].append(value)
    intervals={}
    for key,values in samples.items():
        intervals[key]={'lower':float(np.quantile(values,.025)),'upper':float(np.quantile(values,.975)),'valid_replicates':len(values)} if values else None
    return {'unit':'source_cluster','clusters':count,'replicates':2000,'seed':20261026,'method':'percentile','confidence':.95,'intervals':intervals}

def compact(value):
    supported={k:v for k,v in value['roles'].items() if v['support']>0}
    floor=min((v[m] or 0 for v in supported.values() for m in ('precision','recall')),default=None)
    lowest=min(((v[m] or 0,k,m) for k,v in supported.items() for m in ('precision','recall')),default=None)
    return {'tokens':value['tokens'],'accuracy':value['accuracy'],'macro_f1_present_roles':value['macro_f1'],'nll':value.get('nll'),
            'minimum_present_role_precision_recall':floor,'lowest_present_metric':None if lowest is None else
            {'role':lowest[1],'metric':lowest[2],'value':lowest[0],**{k:supported[lowest[1]][k] for k in ('support','predicted','false_positives')}},
            'per_role':{role:{k:v[k] for k in ('support','precision','recall','f1','false_positives')} for role,v in value['roles'].items()},
            'unsupported_roles':[k for k,v in value['roles'].items() if not v['support']]}

def main():
    end=budget()
    if sys.argv[1:]==['--print-timeout']:
        print(max(1,min(600,int(end-time.time()))));return
    guard(not sys.argv[1:] and os.environ.get('CUDA_VISIBLE_DEVICES')=='','cpu_only_arguments')
    started=time.time();timer=threading.Timer(min(600,end-started),lambda:os.kill(os.getpid(),signal.SIGKILL));timer.daemon=True;timer.start()
    proc=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    print(json.dumps({'independent_validation_started':True,'pid':os.getpid(),'start_ticks':int(proc[19]),'maximum_seconds':600}),file=sys.stderr,flush=True)
    evaluation_path=RUN/'evaluations/test.json';prob_path=RUN/'evaluations/test-probabilities.npy'
    receipt_path=RUN.parent/'campaign/receipts/formal.evaluate.test.json'
    receipt=read(receipt_path);guard(receipt['name']=='formal.evaluate.test','evaluation_receipt_name')
    guard(receipt['proof_sha256']==hashlib.sha256(canonical(receipt['proof']).encode()).hexdigest(),'receipt_proof_sha')
    marker=receipt['stage_marker_snapshot'];guard(marker['status']=='complete' and marker['phase']=='formal' and marker['stage']=='evaluate','receipt_marker')
    evaluation_source=source(evaluation_path);prob_source=source(prob_path)
    guard(evaluation_source['sha256']=='eaa3e98c7e5cdb099a7e96f1807fa5201d344771e2d1d5d7e1a341971ced8ab8','evaluation_original_sha')
    for path,entry in ((evaluation_path,evaluation_source),(prob_path,prob_source)):
        guard(receipt['proof'][str(path)]==entry['sha256'],'receipt_bound_output_sha')
    evaluation=read(evaluation_path);frozen=read(RUN/'freeze.json')
    guard(sha(RUN/'freeze.json')=='ecaf97a27e5cd8f456279360493e619b0e3de2c9efff4a451b53028d7d94eb97','freeze_original_sha')
    guard(sha(RUN/'best.npz')==frozen['probe_sha256']==evaluation['probe_sha256'],'frozen_probe_sha')
    guard(sha(RUN/'selection.json')==frozen['selection_sha256'] and sha(RUN/'training.json')==frozen['training_sha256'],'frozen_selection_training_sha')
    with np.load(RUN/'best.npz',allow_pickle=False) as z:metadata=json.loads(str(z['metadata']))
    guard(metadata==frozen['metadata'] and tuple(metadata['roles'])==ROLES and not metadata['engineering_smoke_only'],'probe_role_metadata')
    p=np.load(prob_path,mmap_mode='r',allow_pickle=False)
    guard(p.shape==(498688,6) and p.dtype==np.dtype('float32') and prob_source['bytes']==11968640,'probability_shape_dtype_bytes')
    guard(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.allclose(p.astype(np.float64).sum(1),1,atol=3e-5),'probability_values')
    manifest_path=FEATURE/'manifest.json';manifest_source=source(manifest_path);manifest=read(manifest_path)
    guard(manifest_source['sha256']==evaluation['feature_manifest_sha256']=='1401482dd7035e001eb699a49a75b6590709c9920c56a8a52c20b1894517041b','evaluation_feature_manifest_sha')
    for key in ('roles','feature_dim','heads','feature_position','model_binding','selection_sha256','pilot_fit_sha256','data_hash','config_hash'):
        guard(manifest[key]==metadata[key],'feature_binding_'+key)
    guard(manifest['progress']['test']['complete'] is True and manifest['progress']['test']['rows']==498688 and manifest['progress']['test']['records']==4748,'complete_test_feature_progress')
    guard(manifest['split_sources']['test']=='b7088b39bb48e785011b4e072aa22388427c7c144e6591e33acb8d9140b69597','audited_test_input_identity')
    chunks=defaultdict(list);attributes=defaultdict(list);shards=[];cursor=0
    for entry in manifest['splits']['test']:
        folder=(FEATURE/entry['path']).resolve();guard(folder.is_relative_to(FEATURE.resolve()),'shard_path')
        commit_path=folder/'complete.json';commit_source=source(commit_path);guard(commit_source['sha256']==entry['sha256'],'shard_commit_sha')
        commit=read(commit_path)
        guard(commit['signature']=={**manifest['signature'],'split':'test','input_sha256':manifest['split_sources']['test']},'shard_signature')
        guard(commit['rows']==entry['rows'] and commit['record_start']==cursor,'contiguous_shard_records');cursor=commit['record_end']
        record={'commit':commit_source,'rows':entry['rows'],'checked_arrays':{}}
        loaded={}
        for name in ('y','source','structure','groups'):
            path=folder/(name+'.npy');proof=source(path);guard(proof['sha256']==commit['files'][path.name],'array_sha_'+name)
            array=np.load(path,mmap_mode='r',allow_pickle=False);guard(array.shape==(entry['rows'],),'array_row_shape_'+name)
            loaded[name]=array;record['checked_arrays'][name]=proof
        guard(loaded['y'].dtype==np.dtype('int64') and np.all((loaded['y']>=0)&(loaded['y']<6)),'label_values')
        for name in ('y','source','structure'):chunks[name].append(np.array(loaded[name]))
        unique,inverse=np.unique(loaded['groups'],return_inverse=True);decoded=[json.loads(str(v)) for v in unique]
        keys=set().union(*(obj.keys() for obj in decoded))
        guard(keys==set(evaluation['by_attribute']),'group_attribute_fields')
        for key in keys:attributes[key].append(np.asarray([str(obj.get(key,'N/A')) for obj in decoded])[inverse])
        shards.append(record)
    guard(cursor==4748,'final_shard_records')
    y=np.concatenate(chunks['y']);sources=np.concatenate(chunks['source']);structures=np.concatenate(chunks['structure'])
    guard(len(y)==len(p),'total_label_probability_rows')
    pred=np.asarray(p.argmax(1),dtype=np.int64)
    losses=-np.log(np.clip(np.asarray(p[np.arange(len(y)),y],dtype=np.float64),1e-30,1))
    matrix=np.bincount(6*y+pred,minlength=36).reshape(6,6)
    overall=stats(matrix,float(losses.mean()))
    errors={'ratios':{'maximum_absolute_error':0.0,'path':None},'nll':{'maximum_absolute_error':0.0,'path':None}}
    compare(overall,evaluation['overall'],'overall',errors)
    independent_structures=grouped(structures,y,pred,losses)
    compare(independent_structures,evaluation['by_structure'],'by_structure',errors)
    structure_equal=float(np.mean([v['macro_f1'] for v in independent_structures.values()]))
    compare(structure_equal,evaluation['structure_equal_macro_f1'],'structure_equal_macro_f1',errors)
    minimum=min(v[m] or 0 for v in overall['roles'].values() for m in ('precision','recall'))
    compare(minimum,evaluation['minimum_precision_recall'],'minimum_precision_recall',errors)
    guard(evaluation['meets_dev_floor']==(minimum>=.9),'stored_floor_flag')
    independent_attributes={}
    for key,parts in attributes.items():
        value=grouped(np.concatenate(parts),y,pred,losses)
        compare(value,evaluation['by_attribute'][key],'by_attribute/'+key,errors)
        independent_attributes[key]=value
    print(json.dumps({'independent_metrics_checked':True,'rows':len(y),'actual_source_clusters':len(np.unique(sources)),'structures':len(independent_structures)}),file=sys.stderr,flush=True)
    ci=cluster_ci(sources,y,pred,losses)
    compare(ci,evaluation['confidence_intervals'],'confidence_intervals',errors)
    guard(sha(prob_path)==prob_source['sha256'] and sha(evaluation_path)==evaluation_source['sha256'],'committed_outputs_changed')
    guard(sha(RUN/'freeze.json')=='ecaf97a27e5cd8f456279360493e619b0e3de2c9efff4a451b53028d7d94eb97','freeze_changed')
    guard(time.time()<end,'deadline_exceeded')
    lowest_precision=min(overall['roles'],key=lambda r:overall['roles'][r]['precision'])
    lowest_recall=min(overall['roles'],key=lambda r:overall['roles'][r]['recall'])
    low_structures=sorted(independent_structures,key=lambda key:(independent_structures[key]['macro_f1'],-independent_structures[key]['tokens'],key))[:10]
    out={'format_version':'synthetic-agent-independent-test-validation-v1','status':'passed_independent_recomputation',
         'started_unix':started,'finished_unix':time.time(),'elapsed_seconds':time.time()-started,
         'script':source(Path(__file__)),'producer_classifier_implementation':source(CODE/'synthetic_agent_classifier.py'),
         'sources':{'evaluation':evaluation_source,'probabilities':prob_source,'evaluation_receipt':source(receipt_path),'feature_manifest':manifest_source,'freeze':source(RUN/'freeze.json'),'probe':source(RUN/'best.npz')},
         'label_and_group_shards':shards,'roles':list(ROLES),'winner':frozen['candidate_id'],
         'test_input_sha256':manifest['split_sources']['test'],'source_identity_definition':manifest.get('source_identity'),
         'independent_overall':overall,'independent_structure_equal_macro_f1':structure_equal,
         'thresholds':{'macro_f1_target':.95,'all_six_role_precision_recall_target':.9,'macro_f1_target_met':overall['macro_f1']>=.95,'all_six_role_precision_recall_target_met':minimum>=.9,'minimum_precision_recall':minimum},
         'lowest_precision':{'role':lowest_precision,**overall['roles'][lowest_precision]},'lowest_recall':{'role':lowest_recall,**overall['roles'][lowest_recall]},
         'independent_confidence_intervals':ci,'ci_rule':'Resample 1024 actual unique source-array clusters with replacement, 1024 clusters per draw, retaining all their rows; 2000 draws with NumPy PCG64 default_rng seed20261026; percentile linear quantiles .025/.975.',
         'producer_agreement':{'all_requested_metrics_and_fields_match':True,'absolute_tolerances':{'ratios_and_confusions':RATIO_TOL,'nll_float64_vs_producer_float32':NLL_TOL},'maximum_absolute_errors':errors,'structures_compared':len(independent_structures),'attribute_groups_compared':{k:len(v) for k,v in independent_attributes.items()},'source_cluster_count':len(np.unique(sources))},
         'diagnostics':{key:{value:compact(metrics) for value,metrics in independent_attributes[key].items()} for key in ('profile','rounds','history_policy','snapshot_stage','position_bin')},
         'lowest_ten_saved_structures':{key:compact(independent_structures[key]) for key in low_structures},
         'position_semantics':'position_bin is the quartile of absolute token position in each whole actual input, not per-message relative body position or the last quarter of each message body.',
         'formulas':{'prediction':'six-class argmax of committed float32 probabilities','confusion':'M[i,j]=count(y=i and argmax(p)=j)','accuracy':'trace(M)/sum(M)','precision':'M[r,r]/sum(M[:,r]); absent predictions => null','recall':'M[r,r]/sum(M[r,:]); absent true support => null','f1':'2*M[r,r]/(support_r+predicted_r); absent true support => null','macro_f1':'arithmetic mean F1 over classes with true support; overall TEST has all six classes','nll':'mean(-log(clip(float64(p[row,y]),1e-30,1)))'},
         'limitations':['This verifies committed labels/probabilities/group metadata; it does not rerun the frozen model or verify x feature vectors.','Subgroup macro-F1 averages only roles with true support; stages with only cot or tool targets are not six-class tests.','Position groups here use whole-input absolute quartiles; per-body tail diagnostics require the separate position report.','TEST diagnostics do not alter the already frozen DEV-only selection.'],
         'privacy_and_execution':{'x_arrays_opened_or_hashed':False,'input_bodies_opened':False,'predictions_or_labels_exported_as_rows':False,'gpu_started':False,'evaluate_called':False,'model_or_core_modified':False}}
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False));timer.cancel()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        print(json.dumps({'status':'validation_failed','error_type':type(error).__name__,'error_code':str(error) if isinstance(error,ValueError) else 'unexpected_api_or_io_failure'}))
        raise SystemExit(1)
