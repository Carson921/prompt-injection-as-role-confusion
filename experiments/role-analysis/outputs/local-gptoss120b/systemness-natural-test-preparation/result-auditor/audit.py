"""Independent CPU audit of completed natural128 four-model results.

Default: help. --preflight opens synthetic /tmp fixtures only. --actual may be
used only after the root confirms inference/report terminal exit0. Actual mode
checks terminal markers before builder cache or test probability access. No
preparation, source-response replay, model inference, tuning or GPU is run.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import tempfile
import zlib

import numpy as np
import pandas as pd

REPOSITORY=Path('/root/prompt-injection-as-role-confusion')
MODULES=REPOSITORY/'experiments/role-analysis'
NAMES=['residual','base-attention','dense128','head-selected']
ROLES=['system','user','cot','assistant','tool']
OBSERVED=['system','user','assistant']
FIELDS=['tp','fp','fn','tn','correct','tokens','system_correct','system_tokens','user_correct','user_tokens','assistant_correct','assistant_tokens']
BOOTSTRAPS,SEED=2000,20261021+128432
PRIMARY_SHA='27a22fea0e784caba9be2d777ecab892db85d30536c3eac46887d31283074266'
PRIMARY_THRESHOLD=.620969696969697
PREPARED_SHA='a3ba614a83feda6687a49b2fb10aca2ab3e9af8d3b1b24935e98ce54cf37175a'


def need(value,message):
    if not value:raise RuntimeError(message)

def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path).resolve();need(path.is_relative_to(Path('/tmp')),'Audit writes are restricted to /tmp.')
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)

def comparison(actual,expected,context='',tolerance=1e-12):
    if expected is None or (isinstance(expected,(float,np.floating)) and np.isnan(expected)):
        need(actual is None or (isinstance(actual,(float,np.floating)) and np.isnan(actual)),context+': expected N/A.');return 0.
    if isinstance(expected,dict):
        need(isinstance(actual,dict) and set(actual)==set(expected),context+': key set differs.')
        return max([0.]+[comparison(actual[k],expected[k],context+'/'+str(k),tolerance) for k in expected])
    if isinstance(expected,(list,tuple)):
        need(isinstance(actual,(list,tuple)) and len(actual)==len(expected),context+': list length differs.')
        return max([0.]+[comparison(a,b,context+'/'+str(i),tolerance) for i,(a,b) in enumerate(zip(actual,expected))])
    if isinstance(expected,(bool,np.bool_)):
        need(isinstance(actual,(bool,np.bool_)) and bool(actual)==bool(expected),context+': Boolean differs.');return 0.
    if isinstance(expected,str):
        need(actual==expected,context+': string differs.');return 0.
    if isinstance(expected,(int,np.integer)):
        need(np.isfinite(actual) and actual==expected,context+': integer differs.');return 0.
    need(np.isfinite(actual) and np.isfinite(expected),context+': unexpected nonfinite value.')
    error=abs(float(actual)-float(expected));need(error<=tolerance,context+': numeric error '+str(error));return error

def compare_records(actual,expected,keys,context,tolerance=1e-12):
    def index(records):
        result={tuple(row[k] for k in keys):row for row in records}
        need(len(result)==len(records),context+': repeated table key.');return result
    return comparison(index(actual),index(expected),context,tolerance)

def compare_frame(actual,expected,keys,context,tolerance=1e-12):
    if expected.empty:
        need(actual.empty,context+': expected no observations.');return 0.
    need(set(actual.columns)==set(expected.columns),context+': columns differ.')
    return compare_records(actual.to_dict('records'),expected.to_dict('records'),keys,context,tolerance)

def read_csv(path):
    if not Path(path).read_bytes().strip():return pd.DataFrame()
    return pd.read_csv(path)

def row_labels(rows):
    need(set(rows.role)<=set(OBSERVED),'Natural test has invented truth role.')
    return rows.role.map({role:i for i,role in enumerate(ROLES)}).to_numpy(dtype=np.int64)

def safe_div(a,b):return float(a/b) if b else None

def metrics(p,y,threshold):
    from sklearn.metrics import average_precision_score,roc_auc_score
    need(len(p)==len(y)>0 and p.dtype==np.float32 and p.shape[1]==5 and np.isfinite(p).all()
        and (p>=0).all() and (p<=1.00001).all() and np.allclose(p.sum(1),1,atol=2e-5,rtol=0),'Invalid probability rows.')
    predicted=p.argmax(1);matrix=np.zeros((5,5),np.int64);np.add.at(matrix,(y,predicted),1)
    support=matrix.sum(1);outcome=matrix.sum(0);diag=matrix.diagonal()
    f1=np.zeros(5,float);np.divide(2*diag,support+outcome,out=f1,where=(support+outcome)>0)
    positive=y==0;detection=p[:,0]>=threshold
    tp=int(np.count_nonzero(positive&detection));fn=int(np.count_nonzero(positive&~detection))
    fp=int(np.count_nonzero(~positive&detection));tn=int(np.count_nonzero(~positive&~detection))
    recall=safe_div(tp,tp+fn);specificity=safe_div(tn,tn+fp)
    return dict(tokens=len(y),accuracy=float(np.count_nonzero(predicted==y)/len(y)),macro_f1=float(f1[support>0].mean()),
        nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()),
        recall={role:safe_div(int(diag[i]),int(support[i])) for i,role in enumerate(ROLES)},confusion=matrix.tolist(),
        system_threshold=float(threshold),system_accuracy=float((tp+tn)/len(y)),system_balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
        system_precision=safe_div(tp,tp+fp),system_recall=recall,system_f1=safe_div(2*tp,2*tp+fp+fn),system_false_positive_rate=safe_div(fp,fp+tn),
        system_auroc=float(roc_auc_score(positive,p[:,0])) if positive.any() and (~positive).any() else None,
        system_average_precision=float(average_precision_score(positive,p[:,0])) if positive.any() else None,
        system_detection_by_role={role:float(detection[y==i].mean()) if np.any(y==i) else None for i,role in enumerate(ROLES)},system_confusion=dict(tp=tp,fn=fn,fp=fp,tn=tn))


def independent_summary(rows,probability,thresholds):
    need(np.array_equal(rows.index,np.arange(len(rows))) and set(probability)==set(thresholds)==set(NAMES),'Global row/model order differs.')
    output=[];source=[];projections=[];union_macro=[]
    for stratum in ['all-prepared','clean-test']:
        subset=rows if stratum=='all-prepared' else rows[rows.eligible_clean_test.eq(True)]
        datasets=['all']+sorted(subset.dataset.unique())
        for dataset in datasets:
            group=subset if dataset=='all' else subset[subset.dataset.eq(dataset)]
            for name in NAMES:
                common=dict(stratum=stratum,dataset=dataset,model=name,source_roots=int(group.source_uid.nunique()),system_families=int(group.family_id.nunique()),dependency_components=int(group.dependency_component_id.nunique()))
                if group.empty:
                    output.append(common|dict(tokens=0,status='no-observations',recall={role:None for role in ROLES},role_support={role:0 for role in ROLES},observed_roles=[],five_class_argmax_accuracy=None,
                        **{key:None for key in ['system_precision','system_recall','system_f1','system_balanced_accuracy','system_false_positive_rate','accuracy']}));continue
                p=probability[name][group.index];y=row_labels(group);result=metrics(p,y,thresholds[name])
                result.update(role_support={role:int(group.role.eq(role).sum()) for role in ROLES},observed_roles=OBSERVED,five_class_argmax_accuracy=result['accuracy'])
                output.append(common|dict(status='observed-SUA-only',**result))
                cm=np.asarray(result['confusion']);denom=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),denom,out=np.zeros(5,float),where=denom>0)
                union_macro.append(common|dict(truth_supported_macro_f1=result['macro_f1'],standard_truth_union_prediction_macro_f1=float(f1[denom>0].mean())))
                for uid,part in group.groupby('source_uid',sort=True):
                    pp=probability[name][part.index];yy=row_labels(part);predicted=pp.argmax(1);detected=pp[:,0]>=thresholds[name];truth=yy==0
                    values=[int(np.count_nonzero(truth&detected)),int(np.count_nonzero(~truth&detected)),int(np.count_nonzero(truth&~detected)),int(np.count_nonzero(~truth&~detected)),int(np.count_nonzero(predicted==yy)),len(part)]
                    for role in OBSERVED:
                        observed=yy==ROLES.index(role);values.extend([int(np.count_nonzero(observed&(predicted==ROLES.index(role)))),int(observed.sum())])
                    first=part.iloc[0]
                    source.append(dict(source_uid=uid,family_id=first.family_id,dependency_component_id=first.dependency_component_id,**dict(zip(FIELDS,values)),stratum=stratum,dataset=dataset,model=name))
        for name in NAMES:
            for key,part in subset.groupby(['source_uid','message_ix','role','family_id','dependency_component_id'],sort=True):
                p=probability[name][part.index]
                projections.append(dict(zip(['source_uid','message_ix','role','family_id','dependency_component_id'],key))|{f'p_{role}':float(p[:,i].mean()) for i,role in enumerate(ROLES)}|dict(tokens=len(part),stratum=stratum,model=name))
    return output,pd.DataFrame(source),pd.DataFrame(projections),union_macro


def counts_to_metrics(a):
    a=np.asarray(a,float)
    ratio=lambda n,d:np.divide(n,d,out=np.full_like(n,np.nan,dtype=float),where=d>0)
    recall=ratio(a[...,0],a[...,0]+a[...,2]);specificity=ratio(a[...,3],a[...,3]+a[...,1])
    return dict(system_precision=ratio(a[...,0],a[...,0]+a[...,1]),system_recall=recall,
        system_f1=ratio(2*a[...,0],2*a[...,0]+a[...,1]+a[...,2]),system_balanced_accuracy=(recall+specificity)/2,
        system_false_positive_rate=ratio(a[...,1],a[...,1]+a[...,3]),accuracy=ratio(a[...,4],a[...,5]),
        role_recall_system=ratio(a[...,6],a[...,7]),role_recall_user=ratio(a[...,8],a[...,9]),role_recall_assistant=ratio(a[...,10],a[...,11]))


def independent_bootstrap(source):
    comparisons=[];intervals=[];component_rows=[]
    if source.empty:return comparisons,intervals,pd.DataFrame()
    keys=sorted(set(zip(source.stratum,source.dataset)))
    for index,(stratum,dataset) in enumerate(keys):
        group=source[source.stratum.eq(stratum)&source.dataset.eq(dataset)]
        need(set(group.model)==set(NAMES) and not group.duplicated(['model','source_uid']).any(),'Missing/duplicate model-source pair.')
        reference=None
        for name in NAMES:
            mapping=group[group.model.eq(name)].sort_values('source_uid')[['source_uid','family_id','dependency_component_id']].to_dict('records')
            if reference is None:reference=mapping
            need(mapping==reference,'Different roots/component IDs between paired models.')
        ids=sorted(group.dependency_component_id.unique());arrays={}
        for name in NAMES:
            values=[]
            for cid in ids:
                part=group[group.model.eq(name)&group.dependency_component_id.eq(cid)]
                counts=[int(sum(part[field])) for field in FIELDS];values.append(counts)
                component_rows.append(dict(model=name,dependency_component_id=cid,**dict(zip(FIELDS,counts)),stratum=stratum,dataset=dataset))
            arrays[name]=np.asarray(values,float)
        draw=np.random.RandomState(SEED+index).randint(len(ids),size=(BOOTSTRAPS,len(ids)))
        points={name:counts_to_metrics(a.sum(0)) for name,a in arrays.items()};samples={name:counts_to_metrics(a[draw].sum(1)) for name,a in arrays.items()}
        common=dict(stratum=stratum,dataset=dataset,source_roots=len(reference),dependency_components=len(ids),system_families=int(group.family_id.nunique()),bootstrap_replicates=BOOTSTRAPS,resampling_unit='shared-family-and-assistant-repeat connected dependency component')
        finite=lambda value:float(value) if np.isfinite(value) else None
        def interval(values):
            defined=values[np.isfinite(values)];count=len(defined)
            reason='fewer_than_two_dependency_components' if len(ids)<2 else 'no_defined_bootstrap_replicates' if not count else None
            limits=np.quantile(defined,[.025,.975]) if reason is None else [None,None]
            return dict(ci95_low=None if limits[0] is None else float(limits[0]),ci95_high=None if limits[1] is None else float(limits[1]),defined_bootstrap_replicates=count,ci_available=reason is None,ci_unavailable_reason=reason)
        for name in NAMES:
            for metric,value in points[name].items():intervals.append(common|dict(model=name,metric=metric,value=finite(value))|interval(samples[name][metric]))
        for before,after in itertools.combinations(NAMES,2):
            for metric in points[before]:comparisons.append(common|dict(baseline_model=before,comparison_model=after,metric=metric,baseline=finite(points[before][metric]),comparison=finite(points[after][metric]),gain_percentage_points=finite(100*(points[after][metric]-points[before][metric])))|interval(100*(samples[after][metric]-samples[before][metric])))
    return comparisons,intervals,pd.DataFrame(component_rows)


def flat_metrics(records):
    return pd.DataFrame([{k:v for k,v in row.items() if not isinstance(v,(dict,list))}|{f'role_recall_{role}':v for role,v in row['recall'].items()}|{f'role_support_{role}':v for role,v in row['role_support'].items()} for row in records])


def pdf_literals(payload):
    strings=[]
    for stream in re.findall(rb'stream\r?\n(.*?)\r?\nendstream',payload,re.S):
        try:content=zlib.decompress(stream)
        except zlib.error:content=stream
        for literal in re.findall(rb'\(((?:\\.|[^()\\])*)\)',content,re.S):
            text=re.sub(rb'\\([()\\])',rb'\1',literal).replace(b'\\n',b'\n').replace(b'\\r',b'\r')
            strings.append(text.decode('latin-1'))
    return ''.join(strings)


def gallery_audit(folder,gallery,ledger,fixture=False):
    from PIL import Image
    ordered=sorted(row['source_uid'] for row in ledger);need(len(ordered)==len(set(ordered))==128,'Fixed128 gallery keys differ.')
    ids=[ordered[i] for i in np.linspace(0,127,12,dtype=int)]
    need(gallery['source_uids']==ids and gallery['observed_truth_roles']==OBSERVED and gallery['model_columns']==4 and gallery['probability_rows']==5 and gallery['png']==gallery['pdf']==13 and gallery['failed_ids_replaced'] is False and gallery['unsmoothed_probabilities'] is True,'Fixed gallery semantics differ.')
    stems={f'natural-test-{i:02d}' for i in range(12)}|{'natural-test-summary'}
    for extension in ['png','pdf']:need({p.stem for p in folder.glob('*.'+extension)}==stems,'Missing/extra fixed gallery files.')
    mapping={row['source_uid']:row for row in ledger};result=[]
    for stem in sorted(stems):
        png=folder/(stem+'.png');pdf=folder/(stem+'.pdf')
        with Image.open(png) as image:image.verify()
        with Image.open(png) as image:
            size=list(image.size)
            need(fixture or size==([1920,560] if stem.endswith('summary') else [3200,1120]),'PNG dimensions differ from5x4/summary layout.')
        payload=pdf.read_bytes();need(payload.startswith(b'%PDF-') and b'%%EOF' in payload[-1024:] and len(re.findall(rb'/Type\s*/Page\b',payload))==1,'Malformed/multipage PDF.')
        item=dict(stem=stem,png_sha256=digest(png),pdf_sha256=digest(pdf),dimensions=size)
        if not stem.endswith('summary'):
            serial=int(stem[-2:]);uid=ids[serial];failed=mapping[uid]['status']=='quality-failed';text=pdf_literals(payload)
            need(uid in text,'Plot title source ID differs from fixed gallery.')
            if failed:need(text.count('Quality failed')==20 and text.count('No invented observations')==20,'Failed source lacks all20 fixed-panel placeholders.')
            else:need('Quality failed' not in text,'Prepared source falsely shown as failed.')
            item.update(source_uid=uid,failed=failed,failed_placeholder_verified=failed)
        result.append(item)
    index=(folder/'gallery-index.md').read_text()
    need(all(f'[{uid}](natural-test-{i:02d}.png)' in index for i,uid in enumerate(ids)),'Gallery index source/file mapping differs.')
    return result


def report_text_audit(path,records):
    table={}
    for line in Path(path).read_text().splitlines():
        fields=line.split()
        if fields and fields[0] in NAMES:
            need(len(fields)==12,'Human report row must contain3 keys+9 numeric/N/A fields.')
            key=tuple(fields[:3]);need(key not in table,'Repeated human report identity.');table[key]=fields[3:]
    expected={}
    for row in records:
        values=[row[key] for key in ['system_precision','system_recall','system_f1','system_balanced_accuracy','system_false_positive_rate','accuracy']]+[row['recall'][role] for role in OBSERVED]
        expected[(row['model'],row['stratum'],row['dataset'])]=['N/A' if v is None else f'{v:.6f}' for v in values]
    need(table==expected,'Human report rounded metrics/identities differ from independent counts.')
    return len(table)


def actual(output):
    source_sha_at_start=digest(__file__)
    sys.path.insert(0,str(MODULES));import systemness_natural_test_confirm as runner
    # No completed-test cache or probability access before both terminal markers.
    need((runner.OUT/'complete.json').is_file(),'No terminal natural-test complete.json; do not read partial results.')
    need(read(runner.OUT/'status.json')['stage']=='complete','Natural-test status is not terminal complete.')
    import systemness_natural_test as builder
    need(digest(builder.OUT/'data-complete.json')==PREPARED_SHA,'Fixed128 prepared data SHA changed.')
    metadata,inputs,rows,tokens,ledger=builder.verified_data()
    runner.validate_layout(metadata,inputs,rows,tokens,ledger)
    inference,probability=runner.completed_probabilities(metadata,inputs,rows)
    complete=read(runner.OUT/'complete.json');frozen=metadata['frozen']
    need(runner.NAMES==NAMES and runner.data.ROLES==ROLES and runner.BOOTSTRAPS==BOOTSTRAPS,'Frozen audit constants differ.')
    need(frozen['primary_candidate']=='joint-mlp128' and frozen['models']['head-selected']==PRIMARY_SHA and frozen['thresholds']['head-selected']==PRIMARY_THRESHOLD,'Fixed primary model/threshold changed.')
    provenance={}
    def bind(path,sha256=None):
        path=Path(path).resolve();value=digest(path)
        if sha256 is not None:need(value==sha256,'Frozen SHA differs: '+str(path))
        provenance[str(path)]=value;return value
    bind(__file__,source_sha_at_start);bind(runner.OUT/'complete.json');bind(runner.OUT/'status.json');bind(runner.OUT/'inference-complete.json',complete['inference_sha256']);bind(builder.OUT/'data-complete.json',PREPARED_SHA)
    protocol=read(builder.OUT/'protocol.json');bind(builder.OUT/'protocol.json',metadata['protocol_sha256']);bind(builder.OUT/'frozen-readouts.json',metadata['frozen_readouts_sha256'])
    for name,sha256 in protocol['dependencies_sha256'].items():bind(MODULES/name,sha256)
    for name,sha256 in metadata['cache_sha256'].items():bind(builder.CACHE/name,sha256)
    for name,sha256 in metadata['metadata_sha256'].items():bind(builder.OUT/name,sha256)
    for name in NAMES:
        bind(frozen['model_paths'][name],frozen['models'][name]);p=probability[name]
        need(p.dtype==np.float32 and p.shape==(len(rows),5),'Probability model/row dtype differs.')
        if len(rows):bind(runner.CACHE/f'{name}-probabilities.f32',inference['probability_sha256'][name])
    need(complete['signature']==inference['signature'] and complete['primary']=='head-selected' and complete['primary_model_sha256']==PRIMARY_SHA and complete['primary_system_threshold']==PRIMARY_THRESHOLD
        and complete['tests_used_for_tuning'] is False and complete['thresholds_retuned'] is False and complete['primary_changed'] is False and complete['source_sha256']==digest(runner.__file__),'Result signature/selection/tuning contract differs.')
    need(complete['fixed_source_roots']==128 and complete['prepared_source_roots']==len(inputs) and complete['quality_failed_roots']==128-len(inputs) and complete['bootstrap_replicates']==BOOTSTRAPS
        and complete['gpu_run']==bool(len(inputs)) and complete['no_observations']==(not len(rows)),'Completion source/inference status differs.')
    required={'metrics.json','metrics.csv','source-counts.csv','component-counts.csv','paired-comparisons.csv','bootstrap-metrics.csv','message-projections.csv.gz','source-status.json','source-status.csv','report.txt','gallery.json','plots/gallery-index.md','inference-complete.json'}
    if len(rows):required.update(['portable-validation.json','hook-validation.json','model_backend.json'])
    required.update({f'plots/natural-test-{i:02d}.{ext}' for i in range(12) for ext in ['png','pdf']});required.update({'plots/natural-test-summary.png','plots/natural-test-summary.pdf'})
    need(required<=set(complete['artifact_sha256']),'Completion manifest omitted required statistics/plot/evidence.')
    for name,sha256 in complete['artifact_sha256'].items():
        path=(runner.OUT/name).resolve();need(path.is_relative_to(runner.OUT.resolve()),'Unsafe completion artifact path.');bind(path,sha256)
    calculated,source,messages,union_macro=independent_summary(rows,probability,frozen['thresholds'])
    metric_difference=compare_records(read(runner.OUT/'metrics.json'),calculated,['stratum','dataset','model'],'metrics JSON')
    csv_difference=compare_frame(read_csv(runner.OUT/'metrics.csv'),flat_metrics(calculated),['stratum','dataset','model'],'metrics CSV')
    compare_frame(read_csv(runner.OUT/'source-counts.csv'),source,['stratum','dataset','model','source_uid'],'source counts',0)
    message_difference=compare_frame(read_csv(runner.OUT/'message-projections.csv.gz'),messages,['stratum','model','source_uid','message_ix','role','family_id','dependency_component_id'],'message projections',2e-6)
    paired,intervals,components=independent_bootstrap(source)
    compare_frame(read_csv(runner.OUT/'component-counts.csv'),components,['stratum','dataset','model','dependency_component_id'],'component counts',0)
    interval_difference=compare_frame(read_csv(runner.OUT/'bootstrap-metrics.csv'),pd.DataFrame(intervals),['stratum','dataset','model','metric'],'bootstrap model intervals',1e-10)
    paired_difference=compare_frame(read_csv(runner.OUT/'paired-comparisons.csv'),pd.DataFrame(paired),['stratum','dataset','baseline_model','comparison_model','metric'],'six-pair intervals',1e-9)
    status_fields=['source_uid','dataset','native_id','status','quality_failure','eligible_clean_test','old_generated_collision','dev_assistant_collision','family_id','dependency_component_id']
    expected_status=[{key:row.get(key) for key in status_fields} for row in ledger]
    comparison(read(runner.OUT/'source-status.json'),dict(fixed_source_roots=128,records=expected_status),'128source JSON status')
    compare_frame(read_csv(runner.OUT/'source-status.csv'),pd.DataFrame(expected_status),['source_uid'],'128source CSV status')
    need(complete['role_support']=={role:int(rows.role.eq(role).sum()) for role in ROLES},'Complete observed/unsupported role counts differ.')
    need(all(record['recall']['cot'] is None and record['recall']['tool'] is None for record in calculated),'Unsupported role recall must be N/A.')
    plots=gallery_audit(runner.OUT/'plots',read(runner.OUT/'gallery.json'),ledger)
    text_rows=report_text_audit(runner.OUT/'report.txt',calculated)
    for path,sha256 in provenance.items():need(digest(path)==sha256,'Bound input changed during CPU audit: '+path)
    result=dict(stage='actual_completed_natural128_result_audit',auditor_source_sha256=digest(__file__),runner_source_sha256=digest(runner.__file__),data_complete_sha256=PREPARED_SHA,
        primary_model_sha256=PRIMARY_SHA,primary_system_threshold=PRIMARY_THRESHOLD,models=frozen['models'],thresholds=frozen['thresholds'],prepared_source_roots=len(inputs),quality_failures=128-len(inputs),
        content_rows=len(rows),input_tokens=sum(map(len,tokens)),metrics_rows=len(calculated),metrics=calculated,source_count_rows=len(source),component_count_rows=len(components),message_projection_rows=len(messages),
        bootstrap_replicates=BOOTSTRAPS,six_pairs=True,paired_comparisons=len(paired),model_intervals=len(intervals),paired_comparisons_recomputed=paired,bootstrap_intervals_recomputed=intervals,
        maximum_metric_difference=metric_difference,maximum_metric_csv_difference=csv_difference,maximum_message_probability_difference=message_difference,maximum_interval_difference=interval_difference,maximum_paired_difference=paired_difference,
        macro_scope='Frozen production macro_f1 averages only truth-supported classes; standard union-truth/pred macro is separately diagnostic.',macro_scope_comparison=union_macro,
        role_support={role:int(rows.role.eq(role).sum()) for role in ROLES},human_report_rows=text_rows,plots=plots,source_status_rows=len(expected_status),provenance_sha256=provenance,
        independent_statistical_math=True,production_summary_or_bootstrap_called=False,no_probability_or_threshold_selection=True,test_probabilities_read=True,stored_validation_probabilities_read_for_completion_verification=True,
        c4_test_data_read=False,source_reply_replay_performed=False,gpu_used=False,models_or_thresholds_modified=False,formal_outputs_written=False)
    write(output,result);print(json.dumps(dict(output=str(output),metrics_rows=len(calculated),maximum_metric_difference=metric_difference,paired_comparisons=len(paired),model_intervals=len(intervals)),sort_keys=True));return result


def synthetic():
    rows=[];ledger=[]
    for i in range(128):
        uid=f'toy-{i:03d}';ok=i!=0;entry=dict(source_uid=uid,dataset='oasst' if i<64 else 'toxicchat0124',native_id=str(i),status='prepared' if ok else 'quality-failed',quality_failure=None if ok else 'toy failure',eligible_clean_test=ok and i%2==0,old_generated_collision=i%2==1,dev_assistant_collision=False,family_id=f'family-{i%32:02d}',dependency_component_id=f'component-{i%16:02d}' if ok else None)
        ledger.append(entry)
        if not ok:continue
        for mi,role in enumerate(OBSERVED):
            for pos in range(4):rows.append({k:entry[k] for k in ['source_uid','dataset','eligible_clean_test','family_id','dependency_component_id']}|dict(role=role,message_ix=mi,token_ix=mi*5+pos))
    frame=pd.DataFrame(rows);labels=row_labels(frame);probability={}
    for ni,name in enumerate(NAMES):
        p=np.full((len(frame),5),.035,np.float32);p[np.arange(len(frame)),labels]=.86
        # Deliberately create genuine false-positive CoT/tool argmax predictions.
        wrong=np.arange(ni+3,len(frame),31+ni);p[wrong]=.035;p[wrong,2 if ni%2 else 4]=.86
        # p(system) below threshold while still argmax: threshold and S recall differ.
        sysrows=np.flatnonzero(labels==0)[ni::29];p[sysrows]=[.38,.17,.15,.15,.15]
        p=p/p.sum(1,keepdims=True);probability[name]=p.astype(np.float32)
    return frame,probability,{name:.5 for name in NAMES},ledger


def preflight(output):
    from sklearn.metrics import confusion_matrix,f1_score,recall_score
    from types import SimpleNamespace
    from unittest.mock import patch
    folder=Path(tempfile.mkdtemp(prefix='confirmation-review-natural-result-',dir='/tmp'));checks=[]
    def passed(name,value=True):need(value,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,ValueError,KeyError,FileNotFoundError):checks.append(name)
        else:raise RuntimeError('Fault accepted: '+name)
    rows,p,thresholds,ledger=synthetic();records,source,messages,macro=independent_summary(rows,p,thresholds)
    passed('full synthetic127prepared yields24 model/stratum/dataset metrics',len(records)==24)
    y=row_labels(rows);v=metrics(p[NAMES[0]],y,.5);pred=p[NAMES[0]].argmax(1)
    passed('five-class confusion independently matches sklearn',v['confusion']==confusion_matrix(y,pred,labels=np.arange(5)).tolist())
    supported=np.unique(y);truth_macro=f1_score(y,pred,labels=supported,average='macro',zero_division=0);union=f1_score(y,pred,average='macro',zero_division=0)
    passed('FP-CoT/tool distinguish truth-supported frozen macro from standard union',abs(v['macro_f1']-truth_macro)<1e-15 and abs(v['macro_f1']-union)>.05 and v['recall']['cot'] is None and v['recall']['tool'] is None)
    passed('threshold detection recall differs from argmax S recall',v['system_recall']!=v['recall']['system'])
    passed('binary system recall matches sklearn',abs(v['system_recall']-recall_score(y==0,p[NAMES[0]][:,0]>=.5))<1e-15)
    # Production helpers are called ONLY on these synthetic arrays for schema/math parity.
    sys.path.insert(0,str(MODULES));import systemness_natural_test_confirm as runner
    expected,expected_source,expected_messages=runner.summarize(rows,p,thresholds)
    metric_error=compare_records(records,expected,['stratum','dataset','model'],'synthetic production metrics')
    compare_frame(source,expected_source,['stratum','dataset','model','source_uid'],'synthetic source counts',0)
    message_error=compare_frame(messages,expected_messages,['stratum','model','source_uid','message_ix','role','family_id','dependency_component_id'],'synthetic projections',2e-6)
    paired,intervals,components=independent_bootstrap(source);ep,ei,ec=runner.bootstrap(expected_source)
    pair_error=compare_records(paired,ep,['stratum','dataset','baseline_model','comparison_model','metric'],'synthetic six-pair exact draws')
    interval_error=compare_records(intervals,ei,['stratum','dataset','model','metric'],'synthetic model intervals')
    compare_frame(components,ec,['stratum','dataset','model','dependency_component_id'],'synthetic components',0)
    passed('independent24metrics source/components projections and2000six-pair draws match synthetic frozen helpers')
    passed('all source group counts pair across four models',len(paired)==6*6*9 and len(intervals)==6*4*9)
    one=source.copy();one['dependency_component_id']='one';pp,ii,_=independent_bootstrap(one)
    passed('one-component CIs unavailable',all(row['ci_available'] is False and row['ci95_low'] is None and row['ci_unavailable_reason']=='fewer_than_two_dependency_components' for row in pp+ii))
    never=source.copy();mask=never.model.eq(NAMES[0]);never.loc[mask,'fn']+=never.loc[mask,'tp'];never.loc[mask,'tn']+=never.loc[mask,'fp'];never.loc[mask,['tp','fp']]=0
    pp,ii,_=independent_bootstrap(never)
    passed('no-detection precision and paired difference remain N/A',all(row['value'] is None and row['defined_bootstrap_replicates']==0 and not row['ci_available'] for row in ii if row['model']==NAMES[0] and row['metric']=='system_precision') and all(row['gain_percentage_points'] is None for row in pp if row['baseline_model']==NAMES[0] and row['metric']=='system_precision'))
    empty=rows.iloc[:0].copy();empty_p={name:np.empty((0,5),np.float32) for name in NAMES};noobs,nc,nm,_=independent_summary(empty,empty_p,thresholds)
    en,enc,enm=runner.summarize(empty,empty_p,thresholds);compare_records(noobs,en,['stratum','dataset','model'],'noobs N/A schema')
    pp,ii,cc=independent_bootstrap(nc);passed('all-failed8 N/A records without invented source/probabilities/bootstrap',len(noobs)==8 and not pp and not ii and cc.empty and all(row['tokens']==0 and all(v is None for v in row['recall'].values()) for row in noobs))
    bad=source.iloc[1:].copy();reject('missing paired source rejected',lambda:independent_bootstrap(bad))
    bad=pd.concat([source,source.iloc[:1]],ignore_index=True);reject('duplicate model/source count rejected',lambda:independent_bootstrap(bad))
    corrupt=p[NAMES[0]].copy();corrupt[0,0]=np.nan;reject('nonfinite probabilities rejected',lambda:metrics(corrupt,y,.5))
    corrupt=p[NAMES[0]].copy();corrupt[0]*=.5;reject('unnormalized probabilities rejected',lambda:metrics(corrupt,y,.5))
    # CSV roundtrips include unsupported-role N/A rather than numeric zero.
    flat_metrics(records).to_csv(folder/'metrics.csv',index=False);compare_frame(read_csv(folder/'metrics.csv'),flat_metrics(records),['stratum','dataset','model'],'CSV N/A roundtrip')
    passed('CSV unsupported CoT/tool recalls roundtrip as N/A')
    lines=[]
    for row in records:
        values=[row[k] for k in ['system_precision','system_recall','system_f1','system_balanced_accuracy','system_false_positive_rate','accuracy']]+[row['recall'][role] for role in OBSERVED]
        lines.append(' '.join([row['model'],row['stratum'],row['dataset']]+['N/A' if value is None else f'{value:.6f}' for value in values]))
    (folder/'report.txt').write_text('\n'.join(lines));passed('24 human-readable rows match frozen six-digit values',report_text_audit(folder/'report.txt',records)==24)
    (folder/'report.txt').write_text('\n'.join(lines).replace('0.','1.',1));reject('human report numeric corruption rejected',lambda:report_text_audit(folder/'report.txt',records))
    # Completion gate must be first; inject fake module with nonexistent OUT.
    never_out=folder/'absent-output';fake=SimpleNamespace(OUT=never_out)
    with patch.dict(sys.modules,{'systemness_natural_test_confirm':fake}):reject('actual refuses missing terminal complete before cache access',lambda:actual(folder/'never.json'))
    passed('failed terminal gate creates no output',not never_out.exists() and not (folder/'never.json').exists())
    # Minimal synthetic5x4 PDFs verify source titles and all20 failed placeholders.
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    plots=folder/'plots';plots.mkdir();chosen=[sorted(row['source_uid'] for row in ledger)[i] for i in np.linspace(0,127,12,dtype=int)];mapping={row['source_uid']:row for row in ledger}
    for serial,uid in enumerate(chosen):
        fig,axes=plt.subplots(5,4,figsize=(4,2));fig.suptitle(uid)
        for ax in axes.flat:
            if mapping[uid]['status']=='quality-failed':ax.text(.5,.5,'Quality failed\nNo invented observations',fontsize=2)
            else:ax.scatter([0,1],[.2,.8],s=1)
        for ext in ['png','pdf']:fig.savefig(plots/f'natural-test-{serial:02d}.{ext}',dpi=40)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(3,1));ax.plot([0,1]);
    for ext in ['png','pdf']:fig.savefig(plots/f'natural-test-summary.{ext}',dpi=40)
    plt.close(fig)
    gallery=dict(source_uids=chosen,observed_truth_roles=OBSERVED,model_columns=4,probability_rows=5,png=13,pdf=13,failed_ids_replaced=False,unsmoothed_probabilities=True)
    (plots/'gallery-index.md').write_text('\n'.join(f'[{uid}](natural-test-{i:02d}.png)' for i,uid in enumerate(chosen)))
    verified_plots=gallery_audit(plots,gallery,ledger,fixture=True);passed('13PNG13PDF fixed12 IDs with20-panel failed-source placeholder verified',len(verified_plots)==13 and verified_plots[0]['failed_placeholder_verified'])
    wrong=deepcopy(gallery);wrong['source_uids']=wrong['source_uids'][::-1];reject('gallery ID replacement/reorder rejected',lambda:gallery_audit(plots,wrong,ledger,fixture=True))
    result=dict(stage='synthetic_natural_result_audit_preflight',auditor_source_sha256=digest(__file__),runner_source_sha256=digest(runner.__file__),checks=checks,checks_passed=len(checks),
        synthetic_roots=128,synthetic_prepared_roots=127,synthetic_content_rows=len(rows),metric_rows=len(records),source_count_rows=len(source),message_projection_rows=len(messages),component_rows=len(components),paired_comparisons=len(paired),intervals=len(intervals),
        maximum_metric_difference=metric_error,maximum_message_difference=message_error,maximum_paired_difference=pair_error,maximum_interval_difference=interval_error,
        plots_png=13,plots_pdf=13,fixture_directory=str(folder),test_data_or_results_read=False,stored_validation_probabilities_read=False,gpu_used=False,formal_outputs_written=False,
        production_summary_called_only_on_synthetic_arrays=True,macro_truth_supported=v['macro_f1'],macro_standard_union=float(union))
    write(output,result);print(json.dumps(dict(output=str(output),checks_passed=len(checks),maximum_metric_difference=metric_error,maximum_paired_difference=pair_error),sort_keys=True));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group();action.add_argument('--actual',action='store_true');action.add_argument('--preflight',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.actual:actual(args.output or Path('/tmp/confirmation-review-natural-result-actual.json'))
    elif args.preflight:preflight(args.output or Path('/tmp/confirmation-review-natural-result-preflight.json'))
    else:parser.print_help()
if __name__=='__main__':main()
