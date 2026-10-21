"""Independent CPU-only guard-first DEV decision audit; default help.

--preflight: small manufactured arrays only. --actual: root only after terminal
producer completion. No producer imports, training, LM or test data access.
"""
import argparse,copy,hashlib,itertools,json,os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
OUT=HERE/'outputs/local-gptoss120b/systemness-complementary-guard-calibration'
SOURCE=HERE/'systemness_complementary_guard_calibration.py'
SOURCE_SHA='70c6ae046cbbb5485047733b593afe23a1d61376f047b2ad29e75c692e7bb85d'
PROTOCOL_SHA='7edc1b5ba2c55a21ebe01bc5af9f850c4865a933bb9bc4b1274520d6b00ec64a'
UPSTREAM_AUDIT_SHA='7bac44572d8ea37fc4cae87d447061228496a218eadaa92583a4a96bb3c4989c'
MODEL_SHA='ef92b421007b7d0c7a11f13e91008e6d2e39b34b730c6ca7d6ecbf03590236ff'
ROLES=['system','user','cot','assistant','tool']
BANKS=['old','c4','public'];COUNTS={'old':173032,'c4':61440,'public':16713}
OLD5=['old-'+s for s in ['single','multi','natural','long-single','long-full']]
PUBLIC=['public-natural-all','public-natural-oasst','public-natural-toxicchat0124']
FULL=['old-long-full','old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']
GROUPS=OLD5+['old-long-full-tail','new-C4-long-full','new-C4-long-full-tail']+PUBLIC
FEASIBLE=OLD5+['new-C4-long-full']+PUBLIC
STRICT=['matched-max33','oldhead-max33','ensemble-max33','zero-max33','priorBinary-max33']
GUARDS=['matched-max33','oldhead-max33','ensemble-max33','oldhead-oldmean','ensemble-original','zero-oldmean','zero-max33','priorBinary-max33']
BINARY=['system_threshold','system_accuracy','system_balanced_accuracy','system_precision','system_recall','system_f1','system_false_positive_rate','system_detection_by_role','system_confusion']
TOL=1e-12
def require(x,message):
    if not bool(x):raise RuntimeError(message)
def sha(path):
    path=Path(path);require('/workspace' not in str(path),'Forbidden filesystem path')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def equal(a,b,path='value'):
    require(type(a) is type(b),f'Type differs at {path}')
    if isinstance(a,dict):
        require(set(a)==set(b),f'Keys differ at {path}')
        for k in a:equal(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        require(len(a)==len(b),f'Length differs at {path}')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    elif isinstance(a,float):require(np.isfinite(a) and np.isfinite(b) and abs(a-b)<=TOL,f'Float differs at {path}')
    else:require(a==b,f'Value differs at {path}')
def atomic(path,value):
    path=Path(path);require(str(path.resolve()).startswith('/tmp/'),'Audit output must be /tmp')
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def gate_first():
    require(isinstance(PROTOCOL_SHA,str) and len(PROTOCOL_SHA)==64,'Root protocol pin missing: actual closed before artifact reads.')
    # Completion and no-live producer are checked before protocol, caches or p.
    require((OUT/'complete.json').is_file(),'Calibration not terminal complete.')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():continue
        try:parts=(process/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if any(v.endswith(b'/systemness_complementary_guard_calibration.py') or v==b'systemness_complementary_guard_calibration.py' for v in parts):
            require(not any(v in [b'--actual',b'--declare'] for v in parts),'Calibration producer still live.')
    complete=read(OUT/'complete.json')
    require(complete['stage']=='completed_guard_constrained_development_calibration' and complete['source_sha256']==SOURCE_SHA and complete['protocol_sha256']==PROTOCOL_SHA and complete['training_updates']==0 and complete['gpu_used'] is False and complete['tests_read'] is False and complete['primary_changed'] is False,'Completed calibration scope changed.')
    return complete
def metrics(score,y,tau):
    """Direct actual F32 comparison, independent of sorted/searchsorted code."""
    truth=y==0;detect=score>=np.float32(tau);tp=int(np.count_nonzero(truth&detect));fp=int(np.count_nonzero(~truth&detect));fn=int(np.count_nonzero(truth&~detect));tn=int(np.count_nonzero(~truth&~detect))
    div=lambda a,b:float(a/b) if b else None
    r=div(tp,tp+fn);s=div(tn,tn+fp)
    return dict(system_threshold=float(tau),system_accuracy=div(tp+tn,len(y)),system_balanced_accuracy=(r+s)/2 if r is not None and s is not None else None,system_precision=div(tp,tp+fp),system_recall=r,system_f1=div(2*tp,2*tp+fp+fn),system_false_positive_rate=div(fp,fp+tn),system_detection_by_role={role:float(detect[y==i].mean()) if np.any(y==i) else None for i,role in enumerate(ROLES)},system_confusion=dict(tp=tp,fn=fn,fp=fp,tn=tn))
def light(values,baseline):
    numbers=[values[g][k] for g in GROUPS for k in ['system_precision','system_recall','system_balanced_accuracy']]
    return dict(groups={g:dict(values[g],accuracy=baseline['groups'][g]['accuracy']) for g in GROUPS},min33=float(min(numbers)) if all(x is not None and np.isfinite(x) for x in numbers) else None,old5_mean_f1=float(np.mean([values[g]['system_f1'] for g in OLD5])))
def one_guard(candidate,control,legacy=False):
    c=candidate['groups'];b=control['groups'];reasons=[]
    oldf={g:b[g]['system_f1']-c[g]['system_f1'] for g in OLD5[:4]};olda={g:b[g]['accuracy']-c[g]['accuracy'] for g in OLD5[:4]};full={g:c[g]['system_f1']-b[g]['system_f1'] for g in FULL};mean=control['old5_mean_f1']-candidate['old5_mean_f1'];pubf={g:b[g]['system_f1']-c[g]['system_f1'] for g in PUBLIC};puba={g:b[g]['accuracy']-c[g]['accuracy'] for g in PUBLIC}
    tests=[(max(oldf.values())>.02+TOL,'old4_system_f1_loss'),(max(olda.values())>.01+TOL,'old4_five_accuracy_loss'),(mean>.005+TOL,'old5_mean_f1_loss'),(min(full.values())< -TOL,'C4_full_tail_f1_regression'),(not legacy and max(pubf.values())>.02+TOL,'public_system_f1_loss'),(not legacy and max(puba.values())>.01+TOL,'public_five_accuracy_loss'),(max(c[g]['system_false_positive_rate'] for g in FEASIBLE)>.05+TOL,'FPR_above_limit'),(not all(np.isfinite(c[g][k]) for g in GROUPS for k in ['system_f1','accuracy','system_false_positive_rate']),'nonfinite_metrics')]
    reasons=[name for failed,name in tests if failed]
    return dict(eligible=not reasons,rejection_reasons=reasons,old4_f1_losses=oldf,old4_accuracy_losses=olda,old5_mean_f1_loss=mean,C4_full_tail_f1_gains=full,public_f1_losses=pubf,public_accuracy_losses=puba,legacy_savedGPU_only=legacy)
def admissibility(candidate,controls,legacy):
    require(set(controls)==set(GUARDS),'Fixed control inventory differs.')
    minimum=candidate['min33'];reasons=[];gs={};gains={}
    if minimum is None or not np.isfinite(minimum) or minimum<.9:reasons.append('absolute_min33_below_90_percent')
    if any(candidate['groups'][g]['system_false_positive_rate']>.05 for g in FEASIBLE):reasons.append('original_exact_FPR_feasibility')
    for name in GUARDS:
        proof=one_guard(candidate,controls[name]);gs[name]=proof;reasons += [name+':'+r for r in proof['rejection_reasons']]
        if name in STRICT:
            gain=minimum-controls[name]['min33'] if minimum is not None and controls[name]['min33'] is not None else None;gains[name]=gain
            if gain is None or not np.isfinite(gain) or gain<=TOL:reasons.append(name+':min33_not_strictly_improved')
    proof=one_guard(candidate,legacy,True);gs['original-savedGPU']=proof;reasons += ['original-savedGPU:'+r for r in proof['rejection_reasons']]
    return dict(eligible=not reasons,rejection_reasons=reasons,guards=gs,min33_gains=gains)
def decide(values,baseline,controls,legacy,grid):
    ledger=[];candidates=[];original=[];gates=[]
    for i,tau in enumerate(grid):
        result=light(values[i],baseline);gate=admissibility(result,controls,legacy);gates.append(gate)
        fpr=all(result['groups'][g]['system_false_positive_rate']<=.05 for g in FEASIBLE)
        ledger.append(dict(index=i,system_threshold=float(tau),effective_F32_threshold=float(np.float32(tau)),min33=result['min33'],old5_mean_f1=result['old5_mean_f1'],original_FPR_feasible=fpr,eligible=gate['eligible'],rejection_reasons=gate['rejection_reasons']))
        if fpr and result['min33'] is not None:original.append(i)
        if gate['eligible']:candidates.append(i)
    require(original,'No defined original FPR-feasible threshold.')
    rank=lambda i:(ledger[i]['min33'],ledger[i]['old5_mean_f1'],-float(grid[i]))
    old=max(original,key=rank);best=max(candidates,key=rank) if candidates else None
    return best,old,ledger,gates
def groups_for(rows):
    groups={}
    for profile in ['single','multi','natural','long-single','long-full']:groups['old-'+profile]=('old',np.flatnonzero(rows['old'].profile.eq(profile)))
    for bank,name in [('old','old'),('c4','new-C4')]:
        ix=np.flatnonzero(rows[bank].profile.eq('long-full')) if bank=='old' else np.arange(len(rows[bank]));part=rows[bank].iloc[ix]
        if 'body_offset' in part:offset=part.body_offset.to_numpy()
        else:offset=part.token_ix.to_numpy()-part.groupby(['prompt_ix','message_ix']).token_ix.transform('min').to_numpy()
        groups[name+'-long-full-tail']=(bank,ix[offset>=512])
        if bank=='c4':groups[name+'-long-full']=(bank,ix)
    groups[PUBLIC[0]]=('public',np.arange(len(rows['public'])))
    for name,dataset in zip(PUBLIC[1:],['oasst','toxicchat0124']):groups[name]=('public',np.flatnonzero(rows['public'].dataset.eq(dataset)))
    require(set(groups)==set(GROUPS) and all(len(ix) for _,ix in groups.values()),'Independent DEV groups incomplete.')
    return groups
def metadata_closure(static,records,zero,controls,original,candidate,audit):
    index={v['record']:v['validation'] for v in records}
    mapped={name:zero['records'][name] if name in zero['records'] else index[name] for name in GUARDS}
    require(static['baseline_validation']==index['new-max33'] and static['controls']==mapped and static['legacy']==controls['original_savedGPU_validation'] and static['original_selection']==original,'Static numeric controls not copied exactly from bound producer JSON.')
    require(candidate['model_sha256']==audit['model_sha256']['new']==static['model_sha256']==MODEL_SHA,'Parent fixed model binding differs.')
    for bank in BANKS:
        p=static['probabilities'][bank];r=static['source_rows'][bank]
        require(isinstance(p['sha256'],str) and len(p['sha256'])==64 and isinstance(r['sha256'],str) and len(r['sha256'])==64 and p['sha256']==candidate['probability_sha256'][bank]==audit['probability_bindings'].get(p['path']) and r['sha256']==audit['provenance_sha256'].get(r['path']),'Parent certificate p/row binding differs: '+bank)
def actual(output):
    complete=gate_first();provenance={str(Path(__file__).resolve()):sha(__file__)}
    def bind(path,expected=None):
        path=Path(path);value=sha(path);require(expected is None or value==expected,'Closed bytes differ: '+str(path));provenance[str(path.resolve())]=value;return value
    bind(SOURCE,SOURCE_SHA);bind(OUT/'protocol.json',PROTOCOL_SHA);complete_sha=bind(OUT/'complete.json');decl=read(OUT/'protocol.json');static=decl['static'];proto=decl['protocol']
    require(proto['groups']==GROUPS and proto['feasible']==FEASIBLE and proto['strict_controls']==STRICT and proto['guard_controls']==GUARDS+['original-savedGPU'] and proto['absolute_min33']==.9 and proto['max_FPR']==.05 and proto['tolerance']==TOL and proto['threshold_budget']==501 and proto['training_updates']==0 and proto['tests_read'] is False and proto['primary_changed'] is False,'Protocol rule inventory changed.')
    require(static['model_sha256']==MODEL_SHA and static['source_sha256']==SOURCE_SHA and static['upstream_actual_audit_sha256']==UPSTREAM_AUDIT_SHA and static['original_selection']['promoted'] is False,'Original fixed failed trial/model ownership changed.')
    for path,want in static['files_sha256'].items():bind(path,want)
    parent=HERE/'outputs/local-gptoss120b/systemness-complementary-head-readout';parent_audit=parent/'actual-audit/actual-development-audit.json';bind(parent_audit,UPSTREAM_AUDIT_SHA)
    for name in ['records.json','zero-controls.json','controls.json','selection.json','candidates/new.json']:
        want=static['files_sha256'].get(str(parent/name));require(isinstance(want,str) and want==provenance.get(str((parent/name).resolve())),'Parent metadata absent from closed files: '+name)
    metadata_closure(static,read(parent/'records.json'),read(parent/'zero-controls.json'),read(parent/'controls.json'),read(parent/'selection.json'),read(parent/'candidates/new.json'),read(parent_audit))
    for name,want in complete['artifact_sha256'].items():bind(OUT/name,want)
    require(set(complete['artifact_sha256'])=={'threshold-ledger.json','selection.json','validation-metrics.csv','report.txt'},'Calibration artifact inventory differs.')
    decision=read(OUT/'selection.json');savedledger=read(OUT/'threshold-ledger.json')
    require(decision['candidate']=='new-guard-max33' and decision['model_path']==static['model_path'] and decision['model_sha256']==MODEL_SHA and decision['protocol_sha256']==PROTOCOL_SHA and decision['source_sha256']==SOURCE_SHA and decision['upstream_actual_audit_sha256']==UPSTREAM_AUDIT_SHA and decision['original_trial_unchanged'] is True and decision['primary_changed'] is False and decision['tests_read'] is False and decision['no_fallback'] is True and decision['new_independent_confirmation_required'] is True,'Decision scope changed.')
    import pyarrow.parquet as pq
    rows={};p={};bindings={}
    for bank in BANKS:
        item=static['source_rows'][bank];rp=Path(item['path']);bindings[str(rp)]=bind(rp,item['sha256'])
        columns=[k for k in ['cache_ix','sample_ix','split','wrapper','role','profile','body_offset','token_ix','prompt_ix','message_ix','dataset'] if k in pq.ParquetFile(rp).schema.names]
        table=pd.read_parquet(rp,columns=columns);require(np.array_equal(table.cache_ix,np.arange(len(table))) and table.sample_ix.is_unique and set(table.split)=={'fit','validation'} and table.wrapper.eq('native').all(),'DEV parent ownership/split changed.')
        frame=table[table.split.eq('validation')].copy().reset_index(drop=True);require(len(frame)==item['rows']==COUNTS[bank] and array_sha(frame.sample_ix.to_numpy())==item['sample_ix_sha256'],'DEV sample/order differs.')
        if bank=='public':require(set(frame.role)=={'system','user','assistant'},'Public must remain SUA-only.')
        rows[bank]=frame;item=static['probabilities'][bank];pp=Path(item['path']);bindings[str(pp)]=bind(pp,item['sha256'])
        with np.load(pp,allow_pickle=False) as z:
            require(set(z.files)=={'p','sample_ix'} and np.array_equal(z['sample_ix'],frame.sample_ix.to_numpy()),'Probability sample namespace differs.');value=z['p'].copy()
        require(value.shape==(len(frame),5) and value.dtype==np.float32 and np.isfinite(value).all() and (value>=0).all() and (value<=1.00001).all() and np.allclose(value.sum(1),1,atol=2e-5,rtol=0),'Invalid fixed DEV probabilities.');p[bank]=value
    equal(complete['data_bindings'],bindings,'data_bindings');equal(complete['probability_bindings'],{s['path']:s['sha256'] for s in static['probabilities'].values()},'probability_bindings')
    grid=np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(np.concatenate([p[b][:,0] for b in BANKS]),np.linspace(0,1,301)),1.000001]);require(len(grid)<=501,'Grid expanded.')
    groups=groups_for(rows);y={b:rows[b].role.map({r:i for i,r in enumerate(ROLES)}).to_numpy(np.int64) for b in BANKS}
    values=[{g:metrics(p[b][ix,0],y[b][ix],tau) for g,(b,ix) in groups.items()} for tau in grid]
    best,old,ledger,gates=decide(values,static['baseline_validation'],static['controls'],static['legacy'],grid);equal(ledger,savedledger,'all_threshold_ledger')
    baseline=static['baseline_validation'];oldresult=light(values[old],baseline)
    require(float(grid[old])==baseline['system_threshold'],'Original max33 tau differs.')
    equal(oldresult['min33'],baseline['min33'],'original_min33');equal(oldresult['old5_mean_f1'],baseline['old5_mean_f1'],'original_mean')
    for g in GROUPS:
        for k in BINARY:equal(values[old][g][k],baseline['groups'][g][k],'old/'+g+'/'+k)
    for k in ['rejection_reasons','guards','min33_gains']:equal(gates[old][k],static['original_selection'][k],'original_gate/'+k)
    proof=dict(grid_thresholds=len(grid),threshold_budget=501,admissible_thresholds=sum(v['eligible'] for v in ledger),original_best_index=old,original_threshold=float(grid[old]),original_min33=oldresult['min33'],original_old5_mean_f1=oldresult['old5_mean_f1'],selected_index=best,no_fallback=True)
    equal(proof,decision['grid_proof'],'grid_proof');require(decision['eligible']==complete['eligible']==(best is not None),'Feasibility decision differs.')
    require(decision['system_threshold']==(float(grid[best]) if best is not None else None),'Selected tie threshold differs.')
    if best is None:require(decision['validation'] is None and decision['gate'] is None and len(pd.read_csv(OUT/'validation-metrics.csv'))==0,'No-candidate fallback invented.')
    else:
        equal(gates[best],decision['gate'],'selected_gate');selected=decision['validation'];expected=copy.deepcopy(baseline);expected['system_threshold']=float(grid[best])
        expected['min33']=light(values[best],baseline)['min33'];expected['old5_mean_f1']=light(values[best],baseline)['old5_mean_f1']
        for g in GROUPS:expected['groups'][g].update(values[best][g])
        for b in BANKS:expected['separate_bank_descriptive'][b].update(metrics(p[b][:,0],y[b],grid[best]))
        expected['pooled_descriptive_only'].update(metrics(np.concatenate([p[b][:,0] for b in BANKS]),np.concatenate([y[b] for b in BANKS]),grid[best]))
        for group in GROUPS:
            require({k:v for k,v in selected['groups'][group].items() if k not in BINARY}=={k:v for k,v in baseline['groups'][group].items() if k not in BINARY},'Threshold-independent group fields not copied exactly.')
        for bank in BANKS:
            require({k:v for k,v in selected['separate_bank_descriptive'][bank].items() if k not in BINARY}=={k:v for k,v in baseline['separate_bank_descriptive'][bank].items() if k not in BINARY},'Threshold-independent bank fields not copied exactly.')
        require({k:v for k,v in selected['pooled_descriptive_only'].items() if k not in BINARY}=={k:v for k,v in baseline['pooled_descriptive_only'].items() if k not in BINARY},'Threshold-independent pooled fields not copied exactly.')
        equal(expected,selected,'selected_all_binary_and_copied_fields')
        csv=pd.read_csv(OUT/'validation-metrics.csv');require(len(csv)==12 and set(csv.group)==set(GROUPS)|{'pooled_descriptive_only'} and csv.record.eq('new-guard-max33').all(),'12row calibration CSV coverage/record differs.')
        for _,row in csv.iterrows():
            value=selected['pooled_descriptive_only'] if row.group=='pooled_descriptive_only' else selected['groups'][row.group]
            for key,want in value.items():
                if isinstance(want,(dict,list)):continue
                got=row[key]
                if want is None:require(pd.isna(got),'CSV N/A changed')
                elif isinstance(want,(float,int)):require(abs(float(got)-want)<=TOL,'CSV numeric differs: '+key)
                else:require(got==want,'CSV value differs: '+key)
    # Every already bound byte is rechecked after arithmetic; no source/output change accepted.
    for path,want in list(provenance.items()):require(sha(path)==want,'Artifact changed during audit: '+path)
    result=dict(stage='actual_completed_guard_constrained_calibration_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=sha(__file__),calibration_source_sha256=SOURCE_SHA,calibration_protocol_sha256=PROTOCOL_SHA,calibration_complete_sha256=complete_sha,calibration_selection_sha256=sha(OUT/'selection.json'),upstream_actual_audit_sha256=UPSTREAM_AUDIT_SHA,model_path=decision['model_path'],model_sha256=MODEL_SHA,decision=decision,threshold_grid_size=len(grid),independently_recomputed_group_thresholds=len(grid)*11,original_max33_and_failed_gate_reproduced=True,threshold_free_metrics_copied_exactly=True,probability_bindings=complete['probability_bindings'],provenance_sha256=provenance,test_data_or_results_read=False,gpu_used=False,training_updates=0,new_independent_confirmation_required=True)
    atomic(output,result);print(json.dumps(dict(report_path=str(output),report_sha256=sha(output),eligible=decision['eligible'],threshold=decision['system_threshold'],thresholds=len(grid))),flush=True)
def preflight():
    checks=[]
    score=np.array([.5,np.nextafter(np.float32(.5),np.float32(1)),.5,.9],np.float32);y=np.array([0,0,1,3]);tau=float(np.quantile(score[:2].astype(np.float64),1/6));v=metrics(score,y,tau)
    require(v['system_confusion']=={'tp':2,'fn':0,'fp':2,'tn':0},'F32 alias semantics changed.');checks.append('Direct F32 weak scalar boundary')
    baseline={'groups':{g:{'system_f1':.98,'accuracy':.99,'system_false_positive_rate':.01} for g in GROUPS},'old5_mean_f1':.98,'min33':.9};controls={name:copy.deepcopy(baseline) for name in GUARDS};legacy=copy.deepcopy(baseline)
    values=[]
    for i in range(3):
        row={g:dict(system_precision=.95 if i==1 else .94,system_recall=.95 if i==1 else .94,system_balanced_accuracy=.95 if i==1 else .94,system_f1=.98,system_false_positive_rate=.01) for g in GROUPS}
        if i==1:row['old-natural']['system_f1']=.959
        values.append(row)
    best,old,ledger,gates=decide(values,baseline,controls,legacy,np.array([.4,.5,.6]));require((best,old)==(0,1) and not ledger[1]['eligible'],'Guard-first differs from max-then-guard.');checks.append('Guard-first and legacy max33 independently distinct')
    for row in values:
        for value in row.values():value['system_recall']=.89
    require(decide(values,baseline,controls,legacy,np.array([.4,.5,.6]))[0] is None,'No fallback violated.');checks.append('No admissible threshold returns None')
    x=copy.deepcopy(baseline);x['min33']=.95
    require(len(admissibility(x,controls,legacy)['guards'])==9,'Guard cardinality changed');checks.append('5strict and 8guards plus savedGPU')
    for name in STRICT:
        c=copy.deepcopy(controls);c[name]['min33']=.95;require(not admissibility(x,c,legacy)['eligible'],'Strict equality passed.')
    checks.append('Every strict control rejects equality')
    # Independent remapping uses original producer JSON, not collect_static's self-description.
    fake_records=[dict(record=name,validation=copy.deepcopy(baseline)) for name in set(GUARDS)|{'new-max33'} if not name.startswith('zero-')];fake_zero={'records':{name:copy.deepcopy(baseline) for name in GUARDS if name.startswith('zero-')}}
    fake_original={'promoted':False};fake_candidate={'model_sha256':MODEL_SHA,'probability_sha256':{b:'a'*64 for b in BANKS}};fake_audit={'model_sha256':{'new':MODEL_SHA},'probability_bindings':{b+'p':'a'*64 for b in BANKS},'provenance_sha256':{b+'r':'b'*64 for b in BANKS}}
    fake_static={'baseline_validation':copy.deepcopy(baseline),'controls':copy.deepcopy(controls),'legacy':copy.deepcopy(legacy),'original_selection':fake_original,'model_sha256':MODEL_SHA,'probabilities':{b:{'path':b+'p','sha256':'a'*64} for b in BANKS},'source_rows':{b:{'path':b+'r','sha256':'b'*64} for b in BANKS}}
    metadata_closure(fake_static,fake_records,fake_zero,{'original_savedGPU_validation':legacy},fake_original,fake_candidate,fake_audit);checks.append('Parent JSON/control/p/row closure reconstructed independently')
    bad=copy.deepcopy(fake_static);bad['controls'][GUARDS[0]]['old5_mean_f1']=.979
    try:metadata_closure(bad,fake_records,fake_zero,{'original_savedGPU_validation':legacy},fake_original,fake_candidate,fake_audit)
    except RuntimeError:pass
    else:raise RuntimeError('Wrong static control copied from parent accepted')
    checks.append('Swapped or altered static control rejected')
    bad=copy.deepcopy(fake_static);bad['probabilities']['old']['sha256']='c'*64
    try:metadata_closure(bad,fake_records,fake_zero,{'original_savedGPU_validation':legacy},fake_original,fake_candidate,fake_audit)
    except RuntimeError:pass
    else:raise RuntimeError('Uncertified static probability binding accepted')
    checks.append('Wrong parent probability SHA rejected')
    bad=copy.deepcopy(fake_static);bad['source_rows']['public']['sha256']='c'*64
    try:metadata_closure(bad,fake_records,fake_zero,{'original_savedGPU_validation':legacy},fake_original,fake_candidate,fake_audit)
    except RuntimeError:pass
    else:raise RuntimeError('Uncertified static row binding accepted')
    checks.append('Wrong parent row SHA rejected')
    saved_pin=globals()['PROTOCOL_SHA'];globals()['PROTOCOL_SHA']=None
    try:
        try:gate_first()
        except RuntimeError:pass
        else:raise RuntimeError('Actual unpinned gate opened')
    finally:globals()['PROTOCOL_SHA']=saved_pin
    checks.append('Missing protocol closes actual before IO')
    folder=Path(tempfile.mkdtemp(prefix='confirmation-review-guard-calibration-results-',dir='/tmp'));output=folder/'report.json';atomic(output,dict(stage='synthetic_guard_calibration_results_audit',source_sha256=sha(__file__),checks=checks,checks_passed=len(checks),actual_artifacts_or_probabilities_read=False,gpu_used=False));print(json.dumps(dict(report_path=str(output),report_sha256=sha(output),script_sha256=sha(__file__))),flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group();a.add_argument('--preflight',action='store_true');a.add_argument('--actual',action='store_true');p.add_argument('--output',type=Path,default=Path('/tmp/confirmation_review_guard_calibration_results_audit.json'));args=p.parse_args()
    if args.preflight:preflight()
    elif args.actual:require(str(args.output.resolve()).startswith('/tmp/'),'Only /tmp output allowed.');actual(args.output)
    else:p.print_help()
if __name__=='__main__':main()
