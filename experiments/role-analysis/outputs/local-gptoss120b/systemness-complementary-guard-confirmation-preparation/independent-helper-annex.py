"""Independent new certificate-layer review, AST+fake metadata only."""
import ast,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np

HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
SOURCE=HERE/'systemness_complementary_guard_confirm_snapshot.py'
PIN='9b3f2afdda695e1e6f5ff74952877af36f559a6320069fe3d5b9ff2781cb3e21'
CHECKS=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(name,ok=True):
    if not ok:raise AssertionError(name)
    CHECKS.append(name)
def reject(name,fn):
    try:fn()
    except (RuntimeError,KeyError,ValueError):CHECKS.append(name)
    else:raise AssertionError('Accepted '+name)
def load():
    t=ast.parse(SOURCE.read_text());ns={'__file__':str(SOURCE),'__name__':'isolated_synthetic_guard_snapshot'}
    exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef))],type_ignores=[]),str(SOURCE),'exec'),ns);return ns,t
def main():
    check('Exact held source pin',sha(SOURCE)==PIN);m,tree=load();pinned=['CALIBRATION_SOURCE_SHA','CALIBRATION_PROTOCOL_SHA','CALIBRATION_COMPLETE_SHA','CALIBRATION_DECISION_SHA','CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA','CALIBRATION_ACTUAL_AUDIT_SHA','CALIBRATION_ACTUAL_AUDIT_PATH']
    def forbidden(*a,**k):raise AssertionError('Actual IO forbidden')
    m['read']=m['digest']=forbidden;m['np']=SimpleNamespace(load=forbidden)
    for name in pinned:
        for key in pinned:m[key]=Path('/tmp/fake') if key.endswith('_PATH') else 'a'*64
        m[name]=None
        reject('Missing '+name+' blocks build before IO',m['build_conditional_snapshot'])
    reject('Public verify unknownpin before content or files',lambda:m['verify_closed_snapshot']({}))
    reject('Public declare unknownpin before files',lambda:m['declare_snapshot'](snapshot={}))
    m['np']=np;m['pin_gate']=lambda:None
    groups=['old-'+p for p in ['single','multi','natural','long-single','long-full']]+['old-long-full-tail','new-C4-long-full','new-C4-long-full-tail','public-natural-all','public-natural-oasst','public-natural-toxicchat0124'];feasible=groups[:5]+['new-C4-long-full']+groups[-3:]
    metric=dict(system_precision=.94,system_recall=.94,system_balanced_accuracy=.94,system_false_positive_rate=.01,system_f1=.98,accuracy=.97,macro_f1=.96,nll=.1,recall={r:.9 for r in m['ROLES']},confusion=[[1]*5 for _ in range(5)],system_auroc=.99,system_average_precision=.99)
    value=dict(system_threshold=.6,min33=.94,old5_mean_f1=.98,groups={g:copy.deepcopy(metric) for g in groups})
    index={name:dict(record=name,validation=copy.deepcopy(value)|dict(min33=.91)) for name in m['DEVELOPMENT_RECORDS']};zero={name:copy.deepcopy(value)|dict(min33=.91) for name in ['zero-oldmean','zero-max33']};legacy=copy.deepcopy(value)
    protocol=dict(models=1,training_updates=0,fixed_model_sha256=m['NEW_MODEL_SHA'],threshold_budget=501,strict_controls=m['STRICT'],guard_controls=m['GUARDS']+['original-savedGPU'],absolute_min33=.9,max_FPR=.05,tolerance=1e-12,tests_read=False,primary_changed=False,groups=groups,feasible=feasible,floor=dict(old4_f1_loss=.02,old4_accuracy_loss=.01,old5_mean_f1_loss=.005,full_tail_nonregression=['old-long-full','old-long-full-tail','new-C4-long-full','new-C4-long-full-tail'],public_f1_loss=.02,public_accuracy_loss=.01))
    gate=dict(eligible=True,rejection_reasons=[],guards={n:dict(eligible=True,rejection_reasons=[],legacy_savedGPU_only=n=='original-savedGPU') for n in m['GUARDS']+['original-savedGPU']},min33_gains={n:.94-.91 for n in m['STRICT']})
    decision=dict(candidate='new-guard-max33',eligible=True,no_fallback=True,model_sha256=m['NEW_MODEL_SHA'],model_path=str(m['model_paths']()['new']),system_threshold=.6,validation=value,gate=gate,grid_proof=dict(no_fallback=True,selected_index=0,grid_thresholds=2,threshold_budget=501,admissible_thresholds=1),original_trial_unchanged=True,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)
    check('Eligible-only schema accepted without promoted',m['validate_calibrated_decision'](decision,protocol,index,zero,legacy)==.6 and 'promoted' not in decision)
    for title,change in [
        ('Ineligible decision',lambda d:d.update(eligible=False)),
        ('Old failed candidate relabelling',lambda d:d.update(candidate='new-max33')),
        ('Original model substitution',lambda d:d.update(model_sha256='b'*64)),
        ('One guard failed',lambda d:d['gate']['guards']['original-savedGPU'].update(eligible=False)),
        ('Strict equality gain',lambda d:d['gate']['min33_gains'].update({m['STRICT'][0]:0.})),
        ('One absolute stat under90',lambda d:d['validation']['groups'][groups[0]].update(system_recall=.89999)),
        ('Exact FPR no tolerance expansion',lambda d:d['validation']['groups'][feasible[0]].update(system_false_positive_rate=.05+1e-13)),
        ('Fiveclass change despite same p claim',lambda d:d['validation']['groups'][groups[0]].update(accuracy=.98)),
        ('AUC change despite only tau decision',lambda d:d['validation']['groups'][groups[0]].update(system_auroc=.9))]:
        bad=copy.deepcopy(decision);change(bad);reject(title,lambda:m['validate_calibrated_decision'](bad,protocol,index,zero,legacy))
    # Metadata-only adapter fixture; all read/digest callbacks resolve fake dicts, never the filesystem.
    root=Path(tempfile.mkdtemp(prefix='confirmation-review-guard-snapshot-',dir='/tmp'));m['TRAINER']=root/'parent';m['CALIBRATION']=root/'cal';m['CALIBRATION_SOURCE']=root/'cal.py';m['CALIBRATION_ACTUAL_AUDIT_PATH']=root/'audit.json'
    m['CALIBRATION_SOURCE_SHA']='1'*64;m['CALIBRATION_PROTOCOL_SHA']='2'*64;m['CALIBRATION_COMPLETE_SHA']='3'*64;m['CALIBRATION_DECISION_SHA']='4'*64;m['CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA']='5'*64;m['CALIBRATION_ACTUAL_AUDIT_SHA']='6'*64
    decision['model_path']=str(m['model_paths']()['new']);decision.update(source_sha256='1'*64,protocol_sha256='2'*64,upstream_actual_audit_sha256=m['ACTUAL_AUDIT_SHA'])
    probabilities={b:dict(path=str(root/(b+'.npz')),sha256='a'*64) for b in ['old','c4','public']};source_rows={b:dict(path=str(root/(b+'.parquet')),sha256='b'*64,rows=n,sample_ix_sha256='f'*64) for b,n in [('old',173032),('c4',61440),('public',16713)]};prob={d['path']:d['sha256'] for d in probabilities.values()};data={**prob,**{d['path']:d['sha256'] for d in source_rows.values()}}
    original=dict(promoted=False,rejection_reasons=['original-guard-fail']);old_audit=dict(probability_bindings=prob,provenance_sha256={d['path']:d['sha256'] for d in source_rows.values()});decl=dict(protocol=dict(groups=groups,feasible=feasible),static=dict(sample_ix_sha256={b:'f'*64 for b in source_rows}))
    static=dict(upstream_actual_audit_sha256=m['ACTUAL_AUDIT_SHA'],upstream_complete_sha256=m['TRAINER_COMPLETE_SHA'],upstream_protocol_sha256=m['TRAINER_PROTOCOL_SHA'],original_records_sha256=m['ORIGINAL_RECORDS_SHA'],all14_plus_zero_untouched=True,model_sha256=m['NEW_MODEL_SHA'],model_path=decision['model_path'],original_selection=original,source_sha256='1'*64,probability_files_opened=False,row_tables_materialized=False,raw_features_needed=False,baseline_validation=index['new-max33']['validation'],controls={n:zero[n] if n in zero else index[n]['validation'] for n in m['GUARDS']},legacy=legacy,probabilities=probabilities,source_rows=source_rows)
    artifacts={'threshold-ledger.json':'7'*64,'selection.json':'4'*64,'validation-metrics.csv':'8'*64,'report.txt':'9'*64};done=dict(stage='completed_guard_constrained_development_calibration',source_sha256='1'*64,protocol_sha256='2'*64,eligible=True,training_updates=0,gpu_used=False,tests_read=False,primary_changed=False,new_independent_confirmation_required=True,artifact_sha256=artifacts,probability_bindings=prob,data_bindings=data)
    ledger=[dict(index=0,eligible=True,rejection_reasons=[],system_threshold=.6,effective_F32_threshold=float(np.float32(.6)),min33=.94),dict(index=1,eligible=False)]
    files={str(m['CALIBRATION_SOURCE']):'1'*64,str(m['CALIBRATION']/'protocol.json'):'2'*64,str(m['CALIBRATION']/'complete.json'):'3'*64,str(m['CALIBRATION_ACTUAL_AUDIT_PATH']):'6'*64,str(root/'auditor.py'):'5'*64}|{str(m['CALIBRATION']/name):s for name,s in artifacts.items()}
    certificate=dict(stage=m['CALIBRATION_AUDIT_STAGE'],audit_passed=True,test_data_or_results_read=False,gpu_used=False,auditor_source_path=str(root/'auditor.py'),auditor_source_sha256='5'*64,calibration_source_sha256='1'*64,calibration_protocol_sha256='2'*64,calibration_complete_sha256='3'*64,calibration_selection_sha256='4'*64,upstream_actual_audit_sha256=m['ACTUAL_AUDIT_SHA'],model_sha256=m['NEW_MODEL_SHA'],decision=decision,probability_bindings=prob,provenance_sha256={**{k:v for k,v in files.items() if k!=str(m['CALIBRATION_ACTUAL_AUDIT_PATH'])},**data})
    fake={str(m['CALIBRATION_ACTUAL_AUDIT_PATH']):certificate,str(m['CALIBRATION']/'complete.json'):done,str(m['CALIBRATION']/'protocol.json'):dict(protocol=protocol,static=static),str(m['CALIBRATION']/'selection.json'):decision,str(m['CALIBRATION']/'threshold-ledger.json'):ledger,str(m['TRAINER']/'selection.json'):original,str(m['TRAINER']/'zero-controls.json'):dict(records=zero),str(m['TRAINER']/'controls.json'):dict(original_savedGPU_validation=legacy)}
    raw=set(data)
    def fake_read(path):
        require_path=str(path);check_raw=require_path not in raw
        if not check_raw or require_path not in fake:raise AssertionError('Unexpected IO '+require_path)
        return copy.deepcopy(fake[require_path])
    def fake_digest(path):
        if str(path) not in files:raise AssertionError('Unexpected byte access '+str(path))
        return files[str(path)]
    m['read']=fake_read;m['digest']=fake_digest
    result=m['calibration_material']({},old_audit,decl,index=index)
    check('Two-layer fake certificate adapter accepts eligible-only decision, no raw IO',result['snapshot_certificate']['eligible'] and result['snapshot_certificate']['original_trial_promoted'] is False and result['snapshot_certificate']['system_threshold']==.6)
    check('New threshold owned by newdecision; old three controls owned by parent records',m['SOURCE_RECORD']['new-guard-max33']['origin']=='guard-calibration-decision' and all(m['SOURCE_RECORD'][n]['origin']=='original-readout-records' for n in m['RECORD_TO_MODEL'] if n!='new-guard-max33'))
    bad=copy.deepcopy(certificate);bad['upstream_actual_audit_sha256']='0'*64;fake[str(m['CALIBRATION_ACTUAL_AUDIT_PATH'])]=bad;reject('Even rehashed certificate cannot replace original7bac',lambda:m['calibration_material']({},old_audit,decl,index=index));fake[str(m['CALIBRATION_ACTUAL_AUDIT_PATH'])]=certificate
    bad=copy.deepcopy(decision);bad['system_threshold']=.5;fake[str(m['CALIBRATION']/'selection.json')]=bad;reject('Same model different threshold is not certified decision',lambda:m['calibration_material']({},old_audit,decl,index=index));fake[str(m['CALIBRATION']/'selection.json')]=decision
    bad=copy.deepcopy(source_rows);bad['old']['sha256']='c'*64;static['source_rows']=bad;reject('Parent row provenance cannot be switched',lambda:m['calibration_material']({},old_audit,decl,index=index));static['source_rows']=source_rows
    oldtree=ast.parse((HERE/'systemness_complementary_confirm_snapshot.py').read_text());f=lambda t,n:ast.dump(next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n),include_attributes=False)
    check('Frozen portable640 geometry unchanged from old helper AST',f(tree,'portable_metadata')==f(oldtree,'portable_metadata'))
    check('No production imports/dynamic global verifier or inventory scan',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and 'systemness_' in ast.unparse(n) for n in ast.walk(tree)) and '.rglob(' not in SOURCE.read_text())
    check('Source still exact held SHA',sha(SOURCE)==PIN)
    output=root/'report.json';report=dict(stage='synthetic_new_guard_snapshot_certificate_audit',audit_passed=True,source_sha256=PIN,checks=CHECKS,checks_passed=len(CHECKS),actual_metadata_ID_body_model_probability_bank_rows_or_tokenizer_read=False,actual_snapshot_declared=False,gpu_used=False,old_geometry_tests_repeated=False)
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report_path=str(output),report_sha256=sha(output),script_sha256=sha(__file__),**report)),flush=True)
if __name__=='__main__':main()
