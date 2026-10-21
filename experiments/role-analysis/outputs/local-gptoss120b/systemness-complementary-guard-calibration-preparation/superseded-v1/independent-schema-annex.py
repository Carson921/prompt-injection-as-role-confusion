"""Small independent source/fake-array audit; never reads actual p/model/rows."""
import ast,copy,hashlib,importlib.metadata,json,os,tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
SOURCE=HERE/'systemness_complementary_guard_calibration.py'
PIN='5c9a5c8dfe0dbf0c74c44cc45ef27f5e9518c0c8a52d08e2c84b20681ad21203'
CHECKS=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(name,value=True):
    if not value:raise AssertionError(name)
    CHECKS.append(name)
def reject(name,fn):
    try:fn()
    except RuntimeError:CHECKS.append(name)
    else:raise AssertionError('Fault accepted: '+name)
def load():
    tree=ast.parse(SOURCE.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef))]
    ns={'__file__':str(SOURCE),'__name__':'synthetic_guard_calibration'}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns
def main():
    check('Exact source pin before review',sha(SOURCE)==PIN);m=load();groups=m['GROUPS'];guards=m['GUARDS']
    check('11groups 9FPR 5strict 8guards fixed',len(groups)==11 and len(m['FEASIBLE'])==9 and len(m['STRICT'])==5 and len(guards)==8)
    p={b:np.array([[0,1,0,0,0],[.5,.1,.2,.1,.1],[1,0,0,0,0]],np.float32) for b in m['BANKS']}
    check('Exact original unique F64 grid recipe',np.array_equal(m['threshold_grid'](p),np.unique(np.r_[np.linspace(.001,.999,199),np.quantile(np.concatenate([p[b][:,0] for b in m['BANKS']]),np.linspace(0,1,301)),1.000001])))
    score=np.array([0,.1,.5,.5,np.nextafter(np.float32(.5),np.float32(1)),1,.1,.5,.9,1],np.float32);y=np.array([0,0,0,1,0,0,2,3,4,1]);grid=np.array([0,.1,.5,np.nextafter(.5,np.inf),float(np.nextafter(np.float32(.5),np.float32(1))),.9,1,1.000001]);curve=m['binary_curves'](score,y,grid)
    diff=0.
    for i,tau in enumerate(grid):
        d=score>=float(tau);s=y==0;tp=np.count_nonzero(d&s);fp=np.count_nonzero(d&~s);fn=np.count_nonzero(~d&s);tn=np.count_nonzero(~d&~s)
        expected=dict(tp=tp,fp=fp,fn=fn,tn=tn,system_precision=tp/(tp+fp) if tp+fp else np.nan,system_recall=tp/(tp+fn),system_balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,system_f1=2*tp/(2*tp+fp+fn),system_false_positive_rate=fp/(fp+tn),system_accuracy=(tp+tn)/len(y))
        for k,v in expected.items():
            check('Direct F32 consumer '+str(i)+'/'+k,np.array_equal(np.array(v),np.array(curve[k][i]),equal_nan=True))
            if np.isfinite(v):diff=max(diff,abs(float(v)-float(curve[k][i])))
    def result(v=.91):return {'groups':{g:{'system_f1':.98,'accuracy':.99,'system_false_positive_rate':.01} for g in groups},'min33':v,'old5_mean_f1':.98}
    base=result();controls={g:result() for g in guards};legacy=result()
    cs={g:{k:np.array(v,np.float64) for k,v in dict(system_precision=[.96,.97,.95],system_recall=[.96,.97,.95],system_balanced_accuracy=[.96,.97,.95],system_f1=[.98,.98,.98],system_false_positive_rate=[.01,.01,.01]).items()} for g in groups}
    cs['old-natural']['system_f1'][1]=.959
    best,proof,ledger=m['constrained_search'](cs,base,controls,legacy,np.array([.4,.5,.6]))
    check('Guard-first differs from old max-then-guard, without relaxing floors',best==0 and proof['original_best_index']==1 and not ledger[1]['eligible'])
    check('All failed tau ledger retained',len(ledger)==3 and any('old4_system_f1_loss' in r for r in ledger[1]['rejection_reasons']))
    no=copy.deepcopy(cs)
    for v in no.values():v['system_recall'][:]=.89
    best,proof,ledger=m['constrained_search'](no,base,controls,legacy,np.array([.4,.5,.6]));check('No feasible threshold gives None nofallback with all ledger',best is None and proof['no_fallback'] and len(ledger)==3)
    tied=copy.deepcopy(cs)
    for v in tied.values():
        for a in v.values():a[:]=a[0]
    best,_,_=m['constrained_search'](tied,base,controls,legacy,np.array([.5,np.nextafter(.5,np.inf),.6]));check('Effective F32 aliases tie by original smaller F64 tau',best==0)
    mean=copy.deepcopy(tied)
    for g in m['OLD5']:mean[g]['system_f1']=np.array([.98,.985,.98])
    best,_,_=m['constrained_search'](mean,base,controls,legacy,np.array([.4,.5,.6]));check('Equal min33 tie uses higher old5 meanF1',best==1)
    r=m['light_result'](cs,base,0)
    for name in m['STRICT']:
        blocked=copy.deepcopy(controls);blocked[name]['min33']=r['min33']-1e-13;reject_reason=m['constraints'](r,blocked,legacy)
        check('Each strict control retains 1e-12 '+name,not reject_reason['eligible'] and name+':min33_not_strictly_improved' in reject_reason['rejection_reasons'])
    high=copy.deepcopy(r);high['groups'][m['FEASIBLE'][0]]['system_false_positive_rate']=.05+5e-13
    check('Exact FPR feasibility stricter than guard tolerance','original_exact_FPR_feasibility' in m['constraints'](high,controls,legacy)['rejection_reasons'])
    high=copy.deepcopy(r);high['min33']=np.nextafter(.9,0);check('Absolute90 not weakened by tolerance','absolute_min33_below_90_percent' in m['constraints'](high,controls,legacy)['rejection_reasons'])
    simple=dict(accuracy=.2,recall={'system':.1},confusion=[[1]*5]*5,macro_f1=.3,nll=.4,system_auroc=.6,system_average_precision=.7)
    updated=m['update_binary'](simple,curve,2,.5)
    check('Fiveclass argmax/recall/confusion/NLL/AUC/AP preserved exactly',all(updated[k]==v for k,v in simple.items()) and simple.keys()==set(simple))
    # Bind actual *JSON schemas only*, while all artifact byte hashing is replaced by the published audit's digest map.
    # No p, portable NPZ, rowtable, bank or real model bytes are opened.
    folder=m['TRAINER'];names=m['ARTIFACTS']+['complete.json','status.json','candidates/new.json','candidates/matched.json','training/new.json','actual-audit/actual-development-audit.json']
    whitelist={str(folder/name):json.loads((folder/name).read_text()) for name in names if name.endswith('.json')}
    audit=whitelist[str(m['AUDIT'])];expected=dict(audit['provenance_sha256']);expected[str(m['AUDIT'])]=m['AUDIT_SHA'];expected[str(SOURCE)]=PIN;expected[str(m['DESIGN'])]=m['DESIGN_SHA'];expected[audit['auditor_source_path']]=m['AUDITOR_SHA']
    originals=(m['digest'],m['read'])
    def fake_digest(path):
        path=str(Path(path));check_key=expected.get(path)
        if check_key is None:raise AssertionError('Unclosed byte read '+path)
        return check_key
    def meta_read(path):
        path=str(Path(path))
        if path not in whitelist:raise AssertionError('Forbidden non-JSON read '+path)
        return copy.deepcopy(whitelist[path])
    m['digest']=fake_digest;m['read']=meta_read
    value=m['collect_static'](m['fixed_spec']())
    check('collect_static accepts real published JSON schema without p/model/row IO',value['model_sha256']==m['MODEL_SHA'] and len(value['controls'])==8 and value['original_selection']['promoted'] is False and set(value['source_rows'])==set(m['BANKS']))
    bad=copy.deepcopy(whitelist);bad[str(folder/'selection.json')]['model_sha256']='0'*64
    saved=whitelist;whitelist=bad;reject('Metadata fixture rejects model/selection mismatch',lambda:m['collect_static'](m['fixed_spec']()));whitelist=saved
    # All callable formal paths fail before digest/read on missing pins.
    def forbidden(*a,**k):raise AssertionError('Formal IO reached')
    m['digest']=forbidden;m['read']=forbidden;m['SOURCE_SHA']=None
    reject('Unknownpins declare before formal IO',m['declare']);reject('Unknownpins actual before formal IO',m['run_actual'])
    m['digest'],m['read']=originals
    tree=ast.parse(SOURCE.read_text());check('No production metrics/verifier imports',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and ('systemness_' in ast.unparse(n)) for n in ast.walk(tree)))
    check('Original readout failed artifact not written',all("TRAINER/" not in ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['immutable_json','atomic_json']))
    check('Source still exact final pin',sha(SOURCE)==PIN)
    out=Path(tempfile.mkdtemp(prefix='confirmation-review-guard-calibration-source-',dir='/tmp'))/'report.json'
    value=dict(stage='source_synthetic_guard_calibration_audit',audit_passed=True,source_sha256=PIN,checks=CHECKS,checks_passed=len(CHECKS),maximum_direct_F32_difference=diff,actual_metadata_schema_read=True,actual_probability_model_bytes_rows_bank_or_test_read=False,actual_calibration_run=False,gpu_used=False,notes='Small published JSON whitelist used only for schema fixture; all byte digest operations replaced by audit digest-map. No actual collect_static filesystem traversal.')
    out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(report_path=str(out),report_sha256=sha(out),script_sha256=sha(__file__),**value)),flush=True)
if __name__=='__main__':main()
