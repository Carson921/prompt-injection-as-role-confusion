"""Guard-first DEV-only fixed16 tomato gallery; default help, explicit --render.

Only extra2-new's binary threshold changes. Same saved p/argmax, sources and
five columns; no five-class improvement, LM, fitting or independent test claim.
"""
import argparse,copy,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import systemness_complementary_head_development_plots as parent
HERE=Path(__file__).resolve().parent;BASE=HERE/'outputs/local-gptoss120b'
OUT=BASE/'systemness-complementary-guard-calibration-development-plots'
CALIB=BASE/'systemness-complementary-guard-calibration'
PARENT_SHA='3cbe2c6fefc3e62246487449fee130bdb3055e9143167420b8f0ee4804ce0785'
CALIB_SOURCE_SHA='70c6ae046cbbb5485047733b593afe23a1d61376f047b2ad29e75c692e7bb85d'
CALIB_PROTOCOL_SHA='7edc1b5ba2c55a21ebe01bc5af9f850c4865a933bb9bc4b1274520d6b00ec64a'
CALIB_COMPLETE_SHA='9d5b032c1b055eda8ccb5620620f873b2e4775cb533729b787d86436857c2e94'
CALIB_SELECTION_SHA='e220bda41077f5931dc81b951f10e55e1160904e303f3299a89dfdc300430a0c'
CALIB_AUDITOR_SHA='3296c36da7efffe9ab45a75c12d8ebf1640b18f126cd64166674ae19002bc323';CALIB_AUDIT_SHA='1a6e545db44a2d76c8da30886409d7c3b8376e670281a130b464a2e8821f4667'
MODELS=list(parent.MODELS);PATHS=dict(parent.PATHS)
require,read,digest,probability=parent.require,parent.read,parent.digest,parent.probability
parent_gate,draw=parent.gate,parent.draw


def pin_gate():
    require(all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in [CALIB_AUDITOR_SHA,CALIB_AUDIT_SHA]),'Root must pin approved calibration auditor and actual audit before any formal IO.')


def gate(parent_audit_path,calibration_audit_path):
    pin_gate();require(digest(parent.__file__)==PARENT_SHA,'Frozen parent gallery changed.')
    upstream,plan=parent_gate(parent_audit_path)
    require(digest(calibration_audit_path)==CALIB_AUDIT_SHA,'Calibration independent actual audit changed.');audit=read(calibration_audit_path)
    require(audit['stage']=='actual_completed_guard_constrained_calibration_audit' and audit['audit_passed'] is True and audit['auditor_source_sha256']==CALIB_AUDITOR_SHA and audit['test_data_or_results_read'] is False and audit['gpu_used'] is False and audit['training_updates']==0,'Approved completed DEV-only calibration audit required.')
    for path,key,want in [(HERE/'systemness_complementary_guard_calibration.py','calibration_source_sha256',CALIB_SOURCE_SHA),(CALIB/'protocol.json','calibration_protocol_sha256',CALIB_PROTOCOL_SHA),(CALIB/'complete.json','calibration_complete_sha256',CALIB_COMPLETE_SHA),(CALIB/'selection.json','calibration_selection_sha256',CALIB_SELECTION_SHA)]:require(digest(path)==audit[key]==want,'Frozen calibration dependency changed: '+str(path))
    decision=read(CALIB/'selection.json');done=read(CALIB/'complete.json')
    require(audit['decision']==decision and decision['candidate']=='new-guard-max33' and decision['eligible'] is True and decision['no_fallback'] is True and decision['original_trial_unchanged'] is True and decision['primary_changed'] is False and decision['tests_read'] is False,'Eligible immutable guard-first decision required.')
    require(audit['upstream_actual_audit_sha256']==decision['upstream_actual_audit_sha256']==parent.APPROVED_ACTUAL_AUDIT_SHA and audit['model_sha256']==decision['model_sha256']==upstream['model_sha256']['new'] and audit['model_path']==decision['model_path'],'Calibration changed the audited model or parent proof.')
    proof=decision['gate'];require(proof['eligible'] is True and not proof['rejection_reasons'] and set(proof['guards'])==set(upstream['selection']['guards']) and all(g['eligible'] is True for g in proof['guards'].values()) and set(proof['min33_gains'])==set(upstream['selection']['min33_gains']) and all(v>1e-12 for v in proof['min33_gains'].values()) and decision['validation']['min33']>=.9,'Original nine guards/five strict/absolute90 not satisfied.')
    folder,prefix=PATHS['extra2-new'];expected={str(folder/f'{prefix}-{b}-validation.npz') for b in ['old','c4','public']}
    bindings=audit['probability_bindings'];require(set(bindings)==expected and bindings==done['probability_bindings'] and all(upstream['probability_bindings'].get(p)==v for p,v in bindings.items()) and done['eligible'] is True and done['tests_read'] is False and done['gpu_used'] is False and done['training_updates']==0,'Calibration probabilities or execution scope changed.')
    require(np.isfinite(decision['system_threshold']) and 0<=decision['system_threshold']<=1.000001,'Invalid calibrated global threshold.');return upstream,plan,audit,decision


def thresholds_for(original,decision):
    require(set(original)==set(MODELS),'Original five columns changed.');out=dict(original);out['extra2-new']=decision['system_threshold']
    require(all(out[m]==original[m] for m in MODELS if m!='extra2-new'),'A control threshold changed.');return out


def load(plan,audit):
    rows={}
    for bank,path,count in [('old',Path('/dev/shm/role-analysis-top8-heads/rows.parquet'),173032),('public',Path('/dev/shm/role-analysis-natural-dev-heads/rows.parquet'),16713)]:
        require(digest(path)==plan['rows_sha256'][bank]==audit['provenance_sha256'][str(path)],'Audited fixed row map changed.');frame=pd.read_parquet(path);rows[bank]=frame[frame.split.eq('validation')].reset_index(drop=True);require(len(rows[bank])==count,'Fixed DEV cohort changed.')
    records={r['record']:r for r in read(parent.TRAIN/'records.json')};thresholds={m:records[r]['validation']['system_threshold'] for m,r in zip(MODELS,parent.RECORDS)}
    values={m:{b:probability(audit,folder/f'{prefix}-{b}-validation.npz',rows[b]) for b in rows} for m,(folder,prefix) in PATHS.items()};return rows,values,thresholds


def render(parent_audit_path,calibration_audit_path):
    upstream,plan,audit,decision=gate(parent_audit_path,calibration_audit_path);require(not (OUT/'complete.json').exists(),'Completed guard gallery cannot be overwritten.')
    rows,values,original=load(plan,upstream);thresholds=thresholds_for(original,decision)
    table,files=draw(plan,rows,values,thresholds,OUT)
    index=OUT/'gallery-index.md';index.write_text('GUARD-FIRST DEVELOPMENT ONLY: same fixed16 sources and saved probabilities.\nOnly extra2-new binary tau is calibrated; five-class predictions and accuracy are unchanged.\nNo independent confirmation or primary change.\n\n'+index.read_text())
    gate(parent_audit_path,calibration_audit_path)
    done=dict(stage='completed_fixed_guard_calibrated_development_gallery',guard_first_DEVELOPMENT_ONLY=True,examples=16,readouts=MODELS,png=16,pdf=16,metrics=80,source_ranks_changed=False,independent_test=False,LM_run=False,probabilities_recomputed=False,argmax_predictions_changed=False,new_fiveclass_improvement=False,primary_changed=False,
        original_thresholds=original,thresholds=thresholds,only_changed_binary_threshold='extra2-new',model_sha256=decision['model_sha256'],parent_audit_sha256=parent.APPROVED_ACTUAL_AUDIT_SHA,calibration_audit_sha256=CALIB_AUDIT_SHA,calibration_auditor_source_sha256=CALIB_AUDITOR_SHA,calibration_complete_sha256=CALIB_COMPLETE_SHA,calibration_selection_sha256=CALIB_SELECTION_SHA,plot_source_sha256=digest(__file__),parent_gallery_source_sha256=PARENT_SHA,gallery_plan_sha256=parent.PLAN_SHA,
        inputs={str(folder/f'{prefix}-{b}-validation.npz'):digest(folder/f'{prefix}-{b}-validation.npz') for folder,prefix in PATHS.values() for b in rows},artifacts={p.name:digest(p) for p in files})
    (OUT/'complete.json').write_text(json.dumps(done,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(out=str(OUT),png=16,pdf=16,metrics=80,complete_sha256=digest(OUT/'complete.json'))))


def preflight():
    """A few mock boundaries; no actual metadata, p, rows, rendering or GPU."""
    import ast,sys
    from unittest.mock import patch
    folder=Path(tempfile.mkdtemp(prefix='complementary-guard-gallery-preflight-',dir='/tmp'));module=sys.modules[__name__];source=digest(__file__);checks=[];old=(parent.OUT,parent.TRAIN,list(parent.MODELS))
    def check(name,value=True):require(value,name);checks.append(name)
    def refuse(name,fn):
        try:fn()
        except RuntimeError:check(name);return
        raise RuntimeError('Expected mock refusal: '+name)
    def forbidden(*a,**k):raise LookupError('Formal IO forbidden in mock.')
    with patch.object(module,'CALIB_AUDITOR_SHA',None),patch.object(module,'CALIB_AUDIT_SHA',None),patch.object(module,'digest',forbidden),patch.object(module,'read',forbidden),patch.object(module,'parent_gate',forbidden):refuse('unknown-pins-before-parent-or-any-IO',lambda:gate(folder/'parent',folder/'calib'))
    strict={str(i):.02 for i in range(5)};guards={str(i):dict(eligible=True) for i in range(9)};pf,px=PATHS['extra2-new'];ps={str(pf/f'{px}-{b}-validation.npz'):str(i)*64 for i,b in enumerate(['old','c4','public'])}
    decision=dict(candidate='new-guard-max33',eligible=True,no_fallback=True,original_trial_unchanged=True,primary_changed=False,tests_read=False,upstream_actual_audit_sha256=parent.APPROVED_ACTUAL_AUDIT_SHA,model_sha256='e'*64,model_path=str(folder/'fixed-model.npz'),system_threshold=.52,validation=dict(min33=.92),gate=dict(eligible=True,rejection_reasons=[],guards=guards,min33_gains=strict))
    upstream=dict(model_sha256={'new':'e'*64},selection=dict(guards=guards,min33_gains=strict),probability_bindings=ps)
    audit=dict(stage='actual_completed_guard_constrained_calibration_audit',audit_passed=True,auditor_source_sha256='a'*64,test_data_or_results_read=False,gpu_used=False,training_updates=0,calibration_source_sha256=CALIB_SOURCE_SHA,calibration_protocol_sha256=CALIB_PROTOCOL_SHA,calibration_complete_sha256=CALIB_COMPLETE_SHA,calibration_selection_sha256=CALIB_SELECTION_SHA,decision=decision,upstream_actual_audit_sha256=parent.APPROVED_ACTUAL_AUDIT_SHA,model_sha256='e'*64,model_path=decision['model_path'],probability_bindings=ps)
    done=dict(probability_bindings=ps,eligible=True,tests_read=False,gpu_used=False,training_updates=0)
    objects={str(folder/'calib'):audit,str(CALIB/'selection.json'):decision,str(CALIB/'complete.json'):done};hashes={str(parent.__file__):PARENT_SHA,str(folder/'calib'):'b'*64,str(HERE/'systemness_complementary_guard_calibration.py'):CALIB_SOURCE_SHA,str(CALIB/'protocol.json'):CALIB_PROTOCOL_SHA,str(CALIB/'complete.json'):CALIB_COMPLETE_SHA,str(CALIB/'selection.json'):CALIB_SELECTION_SHA}
    with patch.object(module,'CALIB_AUDITOR_SHA','a'*64),patch.object(module,'CALIB_AUDIT_SHA','b'*64),patch.object(module,'digest',lambda p:hashes[str(p)]),patch.object(module,'read',lambda p:copy.deepcopy(objects[str(p)])),patch.object(module,'parent_gate',lambda p:(upstream,{'entries':[]})):
        check('closed-mock-certificate-and-same-model-p',gate(folder/'parent',folder/'calib')[3]==decision)
        audit['probability_bindings']=dict(ps,wrong='0'*64);refuse('changed-probability-identity-rejected',lambda:gate(folder/'parent',folder/'calib'));audit['probability_bindings']=ps
        decision['eligible']=False;refuse('ineligible-decision-rejected',lambda:gate(folder/'parent',folder/'calib'));decision['eligible']=True
    original={name:.4+i*.02 for i,name in enumerate(MODELS)};before=dict(original);got=thresholds_for(original,decision);check('only-extra2-new-tau-overridden',original==before and got['extra2-new']==.52 and all(got[m]==original[m] for m in MODELS if m!='extra2-new'))
    tree=ast.parse(Path(__file__).read_text());call=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='draw'][0];check('pure-parent-draw-explicit-arguments',len(call.args)==5 and [ast.unparse(a) for a in call.args]==['plan','rows','values','thresholds','OUT'])
    check('parent-globals-and-source-unchanged',old==(parent.OUT,parent.TRAIN,parent.MODELS) and digest(__file__)==source)
    report=dict(source_sha256=source,checks=checks,checks_passed=len(checks),actual_metadata_or_probability_or_rows_read=False,mock_16_gallery_rendered=False,gpu_used=False,formal_outputs_written=False);path=folder/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(path),report_sha256=digest(path),**report)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--preflight',action='store_true');group.add_argument('--render',action='store_true');parser.add_argument('--parent-audit',type=Path);parser.add_argument('--calibration-audit',type=Path);args=parser.parse_args()
    if args.preflight:preflight()
    elif args.render:require(args.parent_audit is not None and args.calibration_audit is not None,'Provide parent and completed independent calibration audits.');render(args.parent_audit,args.calibration_audit)
    else:parser.print_help()
