#!/usr/bin/env python3
"""Read-only adapter review + tiny/mock CPU. Never execute actual or open formal arrays/results."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd

SOURCE=Path('/tmp/long_data_long_mix_residual_results_audit.py')
EXPECTED='7e4fc4d94ec175905fee9d999dbaf94c3fa86df9df6e9633e258cab44cc9fbff'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
START=sha(SOURCE);assert START==EXPECTED
spec=importlib.util.spec_from_file_location('long_mix_auditor_for_independent_toy',SOURCE)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
CHECKS=[]
def check(name,ok=True):assert bool(ok),name;CHECKS.append(name)
def reject(name,fn):
    try:fn()
    except(RuntimeError,AssertionError,ValueError,KeyError,FileNotFoundError):CHECKS.append(name)
    else:raise AssertionError('Fault accepted: '+name)
def forbidden(*a,**kw):raise AssertionError('Formal file/array API forbidden in independent synthetic annex.')
folder=Path(tempfile.mkdtemp(prefix='feature-review-long-mix-result-auditor-',dir='/tmp'))
math=s.pure_math() # Only hashes/imports the published frozen independent numerical SOURCE, never its actual entrypoint.
old=(math.RECORDS.copy(),math.GUARD_CONTROLS.copy(),math.STRICT_CONTROLS.copy(),math.MAX_DIFFERENCE)
with patch.object(np,'load',side_effect=forbidden),patch.object(np,'memmap',side_effect=forbidden),patch.object(pd,'read_parquet',side_effect=forbidden):
    check('Pinned230/e636 and immutable actualslot/manifest source constants',s.SOURCE_SHA=='230bc5e828a53e4b543b09a6c97693c0eec861ff01e6685424ec8d3f397f364d' and s.PROTOCOL_SHA=='e636d28cd92449daa158cad8ebc8ba9c08060b50b5f1daac88199c6b34e9649f' and s.MANIFEST_SHA=='a7311c5f7ce1709c0c0947569fce4000c754e3e69a0490c8287490b44b93532b' and s.SLOTS_SHA=='5c28655bec5eb6ba2dd90bd1422fb9cac920bb73f7fe2b7c729f5735c0915169')
    # A live-process failure must happen before even metadata read/hash, let alone p/model/slots.
    with patch.object(s,'read',side_effect=forbidden),patch.object(s,'digest',side_effect=forbidden):
        reject('Liveworker rejects before any realhash/status/array access',lambda:s.terminal_gate(out=folder/'absent',active=[123],source_sha='toy',protocol_sha='toy'))
    reject('Missing source/protocol hardpins failclosed',lambda:s.require_pins(None,None))
    (folder/'status.json').write_text(json.dumps({'stage':'training'}));(folder/'complete.json').write_text('{}')
    reject('Nonterminal synthetic status refuses complete marker',lambda:s.terminal_gate(out=folder,active=[],source_sha='toy',protocol_sha='toy'))
    # Execute only the initial fail-fast branch of actual: gate immediately raises,
    # every subsequent IO/numerical loader is forbidden. No actual work is performed.
    reached=[]
    def stop_gate():reached.append('gate');raise RuntimeError('synthetic stop before actual')
    with patch.object(s,'terminal_gate',side_effect=stop_gate),patch.object(s,'pure_math',side_effect=forbidden),patch.object(s,'read',side_effect=forbidden),patch.object(s,'digest',side_effect=forbidden):
        reject('Actual firstinstruction terminalgate precedes numerical source andallIO',lambda:s.actual(folder/'never-written.json'))
    check('Failfast branch leaves no auditresult file',reached==['gate'] and not (folder/'never-written.json').exists())

    fit,fresh=s.synthetic_slots();slots,summary=s.replacement(fit,fresh,old_documents=2,new_documents=4,new_per_half=32)
    check('Independentreplacement maps fitposition not localcache_ix withcolliding prompt IDs',len(slots)==1280 and np.array_equal(fit.iloc[slots.fit_position].cache_ix,slots.old_cache_ix) and slots.old_body_offset.min()>0 and slots.fit_position.min()>0)
    check('Independent rolehalf sampler keeps bijective labels andconstantweight',summary['role_half_buckets']=={r+':'+str(h):128 for r in s.ROLES for h in [0,1]} and slots.new_cache_ix.is_unique and np.array_equal(slots.role,slots.new_role))

    groups=math.GROUPS
    def value(n=.92):return dict(min33=n,old5_mean_f1=.92,groups={g:dict(system_f1=.92,accuracy=.92,system_false_positive_rate=.01) for g in groups})
    records=[dict(record=n,validation=value(.94 if n=='new-max33' else .92)) for n in s.RECORDS]
    zero=[dict(record=n,validation=value()) for n in s.ZERO_RECORDS];legacy=value()
    answer=s.select_records(math,records,zero,legacy)
    check('12plus2 separate records and5strict8guard plusoriginalGPU',answer['promoted'] and len(answer['guards'])==9 and len(s.STRICT)==5 and len(s.GUARDS)==8)
    for name in s.STRICT:
        rr=copy.deepcopy(records);zz=copy.deepcopy(zero)
        next(r['validation'] for r in rr+zz if r['record']==name)['min33']=.94
        a=s.select_records(math,rr,zz,legacy)
        check('Independent exactstrict tie refuses '+name,not a['promoted'] and name+':min33_not_strictly_improved' in a['rejection_reasons'])
    rr=copy.deepcopy(records);next(r['validation'] for r in rr if r['record']=='priorBinary-max33')['groups']['old-long-full-tail']['system_f1']=.99
    check('New priorBinary fulltail F1 floor is compulsory',not s.select_records(math,rr,zero,legacy)['promoted'])
    gpu=value();gpu['groups']['old-natural']['system_f1']=.99
    check('Mandatory originalsavedGPU hardfloor is separate',not s.select_records(math,records,zero,gpu)['promoted'])

    # Count/identity/NA verification is local; no producer/statistical helper is imported.
    metric=dict(tokens=10,accuracy=.9,system_f1=.91,recall={'system':1.,'user':1.,'cot':None,'assistant':1.,'tool':None})
    metricset=dict(groups={g:copy.deepcopy(metric) for g in groups},pooled_descriptive_only=copy.deepcopy(metric))
    rs=[dict(record=n,validation=copy.deepcopy(metricset)) for n in s.RECORDS]
    zr=[dict(record=n,validation=copy.deepcopy(metricset)) for n in s.ZERO_RECORDS]
    def flat(items):
        return pd.DataFrame([dict(record=r['record'],group=g,**{k:v for k,v in m.items() if not isinstance(v,(dict,list))},**{'role_recall_'+role:v for role,v in m['recall'].items()}) for r in items for g,m in dict(r['validation']['groups'],pooled_descriptive_only=r['validation']['pooled_descriptive_only']).items()])
    main,ztable=flat(rs),flat(zr);s.verify_csv(main,rs,s.RECORDS);s.verify_csv(ztable,zr,s.ZERO_RECORDS)
    check('144main+24zero exact168 identities andunsupported-role NA',len(main)==144 and len(ztable)==24)
    reject('Missing CSV row rejects instead of hiding failedrecord',lambda:s.verify_csv(main.iloc[:-1],rs,s.RECORDS))
    bad=main.copy();bad.loc[0,'role_recall_cot']=0
    reject('UndefinedCoT recall cannotbe silentlyconverted tozero',lambda:s.verify_csv(bad,rs,s.RECORDS))

    p={b:np.tile(np.array([[.2,.1,.3,.25,.15]],np.float32),(3,1)) for b in ['old','c4','public']}
    state={'toy':np.arange(6,dtype=np.float32)};budget=dict(best_epoch=3,best_steps=471,epochs=11,actual_steps=1727,actual_instances=7056423)
    rp=s.replication_evidence(math,state,state,dict(matched=p,priorBinary=p),budget,budget)
    check('Matchedprior entire3banks p andallweight bitexact evidence',rp['all_weight_arrays_bitexact'] and rp['all_probability_arrays_bitexact'] and rp['numerical_drift_is_not_source_effect'])
    bad={b:a.copy() for b,a in p.items()};bad['public'][0,0]+=.0006;bad['public'][0,1]-=.0006
    reject('One-cohort matchedprior drift over5e-4 rejects',lambda:s.replication_evidence(math,state,state,dict(matched=bad,priorBinary=p),budget,budget))

    # Static binding/API pass covers the full adapter with actual file opens disabled above.
    source=SOURCE.read_text();tree=ast.parse(source)
    actual=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='actual')
    check('Actual entrypoint starts with ownterminalgate andends with recheck',ast.unparse(actual.body[0])=='terminal_gate()' and sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='terminal_gate' for n in ast.walk(actual))==2)
    forbidden_calls={'actual','preflight','close','compare_raw_evidence','verified_layout','terminal_gate','verify_csv','select_records'}
    check('Imported87c only pure functions no legacyactual/globalverifier/close',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='math' and n.func.attr in forbidden_calls for n in ast.walk(tree)))
    check('No mutation of imported auditor globals orproduction imports',not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='math' for t in n.targets) for n in ast.walk(tree)) and not any(isinstance(n,ast.Import) and any(a.name.startswith('systemness_') for a in n.names) or isinstance(n,ast.ImportFrom) and (n.module or '').startswith('systemness_') for n in ast.walk(tree)))
    check('Realadapter explicit extractiondirectories rawfeatures androwsSHA bound',all(t in inspect.getsource(s.verified_banks) for t in ["BASE/'systemness-long-augmentation'","BASE/'systemness-head-features'","BASE/'systemness-natural-dev-features'","bind(feature,complete['features_sha256'])","bind(cache/'rows.parquet',complete['rows_sha256'])"]))
    actualtext=inspect.getsource(s.actual)
    check('Three fullfunctional replays plusportable/model/savedp SHA coverage',all(t in actualtext for t in ["for arm in ['new','matched']:","computed,oldp,inv=math.residual_replay", "replay('zero'", "model_hashes[name]=bind(path,sha)", "math.binary_model_arrays", "probability_bindings"]))
    check('Prior2600 certificate closes inheritedensemble andNPZ bytechain',all(t in actualtext for t in ["bind(PRIOR_AUDIT,PRIOR_AUDIT_SHA)","for path,want in prior['provenance_sha256'].items():bind(path,want)","for path,want in prior['probability_bindings'].items()", "prior['fixed_four_member_ensemble'] is True", "prior['original_savedGPU_NPZ_and_threshold_unchanged'] is True"]))
    check('Math globals unchanged after everypuretoy call',old==(math.RECORDS,math.GUARD_CONTROLS,math.STRICT_CONTROLS,math.MAX_DIFFERENCE))
    check('No production modules imported',not any(n.startswith('systemness_') for n in sys.modules))
    check('Finalreviewed auditor source unchanged',sha(SOURCE)==START)

report=dict(stage='independent_long_mix_result_auditor_source_and_mock_annex',audit_passed=True,
    reviewed_source_path=str(SOURCE),reviewed_source_sha256=START,reviewed_source_sha256_after=sha(SOURCE),
    checks_passed=len(CHECKS),checks=CHECKS,source_sha256=sha(__file__),
    trainer_source_sha256=s.SOURCE_SHA,trainer_protocol_sha256=s.PROTOCOL_SHA,
    actual_results_or_slots_or_weights_or_probabilities_read=False,actual_entrypoint_work_executed=False,
    test_data_read=False,gpu_used=False,formal_outputs_modified=False)
dest=folder/'report.json';dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(report=str(dest),report_sha256=sha(dest),annex_source_sha256=sha(__file__),**report)))
