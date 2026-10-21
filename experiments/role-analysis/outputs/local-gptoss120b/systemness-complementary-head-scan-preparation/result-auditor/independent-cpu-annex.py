#!/usr/bin/env python3
"""Tiny/mock read-only review of the independent scanner auditor; no actual entrypoint work or fit arrays."""
import ast
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

PATH=Path('/tmp/confirmation_review_complementary_head_results_audit.py')
EXPECTED='3ec5dea460ff2c8facc3bd6cda9ad1fdb7f62ffd3ecdfc69b0528511565129fe'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
START=sha(PATH);assert EXPECTED is not None and START==EXPECTED,'Final auditor source not pinned.'
spec=importlib.util.spec_from_file_location('complementary_auditor_independent_annex',PATH)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
CHECKS=[]
def check(n,v=True):assert bool(v),n;CHECKS.append(n)
def reject(n,fn):
    try:fn()
    except(RuntimeError,AssertionError,KeyError,ValueError,FileNotFoundError):CHECKS.append(n)
    else:raise AssertionError('Fault accepted: '+n)
def forbidden(*a,**k):raise AssertionError('Actual metadata/model/probability API forbidden in tiny annex.')
folder=Path(tempfile.mkdtemp(prefix='feature-review-complementary-result-auditor-',dir='/tmp'))
with patch.object(np,'load',side_effect=forbidden),patch.object(np,'memmap',side_effect=forbidden),patch.object(pd,'read_parquet',side_effect=forbidden):
    check('Exact frozen scanner source70db and plan0db7 pins',s.SOURCE_SHA=='70db2015b90a496dc465c63631911da81d641614fcf0b1d48b04313a078ce156' and s.PLAN_SHA=='0db7b1a351196b5894389fe405f0ca1c93a37d218873b9a0ad119dc603575f70')
    with patch.object(s,'processes',return_value=[123]),patch.object(s,'read',side_effect=forbidden),patch.object(s,'digest',side_effect=forbidden),patch.object(Path,'is_file',side_effect=forbidden):
        reject('Live scanner blocks before evenstatus/file/hash access',s.terminal_gate)
    (folder/'scan-complete.json').write_text('{}');(folder/'status.json').write_text(json.dumps({'stage':'statistics'}))
    with patch.object(s,'OUT',folder),patch.object(s,'processes',return_value=[]):
        reject('Nonterminal synthetic scanner rejects complete marker',s.terminal_gate)
    calls=[]
    def stop_gate():calls.append('gate');raise RuntimeError('synthetic stop before actual work')
    with patch.object(s,'terminal_gate',side_effect=stop_gate),patch.object(s,'digest',side_effect=forbidden),patch.object(s,'read',side_effect=forbidden):
        reject('Actual firstinstruction ownterminalgate before allmetadata andprobability opens',lambda:s.actual(folder/'not-written.json'))
    check('No actual auditwork orresult file produced by failfast test',calls==['gate'] and not (folder/'not-written.json').exists())

    # Preserve actual algorithm interfaces while forbidding conversion of a whole matrix.
    n=8;truth=np.array([True]*4+[False]*4);a=np.full((n,40),.2,np.float32)
    a[:,8]=np.where(truth,.9,.1);a[:,9]=np.where(truth,.8,.15)
    class ColumnOnly:
        def __init__(self,a):self.a=a;self.shape=a.shape;self.dtype=a.dtype;self.columns=[]
        def __array__(self,*a,**k):raise AssertionError('Whole probability matrix conversion is forbidden.')
        def __getitem__(self,key):
            assert isinstance(key,tuple) and len(key)==2 and isinstance(key[0],slice) and key[0]==slice(None) and isinstance(key[1],(int,np.integer))
            self.columns.append(int(key[1]));return self.a[key]
    p=ColumnOnly(a);groups=[(d,'own-unit',np.arange(n),truth) for d in s.DOMAINS]
    result,marginal,pairs,units=s.choose(p,groups,original=list(range(8)),screen=32)
    check('Independent choose consumes columnvectors only, never fullmatF64',len(p.columns)>0 and result['selected_head_ids']==list(range(10)))
    check('All remaining32 toyheads and32choose2=496 jointly audited',len(marginal)==32 and len(pairs)==496 and result['pairs_evaluated']==496)
    expected=np.zeros(n,np.float64)
    for head in [2,8,0]:expected+=a[:,head].astype(np.float64)
    check('Original ranked order F64sum then oneF32 cast retained',np.array_equal(s.total_score(p,[2,8,0]).view(np.uint32),(expected/3).astype(np.float32).view(np.uint32)))
    reject('Nonfinite scalar-score unit rejects rather than fabricate metric',lambda:s.binary_metrics(np.array([np.nan,.5]),np.array([True,False])))

    # Only projected, empty metadata stand-ins are supplied; they must fail scope.
    requests=[]
    def empty_meta(path,**kwargs):requests.append((str(path),kwargs));return pd.DataFrame(columns=kwargs['columns'])
    with patch.object(pd,'read_parquet',side_effect=empty_meta):
        reject('Bad sourcecounts fail before p/model/source materialization',lambda:s.reconstruct({}))
    check('Reconstruct projects onlyfit descriptors, no prompts/tokenJSON',len(requests)==4 and all(('split','=','fit') in k['filters'] and 'prompt' not in k['columns'] for _,k in requests) and ('eligible_fit','=',True) in requests[2][1]['filters'])

    # Exercise NPZ schema rejection with tiny mocked arrays, not real moments.
    class TinyNPZ:
        files=['n','sum','xx','metadata']
        data={'n':np.zeros((2,5),np.int64),'sum':np.zeros((2,5,3),np.float64),'xx':np.zeros((2,3,3),np.float64),'metadata':np.array('{}')}
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def __getitem__(self,k):return self.data[k]
    with patch.object(np,'load',return_value=TinyNPZ()):
        reject('Wrong moment geometry refuses reuse/refit',lambda:s.moments('/tmp/not-opened.npz'))

    text=PATH.read_text();tree=ast.parse(text);actual=inspect.getsource(s.actual);rebuild=inspect.getsource(s.reconstruct)
    def fn(name):return next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    check('Actual wrapper starts andends with its ownterminal gate',ast.unparse(fn('actual').body[0]).startswith('terminal_gate()') and sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='terminal_gate' for n in ast.walk(fn('actual')))==2)
    check('Exact55859sample order andboth descriptors comparedwithout tolerance',all(x in rebuild for x in ["len(views['holdout']['rows'])==55859",'==HOLD_SHA',"plan['data'][phase]",'tol=0']))
    check('Full2296/32/496 counts andCSV scopes hardchecked',all(x in actual for x in ['len(marginal)==2296','len(pairs)==496','len(csv)==2296','len(csv)==496']) and s.DOMAINS==['C4-all','C4-tail','public-oasst','public-toxicchat0124'])
    check('Matrix mmap/F32/shape/whole-fileSHA andchunkfinite/range allverified',all(x in actual for x in ["mmap_mode='r'",'p.shape==(55859,2304)','p.dtype==np.float32',"bind(pp,discovery['probabilities_sha256'])",'np.isfinite(block).all()','((block>=0)&(block<=1)).all()']))
    check('Moment complete/checkpoint/old101additive andLDA refit bind allarrays',all(x in actual for x in ["stat['moments_sha256']","meta['additional_checkpoint_sha256']",'extra[k]+previous[k]','np.broadcast_to(counts,(2304,5))','refit_lda(mixed)',"mm==dict(plan_sha256=PLAN_SHA"]))
    check('Native backend/devicemap/hooks/parity evidence explicitlybound',all(x in actual for x in ["backend['native_mxfp4'] is True","backend['parameters_frozen'] is True","backend['device_map']==expected_map","h['hidden_states_bitexact'] is True","h['statistics_disabled'] is True","h['probability_writes_disabled'] is True","discovery['cpu_probability_max_difference']<1e-8"]))
    check('Selected10 exactweight/bias/head subset tiedto mixedmodel',all(x in actual for x in ["saved['proxy_model_sha256']==digest(proxy)","np.array_equal(z['selected_head_ids'],selected)","np.array_equal(z['weight'],w[selected])","np.array_equal(z['bias'],b[selected])"]))
    check('No production imports or production selection-helper calls',not any(isinstance(n,ast.Import) and any(a.name.startswith('systemness_') for a in n.names) or isinstance(n,ast.ImportFrom) and (n.module or '').startswith('systemness_') for n in ast.walk(tree)) and 'choose_pair' not in text)
    check('Rawactivation nonreplay limitation andpostselection hash-only scope explicit',"native_LM_independently_replayed=False" in actual and 'activations were not persisted' in actual and 'hash-bound only' in actual)
    check('Final reviewed source unchanged',sha(PATH)==START)

report=dict(stage='independent_complementary_scanner_result_auditor_annex',audit_passed=True,
    reviewed_source_path=str(PATH),reviewed_source_sha256=START,reviewed_source_sha256_after=sha(PATH),
    checks_passed=len(CHECKS),checks=CHECKS,annex_source_path=str(Path(__file__).resolve()),annex_source_sha256=sha(__file__),
    actual_metadata_views_or_fit_probabilities_or_models_read=False,actual_audit_work_executed=False,
    validation_or_tests_read=False,LM_loaded=False,gpu_used=False,formal_outputs_modified=False)
dest=folder/'report.json';dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(report=str(dest),report_sha256=sha(dest),**report)))
