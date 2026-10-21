#!/usr/bin/env python3
"""Implementation-only synthetic scanner audit; no actual metadata/body/head/p/LM/declare/scan."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from transformers import AutoTokenizer

ROOT=Path('/root/prompt-injection-as-role-confusion');sys.path.insert(0,str(ROOT/'experiments/role-analysis'))
import systemness_complementary_head_scan as s
EXPECTED='70db2015b90a496dc465c63631911da81d641614fcf0b1d48b04313a078ce156'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
START=sha(s.__file__);assert START==EXPECTED
CHECKS=[]
def check(n,x=True):assert bool(x),n;CHECKS.append(n)
def reject(n,fn):
    try:fn()
    except(RuntimeError,AssertionError,ValueError,KeyError,FileNotFoundError):CHECKS.append(n)
    else:raise AssertionError('Fault accepted: '+n)
def forbidden(*a,**kw):raise AssertionError('Actual model/array/metadata path forbidden in synthetic annex.')
folder=Path(tempfile.mkdtemp(prefix='feature-review-complementary-scanner-core-',dir='/tmp'))
old_globals=(s.old.OUT,s.old.CACHE,s.old.SHRINKAGE,s.old.TOP_K,s.data.runtime.atomic_json,s.data.runtime.status)

with patch.object(np,'load',side_effect=forbidden),patch.object(np,'memmap',side_effect=forbidden),patch.object(s.data.runtime,'load_model',side_effect=forbidden):
    check('Final held source70db exact',START==EXPECTED)
    # Reject unsafe checkpoint strings without stat/open/resolve, never touch the hung mount.
    with patch.object(Path,'stat',side_effect=forbidden),patch.object(Path,'open',side_effect=forbidden),patch.object(Path,'resolve',side_effect=forbidden):
        s.require_checkpoint_path('/dev/shm/role-analysis-gptoss120b')
        check('Exact verified checkpoint guard performs no filesystem access')
        for path in ['/workspace','/workspace/model','/tmp/model','/dev/shm/role-analysis-gptoss120b/../role-analysis-gptoss120b']:
            reject('Checkpoint alias rejected before file access '+path,lambda path=path:s.require_checkpoint_path(path))
        reject('Materialize rejects invalid checkpoint before tokenizer/import/read',lambda:s.materialize({},'/workspace'))

    # Exercise projected metadata API with count-only stand-ins; no real parquet/read is invoked.
    class CountFrame:
        def __init__(self,n,docs=None):self.n=n;self.document_id=SimpleNamespace(nunique=lambda:docs)
        def __len__(self):return self.n
    frames=[CountFrame(660),CountFrame(168960,132),CountFrame(255),CountFrame(63403)]
    requests=[]
    def read_meta(path,**kwargs):requests.append((str(path),kwargs));return frames[len(requests)-1]
    with patch.object(pd,'read_parquet',side_effect=read_meta),patch.object(s,'read',return_value={'synthetic':True}),patch.object(s,'partition_metadata',return_value=('views','partition')):
        assert s.metadata_views()==('views','partition')
    check('All four metadata reads explicitlyfit-only, publicinputs eligible, no prompt column',len(requests)==4 and all(('split','=','fit') in k['filters'] and 'prompt' not in k.get('columns',[]) for _,k in requests) and ('eligible_fit','=',True) in requests[2][1]['filters'])
    check('Metadata uses originalC4 and clean natural caches, excludes new256',requests[0][0]==str(s.native.CACHE/'inputs.parquet') and requests[1][0]==str(s.native.CACHE/'rows.parquet') and requests[3][0]==str(s.NAT_FEATURE_CACHE/'rows.parquet'))

    # Actual Arrow predicate, in-memory tables with deliberate cross-split ID collisions.
    c4_table=pa.table(dict(prompt_ix=[11,11,12,12],split=['fit','validation','fit','validation'],prompt=['c4stat','BAD_VAL','c4hold','BAD_VAL2']))
    nat_table=pa.table(dict(prompt_ix=[21,21,21,22,22,22],split=['fit','validation','fit']*2,
                          eligible_fit=[True,True,False]*2,prompt=['pubstat','BAD_VAL','BAD_UNCLEAN','pubhold','BAD_VAL2','BAD_UNCLEAN2']))
    datasets={'C4':ds.dataset(c4_table),'public':ds.dataset(nat_table)}
    views={}
    for phase,c4ix,pubix,doc in [('statistics',11,21,1),('holdout',12,22,5)]:
        inputs=pd.DataFrame([dict(uid='long-full:'+str(c4ix),source_prompt_ix=c4ix,domain='C4'),
                             dict(uid='public:'+str(pubix),source_prompt_ix=pubix,domain='public')])
        rows=pd.DataFrame([dict(uid='long-full:'+str(c4ix),domain='C4',component_id='doc:'+str(doc),document_id=doc),
                           dict(uid='public:'+str(pubix),domain='public',component_id='public:'+phase,document_id=-1)])
        views[phase]=dict(inputs=inputs,rows=rows)
    token_map={'c4stat':101,'c4hold':102,'pubstat':201,'pubhold':202};seen=[]
    class ToyTokenizer:
        bridge=False
        def decode(self,ids,**kw):return 'body:'+str(101 if self.bridge and ids==[102] else ids[0])
    tok=ToyTokenizer()
    class ToyBundle:
        def __init__(self,inputs,rows,tokenizer):
            self.inputs=inputs;self.rows=rows;seen.extend(inputs.prompt.tolist())
            self.tokens={r.uid:[token_map[r.prompt]] for r in inputs.itertuples()}
    keys=[dict(split='fit',document_id=1,source_url='source://one'),dict(split='fit',document_id=5,source_url='source://five')]
    def dataset(path,**kw):return datasets['C4' if Path(path).parent==s.native.CACHE else 'public']
    with patch.object(ds,'dataset',side_effect=dataset),patch.object(AutoTokenizer,'from_pretrained',return_value=tok),patch.object(s.old,'Bundle',ToyBundle),patch.object(s.native,'parse_native_labels',return_value={0:'user'}),patch.object(s,'read',return_value=keys):
        bundles,proof=s.materialize(views,'/dev/shm/role-analysis-gptoss120b')
        check('Arrow filters validation andunclean colliding IDs before prompt materialization',seen==['c4stat','pubstat','c4hold','pubhold'] and len(bundles)==2 and proof['cross_partition_bridges']==0)
        tok.bridge=True
        reject('Native full/prefix cross-partition body bridge stops without moving sources',lambda:s.materialize(views,'/dev/shm/role-analysis-gptoss120b'))
    check('Metadata duplicate guard permits samephase repeats only',s.fingerprint_audit([('full','same','statistics','a'),('full','same','statistics','b')])['cross_partition_bridges']==0)
    reject('Metadata exactURL bridge halts before any LM',lambda:s.fingerprint_audit([('sourceURL','same','statistics','a'),('sourceURL','same','holdout','b')]))

    # Full old101 reuse contract with mock moment values, no original NPZ is opened.
    desc={'prompts':505,'rows':129280,'input_tokens':520541,'inputs_sha256':'synthetic','rows_sha256':'synthetic','sample_ix_sha256':'synthetic'}
    class Old101:
        rows=range(129280)
        def descriptor(self):return desc
        def batches(self):return [None]*127
    bundle=Old101();moment=SimpleNamespace(n=np.full((2304,5),25856,np.int64))
    meta=dict(plan_sha256=s.OLD_PLAN_SHA,phase='fit',next_batch=127,rows_written=129280,class_counts=[25856]*5,data=desc)
    plan={'signature':dict(old_C4_moment_reuse=True,old_moment_path='/tmp/toy-old101-not-opened.npz',old_fit_complete={'data':desc})}
    with patch.object(s.data,'digest',return_value=s.OLD_MOMENTS_SHA),patch.object(s.old,'load_moments',return_value=(moment,meta)):
        check('Old101 reuse requires full descriptor/plan/density/head count contract',s.reused_moments(bundle,plan) is moment)
        bad=dict(meta,phase='discovery')
        with patch.object(s.old,'load_moments',return_value=(moment,bad)):
            reject('Old discovery orrefit moments cannot masquerade as101fit',lambda:s.reused_moments(bundle,plan))
        wrong=SimpleNamespace(n=moment.n.copy());wrong.n[0,0]-=1
        with patch.object(s.old,'load_moments',return_value=(wrong,meta)):
            reject('Any oldhead classcount discrepancy refuses reuse',lambda:s.reused_moments(bundle,plan))

    # Small complete-batch prefix + incomplete restore/discovery gates before any get_model.
    bundle=object.__new__(s.old.Bundle)
    bundle.inputs=pd.DataFrame(dict(uid=['a','b','c','d','e'],length=[2]*5,prompt=['own-synthetic']*5,domain=['C4']*5))
    bundle.rows=pd.DataFrame([dict(uid=u,sample_ix=i*2+j,token_ix=j,token_id=10+i*2+j,role=['system','user'][j],document_id=i,domain='C4') for i,u in enumerate(bundle.inputs.uid) for j in [0,1]])
    bundle.tokens={u:[10+2*i,11+2*i] for i,u in enumerate(bundle.inputs.uid)};bundle.lookup={u:g for u,g in bundle.rows.groupby('uid',sort=False)}
    prefixes=[s.prefix(bundle,i,'own',plan_sha='own-plan') for i in [0,1,2]]
    check('Prefix covers exact complete batches0/1/2 not token cursor',[(v['rows'],v['class_counts']) for v in prefixes]==[(0,[0]*5),(8,[4,4,0,0,0]),(10,[5,5,0,0,0])])
    reject('Out-of-range moment prefix rejected',lambda:s.prefix(bundle,3,'own',plan_sha='own-plan'))
    out=folder/'own-output';cache=folder/'own-cache';out.mkdir();cache.mkdir();(out/'plan.json').write_text('{}')
    (cache/'additional-moments.npz').write_bytes(b'own-placeholder')
    with patch.object(s,'OUT',out),patch.object(s,'CACHE',cache):
        bad=s.prefix(bundle,1,'additional-statistics');bad['sample_ix_sha256']='wrong-prefix'
        with patch.object(s.old,'load_moments',return_value=(moment,bad)):
            reject('Wrong committed batchprefix rejected before get_model',lambda:s.statistics(forbidden,bundle,{'signature':{'old_C4_moment_reuse':False}}))
        (cache/'discovery-system-probabilities.npy').write_bytes(b'own-incomplete-placeholder')
        reject('Incomplete probability matrix cannot restore bystatus/cursor',lambda:s.discovery(forbidden,bundle,None,None,{},False))

    text=Path(s.__file__).read_text();tree=ast.parse(text)
    signature=inspect.getsource(s.signature);materialize=inspect.getsource(s.materialize);model=inspect.getsource(s.model_context);scan=inspect.getsource(s.scan)
    check('All three checkpoint consumers guard before checkpoint file/tokenizer/model access',signature.index('require_checkpoint_path(checkpoint)')<signature.index("bind(checkpoint/name") and materialize.index('require_checkpoint_path(checkpoint)')<materialize.index('AutoTokenizer.from_pretrained') and model.index('require_checkpoint_path(checkpoint)')<model.index('AutoModelForCausalLM.from_pretrained'))
    check('Old101 reuse validation occurs preGPU andloader islazy',scan.index('previous=reused_moments(')<scan.index('with ExitStack()') and 'if not loaded:loaded.append(stack.enter_context(model_context(plan)))' in scan)
    check('NativeMXFP4 fixed4A40 eager sequential layer mapping retained',all(t in model for t in ['torch.cuda.device_count()==4',"'A40' in torch.cuda.get_device_name(i)",'Mxfp4Config(dequantize=False)',"attn_implementation='eager'",'use_kernels=False',"f'model.layers.{i}':i//9",'parameters_frozen=True']))
    fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['signature','metadata_views','verify_plan']]
    check('No historical source/globalinventory verifier calls in static core',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'source_signature','verified_completion','exclusion_inputs','rglob'} for f in fns for n in ast.walk(f)))
    check('Frozen old namespace andruntime writers unchanged',old_globals==(s.old.OUT,s.old.CACHE,s.old.SHRINKAGE,s.old.TOP_K,s.data.runtime.atomic_json,s.data.runtime.status))
    check('Reviewed finalsource hash unchanged',sha(s.__file__)==START)

report=dict(stage='independent_complementary_head_scanner_implementation_annex',audit_passed=True,
    reviewed_source_sha256=START,reviewed_source_sha256_after=sha(s.__file__),checks_passed=len(CHECKS),checks=CHECKS,
    source_path=str(Path(__file__).resolve()),source_sha256=sha(__file__),
    pure_partial_AUROC_and_32_496_math_review_delegated=True,
    actual_metadata_views_or_source_text_or_moment_values_read=False,actual_heads_or_probabilities_read=False,
    model_or_tokenizer_loaded=False,declare_or_scan_executed=False,gpu_used=False,validation_or_tests_read=False,formal_outputs_modified=False)
dest=folder/'report.json';dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(report=str(dest),report_sha256=sha(dest),**report)))
