"""Independent selected-head/target/cache boundary fixtures, CPU only."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,'/root/prompt-injection-as-role-confusion/experiments/role-analysis')
import systemness_aug_confirmation_infer as m

digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
before=digest(m.__file__); checks=[]
def check(name,value):
    assert value,name
    checks.append(name)
def rejects(name,call):
    try:call()
    except (AssertionError,RuntimeError,ValueError,FileNotFoundError):checks.append(name)
    else:raise AssertionError(name)

heads=[458,462,2045,460,459,2047,456,997]
positions=np.array([4,1,3]); collector=m.Collector(heads);collector.begin(positions)
blocks={}
for layer in reversed(sorted(set(h//64 for h in heads))):
    value=(torch.arange(6*4096).reshape(2,3,4096).float()/100000+layer/10)
    blocks[layer]=value
    collector.hook(layer)(None,(value,))
collector.finish()
expected=np.concatenate([blocks[h//64].numpy().reshape(6,64,64)[positions,h%64,:] for h in heads],axis=1).astype(np.float16)
check('Independent rank-order eight-head and nonsequential positions',
    collector.features.dtype==np.float16 and np.array_equal(collector.features,expected))
rejects('Repeated selected layer fails instead of overwriting rank columns',
    lambda:collector.hook(heads[0]//64)(None,(blocks[heads[0]//64],)))
collector.begin(positions,enabled=False)
for layer,block in blocks.items():collector.hook(layer)(None,(block,))
collector.finish()
check('Disabled capture allocates no feature output',collector.features is None)

tokens=[[11,12,13],[21,22],[31,32,33,34]]
rows=pd.DataFrame({'prompt_ix':[0,0,1,2],'token_ix':[0,2,1,0],'cache_ix':range(4),'sample_ix':np.arange(4)+7*10**12})
ctx=SimpleNamespace(inputs=pd.DataFrame({'prompt_ix':[0,1,2],'length':[3,2,4]}),rows=rows,tokens=tokens)
batch=m.collate(ctx,[2,0,1])
flat=batch['input_ids'].numpy().reshape(-1)
check('Left padding and non-prompt-order targets preserve actual IDs',
    batch['targets'].tolist()==[3,0,1,2] and flat[batch['positions']].tolist()==[31,11,13,22] and
    batch['attention_mask'].sum(1).tolist()==[4,3,2])

folder=Path(tempfile.mkdtemp(prefix='root-aug-infer-annex-'))
ctx=SimpleNamespace(cache=folder/'probabilities',signature={'synthetic':True},
    inputs=pd.DataFrame({'prompt_ix':np.arange(6),'length':[2]*6}),
    rows=pd.DataFrame({'prompt_ix':np.repeat(np.arange(6),2),'cache_ix':np.arange(12),'sample_ix':np.arange(12)+7*10**12}))
values=m.initialize_probabilities(ctx)
check('New probability files begin entirely unwritten',all(np.isnan(v).all() for v in values.values()) and m.restore_progress(ctx,values)==0)
for value in values.values():value[:8]=np.array([.2]*5,np.float32)
m.checkpoint(ctx,values,1)
check('Resume accepts only exact committed rows',m.restore_progress(ctx,values)==1 and
    json.loads((ctx.cache/'progress.json').read_text())['rows_written']==8)
saved=values['new'][0].copy(); values['new'][0]=np.array([.1,.3,.2,.2,.2],np.float32)
rejects('Changed normalized committed prefix is rejected',lambda:m.restore_progress(ctx,values))
values['new'][0]=saved; values['new'].flush()
records=m.records_from_physical(values)
check('Old-head threshold aliases share one array and argmax',
    records['old-head-maximin'] is records['old-head-original'] is values['old-head'] and
    np.array_equal(records['old-head-maximin'][:8].argmax(1),records['old-head-original'][:8].argmax(1)))
check('Reviewed infer source unchanged',digest(m.__file__)==before)
report=dict(stage='independent_synthetic_shared_infer_annex',source_sha256=before,checks=checks,checks_passed=len(checks),
    actual_builder_or_snapshot_read=False,actual_test_probability_read=False,actual_lm_loaded=False,gpu_used=False,
    formal_outputs_written=False,fixture_directory=str(folder))
path=Path('/tmp/root_aug_infer_annex.json');path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report_path':str(path),'report_sha256':digest(path),'checks_passed':len(checks)}))
