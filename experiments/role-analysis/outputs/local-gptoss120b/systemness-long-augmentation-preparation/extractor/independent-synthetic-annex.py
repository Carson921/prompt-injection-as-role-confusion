"""Independent CPU fault audit; no real builder, candidate text, GPU or extraction."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,'/root/prompt-injection-as-role-confusion/experiments/role-analysis')
import systemness_long_augmentation_extract as e

EXPECTED='907204a31cea06bd3885d7148bd2684aaa77b9d3a322737cb8b3ad0c4629fd86'
start=e.data.digest(e.__file__)
assert start==EXPECTED
folder=Path(tempfile.mkdtemp(prefix='long-data-augmentation-extractor-audit-',dir='/tmp'))
checks=[]

def reject(fn,label):
    try:fn()
    except (RuntimeError,FileNotFoundError):checks.append(label)
    else:raise AssertionError('Fault accepted: '+label)

def forbidden(*args,**kwargs):raise AssertionError('Real builder/model path invoked')

assert e.feature_bytes()==398458880 and e.ROWS==389120 and e.PROMPTS==1760
assert e.RESERVE_BYTES==512*2**20 and e.MIN_GPU_FREE_BYTES==22*2**30
checks.append('exact389120x512_F16_bytes_and_reserve')
with patch.object(e,'builder',side_effect=forbidden),patch.object(e,'load_model',side_effect=forbidden):
    inputs,rows,tokens=e.synthetic_data()
    e.validate_layout(inputs,rows,tokens)
    bundle=e.Bundle(inputs,rows,tokens)
    seen=np.zeros(389120,dtype=np.int8)
    for part in bundle.parts:
        batch=bundle.collate(part);width=int(batch['input_ids'].shape[1]);reference=[];targets=[]
        for bi,entry in enumerate(part.itertuples()):
            group=rows[rows.uid.eq(entry.uid)]
            reference.extend((bi*width+width-len(tokens[entry.prompt_ix])+group.token_ix.to_numpy()).tolist())
            targets.extend(group.cache_ix.tolist())
        assert np.array_equal(batch['positions'],reference) and np.array_equal(batch['targets'],targets)
        assert np.array_equal(batch['input_ids'].reshape(-1)[batch['positions']].numpy(),rows.iloc[targets].token_id)
        seen[targets]+=1
    assert np.all(seen==1) and len(bundle.parts)==440
    assert rows.groupby('split').size().to_dict()=={'fit':327680,'validation':61440}
    assert rows.groupby(['document_id','role']).size().isin([128,256]).all()
checks.append('all440_batches_independent_flattened_leftpad_token_target_identity')
bad=rows.copy();bad.loc[0,'token_id']+=1
reject(lambda:e.validate_layout(inputs,bad,tokens),'reject_changed_token_id')
bad=rows.copy();bad.loc[0,'body_offset']=1
reject(lambda:e.validate_layout(inputs,bad,tokens),'reject_changed_body_offset')
bad=rows.copy();bad.loc[0,'split']='test'
reject(lambda:e.validate_layout(inputs,bad,tokens),'reject_test_split')

# Direct expected pre-o_proj rank columns, not a copied Collector implementation.
class Body(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.ModuleList([torch.nn.Module(),torch.nn.Module()])
        for layer in self.layers:layer.self_attn=torch.nn.Module();layer.self_attn.o_proj=torch.nn.Identity()
    def forward(self,input_ids,attention_mask=None,use_cache=False):
        x=self.layers[0].self_attn.o_proj(input_ids+10)
        x=self.layers[1].self_attn.o_proj(x+20)
        return SimpleNamespace(last_hidden_state=x)
class TinyBundle:
    parts=[None]
    def collate(self,part):
        return dict(input_ids=torch.arange(96,dtype=torch.float32).reshape(2,4,12),
            attention_mask=torch.ones(2,4),positions=np.array([6,1,4]),targets=np.array([2,0,1]))
array=np.full((3,16),np.nan,np.float16);selected=[5,0,3,2]
collector=e.Collector(array,selected,heads=3,dimensions=4);model=SimpleNamespace(model=Body())
hookout=folder/'hooks';hookout.mkdir()
with patch.object(e,'OUT',hookout):
    with e.checked_hooks(model,collector,TinyBundle(),'cpu'):
        assert np.isnan(array).all()
        batch=TinyBundle().collate(None);output=e.forward(model,collector,batch,'cpu')
        base=batch['input_ids'].reshape(-1,3,4).numpy();wanted=[]
        for head in selected:
            wanted.append((base+ (10 if head//3==0 else 30))[batch['positions'],head%3])
        expected=np.concatenate(wanted,axis=1).astype(np.float16)
        assert np.array_equal(array[batch['targets']],expected)
        assert torch.equal(output,batch['input_ids']+30)
    assert not collector.enabled and all(not layer.self_attn.o_proj._forward_pre_hooks for layer in model.model.layers)
    assert e.read(hookout/e.HOOK_NAME)['writes_disabled']
checks.append('unsorted_cross_layer_rank_selected_only_disabledwrite_bitexact_and_cleanup')

# Check uncommitted half-batch never advances, committed bytes are immutable.
small_inputs=pd.DataFrame(dict(uid=[f'u{i}' for i in range(6)],prompt_ix=np.arange(6),length=[5,3,6,4,7,2]))
small_rows=pd.DataFrame(dict(cache_ix=np.arange(12),sample_ix=6000000000000+np.arange(12),
    uid=np.repeat(small_inputs.uid,2),token_ix=np.tile([0,1],6),token_id=np.tile([11,22],6),role='system',
    body_offset=np.tile([0,1],6),split=['fit']*8+['validation']*4,source_set='long-augmentation'))
small=e.Bundle(small_inputs,small_rows);source=folder/'source-rows.parquet';small_rows.to_parquet(source,index=False)
cache=folder/'bank';bank=e.initialize(cache,source,'toy-plan',rows=12,dimensions=16)
first=small.targets(small.parts[0]);second=small.targets(small.parts[1])
bank[first]=np.arange(len(first)*16).reshape(-1,16);bank.flush()
parity={'max_difference':0.,'same_saved_f16_vectors':True,'rows':len(first)}
hashes=[e.batch_digest(bank,first)];e.save_progress(cache,small,'toy-plan',hashes,parity)
bank[second[:1],:8]=2;bank.flush()
assert e.restore_progress(cache,small,'toy-plan',bank)[0]==hashes
bank[second]=3;bank.flush()
assert e.restore_progress(cache,small,'toy-plan',bank)[0]==hashes
checks.append('uncommitted_nan_or_finite_partial_rows_do_not_advance')
old=bank[first[0],0].copy();bank[first[0],0]+=1;bank.flush()
reject(lambda:e.restore_progress(cache,small,'toy-plan',bank),'reject_committed_finite_byte_change')
bank[first[0],0]=old;bank[first[0],1]=np.nan;bank.flush()
reject(lambda:e.restore_progress(cache,small,'toy-plan',bank),'reject_committed_single_column_NaN')
bank[first]=np.arange(len(first)*16).reshape(-1,16);bank[second]=3;bank.flush()
hashes=[e.batch_digest(bank,small.targets(part)) for part in small.parts]
e.save_progress(cache,small,'toy-plan',hashes,parity)
reject(lambda:e.restore_progress(cache,small,'wrong-plan',bank),'reject_wrong_resume_plan')
del bank
feature=cache/'features.f16';feature.rename(cache/'saved.f16')
reject(lambda:e.initialize(cache,source,'toy-plan',rows=12,dimensions=16),'missing_initialized_feature_fails_without_recreation')
assert not feature.exists()
feature.write_bytes(b'bad')
reject(lambda:e.initialize(cache,source,'toy-plan',rows=12,dimensions=16),'truncated_initialized_feature_fails')
feature.unlink();(cache/'saved.f16').rename(feature)

# No completed-route builder/model call, and no feature/progress rewrite.
completed=folder/'complete-route';completed.mkdir();e.data.atomic_json(completed/e.PLAN_NAME,{'synthetic':True})
e.data.atomic_json(completed/'extraction-complete.json',{'synthetic':True})
before=e.data.digest(feature)
with patch.object(e,'OUT',completed),patch.object(e,'verify_plan',return_value=None), \
    patch.object(e,'verified_extraction',return_value={'verified_synthetic':True}), \
    patch.object(e,'load_data',side_effect=forbidden),patch.object(e,'load_model',side_effect=forbidden), \
    patch.object(e,'no_competing_jobs',side_effect=forbidden):
    e.run()
assert e.data.digest(feature)==before
checks.append('complete_route_skips_builder_gpu_and_feature_reinitialization')
with patch.object(e,'OUT',folder/'undeclared'),patch.object(e,'builder',side_effect=forbidden):
    reject(e.run,'undeclared_route_rejected_before_builder')

end=e.data.digest(e.__file__);assert end==start==EXPECTED
report=dict(source_sha256=start,source_end_sha256=end,independent_checks=checks,checks_passed=len(checks),
    all389120_rows_mapped=True,all1760_prompts=True,batches=440,feature_bytes=398458880,
    source_review=dict(native_model_exact4A40_MXFP4=True,frozen_eval_parameters_checked=True,
        same_order8head_identity_bound=True,only_sampled_positions_retained=True,
        complete_verifier_mode_readonly=True,cache_row_bytes_identity_bound=True,
        committed_prefix_batch_hashes_recomputed=True,model_config_weights_tokenizer_source_hashes_bound=True),
    real_builder_called=False,new_source_text_decoded=False,real_feature_values_read=False,
    gpu_work_run=False,formal_output_changed=False,no_must_fix_findings=True)
path=folder/'report.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(dict(report_path=str(path),**report)))
