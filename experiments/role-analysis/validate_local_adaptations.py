"""Check the two scientific transformations that differ from the source notebooks."""
import json
from pathlib import Path

import cupy
import numpy as np
import pandas as pd
import torch

from local_runtime import aggregate_conversation_projections, parse_latest_response
from utils.probes import run_projections


class ProbabilityProbe:
    def predict_proba(self, x):
        return x


def main():
    history = '<|start|>assistant<|channel|>analysis<|message|>old thoughts<|end|><|start|>assistant<|channel|>final<|message|>old answer<|return|>'
    latest = '<|start|>assistant<|channel|>analysis<|message|>new thoughts<|end|><|start|>assistant<|channel|>final<|message|>new answer<|return|>'
    assert parse_latest_response(history+latest) == ('new thoughts','new answer')
    assert parse_latest_response('<|start|>assistant<|channel|>analysis<|message|>unfinished') == (None,None)
    # Unequal lengths make incorrect pooling across conversations detectable.
    sample = pd.DataFrame({'sample_ix':range(6),'prompt_ix':[0,0,0,1,2,2],
                           'role':['user','assistant','user','user','assistant','user']})
    inputs = pd.DataFrame({'prompt_ix':[0,1,2],'dataset':['toy']*3,
                           'conv_id':[0,1,2],'conv_type':['tagged','tagged','untagged']})
    hs = torch.tensor([[.9,.1],[.5,.5],[.8,.2],[.1,.9],[.2,.8],[.4,.6]])
    probe = {'role_space':['user','assistant'],'layer_ix':0,'probe':ProbabilityProbe()}
    probability,hard = aggregate_conversation_projections([probe],sample,inputs,{0:hs})
    full = (run_projections(sample,hs,probe).assign(role_space='ua',layer_ix=0)
            .merge(sample,on='sample_ix').merge(inputs,on='prompt_ix'))
    keys = ['conv_type','role_space','layer_ix','role']
    reference_probability = (full[full.target_role==full.role].groupby(keys+['prompt_ix'],as_index=False)
        .agg(mean_acc=('prob','mean')).groupby(keys,as_index=False).agg(mean_acc=('mean_acc','mean')))
    winners = (full.sort_values(keys[:3]+['sample_ix','prob'],ascending=[True]*4+[False])
        .drop_duplicates(keys[:3]+['sample_ix']).assign(is_acc=lambda d:d.target_role==d.role))
    reference_hard = (winners.groupby(keys+['prompt_ix'],as_index=False)
        .agg(mean_acc=('is_acc','mean')).groupby(keys,as_index=False).agg(mean_acc=('mean_acc','mean')))
    pd.testing.assert_frame_equal(probability,reference_probability)
    pd.testing.assert_frame_equal(hard,reference_hard)
    pooled = float(full[(full.target_role=='user')&(full.role=='user')&(full.conv_type=='tagged')].prob.mean())
    conversation_mean = float(probability[(probability.role=='user')&(probability.conv_type=='tagged')].mean_acc.iloc[0])
    assert abs(pooled-conversation_mean)>.1
    result = {'harmony_channel_extraction':'passed','streaming_probability_equals_original':'passed',
              'streaming_argmax_equals_original_including_ties':'passed',
              'unequal_conversation_lengths':'passed'}
    destination = Path(__file__).resolve().parent/'outputs/local-gptoss20b/adaptation_validation.json'
    destination.write_text(json.dumps(result,indent=2))
    print(result)


if __name__=='__main__':
    main()
