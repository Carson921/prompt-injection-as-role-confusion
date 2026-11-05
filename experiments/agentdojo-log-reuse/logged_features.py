"""Forward recorded full inputs without generating or reevaluating their ASR."""
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import save_file

from common import BASE, MERGED, read, write, sha, signature


def capture_logged(model, tokenizer, engine):
    manifest = read(MERGED / 'manifest.json')
    assert manifest['plan_sha256'] == sha(MERGED / 'plan.json')
    protocol = read(BASE / 'manifest.json')
    plan = [r for r in read(MERGED / 'plan.json') if r['feature_method']=='forward_logged_input']
    started = time.time()
    for index, record in enumerate(plan):
        source = MERGED / record['input_path']
        assert sha(source) == record['input_sha256']
        inp = read(source)
        directory = source.parent
        receipt_path = directory / 'capture.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            assert receipt['input_sha256']==sha(source)
            assert receipt['states_sha256']==sha(directory/'states.safetensors')
            continue
        ids = inp['input_ids']
        assert tokenizer.encode(inp['rendered_prompt'], add_special_tokens=False)==ids
        assert len(ids) <= model.config.max_position_embeddings
        positions = np.asarray(inp['positions'],dtype=np.int64)
        assert not np.isin(np.asarray(ids)[positions], protocol['delimiter_ids']).any()
        assert len(positions)==record['rows']
        case_started = time.time()
        features, _ = engine.forward_capture(model, protocol['head_ids_union'], ids, positions)
        payload = np.asarray(inp['payload_mask'],dtype=np.bool_)
        goal = np.asarray(inp['goal_mask'],dtype=np.bool_)
        assert features.shape==(len(positions),len(protocol['head_ids_union']),64)
        arrays = {'features_union':features, 'positions':torch.from_numpy(positions),
                  'token_ids':torch.tensor(ids,dtype=torch.int64)[torch.from_numpy(positions)],
                  'payload_mask':torch.from_numpy(payload),'goal_mask':torch.from_numpy(goal),
                  'template_mask':torch.from_numpy(payload & ~goal),'context_mask':torch.from_numpy(~payload)}
        state_path = directory/'states.safetensors'
        tmp = directory/'states.safetensors.tmp'
        save_file(arrays,str(tmp),metadata={'format':'recorded-full-input-selected-heads-v1',
                  'input_sha256':sha(source),'model_binding':protocol['model_binding']['identity_sha256']})
        tmp.replace(state_path)
        write(receipt_path, {'status':'captured','case_id':record['case_id'], 'input_sha256':sha(source),
              'states_sha256':sha(state_path),'head_ids_union':protocol['head_ids_union'],
              'actual_root_input_ids_verified':True,'original_saved_prompt_and_ids':inp['original_saved_prompt_and_ids'],
              'asr_generation_performed':False,'rows':len(positions),'seconds':time.time()-case_started})
        write(MERGED/'feature_progress.json',{'stage':'forward_existing_logs','completed':index+1,'total':len(plan),
              'last_case':record['case_id'],'elapsed_seconds':time.time()-started})
        print(signature({'case':record['case_id'],'receipt':sha(receipt_path)})+' '+record['case_id'],flush=True)
    write(MERGED/'feature_terminal.json',{'status':'complete','forward_cases':len(plan),'asr_generation_performed':False})
