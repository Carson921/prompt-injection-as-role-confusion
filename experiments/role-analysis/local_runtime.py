"""Memory-bounded local adapters for the original role-space notebooks.

The scientific settings and role templates live in the original notebooks and
config files. These adapters load the local checkpoint, capture actual model
activations with hooks, cache local generations, and stream activations to disk.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
OUT = HERE / 'outputs/local-gptoss20b'
CACHE = OUT / 'cache'
KERNEL_REVISION = '9fbc1b48e710ac8abf18ea2650ee690d52dc3859'
_loaded = None
_capture_layers = None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def load_model_and_tokenizer(model_prefix, device='cuda:0'):
    global _loaded
    assert model_prefix == 'gptoss-20b'
    if _loaded is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
        from transformers.integrations import hub_kernels
        torch.set_num_threads(8)
        original = hub_kernels.get_kernel
        def pinned(name, *args, **kwargs):
            if name == 'kernels-community/gpt-oss-triton-kernels':
                kwargs['revision'] = KERNEL_REVISION
            return original(name, *args, **kwargs)
        hub_kernels.get_kernel = pinned
        dequantize = os.environ.get('ROLE_ANALYSIS_MODEL_BACKEND') == 'bf16'
        model = AutoModelForCausalLM.from_pretrained(
            '/workspace/gpt-oss-20b', local_files_only=True,
            dtype=torch.bfloat16, device_map={'': device},
            quantization_config=Mxfp4Config(dequantize=dequantize),
            attn_implementation='eager', use_kernels=False,
        ).eval()
        if dequantize:
            model.set_experts_implementation('eager')
        else:
            assert type(model.model.layers[0].mlp.experts).__name__ == 'Mxfp4GptOssExperts'
        tokenizer = AutoTokenizer.from_pretrained(
            '/workspace/gpt-oss-20b', local_files_only=True,
            add_eos_token=False, add_bos_token=False, padding_side='left',
        )
        _loaded = (tokenizer, model, 'gptoss', 24)
        print('Loaded local 20B:', 'BF16 dequantized/eager experts' if dequantize else 'MXFP4/Triton experts',
              'BF16 activations, eager attention',flush=True)
    return _loaded


@torch.no_grad()
def run_forward_with_hooks(model, input_ids, attention_mask, return_hidden_states=False):
    selected = set(range(24)) if _capture_layers is None else set(_capture_layers)
    pre = [None] * 24
    handles = []
    if return_hidden_states:
        def hook(index):
            def capture(module, args, output):
                pre[index] = output.reshape(-1, output.shape[-1]).detach().to(device='cpu', dtype=torch.float16)
            return capture
        for i in selected:
            handles.append(model.model.layers[i].post_attention_layernorm.register_forward_hook(hook(i)))
    try:
        result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        return {'logits': result.logits, 'all_pre_mlp_hidden_states': pre, 'all_hidden_states': []}
    finally:
        for handle in handles:
            handle.remove()


def load_custom_forward_pass(model_architecture, model=None, tokenizer=None):
    assert model_architecture == 'gptoss'
    if model is not None:
        encoded = tokenizer(['Hi! I am a dog and I like to bark', 'Vegetables are good for'],
                            return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            reference = model(**encoded, use_cache=False).logits
            captured = run_forward_with_hooks(model, encoded.input_ids, encoded.attention_mask, True)
        difference = float((reference.float() - captured['logits'].float()).abs().max())
        exact = torch.equal(reference, captured['logits'])
        assert torch.allclose(reference, captured['logits'], atol=.05, rtol=.01), difference
        assert len(captured['all_pre_mlp_hidden_states']) == 24
        result = {'exact_logits': exact, 'max_absolute_logit_difference': difference,
                  'layers_captured': 24, 'capture_point': 'post_attention_layernorm output'}
        atomic_json(OUT / 'forward_validation.json', result)
        print('Actual-model forward hook validation:', result, flush=True)
    return run_forward_with_hooks


def parse_latest_response(state_text):
    """Extract the latest analysis/final pair without folding thoughts into history."""
    segments = re.findall(
        r'<\|start\|>assistant<\|channel\|>(analysis|final)<\|message\|>(.*?)(?=<\|(?:end|return|call)\|>|$)',
        state_text, flags=re.DOTALL,
    )
    if not segments or segments[-1][0] != 'final':
        return None, None
    final = segments[-1][1].strip()
    cot = segments[-2][1].strip() if len(segments) >= 2 and segments[-2][0] == 'analysis' else None
    return cot, final or None


def generate_local_batch(message_histories):
    tokenizer, model, _, _ = load_model_and_tokenizer('gptoss-20b')
    params = {'max_new_tokens':4000, 'temperature':1.0, 'max_input_tokens':12288,
              'checkpoint':'/workspace/gpt-oss-20b', 'kernel_revision':KERNEL_REVISION}
    digest = hashlib.sha256(json.dumps({'messages':message_histories, 'params':params}, sort_keys=True).encode()).hexdigest()
    path = CACHE / 'generation_batches' / f'{digest}.json'
    if path.exists():
        result = json.loads(path.read_text())
        # Normalize completion markers; early cache migration could append EOS twice.
        if result.get('parser_version') != 2:
            for i, detail in enumerate(result['details']):
                state = result['state_text'][i]
                if state.endswith(tokenizer.eos_token*2):
                    state = state[:-len(tokenizer.eos_token)]
                    result['state_text'][i] = state
                    detail['generated_tokens'] -= 1
                complete = state.endswith(tokenizer.eos_token)
                cot, final = parse_latest_response(result['state_text'][i])
                result['assistant'][i] = final if complete else None
                detail['ended_with_eos'] = complete
            result['pad_eos_handling'] = 'retain first EOS; remove only subsequent padding'
            result['parser_version'] = 2
            atomic_json(path,result)
        print('Reusing generation batch:', digest[:10], len(message_histories), flush=True)
        return result['assistant'], result['state_text']
    encoded = tokenizer.apply_chat_template(
        message_histories, tokenize=True, add_generation_prompt=True,
        return_tensors='pt', padding=True, truncation=True, max_length=12288, return_dict=True,
    )
    # Long second turns need a smaller prefill batch under eager attention.
    if len(message_histories) > 2 and encoded.input_ids.shape[1] > 2048:
        chunks = [message_histories[i:i+2] for i in range(0, len(message_histories), 2)]
        results = [generate_local_batch(chunk) for chunk in chunks]
        return [v for a,_ in results for v in a], [v for _,s in results for v in s]
    seed = int(digest[:8], 16) % 2**31
    torch.manual_seed(seed)
    started = time.time()
    from transformers import StoppingCriteria, StoppingCriteriaList
    class Progress(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            steps = input_ids.shape[1] - encoded.input_ids.shape[1]
            if steps % 250 == 0:
                elapsed = time.time()-started
                atomic_json(CACHE/'generation-progress.json', {
                    'batch':digest,'batch_size':len(message_histories),
                    'generated_steps':steps,'seconds':elapsed,'unix_time':time.time(),
                })
                print(f'Generation batch {digest[:10]}: {steps}/4000 steps, {elapsed:.1f}s, batch {len(message_histories)}',flush=True)
            return False
    out_of_memory = False
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded.to(model.device), max_new_tokens=4000, do_sample=True,
                temperature=1.0, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True,
                stopping_criteria=StoppingCriteriaList([Progress()]),
            )
    except torch.OutOfMemoryError:
        out_of_memory = True
    if out_of_memory:
        import gc
        del encoded
        gc.collect()
        torch.cuda.empty_cache()
        if len(message_histories) == 1:
            raise RuntimeError('A single conversation exceeded GPU memory')
        midpoint = len(message_histories) // 2
        results = [generate_local_batch(message_histories[:midpoint]), generate_local_batch(message_histories[midpoint:])]
        return [v for a,_ in results for v in a], [v for _,s in results for v in s]
    assistants, states, details = [], [], []
    prompt_length = encoded.input_ids.shape[1]
    for i in range(len(message_histories)):
        prompt = encoded.input_ids[i][encoded.attention_mask[i].bool()].tolist()
        new = generated[i, prompt_length:].tolist()
        complete = tokenizer.eos_token_id in new
        if complete:
            new = new[:new.index(tokenizer.eos_token_id)+1]
        else:
            while new and new[-1] == tokenizer.pad_token_id:
                new.pop()
        state = tokenizer.decode(prompt + new, skip_special_tokens=False)
        cot, final = parse_latest_response(state)
        assistants.append(final if complete else None)
        states.append(state)
        details.append({'generated_tokens':len(new), 'ended_with_eos':complete,
                        'cot_characters':len(cot or ''), 'final_characters':len(final or '')})
    result = {'assistant':assistants, 'state_text':states, 'details':details,
              'seed':seed, 'seconds':time.time()-started, 'params':params,
              'pad_eos_handling':'retain first EOS; remove only subsequent padding',
              'parser_version':2}
    atomic_json(path, result)
    print('Generated local batch:',len(assistants),'seconds',round(result['seconds'],1),
          'tokens',sum(r['generated_tokens'] for r in details),'complete',sum(r['ended_with_eos'] for r in details),flush=True)
    return assistants, states


@torch.no_grad()
def run_and_export_states(model, tokenizer, *, run_model_return_states, dl, layers_to_keep_acts,
                          extraction_key='all_pre_mlp_hidden_states'):
    """Same token order/activations as the original helper, backed by a file map."""
    global _capture_layers
    from utils.store_outputs import convert_outputs_to_df_fast
    assert extraction_key == 'all_pre_mlp_hidden_states'
    layers = list(layers_to_keep_acts)
    ids = dl.dataset.input_ids
    mask = dl.dataset.attention_mask
    backend = type(model.model.layers[0].mlp.experts).__name__
    digest = hashlib.sha256(ids.numpy().tobytes() + mask.numpy().tobytes() + repr(layers).encode()+backend.encode()).hexdigest()[:20]
    folder = CACHE / 'activations' / digest
    folder.mkdir(parents=True, exist_ok=True)
    n = int(mask.sum())
    shape = (len(layers), n, model.config.hidden_size)
    path = folder / 'hidden_states.fp16'
    meta_path = folder / 'complete.json'
    if meta_path.exists() and path.exists():
        meta = json.loads(meta_path.read_text())
        assert meta['shape'] == list(shape)
        assert path.stat().st_size == int(np.prod(shape))*2
        tensor = torch.from_file(str(path), shared=True, size=int(np.prod(shape)), dtype=torch.float16).view(shape)
        print('Reusing activation map:',shape,folder,flush=True)
        return {'sample_df':pd.read_parquet(folder/'tokens.parquet'), 'all_hs':tensor.permute(1,0,2), 'cache_path':str(path)}
    needed = int(np.prod(shape))*2
    import shutil
    if needed + 2**30 > shutil.disk_usage(folder).free:
        raise RuntimeError(f'Activation map requires {needed/2**30:.2f} GiB; insufficient disk space')
    tensor = torch.from_file(str(path), shared=True, size=int(np.prod(shape)), dtype=torch.float16).view(shape)
    _capture_layers = layers
    sample_frames = []
    cursor = 0
    started = time.time()
    try:
        for batch in tqdm(dl, desc='Stream actual-model activations'):
            input_ids = batch['input_ids'].to(model.device)
            attention = batch['attention_mask'].to(model.device)
            output = run_model_return_states(model,input_ids,attention,return_hidden_states=True)
            valid = attention.cpu().reshape(-1).bool()
            count = int(valid.sum())
            for save_index,layer in enumerate(layers):
                tensor[save_index,cursor:cursor+count].copy_(output[extraction_key][layer][valid])
            sample = convert_outputs_to_df_fast(input_ids,attention,output['logits'])
            tokens = pd.DataFrame([(s,t,text) for s,row in enumerate(batch['original_tokens']) for t,text in enumerate(row)],
                                  columns=['sequence_ix','token_ix','token'])
            prompts = pd.DataFrame({'sequence_ix':range(len(batch['prompt_ix'])),'prompt_ix':batch['prompt_ix']})
            sample_frames.append(sample.merge(tokens,on=['sequence_ix','token_ix'],how='left').merge(prompts,on='sequence_ix').drop(columns='sequence_ix'))
            cursor += count
            del output
        assert cursor == n
    finally:
        _capture_layers = None
    sample_df = pd.concat(sample_frames,ignore_index=True)
    assert len(sample_df) == n
    sample_df.to_parquet(folder/'tokens.parquet',index=False)
    atomic_json(meta_path,{'shape':list(shape),'layers':layers,'seconds':time.time()-started,'dtype':'float16'})
    print('Completed disk-backed activation map:',shape,'GiB',round(needed/2**30,2),flush=True)
    return {'sample_df':sample_df,'all_hs':tensor.permute(1,0,2),'cache_path':str(path)}


def aggregate_conversation_projections(all_probes, sample_df, inputs, hidden_states):
    """Original notebook statistics, aggregated one probe at a time.

    First average tokens within each conversation/semantic role, then give every
    conversation equal weight. Do not pool all tokens across conversations.
    """
    from utils.probes import run_projections
    probabilities, accuracies = [], []
    keys = ['conv_type','role_space','layer_ix','role']
    token_metadata = sample_df[['prompt_ix','sample_ix','role']]
    prompt_metadata = inputs[['prompt_ix','dataset','conv_id','conv_type']]
    for probe in tqdm(all_probes, desc='Aggregate all 96 conversation projections'):
        valid = sample_df[sample_df['role'].isin(probe['role_space'])]
        projected = (run_projections(valid,hidden_states[probe['layer_ix']],probe)
            .assign(layer_ix=probe['layer_ix'],role_space=''.join(r[0] for r in probe['role_space']))
            .merge(token_metadata,on='sample_ix',validate='many_to_one')
            .merge(prompt_metadata,on='prompt_ix',validate='many_to_one'))
        true_class = projected[projected.target_role == projected.role]
        probabilities.append(true_class.groupby(keys+['prompt_ix'],as_index=False)
            .agg(mean_acc=('prob','mean')).groupby(keys,as_index=False).agg(mean_acc=('mean_acc','mean')))
        winners = (projected.sort_values(keys[:3]+['sample_ix','prob'],ascending=[True]*4+[False])
            .drop_duplicates(keys[:3]+['sample_ix']).assign(is_acc=lambda d:d.target_role == d.role))
        accuracies.append(winners.groupby(keys+['prompt_ix'],as_index=False)
            .agg(mean_acc=('is_acc','mean')).groupby(keys,as_index=False).agg(mean_acc=('mean_acc','mean')))
        del projected, true_class, winners
    return pd.concat(probabilities,ignore_index=True), pd.concat(accuracies,ignore_index=True)


def diagnose_midlayer_probe(sample_df, hidden_states, all_probes, seed=123):
    """A float64 sensitivity check; does not overwrite the original 96 probes."""
    import cupy
    import cuml
    import gc
    import pickle
    from sklearn.pipeline import Pipeline
    roles = ['system','user','cot','assistant']
    sample = sample_df[sample_df.role.isin(roles)]
    baseline = next(p for p in all_probes if p['role_space']==roles and p['layer_ix']==12)
    train,test = cuml.train_test_split(sample.prompt_ix.unique(),test_size=.1,random_state=seed)
    training = sample[sample.prompt_ix.isin(train)]
    testing = sample[sample.prompt_ix.isin(test)]
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    old_limit = pool.get_limit()
    pool.set_limit(size=32*2**30)
    started = time.time()
    try:
        x = cupy.asarray(hidden_states[12][training.sample_ix.tolist()].to(torch.float64).numpy())
        xt = cupy.asarray(hidden_states[12][testing.sample_ix.tolist()].to(torch.float64).numpy())
        y = cupy.asarray(training.role.map({r:i for i,r in enumerate(roles)}).values)
        yt = cupy.asarray(testing.role.map({r:i for i,r in enumerate(roles)}).values)
        classifier = Pipeline([('clf',cuml.linear_model.LogisticRegression(
            penalty='l2',max_iter=5000,linesearch_max_iter=100,fit_intercept=True,C=.005))])
        classifier.fit(x,y)
        score = float(classifier.score(xt,yt))
        nll = float(cuml.metrics.log_loss(yt,classifier.predict_proba(xt)))
        result = {'role_space':'suca','layer_ix':12,'float32_accuracy':float(baseline['acc']),
                  'float64_accuracy':score,'accuracy_difference_percentage_points':100*(score-float(baseline['acc'])),
                  'float32_nll':float(baseline['nll']),'float64_nll':nll,'seconds':time.time()-started,
                  'training_tokens':len(training),'test_tokens':len(testing)}
        with open(OUT/'midlayer-float64-probe.pkl','wb') as f:
            pickle.dump(classifier,f)
        atomic_json(OUT/'midlayer-float64-diagnostic.json',result)
        print('Float64 sensitivity diagnostic:',result,flush=True)
        del x,xt,y,yt,classifier
    finally:
        gc.collect()
        pool.free_all_blocks()
        pool.set_limit(size=old_limit)
    return result
