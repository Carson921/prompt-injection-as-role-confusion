"""Hot-reloadable operations for ds_probe_server.py (edit freely; the server reloads on change)."""
from __future__ import annotations

import copy
import time

import numpy as np


def _probabilities(state, features):
    return state.probe.predict_proba(features)


def full(state, input_ids, positions):
    """Exact measurement: full no-cache forward (the probe's training protocol)."""
    features = state.backbone.extract_selected(state.model, list(map(int, input_ids)), np.asarray(positions, dtype=np.int64),
                                               selected=state.heads)
    return dict(probabilities=_probabilities(state, features).tolist())


def prefix(state, key, input_ids):
    """Run the constant prompt prefix once and keep its cache (surrogate fast path)."""
    import torch
    if key in state.caches and state.caches[key]['ids'] == list(input_ids):
        return dict(cached=True, tokens=len(input_ids))
    device = state.model.model.embed_tokens.weight.device
    ids = torch.as_tensor([list(map(int, input_ids))], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = state.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True)
    state.caches[key] = dict(ids=list(input_ids), cache=out.past_key_values)
    return dict(cached=False, tokens=len(input_ids))


def chunk(state, key, chunk_ids, positions):
    """Continue from a stored prefix cache with the payload chunk; positions are chunk-relative."""
    import torch
    entry = state.caches[key]
    cache = copy.deepcopy(entry['cache'])
    device = state.model.model.embed_tokens.weight.device
    ids = torch.as_tensor([list(map(int, chunk_ids))], dtype=torch.long, device=device)
    mask = torch.ones((1, len(entry['ids']) + len(chunk_ids)), dtype=torch.long, device=device)
    capture = state.backbone.HeadCapture(np.asarray(positions, dtype=np.int64), selected=state.heads)
    with capture.install(state.model):
        with torch.inference_mode():
            state.model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
    del cache
    return dict(probabilities=_probabilities(state, capture.matrix()).tolist())


def drop(state, keys=None):
    import torch
    for key in list(state.caches) if keys is None else keys:
        state.caches.pop(key, None)
    torch.cuda.empty_cache()
    return dict(caches=len(state.caches))


def memory(state):
    import torch
    return [dict(device=i, allocated_gb=round(torch.cuda.memory_allocated(i) / 2**30, 2),
                 reserved_gb=round(torch.cuda.memory_reserved(i) / 2**30, 2)) for i in range(torch.cuda.device_count())]


def info(state):
    """Identity of the loaded probe (clients check it against ds_common.PROBE_SHA256)."""
    return dict(probe_sha256=state.prober.protocol['probe_sha256'], heads=list(map(int, state.heads)),
                candidate=state.probe.metadata.get('candidate_id'), roles=list(state.probe.roles))


def chunks(state, items):
    """Several cached-prefix continuations in ONE pass over the layers (lockstep).

    Expanding the packed GPTQ experts is a fixed cost per layer and forward, so the token mixers (attention /
    Gated DeltaNet, which need each context's own cache) run per item while the MoE block runs once on the
    concatenated tokens of all items.  items: [{key, chunk_ids, positions}] with chunk-relative positions.
    """
    import torch
    base = state.model.model
    first = base.embed_tokens.weight.device
    heads = np.asarray(state.heads, dtype=np.int64)
    per_layer, full_layers = state.backbone.HEADS, state.backbone.FULL_LAYERS
    wanted = sorted(set(map(int, heads // per_layer)))
    captured = {layer: [] for layer in wanted}
    handles = [base.layers[full_layers[layer]].self_attn.o_proj.register_forward_pre_hook(
        lambda module, args, layer=layer: captured[layer].append(args[0].detach())) for layer in wanted]
    contexts = []
    try:
        with torch.inference_mode():
            for item in items:
                entry = state.caches[item['key']]
                prefix, ids = len(entry['ids']), torch.as_tensor([list(map(int, item['chunk_ids']))], dtype=torch.long, device=first)
                hidden, count = base.embed_tokens(ids), ids.shape[1]
                position = (torch.arange(count, device=first) + prefix).view(1, 1, -1).expand(4, 1, -1)
                mask = torch.full((count, prefix + count), float('-inf'), dtype=hidden.dtype, device=first).triu(prefix + 1)
                contexts.append(dict(hidden=hidden, cache=copy.deepcopy(entry['cache']), rotary=base.rotary_emb(hidden, position[1:]),
                                     mask=mask[None, None], text_position=position[0], count=count))
            for layer in base.layers:
                device = layer.input_layernorm.weight.device
                for context in contexts:
                    hidden = context['hidden'].to(device)
                    mixed = layer.input_layernorm(hidden)
                    if layer.block_type == 'linear_attention':
                        mixed = layer.linear_attn(hidden_states=mixed, cache_params=context['cache'], attention_mask=None)
                    else:
                        mixed, _ = layer.self_attn(hidden_states=mixed, attention_mask=context['mask'].to(device),
                                                   position_ids=context['text_position'].to(device), past_key_values=context['cache'],
                                                   position_embeddings=tuple(t.to(device) for t in context['rotary']))
                    context['hidden'] = hidden + mixed
                joined = torch.cat([context['hidden'] for context in contexts], dim=1)
                joined = joined + layer.mlp(layer.post_attention_layernorm(joined))
                cursor = 0
                for context in contexts:
                    context['hidden'] = joined[:, cursor:cursor + context['count']]
                    cursor += context['count']
    finally:
        for handle in handles:
            handle.remove()
    results = []
    for index, item in enumerate(items):
        positions = torch.as_tensor(list(map(int, item['positions'])), dtype=torch.long)
        features = np.empty((len(positions), len(heads), state.backbone.HEAD_DIM), dtype=np.float16)
        for layer in wanted:
            value = captured[layer][index]
            value = value.reshape(value.shape[1], per_layer, state.backbone.HEAD_DIM).index_select(0, positions.to(value.device))
            columns = np.flatnonzero(heads // per_layer == layer)
            features[:, columns] = value.index_select(1, torch.as_tensor(heads[columns] % per_layer, device=value.device)).to(
                device='cpu', dtype=torch.float16).numpy()
        results.append(dict(probabilities=_probabilities(state, features.reshape(len(positions), -1)).tolist()))
    for context in contexts:
        del context['cache']
    return results


def dump(state, input_ids, positions, path):
    """Diagnostic: all-head pre-o_proj features [positions, 15, 32, 256] (float16) saved to ``path``."""
    import torch
    blocks = {}
    state.backbone.scan_heads(state.model, list(map(int, input_ids)), np.asarray(positions, dtype=np.int64),
                              lambda layer, x: blocks.__setitem__(layer, x.to(device='cpu', dtype=torch.float16).numpy()))
    np.save(path, np.stack([blocks[layer] for layer in sorted(blocks)], axis=1))
    return dict(positions=len(positions))
