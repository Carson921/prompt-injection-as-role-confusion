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
    """Several cached-prefix continuations: [{key, chunk_ids, positions}] -> one result per item.
    (The Qwen port ran these in lockstep to expand packed experts once; Llama 4 is dense, so they simply run in turn.)"""
    return [chunk(state, item['key'], item['chunk_ids'], item['positions']) for item in items]


def dump(state, input_ids, positions, path):
    """Diagnostic: all-head pre-o_proj features [positions, 48, 40, 128] (float16) saved to ``path``."""
    import torch
    blocks = {}
    state.backbone.scan_heads(state.model, list(map(int, input_ids)), np.asarray(positions, dtype=np.int64),
                              lambda layer, x: blocks.__setitem__(layer, x.to(device='cpu', dtype=torch.float16).numpy()))
    np.save(path, np.stack([blocks[layer] for layer in sorted(blocks)], axis=1))
    return dict(positions=len(positions))
