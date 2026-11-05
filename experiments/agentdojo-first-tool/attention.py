"""Original eager attention, query-tiled without splitting model prefills."""

def require(value, message):
    if not value:
        raise ValueError(message)

def install_tiled_attention(query_tile):
    """Evaluate the unchanged eager attention equation in bounded query tiles.

    The complete sequence still traverses every layer and MoE together. Right
    alignment describes both a full prefill and native growing/sliding caches.
    No dense N-by-N mask is materialized. This has no approximation or truncation.
    """
    import torch
    from transformers.models.gpt_oss import modeling_gpt_oss as implementation
    original = implementation.eager_attention_forward
    def tiled(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        require(not module.training and dropout == 0.0, "Capture requires inference attention")
        keys = implementation.repeat_kv(key, module.num_key_value_groups)
        values = implementation.repeat_kv(value, module.num_key_value_groups)
        batch, heads, q_length, width = query.shape
        k_length = keys.shape[-2]
        result = torch.empty((batch, heads, q_length, width), device=query.device, dtype=values.dtype)
        key_positions = torch.arange(k_length, device=query.device)
        for start in range(0, q_length, query_tile):
            end = min(start + query_tile, q_length)
            logits = torch.matmul(query[:, :, start:end], keys.transpose(2, 3)) * scaling
            if attention_mask is not None:
                mask = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[..., start:end, :]
            else:
                query_positions = torch.arange(start, end, device=query.device) + k_length - q_length
                allowed = key_positions[None, :] <= query_positions[:, None]
                if module.sliding_window is not None:
                    allowed &= key_positions[None, :] > query_positions[:, None] - module.sliding_window
                mask = torch.zeros((end - start, k_length), device=query.device, dtype=query.dtype)
                mask.masked_fill_(~allowed, torch.finfo(query.dtype).min)
            logits = logits + mask
            sinks = module.sinks.reshape(1, -1, 1, 1).expand(batch, -1, end - start, -1)
            combined = torch.cat([logits, sinks], dim=-1)
            combined = combined - combined.max(dim=-1, keepdim=True).values
            probabilities = torch.nn.functional.softmax(combined, dim=-1, dtype=combined.dtype)
            weights = probabilities[..., :-1].to(values.dtype)
            result[:, :, start:end] = torch.matmul(weights, values)
        return result.transpose(1, 2).contiguous(), None
    implementation.eager_attention_forward = tiled
    return original
