from types import SimpleNamespace

import pytest

from deepseek_agentdojo_preflight import check_cached_forward, strict_same_shape_and_observe_split, check_all_head_writer


def test_real_tiny_v4_cache_crosses_compression_and_sliding_boundaries():
    import torch
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
    torch.set_num_threads(2)
    torch.manual_seed(19)
    config = DeepseekV4Config(
        vocab_size=64, hidden_size=64, moe_intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, q_lora_rank=64, o_groups=2, o_lora_rank=64,
        n_routed_experts=1, n_shared_experts=1, num_experts_per_tok=1,
        layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
        mlp_layer_types=["hash_moe", "moe", "moe"], index_n_heads=4,
        index_head_dim=16, index_topk=2, compress_rates={"compressed_sparse_attention":4,"heavily_compressed_attention":4},
        sliding_window=8, max_position_embeddings=128, qk_rope_head_dim=8,
        hc_mult=2, hc_sinkhorn_iters=4, use_cache=True)
    config._attn_implementation = "eager"
    model = DeepseekV4ForCausalLM(config).eval().requires_grad_(False)
    adapter = SimpleNamespace(model=model)
    for n in (8, 17, 33):
        result = check_cached_forward(adapter, [i % 63 for i in range(n)],
                                      selected=[0,4,8], tolerance=1e-5, rms_tolerance=1e-5)
        assert result["hook_nohook_bitexact"]
        assert result["cached_full_next_token_equal"]
        assert result["selected_features_shape"] == [1,96]
    with pytest.raises(ValueError, match="at least two"):
        check_cached_forward(adapter, [1], selected=[0])


def test_frozen_capture_and_full_native_bf16_writer_are_observational(tmp_path):
    import numpy as np
    import torch
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
    from deepseek_agentdojo_model import ForwardCapture
    torch.set_num_threads(2)
    config = DeepseekV4Config(vocab_size=64, hidden_size=64, moe_intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=1, head_dim=32,
        q_lora_rank=64, o_groups=2, o_lora_rank=64, n_routed_experts=1, n_shared_experts=1,
        num_experts_per_tok=1, layer_types=['sliding_attention','compressed_sparse_attention','heavily_compressed_attention'],
        mlp_layer_types=['hash_moe','moe','moe'], index_n_heads=4, index_head_dim=16, index_topk=2,
        compress_rates={'compressed_sparse_attention':4,'heavily_compressed_attention':4}, sliding_window=8,
        max_position_embeddings=128, qk_rope_head_dim=8, hc_mult=2, hc_sinkhorn_iters=4, use_cache=True)
    config._attn_implementation='eager'
    model=DeepseekV4ForCausalLM(config).eval().requires_grad_(False).to(torch.bfloat16)
    class Probe:
        roles=('system','user','cot','assistant','tool')
        def predict_proba(self, values):
            return np.full((len(values),5),.2,np.float32)
    adapter=SimpleNamespace(model=model,selected_heads=[8,0,4],probe=Probe())
    ids=list(range(2,10))
    capture=ForwardCapture(model,adapter.selected_heads,num_layers=3,num_heads=4,head_dim=32)
    with torch.inference_mode(),capture:
        model(input_ids=torch.tensor([ids]),use_cache=True,logits_to_keep=1)
    result=strict_same_shape_and_observe_split(adapter,ids,selected=adapter.selected_heads,reference_features=capture.features)
    assert result['same_shape_cache_toggle_bitexact'] and result['frozen_training_vs_generation_head_capture_bitexact']
    assert result['full_vs_split_observation']['is_acceptance_gate'] is False
    writer=check_all_head_writer(adapter,ids,tmp_path)
    assert writer['status']=='passed' and writer['shape']==[2,3,4,32]
    with np.load(tmp_path/'full-head-observation.npz',allow_pickle=False) as values:
        assert values['bf16_uint16'].dtype==np.uint16
        assert values['selected_fp16'].shape==(2,96)
