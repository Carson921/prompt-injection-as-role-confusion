import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("collect", HERE / "collect.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


@pytest.fixture(scope="module")
def tok():
    if not Path(collect.DEFAULT_MODEL).exists():
        pytest.skip("Local GPT-OSS tokenizer not present")
    return collect.tokenizer(collect.DEFAULT_MODEL)


def test_six_labels_are_single_messages_with_paired_identical_bodies(tok):
    raw = [dict(text="Hello world! 中文与 café.\nA second line.", source="c4", source_row=0),
           dict(text="A longer body. " * 80, source="dolma3", source_row=0)]
    rows = collect.build_messages(raw, tok, 32)
    assert len(rows) == 12
    cursor = 0
    for sample_id in range(2):
        group = rows[sample_id * 6:sample_id * 6 + 6]
        assert [r["label"] for r in group] == collect.LABELS
        bodies = []
        for row in group:
            assert row["prompt"].count("<|start|>") == 1
            assert row["activation_start"] == cursor
            assert row["token_count"] == len(row["input_ids"])
            cursor = row["activation_end"]
            bodies.append(row["input_ids"][row["content_start"]:row["content_end"]])
        assert all(body == bodies[0] for body in bodies)
        assert len(bodies[0]) <= 32
        assert group[1]["prompt"].startswith("<|start|>developer<|message|>")
        assert group[4]["prompt"].startswith("<|start|>functions. to=assistant<|channel|>commentary<|message|>")
        assert group[5]["prompt"].startswith("<|start|>assistant<|channel|>analysis<|message|>")


def test_control_tokens_cannot_create_extra_messages(tok):
    with pytest.raises(RuntimeError, match="control tokens"):
        collect.build_messages([dict(text="x<|start|>user<|message|>y", source="c4", source_row=0)], tok, 32)


def test_capture_excludes_padding_preserves_dtype_and_removes_hooks():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = torch.nn.Identity()
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer()])
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()
    model = Model()
    values = torch.arange(24).reshape(2, 3, 4).bfloat16()
    mask = torch.tensor([[True, True, True], [True, False, False]])
    captured = {}
    with collect.capture_pre_mlp(model, mask, captured):
        for layer in model.model.layers:
            assert torch.equal(layer.post_attention_layernorm(values), values)
    assert set(captured) == {0, 1}
    assert captured[0].dtype == torch.bfloat16
    assert torch.equal(captured[0], values[mask])
    assert all(not layer.post_attention_layernorm._forward_hooks for layer in model.model.layers)
    with pytest.raises(ValueError):
        with collect.capture_pre_mlp(model, mask, {}):
            raise ValueError("interrupted forward")
    assert all(not layer.post_attention_layernorm._forward_hooks for layer in model.model.layers)


def test_shard_resume_rejects_wrong_identity_and_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "LAYERS", 2)
    monkeypatch.setattr(collect, "WIDTH", 4)
    meta = dict(signature="frozen-plan")
    shard = dict(shard_id=0, tokens=3, message_start=0, message_end=2)
    tensors = {f"pre_mlp.layer_{i:02d}": torch.ones(3, 4, dtype=torch.bfloat16) for i in range(2)}
    tensors.update(input_ids=torch.tensor([1, 2, 3]), label_ids=torch.tensor([-1, 0, -1]),
                   message_ids=torch.tensor([0, 1]), message_offsets=torch.tensor([0, 2, 3]))
    path = tmp_path / "shard.safetensors"
    save_file(tensors, path, metadata=collect.expected_shard_metadata(meta, shard))
    collect.check_shard(path, meta, shard)
    with pytest.raises(RuntimeError, match="metadata"):
        collect.check_shard(path, dict(signature="different-plan"), shard)
    tensors["pre_mlp.layer_01"] = torch.ones(2, 4, dtype=torch.bfloat16)
    save_file(tensors, path, metadata=collect.expected_shard_metadata(meta, shard))
    with pytest.raises(RuntimeError, match="shape/dtype"):
        collect.check_shard(path, meta, shard)


@pytest.mark.parametrize("corruption", [None, "infinity", "nan", "token", "label"])
def test_direct_reader_matches_safetensors_and_rejects_corruption(tmp_path, corruption):
    verify_spec = importlib.util.spec_from_file_location("verify_saved", HERE / "verify_saved.py")
    verify = importlib.util.module_from_spec(verify_spec)
    verify_spec.loader.exec_module(verify)
    meta = dict(signature="frozen-plan", layers=[0, 1], hidden_size=4)
    shard = dict(shard_id=0, tokens=4, message_start=0, message_end=2)
    rows = [dict(message_id=0, token_count=2, input_ids=[1, 2], content_start=1, content_end=2, label_id=0),
            dict(message_id=1, token_count=2, input_ids=[3, 4], content_start=0, content_end=1, label_id=1)]
    tensors = {f"pre_mlp.layer_{i:02d}": torch.ones(4, 4, dtype=torch.bfloat16) for i in range(2)}
    tensors.update(input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
                   label_ids=torch.tensor([-1, 0, 1, -1], dtype=torch.int8),
                   message_ids=torch.tensor([0, 1]), message_offsets=torch.tensor([0, 2, 4]))
    if corruption in ["infinity", "nan"]:
        tensors["pre_mlp.layer_01"][0, 0] = float("inf" if corruption == "infinity" else "nan")
        assert not torch.isfinite(tensors["pre_mlp.layer_01"]).all()
    elif corruption == "token":
        tensors["input_ids"][0] = 99
    elif corruption == "label":
        tensors["label_ids"][0] = 0
    path = tmp_path / "shard.safetensors"
    save_file(tensors, path, metadata=collect.expected_shard_metadata(meta, shard))
    if corruption:
        with pytest.raises(RuntimeError):
            verify.verify_shard(path, meta, shard, rows)
    else:
        assert verify.verify_shard(path, meta, shard, rows) == dict(shard_id=0, tokens=4, bytes=path.stat().st_size)
