"""CPU checks for mask bounds, FIT partitions, head order, and durable caches."""
import json
from pathlib import Path
import tempfile

import numpy as np

import synthetic_agent_gpu as gpu


def check_numpy():
    with tempfile.TemporaryDirectory(prefix="six-role-gpu-check-") as temporary:
        root = Path(temporary)
        inputs = root / "fit.jsonl"
        records = []
        for source in range(40):
            record = {"id": str(source), "source_id": str(source), "cluster_id": str(source),
                      "split": "fit", "input_ids": list(range(12)), "targets": []}
            for role in range(6):
                record["targets"].append({"token_ix": role + 3, "label": role,
                                          "role": gpu.ROLES[role], "family_id": f"{source}:{role}", "weight": .5})
            records.append(record)
        inputs.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        plan = gpu._scan_plan(inputs, 60, 30)
        assert plan == gpu._scan_plan(inputs, 60, 30)
        assert not set(plan["clusters"]["statistics"]) & set(plan["clusters"]["inner"])
        assert len(set(plan["clusters"]["statistics"]) | set(plan["clusters"]["inner"])) == 30
        for part, chosen in plan["records"].items():
            for record_ix, target_ixs in chosen.items():
                record = records[int(record_ix)]
                assert gpu._source_split(record["cluster_id"]) == part
                assert all(0 <= ix < len(record["targets"]) for ix in target_ixs)
        bad = dict(records[0], targets=[dict(records[0]["targets"][0], token_ix=12)])
        try:
            gpu._checked_record(bad)
            raise AssertionError("Out-of-range body target was accepted")
        except RuntimeError:
            pass
        y = np.arange(6)
        x = np.arange(6 * 1024, dtype=np.float16).reshape(6, 1024)
        arrays = gpu._row_arrays({**records[0], "profile": "single", "template_id": "six", "round_segments": []}, x)
        buffers = {key: [value] for key, value in arrays.items()}
        signature = {"data": "fixed", "heads": list(range(16))}
        entry = gpu._save_shard(root, "fit", 0, buffers, 0, 1, signature)
        meta = gpu._verify_shard(root, entry, signature)
        assert meta["record_end"] == 1 and entry["rows"] == 6
        actual = np.load(root / entry["path"] / "x.npy", mmap_mode="r", allow_pickle=False)
        assert np.array_equal(actual, x)
        with (root / entry["path"] / "x.npy").open("ab") as stream:
            stream.write(b"corruption")
        try:
            gpu._verify_shard(root, entry, signature)
            raise AssertionError("Modified durable feature shard was accepted")
        except RuntimeError:
            pass
    # A known sufficient-statistics problem checks class/feature axes and solve.
    means = np.zeros((6, 64))
    means[np.arange(6), np.arange(6)] = 8
    n = np.full((36, 6), 100., dtype=np.float64)
    total = np.broadcast_to(means[None, None] * 100, (36, 64, 6, 64)).copy()
    xx_one = 600 * np.eye(64) + 100 * means.T @ means
    xx = np.broadcast_to(xx_one[None, None], (36, 64, 64, 64)).copy()
    weight, bias = gpu.fit_head_lda({"n": n, "sum": total, "xx": xx})
    pred = (np.einsum("cd,hkd->hck", means, weight) + bias[:, None]).argmax(-1)
    assert np.array_equal(pred, np.broadcast_to(np.arange(6), pred.shape))
    confusion = np.broadcast_to(np.eye(6)[None], (2304, 6, 6)).copy()
    macro, weakest = gpu._head_scores(confusion)
    assert np.all(macro == 1) and np.all(weakest == 1)
    print(json.dumps({"numpy_checks": "passed", "checks": [
        "deterministic FIT-only cluster sampling", "source-disjoint head evaluation", "body bounds",
        "FP16 memmap round trip", "durable shard corruption rejection", "six-class LDA axes", "six-class metrics"]}))


def check_torch():
    import torch
    from types import SimpleNamespace

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = torch.nn.Identity()

    layers = [SimpleNamespace(self_attn=Attention()) for _ in range(36)]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    values = torch.arange(3 * 4096, dtype=torch.float32).reshape(1, 3, 4096) / 16
    selected = np.asarray([2248, 0, 64, 35, 458, 6, 7, 9, 10, 20, 30, 40, 50, 60, 70, 80])
    capture = gpu.HeadCapture([2, 0], selected=selected)
    before = values.clone()
    with capture.install(model):
        for layer in capture.layers:
            result = layers[layer].self_attn.o_proj(values + layer)
            assert torch.equal(result, values + layer)
    matrix = capture.matrix().reshape(2, 16, 64)
    for j, head in enumerate(selected):
        expected = (values + int(head // 64)).reshape(3, 64, 64)[[2, 0], int(head % 64)].half().numpy()
        assert np.array_equal(matrix[:, j], expected)
    assert torch.equal(values, before)
    # Weighted covariance hooks agree with a direct CPU sufficient-statistic sum.
    moment = gpu.OnlineMoments()
    x = torch.arange(4 * 64 * 64, dtype=torch.float32).reshape(4, 64, 64) / 100
    labels, weights = np.asarray([0, 1, 0, 1]), np.asarray([.2, .3, .4, .5])
    for layer in range(36):
        moment.update(layer, x, labels, weights)
    arrays = moment.export()
    expected = np.einsum("nhd,nhe,n->hde", x.double().numpy(), x.double().numpy(), weights)
    assert np.allclose(arrays["xx"][0], expected, rtol=1e-12, atol=1e-9)
    restored = gpu.OnlineMoments(arrays).export()
    assert all(np.array_equal(arrays[key], restored[key]) for key in arrays)
    print(json.dumps({"torch_cpu_checks": "passed", "checks": [
        "noncontiguous unsorted head concatenation", "no hook mutation", "weighted online covariance", "moment checkpoint round trip"]}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", action="store_true")
    args = parser.parse_args()
    check_numpy()
    if args.torch:
        check_torch()
