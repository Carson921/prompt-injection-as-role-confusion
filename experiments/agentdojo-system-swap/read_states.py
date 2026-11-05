"""Read committed native selected-head states without losing BF16 precision.

Example (from the repository's .venv)::

    python read_states.py OUTPUT banking/user_task_0 0 output --body-only

Python: read_call(output, 'banking/user_task_0', 0, 'output') returns aligned
features, token IDs, original stream positions and body/role masks. By default
all non-delimiter tokens are returned, including ordinary header text. Each
1024-column row contains the sixteen head IDs in manifest order, 64 columns
per head. Inputs repeated in later agent calls retain their actual new states.
"""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def read_call(output, task, call_index, scope, *, body_only=False, head=None):
    if scope not in ("input", "output"):
        raise ValueError("scope must be input or output")
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if task not in {row["task_key"] for row in manifest["initial_requests"]}:
        raise ValueError("Unknown task")
    directory = output / "calls/no_payload" / task / f"call_{call_index:04d}"
    receipt = json.loads((directory / "receipt.json").read_text())
    path = directory / "states.safetensors"
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    if h.hexdigest() != receipt["files"][path.name]:
        raise ValueError("Activation file does not match its committed receipt")
    with safe_open(str(path), framework="pt", device="cpu") as arrays:
        values = {key: arrays.get_tensor(scope + "_" + key) for key in
                  ("features", "token_ids", "positions", "body_mask", "role_ids")}
        heads = arrays.get_tensor("head_ids").tolist()
    if heads != manifest["head_ids"]:
        raise ValueError("Head order differs from the frozen selection")
    if body_only:
        mask = values["body_mask"]
        values = {key: tensor[mask] for key, tensor in values.items()}
    if head is not None:
        slot = heads.index(head)
        values["features"] = values["features"][:, slot * 64:(slot + 1) * 64]
        heads = [head]
    return values | {"head_ids": heads, "role_names": ["system", "developer", "user", "cot", "assistant", "tool"],
                     "scope": scope, "call_id": receipt["call_id"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("task")
    parser.add_argument("call_index", type=int)
    parser.add_argument("scope", choices=["input", "output"])
    parser.add_argument("--body-only", action="store_true")
    parser.add_argument("--head", type=int)
    args = parser.parse_args()
    result = read_call(args.output, args.task, args.call_index, args.scope, body_only=args.body_only, head=args.head)
    print(json.dumps({"call_id": result["call_id"], "scope": result["scope"],
                      "shape": list(result["features"].shape), "dtype": str(result["features"].dtype),
                      "head_ids": result["head_ids"], "body_only": args.body_only,
                      "position_range": [int(result["positions"].min()), int(result["positions"].max())] if len(result["positions"]) else []}))
