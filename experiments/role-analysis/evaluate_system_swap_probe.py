"""Apply a frozen six-role MLP to the user's private system-swap export.

Only numeric arrays and JSON are read; no code supplied in the archive is run.
Raw inputs and per-position predictions stay outside the repository.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
from scipy.special import erf

ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
SEGMENTS = ("channel_rules", "tool_routing", "identity_and_date", "reasoning_setting")
ANCHORS = {"channel_rules": "# Valid channels:", "tool_routing": "Calls to these tools",
           "identity_and_date": "You are ChatGPT", "reasoning_setting": "Reasoning:"}
NEW_SHA = "45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df"
PARENT_SHA = "22cb7251cd96c81d8d1225311b1e3048499850f738b56f4b57040ad7bedf4125"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class Readout:
    def __init__(self, path):
        require(sha(path) == NEW_SHA, "Not the user-selected scratch epoch11 checkpoint")
        with np.load(path, allow_pickle=False) as a:
            self.metadata = json.loads(str(a["metadata"].item()))
            self.mean, self.scale = a["mean"].copy(), a["scale"].copy()
            self.weights = {k[8:]: a[k].copy() for k in a.files if k.startswith("weight::")}
        m = self.metadata
        require(tuple(m["roles"]) == ROLES and m["hidden"] == 128 and m["activation"] == "gelu_exact", "Probe architecture differs")
        require(m["candidate_id"] == "scratch" and m["best_epoch"] == 11, "Probe candidate differs")
        require(self.mean.shape == self.scale.shape == (1024,) and self.mean.dtype == self.scale.dtype == np.float32,
                "Invalid inherited standardizer")
        require(np.isfinite(self.mean).all() and np.isfinite(self.scale).all() and (self.scale > 0).all(), "Invalid standardizer values")
        for k, shape in {"0.weight": (128, 1024), "0.bias": (128,), "3.weight": (6, 128), "3.bias": (6,)}.items():
            require(self.weights[k].shape == shape and self.weights[k].dtype == np.float32 and np.isfinite(self.weights[k]).all(), "Invalid probe parameter")

    def predict(self, x):
        result = []
        for start in range(0, len(x), 4096):
            z = (np.asarray(x[start:start+4096], np.float32) - self.mean) / self.scale
            z = z @ self.weights["0.weight"].T + self.weights["0.bias"]
            z = .5 * z * (1 + erf(z / np.float32(np.sqrt(2))))
            z = z @ self.weights["3.weight"].T + self.weights["3.bias"]
            z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            result.append(p)
        p = np.concatenate(result)
        require(p.shape == (len(x), 6) and p.dtype == np.float32 and np.isfinite(p).all()
                and np.max(abs(p.sum(1)-1)) < 3e-5, "Invalid six-role predictions")
        return p


def summarize(p):
    require(len(p) > 0, "Empty segment")
    counts = np.bincount(p.argmax(1), minlength=6)
    return {"tokens": len(p), "mean_probability": dict(zip(ROLES, p.mean(0, dtype=np.float64).tolist())),
            "argmax_counts": dict(zip(ROLES, map(int, counts))),
            "argmax_fraction": dict(zip(ROLES, (counts/len(p)).tolist())),
            "system_recall": float(counts[0]/len(p))}


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.8), layout="constrained")
    count = result["tasks"]
    for ax, (label, rows) in zip(axes, [(f"Original order: historical matched {count} tasks", result["original_matched_tasks"]),
                                      (f"{result['system_order'].title()} order: first call of {count} tasks", result["scopes"]["initial_calls"]) ]):
        matrix = np.array([[rows[s]["mean_probability"][r] for r in ROLES] for s in SEGMENTS])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(6), ROLES, rotation=30, ha="right")
        ax.set_yticks(range(4), [f"{s}\n(n={rows[s]['tokens']:,})" for s in SEGMENTS])
        ax.set_title(label); ax.set_xlabel("Probe role")
        for i in range(4):
            for j in range(6):
                ax.text(j, i, f"{matrix[i,j]*100:.2f}%", ha="center", va="center",
                        color="white" if matrix[i,j]>.5 else "black", fontsize=9)
    fig.suptitle("Frozen scratch epoch11 | Mean six-role probability for true SYSTEM segments", fontsize=13)
    fig.colorbar(im, ax=list(axes), shrink=.8, label="Mean softmax probability")
    fig.supxlabel("Same task IDs; first system occurrence per task. No training or new model forward.\n"
                  "Historical reference differs in prompt date, hardware and activation storage; this is not a single-variable causal control.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix("."+ext), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "checkpoint", "freeze", "parent", "tokenizer", "baseline", "private-output", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, choices=(16, 37, 57), default=37)
    parser.add_argument("--system-order", choices=("normal", "swapped"), default="swapped")
    args = parser.parse_args(); started = time.time()
    root, output, private = args.inputs.resolve(), args.output.resolve(), args.private_output.resolve()
    repository = Path(__file__).resolve().parents[2]
    require(not root.is_relative_to(repository) and not private.is_relative_to(repository), "Raw arrays must stay private")
    require(not output.exists() and not private.exists(), "Use new output directories")
    require(not output.is_relative_to(root) and not root.is_relative_to(output), "Report overlaps raw inputs")
    package, calls, tasks, messages = [read(root/n) for n in ("manifest.json", "calls.json", "tasks.json", "system_messages.json")]
    require(package["format"] == "agentdojo-system-body-probe-input-v1", "Unsupported system export format")
    n_tasks, n_calls = args.expected_tasks, int(package["calls"])
    expected_suites = {"banking":16, **({"slack":21} if n_tasks >= 37 else {}), **({"travel":20} if n_tasks == 57 else {})}
    require(package["tasks"] == n_tasks and package["suite_tasks"] == expected_suites, "Unexpected suite/task coverage")
    require(package.get("system_order", "swapped") == args.system_order, "Source order differs from requested order")
    probe = Readout(args.checkpoint); frozen = read(args.freeze)
    require(frozen["probe_sha256"] == NEW_SHA and frozen["parent_probe_sha256"] == PARENT_SHA and frozen["eligible"] is False,
            "Frozen candidate identity changed")
    require(package["compatible_probe_sha256"] == sha(args.parent) == PARENT_SHA, "Archive parent identity differs")
    with np.load(args.parent, allow_pickle=False) as parent:
        require(np.array_equal(parent["mean"], probe.mean) and np.array_equal(parent["scale"], probe.scale), "Parent/new normalization differs")
    for key in ("heads", "feature_dim", "feature_position", "model_binding", "selection_sha256"):
        require(package["probe_binding"][key] == probe.metadata[key], "Archive feature binding differs: " + key)
    require(tuple(package["roles"]) == ROLES and package["capture"] == "pre-o_proj", "Raw feature role/interface differs")
    a = {name: np.load(root/(name+".npy"), mmap_mode="r", allow_pickle=False) for name in
         ("X", "y", "positions", "token_ids", "call_index", "task_index", "call_offsets", "task_offsets", "initial_call_mask", "head_ids")}
    x, n = a["X"], int(package["rows"])
    require(n > 0 and package["feature_shape"] == [n, 1024] and x.shape == (n, 1024)
            and x.dtype == np.float32 and np.isfinite(x).all(), "Invalid raw feature rows")
    require(all(a[k].shape == (n,) for k in ("y", "positions", "token_ids", "call_index", "task_index", "initial_call_mask")),
            "Per-position metadata shape differs")
    require(np.array_equal(a["head_ids"], probe.metadata["heads"]) and np.all(a["y"] == 0), "Wrong heads or non-system labels")
    require(a["initial_call_mask"].dtype == bool and len(calls) == n_calls and len(tasks) == n_tasks,
            "Task/call identity differs")
    require(a["call_offsets"].shape == (n_calls+1,) and a["task_offsets"].shape == (n_tasks+1,)
            and a["call_offsets"][0] == a["task_offsets"][0] == 0
            and a["call_offsets"][-1] == a["task_offsets"][-1] == n, "Invalid row offsets")
    with (root/"system_states.safetensors").open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]; require(header_size < 1_000_000, "Invalid safetensors header")
        header = json.loads(stream.read(header_size)); raw = header["features"]
    require(raw["dtype"] == "BF16" and raw["shape"] == [n, 1024], "Safetensors BF16 shape differs")
    storage = np.memmap(root/"system_states.safetensors", dtype="<u2", mode="r", offset=8+header_size+raw["data_offsets"][0], shape=(n,1024))
    require(raw["data_offsets"][1]-raw["data_offsets"][0] == n*1024*2, "Safetensors byte length differs")
    for start in range(0, n, 4096):
        decoded = (np.asarray(storage[start:start+4096], np.uint32) << 16).view(np.float32)
        require(np.array_equal(decoded, x[start:start+4096]), "X is not an exact conversion of captured BF16 features")
    from transformers import AutoTokenizer
    assets = package["probe_binding"]["model_binding"]["assets_sha256"]
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        require(sha(args.tokenizer/name) == assets[name], "Tokenizer bytes differ")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    segment_ids = np.empty(n, dtype=np.int8); template_proof = {}
    for digest, value in messages.items():
        text, transform = value.get("system_body", value.get("swapped_system")), value["transformation"]
        require(hashlib.sha256(text.encode()).hexdigest() == digest, "System body hash differs")
        original, cut = transform["original_system"], transform["split_char_in_system"]
        expected_text = original[cut:]+original[:cut] if args.system_order == "swapped" else original
        require(text == expected_text == transform["swapped_system"], "Source text is not the declared order transformation")
        expected_order = SEGMENTS if args.system_order == "swapped" else ("identity_and_date", "reasoning_setting", "channel_rules", "tool_routing")
        anchors = sorted((text.index(ANCHORS[s]), s) for s in SEGMENTS)
        require(all(text.count(ANCHORS[s]) == 1 for s in SEGMENTS) and tuple(s for _,s in anchors) == expected_order, "Unexpected section order")
        anchors[0] = (0, anchors[0][1])  # Leading whitespace belongs to the first textual section.
        prefix = "<|start|>system<|message|>"
        require(len(prefix) == transform["system_char_start"], "System header character coordinates differ")
        encoded = tok(prefix+text+"<|end|>", add_special_tokens=False, return_offsets_mapping=True)
        positions = np.array([i for i,(b,e) in enumerate(encoded["offset_mapping"]) if max(b,len(prefix)) < min(e,len(prefix)+len(text))], np.int64)
        ids = np.array(encoded["input_ids"], np.int64)[positions]
        offsets = np.asarray(encoded["offset_mapping"], np.int64)[positions]-len(prefix)
        mids = offsets.mean(1); boundaries = np.array([b for b,_ in anchors])
        groups = np.asarray([SEGMENTS.index(s) for _,s in anchors], dtype=np.int8)[np.searchsorted(boundaries, mids, side="right")-1]
        spans = [{"name":s,"char_start":b,"char_end":anchors[i+1][0] if i+1<len(anchors) else len(text),
                  "tokens_per_call":int(np.sum(groups==SEGMENTS.index(s)))} for i,(b,s) in enumerate(anchors)]
        template_proof[digest] = {"body_characters":len(text),"body_tokens":len(ids),"spans":spans,
                                  "boundary_crossing_tokens":int(sum(any(b<c<e for c in boundaries[1:]) for b,e in offsets))}
        for call in calls:
            if call["system_text_sha256"] != digest: continue
            i = call["index"]; lo,hi = int(a["call_offsets"][i]),int(a["call_offsets"][i+1]);t=call["task_index"]
            require(lo == call["row_start"] and hi == call["row_end"] and hi-lo == len(ids), "Call/body row count differs")
            require(np.all(a["call_index"][lo:hi] == i) and np.all(a["task_index"][lo:hi] == t), "Call/task row mapping differs")
            require(np.array_equal(a["positions"][lo:hi], positions) and np.array_equal(a["token_ids"][lo:hi],ids), "Full-input token coordinates differ")
            require(np.all(a["initial_call_mask"][lo:hi] == (call["call_number"] == 0)), "Initial call mask differs")
            segment_ids[lo:hi] = groups
    require(set(c["system_text_sha256"] for c in calls) == set(messages), "Unknown system template")
    for i,t in enumerate(tasks):
        lo,hi = map(int,a["task_offsets"][i:i+2]); grouped=[c for c in calls if c["task_index"]==i]
        require(t["index"] == i and t["row_start"] == lo and t["row_end"] == hi
                and t["calls"] == len(grouped) and np.all(a["task_index"][lo:hi] == i), "Task offsets/calls differ")
        require(sum(c["call_number"]==0 for c in grouped)==1, "Task initial call is not unique")
    require(Counter(t["task_key"].split('/')[0] for t in tasks) == expected_suites, "Unexpected task suites")
    initial = np.asarray(a["initial_call_mask"])
    require(initial.sum() == package["initial_call_rows"] and len(np.unique(a["task_index"][initial])) == n_tasks,
            "Initial system denominator differs")
    p = probe.predict(x); fp16_x = np.asarray(x).astype(np.float16).astype(np.float32)
    require(np.isfinite(fp16_x).all(), "Features exceed old FP16 storage range")
    q = probe.predict(fp16_x)
    sensitivity = {"raw_fp32_is_exact_bf16_conversion":True,"fp16_storage_feature_max_absolute_delta":float(np.max(abs(x-fp16_x))),
                   "fp16_storage_probability_max_absolute_delta":float(np.max(abs(p-q))),
                   "fp16_storage_argmax_disagreements":int(np.sum(p.argmax(1)!=q.argmax(1))),
                   "primary_readout":"Raw X float32, inherited FIT mean/scale, no extra rounding or standardizer fitting"}
    scopes = {scope:{s:summarize(p[mask & (segment_ids==i)]) for i,s in enumerate(SEGMENTS)}
              for scope,mask in [("initial_calls",initial),("all_calls",np.ones(n,dtype=bool))]}
    require(sum(z["tokens"] for z in scopes["initial_calls"].values()) == initial.sum()
            and sum(z["tokens"] for z in scopes["all_calls"].values()) == n, "Segments do not cover all system positions")
    baseline_manifest = read(args.baseline/"result-manifest.json")
    require(baseline_manifest["probe_sha256"] == NEW_SHA, "Historical comparison used another checkpoint")
    old_path = args.baseline/"first-position-probabilities.npz"
    expected = next(e["sha256"] for e in baseline_manifest["files"] if e["path"] == old_path.name)
    require(sha(old_path) == expected, "Historical probability packet changed")
    task_keys = {t["task_key"] for t in tasks}; old_rows = {}
    require(len(task_keys) == n_tasks, "Repeated task key")
    with np.load(old_path, allow_pickle=False) as old:
        keep = np.isin(old["task"],list(task_keys)) & (old["scope"]=="actual_input") & (old["tag"]=="system")
        require(set(old["task"][keep]) == task_keys and keep.sum() == 71*n_tasks,
                "Historical matched task/position count differs")
        for s in SEGMENTS:old_rows[s]=summarize(old["new"][keep & (old["segment"]=="system/"+s)])
    result = {"schema_version":"system-order-readout-v3","system_order":args.system_order,"status":"complete","checkpoint_sha256":NEW_SHA,"checkpoint":"scratch epoch11; DEV ineligible; frozen",
              "tasks":n_tasks,"suite_tasks":expected_suites,"calls":n_calls,"rows":n,"initial_call_rows":int(initial.sum()),
              "segment_order":list(SEGMENTS),"aggregation":"Tokenwise softmax probabilities averaged in float64; argmax rates reported separately",
              "scopes":scopes,"original_matched_tasks":old_rows,"template_alignment":template_proof,
              "overall_initial_calls":summarize(p[initial]),"overall_all_calls":summarize(p),
              "storage_precision_sensitivity":sensitivity,"new_backbone_forward":False,"training_performed":False,
              "input_archive_proof":read(root.parent/"input-verification.json"),"historical_reference_sha256":sha(old_path),
              "limitations":[f"Primary is one initial call per task; all{n_calls} repeated-call aggregate is separate.",
                             f"One repeated system template;{n_tasks}tasks are not{n_tasks} independent system templates.",
                             "Historical baseline uses2026-10-21 and4A40/FP16 cache; new run uses2026-11-02,H100,BF16 states.",
                             "Source text/whitespace is used exactly; normal order preserves original text, swapped order uses AB-to-BA without inserting a newline.",
                             "Different date/hardware/section-boundary tokenization prevent attributing differences solely to position."],
              "implementation_sha256":sha(__file__),"elapsed_seconds":time.time()-started}
    output.mkdir(parents=True); private.mkdir(parents=True)
    np.save(private/"probabilities.npy",p,allow_pickle=False);np.save(private/"segment_ids.npy",segment_ids,allow_pickle=False)
    write(private/"alignment.json",template_proof)
    write(output/"metrics.json",result)
    rows=[]
    for scope, data in [(args.system_order+"_initial_calls",scopes["initial_calls"]),(args.system_order+"_all_calls",scopes["all_calls"]),(f"original_matched{n_tasks}",old_rows)]:
        for s in SEGMENTS:
            z=data[s];rows.append({"scope":scope,"segment":s,"tokens":z["tokens"],**{"p_"+r:z["mean_probability"][r] for r in ROLES},
                                 **{"argmax_"+r:z["argmax_fraction"][r] for r in ROLES}})
    csv_write(output/"segment-probabilities.csv",rows)
    per_task=[]
    for t in tasks:
        mask=initial & (a["task_index"]==t["index"])
        for i,s in enumerate(SEGMENTS):
            z=summarize(p[mask & (segment_ids==i)])
            per_task.append({"task":t["task_key"],"segment":s,"tokens":z["tokens"],**{"p_"+r:z["mean_probability"][r] for r in ROLES}})
    csv_write(output/"per-task-initial-probabilities.csv",per_task)
    plot(result,output/("system-normal-probabilities" if args.system_order == "normal" else "system-swap-probabilities"))
    write(output/"run-validation.json",{"status":"passed","binding_matches_parent_and_new":True,"normalization_inherited":True,
          "archive_code_executed":False,"all_exported_full_input_ids_positions_and_BF16_states_verified":True,
          "verified_rows":n,"verified_calls":n_calls,"verified_tasks":n_tasks,
          "no_training_or_new_LM_forward":True,"frozen_checkpoint_sha256":NEW_SHA,
          "private_probabilities_sha256":sha(private/"probabilities.npy"),"private_segments_sha256":sha(private/"segment_ids.npy"),
          "source_SHA256SUMS_sha256":sha(root/"SHA256SUMS"),"source_manifest_sha256":sha(root/"manifest.json")})
    print(json.dumps({"status":"complete","initial_call_probabilities":scopes["initial_calls"],"storage_sensitivity":sensitivity,
                      "output":str(output),"seconds":time.time()-started}),flush=True)


if __name__=="__main__":
    main()
