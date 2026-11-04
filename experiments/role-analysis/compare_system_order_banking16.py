"""Matched initial-request comparison of normal/swapped banking16 readouts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from evaluate_system_swap_probe import ROLES, SEGMENTS, summarize, csv_write, write, require, sha, read


def capture_sources(archive):
    result = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isfile() and "/capture_code/" in member.name:
                result[Path(member.name).name] = tar.extractfile(member).read().decode()
                if len(result) == 3:
                    break
    require(set(result) == {"collect.py", "run_fresh.py", "agentdojo_bridge.py"}, "Missing capture sources")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("normal-private", "swapped-private", "normal-result", "swapped-result", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    require(not (output/"matched-banking16-comparison.json").exists(), "Comparison already committed")
    roots = {"normal":args.normal_private.resolve(), "swapped":args.swapped_private.resolve()}
    reports = {"normal":args.normal_result.resolve(), "swapped":args.swapped_result.resolve()}
    metadata = {k:read(p/"inputs/provenance/manifest.json") for k,p in roots.items()}
    backends = {k:read(p/"inputs/provenance/backend.json") for k,p in roots.items()}
    require(backends["normal"] == backends["swapped"], "Backend settings differ")
    for key in ("model_identity", "head_ids", "selection_sha256", "generation", "run_date", "tools", "agentdojo_source_sha256"):
        require(metadata["normal"][key] == metadata["swapped"][key], "Paired control differs: "+key)
    source_messages = {k:next(iter(read(p/"inputs/system_messages.json").values())) for k,p in roots.items()}
    normal_text = source_messages["normal"]["system_body"]
    transform = source_messages["swapped"]["transformation"]
    require(normal_text == transform["original_system"], "Original system text differs")
    cut = transform["split_char_in_system"]
    require(source_messages["swapped"]["system_body"] == normal_text[cut:]+normal_text[:cut], "Not exactly the system rotation")
    inventories = {k:read(p/"bundle-inventory.json") for k,p in roots.items()}
    request_matches=[]
    for i in range(16):
        name=f"collection/calls/no_payload/banking/user_task_{i}/call_0000/original_request.json"
        require(inventories["normal"][name]["sha256"] == inventories["swapped"][name]["sha256"], "Original first request differs")
        request_matches.append({"task":f"banking/user_task_{i}","sha256":inventories["normal"][name]["sha256"]})
    sources={k:capture_sources(Path(read(p/"input-verification.json")["archive_path"])) for k,p in roots.items()}
    ast_review={}
    for name in sources["normal"]:
        for k in roots:
            require(hashlib.sha256(sources[k][name].encode()).hexdigest() == metadata[k]["code_sha256"][name], "Captured source SHA differs")
        nodes={k:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(sources[k][name]).body
                  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))} for k in roots}
        require(set(nodes["normal"]) == set(nodes["swapped"]), "Capture function names changed")
        changed=[n for n in nodes["normal"] if nodes["normal"][n] != nodes["swapped"][n]]
        if name=="collect.py":require(not changed,"Numerical capture helper changed")
        if name=="run_fresh.py":require(set(changed)<={"render","prepare","run","verify","main"},"Numerical forward/decode implementation changed")
        ast_review[name]={"unchanged_function_class_nodes":[n for n in nodes["normal"] if n not in changed],
                          "changed_nodes":changed,"archive_code_executed":False}
    stats={};csv_rows=[];selected_calls={};expected_tasks={f"banking/user_task_{i}" for i in range(16)}
    for order,p in roots.items():
        raw=p/"inputs";tasks=read(raw/"tasks.json");calls=read(raw/"calls.json")
        selected_ids={t["index"] for t in tasks if t["task_key"] in expected_tasks}
        require({t["task_key"] for t in tasks if t["index"] in selected_ids}==expected_tasks,"Banking task IDs differ")
        probs=np.load(p/"readout/probabilities.npy",allow_pickle=False)
        require(sha(p/"readout/probabilities.npy")==read(reports[order]/"run-validation.json")["private_probabilities_sha256"],"Sealed readout changed")
        groups=np.load(p/"readout/segment_ids.npy",allow_pickle=False);task_index=np.load(raw/"task_index.npy",allow_pickle=False)
        cohort=np.isin(task_index,list(selected_ids));initial=np.load(raw/"initial_call_mask.npy",allow_pickle=False)
        stats[order]={};selected_calls[order]={}
        for scope,mask in [("initial_calls",cohort&initial),("all_calls",cohort)]:
            direct=np.zeros(len(probs),bool);count=0
            for call in calls:
                if call["task_index"] in selected_ids and (scope=="all_calls" or call["call_number"]==0):
                    direct[call["row_start"]:call["row_end"]]=True;count+=1
            require(np.array_equal(mask,direct),"Independent call-ID mask differs from task-index selection")
            if scope=="initial_calls":require(count==16 and mask.sum()==1136,"Initial banking denominator differs")
            stats[order][scope]={s:summarize(probs[mask&(groups==i)]) for i,s in enumerate(SEGMENTS)}
            selected_calls[order][scope]=count
            for s,z in stats[order][scope].items():
                csv_rows.append({"scope":scope,"order":order,"segment":s,"tasks":16,"calls":count,"tokens":z["tokens"],
                                 **{"p_"+r:z["mean_probability"][r] for r in ROLES},
                                 **{"argmax_"+r:z["argmax_fraction"][r] for r in ROLES}})
    proof={"status":"passed","task_count":16,"initial_request_hash_matches":request_matches,"same_original_system_text":True,
           "same_model_weights_and_tokenizer":True,"same_backend_config":True,"same_run_date":metadata["normal"]["run_date"],
           "same_generation_settings":metadata["normal"]["generation"],"capture_source_AST_review":ast_review,
           "scope":"Same banking16 initial requests. Rotation jointly changes token position, preceding context and whitespace boundaries; not an isolated absolute-position intervention."}
    result={"status":"complete","checkpoint_sha256":read(reports["normal"]/"metrics.json")["checkpoint_sha256"],
            "tasks":16,"matched_task_keys":sorted(expected_tasks),"selected_calls":selected_calls,"stats":stats,
            "controls":proof,"implementation_sha256":sha(__file__)}
    require(result["checkpoint_sha256"] == read(reports["swapped"]/"metrics.json")["checkpoint_sha256"],"Checkpoints differ")
    write(output/"matched-banking16-comparison.json",result);write(output/"paired-control-validation.json",proof)
    csv_write(output/"matched-banking16-comparison.csv",csv_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15.2,5.8),layout="constrained")
    for ax,order in zip(axes,("normal","swapped")):
        rows=stats[order]["initial_calls"];m=np.array([[rows[s]["mean_probability"][r] for r in ROLES] for s in SEGMENTS])
        im=ax.imshow(m,cmap="Blues",vmin=0,vmax=1,aspect="auto");ax.set_title(order.title()+" system order | matched banking16")
        ax.set_xticks(range(6),ROLES,rotation=30,ha="right");ax.set_yticks(range(4),[f"{s}\n(n={rows[s]['tokens']})" for s in SEGMENTS]);ax.set_xlabel("Probe role")
        for i in range(4):
            for j in range(6):ax.text(j,i,f"{100*m[i,j]:.2f}%",ha="center",va="center",fontsize=9,color="white" if m[i,j]>.5 else "black")
    fig.suptitle("Frozen scratch epoch11 | Mean six-role probability for true SYSTEM segments",fontsize=13)
    fig.colorbar(im,ax=list(axes),shrink=.8,label="Mean softmax probability")
    fig.supxlabel("16 byte-identical original first requests; same date, model/tokenizer and H100 backend settings. One initial call per task.\n"
                  "AB-to-BA changes both position and prefix context; whitespace preserved. Not a pure absolute-position intervention.",fontsize=9)
    for ext in ("png","pdf"):fig.savefig(output/("banking16-normal-vs-swapped."+ext),dpi=180)
    plt.close(fig)
    print(json.dumps({"status":"complete","selected_calls":selected_calls,"normal_initial":stats["normal"]["initial_calls"],"swapped_initial":stats["swapped"]["initial_calls"]}),flush=True)


if __name__=="__main__":main()
