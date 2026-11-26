"""Prepare exact original CoT-forgery payloads at native AgentDojo tool exposures.

No model inference or tools are executed. Every benign input token is preserved;
only newly appended payload text is tokenized. Positions are zero-based and all
intervals are half open. Native CoT references retain the original raw generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/role-analysis"))
from agentdojo_trace_interventions_data import parse_native_blocks, require


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def prefix(trace_id):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "__", trace_id)
    return label + "-" + hashlib.sha256(trace_id.encode()).hexdigest()[:10]


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    require(not (args.output / "inputs.jsonl").exists(), "Refusing to overwrite immutable prepared inputs")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    require(tokenizer.is_fast, "Payload paragraph alignment requires tokenizer offsets")
    traces = [json.loads(line) for line in args.traces.read_text().splitlines() if line]
    require(len(traces) == 57 and Counter(x["source_split"] for x in traces) ==
            dict(fit=40, dev=8, test=9), "Original 57-task split inventory differs")
    config = yaml.safe_load(args.payloads.read_text())
    base = {x["type"]: x["prompt"] for x in config["base_injections"]}
    forged = {x["type"]: x["prompt"] for x in config["prompt_injections"]}
    require(len(base) == len(forged) == 5 and set(base) == set(forged), "Original five payload pairs differ")
    payload_records = []
    for payload_type in base:
        for condition, text in (("base-injection", base[payload_type]),
                                ("cot-forgery-injection", forged[payload_type])):
            require(text == text.strip(), "YAML payload has unexpected exterior whitespace")
            require(("\n\n" in text) == (condition == "cot-forgery-injection"),
                    "Unexpected original payload paragraph structure")
            full = "\n\n" + text
            encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
            require(tokenizer.decode(encoded["input_ids"], skip_special_tokens=False) == full,
                    "Payload tokenization does not round-trip exactly")
            split = full.find("\n\n", 2) if condition == "cot-forgery-injection" else len(full)
            direct_chars = (2, split)
            cot_chars = (split + 2, len(full)) if condition == "cot-forgery-injection" else (len(full), len(full))
            def overlaps(span):
                return [i for i, (lo, hi) in enumerate(encoded["offset_mapping"])
                        if hi > span[0] and lo < span[1]]
            payload_records.append(dict(condition=condition, payload_type=payload_type,
                                        payload_text=text, appended_text=full,
                                        payload_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                        ids=encoded["input_ids"], direct_relative=overlaps(direct_chars),
                                        cot_relative=overlaps(cot_chars)))
    records, sources = [], []
    for trace in sorted(traces, key=lambda row: row["trace_id"]):
        trace_id = trace["trace_id"]
        lineage = trace["provenance"]["first_role_training_lineage"]
        tool_lineage = lineage["tool"]
        require(tool_lineage["scope"] == "actual_input", "Tool exposure is not a live archived input")
        tool_dir = args.source_root / "calls" / tool_lineage["call_id"]
        tool_request = read(tool_dir / "request.json")
        require(tool_request["status"] == "complete", "Archived tool exposure request is incomplete")
        with np.load(tool_dir / "input.npz", allow_pickle=False) as source:
            original = source["input_ids"].tolist()
        require(len(original) == tool_request["input_tokens"], "Original live input length differs")
        require(tokenizer.decode(original, skip_special_tokens=False) == tool_request["rendered_prompt"],
                "Original live input IDs differ from native request")
        blocks = parse_native_blocks(original, tokenizer)
        tool = next(block for block in blocks if block["role"] == "tool")
        require(list(range(tool["body_start"], tool["body_end"])) == tool_lineage["source_token_indices"],
                "First live tool body does not match checkpoint-retained lineage")
        require(ids_sha(tool["body_ids"]) == tool_lineage["body_token_ids_sha256"], "Tool body hash differs")
        require(not blocks[-1]["complete"] and blocks[-1]["header_text"] == "assistant",
                "Native final open assistant header is missing")
        provenance = dict(source_call_id=tool_lineage["call_id"], source_scope="actual_input",
                          source_input_npz=str((tool_dir / "input.npz").resolve()),
                          source_input_npz_sha256=sha(tool_dir / "input.npz"),
                          source_request_json=str((tool_dir / "request.json").resolve()),
                          source_request_json_sha256=sha(tool_dir / "request.json"),
                          original_input_ids_sha256=ids_sha(original), original_input_tokens=len(original),
                          original_tool_body_start=tool["body_start"], original_tool_body_end=tool["body_end"],
                          original_tool_body_token_ids_sha256=ids_sha(tool["body_ids"]),
                          first_tool_training_lineage=tool_lineage,
                          trailing_open_header_ids=blocks[-1]["header_ids"],
                          original_bodies_retokenized=False, native_outer_headers_changed=False)
        common = dict(trace_id=trace_id, source_split=trace["source_split"],
                      suite=trace["source_suite"], source_suite=trace["source_suite"],
                      primary_heldout=trace["source_split"] == "test", context_kind="original_live_first_tool_exposure")

        def tool_row(condition, payload=None):
            inserted = payload["ids"] if payload else []
            insertion = tool["body_end"]
            ids = original[:insertion] + inserted + original[insertion:]
            original_positions = list(range(tool["body_start"], tool["body_end"]))
            segments = dict(original_tool=original_positions,
                            direct_instruction=[insertion+i for i in payload["direct_relative"]] if payload else [],
                            forged_cot=[insertion+i for i in payload["cot_relative"]] if payload else [],
                            payload_all=list(range(insertion, insertion+len(inserted))),
                            entire_tool=list(range(tool["body_start"], insertion+len(inserted))))
            parsed = parse_native_blocks(ids, tokenizer)
            require([x["header_ids"] for x in parsed] == [x["header_ids"] for x in blocks],
                    "A native outer header changed")
            require([x["ending_ids"] for x in parsed] == [x["ending_ids"] for x in blocks],
                    "A native outer ending changed")
            require(ids[:insertion] == original[:insertion] and ids[insertion+len(inserted):] == original[insertion:],
                    "Insertion changed benign source tokens")
            suffix = condition + ("--"+payload["payload_type"] if payload else "")
            return dict(**common, input_id=prefix(trace_id)+"--"+suffix,
                        condition=condition, payload_type=payload["payload_type"] if payload else None,
                        payload_text=payload["payload_text"] if payload else None,
                        payload_text_sha256=payload["payload_text_sha256"] if payload else None,
                        input_ids=ids, input_tokens=len(ids), input_ids_sha256=ids_sha(ids),
                        segments=segments, target_positions=sorted(set(sum(segments.values(), []))),
                        injection_start=insertion, injection_tokens=len(inserted), provenance=provenance)
        records.append(tool_row("no_injection"))
        records.extend(tool_row(payload["condition"], payload) for payload in payload_records)

        cot_lineage = lineage["cot"]
        require(cot_lineage["scope"] == "actual_generated", "Native CoT reference is not raw generated text")
        cot_dir = args.source_root / "calls" / cot_lineage["call_id"]
        cot_request = read(cot_dir / "request.json")
        cot_blocks = parse_native_blocks(trace["input_ids"], tokenizer,
                                         allow_terminal_body=trace.get("allow_terminal_body", False))
        cot = cot_blocks[cot_lineage["block_index"]]
        require(cot["role"] == "cot" and cot["body_start"] == cot_lineage["trace_body_start"] and
                cot["body_end"] == cot_lineage["trace_body_end"], "Native CoT trace lineage differs")
        require(ids_sha(cot["body_ids"]) == cot_lineage["body_token_ids_sha256"], "Native CoT body hash differs")
        with np.load(cot_dir / "input.npz", allow_pickle=False) as source:
            cot_original_input = source["input_ids"].tolist()
        native_prefix = trace["input_ids"][:cot["end"]]
        source_full = cot_original_input + cot_request["generated_ids"]
        require(source_full[:len(native_prefix)] == native_prefix,
                "Native CoT reference is not the exact original live generation prefix")
        segments = dict(native_cot=list(range(cot["body_start"], cot["body_end"])))
        records.append(dict(**{**common, "context_kind":"original_live_first_cot_generation_prefix"},
                            input_id=prefix(trace_id)+"--native_cot", condition="native_cot", payload_type=None,
                            input_ids=native_prefix, input_tokens=len(native_prefix), input_ids_sha256=ids_sha(native_prefix),
                            segments=segments, target_positions=segments["native_cot"],
                            provenance=dict(source_call_id=cot_lineage["call_id"], source_scope="actual_generated",
                                            source_request_json=str((cot_dir/"request.json").resolve()),
                                            source_request_json_sha256=sha(cot_dir/"request.json"),
                                            source_input_npz=str((cot_dir/"input.npz").resolve()),
                                            source_input_npz_sha256=sha(cot_dir/"input.npz"),
                                            first_cot_training_lineage=cot_lineage,
                                            body_retokenization=False, artificial_ending_tokens_added=0)))
        sources.append(dict(**common, tool_source_call_id=tool_lineage["call_id"], cot_source_call_id=cot_lineage["call_id"],
                            original_live_input_tokens=len(original), native_cot_prefix_tokens=len(native_prefix)))
    require(len(records) == 684 and len({x["input_id"] for x in records}) == len(records), "Input inventory differs")
    with (args.output / "inputs.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n")
    write(args.output / "source_inventory.json", sources)
    protocol = dict(status="prepared", schema_version="cot-forgery-agentdojo-native-v1",
                    records=len(records), tool_exposure_records=627, native_cot_references=57,
                    tasks=57, task_splits=dict(Counter(x["source_split"] for x in traces)),
                    conditions=dict(Counter(x["condition"] for x in records)),
                    positions="Zero-based token positions; source intervals are half open. Payload paragraph masks include all overlapping BPE tokens, and may overlap if a token crosses a paragraph boundary.",
                    source_root=str(args.source_root.resolve()), traces_path=str(args.traces.resolve()),
                    traces_sha256=sha(args.traces), payload_yaml=str(args.payloads.resolve()), payload_yaml_sha256=sha(args.payloads),
                    probe=str(args.probe.resolve()), probe_sha256=sha(args.probe),
                    tokenizer=str(args.tokenizer.resolve()), tokenizer_files={name:sha(args.tokenizer/name)
                        for name in ("tokenizer.json","tokenizer_config.json")},
                    inputs_sha256=sha(args.output/"inputs.jsonl"), script_sha256=sha(Path(__file__)),
                    primary_analysis="TEST 9 task-equal paired contrasts; average the five payload pairs within task before task-level inference.",
                    secondary_analysis="ALL 57 descriptive only, because FIT and DEV bodies contributed to probe fitting or selection.",
                    experimental_unit="Benign task; the five payload types are repeated conditions, not independent tasks.",
                    method_differences=[
                        "Original agent experiment injected five exfiltration instructions and five instruction-plus-forged-policy pairs into Wikipedia HTML; here the exact same YAML payloads are appended to the first native tool result of each benign AgentDojo task.",
                        "The original native input, including JSON-escaped benign tool body, is preserved token-for-token. A literal double-newline plus the unescaped YAML payload is inserted inside the outer tool frame immediately before its original END token.",
                        "The supplied six-role gpt-oss-120b probe replaces the original experiment's trained probes. Probe layer and feature standardization must follow checkpoint metadata, rather than assuming the original plot's layer 16.",
                        "No exfiltration payload continuation or tool action is executed. These are teacher-forced role-probe measurements, not an attack-success evaluation.",
                        "Native CoT references use their exact original first-generation prefix; first-tool contexts are exact original live archived requests, which omit earlier analysis as the native API did.",
                        "Original CoT-forgery styling is first-person policy reasoning; textual destyling is a separate generated paraphrase experiment. These original agent pairs do not inject structural role tags.",
                        "Original published agent ASR plotting combines ATTEMPTED_SUCCESSFUL and ATTEMPTED_UNSUCCESSFUL; that plotted metric is attempt rate. No equivalent ASR is inferred here."
                    ],
                    interpretation="Probe probability describes resemblance to training role classes, not proof of policy adoption or successful task compromise.",
                    behavioral_evaluation_included=False)
    write(args.output/"protocol.json", protocol)
    print(json.dumps({k:protocol[k] for k in ("status","records","tasks","task_splits","conditions","inputs_sha256")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=REPO/"experiments/role-analysis/outputs/agentdojo-trace-interventions-20261112/sources/traces.jsonl")
    parser.add_argument("--source-root", type=Path, default=Path("/scratch/wz55191/agentdojo-trace-source-20261112/full57_no_payload_first_occurrence_probe_v7_20261027"))
    parser.add_argument("--payloads", type=Path, default=REPO/"experiments/cot-forgery-agent-evals/prompts/injections.yaml")
    parser.add_argument("--tokenizer", type=Path, default=Path("/tmp/wz55191-agentdojo-first-tool-model"))
    parser.add_argument("--probe", type=Path, default=REPO/"experiments/role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz")
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
