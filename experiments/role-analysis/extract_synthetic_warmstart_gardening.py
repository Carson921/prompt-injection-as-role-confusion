"""Capture the fixed public gardening case for frozen parent/new readouts.

This separate diagnostic never trains or edits the sealed six-hour experiment.
It retains raw selected-head features privately so future readouts need no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np

import evaluate_synthetic_gardening as gardening
import synthetic_agent_gpu as runtime
from synthetic_agent_classifier import Probe
from synthetic_role_warmstart_gpu import check_budget, validate_test_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--parent-probe', type=Path, required=True)
    parser.add_argument('--new-probe', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--budget-file', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    require = runtime.require
    budget = check_budget(args.budget_file)
    parent, new = Probe(args.parent_probe), Probe(args.new_probe)
    require(runtime.digest(args.parent_probe) == '22cb7251cd96c81d8d1225311b1e3048499850f738b56f4b57040ad7bedf4125', 'Wrong parent')
    require(runtime.digest(args.new_probe) == '45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df', 'Wrong candidate')
    freeze = validate_test_freeze(args.freeze, parent)
    require(runtime.digest(args.new_probe) == freeze['probe_sha256'], 'Candidate differs from article DEV freeze')
    require(runtime.digest(args.budget_file) == freeze['gpu_budget']['budget_sha256'], 'Not the original frozen six-hour authorization')
    require(budget['started_unix'] == freeze['gpu_budget']['started_unix'] and budget['deadline_unix'] == freeze['gpu_budget']['deadline_unix'], 'Frozen budget window differs')
    binding = runtime.model_binding(args.model_dir)
    for key in ('heads', 'roles', 'feature_dim', 'feature_position', 'model_binding'):
        require(parent.metadata[key] == new.metadata[key], 'Probe feature identity differs: ' + key)
    require(binding == parent.metadata['model_binding'], 'Backbone differs from frozen probes')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    samples, preparation = gardening.prepare_samples(tokenizer, args.source_dir/'inputs.json', args.source_dir/'tomato.yaml', args.source_dir/'gptoss.j2')
    preparation['tokenizer_assets_sha256'] = {p.name:runtime.digest(p) for p in sorted(args.model_dir.iterdir()) if p.is_file() and (p.name.startswith('tokenizer') or p.name=='special_tokens_map.json')}
    if args.check_only:
        print(json.dumps({'status':'ready','formats':list(gardening.FORMATS),'body_rows_per_format':916,'head_dim':1024,'budget_deadline_unix':budget['deadline_unix']}), flush=True)
        return
    require(not (args.root/'manifest.json').exists(), 'Committed diagnostic exists; use its saved cache')
    args.root.mkdir(parents=True, exist_ok=True)
    runtime.atomic_json(args.root/'prepared-inputs.json', samples)
    runtime.atomic_json(args.root/'prepare-manifest.json', preparation)
    started = time.time()
    stop_at = min(budget['deadline_unix'], started+600)
    timer = threading.Timer(max(0, stop_at-time.time()), lambda:os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    report = {'format_version':'warmstart-gardening-raw-cache-v1','status':'running','started_unix':started,'local_deadline_unix':stop_at,'original_budget_sha256':runtime.digest(args.budget_file),'original_budget_deadline_unix':budget['deadline_unix'],'original_budget_start_unix':budget['started_unix'],'budget_not_rearmed':True,'parent_probe_sha256':runtime.digest(args.parent_probe),'new_probe_sha256':runtime.digest(args.new_probe),'freeze_sha256':runtime.digest(args.freeze),'feature_dim':1024,'feature_position':parent.metadata['feature_position'],'heads':parent.metadata['heads'],'roles':parent.metadata['roles'],'model_binding':binding,'prepared_inputs_sha256':runtime.digest(args.root/'prepared-inputs.json'),'prepare_manifest_sha256':runtime.digest(args.root/'prepare-manifest.json'),'extractor_sha256':runtime.digest(__file__),'runtime_sha256':runtime.digest(runtime.__file__),'source_parser_sha256':runtime.digest(gardening.__file__),'selection_unchanged':True,'new_agent_rollout':False,'new_response_generation':False,'fixed_original_conversation_forward':True,'devices':[0,1,2],'formats':{}}
    runtime.atomic_json(args.root/'started.json',report)
    try:
        import torch
        with runtime.MemoryMonitor(3) as monitor:
            check_budget(args.budget_file)
            model = runtime.load_model(args.model_dir,args.root/'backend',devices='0,1,2',cache_dir=args.root/'cache')
            for sample in samples:
                check_budget(args.budget_file)
                require(time.time()<stop_at,'Diagnostic deadline expired')
                before=time.monotonic()
                positions=np.asarray([r['token_position'] for r in sample['content']],dtype=np.int64)
                capture=runtime.HeadCapture(positions,selected=parent.metadata['heads'])
                with torch.inference_mode(),capture.install(model):
                    hidden=runtime._forward(model,sample['input_ids'])
                    require(bool(torch.isfinite(hidden).all()),'Nonfinite hidden states')
                x=capture.matrix()
                require(x.shape==(916,1024) and x.dtype==np.float16 and np.isfinite(x).all(),'Invalid retained selected-head features')
                path=args.root/(sample['prompt_key']+'.npz')
                runtime.atomic_npz(path,head_features=x,positions=positions,input_ids=np.asarray(sample['input_ids'],dtype=np.int64),token_ids=np.asarray([r['token_id'] for r in sample['content']],dtype=np.int64),head_ids=np.asarray(parent.metadata['heads'],dtype=np.int64),roles=np.asarray(parent.metadata['roles']))
                report['formats'][sample['prompt_key']]={'file':path.name,'sha256':runtime.digest(path),'input_tokens':len(sample['input_ids']),'body_rows':len(x),'prompt_sha256':sample['prompt_sha256'],'elapsed_seconds':time.monotonic()-before,'rows_in_original_message_order':True}
                print(json.dumps({'stage':'gardening_capture','format':sample['prompt_key'],**report['formats'][sample['prompt_key']]}),flush=True)
                del hidden,x,capture
            require(len(report['formats'])==3,'Missing an original format')
            del model
            for device in range(3):
                with torch.cuda.device(device):torch.cuda.empty_cache()
        require(not monitor.errors,'GPU telemetry failed')
        check_budget(args.budget_file)
        require(time.time()<stop_at,'Diagnostic completed after its deadline')
        report.update(status='complete',forward_count=3,finished_unix=time.time(),device_wide_sampled_peak_mib=monitor.peak,memory_monitor_errors=monitor.errors)
        runtime.atomic_json(args.root/'manifest.json',report)
    finally:
        timer.cancel()


if __name__=='__main__':
    main()
