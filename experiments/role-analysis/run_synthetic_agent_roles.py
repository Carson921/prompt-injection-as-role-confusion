"""Run the article-only six-role GPT-OSS-120B experiment in explicit stages.

All state is scoped by --run-dir and --data-dir; existing probes are never loaded.
Use separate pilot/formal run directories. TEST is evaluated only after freeze.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_deadline(config):
    deadline = config.get('budget', {}).get('deadline_utc')
    if deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline.replace('Z', '+00:00')):
        raise RuntimeError('The user-authorized 24-hour execution window has expired.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'preflight', 'scan', 'extract', 'train', 'freeze', 'evaluate', 'report', 'plot', 'external-protocol', 'external-prepare'])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--devices', default='0,1,2,3')
    parser.add_argument('--phase', choices=['pilot', 'formal'])
    parser.add_argument('--split', default='dev', choices=['dev', 'test', 'external'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase:
        config['phase'] = args.phase
    phase = config.get('phase', 'pilot')
    config.setdefault('classifier', {})['max_epochs'] = 15 if phase == 'pilot' else 45
    check_deadline(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HOME', str(args.cache_dir / 'huggingface'))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    bound = args.run_dir / 'config.json'
    if bound.exists() and json.loads(bound.read_text()) != config:
        raise RuntimeError('Run configuration changed; use a distinct run directory.')
    atomic_json(bound, config)
    status = dict(stage=args.stage, phase=phase, status='running', started_unix=time.time(), config_hash=identity)
    atomic_json(args.run_dir / 'status.json', status)
    print(json.dumps(status), flush=True)
    try:
        if args.stage in {'external-protocol', 'external-prepare'}:
            import synthetic_agent_external as external
            if phase != 'formal':
                raise RuntimeError('External dialogue testing belongs to the formal experiment only.')
            if args.stage == 'external-protocol':
                result = external.prepare_external_protocol(config, args.data_dir, args.run_dir)
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
                result = external.prepare_external(config, tokenizer, args.data_dir, args.run_dir)
        elif args.stage == 'prepare':
            from transformers import AutoTokenizer
            from synthetic_agent_data import prepare_data
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            result = prepare_data(config, tokenizer, args.data_dir, args.run_dir, phase=phase, resume=args.resume)
        elif args.stage in {'preflight', 'scan', 'extract'}:
            import synthetic_agent_gpu as gpu
            kwargs = dict(model_dir=args.model_dir, run_dir=args.run_dir, config=config,
                          devices=args.devices, cache_dir=args.cache_dir)
            if args.stage == 'preflight':
                result = gpu.preflight(**kwargs)
            else:
                kwargs.update(data_dir=args.data_dir, phase=phase, resume=args.resume)
                if args.stage == 'extract' and args.split in {'test', 'external'}:
                    if not (args.run_dir / 'freeze.json').exists():
                        raise RuntimeError('Freeze the selected classifier before extracting TEST.')
                    kwargs['splits'] = ('test',)
                    if args.split == 'external':
                        kwargs['phase'] = 'external'
                result = getattr(gpu, args.stage)(**kwargs)
        elif args.stage in {'train', 'freeze', 'evaluate'}:
            import synthetic_agent_classifier as classifier
            feature_dir = args.run_dir / 'features' / phase
            if args.stage == 'train':
                result = classifier.train_candidates(feature_dir, args.run_dir, config, resume=args.resume)
            elif args.stage == 'freeze':
                result = classifier.freeze(args.run_dir)
            else:
                if phase == 'pilot' and args.split != 'dev':
                    raise RuntimeError('Pilot may evaluate DEV only.')
                if args.split == 'external':
                    feature_dir = args.run_dir / 'features' / 'external'
                    result = classifier.evaluate(feature_dir, args.run_dir, split='test', output_name='external')
                else:
                    result = classifier.evaluate(feature_dir, args.run_dir, split=args.split)
        elif args.stage == 'plot':
            from synthetic_agent_plots import plot_run
            result = plot_run(args.run_dir, args.data_dir, phase=phase, split=args.split)
        else:
            from synthetic_agent_report import report
            result = report(args.run_dir, args.data_dir, phase=phase)
        status.update(status='complete', seconds=time.time()-status['started_unix'])
        atomic_json(args.run_dir / (args.stage + '-stage.json'), status)
        atomic_json(args.run_dir / 'status.json', status)
        print(json.dumps(status), flush=True)
    except BaseException as error:
        status.update(status='failed', error=repr(error), seconds=time.time()-status['started_unix'])
        atomic_json(args.run_dir / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
