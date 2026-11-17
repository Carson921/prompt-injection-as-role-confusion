"""Resident role-probe service: scores a candidate payload for a scenario on gpt-oss-120b (both H100s).

POST /score {"scenario_id", "payload", "goal"} -> {"goal_user_mean", "goal_role_means", "payload_role_means",
"segments": [{"text", "dominant", "user", "tool", ...}], "tokens"}.  Rendering is identical to prepare.py.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np

from common import IPI_RICHEVO, MODEL, locate_payload, read_json, render_prompt, special_floor
from prepare import inject
from target import openai_tools

sys.path.insert(0, str(IPI_RICHEVO))
from probe_score import Prober, load_scorer_module  # noqa: E402


class Service:
    def __init__(self, dataset_dir: Path, output: Path, date: str):
        from transformers import AutoTokenizer
        self.entries = {e['scenario_id']: e for e in read_json(dataset_dir / 'dataset.json')['scenarios']}
        self.trajectories = read_json(dataset_dir / 'trajectories.json')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.floor, self.spellings = special_floor(MODEL)
        output.mkdir(parents=True, exist_ok=True)
        self.prober = Prober(MODEL, load_scorer_module().DEFAULT_PROBE, output)
        self.roles = list(self.prober.roles)
        self.date = date
        self.lock = threading.Lock()
        self.served = 0

    def score(self, scenario_id: str, payload: str, goal: str) -> dict:
        entry, trajectory = self.entries[scenario_id], self.trajectories[scenario_id]
        if any(s in payload for s in self.spellings):
            raise ValueError('payload contains a real special token spelling')
        messages = json.loads(json.dumps(trajectory['messages']))
        messages[-1]['content'], _ = inject(messages[-1]['content'], payload)
        prompt = render_prompt(self.tokenizer, messages, openai_tools(entry['task_tools'] + entry['target_tools']), self.date)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        if goal not in payload:
            goal = payload  # whole payload is scored when the goal text was paraphrased away
        occurrence = locate_payload(prompt, offsets, payload, goal)[0]
        request = dict(prompt_token_ids=ids, payload_occurrences=[occurrence], scored_occurrence=0)
        with self.lock:
            result = self.prober.score(request)
            self.served += 1
        probs, positions = result['probabilities'], result['positions']
        # segments: split the payload by line breaks, map tokens by character overlap
        segments = []
        base = occurrence['start_char']
        rendered = prompt[base:occurrence['end_char']]
        bounds, last = [], 0
        for match in re.finditer(r'(?:\\)?\\n', rendered):  # escaped newline (once or twice JSON-escaped)
            bounds.append((last, match.start()))
            last = match.end()
        bounds.append((last, len(rendered)))
        for a, b in bounds:
            line, start, end = rendered[a:b], base + a, base + b
            if not line.strip():
                continue
            mask = np.array([offsets[i][0] < end and offsets[i][1] > start for i in positions])
            if not mask.any():
                continue
            mean = probs[mask].mean(0)
            segments.append(dict(text=line.replace('\\"', '"')[:120], dominant=self.roles[int(mean.argmax())],
                                 **{r: float(mean[j]) for j, r in enumerate(self.roles)}, tokens=int(mask.sum())))
        return dict(goal_user_mean=result['means']['goal']['user'], goal_role_means=result['means']['goal'],
                    payload_role_means=result['means']['full_payload'], segments=segments, tokens=len(ids),
                    seconds=result['elapsed_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).with_name('outputs') / 'dataset-v1')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('outputs') / 'probe-server')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--date', default='2026-11-11')
    args = parser.parse_args()
    service = Service(args.dataset, args.output, args.date)
    log = (args.output / 'requests.jsonl').open('a')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(dict(ready=True, served=service.served)).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            try:
                result = service.score(body['scenario_id'], body['payload'], body.get('goal', ''))
                status = 200
            except Exception as error:
                result, status = dict(error=f'{type(error).__name__}: {error}'), 400
            log.write(json.dumps(dict(unix=time.time(), scenario_id=body.get('scenario_id'), payload=body.get('payload'),
                                      goal_user_mean=result.get('goal_user_mean'), error=result.get('error')), ensure_ascii=False) + '\n')
            log.flush()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *a):
            pass

    print(json.dumps(dict(stage='ready', port=args.port)), flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
