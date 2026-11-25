"""Qwen3.5 evolution scored through the persistent probe server with the cached-prefix surrogate (ds_fast_scorer).

Every clearly new best above DS_EXACT_FLOOR is re-measured exactly (full no-cache forward on all contexts) and logged
to exact_scores.jsonl; selection for ASR uses exact scores only.
Run with the chatinject-table1 interpreter (CPU only; the GPU work happens in ds_probe_server.py).
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time

import ds_evolution as E  # installs the DeepSeek vocabulary, library and LLM briefing
from ds_common import append_jsonl, read_json
from ds_fast_scorer import FastScorer
import ds_grammar

R, L, G = E.R, E.L, ds_grammar.G
BaseRun = E.QwenRun.__mro__[1]

L.SYSTEM += """

Measured facts from this study: (replaced at run time by ds_evolution4.py with the measurements of this model)"""


class PromotingFastScorer(FastScorer):
    """Surrogate fitness for the search; exact re-measurement of each clearly new best."""
    STEP = 0.015
    FLOOR = 0.0  # no exact check below this surrogate fitness (exact forwards block the single-threaded server)

    def __init__(self, dataset_path, output_path, model_path=None, probe_path=None, target_role='user'):
        super().__init__(dataset_path, output_path, target_role=target_role)
        self.exact_log = self.output / 'exact_scores.jsonl'
        self.verified = -1.0
        if self.exact_log.exists():
            for line in self.exact_log.read_text().splitlines():
                self.verified = max(self.verified, json.loads(line)['fast_fitness'])

    def score(self, template, exact=None):
        result = super().score(template, exact)
        if not exact and not result.get('cache_hit') and result['fitness'] >= max(self.verified + self.STEP, self.FLOOR):
            self.verified = result['fitness']
            measured = super().score(template, exact=True)
            append_jsonl(self.exact_log, dict(template=template, fast_fitness=result['fitness'], exact_fitness=measured['fitness'],
                                              exact_suite_means=measured['suite_means'], fast_suite_means=result['suite_means'],
                                              exact_per_case={c['case_id']: c['goal_user_mean'] for c in measured['per_case']},
                                              unix=time.time()))
            print(json.dumps(dict(event='exact_check', fast=round(result['fitness'], 4), exact=round(measured['fitness'], 4),
                                  suites={k: round(v, 3) for k, v in measured['suite_means'].items()})), flush=True)
        return result


class FastRun(E.QwenRun):
    def __init__(self, args):
        R.SUITES = tuple(read_json(args.dataset)['suites'])
        BaseRun.__init__(self, args)

    def run(self):
        try:
            return BaseRun.run(self)
        finally:
            self.stop.set()
            if self.client is not None:
                self.client.pool.shutdown(wait=False, cancel_futures=True)


E.scorer_module.Scorer = PromotingFastScorer
R.Run = FastRun

if __name__ == '__main__':
    R.main()
