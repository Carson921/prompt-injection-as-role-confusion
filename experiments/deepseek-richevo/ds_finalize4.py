"""After a benchmark-specific evolution (corrected probe): exact re-measurement of the leading candidates.

usage: ds_finalize4.py EVOLUTION_DIR [verify_top]
Writes EVOLUTION_DIR/exact_top.json (sorted by exact fitness) with a `no_pretext` flag: wrapper starts directly
with the literal </tool_result> (round-2 behavior tests: text before the close tag lowers ASR).
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

from ds_common import write_json
from ds_fast_scorer import FastScorer


def main():
    evolution, top = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 10
    scorer = FastScorer(evolution / 'dataset.json', evolution)
    best = {}
    for line in (evolution / 'scores.jsonl').read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get('status') == 'complete' and (r['template'] not in best or r['fitness'] > best[r['template']]['fitness']):
            best[r['template']] = r
    ranked = sorted(best.values(), key=lambda r: -r['fitness'])
    chosen = []
    for r in ranked:  # leading candidates, skipping near-duplicates of an already chosen one
        if len(chosen) >= top:
            break
        if all(difflib.SequenceMatcher(None, r['template'], c['template']).ratio() < 0.9 for c in chosen):
            chosen.append(r)
    short = [r for r in ranked if r['template'].lstrip().startswith('</tool_result>')]
    for r in short[:max(3, top // 3)]:  # always include the best wrappers without text before the close tag
        if r not in chosen:
            chosen.append(r)
    exact = []
    for r in chosen:
        measured = scorer.score(r['template'], exact=True)
        exact.append(dict(template=r['template'], exact_fitness=measured['fitness'], fast_fitness=r['fitness'],
                          suite_means=measured['suite_means'], goal_role_means=measured['goal_role_means'],
                          per_case={c['case_id']: c['goal_user_mean'] for c in measured['per_case']},
                          wrapper_tokens=r['wrapper_tokens'], no_pretext=r['template'].lstrip().startswith('</tool_result>'),
                          source=r.get('source'), operator=r.get('operator'), evaluation_index=r.get('evaluation_index')))
        print(json.dumps(dict(fast=round(r['fitness'], 4), exact=round(measured['fitness'], 4), tokens=r['wrapper_tokens'],
                              suites={k: round(v, 3) for k, v in measured['suite_means'].items()}, template=r['template'][:140]),
                         ensure_ascii=False), flush=True)
        write_json(evolution / 'exact_top.json', sorted(exact, key=lambda x: -x['exact_fitness']))


if __name__ == '__main__':
    main()
