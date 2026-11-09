"""Score the top panel templates of an ipievo run on all 36 primary behaviors (needs both H100s)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import MODEL, read_json, write_json
from probe_score import load_scorer_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    output = args.output.resolve()
    rows = [json.loads(l) for l in (output / 'scores.jsonl').read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get('status') == 'complete']
    best = {}
    for r in rows:
        if r['template'] not in best or r['fitness'] > best[r['template']]['fitness']:
            best[r['template']] = r
    ranked = sorted(best.values(), key=lambda r: -r['fitness'])[:args.top]
    done = {}
    full_path = output / 'full_scores.jsonl'
    if full_path.exists():
        for l in full_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done[r['template']] = r
    import ipi_scorer
    scorer = ipi_scorer.Scorer(output / 'ipi_dataset.json', output, MODEL, load_scorer_module().DEFAULT_PROBE)
    results = []
    for r in ranked:
        full = done.get(r['template']) or scorer.score_full(r['template'])
        results.append(dict(panel_fitness=r['fitness'], full_mean=full['full_mean'], suite_means=full['suite_means'],
                            wrapper_tokens=r['wrapper_tokens'], per_case=full['per_case'], template=r['template']))
        print(json.dumps(dict(panel=round(r['fitness'], 4), full=round(full['full_mean'], 4), tokens=r['wrapper_tokens'])), flush=True)
    results.sort(key=lambda x: -x['full_mean'])
    write_json(output / 'full_validation.json', results)
    scorer.close()


if __name__ == '__main__':
    main()
