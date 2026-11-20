"""Compare the cached-prefix surrogate with recorded exact (full no-cache) scores of the first run."""
import json, sys, time
from pathlib import Path
import numpy as np
from ds_common import write_json
from ds_fast_scorer import FastScorer

out = Path('outputs/dsevo2-20261119')
rows = [json.loads(l) for l in open('outputs/dsevo-20261118/scores.jsonl') if l.strip()]
rows = sorted((r for r in rows if r.get('status') == 'complete'), key=lambda r: -r['fitness'])
count = int(sys.argv[1]) if len(sys.argv) > 1 else 30
picked = rows[:count // 2] + rows[count // 2::max(1, (len(rows) - count // 2) // (count - count // 2))][:count - count // 2]
t0 = time.time()
scorer = FastScorer('outputs/dsevo-20261118/dataset.json', out / 'fast_validation')
print('prefix caches built in', round(time.time() - t0, 1), 's', scorer.protocol['prefix_tokens'], flush=True)
pairs, modes, seconds = [], {}, []
for r in picked:
    t = time.time()
    result = scorer.score(r['template'])
    seconds.append(time.time() - t)
    exact = {c['case_id']: c['goal_user_mean'] for c in r['per_case']}
    for c in result['per_case']:
        pairs.append((exact[c['case_id']], c['goal_user_mean'], c['case_id']))
        modes[c['mode']] = modes.get(c['mode'], 0) + 1
    print(json.dumps(dict(exact=round(r['fitness'], 4), fast=round(result['fitness'], 4), seconds=round(seconds[-1], 1),
                          template=r['template'][:60])), flush=True)
a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
fit_exact = np.array([r['fitness'] for r in picked])
summary = dict(templates=len(picked), case_pairs=len(pairs), pearson_case=float(np.corrcoef(a, b)[0, 1]),
               mean_abs_diff_case=float(np.abs(a - b).mean()), max_abs_diff_case=float(np.abs(a - b).max()),
               modes=modes, median_seconds_per_candidate=float(np.median(seconds)))
write_json(out / 'fast_validation.json', dict(summary=summary, pairs=pairs))
print(json.dumps(summary))
