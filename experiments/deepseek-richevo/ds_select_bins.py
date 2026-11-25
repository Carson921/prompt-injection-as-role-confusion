"""Select one searched template per P(user) decile: intermediate candidates of both dsevo4 searches, ten equal bins.

Purpose: a P(user) -> ASR curve over the whole range from candidates the search produced on the way, not from its
best ones.  Rule (fixed before any ASR run of these templates, same as experiments/llama4-richevo/ds_select_bins.py):
per search, split [0, 1] into ten equal bins of the surrogate panel P(user); in every non-empty bin take the candidate
closest to the bin centre (ties: earlier evaluation) among the complete candidates without genuine special tokens
that were not tested before.  The conditions are written in an order that covers the range early (the ASR run may be
cut by the end of a GPU job and resumed).
usage: ds_select_bins.py EVOLUTION_ROOT OUT_DIR TESTED_DIR [TESTED_DIR ...]
"""
import json
import sys
from pathlib import Path

from ds_common import Codec, read_json, write_json

BINS = 10
ORDER = (0, 9, 4, 2, 6, 1, 8, 3, 5, 7)
root, out, tested = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
codec = Codec()
used = {c['template'] for directory in tested for c in read_json(directory / 'conditions.json').values()}
conditions = {}
for target, tag in (('agentdojo', 'ad'), ('ipi', 'ipi')):
    rows = [json.loads(line) for line in (root / target / 'scores.jsonl').read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['status'] == 'complete' and not codec.has_special(r['template'].replace('{goal}', ' '))]
    for k in ORDER:
        low, high, centre = k / BINS, (k + 1) / BINS, (k + 0.5) / BINS
        inside = [r for r in rows if low <= r['fitness'] < high or (k == BINS - 1 and r['fitness'] == 1.0)]
        fresh = [r for r in inside if r['template'] not in used]
        if not fresh:
            print(f'{target} bin [{low:.1f}, {high:.1f}): no candidate ({len(inside)} in bin)')
            continue
        row = min(fresh, key=lambda r: (abs(r['fitness'] - centre), r['evaluation_index']))
        used.add(row['template'])
        conditions[f'bin_{tag}_{round(centre * 100):02d}'] = dict(
            template=row['template'], source=f'dsevo4 {target} candidate closest to the centre of surrogate P(user) bin [{low:.1f}, {high:.1f})',
            optimized_for=target, bin=[low, high], bin_candidates=len(inside), fast_fitness=row['fitness'], suite_means=row['suite_means'],
            family=row.get('family'), candidate_source=row.get('source'), operator=row.get('operator'), wrapper_tokens=row['wrapper_tokens'],
            evaluation_index=row['evaluation_index'], evaluations=len(rows))
out.mkdir(parents=True, exist_ok=True)
write_json(out / 'conditions.json', conditions)
for name, c in sorted(conditions.items()):
    print(name, round(c['fast_fitness'], 4), c['bin_candidates'], c['wrapper_tokens'], c['evaluation_index'], c['candidate_source'], repr(c['template'])[:230])
