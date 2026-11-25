"""Select one searched template per P(user) decile: intermediate candidates of both evolutions, ten equal bins.

Purpose: a P(user) -> ASR curve over the whole range from candidates the search produced on the way, not from its
best ones.  Rule (fixed before any ASR run of these templates): per evolution, split [0, 1] into ten equal bins of the
surrogate panel P(user); in every bin take the candidate closest to the bin centre (ties: earlier evaluation) among
the complete candidates without genuine special tokens that were not tested before (optimized and graded sets).
usage: ds_select_bins.py EVOLUTION_ROOT OUT_DIR TESTED_DIR [TESTED_DIR ...]
"""
import json
import sys
from pathlib import Path

from ds_common import Codec, MODEL, read_json, write_json

BINS = 10
root, out, tested = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
codec = Codec(MODEL)
used = {c['template'] for directory in tested for c in read_json(directory / 'conditions.json').values()}
conditions = {}
for target, tag in (('agentdojo', 'ad'), ('ipi', 'ipi')):
    rows = [json.loads(line) for line in (root / target / 'scores.jsonl').read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['status'] == 'complete' and not codec.has_special(r['template'].replace('{goal}', ' '))]
    for k in range(BINS):
        low, high, centre = k / BINS, (k + 1) / BINS, (k + 0.5) / BINS
        inside = [r for r in rows if low <= r['fitness'] < high or (k == BINS - 1 and r['fitness'] == 1.0)]
        fresh = [r for r in inside if r['template'] not in used]
        row = min(fresh, key=lambda r: (abs(r['fitness'] - centre), r['evaluation_index']))
        used.add(row['template'])
        conditions[f'bin_{tag}_{round(centre * 100):02d}'] = dict(
            template=row['template'], source=f'l4evo {target} candidate closest to the centre of surrogate P(user) bin [{low:.1f}, {high:.1f})',
            optimized_for=target, bin=[low, high], bin_candidates=len(inside), fast_fitness=row['fitness'], family=row.get('family'),
            candidate_source=row.get('source'), operator=row.get('operator'), wrapper_tokens=row['wrapper_tokens'],
            evaluation_index=row['evaluation_index'], evaluations=len(rows))
out.mkdir(parents=True, exist_ok=True)
write_json(out / 'conditions.json', conditions)
for name, c in conditions.items():
    print(name, round(c['fast_fitness'], 4), c['bin_candidates'], c['wrapper_tokens'], c['evaluation_index'], repr(c['template'])[:220])
