"""Select a graded set of searched templates: candidates of both evolutions at fixed surrogate P(user) levels.

Purpose: a P(user) -> ASR curve from templates that were not picked for being the best.  Rule (fixed before any ASR
run of these templates): per evolution and per level in LEVELS, the candidate whose surrogate panel P(user) is
closest to the level (ties: earlier evaluation), then the closest candidate of a different structure family.
Also adds a rerun of opt_ad_top (same template, second run) as a run-to-run noise check.
usage: ds_select_graded.py EVOLUTION_ROOT OPTIMIZED_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

from ds_common import Codec, MODEL, read_json, write_json

LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
root, optimized, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
codec = Codec(MODEL)
picked = read_json(optimized / 'conditions.json')
conditions, used = {}, {c['template'] for c in picked.values()}
for target, tag in (('agentdojo', 'ad'), ('ipi', 'ipi')):
    rows = [json.loads(line) for line in (root / target / 'scores.jsonl').read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['status'] == 'complete' and not codec.has_special(r['template'].replace('{goal}', ' '))]
    for level in LEVELS:
        order = sorted(rows, key=lambda r: (abs(r['fitness'] - level), r['evaluation_index']))
        first = next(r for r in order if r['template'] not in used)
        used.add(first['template'])
        second = next((r for r in order if r['template'] not in used and r.get('family') != first.get('family')), None) \
            or next(r for r in order if r['template'] not in used)
        used.add(second['template'])
        for letter, row in (('a', first), ('b', second)):
            conditions[f'grade_{tag}_{round(level * 100):02d}{letter}'] = dict(
                template=row['template'], source=f'l4evo {target} candidate closest to surrogate P(user) {level}', optimized_for=target,
                level=level, fast_fitness=row['fitness'], family=row.get('family'), candidate_source=row.get('source'),
                wrapper_tokens=row['wrapper_tokens'], evaluation_index=row['evaluation_index'])
conditions['opt_ad_top_rerun'] = dict(picked['opt_ad_top'], source='second run of opt_ad_top (run-to-run noise check)')
out.mkdir(parents=True, exist_ok=True)
write_json(out / 'conditions.json', conditions)
for name, c in conditions.items():
    print(name, round(c['fast_fitness'], 4), c['wrapper_tokens'], c.get('family'), repr(c['template'])[:200])
