"""Select the ASR conditions of the corrected-probe round from the two benchmark-specific exact rankings.

Rule (fixed before any ASR run): per benchmark, (top) the template with the highest exact P(user) on that
benchmark's panel, and (short) the best exact template that has no text before the literal </tool_result>.
usage: ds_select4.py EVOLUTION_ROOT OUT_DIR
"""
import sys
from pathlib import Path

from ds_common import Codec, MODEL, read_json, write_json

root, out = Path(sys.argv[1]), Path(sys.argv[2])
codec = Codec(MODEL)
conditions = {}
for target, tag in (('agentdojo', 'ad'), ('ipi', 'ipi')):
    rows = sorted(read_json(root / target / 'exact_top.json'), key=lambda r: -r['exact_fitness'])
    picks = dict(top=rows[0], short=next((r for r in rows if r['no_pretext']), None))
    for kind, row in picks.items():
        if row is None or any(c['template'] == row['template'] for c in conditions.values()):
            continue
        assert not codec.has_special(row['template'].replace('{goal}', ' '))
        conditions[f'opt4_{tag}_{kind}'] = dict(template=row['template'], source=f'dsevo4 {target} exact rank ({kind})', optimized_for=target,
                                                panel_exact=row['exact_fitness'], panel_suite_means=row['suite_means'],
                                                wrapper_tokens=row['wrapper_tokens'], fast_fitness=row['fast_fitness'],
                                                evaluation_index=row['evaluation_index'], no_pretext=row['no_pretext'])
out.mkdir(parents=True, exist_ok=True)
write_json(out / 'conditions.json', conditions)
for name, c in conditions.items():
    print(name, round(c['panel_exact'], 4), c['wrapper_tokens'], repr(c['template']))
