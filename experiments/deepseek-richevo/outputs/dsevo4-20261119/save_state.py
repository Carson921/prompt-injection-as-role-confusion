"""Record where each dsevo4 evolution stopped (read-only on the run dirs; writes stop_state.json)."""
import json, sys, time
from pathlib import Path
D = Path(__file__).resolve().parent
PLAN = dict(agentdojo=dict(hours=2.5, first_cap_minutes=60), ipi=dict(hours=3.5, first_cap_minutes=75))
state = dict(saved_unix=time.time(), saved=time.strftime('%F %T %Z'), probe_sha256='66c100b8c9e84c7921a413093ce3d70806510f741aaebaf3063752b48cb90c66')
for target, plan in PLAN.items():
    E = D / target
    rows = [json.loads(l) for l in (E / 'scores.jsonl').read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get('status') == 'complete']
    stamps = [json.loads(l)['unix'] for l in (E / 'run.log').read_text().splitlines() if l.startswith('{') and '"unix"' in l]
    # running time = sum of gaps shorter than 10 minutes (restarts and the stop leave longer gaps)
    elapsed = sum(b - a for a, b in zip(stamps, stamps[1:]) if b - a < 600)
    exact = [json.loads(l) for l in (E / 'exact_scores.jsonl').read_text().splitlines() if l.strip()] if (E / 'exact_scores.jsonl').exists() else []
    best = max(rows, key=lambda r: r['fitness'])
    best_exact = max(exact, key=lambda r: r['exact_fitness']) if exact else None
    state[target] = dict(planned_hours=plan['hours'], elapsed_hours=round(elapsed / 3600, 3),
                         remaining_hours=round(max(0.0, plan['hours'] - elapsed / 3600), 3),
                         remaining_first_cap_minutes=round(max(1.0, plan['first_cap_minutes'] - elapsed / 60), 1),
                         scored=len(rows), best_fast=dict(fitness=best['fitness'], suite_means=best['suite_means'], template=best['template']),
                         best_exact=best_exact and dict(exact_fitness=best_exact['exact_fitness'], fast_fitness=best_exact['fast_fitness'],
                                                        exact_suite_means=best_exact['exact_suite_means'], template=best_exact['template']))
(D / 'stop_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=1))
print(json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a not in ('best_fast', 'best_exact')}) for k, v in state.items()}, ensure_ascii=False))
