"""Build conditions.json for an ASR run of IPI-optimized (ipievo) templates plus reference conditions.

Reference conditions (plain baseline and selected AgentDojo band templates) are copied from an
earlier run so the comparison is made within one WorldSim session.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evolution', type=Path, required=True, help='ipievo run dir with full_validation.json')
    parser.add_argument('--reference', type=Path, required=True, help='earlier ASR run dir with conditions.json')
    parser.add_argument('--reference-conditions', default='plain,band_0.99_1.00,band_0.80_0.90')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reference = read_json(args.reference / 'conditions.json')
    validation = read_json(args.evolution / 'full_validation.json')
    conditions = {}
    for name in args.reference_conditions.split(','):
        conditions[name] = {**reference[name], 'source_run': str(args.reference)}
    seen = set()
    rank = 0
    for row in validation:  # sorted by full_mean descending
        if row['template'] in seen:
            continue
        seen.add(row['template'])
        rank += 1
        if rank > args.top:
            break
        conditions[f'ipievo_{rank}'] = dict(condition=f'ipievo_{rank}', template=row['template'], p_user_agentdojo=None,
                                            asr_agentdojo=None, successes_agentdojo=None, scored_agentdojo=None,
                                            p_user_ipi_optimization=row['full_mean'], panel_fitness=row['panel_fitness'],
                                            suite_means_optimization=row['suite_means'], wrapper_tokens=row['wrapper_tokens'],
                                            source='ipievo', source_run=str(args.evolution))
    write_json(args.out, conditions)
    for name, spec in conditions.items():
        print(name, spec.get('p_user_ipi_optimization'), spec.get('p_user_agentdojo'), len(spec['template']))


if __name__ == '__main__':
    main()
