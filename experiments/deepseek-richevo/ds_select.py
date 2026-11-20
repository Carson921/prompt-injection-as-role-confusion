"""Pick optimized templates from a finished DeepSeek evolution run for ASR testing.

opt_top1: highest objective.  opt_top2..k: next best that are textually different from every
earlier pick (difflib ratio below a threshold).  band_X: template whose objective is closest to X,
to trace ASR against P(user) (same idea as the GPT-OSS band tables). Also retain the best
AgentDojo-six and IPI-six panel means from this same joint-search ledger. These are post-hoc
benchmark rankings, not independent optimizations. Duplicate picks receive extra labels.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
from pathlib import Path

from ds_common import read_json, write_json


def panel_scores(row, agentdojo_ids, ipi_ids):
    values = {r['case_id']: r['goal_user_mean'] for r in row['per_case']}
    if len(row['per_case']) != 12 or set(values) != agentdojo_ids | ipi_ids:
        raise ValueError(f"Incomplete scoring panel at evaluation {row['evaluation_index']}")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
        raise ValueError(f"Invalid probability at evaluation {row['evaluation_index']}")
    result = dict(joint=math.fsum(values.values()) / 12,
                  agentdojo_mean=math.fsum(values[k] for k in agentdojo_ids) / 6,
                  ipi_mean=math.fsum(values[k] for k in ipi_ids) / 6)
    if not math.isclose(row['fitness'], result['joint'], rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"Fitness differs from the 12-case mean at evaluation {row['evaluation_index']}")
    return result


def select_templates(rows, dataset, top=2, bands=(0.2, 0.4, 0.6, 0.8), max_similarity=0.6):
    agentdojo_ids = {r['case_id'] for r in dataset['contexts']}
    ipi_ids = set(dataset['ipi_cases'])
    if len(agentdojo_ids) != 6 or len(ipi_ids) != 6 or agentdojo_ids & ipi_ids:
        raise ValueError('Selection requires the frozen AD6 + IPI6 joint panel')
    best, scores, evaluation_hashes = {}, {}, {}
    for row in rows:
        if row.get('status') != 'complete':
            continue
        index = row['evaluation_index']
        template_hash = hashlib.sha256(row['template'].encode()).hexdigest()
        if index in evaluation_hashes and evaluation_hashes[index] != template_hash:
            raise ValueError(f'Conflicting template identities at evaluation {index}')
        evaluation_hashes[index] = template_hash
        scores[index] = panel_scores(row, agentdojo_ids, ipi_ids)
        if template_hash not in best or row['fitness'] > best[template_hash]['fitness']:
            best[template_hash] = row
    ranked = sorted(best.values(), key=lambda r: (-r['fitness'], r['evaluation_index']))
    if not ranked:
        raise ValueError('No complete candidate is available for selection')
    picks, selected_hashes, selected_indices = {}, {}, {}

    def add(name, row, kind, label):
        index = row['evaluation_index']
        template_hash = hashlib.sha256(row['template'].encode()).hexdigest()
        existing = selected_hashes.get(template_hash, selected_indices.get(index))
        if existing is not None:
            if label not in picks[existing]['selection_labels']:
                picks[existing]['selection_labels'].append(label)
            return
        picks[name] = dict(template=row['template'], template_sha256=template_hash,
                           p_user_optimization=row['fitness'], suite_means=row['suite_means'],
                           wrapper_tokens=row['wrapper_tokens'], source=kind, evolution_source=row.get('source'),
                           operator=row.get('operator'), evaluation_index=index,
                           selection_labels=[label], selection_scores=scores[index],
                           selection_population='same AD6+IPI6 joint-search ledger')
        selected_hashes[template_hash] = selected_indices[index] = name

    for row in ranked:
        if len(picks) >= top:
            break
        if all(difflib.SequenceMatcher(None, row['template'], p['template']).ratio() < max_similarity for p in picks.values()):
            add(f'opt_top{len(picks) + 1}', row, 'top', 'joint_top')
    ceiling = ranked[0]['fitness']
    for center in bands:
        if center >= ceiling - 0.05:
            continue
        row = min(ranked, key=lambda r: abs(r['fitness'] - center))
        add(f'band_{center:.2f}', row, 'band', f'joint_band_{center:.2f}')
    for benchmark, name in (('agentdojo', 'opt_ad_champion'), ('ipi', 'opt_ipi_champion')):
        champion = max(ranked, key=lambda r: (scores[r['evaluation_index']][benchmark + '_mean'],
                                             r['fitness'], -r['evaluation_index']))
        add(name, champion, 'benchmark_posthoc', f'posthoc_{benchmark}_champion')
    return picks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evolution', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--top', type=int, default=2)
    parser.add_argument('--bands', default='0.2,0.4,0.6,0.8')
    parser.add_argument('--max-similarity', type=float, default=0.6)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.evolution / 'scores.jsonl').read_text().splitlines() if line.strip()]
    picks = select_templates(rows, read_json(args.evolution / 'dataset.json'), args.top,
                             [float(b) for b in args.bands.split(',') if b], args.max_similarity)
    write_json(args.out, picks)
    for name, p in picks.items():
        print(json.dumps(dict(condition=name, p_user=round(p['p_user_optimization'], 4), tokens=p['wrapper_tokens'],
                              suites={k: round(v, 3) for k, v in p['suite_means'].items()}, evaluation_index=p['evaluation_index'],
                              selection_labels=p['selection_labels'], selection_scores=p['selection_scores'],
                              template_sha256=p['template_sha256']), ensure_ascii=False))


if __name__ == '__main__':
    main()
