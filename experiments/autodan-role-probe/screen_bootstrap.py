"""Initial wide screening, using the campaign's shared frozen native cache."""
import argparse
import json
from pathlib import Path
import random
import shutil
import time

from advanced_templates import generate_seeds
from near_delimiters import NearDelimiterLibrary
from scorer import Scorer, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/tmp/wz55191-agentdojo-first-tool-model'))
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=48)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    destination = args.output / 'dataset.json'
    if destination.exists():
        assert destination.read_bytes() == args.dataset.read_bytes()
    else:
        shutil.copyfile(args.dataset, destination)
    scorer = Scorer(destination, args.output, args.model, args.probe, target_role='user')
    library = NearDelimiterLibrary(args.model)
    candidates = []
    bootstrap = json.loads((args.output / 'bootstrap_templates.json').read_text())['templates']
    candidates.extend(dict(row, origin='historical_user_ranked') for row in bootstrap[:4])
    seen_families = set()
    for max_tokens in (512, 1024, 2048):
        count = 0
        for record in generate_seeds(scorer.tokenizer, max_tokens, library=library, rng=random.Random(20261104)):
            family = record['family']
            if family == 'historical_user_ranked':
                continue
            key = (max_tokens, family)
            if key not in seen_families:
                seen_families.add(key)
                candidates.append(dict(record, origin='structured_initial_screen', length_cap=max_tokens))
                count += 1
            if count >= 8:
                break
    embedding = json.loads((args.output / 'embedding_seed_templates.json').read_text())['templates']
    candidates.extend(dict(row, origin='embedding_neighbor_initial_screen')
                      for row in embedding[::max(1, len(embedding) // 16)][:16])
    unique, seen = [], set()
    for row in candidates:
        text = row['template']
        if text not in seen and library.validate(text, tokenizer=scorer.tokenizer):
            seen.add(text)
            unique.append(row)
    write(args.output / 'bootstrap_screen_plan.json', dict(candidates=unique[:args.limit],
          target_role='user', metric='equal six-task mean of complete goal token P(user)',
          no_api_calls=True, initial_seed=20261104))
    try:
        for index, row in enumerate(unique[:args.limit]):
            template = row['template']
            try:
                # Reject serialization/offset problems before any GPU forward.
                for context in scorer.contexts:
                    scorer.prepare(context, template)
                result = scorer.score(template)
                event = dict(index=index, origin=row['origin'], family=row.get('family'),
                             fitness=result['fitness'], template=template,
                             per_case=[x['goal_user_mean'] for x in result['per_case']],
                             artifact=result['artifact'], cache_hit=result['cache_hit'],
                             time=time.time())
            except ValueError as exc:
                event = dict(index=index, origin=row['origin'], status='rejected', error=str(exc))
            with (args.output / 'bootstrap_screen.jsonl').open('a') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            print(json.dumps(event, ensure_ascii=False), flush=True)
    finally:
        scorer.close()


if __name__ == '__main__':
    main()
