"""Exercise original AutoDAN GA/HGA and all content-policy admission paths."""
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
from transformers import AutoTokenizer
import content_campaign as c
import content_mutations as m
from content_policy import validate_content_template


def main():
    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(c.DEFAULT_MODEL, local_files_only=True)
    lib = c.NearDelimiterLibrary(c.DEFAULT_MODEL)
    cache = {}
    for path in (c.HERE / 'outputs/user90-precise-20261104/scores').glob('*/score.json'):
        row = json.loads(path.read_text())
        text = row['template']
        count = c.tokens(tok, text)
        if count > 3165:
            continue
        row.update(template_sha256=c.search.digest(text), wrapper_tokens=count)
        cache[row['template_sha256']] = row
    class NoAPI:
        key = ''
        calls = 0
        target_role = 'user'
        def mutate(self, offspring, *args, **kwargs):
            return offspring
    api = NoAPI()
    operators, provenance = c.search.load_autodan(c.search.AUTODAN, api)
    c.install_empty_join_guard(operators, provenance)
    args = SimpleNamespace(population=16, batch_size=8, breadth_per_stage=4,
                           selection_temperature=.1, ga_every=2, crossover=.65,
                           mutation=0, num_points=5, advanced_mutation_rate=.85,
                           api_every=2, api_immigrants=2)
    seeds = m.generate_seeds(tok, 3165, random.Random(20261105), lib)
    phase = dict(max_wrapper_tokens=3165, evaluated=0, batch_index=0)
    cursor, words, origins = 0, {}, []
    for _ in range(2):
        rows, cursor, words = c.propose_batch(args, phase, cache, [], seeds,
            cursor, m, tok, lib, api, operators, words,
            random.Random(12 + phase['batch_index']),
            lambda text: c.valid_template(text, tok, 1, 3165, lib))
        assert len(rows) == 8
        for row in rows:
            assert validate_content_template(row['template'], tok, lib,
                parent=row['metadata'].get('parent_template'), hard_cap=3165)['allowed']
        origins.extend(row['origin'] for row in rows)
        phase['evaluated'] += len(rows)
    report = dict(passed=True, ga_and_hga_batches=2, accepted_checked=len(origins),
                  origins=origins, import_eligible_native_parents=len(cache),
                  api_calls=0, backbone_loaded=False,
                  elapsed_seconds=time.monotonic() - start)
    (c.HERE / 'outputs/user90-content-20261105/content_driver_preflight.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
