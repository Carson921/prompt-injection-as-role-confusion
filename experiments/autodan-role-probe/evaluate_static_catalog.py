"""Score an audited finite catalog of pre-existing templates, without mutation.

Every fitness is the original native complete-six-context P(user). Candidates
are resolved back to their source JSON; this runner never generates text.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from precise_scorer import PreciseScorer
from scorer import DEFAULT_MODEL, DEFAULT_PROBE, ROLES
import search

HERE = Path(__file__).resolve().parent
CAPS = (256, 512, 1024, 2048, 4096, 8192, 16384)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def pointer_value(value, pointer):
    if pointer:
        if not pointer.startswith('/'):
            raise ValueError('Catalog source requires a JSON pointer')
        for part in pointer[1:].split('/'):
            key = part.replace('~1', '/').replace('~0', '~')
            value = value[int(key)] if isinstance(value, list) else value[key]
    return value.get('template') if isinstance(value, dict) else value


def load_catalog(path, tokenizer, library):
    data = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    source_cache, entries = {}, {}
    for row in data:
        source = Path(row['source_path']).resolve()
        expected = row['source_sha256']
        if source not in source_cache:
            if file_sha(source) != expected:
                raise ValueError('A fixed catalog source changed')
            source_cache[source] = json.loads(source.read_text())
        original = pointer_value(source_cache[source], row['json_pointer'])
        text = row.get('template', original)
        if not isinstance(text, str) or text != original:
            raise ValueError('Candidate is not byte-identical to pre-existing source text')
        identity = raw_sha(text)
        if identity != row.get('raw_sha256', row.get('template_sha256', identity)):
            raise ValueError('Catalog template identity changed')
        policy = validate_diverse_template(text, tokenizer, library, maximum=CAPS[-1])
        if not policy['allowed']:
            continue
        if identity not in entries:
            entries[identity] = dict(row, template=text, raw_sha256=identity,
                                    wrapper_tokens=policy['metrics']['wrapper_tokens'],
                                    family=row.get('family', source.stem), policy=policy)
    return entries


def verify_complete(result, contexts):
    rows = result.get('per_case', [])
    expected = [row['case_id'] for row in contexts]
    if (result.get('status') != 'complete' or result.get('complete') is not True
            or result.get('target_role') != 'user' or result.get('target_role_index') != 2
            or len(rows) != 6 or [row.get('case_id') for row in rows] != expected):
        raise ValueError('Fitness requires the complete fixed six cases, actual user class2')
    values = []
    for row, context in zip(rows, contexts):
        value = row['probabilities']['goal']['user']
        if (not math.isfinite(value) or not 0 <= value <= 1
                or row.get('target_role_index') != 2 or row.get('target_role') != 'user'
                or row.get('goal_user_mean') != value
                or row.get('actual_goal_sha256') != raw_sha(context['actual_goal'])):
            raise ValueError('Incomplete or altered goal/user-role evidence')
        values.append(value)
    mean = math.fsum(values) / 6
    if abs(mean - result['fitness']) > 1e-12 or abs(mean - result['goal_role_means']['user']) > 1e-12:
        raise ValueError('Mean P(user) is not the equal-six complete native mean')
    return mean


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def select_candidate(available, records, rng):
    """Family exploration ranks existing entries only; no predicted fitness."""
    groups = defaultdict(list)
    for row in available:
        groups[row['family']].append(row)
    stats = defaultdict(list)
    for row in records.values():
        stats[row.get('family', 'legacy')].append(row['fitness'])
    total = max(2, len(records))
    def priority(family):
        values = stats[family]
        quality = (.5 * max(values) + .5 * math.fsum(values) / len(values)) if values else 0.
        return quality + .18 * math.sqrt(math.log(total + 1) / (1 + len(values)))
    best = max(groups, key=lambda family: (priority(family), rng.random()))
    return rng.choice(groups[best])


def save_leaderboard(output, records, phase):
    ordered = sorted(records.values(), key=lambda row: (-row['fitness'], row['raw_sha256']))
    if not ordered:
        return
    best = ordered[0]
    search.write(output / 'winner.json', dict(best, optimization_target='maximize complete-six mean P(user)',
                 fixed_catalog_only=True, phase=phase, updated_unix=time.time()))
    (output / 'optimized_template.txt').write_text(best['template'])
    bins = (0., .05, .1, .2, .3, .4, .5, .65, .8, 1.0000000001)
    interval_best = []
    for low, high in zip(bins, bins[1:]):
        candidates = [row for row in ordered if low <= row['fitness'] < high]
        if candidates:
            row = candidates[0]
            folder = output / 'templates' / row['raw_sha256'][:24]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'template.txt').write_text(row['template'])
            search.write(folder / 'metadata.json', row)
            interval_best.append(dict(template=row['template'], score_path=row['source_score_path'],
                raw_sha256=row['raw_sha256'], fitness=row['fitness'], wrapper_tokens=row['wrapper_tokens'],
                p_user_interval=[low, min(high, 1.)]))
    search.write(output / 'interval_candidates.json', dict(candidates=interval_best,
                 selection_metric='actual full-six P(user), no ASR used for selection'))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    budget = json.loads(args.budget.read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    library = BroadDelimiterLibrary(args.model)
    catalog = load_catalog(args.catalog, tokenizer, library)
    if not catalog:
        raise ValueError('No unchanged policy-compliant catalog entries')
    contexts = json.loads(args.dataset.read_text())['contexts']
    sources = [Path(__file__), HERE / 'diverse_policy.py', HERE / 'precise_scorer.py',
               HERE / 'scorer.py', HERE / 'deterministic_spans.py', HERE / 'broad_validation.py',
               HERE / 'near_delimiters.py']
    config = dict(schema='static-catalog-native-probe-v1', sources={str(p.resolve()): file_sha(p) for p in sources},
        catalog_sha256=file_sha(args.catalog), legacy_cache_index_sha256=file_sha(args.cache_index),
        extra_catalog_paths=[str(path.resolve()) for path in args.extra_catalog],
        extra_catalog_policy='immutable indexed chunks; each original source JSON and literal text independently matched',
        dataset_sha256=file_sha(args.dataset), probe_sha256=file_sha(args.probe),
        model_path=str(args.model), caps=list(CAPS), seed=args.seed,
        plateau_fresh_candidates=96, plateau_minimum_seconds=1200,
        probe_deadline_unix=budget['probe_deadline_unix'],
        no_new_text_generation=True, no_API_generation=True, full_six_scores_only=True,
        genuine_special_tokens_allowed=False, no_parent_repetition_exemption=True,
        known_unimplemented_branch='New wrapper generator rejected by automatic safety review')
    protocol_path = output / 'run_protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != config:
            raise ValueError('Run protocol changed; do not overwrite prior experiment')
    else:
        search.write(protocol_path, config)
        for path in sources:
            target = output / 'provenance' / path.name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(path.read_bytes())
    records = {row['raw_sha256']: row for row in read_lines(output / 'score_records.jsonl')}
    rejected = {row['raw_sha256'] for row in read_lines(output / 'rejections.jsonl')}
    for entry in json.loads(args.cache_index.read_text())['entries']:
        path = Path(entry.get('source_score_path', entry['source_path']))
        if file_sha(path) != entry['source_sha256']:
            raise ValueError('Complete cached score changed')
        result = json.loads(path.read_text())
        identity = raw_sha(result['template'])
        policy = validate_diverse_template(result['template'], tokenizer, library, maximum=CAPS[-1])
        if not policy['allowed'] or identity in records:
            continue
        fitness = verify_complete(result, contexts)
        for case in result['per_case']:
            for name, expected in case['artifact_sha256'].items():
                if file_sha(Path(case['token_artifact']).parent / name) != expected:
                    raise ValueError('Native cache artifacts changed')
        row = dict(result, raw_sha256=identity, wrapper_tokens=policy['metrics']['wrapper_tokens'],
                   source_score_path=str(path), source_score_sha256=file_sha(path), family='legacy',
                   origin='verified_complete_legacy', policy=policy, fitness=fitness)
        search.append(output / 'score_records.jsonl', row)
        records[identity] = row
    state_path = output / 'checkpoint.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    phase = state.get('phase_index', 0)
    phase_count = state.get('phase_fresh_scores', 0)
    since_improvement = state.get('fresh_since_improvement', 0)
    last_improvement = state.get('last_improvement_unix', time.time())
    rng = random.Random(args.seed + len(records))
    scorer = None
    stop = None
    ingested = {row['path']: row['sha256'] for row in read_lines(output / 'catalog_ingestions.jsonl')}
    for path in args.extra_catalog:
        absolute = str(path.resolve())
        if absolute in ingested:
            if file_sha(path) != ingested[absolute]:
                raise ValueError('An ingested catalog chunk changed')
            catalog.update(load_catalog(path, tokenizer, library))
    def checkpoint(status):
        search.write(state_path, dict(status=status, phase_index=phase, cap=CAPS[phase],
            complete_scores=len(records), fresh_scores=sum(row.get('origin') == 'fresh_fixed_catalog' for row in records.values()),
            phase_fresh_scores=phase_count, fresh_since_improvement=since_improvement,
            last_improvement_unix=last_improvement, updated_unix=time.time(),
            best_fitness=max((row['fitness'] for row in records.values()), default=None)))
    save_leaderboard(output, records, phase)
    try:
        scorer = PreciseScorer(args.dataset, output / 'native', model_path=args.model,
                               probe_path=args.probe, target_role='user')
        while time.time() < budget['probe_deadline_unix']:
            for path in args.extra_catalog:
                absolute = str(path.resolve())
                if absolute not in ingested and path.exists():
                    digest = file_sha(path)
                    extra = load_catalog(path, tokenizer, library)
                    catalog.update(extra)
                    search.append(output / 'catalog_ingestions.jsonl', dict(path=absolute,
                        sha256=digest, accepted=len(extra), unix=time.time(), generated_text=False))
                    ingested[absolute] = digest
            cap = CAPS[phase]
            available = [row for key, row in catalog.items()
                         if key not in records and key not in rejected and row['wrapper_tokens'] <= cap]
            plateau = (phase_count >= 96 and since_improvement >= 96
                       and time.time() - last_improvement >= 1200)
            if not available or plateau:
                if phase < len(CAPS) - 1:
                    search.append(output / 'phase_transitions.jsonl', dict(from_cap=cap, to_cap=CAPS[phase+1],
                        reason='catalog_exhausted_within_cap' if not available else 'measured_plateau',
                        phase_fresh_scores=phase_count, fresh_since_improvement=since_improvement,
                        seconds_since_improvement=time.time()-last_improvement, unix=time.time()))
                    phase += 1
                    phase_count, since_improvement, last_improvement = 0, 0, time.time()
                    checkpoint('scoring')
                    continue
                if not available:
                    stop = 'audited_finite_catalog_exhausted'
                    break
            candidate = select_candidate(available, records, rng)
            identity, text = candidate['raw_sha256'], candidate['template']
            policy = validate_diverse_template(text, tokenizer, library, maximum=cap)
            if not policy['allowed']:
                raise ValueError('Frozen candidate violates current admission policy')
            try:
                for context in scorer.contexts:
                    scorer.prepare(context, text)
            except Exception as error:
                search.append(output / 'rejections.jsonl', dict(raw_sha256=identity,
                    source_path=candidate['source_path'], reason='full_six_context_prepare',
                    error=f'{type(error).__name__}: {error}', fitness=None, native_forwards=0))
                rejected.add(identity)
                continue
            prior = max((row['fitness'] for row in records.values() if row['wrapper_tokens'] <= cap), default=0.)
            started = time.monotonic()
            result = scorer.score(text)
            fitness = verify_complete(result, contexts)
            path = Path(result['artifact'])
            row = dict(result, raw_sha256=identity, wrapper_tokens=candidate['wrapper_tokens'],
                source_score_path=str(path.resolve()), source_score_sha256=file_sha(path),
                origin='fresh_fixed_catalog', family=candidate['family'], policy=policy,
                original_catalog_source={k:candidate[k] for k in ('source_path','source_sha256','json_pointer')},
                evaluated_cap=cap, completed_unix=time.time())
            search.append(output / 'score_records.jsonl', row)
            records[identity] = row
            phase_count += 1
            since_improvement += 1
            if fitness > prior + .002:
                since_improvement, last_improvement = 0, time.time()
            save_leaderboard(output, records, phase)
            checkpoint('scoring')
            print(json.dumps(dict(event='complete_native_six_score', fresh=phase_count,
                total_complete=len(records), cap=cap, wrapper_tokens=row['wrapper_tokens'],
                fitness=fitness, best=max(r['fitness'] for r in records.values()),
                elapsed_seconds=time.monotonic()-started, raw_sha256=identity)), flush=True)
        if stop is None:
            stop = 'ASR_reserved_deadline'
    except KeyboardInterrupt:
        stop = 'interrupted'
    finally:
        if scorer is not None:
            scorer.close()
        checkpoint(stop or 'error')
        save_leaderboard(output, records, phase)
    return stop


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','cache-index','dataset','output','budget'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--probe', type=Path, default=DEFAULT_PROBE)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--extra-catalog', type=Path, action='append', default=[])
    return parser.parse_args()


if __name__ == '__main__':
    run(arguments())
