"""Independent, read-only CPU audit of DeepSeek experiment counts and probe aggregates.

Prints only identifiers, counts, probabilities and audit metadata. JSON readers discard
free text and model transcripts. No experiment module, model runtime, GPU or API is used.
Missing or still-running outputs are pending, not successful. Exit 1 means a failure;
with --require-complete, pending also exits 2. Redirect stdout to save an audit artifact.
--asr-only-current-best audits the single immutable user-stop choice and explicitly
records the additional native probe passes as skipped by the user's ASR-only request.
"""
from __future__ import annotations

import argparse
import array
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

ROLES = ('system', 'user', 'cot', 'assistant', 'tool')
SUITE_SHAPES = {'banking': (16, 9), 'slack': (21, 5), 'travel': (20, 7)}
TEXT_FIELDS = {'template', 'goal', 'payload', 'rendered_prompt', 'description', 'content',
               'reasoning', 'reasoning_content', 'final_response', 'completion', 'error', 'traceback'}
TRANSCRIPT_FIELDS = {'messages', 'transcripts', 'calls', 'steps', 'tool_calls_made'}


def metadata_only(row):
    if isinstance(row.get('template'), str):
        row['template_sha256_observed'] = hashlib.sha256(row['template'].encode()).hexdigest()
    return {k: v for k, v in row.items()
            if k not in TRANSCRIPT_FIELDS and not (k in TEXT_FIELDS and isinstance(v, str))}


def read(path, identity_base=None):
    def discard_text(row):
        if identity_base is not None and isinstance(row.get('template'), str):
            # Compute the runner's identity while parsing, then immediately discard
            # the template string as with every other reader in this audit.
            row['asr_identity_sha256_observed'] = hashlib.sha256(json.dumps(
                dict(identity_base, template=row['template']), sort_keys=True).encode()).hexdigest()
        return metadata_only(row)
    return json.loads(Path(path).read_text(), object_hook=discard_text)


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def close(a, b):
    return a == b if a is None or b is None else math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8)


def probability(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def counts(records, expected, success_key):
    complete = [r for r in records if r['status'] == 'complete']
    errors = sum(r['status'] == 'error' for r in records)
    successes = sum(bool(r[success_key]) for r in complete)
    return dict(expected=len(expected), complete=len(complete), errors=errors,
                running=sum(r['status'] not in ('complete', 'error') for r in records),
                missing=len(expected) - len(records), successes=successes,
                utility_successes=sum(bool(r.get('utility')) for r in complete),
                asr_scored=successes / len(complete) if complete else None,
                asr_lower=successes / len(expected) if expected else None,
                asr_upper=(successes + len(expected) - len(complete)) / len(expected) if expected else None,
                utility_scored=mean(int(bool(r.get('utility'))) for r in complete))


def npy(archive, name):
    """Read the numeric .npy types emitted by np.savez_compressed using stdlib only."""
    raw = archive.read(name + '.npy')
    if raw[:6] != b'\x93NUMPY':
        raise ValueError('invalid NPY magic')
    version = raw[6]
    size, start = (struct.unpack_from('<H', raw, 8)[0], 10) if version == 1 else (struct.unpack_from('<I', raw, 8)[0], 12)
    header = ast.literal_eval(raw[start:start + size].decode('latin1'))
    if header['fortran_order']:
        raise ValueError('unexpected Fortran-order array')
    dtype = header['descr']
    code = {'f4': 'f', 'f8': 'd', 'b1': 'b', 'i8': 'q', 'i4': 'i'}.get(dtype[1:])
    if code is None:
        raise ValueError('unsupported numeric dtype')
    data = array.array(code)
    data.frombytes(raw[start + size:])
    if dtype[0] in '<>' and ((dtype[0] == '<') != (sys.byteorder == 'little')):
        data.byteswap()
    if len(data) != math.prod(header['shape']):
        raise ValueError('NPY shape differs from numeric data size')
    return tuple(header['shape']), data


class Audit:
    def __init__(self):
        self.checks, self.metrics = [], {}
        self.agentdojo_cases = set()
        self.panel_groups = {}

    def check(self, name, ok, actual=None, expected=None, pending=False):
        row = dict(name=name, status='pass' if ok else ('pending' if pending else 'fail'))
        if actual is not None:
            row['actual'] = json_safe(actual)
        if expected is not None:
            row['expected'] = json_safe(expected)
        self.checks.append(row)

    def optional(self, path, name):
        self.check(name, path.exists(), pending=True)
        return read(path) if path.exists() else None

    def skip_asr_only(self, name):
        self.checks.append(dict(name=name, status='skipped-by-user-ASR-only',
                                reason='User stopped optimization and requested ASR of the current complete best only; no additional native probe pass.'))

    def current_best(self, selection, snapshot_path, ledger_path, panel):
        self.check('selection.current_best_only', set(selection or {}) == {'opt_top1'})
        self.check('user_stop_snapshot.exists', snapshot_path.exists())
        if not snapshot_path.exists():
            return
        snapshot = read(snapshot_path)
        self.check('user_stop_snapshot.schema', snapshot.get('schema') == 'deepseek-user-stop-v1')
        best = snapshot.get('best', {})
        self.check('user_stop_snapshot.best_complete', best.get('status') == 'complete')
        if best.get('status') != 'complete':
            return
        best_hash = best.get('template_sha256_observed')
        self.check('user_stop_snapshot.best_template_hash', isinstance(best_hash, str) and len(best_hash) == 64)
        self.check('user_stop_snapshot.best_evaluation', type(best.get('evaluation_index')) is int)
        self.objective('user_stop_snapshot.best', best, panel)
        wanted = snapshot.get('completed_at_snapshot')
        self.check('user_stop_snapshot.completed_count', type(wanted) is int and wanted > 0)
        self.check('user_stop_snapshot.ledger_exists', ledger_path.exists())
        prefix, completed = bytearray(), []
        if ledger_path.exists() and type(wanted) is int and wanted > 0:
            for line in ledger_path.read_bytes().splitlines(keepends=True):
                prefix.extend(line)
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, object_hook=metadata_only)
                except json.JSONDecodeError:
                    self.check('user_stop_snapshot.ledger_valid_json', False)
                    break
                if row.get('status') == 'complete':
                    completed.append(row)
                if len(completed) == wanted:
                    break
        self.same('user_stop_snapshot.ledger_count', len(completed), wanted)
        self.check('user_stop_snapshot.ledger_sha256',
                   hashlib.sha256(prefix).hexdigest() == snapshot.get('ledger_snapshot_sha256'))
        rows = [r for r in completed if r.get('evaluation_index') == best.get('evaluation_index')]
        self.check('user_stop_snapshot.best_ledger_identity', len(rows) == 1 and
                   rows[0].get('template_sha256_observed') == best.get('template_sha256_observed') and
                   rows[0].get('fitness') == best.get('fitness'))
        if completed:
            self.same('user_stop_snapshot.best_at_stop', best.get('fitness'), max(r['fitness'] for r in completed))
        spec = (selection or {}).get('opt_top1')
        if spec is not None:
            self.check('selection.current_best_hash',
                       spec.get('template_sha256') == spec.get('template_sha256_observed') == best.get('template_sha256_observed'))
            self.check('selection.current_best_evaluation', spec.get('evaluation_index') == best.get('evaluation_index'))
            self.check('selection.current_best_score', spec.get('p_user_optimization') == best.get('fitness'))
        return best

    def reused_best_score(self, row, best):
        """The retained stop-time measurement is audited even without a new pass."""
        values = row.get('per_case', {})
        if isinstance(values, list):
            values = {r['case_id']: r['goal_user_mean'] for r in values}
        expected = {r['case_id']: r['goal_user_mean'] for r in best['per_case']}
        for name, actual, reference in (
            ('per_case', values, expected), ('suite_means', row.get('suite_means'), best['suite_means']),
            ('role_means', row.get('goal_role_means'), best['goal_role_means']),
            ('fitness', row.get('p_user', row.get('fitness')), best['fitness']),
            ('template_hash', row.get('template_sha256_observed'), best.get('template_sha256_observed'))):
            self.check('final_condition_scores.opt_top1.snapshot_' + name, actual == reference)

    def same(self, name, actual, expected):
        try:
            ok = close(actual, expected)
        except (TypeError, ValueError):
            ok = False
        self.check(name, ok, actual, expected)

    def vector(self, name, vector):
        self.check(name, set(vector) == set(ROLES) and all(probability(v) for v in vector.values())
                   and close(math.fsum(vector.values()), 1), expected='finite normalized five-role probabilities')

    def asr_bindings(self, label, directory, benchmark, expected_specs):
        """Bind actual ASR records to the selected conditions without retaining text."""
        conditions_path = directory / benchmark / 'conditions.json'
        protocol_path = directory / 'ipi/protocol.json'
        if not conditions_path.exists() or not protocol_path.exists():
            return  # The ordinary benchmark audit already marks these as pending.
        protocol = read(protocol_path)
        identity_base = None
        if benchmark == 'agentdojo':
            identity_base = dict(benchmark_version='v1.2.1', generation=protocol['generation'],
                                 codec=protocol['codec'], pipeline='deepseek_v4_flash_vllm',
                                 max_model_len=protocol['max_model_len'])
        conditions = read(conditions_path, identity_base)
        for name, expected in expected_specs.items():
            actual = conditions.get(name, {})
            self.check(label + '.' + name + '.condition_template_hash',
                       actual.get('template_sha256_observed') == expected.get('template_sha256_observed')
                       and isinstance(expected.get('template_sha256_observed'), str))
            self.check(label + '.' + name + '.condition_special_policy',
                       bool(actual.get('allow_special')) == bool(expected.get('allow_special')))
            if benchmark == 'agentdojo':
                rows = [read(p) for p in (directory / benchmark / 'cases' / name).glob('*/*/*.json')]
                self.check(label + '.' + name + '.record_identity_hash', all(
                    r.get('identity_sha256') == actual.get('asr_identity_sha256_observed') and
                    r.get('case_id') == name + '/' + r['base_case_id'] for r in rows))
        if benchmark != 'ipi':
            return
        identity_path = directory / 'ipi/asr_identity.json'
        identity = self.optional(identity_path, label + '.asr_identity')
        if identity is None:
            return
        settings = {k: v for k, v in identity.items() if k not in ('identity_sha256', 'judge')}
        identity_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        self.check(label + '.asr_identity.self_hash', identity_hash == identity.get('identity_sha256'))
        self.check(label + '.asr_identity.protocol_hash',
                   identity.get('protocol_sha256') == hashlib.sha256(protocol_path.read_bytes()).hexdigest())
        for entry in protocol['rows']:
            if entry['condition'] not in expected_specs:
                continue
            prefix = label + '.' + entry['condition'] + '.' + entry['case_id']
            request = directory / 'ipi' / entry['request']
            self.check(prefix + '.request_file', request.exists(), pending=True)
            if request.exists():
                self.check(prefix + '.request_hash', hashlib.sha256(request.read_bytes()).hexdigest() == entry['request_sha256'])
            path = directory / 'ipi/cases' / entry['condition'] / (entry['case_id'] + '.json')
            if path.exists():
                row = read(path)
                self.check(prefix + '.record_hashes', row.get('identity_sha256') == identity_hash and
                           row.get('request_sha256') == entry['request_sha256'])

    def objective(self, label, row, panel):
        records = row['per_case']
        values = records if isinstance(records, dict) else {r['case_id']: r['goal_user_mean'] for r in records}
        self.check(label + '.case_ids', set(values) == panel and len(records) == len(panel),
                   dict(count=len(records), unique=len(values)), dict(count=len(panel)))
        self.check(label + '.finite', all(probability(v) for v in values.values()))
        if values and all(probability(v) for v in values.values()):
            self.same(label + '.equal_case_mean', row.get('p_user', row.get('fitness')), mean(values.values()))
            for group in set(self.panel_groups.values()):
                self.same(label + '.group_mean.' + group, row['suite_means'].get(group),
                          mean(v for k, v in values.items() if self.panel_groups.get(k) == group))
        if 'goal_role_means' in row:
            self.vector(label + '.role_means', row['goal_role_means'])
            self.same(label + '.role_user', row.get('p_user', row.get('fitness')), row['goal_role_means']['user'])
        if isinstance(records, list):
            for item in records:
                if 'probabilities' in item:
                    for region, vector in item['probabilities'].items():
                        if vector is not None:
                            self.vector(label + '.' + item['case_id'] + '.' + region, vector)
                    self.same(label + '.' + item['case_id'] + '.user_mean', item['goal_user_mean'],
                              item['probabilities']['goal']['user'])
            if all('probabilities' in item for item in records):
                for role in ROLES:
                    self.same(label + '.role_aggregate.' + role, row['goal_role_means'][role],
                              mean(item['probabilities']['goal'][role] for item in records))
        return values

    def benchmark(self, label, directory, benchmark, names, excluded):
        summary = self.optional(directory / benchmark / 'summary.json', label + '.summary')
        protocol = read(directory / 'ipi/protocol.json') if (directory / 'ipi/protocol.json').exists() else None
        if benchmark == 'ipi' and protocol is None:
            self.check(label + '.protocol', False, pending=True)
            return
        conditions_path = directory / benchmark / 'conditions.json'
        if conditions_path.exists():
            self.check(label + '.conditions', set(read(conditions_path)) == names)
        else:
            self.check(label + '.conditions', False, pending=True)
        for name in sorted(names):
            if benchmark == 'agentdojo':
                expected = self.agentdojo_cases
                primary = expected
                files = list((directory / benchmark / 'cases' / name).glob('*/*/*.json'))
            else:
                plan = [r for r in protocol['rows'] if r['condition'] == name]
                expected, primary = {r['case_id'] for r in plan}, {r['case_id'] for r in plan if r['primary']}
                self.check(label + '.' + name + '.plan', len(expected) == 38 and len(primary) == 36,
                           dict(all=len(expected), primary=len(primary)), dict(all=38, primary=36))
                files = list((directory / benchmark / 'cases' / name).glob('*.json'))
            rows = [read(p) for p in files]
            id_key, success_key = ('base_case_id', 'security') if benchmark == 'agentdojo' else ('case_id', 'success')
            ids = [r[id_key] for r in rows]
            self.check(label + '.' + name + '.identities', len(ids) == len(set(ids)) and set(ids) <= expected
                       and all(r['condition'] == name for r in rows))
            self.check(label + '.' + name + '.complete_boolean_scores', all(
                type(r.get(success_key)) is bool and (benchmark != 'agentdojo' or type(r.get('utility')) is bool)
                for r in rows if r['status'] == 'complete'))
            if benchmark == 'ipi':
                self.check(label + '.' + name + '.primary_flags', all(r['primary'] == (r[id_key] in primary) for r in rows))
            scored_rows = [r for r in rows if r[id_key] in primary]
            result = counts(scored_rows, primary, success_key)
            heldout = counts([r for r in scored_rows if r[id_key] not in excluded], primary - excluded, success_key)
            digest = hashlib.sha256()
            for p in sorted(files):
                digest.update(str(p.relative_to(directory)).encode())
                digest.update(hashlib.sha256(p.read_bytes()).digest())
            key = label + '.' + name
            self.metrics[key] = dict(full=result, heldout=heldout, records=len(rows), raw_records_sha256=digest.hexdigest())
            terminal = len(rows) == len(expected) and not result['running']
            self.check(key + '.records', terminal, dict(records=len(rows), running=result['running']),
                       dict(records=len(expected), running=0), pending=len(rows) < len(expected) or bool(result['running']))
            if not terminal or not summary:
                continue
            stored = summary['conditions'].get(name)
            self.check(key + '.summary_condition', stored is not None)
            if stored is None:
                continue
            keys = {'expected': 'expected', 'scored': 'complete', 'errors': 'errors', 'successes': 'successes',
                    'utility_successes': 'utility_successes', 'asr_scored': 'asr_scored', 'utility_scored': 'utility_scored'} if benchmark == 'agentdojo' else {
                        'scheduled': 'expected', 'complete': 'complete', 'errors': 'errors', 'successes': 'successes',
                        'asr': 'asr_scored', 'asr_over_scheduled': 'asr_lower'}
            for stored_key, result_key in keys.items():
                self.same(key + '.summary.' + stored_key, stored.get(stored_key), result[result_key])
            if benchmark == 'agentdojo':
                self.same(key + '.summary.asr', stored.get('asr'), result['asr_lower'] if result['complete'] == result['expected'] else None)
            else:
                supplementary = [r for r in rows if r[id_key] not in primary and r['status'] == 'complete']
                self.same(key + '.summary.supplementary.complete', stored['supplementary']['complete'], len(supplementary))
                self.same(key + '.summary.supplementary.successes', stored['supplementary']['successes'], sum(r['success'] for r in supplementary))

    def ipi_probe(self, label, directory, names):
        protocol = self.optional(directory / 'protocol.json', label + '.protocol')
        summary = self.optional(directory / 'probe_summary.json', label + '.summary')
        if not protocol:
            return
        for name in sorted(names):
            plan = [r for r in protocol['rows'] if r['condition'] == name]
            rows = []
            for entry in plan:
                path = directory / 'probe' / name / (entry['case_id'] + '.json')
                if not path.exists():
                    continue
                row = read(path)
                rows.append(row)
                prefix = label + '.' + name + '.' + entry['case_id']
                self.check(prefix + '.identity', row['case_id'] == entry['case_id'] and row['primary'] == entry['primary']
                           and row['request_sha256'] == entry['request_sha256'] and row['status'] == 'complete')
                self.vector(prefix + '.goal', row['region_means']['goal'])
                self.same(prefix + '.goal_user', row['goal_user_mean'], row['region_means']['goal']['user'])
                archive_path = path.with_suffix('.npz')
                self.check(prefix + '.archive', archive_path.exists(), pending=True)
                if archive_path.exists():
                    try:
                        with zipfile.ZipFile(archive_path) as archive:
                            shape, probabilities = npy(archive, 'probabilities')
                            mask_shape, mask = npy(archive, 'goal_mask')
                            shape_valid = shape == (len(mask), 5) and mask_shape == (len(mask),)
                            self.check(prefix + '.archive_shape', shape_valid)
                            if not shape_valid:
                                continue
                            valid, goal_rows = True, []
                            for i in range(len(mask)):
                                vector = probabilities[i * 5:(i + 1) * 5]
                                valid &= all(probability(v) for v in vector) and abs(math.fsum(vector) - 1) < 1e-5
                                if mask[i]:
                                    goal_rows.append(vector)
                            self.check(prefix + '.archive_probabilities', valid and bool(goal_rows))
                            self.same(prefix + '.archive_goal_tokens', len(goal_rows), row['goal_tokens'])
                            if goal_rows:
                                self.same(prefix + '.archive_goal_user', mean(v[1] for v in goal_rows), row['goal_user_mean'])
                    except (ValueError, KeyError, EOFError, OSError, zipfile.BadZipFile):
                        self.check(prefix + '.archive_read', False)
            self.check(label + '.' + name + '.records', len(rows) == 38, len(rows), 38, pending=len(rows) < 38)
            if len(rows) == 38 and summary:
                primary = [r for r in rows if r['primary']]
                stored = summary.get(name, {})
                self.same(label + '.' + name + '.summary.cases', stored.get('cases'), 36)
                self.same(label + '.' + name + '.summary.mean', stored.get('p_user_mean'), mean(r['goal_user_mean'] for r in primary))
                for role in ROLES:
                    self.same(label + '.' + name + '.summary.role.' + role, stored.get('goal_role_means', {}).get(role),
                              mean(r['region_means']['goal'][role] for r in primary))
                for category in ('browser', 'coding', 'tool'):
                    self.same(label + '.' + name + '.summary.category.' + category, stored.get('per_category', {}).get(category),
                              mean(r['goal_user_mean'] for r in primary if r['category'] == category))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent / 'outputs'
    parser.add_argument('--baseline', type=Path, default=base / 'baseline-20261118')
    parser.add_argument('--optimized', type=Path, default=base / 'optimized-20261118')
    parser.add_argument('--evolution', type=Path, default=base / 'dsevo-20261118')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--asr-only-current-best', action='store_true',
                        help='Audit only opt_top1 bound to user_stop_snapshot.json; explicitly skip extra native probes')
    args = parser.parse_args()
    audit = Audit()
    dataset = read(args.evolution / 'dataset.json')
    ad_panel, ipi_panel = {r['case_id'] for r in dataset['contexts']}, set(dataset['ipi_cases'])
    panel = ad_panel | ipi_panel
    audit.panel_groups = {r['case_id']: r['suite'] for r in dataset['contexts']}
    ipi_protocol = read(args.baseline / 'ipi/protocol.json')
    audit.panel_groups.update({r['case_id']: r['category'] for r in ipi_protocol['rows'] if r['case_id'] in ipi_panel})
    audit.check('optimization_panel', len(ad_panel) == 6 and len(ipi_panel) == 6 and len(panel) == 12,
                dict(agentdojo=len(ad_panel), ipi=len(ipi_panel), unique=len(panel)))
    baseline_ad_specs = read(args.baseline / 'conditions_agentdojo.json')
    baseline_ipi_specs = read(args.baseline / 'conditions_ipi.json')
    baseline_ad, baseline_ipi = set(baseline_ad_specs), set(baseline_ipi_specs)
    # Native task ids need not be contiguous integers. Derive their names from the
    # completed original baseline, then independently require each full Cartesian grid.
    reference = args.baseline / 'agentdojo/cases/original'
    for suite, shape in SUITE_SHAPES.items():
        paths = list((reference / suite).glob('*/*.json'))
        users, injections = {p.parent.name for p in paths}, {p.stem for p in paths}
        expected = {f'{suite}/{user}/{injection}' for user in users for injection in injections}
        actual = {str(p.relative_to(reference).with_suffix('')) for p in paths}
        audit.check('agentdojo_reference.' + suite, (len(users), len(injections)) == shape and actual == expected,
                    dict(users=len(users), injections=len(injections), pairs=len(actual)),
                    dict(users=shape[0], injections=shape[1], pairs=math.prod(shape)))
        audit.agentdojo_cases.update(expected)
    audit.check('agentdojo_reference.pairs', len(audit.agentdojo_cases) == 389, len(audit.agentdojo_cases), 389)
    selection = audit.optional(args.optimized / 'conditions.json', 'selection')
    selected_names = set(selection or {})
    if selection is not None:
        audit.check('selection.opt_top1', 'opt_top1' in selection)
    for label, directory, ad_names, ipi_names in [('baseline', args.baseline, baseline_ad, baseline_ipi),
                                                  ('optimized', args.optimized, selected_names, selected_names)]:
        for benchmark, names, excluded in [('agentdojo', ad_names, ad_panel), ('ipi', ipi_names, ipi_panel)]:
            audit.benchmark(label + '.' + benchmark, directory, benchmark, names, excluded)
            if args.asr_only_current_best:
                specs = (baseline_ad_specs if benchmark == 'agentdojo' else baseline_ipi_specs) if label == 'baseline' else selection or {}
                audit.asr_bindings(label + '.' + benchmark, directory, benchmark, specs)
        if args.asr_only_current_best:
            audit.skip_asr_only(label + '.ipi_probe')
        else:
            audit.ipi_probe(label + '.ipi_probe', directory / 'ipi', ipi_names)
    if args.asr_only_current_best:
        audit.skip_asr_only('final_condition_scores.additional_native_baselines')
        stopped_best = audit.current_best(selection, args.evolution / 'user_stop_snapshot.json',
                                          args.evolution / 'scores.jsonl', panel)
        score_path = args.optimized / 'condition_scores.json'
        scores = read(score_path) if score_path.exists() else None
        if scores is not None:
            audit.check('final_condition_scores.names', set(scores) == {'opt_top1'})
        if scores is not None and stopped_best is not None and 'opt_top1' in scores:
            audit.reused_best_score(scores['opt_top1'], stopped_best)
    else:
        scores = audit.optional(args.optimized / 'condition_scores.json', 'final_condition_scores')
    if scores:
        if not args.asr_only_current_best:
            audit.check('final_condition_scores.names', set(scores) == baseline_ad | selected_names)
        for name, row in scores.items():
            values = audit.objective('final_condition_scores.' + name, row, panel)
            audit.same('final_condition_scores.' + name + '.agentdojo_mean', row['agentdojo_mean'], mean(v for k, v in values.items() if k in ad_panel))
    ledger_path = args.evolution / 'scores.jsonl'
    ledger = {}
    if ledger_path.exists():
        lines = ledger_path.read_text().splitlines()
        for index, line in enumerate(lines):
            if line.strip():
                try:
                    row = json.loads(line, object_hook=metadata_only)
                except json.JSONDecodeError:
                    audit.check('evolution_ledger.valid_json.' + str(index), False, pending=index == len(lines) - 1)
                    continue
                if row.get('status') == 'complete':
                    ledger[row['evaluation_index']] = row
    evolution_summary = audit.optional(args.evolution / 'summary.json', 'evolution_summary')
    if evolution_summary:
        audit.same('evolution_summary.scored', evolution_summary['scored'], len(ledger))
        audit.check('evolution_summary.elapsed', math.isfinite(evolution_summary['elapsed_hours']) and evolution_summary['elapsed_hours'] >= 0)
        if ledger:
            audit.same('evolution_summary.best', evolution_summary['best']['fitness'], max(r['fitness'] for r in ledger.values()))
        if 'per_case' in evolution_summary['best']:
            audit.objective('evolution_summary.best', evolution_summary['best'], panel)
    for name, spec in (selection or {}).items():
        row = ledger.get(spec['evaluation_index'])
        audit.check('selected.' + name + '.ledger', row is not None)
        if row is None:
            continue
        audit.objective('selected.' + name, row, panel)
        audit.same('selected.' + name + '.fitness', spec['p_user_optimization'], row['fitness'])
        audit.check('selected.' + name + '.template_identity', spec['template_sha256'] == spec['template_sha256_observed']
                    == row['template_sha256_observed'])
        values = {r['case_id']: r['goal_user_mean'] for r in row['per_case']}
        for label, expected in (('joint', row['fitness']), ('agentdojo_mean', mean(values[k] for k in ad_panel)),
                                ('ipi_mean', mean(values[k] for k in ipi_panel))):
            audit.same('selected.' + name + '.selection_scores.' + label, spec['selection_scores'][label], expected)
        artifact = Path(row['score_artifact'])
        cached = audit.optional(artifact, 'selected.' + name + '.artifact')
        if cached:
            audit.objective('selected.' + name + '.artifact', cached, panel)
            audit.same('selected.' + name + '.artifact_fitness', row['fitness'], cached['fitness'])
            audit.check('selected.' + name + '.artifact_template_identity', cached['template_sha256_observed'] == spec['template_sha256'])
    if selection:
        audit.check('selection.unique_hashes_and_evaluations',
                    len({r['template_sha256'] for r in selection.values()}) == len(selection)
                    and len({r['evaluation_index'] for r in selection.values()}) == len(selection))
        for benchmark, case_ids in (() if args.asr_only_current_best else (('agentdojo', ad_panel), ('ipi', ipi_panel))):
            champions = [r for r in selection.values() if 'posthoc_' + benchmark + '_champion' in r['selection_labels']]
            audit.check('selection.' + benchmark + '.champion_label', len(champions) == 1)
            if len(champions) == 1 and ledger:
                maximum = max(mean(r['goal_user_mean'] for r in row['per_case'] if r['case_id'] in case_ids) for row in ledger.values())
                audit.same('selection.' + benchmark + '.champion_score', champions[0]['selection_scores'][benchmark + '_mean'], maximum)
    report_path = args.optimized / 'REPORT.zh-CN.md'
    audit.check('report.exists', report_path.exists(), pending=True)
    if report_path.exists():
        # Do not inspect or emit the fenced template appendix.
        report = report_path.read_text().split('## 5.')[0]
        for label, terms in {'planned_denominator': ('成功/计划',), 'error_and_missing': ('错误', '尚无结果'),
                             'exact_case_holdout': ('上下文留出集', '不是 goal-disjoint'),
                             'surrogate_limit': ('代理指标', '不保证'), 'selection_bias': ('过拟合', '多次选择'),
                             'numerical_path': ('Transformers', 'vLLM', 'FP8 KV'), 'single_run': ('只运行一次',),
                             'cpu_cache_scope': ('8/257/1029/8309', '任意配置', 'CPU')}.items():
            audit.check('report.' + label, all(term in report for term in terms))
        report_rows = audit.optional(args.optimized / 'report_rows.json', 'report_rows')
        if report_rows is not None:
            audit.check('report_rows.conditions', {r['condition'] for r in report_rows} == baseline_ipi | selected_names)
        for row in report_rows or []:
            name = row['condition']
            run = 'baseline' if name in baseline_ipi else 'optimized'
            if args.asr_only_current_best and name == 'opt_top1' and selection and name in selection:
                spec = selection[name]
                audit.check('report_rows.opt_top1.template_identity',
                            row.get('template_sha256') == row.get('template_sha256_observed') == spec['template_sha256'])
                audit.check('report_rows.opt_top1.evaluation_identity', row.get('evaluation_index') == spec['evaluation_index'])
                audit.same('report_rows.opt_top1.objective', row.get('p_user_objective'), spec['p_user_optimization'])
            for benchmark in ('agentdojo', 'ipi'):
                independent = audit.metrics.get(run + '.' + benchmark + '.' + name)
                if args.asr_only_current_best and (benchmark == 'ipi' or name in baseline_ad | selected_names):
                    audit.check('report_rows.' + name + '.' + benchmark + '.present',
                                independent is not None and row.get(benchmark) is not None)
                if not independent or row.get(benchmark) is None:
                    continue
                for part, output_key in [('full', benchmark), ('heldout', benchmark + '_heldout')]:
                    for key in ('successes', 'complete', 'expected', 'errors'):
                        audit.same('report_rows.' + name + '.' + output_key + '.' + key,
                                   row[output_key].get(key), independent[part][key])
                    if part == 'full':
                        audit.same('report_rows.' + name + '.' + output_key + '.asr', row[output_key].get('asr'), independent[part]['asr_scored'])
                        if benchmark == 'agentdojo':
                            audit.same('report_rows.' + name + '.utility', row[output_key].get('utility'), independent[part]['utility_scored'])
    statuses = Counter(r['status'] for r in audit.checks)
    status = 'fail' if statuses['fail'] else ('pending' if statuses['pending'] else 'pass')
    print(json.dumps(dict(schema='deepseek-independent-results-audit-v1', status=status,
                          mode='asr-only-current-best' if args.asr_only_current_best else 'full',
                          check_counts=dict(statuses), metrics=audit.metrics, checks=audit.checks), ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if status == 'fail' else (2 if status == 'pending' and args.require_complete else 0)


if __name__ == '__main__':
    raise SystemExit(main())
