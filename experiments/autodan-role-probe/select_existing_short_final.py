"""Select fixed final references from terminal, independently audited scores.

This CPU helper never edits template strings, runs a model, or requests an API.
It refuses to publish a selection from a running/interrupted/error checkpoint.
No candidate file is written unless --finalize is explicitly supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
from evaluate_fixed_candidates import bind_probe_score, text_sha
from evaluate_static_catalog import verify_complete
from prepare import PLAIN, REMEMBER, ROOT, read, sha
from strict_header_policy import strict_header_metrics


BINS = (0., .05, .1, .2, .3, .4, .5, .65, .8, 1.)
BASELINE_SHAS = {
    'f9d007dadefae74918aabc30648e92848933adf92ef8edc033b9babed7412606',
    '128b9c61bf6c95fd1f45b9eb1cc3249a5d64f73b4f4e43f11652a6a8227c1ec7',
}
TERMINAL_STATUSES = {'audited_finite_catalog_exhausted', 'ASR_reserved_deadline'}
FINAL_AUDIT_SCHEMA = 'independent-finite-short128-result-audit-v1'
COVERAGE_AUDIT_SCHEMA = 'independent-short128-static-native-supervision-v1'
EXPECTED_COVERAGE_AUDIT_SHA256 = '114c4f8c78fa3ba8613a4f4731da0b691c9734d38ede30e2918b2cf4921ecef6'


def require(value, message):
    if not value:
        raise ValueError(message)


def probability_bin(value):
    require(0 <= value <= 1, 'P(user) must lie in [0,1]')
    for low, high in zip(BINS, BINS[1:]):
        if low <= value < high or value == high == 1:
            return low, high
    raise ValueError('No probability bin')


def choose(records_paths, dataset, tokenizer, library):
    counts = Counter()
    unique = {}
    sources = []
    for path in records_paths:
        path = path.resolve()
        digest = sha(path)
        sources.append(dict(path=str(path), sha256=digest))
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                counts['records_seen'] += 1
                if row.get('status') != 'complete' or row.get('complete') is not True:
                    counts['partial_or_noncomplete_excluded'] += 1
                    continue
                template = row.get('template')
                require(isinstance(template, str), 'Complete score lacks an actual template string')
                identity = text_sha(template)
                if identity in BASELINE_SHAS or template in (PLAIN, REMEMBER):
                    counts['original_baseline_bytes_excluded'] += 1
                    continue
                if not validate_diverse_template(template, tokenizer, library, maximum=128)['allowed']:
                    counts['short_special_or_nonrepeat_policy_excluded'] += 1
                    continue
                strict = strict_header_metrics(template)
                if not strict['allowed']:
                    counts['duplicate_header_line_or_paragraph_excluded'] += 1
                    continue
                fitness = verify_complete(row, dataset['contexts'])
                require(identity == row['raw_sha256'], 'Complete record raw template SHA differs')
                score_path = Path(row.get('source_score_path', row.get('artifact', ''))).resolve()
                require(score_path.is_file(), 'Complete record has no original native score JSON')
                score_sha = sha(score_path)
                if row.get('source_score_sha256') is not None:
                    require(score_sha == row['source_score_sha256'], 'Original native score JSON changed')
                candidate = dict(template=template, raw_sha256=identity, fitness=fitness,
                    source_score_path=str(score_path), source_score_sha256=score_sha,
                    source_record_file=str(path), source_record_file_sha256=digest,
                    source_record_line_number=line_number, strict_headers=strict,
                    original_catalog_source=row.get('original_catalog_source'))
                if identity in unique:
                    counts['duplicate_complete_records'] += 1
                # Any repeated literal remains bound to the actual chosen
                # measurement; no averaging or predicted fitness is introduced.
                prior = unique.get(identity)
                if prior is None or (fitness, str(score_path)) > (prior['fitness'], prior['source_score_path']):
                    unique[identity] = candidate
    by_bin = {}
    for candidate in unique.values():
        interval = probability_bin(candidate['fitness'])
        prior = by_bin.get(interval)
        if prior is None or (candidate['fitness'], candidate['raw_sha256']) > (prior['fitness'], prior['raw_sha256']):
            by_bin[interval] = candidate
    selected_bins = sorted(by_bin, reverse=True)[:6]
    require(3 <= len(selected_bins) <= 6, 'Need three to six nonempty eligible probability bins')
    return [(interval, by_bin[interval]) for interval in sorted(selected_bins)], dict(counts), sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, action='append', required=True)
    parser.add_argument('--terminal-checkpoint', type=Path, required=True)
    parser.add_argument('--numerical-audit', type=Path, required=True)
    parser.add_argument('--expected-numerical-audit-sha256', required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT/'outputs/diverse8h-20261105/dataset.json')
    parser.add_argument('--model', type=Path, default=Path('/tmp/wz55191-agentdojo-first-tool-model'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--finalize', action='store_true', help='Explicitly publish only after root authorizes final selection')
    args = parser.parse_args()
    require(args.finalize, 'Prepared helper only: publishing requires an explicit --finalize handoff')
    checkpoint = read(args.terminal_checkpoint)
    require(checkpoint.get('status') in TERMINAL_STATUSES,
            'Native fixed-catalog evaluation is not normally terminal; do not freeze changing candidates')
    require(sha(args.numerical_audit) == args.expected_numerical_audit_sha256,
            'Final independent numerical audit SHA differs from the explicit handoff')
    audit = read(args.numerical_audit)
    require(audit.get('schema') == FINAL_AUDIT_SCHEMA and audit.get('status') == 'passed',
            'Independent final numerical audit schema or passed conclusion differs')
    require(audit.get('catalog_evaluation_complete') is True
            and audit.get('terminal_checkpoint') == checkpoint
            and audit.get('catalog_complete') == audit.get('catalog_candidates')
            and audit.get('catalog_candidates', 0) > 0
            and audit.get('catalog_remaining') == 0 and audit.get('catalog_rejected') == 0,
            'Independent final audit does not prove complete terminal catalog coverage')
    coverage_path = Path(audit['coverage_audit_path'])
    require(audit.get('coverage_audit_sha256') == EXPECTED_COVERAGE_AUDIT_SHA256
            and sha(coverage_path) == EXPECTED_COVERAGE_AUDIT_SHA256,
            'Expected independent coverage audit SHA differs')
    coverage = read(coverage_path)
    require(coverage.get('schema') == COVERAGE_AUDIT_SCHEMA and coverage.get('status') == 'passed'
            and coverage.get('terminal_checkpoint') == checkpoint
            and coverage.get('catalog_complete') == coverage.get('catalog_candidates')
            and coverage.get('catalog_candidates') == audit['catalog_candidates']
            and coverage.get('catalog_remaining') == 0 and coverage.get('catalog_rejected') == 0,
            'Bound independent coverage audit lacks the passed complete-coverage conclusion')
    require({text_sha(PLAIN), text_sha(REMEMBER)} == BASELINE_SHAS, 'Baseline constant byte identities changed')
    require(not args.output.exists(), 'Final manifest already exists; preserve the existing frozen selection')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    library = BroadDelimiterLibrary(args.model)
    dataset = read(args.dataset)
    selected, counts, sources = choose(args.records, dataset, tokenizer, library)
    candidates = []
    for interval, row in selected:
        score_path = Path(row['source_score_path'])
        evidence = bind_probe_score(dict(score_path=str(score_path)), row['template'],
                                    Path.cwd(), dataset, sha(args.dataset))
        require(abs(evidence['goal_user_mean'] - row['fitness']) <= 1e-12,
                'Selected original score differs from the ranking record')
        tokens = len(tokenizer.encode(row['template'].replace('{goal}', ''), add_special_tokens=False))
        candidates.append(dict(template_path=str(score_path), score_path=str(score_path),
            score_protocol_path=evidence['score_protocol_path'], p_user_interval=list(interval),
            native_fitness=evidence['goal_user_mean'], wrapper_tokens=tokens,
            raw_sha256=row['raw_sha256'], score_file_sha256=evidence['score_file_sha256'],
            score_protocol_file_sha256=evidence['score_protocol_file_sha256'],
            source_record_file=row['source_record_file'], source_record_file_sha256=row['source_record_file_sha256'],
            source_record_line_number=row['source_record_line_number'], strict_headers=row['strict_headers'],
            original_catalog_source=row['original_catalog_source']))
    checkpoint_sha = sha(args.terminal_checkpoint)
    require(read(args.terminal_checkpoint) == checkpoint, 'Native checkpoint changed during final selection')
    for source in sources:
        require(sha(Path(source['path'])) == source['sha256'], 'Complete-record stream changed during final selection')
    manifest = dict(schema='fixed-candidates-after-short128-final-v1', final=True, selected_unix=time.time(),
        selection_metric='Highest complete native equal-six-case P(user) within each of the highest six eligible nonoverlapping bins',
        no_ASR_feedback=True, new_candidate_generation=False, template_mutation=False,
        maximum_wrapper_tokens=128, strict_no_duplicate_headers_lines_paragraphs=True,
        excluded_exact_baseline_sha256=sorted(BASELINE_SHAS), candidate_count=len(candidates),
        terminal_checkpoint_path=str(args.terminal_checkpoint.resolve()), terminal_checkpoint_sha256=checkpoint_sha,
        terminal_checkpoint_status=checkpoint['status'], numerical_audit_path=str(args.numerical_audit.resolve()),
        numerical_audit_sha256=args.expected_numerical_audit_sha256,
        numerical_audit_schema=FINAL_AUDIT_SCHEMA, numerical_audit_status='passed',
        coverage_audit_path=str(coverage_path.resolve()), coverage_audit_sha256=EXPECTED_COVERAGE_AUDIT_SHA256,
        coverage_audit_schema=COVERAGE_AUDIT_SCHEMA, coverage_audit_status='passed', records_sources=sources,
        filter_counts=counts, dataset_sha256=sha(args.dataset),
        selector_source_sha256=sha(Path(__file__)), strict_helper_source_sha256=sha(ROOT/'strict_header_policy.py'),
        candidates=candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='final_selection_frozen',manifest=str(args.output.resolve()),
                         manifest_sha256=sha(args.output),candidate_count=len(candidates),
                         model_generation_calls=0,ASR_started=False)))


if __name__ == '__main__':
    main()
