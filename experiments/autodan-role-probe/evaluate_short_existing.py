"""Evaluate unchanged existing candidates under a fixed short-template cap.

This wrapper performs selection and native scoring only. It has no mutation,
text-generation, API, or automatic length-expansion branch.
"""
import json
from pathlib import Path

from broad_validation import BroadDelimiterLibrary
from diverse_policy import validate_diverse_template
import evaluate_static_catalog as existing
from strict_header_policy import strict_header_metrics
import search


MAXIMUM_WRAPPER_TOKENS = 128
_original_load_catalog = existing.load_catalog


def short_catalog(path, tokenizer, library):
    entries = _original_load_catalog(path, tokenizer, library)
    return {key: row for key, row in entries.items()
            if row['wrapper_tokens'] <= MAXIMUM_WRAPPER_TOKENS
            and strict_header_metrics(row['template'])['allowed']}


def run(args):
    existing.CAPS = (MAXIMUM_WRAPPER_TOKENS,)
    existing.load_catalog = short_catalog
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    library = BroadDelimiterLibrary(args.model)
    contexts = json.loads(args.dataset.read_text())['contexts']
    index = json.loads(args.cache_index.read_text())
    for entry in index['entries']:
        path = Path(entry['source_path'])
        if existing.file_sha(path) != entry['source_sha256']:
            raise ValueError('An existing complete score file changed')
        result = json.loads(path.read_text())
        existing.verify_complete(result, contexts)
        if not strict_header_metrics(result['template'])['allowed']:
            raise ValueError('Cached template has repeated headers, lines or paragraphs')
        if not validate_diverse_template(result['template'], tokenizer, library,
                                         maximum=MAXIMUM_WRAPPER_TOKENS)['allowed']:
            raise ValueError('Cached template violates the fixed short-template constraint')
    catalogs = [args.catalog, *args.extra_catalog]
    admission_counts = {str(path.resolve()): len(short_catalog(path, tokenizer, library))
                        for path in catalogs if path.exists()}
    sources = [Path(__file__).resolve(), Path(existing.__file__).resolve(),
               Path(__file__).resolve().parent / 'strict_header_policy.py']
    protocol = dict(schema='short-existing-native-evaluation-constraint-v1',
        maximum_wrapper_tokens=MAXIMUM_WRAPPER_TOKENS, automatic_length_expansion=False,
        text_generation=False, template_mutation=False, API_generation=False,
        strict_duplicate_headers_lines_paragraphs_excluded=True,
        full_six_native_scores_only=True,
        sources={str(path): existing.file_sha(path) for path in sources},
        catalogs={str(path.resolve()): existing.file_sha(path) for path in catalogs if path.exists()},
        catalog_accepted_counts=admission_counts,
        cache_index_sha256=existing.file_sha(args.cache_index),
        complete_cached_candidates=len(index['entries']),
        dataset_sha256=existing.file_sha(args.dataset))
    protocol_path = output / 'short_constraint_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Short-template evaluation constraint changed')
    search.write(protocol_path, protocol)
    for path in sources:
        target = output / 'constraint_provenance' / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    return existing.run(args)


if __name__ == '__main__':
    run(existing.arguments())
