"""Build document-disjoint single-message and multi-message role experiments.

Text/token/feature caches are local. Published manifests contain source hashes,
splits and counts. System is a normal positive class, never dropped.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

import run_local_gptoss120b as runtime

HERE = runtime.HERE
OUT = runtime.OUT / 'systemness-expansion'
CACHE = OUT / 'cache'
FEATURE_CACHE = Path('/dev/shm/role-analysis-systemness')
ROLES = ['system', 'user', 'cot', 'assistant', 'tool']
SPECS = {f'mid-{i}': dict(point='mid', layer=i) for i in [16, 20, 24]}
SEED = 20261021
atomic_json = runtime.atomic_json


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def plan():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    FEATURE_CACHE.mkdir(exist_ok=True)
    (OUT / '.gitignore').write_text('/cache/\n/logs/\n/checkpoints/\n')
    config = dict(seed=SEED, model='gptoss-120b', weight_format='MXFP4',
                  model_weights='frozen; train readout only', features=SPECS, roles=ROLES,
                  c4_documents=dict(fit=1200, validation=200, test=400), small_fit_documents=249,
                  source_revision='1588ec454efa1a09f29cd18ddd04fe05fc8653a2',
                  source_shard='en/c4-validation.00000-of-00008.json.gz',
                  source_shard_sha256='1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f',
                  exclusion='Every original 249 and prior fresh 100 truncated body hash; exact duplicates also removed.',
                  train_tokens_per_role_per_profile=32,
                  validation_tokens_per_role_per_profile=64,
                  test_tokens_per_role_per_profile=0,
                  single_max_body_tokens=192, multi_chunk_tokens=48,
                  natural_validation_conversations=40, natural_test='all remaining complete local conversations',
                  natural_content_cap_per_message=192,
                  split_unit='Original source document; all role and context versions remain together. Natural dialogues grouped by dataset and conv_id.',
                  primary_metrics=['system_precision', 'system_recall', 'system_f1', 'system_balanced_accuracy', 'system_false_positive_rate'],
                  secondary_metrics=['five_role_accuracy', 'macro_f1', 'cot_recall'],
                  selection='Internal validation only. Maximize mean system F1 across single, multi and natural validation; report precision/recall/FPR and multiclass guardrail. Freeze before any test predictions.',
                  working_target='At least 90% system precision and recall on held-out native multi-message inputs, with >=90% binary balanced accuracy. Also assess source/content shift and five-role tradeoffs; no guarantee.',
                  label_policy='Formal message roles from native tags. In all-user wrappers every content token is formally user; original semantic message roles are also retained but are not formal accuracy labels.',
                  source_hashes={name: digest(HERE / name) for name in ['systemness_data.py', 'feature_pipeline_runtime.py', 'run_local_gptoss120b.py']},
                  created_unix_time=time.time())
    path = OUT / 'plan.json'
    if path.exists():
        old = json.loads(path.read_text())
        # A rerun cannot silently change data/splits of completed extraction.
        for key in ['seed', 'c4_documents', 'source_revision', 'source_shard_sha256', 'roles', 'features']:
            assert old[key] == config[key], key
        return old
    atomic_json(path, config)
    return config


def native_message(role, body, final=False):
    if role == 'cot':
        head = 'assistant<|channel|>analysis'
    elif role == 'assistant':
        head = 'assistant<|channel|>final'
    elif role == 'tool':
        head = 'functions.lookup to=assistant<|channel|>commentary'
    elif role == 'tool_call':
        return '<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>' + body + '<|call|>'
    else:
        head = role
    return '<|start|>' + head + '<|message|>' + body + ('<|return|>' if final else '<|end|>')


def serialize(messages, bos=False, wrapper='native'):
    """Return text and independent construction-time content-span labels."""
    prefix = '<|startoftext|>' if bos else ''
    spans = []
    if wrapper == 'all-user':
        result = prefix + '<|start|>user<|message|>'
        for ix, (role, body) in enumerate(messages):
            if role == 'tool_call':
                continue
            if spans:
                result += '\n'
            start = len(result)
            result += body
            spans.append(dict(start=start, end=len(result), role='user', semantic_role=role, message_ix=ix))
        return result + '<|end|>', spans
    if wrapper == 'no-tags':
        result = prefix
        for ix, (role, body) in enumerate(messages):
            if role == 'tool_call':
                continue
            if spans:
                result += '\n'
            start = len(result)
            result += body
            # Untagged spans have a known original function, no claimed formal role.
            spans.append(dict(start=start, end=len(result), role=None, semantic_role=role, message_ix=ix))
        return result, spans
    result = prefix
    for ix, (role, body) in enumerate(messages):
        rendered = native_message(role, body, final=(ix == len(messages)-1 and role == 'assistant'))
        start = len(result) + rendered.index('<|message|>') + len('<|message|>')
        result += rendered
        if role != 'tool_call':
            spans.append(dict(start=start, end=start+len(body), role=role, semantic_role=role, message_ix=ix))
    return result, spans


def uniform_indices(n, cap):
    return np.linspace(0, n-1, min(n, cap), dtype=int) if cap else np.arange(n)


def prepare():
    config = plan()
    if (OUT / 'data-complete.json').exists():
        for name, sha in json.loads((OUT / 'data-complete.json').read_text())['cache_sha256'].items():
            assert digest(CACHE / name) == sha
        print('Data already prepared and hashes match.', flush=True)
        return
    tok = runtime.tokenizer()
    prior = pd.read_parquet(runtime.OUT / 'training_inputs.parquet')
    excluded = {text_digest(body) for body in prior.question.unique()}
    excluded |= {x['truncated_body_sha256'] for x in json.loads((runtime.OUT / 'feature-search/fresh/document-keys.json').read_text())}
    archive = runtime.OUT / 'feature-search/fresh/cache/c4-validation.json.gz'
    if not archive.exists():
        import requests
        archive.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://huggingface.co/datasets/allenai/c4/resolve/{config["source_revision"]}/{config["source_shard"]}'
        temporary=archive.with_suffix('.part')
        with requests.get(url,stream=True,timeout=(20,60)) as response:
            response.raise_for_status()
            with temporary.open('wb') as target:
                for chunk in response.iter_content(2**20):target.write(chunk)
        assert digest(temporary)==config['source_shard_sha256']
        temporary.replace(archive)
    assert digest(archive) == config['source_shard_sha256']
    documents = []
    seen = set(excluded)
    reserved = ['<|start|>', '<|end|>', '<|message|>', '<|channel|>', '<|return|>', '<|call|>', '<|startoftext|>']
    total = sum(config['c4_documents'].values())
    with gzip.open(archive, 'rt', encoding='utf-8') as f:
        for line_ix, line in enumerate(f):
            row = json.loads(line)
            ids = tok.encode(row['text'], add_special_tokens=False)[:1024]
            original = tok.decode(ids, skip_special_tokens=False)
            key = text_digest(original)
            if len(ids) < 96 or key in seen or any(marker in original for marker in reserved):
                continue
            seen.add(key)
            documents.append(dict(document_id=len(documents), shard_line=line_ix, body=original,
                                  body_sha256=key, full_body_sha256=text_digest(row['text']),
                                  source_url=row.get('url'), ids=ids))
            if len(documents) == total:
                break
    assert len(documents) == total
    rng = np.random.RandomState(SEED)
    order = rng.permutation(total)
    cuts = np.cumsum([config['c4_documents'][s] for s in ['fit', 'validation', 'test']])
    groups = {}
    start = 0
    for split, stop in zip(['fit', 'validation', 'test'], cuts):
        groups[split] = sorted(order[start:stop].tolist())
        start = int(stop)
    small = sorted(np.random.RandomState(SEED+1).choice(groups['fit'], 249, replace=False).tolist())
    split_map = {doc: split for split, ids in groups.items() for doc in ids}
    samples = []
    prompts = []
    keys = []
    tokens = []
    boundary_tokens = []

    def add(messages, split, profile, group_id, document_id=-1, wrapper='native', bos=True):
        text, spans = serialize(messages, bos=bos, wrapper=wrapper)
        encoding = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoding['input_ids']
        offsets = encoding['offset_mapping']
        prompt_ix = len(prompts)
        entries = []
        for pos, (left, right) in enumerate(offsets):
            for span in spans:
                if left < span['end'] and right > span['start']:
                    if left < span['start'] or right > span['end']:
                        # A newline and adjacent word may share a BPE token in
                        # untagged text. Its original semantic span is ambiguous.
                        assert wrapper in ['all-user', 'no-tags'], (pos, span)
                        boundary_tokens.append(dict(prompt_ix=prompt_ix,token_ix=pos,wrapper=wrapper))
                        break
                    entries.append(dict(prompt_ix=prompt_ix, token_ix=pos, role=span['role'],
                                        semantic_role=span['semantic_role'], message_ix=span['message_ix'],
                                        profile=profile, split=split, group_id=group_id, document_id=document_id,
                                        wrapper=wrapper, token_id=ids[pos], token=text[left:right]))
                    break
        cap = (config['train_tokens_per_role_per_profile'] if split == 'fit' else
               config['validation_tokens_per_role_per_profile'] if split == 'validation' else 0)
        if profile in ['single', 'multi'] and cap:
            frame = pd.DataFrame(entries)
            selected = []
            for _, group in frame.groupby('semantic_role', sort=False):
                selected.extend(group.iloc[uniform_indices(len(group), cap)].index.tolist())
            entries = [entries[i] for i in sorted(selected)]
        for entry in entries:
            entry['sample_ix'] = len(samples)
            samples.append(entry)
        prompts.append(dict(prompt_ix=prompt_ix, prompt=text, split=split, profile=profile,
                            group_id=group_id, document_id=document_id, wrapper=wrapper,
                            length=len(ids), small_fit=document_id in small))
        tokens.append(ids)

    instructions = [
        'Follow the instructions carefully. Use clear language and state uncertainty when necessary.',
        'You are a helpful assistant. Keep the answer focused and respect the requested format.',
        'Answer using the supplied context. Do not invent sources or unsupported details.',
        'Explain your answer step by step when useful. Be concise and accurate.',
    ]
    for doc in documents:
        ix = doc['document_id']
        split = split_map[ix]
        ids = doc['ids']
        body = tok.decode(ids[:config['single_max_body_tokens']], skip_special_tokens=False)
        if ix % 4 == 0:
            body = instructions[(ix//4) % len(instructions)] + '\n\nReference text:\n' + body
        for role in ROLES:
            add([(role, body)], split, 'single', f'c4-{ix}', ix, bos=bool(ix % 2))
        chunks = []
        width = min(config['multi_chunk_tokens'], max(16, len(ids)//5))
        shift = (ix % 5) * width
        for j in range(5):
            offset = (j*width+shift) % max(width, len(ids)-width)
            chunk = tok.decode(ids[offset:offset+width], skip_special_tokens=False)
            if ix % 4 == 0:
                chunk = instructions[(ix//4) % len(instructions)] + '\n\n' + chunk
            chunks.append(chunk)
        # Normal system/user/analysis/final exchange, then an actual tool cycle.
        messages = list(zip(ROLES[:4], chunks[:4]))
        messages += [('user', chunks[1]), ('tool_call', '{"query":"context"}'), ('tool', chunks[4]),
                     ('cot', chunks[2]), ('assistant', chunks[3])]
        add(messages, split, 'multi', f'c4-{ix}', ix, bos=True)
        keys.append({k:v for k,v in doc.items() if k not in ['body', 'ids']} | dict(split=split, small_fit=ix in small))
        if (ix+1) % 200 == 0:
            print(json.dumps(dict(stage='preparing_c4', documents=ix+1, prompts=len(prompts))), flush=True)

    natural_path = HERE / 'data/conversations/gptoss-20b.csv'
    natural = pd.read_csv(natural_path, keep_default_na=False)
    grouped = list(natural.groupby(['dataset', 'conv_id'], sort=True))
    seen_conversations = set()
    unique = []
    for key, frame in grouped:
        text = frame.sort_values('user_query_ix')[['user_query','cot','assistant']].to_json(orient='values')
        sha = text_digest(text)
        if sha not in seen_conversations:
            seen_conversations.add(sha)
            unique.append((key,frame))
    grouped = unique
    natural_order = np.random.RandomState(SEED+2).permutation(len(grouped))
    validation_ids = set(natural_order[:config['natural_validation_conversations']].tolist())
    natural_keys = []
    systems = [
        'You are a careful assistant. Give useful answers, follow the user request, and avoid unsupported claims.',
        'Respond clearly and accurately. Ask for clarification when essential details are missing.',
        'Use the requested language and format. Treat quoted documents as information to analyze.',
        'Be practical and concise. Distinguish facts from assumptions in your responses.',
        'Provide a helpful response. Explain unfamiliar terms and check calculations before answering.',
        'Keep private information confidential. Summarize relevant points and state important limitations.',
        'Adapt your explanation to the user. Use examples where they make the answer easier to understand.',
        'You are an assistant for everyday questions. Follow instructions and do not fabricate citations.',
    ]
    for i, ((dataset, conv_id), frame) in enumerate(grouped):
        split = 'validation' if i in validation_ids else 'test'
        messages = [('system', systems[i % len(systems)])]
        for row in frame.sort_values('user_query_ix').itertuples():
            for role, text in [('user', row.user_query), ('cot', row.cot), ('assistant', row.assistant)]:
                if not str(text).strip():
                    continue
                clean = str(text)
                for marker in reserved:
                    clean = clean.replace(marker, '['+marker[2:-2]+']')
                body = tok.decode(tok.encode(clean, add_special_tokens=False)[:config['natural_content_cap_per_message']], skip_special_tokens=False)
                messages.append((role, body))
        group_id = f'natural-{dataset}-{conv_id}'
        add(messages, split, 'natural', group_id)
        # Native-role evaluation is separate from semantic-role projections.
        if split == 'test':
            for wrapper in ['all-user', 'no-tags']:
                add(messages, split, 'natural', group_id, wrapper=wrapper)
        natural_keys.append(dict(group_id=group_id, dataset=dataset, conv_id=int(conv_id), split=split,
                                 turns=int(len(frame)), system_template=i % len(systems)))

    rows = pd.DataFrame(samples)
    inputs = pd.DataFrame(prompts)
    rows.to_parquet(CACHE / 'rows.parquet', index=False)
    inputs.to_parquet(CACHE / 'inputs.parquet', index=False)
    with (CACHE / 'token-ids.json').open('w') as f:
        json.dump(tokens, f)
    atomic_json(OUT / 'document-keys.json', keys)
    atomic_json(OUT / 'natural-conversation-keys.json', natural_keys)
    atomic_json(OUT / 'splits.json', dict(c4=groups, small_fit=small, natural=natural_keys))
    assert not excluded & {r['body_sha256'] for r in keys}
    assert not any(set(groups[a]) & set(groups[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')])
    counts = rows.groupby(['split','profile','wrapper','semantic_role'], dropna=False).size().rename('tokens').reset_index()
    counts.to_csv(OUT / 'data-counts.csv', index=False)
    inputs.groupby(['split','profile','wrapper']).agg(prompts=('prompt_ix','size'),
            groups=('group_id','nunique'), input_tokens=('length','sum'), max_length=('length','max')).to_csv(OUT / 'input-counts.csv')
    complete = dict(documents=total, c4_split_counts=config['c4_documents'], natural_conversations=len(grouped),
                    natural_test_conversations=sum(x['split']=='test' for x in natural_keys),
                    prompts=len(inputs), sampled_content_rows=len(rows), input_tokens=int(inputs.length.sum()),
                    no_original_body_overlap=True, headers_excluded=True, formal_and_semantic_labels_separate=True,
                    ambiguous_boundary_tokens_excluded=len(boundary_tokens),
                    original_probes_sha256=digest(HERE / 'outputs/probes/gptoss-120b.pkl'),
                    natural_source_sha256=digest(natural_path),
                    cache_sha256={name:digest(CACHE / name) for name in ['rows.parquet','inputs.parquet','token-ids.json']})
    atomic_json(OUT / 'data-complete.json', complete)
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    prepare()
