"""Independent new1024-token C4 fit/validation sources, CPU preparation only.

Default: help. --preflight uses synthetic bodies and /tmp, never the new window.
Explicit --stage declare freezes metadata before source decoding. Only the root
may later run prepare/verify after source review and publication. No LM/GPU,
download, classifier probability or metric is used by this source builder.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_long_data as old


OUT = data.runtime.OUT / 'systemness-long-augmentation'
CACHE = data.CACHE / 'long-augmentation'
ARCHIVE = old.ARCHIVE
REVISION = '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
SHARD = 'en/c4-validation.00000-of-00008.json.gz'
ARCHIVE_SHA256 = '1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f'
PROPOSAL_SHA256 = '05b234857c75c9681a0f22e2f2ae5fc5ea72f72787675de32c8471571885e519'
FIRST, LAST = 4111, 12302
COUNTS = {'fit': 256, 'validation': 96}
SAMPLES = {'fit': 256, 'validation': 128}
BODY_CAP, SAMPLE_START, DOCUMENT_START = 1024, 6_000_000_000_000, 6_000_000
SELECT_SEED, SPLIT_SEED = 20296221, 20296222
SOURCE_SET = 'long-augmentation'
FILES = ['inputs.parquet', 'rows.parquet', 'token-ids.json']
HASH_FIELDS = ['body_sha256', 'full_body_sha256', 'truncated_body_sha256', 'short_body_sha256',
    'prefix192_sha256', 'prefix1024_sha256', 'full_text_sha256']
DEPENDENCIES = ['systemness_long_augmentation_data.py', 'systemness_long_data.py',
    'systemness_data.py', 'run_local_gptoss120b.py']
MANIFESTS = {
    'feature-search/fresh/document-keys.json': '244ab2eab8ea1472829eb803ab98209954976950321bdc45efb792869112f55b',
    'systemness-expansion/document-keys.json': '7375b9ea40a87f39d587424a0a7ff0a5267768b3fab1b9cbdf3b251856769797',
    'systemness-long-full/document-keys.json': 'aa46851d242dc1b06218f747f0fd14e5670e0a933c07193f66b07bfb025c4e8b',
    'systemness-long-full/confirmation/document-keys.json': '8a8fd1fae47228add6a5cff98d6e8457c15b2a6b6c990ae03783a9a9f5cf9779',
    'systemness-head-confirmation/document-keys.json': '34be14a707c6a4f4a719f38bbadc554683582d9f66237a497220b60919355440',
}
ORIGINAL = {
    data.runtime.OUT / 'training_inputs.parquet': 'f9a1d9a593899d0b608bdb9d9ed04e265331a2009efc6250f8cab871be5e2905',
    data.runtime.OUT.parent / 'local-gptoss20b/raw_training_data.json': '4a3ad71177d99d86c93a680b53ae834896a718d56f8a924024b73632f23571f3',
}
require = old.require


def read(path):
    return json.loads(Path(path).read_text())


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')))


def immutable_json(path, value):
    if path.exists(): require(read(path) == value, 'Immutable artifact differs: ' + str(path))
    else: data.atomic_json(path, value)


def canonical_url(value):
    """Conservative exact URL equivalence, not domain/semantic deduplication."""
    if not value: return None
    require(isinstance(value, str), 'C4 URL must be a string or null.')
    try:
        parts = urlsplit(value)
        if not parts.scheme or not parts.hostname: return value
        hostname = parts.hostname.lower()
        if ':' in hostname: hostname = '[' + hostname + ']'
        port = parts.port
        if port is not None and not ((parts.scheme.lower() == 'http' and port == 80)
            or (parts.scheme.lower() == 'https' and port == 443)): hostname += ':' + str(port)
        userinfo = parts.netloc.rsplit('@', 1)[0] + '@' if '@' in parts.netloc else ''
        return urlunsplit((parts.scheme.lower(), userinfo + hostname, parts.path, parts.query, ''))
    except ValueError: return value


def protocol_spec():
    return dict(version=1, experiment='systemness-long-augmentation', proposal_sha256=PROPOSAL_SHA256,
        source_revision=REVISION, source_shard=SHARD, source_shard_sha256=ARCHIVE_SHA256,
        window=dict(first_line=FIRST, last_line_inclusive=LAST, records=LAST-FIRST+1),
        documents=COUNTS, roles=data.ROLES, body_tokens=BODY_CAP, sampled_tokens_per_role=SAMPLES,
        selection_seed=SELECT_SEED, split_seed=SPLIT_SEED,
        selection='Collect all eligible records in the fixed window; SHA256(seed,NUL,revision,NUL,shard,NUL,line,NUL,full-bodySHA) order, greedy exact fingerprint/URL dedupe, first352. Independently seeded SHA order assigns first256 fit and remaining96 validation. No model/role result is involved.',
        failure='Insufficient352 qualified unique sources, missing/corrupt inputs or mismatched hashes/schema/parser cause failure. Never extend the window, replace selected IDs, silently overwrite a protocol or consult scores.',
        exclusions='Explicit five historical manifests including flat systemness-head-confirmation; original249 two text inventories; all stored full/body/short/truncated/prefix hashes; recover historical pinned source full/192/1024 fingerprints with the frozen tokenizer. Exact URLs and conservative canonical URL equivalence. Original249 has no complete line/URL provenance.',
        canonical_url='Lower scheme/hostname; drop fragment/default80/http or443/https port. Preserve path case, query bytes/order, userinfo, www and percent encoding. No tracking-query stripping, registrable-domain or semantic dedupe claim.',
        source_eligibility='Original text at least1024 tokens; decode the first1024 then re-encode to exactly the same1024 IDs; no reserved native Harmony marker in body. The same body is serialized in all five formal roles.',
        serialization='Frozen data.serialize native system/user/cot-analysis/assistant-final/tool headers. Normal system tag retained. Balanced BOS=document_id parity; document_id=6000000+new_document_ix, fit block then validation block.',
        sampling='Frozen uniform_indices on actual body positions, including0 and1023; fit256 has128 positions512+, validation128 has64. Independent native parser checks every1024 body token before sampling.',
        namespace=dict(source_set=SOURCE_SET, cohort='new-long', profile='long-full', sample_ix_start=SAMPLE_START,
            document_start=DOCUMENT_START, group_prefix='long-augmentation-c4-', uid_prefix='long-augmentation:',
            cache_ix='Bank-local0..389119; consumers must use bank_id+cache_ix when joining old bank.'),
        order='Selected sources split first; within each split selection-rank order. Fit256 then validation96; each document five ROLES in fixed order; sampled body_offset ascending.',
        historical_source_read_scope='Old sources, including historical tests, are read only to recover exclusion fingerprints, never as new examples. No historical/new classifier predictions or metrics are read.',
        new_test_inputs_prepared=False, test_probabilities_read=False, test_metrics_read=False,
        model_generation_run=False, gpu_work_run=False, downloads_run=False,
        independent_documents='352 genuinely new source documents, not denser coverage of existing132/31. Five role versions and token positions are paired observations, not independent sources.',
        storage=dict(cache_path=str(CACHE), maximum_cpu_cache_bytes=128*2**20,
            minimum_disk_free_before_prepare=128*2**20, reuse_archive=True, delete_existing_banks=False))


def source_signature():
    """Hashes and historical key metadata only; no source text decoding."""
    require(data.digest(ARCHIVE) == ARCHIVE_SHA256, 'Pinned archive changed.')
    inventories = {}; lines = set()
    expected_paths={data.runtime.OUT/name for name in MANIFESTS}
    discovered={path for path in data.runtime.OUT.rglob('document-keys.json') if OUT not in path.parents}
    require(discovered==expected_paths,'Historical C4 manifest inventory changed; review instead of silently omitting a source.')
    for name, expected in MANIFESTS.items():
        path = data.runtime.OUT / name
        require(data.digest(path) == expected, 'Pinned historical manifest changed: ' + name)
        keys = read(path)
        require(isinstance(keys, list) and all(type(row.get('shard_line')) is int for row in keys), 'Invalid historical C4 identity.')
        lines.update(row['shard_line'] for row in keys)
        inventories[str(path)] = dict(sha256=expected, records=len(keys))
    require(len(lines) == 2828 and max(lines) == FIRST-1, 'Historical C4 exclusion line inventory changed.')
    for path, expected in ORIGINAL.items(): require(data.digest(path) == expected, 'Original249 inventory changed.')
    checkpoint, tokenizer_sha = old.tokenizer_assets()
    return dict(source_sha256=data.digest(__file__), dependencies_sha256={name:data.digest(data.HERE/name) for name in DEPENDENCIES},
        archive_path=str(ARCHIVE), archive_sha256=ARCHIVE_SHA256, archive_bytes=ARCHIVE.stat().st_size,
        manifests=inventories, original249_sha256={str(path):digest for path,digest in ORIGINAL.items()},
        tokenizer_path=str(checkpoint), tokenizer_sha256=tokenizer_sha, known_historical_lines=len(lines),
        known_historical_max_line=max(lines))


def plan_value():
    signature = source_signature()
    return dict(protocol=protocol_spec(), source_signature=signature, source_sha256=signature['source_sha256'],
        dependencies_sha256=signature['dependencies_sha256'])


def declare():
    value = plan_value()
    OUT.mkdir(parents=True, exist_ok=True)
    immutable_json(OUT/'plan.json', value)
    with (OUT/'plan.json').open('rb') as stream: os.fsync(stream.fileno())
    descriptor = os.open(OUT, os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return value


def verify_plan(plan):
    require(plan == plan_value() and read(OUT/'plan.json') == plan, 'Source/protocol/dependencies changed after declaration.')


def guarded_source_access(plan):
    require((OUT/'plan.json').is_file(), 'Persist immutable plan before any source text decode.')
    verify_plan(plan)


def text_fingerprints(text, tokenizer):
    ids = tokenizer.encode(text, add_special_tokens=False)
    short = tokenizer.decode(ids[:192], skip_special_tokens=False)
    body = tokenizer.decode(ids[:BODY_CAP], skip_special_tokens=False)
    return {sha(text), sha(short), sha(body)}, ids, short, body


def historical_exclusions(plan, tokenizer):
    guarded_source_access(plan)
    hashes, urls, lines = set(), set(), set()
    for path in plan['source_signature']['manifests']:
        for key in read(path):
            lines.add(key['shard_line'])
            hashes.update(key[field] for field in HASH_FIELDS if key.get(field))
            if key.get('source_url'): urls.update([key['source_url'], canonical_url(key['source_url'])])
    prior = pd.read_parquet(data.runtime.OUT/'training_inputs.parquet', columns=['question']).question.unique().tolist()
    raw = read(data.runtime.OUT.parent/'local-gptoss20b/raw_training_data.json')
    require(len(prior) == len(raw) == 249, 'Original249 text inventory count changed.')
    for text in prior + [row['text'] for row in raw]: hashes.update(text_fingerprints(text, tokenizer)[0])
    remaining = set(lines)
    with gzip.open(ARCHIVE, 'rt', encoding='utf-8') as stream:
        for line_ix, line in enumerate(stream):
            if line_ix not in remaining: continue
            article = json.loads(line); hashes.update(text_fingerprints(article['text'], tokenizer)[0])
            if article.get('url'): urls.update([article['url'], canonical_url(article['url'])])
            remaining.remove(line_ix)
            if not remaining: break
    require(not remaining and max(lines) < FIRST, 'Historical source fingerprint recovery incomplete/overlapping window.')
    proof = dict(stored_manifests=5, unique_known_lines=len(lines), recovered_historical_lines=len(lines),
        original249_inventory_counts=[len(prior),len(raw)], hash_count=len(hashes), url_keys=len(urls),
        fingerprint_set_sha256=canonical_sha(sorted(hashes)), url_set_sha256=canonical_sha(sorted(urls)),
        historical_test_source_text_read_for_exclusion_only=True, original249_line_url_provenance_complete=False,
        test_probabilities_read=False, test_metrics_read=False)
    return hashes, urls, lines, proof


def make_candidate(article, line_ix, tokenizer):
    require(isinstance(article, dict) and isinstance(article.get('text'), str), 'Malformed pinned C4 source record.')
    fingerprints, ids, short, body = text_fingerprints(article['text'], tokenizer)
    if len(ids) < BODY_CAP: return None, 'shorter-than1024'
    if tokenizer.encode(body, add_special_tokens=False) != ids[:BODY_CAP]: return None, 'non-roundtrip1024-prefix'
    if any(marker in body for marker in old.RESERVED): return None, 'reserved-native-marker'
    url = article.get('url'); normalized_url = canonical_url(url)
    full_hash, body_hash, short_hash = sha(article['text']), sha(body), sha(short)
    rank = sha('\0'.join(map(str,[SELECT_SEED, REVISION, SHARD, line_ix, full_hash])))
    return dict(shard_line=line_ix, source_revision=REVISION, source_shard=SHARD, source_url=url,
        canonical_source_url=normalized_url, full_body_sha256=full_hash, full_text_sha256=full_hash,
        body_sha256=body_hash, short_body_sha256=short_hash, prefix192_sha256=short_hash,
        prefix1024_sha256=body_hash, original_tokens=len(ids), body_tokens=BODY_CAP, rank_sha256=rank,
        _body=body), None


def select_candidates(candidates, historical_hashes, historical_urls, counts=None):
    counts = COUNTS if counts is None else counts
    chosen = []; seen_hashes=set(historical_hashes); seen_urls=set(historical_urls); reasons=Counter()
    ordered = sorted(candidates, key=lambda row:(row['rank_sha256'],row['shard_line']))
    require(len({row['shard_line'] for row in ordered}) == len(ordered), 'Duplicate candidate source line.')
    for row in ordered:
        fingerprints={row[field] for field in HASH_FIELDS if row.get(field)}
        urls={value for value in [row.get('source_url'),row.get('canonical_source_url')] if value}
        if fingerprints & seen_hashes: reasons['history-or-pool-text-prefix-duplicate']+=1; continue
        if urls & seen_urls: reasons['history-or-pool-URL-duplicate']+=1; continue
        seen_hashes.update(fingerprints); seen_urls.update(urls); chosen.append(row)
    require(len(chosen) >= sum(counts.values()), 'Fewer352 qualified unique sources; fixed-window failure, no replacement/extension.')
    selected = chosen[:sum(counts.values())]
    split_order=sorted(selected,key=lambda row:sha('\0'.join(map(str,[SPLIT_SEED,REVISION,SHARD,row['shard_line'],row['full_body_sha256']]))))
    split_map={row['shard_line']:('fit' if i<counts['fit'] else 'validation') for i,row in enumerate(split_order)}
    selection_rank={row['shard_line']:i for i,row in enumerate(selected)}
    final=sorted(selected,key=lambda row:(0 if split_map[row['shard_line']]=='fit' else 1,selection_rank[row['shard_line']]))
    result=[]
    for index,row in enumerate(final):
        result.append(dict(row,split=split_map[row['shard_line']],source_rank=selection_rank[row['shard_line']],
            new_document_ix=index,document_id=DOCUMENT_START+index,group_id=f'long-augmentation-c4-{index}'))
    require(Counter(row['split'] for row in result)==counts,'Independent split count changed.')
    return result, dict(eligible_candidates=len(candidates), unique_eligible=len(chosen), selected=len(result),
        duplicate_rejections=dict(reasons), ranked_candidate_keys_sha256=canonical_sha([
            {key:value for key,value in row.items() if key!='_body'} for row in ordered]),
        no_window_extension=True, no_model_or_role_scores_used=True)


def scan_window(plan, tokenizer, hashes, urls, excluded_lines):
    guarded_source_access(plan)
    candidates=[]; reasons=Counter(); scanned=0
    with gzip.open(ARCHIVE,'rt',encoding='utf-8') as stream:
        for line_ix,line in enumerate(stream):
            if line_ix<FIRST: continue
            if line_ix>LAST: break
            require(line_ix not in excluded_lines,'Fixed new window overlaps historical source line.')
            scanned+=1; candidate,reason=make_candidate(json.loads(line),line_ix,tokenizer)
            if reason: reasons[reason]+=1
            else: candidates.append(candidate)
    require(scanned==LAST-FIRST+1,'Fixed source window is truncated; no extension permitted.')
    selected,audit=select_candidates(candidates,hashes,urls)
    return selected,dict(audit,window_records_scanned=scanned,quality_rejections=dict(reasons),
        window=[FIRST,LAST],new_source_model_probabilities_read=False,test_metrics_read=False)


def render(selected,tokenizer):
    inputs=[];rows=[];tokens=[];keys=[]
    for item in selected:
        body=item['_body'];body_ids=tokenizer.encode(body,add_special_tokens=False)
        require(len(body_ids)==BODY_CAP and sha(body)==item['body_sha256'],'Actual1024 body/hash changed.')
        key={name:value for name,value in item.items() if name!='_body'};keys.append(key)
        for role in data.ROLES:
            prompt,spans=data.serialize([(role,body)],wrapper='native',bos=bool(item['document_id']%2))
            encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
            actual=old.parse_native_labels(encoded['input_ids'],tokenizer); entries=[]
            for position,(left,right) in enumerate(encoded['offset_mapping']):
                if left<spans[0]['end'] and right>spans[0]['start']:
                    require(left>=spans[0]['start'] and right<=spans[0]['end'] and actual.get(position)==role,
                        'Native content boundary/independent formal role mismatch.')
                    entries.append(dict(token_ix=position,token_id=encoded['input_ids'][position],token=prompt[left:right]))
            require(len(entries)==len(actual)==BODY_CAP and list(actual.values())==[role]*BODY_CAP,
                'Independent native parser does not cover exactly1024 body tokens.')
            prompt_ix=len(inputs); uid=f'long-augmentation:{prompt_ix}'
            common=dict(prompt_ix=prompt_ix,uid=uid,source_set=SOURCE_SET,cohort='new-long',profile='long-full',
                split=item['split'],group_id=item['group_id'],document_id=item['document_id'],
                new_document_ix=item['new_document_ix'],source_rank=item['source_rank'],wrapper='native',role=role)
            count=SAMPLES[item['split']]; chosen=data.uniform_indices(BODY_CAP,count)
            require(len(chosen)==count and chosen[0]==0 and chosen[-1]==1023
                and int(np.sum(chosen>=512))==count//2,'Uniform sample/tail positions changed.')
            start=len(rows)
            for offset in chosen:
                entry=entries[int(offset)]
                rows.append(dict(common,cache_ix=len(rows),sample_ix=SAMPLE_START+len(rows),
                    semantic_role=role,message_ix=0,body_offset=int(offset),**entry))
            inputs.append(dict(common,prompt=prompt,length=len(encoded['input_ids']),body_tokens=BODY_CAP,
                sampled_body_tokens=count,bos=bool(item['document_id']%2),row_start=start,row_stop=len(rows)))
            tokens.append(encoded['input_ids'])
    return pd.DataFrame(inputs),pd.DataFrame(rows),tokens,keys


def verify_content(inputs,rows,tokens,keys,tokenizer,counts=None):
    counts=COUNTS if counts is None else counts
    require(Counter(key['split'] for key in keys)==counts and len(keys)==sum(counts.values()),'Fixed source split count.')
    require([key['new_document_ix'] for key in keys]==list(range(len(keys)))
        and [key['split'] for key in keys]==['fit']*counts['fit']+['validation']*counts['validation']
        and len({key['source_rank'] for key in keys})==len(keys)
        and all(key['document_id']==DOCUMENT_START+i and key['group_id']==f'long-augmentation-c4-{i}'
            for i,key in enumerate(keys)), 'Document namespace/split-block order changed.')
    require(inputs.prompt_ix.tolist()==list(range(len(inputs))) and len(tokens)==len(inputs)==len(keys)*5
        and np.array_equal(rows.cache_ix,np.arange(len(rows)))
        and np.array_equal(rows.sample_ix,SAMPLE_START+np.arange(len(rows))) and rows.sample_ix.is_unique,
        'Cache identity/order changed.')
    require(set(rows.split)==set(counts) and set(rows.role)==set(data.ROLES)
        and rows.source_set.eq(SOURCE_SET).all() and rows.profile.eq('long-full').all()
        and rows.wrapper.eq('native').all() and rows.semantic_role.eq(rows.role).all()
        and rows.message_ix.eq(0).all() and inputs.source_set.eq(SOURCE_SET).all()
        and inputs.profile.eq('long-full').all() and inputs.wrapper.eq('native').all(),
        'Unexpected data namespace/role/split.')
    require(inputs.document_id.tolist()==[key['document_id'] for key in keys for role in data.ROLES]
        and inputs.role.tolist()==data.ROLES*len(keys),'Missing/duplicated/reordered five-role document packet.')
    by_doc={key['document_id']:key for key in keys}; lookup={i:g for i,g in rows.groupby('prompt_ix',sort=False)}
    for entry in inputs.itertuples():
        key=by_doc[entry.document_id];ids=tokenizer.encode(entry.prompt,add_special_tokens=False)
        actual=old.parse_native_labels(ids,tokenizer);group=lookup[entry.prompt_ix]
        require(ids==tokens[entry.prompt_ix] and len(ids)==entry.length and len(actual)==BODY_CAP,
            'Cache tokenization/complete-body parser changed.')
        body_positions=sorted(actual); expected=data.uniform_indices(BODY_CAP,SAMPLES[key['split']])
        require(np.array_equal(group.body_offset,expected) and np.array_equal(group.token_ix,
            np.asarray(body_positions)[expected]) and group.role.eq(entry.role).all()
            and all(ids[r.token_ix]==r.token_id and actual[r.token_ix]==r.role for r in group.itertuples()),
            'Actual sampled content/role/body-offset mapping changed.')
        body=tokenizer.decode([ids[p] for p in body_positions],skip_special_tokens=False)
        require(sha(body)==key['body_sha256'] and len(tokenizer.encode(body,add_special_tokens=False))==BODY_CAP,
            'Cached1024 body hash changed.')
        require(sha(tokenizer.decode([ids[p] for p in body_positions[:192]],skip_special_tokens=False))==key['prefix192_sha256']
            and key['prefix1024_sha256']==key['body_sha256'],'Cached source prefix fingerprint changed.')
        require(entry.uid==f'long-augmentation:{entry.prompt_ix}' and entry.split==key['split']
            and entry.group_id==key['group_id'] and entry.document_id==DOCUMENT_START+entry.new_document_ix,
            'Source/prompt namespace mapping changed.')
        for field in ['uid','split','group_id','document_id','new_document_ix','source_rank','role']:
            require(all(group[field]==getattr(entry,field)),'Row/input metadata mismatch: '+field)
    require(not set(rows[rows.split=='fit'].group_id)&set(rows[rows.split=='validation'].group_id),'Fit/validation source leakage.')
    return dict(independent_native_labels_match=True,all_body_tokens1024=True,
        full_content_positions_parsed=len(inputs)*BODY_CAP,sampled_rows=len(rows),
        fit_rows=int(rows.split.eq('fit').sum()),validation_rows=int(rows.split.eq('validation').sum()),
        tail_rows={split:int((rows.split.eq(split)&rows.body_offset.ge(512)).sum()) for split in counts},
        role_counts={split:rows[rows.split==split].role.value_counts().to_dict() for split in counts},
        first_last_positions_retained=True,fit_validation_sources_disjoint=True)


def save_parquet(path,frame):
    descriptor,name=tempfile.mkstemp(prefix=path.name+'-',suffix='.tmp',dir=path.parent);os.close(descriptor);temp=Path(name)
    try:
        frame.to_parquet(temp,index=False)
        if path.exists():require(data.digest(path)==data.digest(temp),'Different existing cache: '+str(path))
        else:temp.replace(path)
    finally:
        if temp.exists():temp.unlink()


def prepare():
    plan=declare()
    if (OUT/'data-complete.json').exists():
        metadata,_,_,_=load_verified_data();print(json.dumps(dict(verified=True,rows=metadata['rows'])),flush=True);return
    require(shutil.disk_usage(data.CACHE).free>=128*2**20,'Insufficient disk; do not delete existing caches.')
    tokenizer=old.load_tokenizer();hashes,urls,lines,history=historical_exclusions(plan,tokenizer)
    selected,scan=scan_window(plan,tokenizer,hashes,urls,lines)
    inputs,rows,tokens,keys=render(selected,tokenizer);proof=verify_content(inputs,rows,tokens,keys,tokenizer)
    require(len(inputs)==1760 and len(rows)==389120 and proof['fit_rows']==327680
        and proof['validation_rows']==61440,'Fixed preparation count changed.')
    verify_plan(plan);CACHE.mkdir(parents=True,exist_ok=True)
    owner=dict(plan_sha256=data.digest(OUT/'plan.json'),source_sha256=data.digest(__file__))
    if not (CACHE/'initialized.json').exists():require(not any(CACHE.iterdir()),'Unowned partial cache; inspect before retry.')
    immutable_json(CACHE/'initialized.json',owner)
    save_parquet(CACHE/'inputs.parquet',inputs);save_parquet(CACHE/'rows.parquet',rows);immutable_json(CACHE/'token-ids.json',tokens)
    require(sum((CACHE/name).stat().st_size for name in FILES)<=128*2**20,'CPU cache exceeds declared128MiB cap.')
    immutable_json(OUT/'document-keys.json',keys);immutable_json(OUT/'exclusion-audit.json',history)
    immutable_json(OUT/'scan-audit.json',scan);immutable_json(OUT/'independent-validation.json',proof)
    complete=dict(plan_sha256=data.digest(OUT/'plan.json'),source_sha256=data.digest(__file__),
        dependencies_sha256=plan['dependencies_sha256'],source_signature=plan['source_signature'],
        document_keys_sha256=data.digest(OUT/'document-keys.json'),cache_path=str(CACHE),
        cache_sha256={name:data.digest(CACHE/name) for name in FILES},
        metadata_sha256={name:data.digest(OUT/name) for name in ['document-keys.json','exclusion-audit.json','scan-audit.json','independent-validation.json']},
        documents=COUNTS,inputs=len(inputs),rows=len(rows),fit_rows=proof['fit_rows'],validation_rows=proof['validation_rows'],
        input_tokens=int(inputs.length.sum()),independent_native_labels_match=True,all_body_tokens1024=True,
        test_inputs_prepared=False,test_probabilities_read=False,test_metrics_read=False,
        historical_test_source_text_read_for_exclusion_only=True,gpu_work_run=False,model_generation_run=False,downloads_run=False)
    immutable_json(OUT/'data-complete.json',complete)
    print(json.dumps(dict(output_path=str(OUT),data_complete_sha256=data.digest(OUT/'data-complete.json'),
        rows=len(rows),inputs=len(inputs),fit_rows=proof['fit_rows'],validation_rows=proof['validation_rows'],gpu_work_run=False)),flush=True)


def load_verified_data():
    """Verify cached source data only, with no source-window decode or model."""
    plan=read(OUT/'plan.json');verify_plan(plan);metadata=read(OUT/'data-complete.json')
    require(metadata['plan_sha256']==data.digest(OUT/'plan.json') and metadata['source_sha256']==data.digest(__file__)
        and metadata['source_signature']==plan['source_signature'] and metadata['dependencies_sha256']==plan['dependencies_sha256']
        and metadata['cache_path']==str(CACHE) and metadata['documents']==COUNTS,'Completed source binding changed.')
    require(set(metadata['cache_sha256'])==set(FILES),'Incomplete cache manifest.')
    require(set(metadata['metadata_sha256'])=={'document-keys.json','exclusion-audit.json',
        'scan-audit.json','independent-validation.json'},'Incomplete source audit manifest.')
    require(read(CACHE/'initialized.json')==dict(plan_sha256=metadata['plan_sha256'],source_sha256=metadata['source_sha256']),
        'Cache ownership changed.')
    for name,digest in metadata['cache_sha256'].items():require(data.digest(CACHE/name)==digest,'Changed cache: '+name)
    for name,digest in metadata['metadata_sha256'].items():require(data.digest(OUT/name)==digest,'Changed source audit: '+name)
    require(metadata['document_keys_sha256']==data.digest(OUT/'document-keys.json') and not metadata['test_inputs_prepared']
        and not metadata['test_probabilities_read'] and not metadata['test_metrics_read'] and not metadata['gpu_work_run'],
        'Source-only development boundary changed.')
    require(metadata['independent_native_labels_match'] is True and metadata['all_body_tokens1024'] is True
        and metadata['model_generation_run'] is False and metadata['downloads_run'] is False,
        'Completed parser/body/source-only flags changed.')
    inputs=pd.read_parquet(CACHE/'inputs.parquet');rows=pd.read_parquet(CACHE/'rows.parquet');tokens=read(CACHE/'token-ids.json')
    proof=verify_content(inputs,rows,tokens,read(OUT/'document-keys.json'),old.load_tokenizer())
    require(proof==read(OUT/'independent-validation.json') and len(rows)==metadata['rows']==389120
        and len(inputs)==metadata['inputs']==1760 and proof['fit_rows']==metadata['fit_rows']==327680
        and proof['validation_rows']==metadata['validation_rows']==61440,'Completed content/count proof changed.')
    return metadata,inputs,rows,tokens


def verify():
    metadata,inputs,rows,tokens=load_verified_data();plan=read(OUT/'plan.json');tokenizer=old.load_tokenizer()
    hashes,urls,lines,history=historical_exclusions(plan,tokenizer);selected,scan=scan_window(plan,tokenizer,hashes,urls,lines)
    replay=render(selected,tokenizer)
    pd.testing.assert_frame_equal(inputs,replay[0],check_exact=True);pd.testing.assert_frame_equal(rows,replay[1],check_exact=True)
    require(tokens==replay[2] and read(OUT/'document-keys.json')==replay[3]
        and history==read(OUT/'exclusion-audit.json') and scan==read(OUT/'scan-audit.json'),'Source preparation replay changed.')
    return dict(data_complete_sha256=data.digest(OUT/'data-complete.json'),rows=len(rows),verified_source_replay=True,gpu_work_run=False)


def preflight():
    """Synthetic source bodies only; no old/new article text is materialized."""
    existed=(OUT.exists(),CACHE.exists());signature=source_signature();tokenizer=old.load_tokenizer()
    folder=Path(tempfile.mkdtemp(prefix='systemness-long-augmentation-preflight-',dir='/tmp'))
    previous_out=globals()['OUT'];globals()['OUT']=folder/'unfrozen'
    try:
        try:guarded_source_access({})
        except RuntimeError:pass
        else:raise RuntimeError('Unfrozen source access was accepted.')
    finally:globals()['OUT']=previous_out
    require(canonical_url('HTTPS://Example.COM:443/a?b=2&a=1#part')=='https://example.com/a?b=2&a=1'
        and canonical_url('https://example.com/A')!='https://example.com/a','Conservative exact URL rule changed.')
    candidates=[]
    for index in range(9):
        text=f'Synthetic source number {index}. '+('A local synthetic sentence for CPU validation. '*300)
        candidate,reason=make_candidate(dict(text=text,url=f'https://fixture.invalid/article/{index}'),100000+index,tokenizer)
        require(reason is None,'Synthetic1024 source unexpectedly rejected.');candidates.append(candidate)
    duplicate=dict(candidates[0],shard_line=200000,rank_sha256='0'*64);candidates.append(duplicate)
    short,reason=make_candidate(dict(text='Short.',url=None),200001,tokenizer)
    require(short is None and reason=='shorter-than1024','Actual1024 minimum guard failed.')
    marker,reason=make_candidate(dict(text='<|start|>'+(' synthetic '*2000),url=None),200002,tokenizer)
    require(marker is None and reason=='reserved-native-marker','Reserved native marker guard failed.')
    counts={'fit':4,'validation':2}
    selected,audit=select_candidates(candidates,{candidates[1]['body_sha256']},{candidates[2]['source_url']},counts)
    again,audit2=select_candidates(list(reversed(candidates)),{candidates[1]['body_sha256']},{candidates[2]['source_url']},counts)
    require(selected==again and audit==audit2 and len(selected)==6,'SHA selection/split must not depend on iteration order.')
    require(not {row['shard_line'] for row in selected}&{100001,100002},'History fingerprint/URL exclusion failed.')
    try:select_candidates(candidates,set(),set(),{'fit':20,'validation':2})
    except RuntimeError:pass
    else:raise RuntimeError('Insufficient fixed-window sources did not fail.')
    inputs,rows,tokens,keys=render(selected,tokenizer)
    proof=verify_content(inputs,rows,tokens,keys,tokenizer,counts)
    require(len(rows)==4*5*256+2*5*128 and proof['tail_rows']=={'fit':4*5*128,'validation':2*5*64},
        'Fit/validation sample/tail count mismatch.')
    wrong=rows.copy();wrong.loc[0,'body_offset']=1
    try:verify_content(inputs,wrong,tokens,keys,tokenizer,counts)
    except RuntimeError:pass
    else:raise RuntimeError('Offset tamper guard failed.')
    inputs.to_parquet(folder/'inputs.parquet',index=False);rows.to_parquet(folder/'rows.parquet',index=False)
    data.atomic_json(folder/'token-ids.json',tokens)
    pd.testing.assert_frame_equal(inputs,pd.read_parquet(folder/'inputs.parquet'),check_exact=True)
    pd.testing.assert_frame_equal(rows,pd.read_parquet(folder/'rows.parquet'),check_exact=True)
    require(tokens==read(folder/'token-ids.json') and existed==(OUT.exists(),CACHE.exists()),'Preflight changed formal state.')
    report=dict(source_sha256=data.digest(__file__),source_signature=signature,synthetic_sources_only=True,
        real_candidate_sources_decoded=0,real_candidate_IDs_selected=0,historical_source_text_decoded=0,
        synthetic_documents=len(keys),synthetic_prompts=len(inputs),synthetic_rows=len(rows),
        fixed_window_failure_guard=True,predecode_plan_guard=True,
        independent_hash_selection_split=True,iteration_order_invariant=True,
        exact_URL_text_prefix_dedupe=True,native_five_role_same1024_body=True,normal_system_tag=True,
        complete1024_native_parser=True,uniform_first_last_and_tail_coverage=True,offset_tamper_rejected=True,
        parquet_token_roundtrip=True,formal_plan_declared=False,formal_cache_created=False,
        gpu_work_run=False,model_generation_run=False,test_probabilities_read=False,test_metrics_read=False)
    path=folder/'report.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(dict(report_path=str(path),**report)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true',help='SyntheticCPU/tmp only; zero new candidate source decodes.')
    action.add_argument('--stage',choices=['declare','prepare','verify'],help='Explicit metadata declaration or root-only CPU source preparation/replay.')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.stage=='declare':
        declare();print(json.dumps(dict(plan_sha256=data.digest(OUT/'plan.json'))),flush=True)
    elif args.stage=='prepare':prepare()
    elif args.stage=='verify':print(json.dumps(verify()),flush=True)
    else:parser.print_help()
