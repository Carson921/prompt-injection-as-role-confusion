"""Conditional complementary640 fresh C4 confirmation sources; no model inference.

Default help. --preflight uses synthetic articles and local tokenizer CPU only.
The root may explicitly declare/prepare only after the audited promotion gate
and immutable closed readout snapshot exist. Never scan the new window on import
or reconstruct a historical global inventory. Frozen old modules are unchanged.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_long_data as native
import systemness_long_augmentation_data as pure
import systemness_complementary_confirm_snapshot as closed
import systemness_natural_sources as natural_sources


OUT = data.runtime.OUT / 'systemness-complementary-confirmation'
CACHE = data.CACHE / 'complementary-confirmation'
SNAPSHOT = closed.OUT / 'readout-snapshot.json'
DESIGN = closed.OUT / 'root-prospective-design.json'
DESIGN_SHA256 = '6999ea73bb0207ef3d6a00610d96e47de9e05cca38bac5b5ef08bbd2c17bd824'
ARCHIVE = native.ARCHIVE
REVISION, SHARD, ARCHIVE_SHA256 = pure.REVISION, pure.SHARD, pure.ARCHIVE_SHA256
FIRST, LAST = 12303, 20494
GENERAL, FULL, PAIRED = 400, 128, 64
SELECT_SEED = 20296231
DOCUMENT_START, SAMPLE_START = 9_000_000, 9_000_000_000_000
SOURCE_SET = 'complementary-confirm-c4'
FILES = ['inputs.parquet', 'rows.parquet', 'token-ids.json', 'selected-bodies.json']
AUDITS = ['document-keys.json', 'exclusion-audit.json', 'scan-audit.json',
    'independent-validation.json', 'gallery-index.json']
MANIFESTS = dict(pure.MANIFESTS, **{
    'systemness-long-augmentation/document-keys.json':
        '7fe8812b69b4992fa8fc8fbc833c39956724737e642bc01330fb3fbf3f36f7cf'})
ORIGINAL = dict(pure.ORIGINAL)
DEPENDENCIES = ['systemness_complementary_confirmation_data.py', 'systemness_complementary_confirm_snapshot.py',
    'systemness_long_augmentation_data.py', 'systemness_long_data.py',
    'systemness_data.py', 'systemness_natural_sources.py', 'run_local_gptoss120b.py']
NATURAL_HISTORY = {
    'dev320': dict(data_complete=data.runtime.OUT/'systemness-natural-transcripts/data-complete.json',
        cache=data.CACHE/'natural-transcripts-dev', data_complete_sha256='b6c411c920f583a128401c0d787869e2eba17b632136a5cec9613fd227bb9dfd', prompts=320, rows=80436),
    'test128': dict(data_complete=data.runtime.OUT/'systemness-natural-test/data-complete.json',
        cache=data.CACHE/'natural-test', data_complete_sha256='a3ba614a83feda6687a49b2fb10aca2ab3e9af8d3b1b24935e98ce54cf37175a', prompts=128, rows=33476)}
NATURAL_FILES = ['inputs.parquet','rows.parquet','token-ids.json']
INSTRUCTIONS = [
    'Follow the instructions carefully. Use clear language and state uncertainty when necessary.',
    'You are a helpful assistant. Keep the answer focused and respect the requested format.',
    'Answer using the supplied context. Do not invent sources or unsupported details.',
    'Explain your answer step by step when useful. Be concise and accurate.']
require, read, sha = native.require, pure.read, pure.sha
load_tokenizer = native.load_tokenizer
canonical_sha, immutable_json = pure.canonical_sha, pure.immutable_json
canonical_url = pure.canonical_url


def protocol_spec():
    return dict(version=1, experiment='systemness-complementary-confirmation',
        prospective_design_sha256=DESIGN_SHA256, source_revision=REVISION,
        source_shard=SHARD, source_shard_sha256=ARCHIVE_SHA256,
        window=dict(first_line=FIRST, last_line_inclusive=LAST, records=LAST-FIRST+1),
        documents=dict(general=GENERAL, full_length=FULL, paired_general_long=PAIRED),
        roles=data.ROLES, profiles=['single', 'multi', 'long-single', 'long-full'],
        selection_seed=SELECT_SEED,
        selection='Fixed-window eligible records sorted by SHA256(seed,NUL,revision,NUL,shard,NUL,line,NUL,full-raw-SHA). Greedily choose400 general >=96tokens with exact fingerprint/URL dedupe; then128 disjoint remaining true1024 sources in the same rank order. First64 general cohort ranks receive paired up-to1024 long-single variants. No score or role outcome.',
        failure='Insufficient sources, truncated fixed window, non-roundtrip source, corrupt cache or changed dependency fails. Never extend the window or replace frozen IDs.',
        exclusions='Explicit six historical C4 manifests plus original249 two inventories. Recover every historical pinned source full/first192/first1024 text fingerprint. Exclude source lines, stored and recovered full/body/truncated/prefix fingerprints, exact URLs and conservative canonical URLs. No semantic or domain independence claim; original249 missing line/URL provenance remains explicit.',
        canonical_url=pure.protocol_spec()['canonical_url'],
        natural_crosscorpus_exclusion='All320 prepared historical natural dev prompts including quarantine and both splits plus128 prior public-test prompts. Reconstruct every already-used capped S/U/A body from native content token IDs; ContentKeys raw/NFKC-casefold/sanitized full and192-token-prefix plus canonical URLs. Compare candidate complete original article and source URL; no new natural-length C4 filter. Fixed data-complete/cache SHA before decoding; no source probabilities.',
        source_eligibility='Original >=96 tokens, exact decode/encode roundtrip of first min(1024,N) IDs, no reserved Harmony marker or <| substring. Additional full cohort must have exactly1024 retained tokens. User content never selects a model or threshold.',
        input_rules='Preserve frozen head_confirm._prepare_data single/multi/long-single pure sequence rules. Relative source index controls instruction prefixes/chunk rotation/BOS. Full source is same1024 body in five formal native roles. Normal system tag is retained. All normal role content tokens are scored; the native tool_call structural message has no five-class content rows.',
        content_scoring='Every system/user/cot/assistant/tool body token, no sparse sampling. Independent native token-state parser and serialized character spans must agree on all positions. Body offsets reset for every native message.',
        namespace=dict(source_set=SOURCE_SET, document_start=DOCUMENT_START,
            sample_ix_start=SAMPLE_START, group_prefix='complementary-confirm-c4-', uid_prefix='complementary-confirm-c4:',
            cache_ix='Contiguous bank-local row identity; combine source_set with cache_ix in joins.'),
        expected_inputs=GENERAL*6+PAIRED*5+FULL*5,
        gallery=dict(general_multi='12 evenly spaced general cohort ranks including endpoints',
            full_system_ranks=[0,42,84,127], position_curve='All128 full sources; no smoothing or score-selected examples.',
            expected_png_pdf_pairs=17),
        readout='One predeclared audited closed snapshot: new640 residual, duplicated-coordinate matched640 residual, frozen original512 MLP; four fixed threshold records. oldhead-original and oldhead-max33 share identical probabilities. Capture original8+added2 heads in frozen order; matched uses original512+originalrank0/1. No dynamic old verifier, rglob, model/threshold reselection or dependence on current confirmation results.',
        uncertainty='Source-paired bootstrap; single/multi/general-long subset remain paired. Full tail512+ and position bins reuse each original source. Report undefined ratios and five-class tradeoffs.',
        joint_goal='C4 success alone is insufficient; prospectively frozen genuinely fresh128 natural companion is also required. This builder does not read its sources or replies.',
        storage=dict(cache_path=str(CACHE), maximum_cpu_cache_bytes=128*2**20,
            minimum_disk_free_before_prepare=128*2**20, delete_existing_banks=False),
        gpu_work_run=False, model_generation_run=False, downloads_run=False,
        classifier_probabilities_read=False, test_metrics_read=False,
        historical_test_source_text_read_for_exclusion_only=True)


def actual_gate():
    """Unknown final audited640 pins reject before any formal filesystem IO."""
    closed.pin_gate()


def natural_history_signature():
    """Closed metadata+byte hashes only; no response/ID/token JSON decoding."""
    actual_gate();result={}
    for name,pin in NATURAL_HISTORY.items():
        path=pin['data_complete'];require(data.digest(path)==pin['data_complete_sha256'],'Pinned historical natural completion changed: '+name)
        complete=read(path);require(set(complete['cache_sha256'])==set(NATURAL_FILES) and complete['cache_path']==str(pin['cache'])
            and complete['prepared_prompts']==pin['prompts'] and complete['content_rows']==pin['rows'],'Historical natural cache scope changed: '+name)
        require(complete['model_generation_run'] is False and complete['gpu_work_run'] is False,'Prepared source cache claims model work.')
        for file in NATURAL_FILES:require(data.digest(pin['cache']/file)==complete['cache_sha256'][file],'Historical natural cache bytes changed: '+name+'/'+file)
        result[name]=dict(data_complete_path=str(path),data_complete_sha256=pin['data_complete_sha256'],cache_path=str(pin['cache']),cache_sha256=complete['cache_sha256'],prompts=pin['prompts'],rows=pin['rows'])
    return result


def source_signature():
    """Static explicit metadata closure only; never scan source JSON or rglob."""
    actual_gate()
    require(SNAPSHOT.is_file(), 'Root must first declare the audited closed readout snapshot.')
    snapshot = read(SNAPSHOT)
    closed.verify_closed_snapshot(snapshot)
    require(data.digest(DESIGN) == DESIGN_SHA256, 'Prospective root design changed.')
    require(data.digest(ARCHIVE) == ARCHIVE_SHA256, 'Pinned C4 archive changed.')
    expected = {str(data.runtime.OUT/name): value for name,value in MANIFESTS.items()}
    original = {str(path): value for path,value in ORIGINAL.items()}
    require(snapshot['exclusion_inputs_sha256'] == expected
        and snapshot['original249_inputs_sha256'] == original, 'Closed explicit six-manifest/original249 inventory differs.')
    lines=set(); inventories={}
    for path,expected_sha in expected.items():
        require(data.digest(path) == expected_sha, 'Historical key manifest changed: '+path)
        keys=read(path)
        require(isinstance(keys,list) and all(type(k.get('shard_line')) is int for k in keys), 'Invalid historical C4 key metadata.')
        lines.update(k['shard_line'] for k in keys)
        inventories[path]=dict(sha256=expected_sha,records=len(keys))
    require(len(lines)==3180 and max(lines)<FIRST, 'Historical source count/window separation changed.')
    for path,value in original.items(): require(data.digest(path)==value,'Original249 inventory changed.')
    checkpoint,tokenizer_sha=native.tokenizer_assets()
    require(str(checkpoint)==snapshot['checkpoint_path'] and all(
        snapshot['runtime_files_sha256'].get(str(checkpoint/name))==value for name,value in tokenizer_sha.items()),
        'Tokenizer differs from closed readout checkpoint.')
    return dict(source_sha256=data.digest(__file__),
        dependencies_sha256={name:data.digest(data.HERE/name) for name in DEPENDENCIES},
        snapshot_path=str(SNAPSHOT), snapshot_file_sha256=data.digest(SNAPSHOT),
        frozen_sha256=snapshot['snapshot_sha256'], prospective_design_sha256=data.digest(DESIGN),
        archive_path=str(ARCHIVE),archive_sha256=ARCHIVE_SHA256,archive_bytes=ARCHIVE.stat().st_size,
        manifests=inventories,original249_sha256=original,natural_history=natural_history_signature(),known_historical_lines=len(lines),
        known_historical_max_line=max(lines),tokenizer_path=str(checkpoint),tokenizer_sha256=tokenizer_sha)


def plan_value():
    signature=source_signature()
    return dict(protocol=protocol_spec(),source_signature=signature,
        source_sha256=signature['source_sha256'],dependencies_sha256=signature['dependencies_sha256'],
        frozen=read(SNAPSHOT))


def declare():
    value=plan_value()
    path=OUT/'plan.json'
    if not path.exists():
        require(not any((OUT/name).exists() for name in AUDITS+['data-complete.json'])
            and not any((CACHE/name).exists() for name in FILES+['initialized.json']),
            'New test data predates the required source protocol; refusing retrospective declaration.')
    OUT.mkdir(parents=True,exist_ok=True);immutable_json(path,value)
    with path.open('rb') as stream: os.fsync(stream.fileno())
    descriptor=os.open(OUT,os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return value


def verify_plan(plan):
    actual_gate()
    require((OUT/'plan.json').is_file() and read(OUT/'plan.json')==plan==plan_value(),
        'Static source/readout/protocol/dependencies changed after declaration.')


def guarded_source_access(plan):
    actual_gate()
    require((OUT/'plan.json').is_file(),'Persist source protocol before any new source body decoding.')
    verify_plan(plan)


def historical_natural_keys(plan,tokenizer):
    """Read only already-prepared native content, after immutable source freeze."""
    guarded_source_access(plan);content_keys=natural_sources.ContentKeys(tokenizer);counts=Counter();key_scopes={};sets={};total_messages=0
    for name,descriptor in plan['source_signature']['natural_history'].items():
        folder=Path(descriptor['cache_path']);require(data.digest(descriptor['data_complete_path'])==descriptor['data_complete_sha256'],'Natural completion changed after freeze.')
        for file,want in descriptor['cache_sha256'].items():require(data.digest(folder/file)==want,'Natural cache changed after freeze.')
        inputs=pd.read_parquet(folder/'inputs.parquet',columns=['prompt_ix','length','split','wrapper'])
        rows=pd.read_parquet(folder/'rows.parquet',columns=['prompt_ix','message_ix','token_ix','token_id','role'])
        tokens=read(folder/'token-ids.json')
        require(len(inputs)==descriptor['prompts']==len(tokens) and len(rows)==descriptor['rows'] and inputs.prompt_ix.tolist()==list(range(len(inputs)))
            and inputs.wrapper.eq('native').all() and set(rows.role)<={'system','user','assistant'},'Prepared historical natural layout changed.')
        require(set(rows.prompt_ix)==set(inputs.prompt_ix) and not rows.duplicated(['prompt_ix','token_ix']).any(),'Historical content prompt coverage/position uniqueness changed.')
        role_split=Counter();messages=0
        for prompt_ix,group in rows.groupby('prompt_ix',sort=True):
            prompt_ix=int(prompt_ix);entry=inputs.iloc[prompt_ix];ids=tokens[prompt_ix];require(len(ids)==entry.length,'Historical native input length changed.')
            for message_ix,message in group.groupby('message_ix',sort=True):
                message=message.sort_values('token_ix');positions=message.token_ix.to_numpy(dtype=np.int64)
                require(message.role.nunique()==1 and int(message_ix)>=0 and len(positions)>0 and np.array_equal(positions,np.arange(positions[0],positions[0]+len(positions)))
                    and positions[0]>=0 and positions[-1]<len(ids) and message.token_id.tolist()==[ids[int(i)] for i in positions],'Historical native body token IDs/range changed.')
                body=tokenizer.decode(message.token_id.tolist(),skip_special_tokens=False);keys=content_keys(body)
                require(keys,'Empty historical body fingerprint.');counts.update(keys);scope=name+':'+str(entry.split)
                for key in keys:key_scopes.setdefault(key,set()).add(scope)
                messages+=1;role_split[str(entry.split)+':'+str(message.role.iloc[0])]+=1
        total_messages+=messages;sets[name]=dict(prompts=len(inputs),rows=len(rows),messages=messages,split_prompts=inputs.groupby('split').size().to_dict(),message_role_split_counts=dict(role_split))
    keys=set(counts)
    audit=dict(inputs=plan['source_signature']['natural_history'],sets=sets,messages=total_messages,unique_key_count=len(keys),key_type_counts=dict(Counter(k[0] for k in keys)),
        repeated_key_occurrences=sum(v-1 for v in counts.values()),cross_scope_shared_keys=sum(len(v)>1 for v in key_scopes.values()),fingerprint_set_sha256=canonical_sha(sorted(keys)),
        all_prepared_splits_and_quarantine_included=True,native_used_capped_body_only=True,raw_unused_replies_read=False,classifier_probabilities_read=False,test_metrics_read=False,
        comparison='ContentKeys(full,NFKC-casefold,sanitized,normalized-sanitized,prefix192,URLs); no minimum-length filter.')
    return keys,audit


def historical_exclusions(plan,tokenizer):
    """Recover old source fingerprints only after immutable prospective freeze."""
    guarded_source_access(plan)
    hashes,urls,lines=set(),set(),set()
    for path in plan['source_signature']['manifests']:
        for key in read(path):
            lines.add(key['shard_line'])
            hashes.update(key[field] for field in pure.HASH_FIELDS if key.get(field))
            if key.get('source_url'): urls.update([key['source_url'],canonical_url(key['source_url'])])
    prior=pd.read_parquet(data.runtime.OUT/'training_inputs.parquet',columns=['question']).question.unique().tolist()
    raw=read(data.runtime.OUT.parent/'local-gptoss20b/raw_training_data.json')
    require(len(prior)==len(raw)==249,'Original249 source inventory changed.')
    for text in prior+[row['text'] for row in raw]: hashes.update(pure.text_fingerprints(text,tokenizer)[0])
    remaining=set(lines)
    with gzip.open(ARCHIVE,'rt',encoding='utf-8') as stream:
        for line_ix,line in enumerate(stream):
            if line_ix not in remaining: continue
            article=json.loads(line);hashes.update(pure.text_fingerprints(article['text'],tokenizer)[0])
            if article.get('url'): urls.update([article['url'],canonical_url(article['url'])])
            remaining.remove(line_ix)
            if not remaining: break
    require(not remaining and lines and max(lines)<FIRST,'Historical line recovery incomplete/overlaps new window.')
    natural_keys,natural_audit=historical_natural_keys(plan,tokenizer)
    return hashes,urls,lines,natural_keys,dict(stored_manifests=6,unique_known_lines=len(lines),natural_history=natural_audit,
        recovered_historical_lines=len(lines),original249_inventory_counts=[len(prior),len(raw)],
        hash_count=len(hashes),url_keys=len(urls),fingerprint_set_sha256=canonical_sha(sorted(hashes)),
        url_set_sha256=canonical_sha(sorted(urls)),historical_test_source_text_read_for_exclusion_only=True,
        original249_line_url_provenance_complete=False,semantic_deduplication_claimed=False,
        classifier_probabilities_read=False,test_metrics_read=False)


def make_candidate(article,line_ix,tokenizer):
    require(isinstance(article,dict) and isinstance(article.get('text'),str),'Malformed pinned C4 article.')
    fingerprints,ids,short,body=pure.text_fingerprints(article['text'],tokenizer)
    if len(ids)<96:return None,'shorter-than96'
    retained=tokenizer.encode(body,add_special_tokens=False)
    if retained!=ids[:1024]:return None,'non-roundtrip-retained-prefix'
    if '<|' in body or any(marker in body for marker in native.RESERVED):return None,'reserved-native-marker'
    full_hash=sha(article['text']);body_hash=sha(body);short_hash=sha(short)
    rank=sha('\0'.join(map(str,[SELECT_SEED,REVISION,SHARD,line_ix,full_hash])))
    return dict(shard_line=line_ix,source_revision=REVISION,source_shard=SHARD,
        source_url=article.get('url'),canonical_source_url=canonical_url(article.get('url')),
        full_body_sha256=full_hash,full_text_sha256=full_hash,body_sha256=body_hash,
        short_body_sha256=short_hash,prefix192_sha256=short_hash,prefix1024_sha256=body_hash,
        original_tokens=len(ids),body_tokens=len(retained),rank_sha256=rank,
        natural_content_keys=[list(k) for k in sorted(natural_sources.ContentKeys(tokenizer)(article['text'])|{('url',u) for u in natural_sources.canonical_urls(article.get('url') or '')})],_body=body),None


def select_candidates(candidates,historical_hashes,historical_urls,counts=None,*,natural_keys=None):
    counts=(GENERAL,FULL,PAIRED) if counts is None else counts
    general_n,full_n,paired_n=counts
    require(0<paired_n<=general_n and full_n>0,'Invalid declared cohort sizes.')
    ordered=sorted(candidates,key=lambda row:(row['rank_sha256'],row['shard_line']))
    require(len({row['shard_line'] for row in ordered})==len(ordered),'Duplicate candidate line.')
    seen_hashes=set(historical_hashes);seen_urls=set(historical_urls);chosen=[];reasons=Counter();natural_keys=set() if natural_keys is None else set(natural_keys);natural_matches=Counter()
    def take(row):
        matched={tuple(k) for k in row['natural_content_keys']}&natural_keys
        if matched:
            reasons['historical-natural-content-or-URL-duplicate']+=1;natural_matches.update(k[0] for k in matched);return False
        fingerprints={row[field] for field in pure.HASH_FIELDS if row.get(field)}
        urls={v for v in [row.get('source_url'),row.get('canonical_source_url')] if v}
        if fingerprints&seen_hashes:reasons['history-or-selected-text-prefix-duplicate']+=1;return False
        if urls&seen_urls:reasons['history-or-selected-URL-duplicate']+=1;return False
        seen_hashes.update(fingerprints);seen_urls.update(urls);return True
    general=[]
    for row in ordered:
        if take(row):general.append(row)
        if len(general)==general_n:break
    require(len(general)==general_n,'Fewer400 eligible general sources: fixed-window failure, no extension.')
    general_lines={row['shard_line'] for row in general};full=[]
    for row in ordered:
        if row['shard_line'] in general_lines or row['body_tokens']!=1024:continue
        if take(row):full.append(row)
        if len(full)==full_n:break
    require(len(full)==full_n,'Fewer128 disjoint true1024 sources: fixed-window failure, no extension.')
    for cohort,subset in [('general',general),('full-length',full)]:
        for cohort_rank,row in enumerate(subset):
            index=len(chosen)
            chosen.append(dict(row,cohort=cohort,cohort_rank=cohort_rank,source_rank=index,
                source_index=index,document_id=DOCUMENT_START+index,group_id=f'complementary-confirm-c4-{index}',
                split='test',up_to1024_variant=cohort=='general' and cohort_rank<paired_n))
    return chosen,dict(eligible_candidates=len(candidates),selected=len(chosen),
        general_documents=general_n,full_length_documents=full_n,paired_general_documents=paired_n,
        duplicate_rejections=dict(reasons),natural_collision_key_types=dict(natural_matches),natural_history_key_count=len(natural_keys),ranked_candidate_keys_sha256=canonical_sha([
            {k:v for k,v in row.items() if k!='_body'} for row in ordered]),
        no_window_extension=True,no_model_or_role_scores_used=True)


def scan_window(plan,tokenizer,hashes,urls,excluded_lines,natural_keys):
    guarded_source_access(plan)
    candidates=[];reasons=Counter();scanned=0
    with gzip.open(ARCHIVE,'rt',encoding='utf-8') as stream:
        for line_ix,line in enumerate(stream):
            if line_ix<FIRST:continue
            if line_ix>LAST:break
            require(line_ix not in excluded_lines,'New window reuses historical line.')
            scanned+=1;candidate,reason=make_candidate(json.loads(line),line_ix,tokenizer)
            if reason:reasons[reason]+=1
            else:candidates.append(candidate)
    require(scanned==LAST-FIRST+1,'Fixed source window truncated; no extension permitted.')
    selected,audit=select_candidates(candidates,hashes,urls,natural_keys=natural_keys)
    return selected,dict(audit,window=[FIRST,LAST],window_records_scanned=scanned,
        quality_rejections=dict(reasons),classifier_probabilities_read=False,test_metrics_read=False)


def message_specs(item,tokenizer):
    """Frozen pure old single/multi construction, using relative source index."""
    body=item['_body'];ids=tokenizer.encode(body,add_special_tokens=False);ix=item['source_index']
    require(len(ids)==item['body_tokens'] and sha(body)==item['body_sha256'],'Selected body cache changed.')
    bos=bool(ix%2)
    if item['cohort']=='full-length':
        return [([(role,body)],'long-full',bos) for role in data.ROLES]
    short=tokenizer.decode(ids[:192])
    if ix%4==0:short=INSTRUCTIONS[(ix//4)%4]+'\n\nReference text:\n'+short
    specs=[([(role,short)],'single',bos) for role in data.ROLES]
    width=min(48,max(16,len(ids)//5));shift=(ix%5)*width;chunks=[]
    for j in range(5):
        offset=(j*width+shift)%max(width,len(ids)-width)
        chunk=tokenizer.decode(ids[offset:offset+width])
        if ix%4==0:chunk=INSTRUCTIONS[(ix//4)%4]+'\n\n'+chunk
        chunks.append(chunk)
    messages=list(zip(data.ROLES[:4],chunks[:4]))
    messages += [('user',chunks[1]),('tool_call','{"query":"context"}'),
        ('tool',chunks[4]),('cot',chunks[2]),('assistant',chunks[3])]
    specs.append((messages,'multi',True))
    if item['up_to1024_variant']:specs += [([(role,body)],'long-single',bos) for role in data.ROLES]
    return specs


def parse_labels(ids,tokenizer):
    """Independent native state parser, including explicit structural tool calls."""
    special={name:tokenizer.convert_tokens_to_ids(name) for name in native.SPECIAL}
    labels={};active=None;header=[];in_header=False
    for position,token in enumerate(ids):
        if token==special['<|start|>']:
            require(active is None and not in_header,'Nested native message.')
            header=[];in_header=True
        elif token==special['<|message|>']:
            require(in_header,'Native body without header.');head=tokenizer.decode(header,skip_special_tokens=False)
            if head.startswith('functions.'):active='tool'
            elif head.startswith('assistant') and 'to=functions.' in head:active='tool_call'
            elif head.startswith('assistant'):active='cot' if '<|channel|>analysis' in head else 'assistant'
            else:
                require(head in ['system','user'],'Unexpected native formal role.');active=head
            in_header=False
        elif token in [special['<|end|>'],special['<|return|>'],special['<|call|>']]:
            require(active is not None and not in_header,'Native end marker without body.');active=None
        elif in_header:header.append(token)
        elif active is not None:labels[position]=active
    require(active is None and not in_header,'Unterminated native input.')
    return labels


def append_input(messages,profile,item,bos,tokenizer,inputs,rows,tokens):
    prompt,spans=data.serialize(messages,bos=bos,wrapper='native')
    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
    labels=parse_labels(encoded['input_ids'],tokenizer)
    prompt_ix=len(inputs);start=len(rows);positions=set();offsets=Counter()
    common=dict(prompt_ix=prompt_ix,uid=f'complementary-confirm-c4:{prompt_ix}',source_set=SOURCE_SET,
        profile=profile,split='test',group_id=item['group_id'],document_id=item['document_id'],
        source_index=item['source_index'],source_rank=item['source_rank'],cohort_rank=item['cohort_rank'],
        cohort=item['cohort'],wrapper='native')
    for token_ix,(left,right) in enumerate(encoded['offset_mapping']):
        for span in spans:
            if left<span['end'] and right>span['start']:
                require(left>=span['start'] and right<=span['end'] and labels.get(token_ix)==span['role'],
                    'Independent native role or body boundary mismatch.')
                rows.append(dict(common,cache_ix=len(rows),sample_ix=SAMPLE_START+len(rows),
                    token_ix=token_ix,token_id=encoded['input_ids'][token_ix],token=prompt[left:right],
                    role=span['role'],semantic_role=span['semantic_role'],message_ix=span['message_ix'],
                    body_offset=offsets[span['message_ix']]))
                offsets[span['message_ix']]+=1;positions.add(token_ix);break
    require(positions=={p for p,r in labels.items() if r!='tool_call'},'Not all normal native content positions were scored.')
    if profile=='long-full':require(len(positions)==1024 and len(messages)==1,'Full body is not exactly1024 tokens.')
    inputs.append(dict(common,prompt=prompt,length=len(encoded['input_ids']),bos=bos,
        row_start=start,row_stop=len(rows),role=messages[0][0] if len(messages)==1 else 'mixed'))
    tokens.append(encoded['input_ids'])


def render(selected,tokenizer):
    inputs=[];rows=[];tokens=[]
    for item in selected:
        for messages,profile,bos in message_specs(item,tokenizer):
            append_input(messages,profile,item,bos,tokenizer,inputs,rows,tokens)
    return pd.DataFrame(inputs),pd.DataFrame(rows),tokens


def gallery_index(keys,counts=None):
    general_n,full_n,_=(GENERAL,FULL,PAIRED) if counts is None else counts
    general=[k for k in keys if k['cohort']=='general'];full=[k for k in keys if k['cohort']=='full-length']
    indices=data.uniform_indices(general_n,min(12,general_n)).tolist()
    full_ranks=[0,42,84,127] if full_n==128 else data.uniform_indices(full_n,min(4,full_n)).tolist()
    return dict(general_multi=[dict(general[i],profile='multi') for i in indices],
        full_system=[dict(full[i],profile='long-full',role='system') for i in full_ranks],
        full_position_curve=dict(cohort='full-length',documents=full_n,profile='long-full',role='system'),
        selected_before_probabilities=True,score_based_selection=False,
        physical_columns=['new','matched','old-head'],threshold_records=list(closed.RECORD_TO_MODEL),
        old_head_shared_probabilities=True)


def verify_content(inputs,rows,tokens,keys,bodies,tokenizer,counts=None):
    general_n,full_n,paired_n=(GENERAL,FULL,PAIRED) if counts is None else counts
    require(len(keys)==len(bodies)==general_n+full_n and len({k['shard_line'] for k in keys})==len(keys),
        'Distinct fixed source count changed.')
    require([k['cohort'] for k in keys]==['general']*general_n+['full-length']*full_n
        and [k['source_index'] for k in keys]==list(range(len(keys)))
        and [k['source_rank'] for k in keys]==list(range(len(keys))), 'Source layout/order changed.')
    selected=[];seen_hashes=set();seen_urls=set()
    for i,(key,body) in enumerate(zip(keys,bodies)):
        require(key['document_id']==DOCUMENT_START+i and key['group_id']==f'complementary-confirm-c4-{i}'
            and key['split']=='test' and key['cohort_rank']==(i if i<general_n else i-general_n)
            and key['up_to1024_variant']==(i<paired_n)
            and key['source_revision']==REVISION and key['source_shard']==SHARD
            and FIRST<=key['shard_line']<=LAST,'Source identity/cohort namespace changed.')
        ids=tokenizer.encode(body,add_special_tokens=False)
        require(96<=len(ids)<=1024 and len(ids)==key['body_tokens']==min(1024,key['original_tokens'])
            and tokenizer.decode(ids,skip_special_tokens=False)==body and '<|' not in body
            and sha(body)==key['body_sha256']==key['prefix1024_sha256']
            and sha(tokenizer.decode(ids[:192],skip_special_tokens=False))==key['short_body_sha256']==key['prefix192_sha256'],
            'Cached source body/prefix tokens/hash changed.')
        require(key['cohort']!='full-length' or len(ids)==1024,'True1024 cohort changed.')
        require(isinstance(key['natural_content_keys'],list) and key['natural_content_keys']==[list(k) for k in sorted(set(tuple(k) for k in key['natural_content_keys']))]
            and all(len(k)==2 and k[0] in ['full','prefix192','url'] and isinstance(k[1],str) for k in key['natural_content_keys']),'Malformed full-source natural comparison keys.')
        require(key['rank_sha256']==sha('\0'.join(map(str,[SELECT_SEED,REVISION,SHARD,key['shard_line'],key['full_body_sha256']])))
            and key['full_text_sha256']==key['full_body_sha256']
            and key['canonical_source_url']==canonical_url(key['source_url']),'Source rank/raw/URL fingerprint changed.')
        fingerprints={key[field] for field in pure.HASH_FIELDS if key.get(field)}
        urls={value for value in [key.get('source_url'),key.get('canonical_source_url')] if value}
        require(not fingerprints&seen_hashes and not urls&seen_urls,'Selected source text/prefix/URL collision.')
        seen_hashes.update(fingerprints);seen_urls.update(urls)
        selected.append(dict(key,_body=body))
    for subset in [selected[:general_n],selected[general_n:]]:
        require([r['rank_sha256'] for r in subset]==sorted(r['rank_sha256'] for r in subset),'Within-cohort SHA order changed.')
    require(len(inputs)==len(tokens)==general_n*6+paired_n*5+full_n*5
        and inputs.prompt_ix.tolist()==list(range(len(inputs))) and rows.cache_ix.tolist()==list(range(len(rows)))
        and rows.sample_ix.tolist()==list(SAMPLE_START+np.arange(len(rows))),'Prompt/row identity/order changed.')
    require(set(rows.role)==set(data.ROLES) and rows.semantic_role.eq(rows.role).all()
        and rows.split.eq('test').all() and rows.wrapper.eq('native').all()
        and rows.source_set.eq(SOURCE_SET).all(),'Unexpected scored role/split/source wrapper.')
    lookup={ix:g for ix,g in rows.groupby('prompt_ix',sort=False)};prompt_ix=0;body_total=0
    for item in selected:
        for messages,profile,bos in message_specs(item,tokenizer):
            entry=inputs.iloc[prompt_ix];group=lookup[prompt_ix]
            prompt,_=data.serialize(messages,bos=bos,wrapper='native');encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
            ids=encoded['input_ids'];labels=parse_labels(ids,tokenizer)
            require(entry.prompt==prompt and entry.profile==profile and entry.bos==bos
                and ids==tokens[prompt_ix] and len(ids)==entry.length
                and dict(zip(group.token_ix,group.role))=={p:r for p,r in labels.items() if r!='tool_call'},
                'Independent reserialization/native full-content coverage changed.')
            require(entry.row_start==int(group.cache_ix.iloc[0]) and entry.row_stop==int(group.cache_ix.iloc[-1])+1
                and group.token_ix.tolist()==sorted(group.token_ix.tolist()) and group.token_ix.is_unique,
                'Prompt cache contiguous range changed.')
            spans=data.serialize(messages,bos=bos,wrapper='native')[1]
            for record in group.itertuples():
                require(record.token_id==ids[record.token_ix] and record.role==labels[record.token_ix], 'Cached token/label changed.')
                left,right=encoded['offset_mapping'][record.token_ix]
                span=next(s for s in spans if left>=s['start'] and right<=s['end'])
                require(record.message_ix==span['message_ix'] and record.token==prompt[left:right], 'Message/span/token string changed.')
            for _,message in group.groupby('message_ix',sort=False):
                require(message.body_offset.tolist()==list(range(len(message))), 'Body offsets must cover every content position.')
            for field in ['uid','source_set','split','wrapper','profile','document_id','group_id','source_index','source_rank','cohort_rank','cohort']:
                require(group[field].eq(entry[field]).all(),'Row/input metadata changed: '+field)
            require(entry.uid==f'complementary-confirm-c4:{prompt_ix}' and entry.document_id==item['document_id']
                and entry.group_id==item['group_id'],'Input/source identity changed.')
            if profile=='long-full':require(len(group)==1024 and len(messages)==1,'Full role body length changed.')
            body_total+=len(group);prompt_ix+=1
    expected={'single':general_n,'multi':general_n,'long-single':paired_n,'long-full':full_n}
    require(rows.groupby('profile').group_id.nunique().to_dict()==expected,'Profile source counts changed.')
    require(not set(rows[rows.profile=='long-full'].group_id)&set(rows[rows.profile!='long-full'].group_id),
        'Independent full cohort overlaps general.')
    return dict(all_content_tokens_scored=True,independent_native_labels_match=True,
        normal_system_tag_retained=True,inputs=len(inputs),rows=len(rows),content_positions=body_total,
        profile_source_counts=expected,role_counts=rows.groupby(['profile','role']).size().unstack(fill_value=0).to_dict(),
        full_system_tail_rows=int((rows.profile.eq('long-full')&rows.role.eq('system')&rows.body_offset.ge(512)).sum()),
        source_documents=len(keys),general_documents=general_n,full_length_documents=full_n,
        paired_general_long_documents=paired_n,tool_call_content_scored=False)


def save_result(plan,selected,history,scan,tokenizer):
    """Write immutable CPU cache after complete validation; no implicit selection."""
    actual_gate();guarded_source_access(plan)
    inputs,rows,tokens=render(selected,tokenizer)
    keys=[{k:v for k,v in item.items() if k!='_body'} for item in selected]
    bodies=[item['_body'] for item in selected]
    proof=verify_content(inputs,rows,tokens,keys,bodies,tokenizer);verify_plan(plan)
    CACHE.mkdir(parents=True,exist_ok=True)
    owner=dict(plan_sha256=data.digest(OUT/'plan.json'),source_sha256=data.digest(__file__))
    if not (CACHE/'initialized.json').exists():require(not any(CACHE.iterdir()),'Unowned partial cache; inspect before retry.')
    immutable_json(CACHE/'initialized.json',owner)
    pure.save_parquet(CACHE/'inputs.parquet',inputs);pure.save_parquet(CACHE/'rows.parquet',rows)
    immutable_json(CACHE/'token-ids.json',tokens);immutable_json(CACHE/'selected-bodies.json',bodies)
    require(sum((CACHE/name).stat().st_size for name in FILES)<=128*2**20,'CPU cache exceeds declared128MiB cap.')
    for name,value in [('document-keys.json',keys),('exclusion-audit.json',history),('scan-audit.json',scan),
        ('independent-validation.json',proof),('gallery-index.json',gallery_index(keys))]:immutable_json(OUT/name,value)
    signature=plan['source_signature']
    complete=dict(plan_sha256=data.digest(OUT/'plan.json'),source_sha256=data.digest(__file__),
        dependencies_sha256=plan['dependencies_sha256'],source_signature=signature,
        frozen=plan['frozen'],frozen_sha256=signature['frozen_sha256'],snapshot_path=str(SNAPSHOT),
        snapshot_file_sha256=signature['snapshot_file_sha256'],cache_path=str(CACHE),
        cache_sha256={name:data.digest(CACHE/name) for name in FILES},
        metadata_sha256={name:data.digest(OUT/name) for name in AUDITS},
        document_keys_sha256=data.digest(OUT/'document-keys.json'),documents=GENERAL+FULL,
        independent_source_documents=GENERAL+FULL,general_documents=GENERAL,full_length_documents=FULL,
        up_to1024_subset_documents=PAIRED,inputs=len(inputs),rows=len(rows),input_tokens=int(inputs.length.sum()),
        coverage=proof,all_content_tokens_scored=True,independent_native_label_parser_all_passed=True,
        exclusion_input_sha256={path:item['sha256'] for path,item in signature['manifests'].items()},
        known_prior_source_lines=history['unique_known_lines'],exclusion_fingerprints=history['hash_count'],
        known_prior_urls=history['url_keys'],original249_url_line_provenance_complete=False,
        semantic_near_duplicates_or_domain_exclusion_claimed=False,
        test_inputs_prepared=True,test_probabilities_read=False,classifier_probabilities_read=False,test_metrics_read=False,
        gpu_work_run=False,model_generation_run=False,downloads_run=False,
        historical_test_source_text_read_for_exclusion_only=True)
    immutable_json(OUT/'data-complete.json',complete)
    return complete


def prepare():
    actual_gate()
    require((OUT/'plan.json').is_file(),'Explicit --stage declare required before preparation.')
    plan=read(OUT/'plan.json');verify_plan(plan)
    if (OUT/'data-complete.json').exists():return load_verified_data()[0]
    require(not (OUT/'inference').exists() and not (CACHE/'inference').exists(),
        'Inference state exists before preparation completion; inspect instead of restarting.')
    require(shutil.disk_usage(data.CACHE).free>=128*2**20,'Need128MiB free; no existing bank deletion permitted.')
    tokenizer=load_tokenizer();hashes,urls,lines,natural_keys,history=historical_exclusions(plan,tokenizer)
    selected,scan=scan_window(plan,tokenizer,hashes,urls,lines,natural_keys)
    return save_result(plan,selected,history,scan,tokenizer)


def load_verified_data():
    """Static closure/cache verification only; never rescan new source window."""
    actual_gate()
    plan=read(OUT/'plan.json');verify_plan(plan);complete=read(OUT/'data-complete.json')
    signature=plan['source_signature']
    require(complete['plan_sha256']==data.digest(OUT/'plan.json') and complete['source_sha256']==data.digest(__file__)
        and complete['dependencies_sha256']==plan['dependencies_sha256'] and complete['source_signature']==signature
        and complete['frozen']==plan['frozen'] and complete['frozen_sha256']==signature['frozen_sha256']
        and complete['snapshot_path']==str(SNAPSHOT) and complete['snapshot_file_sha256']==signature['snapshot_file_sha256']
        and complete['cache_path']==str(CACHE),'Completed static protocol/readout/source identity changed.')
    require(set(complete['cache_sha256'])==set(FILES) and set(complete['metadata_sha256'])==set(AUDITS),
        'Incomplete owned cache/audit manifest.')
    require(read(CACHE/'initialized.json')==dict(plan_sha256=complete['plan_sha256'],source_sha256=complete['source_sha256']),
        'Cache owner changed.')
    for name,value in complete['cache_sha256'].items():require(data.digest(CACHE/name)==value,'Changed cache: '+name)
    for name,value in complete['metadata_sha256'].items():require(data.digest(OUT/name)==value,'Changed source audit: '+name)
    require(complete['document_keys_sha256']==data.digest(OUT/'document-keys.json')
        and (complete['documents'],complete['general_documents'],complete['full_length_documents'],complete['up_to1024_subset_documents'])
            ==(GENERAL+FULL,GENERAL,FULL,PAIRED)
        and complete['independent_source_documents']==GENERAL+FULL,'Completed cohort counts changed.')
    for field in ['test_inputs_prepared','all_content_tokens_scored','independent_native_label_parser_all_passed',
        'historical_test_source_text_read_for_exclusion_only']:require(complete[field] is True,'Required true flag changed: '+field)
    for field in ['test_probabilities_read','classifier_probabilities_read','test_metrics_read','gpu_work_run',
        'model_generation_run','downloads_run','original249_url_line_provenance_complete',
        'semantic_near_duplicates_or_domain_exclusion_claimed']:require(complete[field] is False,'Source-only boundary changed: '+field)
    inputs=pd.read_parquet(CACHE/'inputs.parquet');rows=pd.read_parquet(CACHE/'rows.parquet')
    tokens=read(CACHE/'token-ids.json');keys=read(OUT/'document-keys.json');bodies=read(CACHE/'selected-bodies.json')
    proof=verify_content(inputs,rows,tokens,keys,bodies,load_tokenizer())
    history=read(OUT/'exclusion-audit.json');scan=read(OUT/'scan-audit.json')
    require(history['stored_manifests']==6 and history['unique_known_lines']==history['recovered_historical_lines']==3180
        and history['original249_inventory_counts']==[249,249]
        and history['historical_test_source_text_read_for_exclusion_only'] is True
        and history['original249_line_url_provenance_complete'] is False
        and history['classifier_probabilities_read'] is False and history['test_metrics_read'] is False
        and complete['known_prior_source_lines']==3180 and complete['exclusion_fingerprints']==history['hash_count']
        and complete['known_prior_urls']==history['url_keys'] and history['natural_history']['inputs']==signature['natural_history']
        and history['natural_history']['messages']>0 and history['natural_history']['unique_key_count']>0
        and history['natural_history']['all_prepared_splits_and_quarantine_included'] is True
        and history['natural_history']['classifier_probabilities_read'] is False and history['natural_history']['test_metrics_read'] is False,'Completed exclusion audit changed.')
    require(scan['window']==[FIRST,LAST] and scan['window_records_scanned']==LAST-FIRST+1
        and scan['selected']==GENERAL+FULL and scan['general_documents']==GENERAL
        and scan['full_length_documents']==FULL and scan['paired_general_documents']==PAIRED
        and scan['no_window_extension'] is True and scan['no_model_or_role_scores_used'] is True
        and scan['classifier_probabilities_read'] is False and scan['test_metrics_read'] is False,
        'Completed fixed-window selection audit changed.')
    require(proof==complete['coverage']==read(OUT/'independent-validation.json')
        and complete['inputs']==len(inputs) and complete['rows']==len(rows)
        and complete['input_tokens']==int(inputs.length.sum())
        and read(OUT/'gallery-index.json')==gallery_index(keys),'Completed coverage/count/gallery proof changed.')
    return complete,inputs,rows,tokens


def verify():
    complete,inputs,rows,tokens=load_verified_data();plan=read(OUT/'plan.json');tokenizer=load_tokenizer()
    hashes,urls,lines,natural_keys,history=historical_exclusions(plan,tokenizer);selected,scan=scan_window(plan,tokenizer,hashes,urls,lines,natural_keys)
    replay=render(selected,tokenizer)
    pd.testing.assert_frame_equal(inputs,replay[0],check_exact=True);pd.testing.assert_frame_equal(rows,replay[1],check_exact=True)
    require(tokens==replay[2] and read(CACHE/'selected-bodies.json')==[item['_body'] for item in selected]
        and read(OUT/'document-keys.json')==[{k:v for k,v in item.items() if k!='_body'} for item in selected]
        and history==read(OUT/'exclusion-audit.json') and scan==read(OUT/'scan-audit.json'),'Source replay changed.')
    return dict(data_complete_sha256=data.digest(OUT/'data-complete.json'),rows=complete['rows'],
        verified_source_replay=True,gpu_work_run=False,test_probabilities_read=False)


@contextmanager
def patched(**values):
    original={key:globals()[key] for key in values}
    globals().update(values)
    try:yield
    finally:globals().update(original)


class SyntheticTokenizer:
    """Reversible character+Harmony-marker fixture, never a real model tokenizer."""
    def __init__(self):
        self.special={m:i+1 for i,m in enumerate(native.RESERVED)};self.inverse={i:m for m,i in self.special.items()};self.markers=sorted(self.special,key=len,reverse=True)
    def convert_tokens_to_ids(self,name):return self.special[name]
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        ids=[];offsets=[];i=0
        while i<len(text):
            marker=next((m for m in self.markers if text.startswith(m,i)),None)
            end=i+len(marker) if marker else i+1;ids.append(self.special[marker] if marker else 1000+ord(text[i]));offsets.append((i,end));i=end
        return dict(input_ids=ids,offset_mapping=offsets) if return_offsets_mapping else dict(input_ids=ids)
    def encode(self,text,add_special_tokens=False):return self(text,add_special_tokens=add_special_tokens)['input_ids']
    def decode(self,ids,skip_special_tokens=False):return ''.join(self.inverse[int(i)] if int(i) in self.inverse else chr(int(i)-1000) for i in ids if not(skip_special_tokens and int(i) in self.inverse))


def preflight():
    """CPU synthetic fixture; zero real new-window/assistant/probability reads."""
    folder=Path(tempfile.mkdtemp(prefix='systemness-complementary-confirmation-data-preflight-',dir='/tmp'))
    source_sha=data.digest(__file__);formal_paths=(str(OUT),str(CACHE),str(SNAPSHOT))
    tokenizer=SyntheticTokenizer();checks=[]
    def check(name,value=True):require(value,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,FileNotFoundError,ValueError,KeyError,AssertionError):checks.append(name)
        else:raise RuntimeError('Fault accepted: '+name)
    # Explicitly synthetic pin test stays synthetic even after future root pins.
    # Only this new module's references are patched; frozen helpers stay untouched.
    from types import SimpleNamespace
    class ForbiddenFormalIO(Exception):pass
    class NoFormalPath:
        def __truediv__(self,name):raise ForbiddenFormalIO('Formal path accessed before pin gate.')
        def is_file(self):raise ForbiddenFormalIO('Formal stat before pin gate.')
    def unavailable_pins():raise RuntimeError('Synthetic unknown final640 pins.')
    def no_formal_read(*args,**kwargs):raise ForbiddenFormalIO('Formal read before pin gate.')
    with patched(closed=SimpleNamespace(pin_gate=unavailable_pins),SNAPSHOT=NoFormalPath(),OUT=NoFormalPath(),read=no_formal_read):
        for name,fn in [('signature',source_signature),('declare',declare),('prepare',prepare),('load',load_verified_data),('verify',verify),('natural-metadata',natural_history_signature),('natural-bodies',lambda:historical_natural_keys({},tokenizer))]:
            reject('Unknown640 pins before formal IO: '+name,fn)
    with patched(OUT=folder/'unfrozen',actual_gate=lambda:None):
        reject('No source decode without declared protocol',lambda:guarded_source_access({}))
    candidates=[]
    for index in range(16):
        text=f'Synthetic CPU article number {index}. '+('Only a fabricated article sentence for local checking. '*320)
        candidate,reason=make_candidate(dict(text=text,url=f'https://fixture.invalid/{index}'),FIRST+index,tokenizer)
        check('Synthetic candidate '+str(index),reason is None);candidates.append(candidate)
    short,reason=make_candidate(dict(text='Short.',url=None),FIRST+20,tokenizer)
    check('Minimum96 token guard',short is None and reason=='shorter-than96')
    bad,reason=make_candidate(dict(text='<|start|>'+(' harmless text '*2000),url=None),FIRST+21,tokenizer)
    check('Reserved native marker rejected',bad is None and reason=='reserved-native-marker')
    counts=(4,2,2)
    selected,audit=select_candidates(candidates,{candidates[1]['body_sha256']},{candidates[2]['source_url']},counts)
    again,audit2=select_candidates(candidates[::-1],{candidates[1]['body_sha256']},{candidates[2]['source_url']},counts)
    check('Fixed SHA selection independent of iteration',selected==again and audit==audit2)
    check('History text and URL excluded',not {r['shard_line'] for r in selected}&{FIRST+1,FIRST+2})
    check('General/full disjoint; paired subset',len(selected)==6 and sum(r['up_to1024_variant'] for r in selected)==2)
    reject('Insufficient general fails no extension',lambda:select_candidates(candidates,set(),set(),(100,2,2)))
    reject('Insufficient disjoint full fails no extension',lambda:select_candidates(candidates,set(),set(),(15,2,2)))
    inputs,rows,tokens=render(selected,tokenizer);keys=[{k:v for k,v in r.items() if k!='_body'} for r in selected]
    bodies=[r['_body'] for r in selected];proof=verify_content(inputs,rows,tokens,keys,bodies,tokenizer,counts)
    check('Native all-content independent parser and same fullbody',proof['all_content_tokens_scored'] and proof['independent_native_labels_match']
        and proof['full_system_tail_rows']==2*512 and len(inputs)==44 and rows.role.eq('system').any())
    gallery=gallery_index(keys,counts)
    check('Pre-probability fixed source galleries',gallery['selected_before_probabilities'] and not gallery['score_based_selection'])
    fixed_keys=[dict(cohort='general',cohort_rank=i) for i in range(400)]+[dict(cohort='full-length',cohort_rank=i) for i in range(128)]
    fixed_gallery=gallery_index(fixed_keys)
    check('Fixed12general4fullrankspluscurve17 pairs',len(fixed_gallery['general_multi'])==12 and [r['cohort_rank'] for r in fixed_gallery['full_system']]==[0,42,84,127] and fixed_gallery['full_position_curve']['documents']==128 and len(fixed_gallery['general_multi'])+len(fixed_gallery['full_system'])+1==17)
    wrong=rows.copy();wrong.loc[0,'body_offset']=1
    reject('Tampered body offset rejected',lambda:verify_content(inputs,wrong,tokens,keys,bodies,tokenizer,counts))
    wrong=rows.copy();wrong.loc[0,'token_id']+=1
    reject('Tampered token ID rejected',lambda:verify_content(inputs,wrong,tokens,keys,bodies,tokenizer,counts))
    wrong=rows.copy();wrong.loc[0,'role']='user'
    reject('Tampered native role rejected',lambda:verify_content(inputs,wrong,tokens,keys,bodies,tokenizer,counts))
    bad_keys=[dict(r) for r in keys];bad_keys[-1]['cohort']='general'
    reject('Cohort overlap/tamper rejected',lambda:verify_content(inputs,rows,tokens,bad_keys,bodies,tokenizer,counts))
    for name,frame in [('inputs.parquet',inputs),('rows.parquet',rows)]:
        pure.save_parquet(folder/name,frame);pd.testing.assert_frame_equal(frame,pd.read_parquet(folder/name),check_exact=True)
    immutable_json(folder/'token-ids.json',tokens);check('Token cache roundtrip',read(folder/'token-ids.json')==tokens)
    reject('Immutable artifact refuses overwrite',lambda:immutable_json(folder/'token-ids.json',[]))
    with patched(OUT=folder/'owned',CACHE=folder/'cache',plan_value=lambda:dict(fixture=True),actual_gate=lambda:None):
        (OUT).mkdir();immutable_json(OUT/'document-keys.json',[])
        reject('No retrospective plan declaration',declare)
    # Six fabricated inventories plus both249-item fabricated original stocks.
    # Replay exact full/192/1024/URL/line exclusions without any real source reader.
    fake_root=folder/'historical-fixture';runtime_out=fake_root/'local-gptoss120b';runtime_out.mkdir(parents=True)
    stock=[f'Fabricated original inventory source {i}: '+('synthetic words '*100) for i in range(249)]
    pure.save_parquet(runtime_out/'training_inputs.parquet',pd.DataFrame(dict(question=stock)))
    original_raw=runtime_out.parent/'local-gptoss20b/raw_training_data.json';original_raw.parent.mkdir();immutable_json(original_raw,[dict(text=t) for t in stock])
    old_articles=[dict(text=f'Fabricated historical article {i}: '+('history only fixture '*100),url=f'https://history.invalid/{i}') for i in range(6)]
    manifests={}
    for i,article in enumerate(old_articles):
        _,ids,short,body=pure.text_fingerprints(article['text'],tokenizer);path=fake_root/f'manifest-{i}.json'
        immutable_json(path,[dict(shard_line=i,full_body_sha256=sha(article['text']),body_sha256=sha(body),prefix192_sha256=sha(short),prefix1024_sha256=sha(body),source_url=article['url'])]);manifests[str(path)]=dict(sha256=data.digest(path),records=1)
    archive=fake_root/'fixture.json.gz'
    with gzip.open(archive,'wt',encoding='utf-8') as stream:
        for article in old_articles:stream.write(json.dumps(article)+'\n')
    # Prepared native natural caches, never source dumps or classifier p.
    # Include clean fit, quarantined fit, validation, and old public-test splits.
    fake_descriptors={};fake_natural_pins={}
    def natural_cache(name,entries):
        cache=fake_root/name;cache.mkdir();ni=[];nr=[];nt=[]
        for prompt_ix,(split,eligible,messages) in enumerate(entries):
            prompt,spans=data.serialize(messages,bos=False,wrapper='native');encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=encoded['input_ids']
            ni.append(dict(prompt_ix=prompt_ix,length=len(ids),split=split,wrapper='native',eligible_fit=eligible));nt.append(ids)
            for token_ix,(left,right) in enumerate(encoded['offset_mapping']):
                for span in spans:
                    if left>=span['start'] and right<=span['end']:
                        nr.append(dict(prompt_ix=prompt_ix,message_ix=span['message_ix'],token_ix=token_ix,token_id=ids[token_ix],role=span['role']));break
        pure.save_parquet(cache/'inputs.parquet',pd.DataFrame(ni));pure.save_parquet(cache/'rows.parquet',pd.DataFrame(nr));immutable_json(cache/'token-ids.json',nt)
        done=cache/'data-complete.json';value=dict(cache_path=str(cache),cache_sha256={f:data.digest(cache/f) for f in NATURAL_FILES},prepared_prompts=len(ni),content_rows=len(nr),model_generation_run=False,gpu_work_run=False);immutable_json(done,value)
        fake_natural_pins[name]=dict(data_complete=done,cache=cache,data_complete_sha256=data.digest(done),prompts=len(ni),rows=len(nr))
        return dict(data_complete_path=str(done),data_complete_sha256=data.digest(done),cache_path=str(cache),cache_sha256=value['cache_sha256'],prompts=len(ni),rows=len(nr))
    same_system='Historical repeated system wording, only synthetic.'
    fake_descriptors['dev320']=natural_cache('dev320',[
        ('fit',True,[('system',same_system),('user',candidates[0]['_body']),('assistant','Synthetic clean assistant body.')]),
        ('fit',False,[('system',same_system),('user',candidates[1]['_body']),('assistant','Synthetic quarantined assistant body.')]),
        ('validation',False,[('system',same_system),('user',candidates[3]['_body'].upper()),('assistant','Synthetic validation assistant body.')])])
    fake_descriptors['test128']=natural_cache('test128',[
        ('test',False,[('system',same_system),('user','Tiny historical URL https://fixture.invalid/4'),('assistant',candidates[5]['_body'])])])
    with patched(NATURAL_HISTORY=fake_natural_pins,actual_gate=lambda:None):
        check('Historical natural metadata and threecache SHA frozen without decoder',natural_history_signature()==fake_descriptors)
        fault=Path(fake_descriptors['test128']['cache_path'])/'token-ids.json';prior=fault.read_bytes();fault.write_text('[]')
        reject('Historical natural token-cache SHA tamper before decoder',natural_history_signature);fault.write_bytes(prior)
    history_plan=dict(source_signature=dict(manifests=manifests,natural_history=fake_descriptors));history_out=fake_root/'builder';history_out.mkdir();immutable_json(history_out/'plan.json',history_plan)
    from types import SimpleNamespace
    with patched(OUT=history_out,ARCHIVE=archive,data=SimpleNamespace(runtime=SimpleNamespace(OUT=runtime_out),digest=data.digest),plan_value=lambda:history_plan,actual_gate=lambda:None):
        fake_hashes,fake_urls,fake_lines,fake_natural_keys,fake_history=historical_exclusions(history_plan,tokenizer)
    check('Six synthetic manifests and two249 inventories recovered',fake_lines==set(range(6)) and fake_history['recovered_historical_lines']==6 and fake_history['original249_inventory_counts']==[249,249] and all(sha(t) in fake_hashes for t in stock))
    check('Synthetic full prefix192 prefix1024 URLs and lines excluded',all(pure.text_fingerprints(a['text'],tokenizer)[0]<=fake_hashes and canonical_url(a['url']) in fake_urls for a in old_articles))
    check('All natural splits quarantine and oldtest capped bodies recovered',fake_history['natural_history']['sets']['dev320']['prompts']==3 and fake_history['natural_history']['sets']['dev320']['split_prompts']=={'fit':2,'validation':1} and fake_history['natural_history']['sets']['test128']['split_prompts']=={'test':1} and fake_history['natural_history']['cross_scope_shared_keys']>0)
    selected,scan_natural=select_candidates(candidates,{candidates[1]['body_sha256']},{candidates[2]['source_url']},counts,natural_keys=fake_natural_keys)
    _,all_collision_audit=select_candidates(candidates,{candidates[1]['body_sha256']},{candidates[2]['source_url']},(9,1,1),natural_keys=fake_natural_keys)
    check('Crosscorpus raw normalized prefix URL and allhistory rejected',not {r['shard_line'] for r in selected}&{FIRST,FIRST+1,FIRST+2,FIRST+3,FIRST+4,FIRST+5} and all_collision_audit['natural_collision_key_types'].get('prefix192',0)>0 and all_collision_audit['natural_collision_key_types'].get('url',0)>0 and ('full',sha(candidates[0]['_body'])) in fake_natural_keys)
    inputs,rows,tokens=render(selected,tokenizer);keys=[{k:v for k,v in r.items() if k!='_body'} for r in selected];bodies=[r['_body'] for r in selected];proof=verify_content(inputs,rows,tokens,keys,bodies,tokenizer,counts)
    audit=scan_natural

    # Complete public cache roundtrip under synthetic-only metadata. No real gate,
    # archive text or probability path is accessed by these failure fixtures.
    synthetic_snapshot={'snapshot_sha256':'a'*64,'fixture_only':True}
    signature=dict(source_sha256=source_sha,dependencies_sha256={},frozen_sha256='a'*64,
        snapshot_file_sha256='b'*64,manifests={},natural_history=fake_descriptors)
    fixture_plan=dict(source_signature=signature,source_sha256=source_sha,dependencies_sha256={},frozen=synthetic_snapshot)
    history=dict(stored_manifests=6,unique_known_lines=3180,recovered_historical_lines=3180,
        original249_inventory_counts=[249,249],hash_count=7,url_keys=3,
        historical_test_source_text_read_for_exclusion_only=True,original249_line_url_provenance_complete=False,
        classifier_probabilities_read=False,test_metrics_read=False,natural_history=fake_history['natural_history'])
    scan=dict(audit,window=[FIRST,LAST],window_records_scanned=LAST-FIRST+1,
        classifier_probabilities_read=False,test_metrics_read=False)
    with patched(OUT=folder/'roundtrip',CACHE=folder/'roundtrip-cache',GENERAL=4,FULL=2,PAIRED=2,
        plan_value=lambda:fixture_plan,actual_gate=lambda:None,load_tokenizer=lambda:tokenizer):
        OUT.mkdir();immutable_json(OUT/'plan.json',fixture_plan)
        complete=save_result(fixture_plan,selected,history,scan,tokenizer)
        replay=load_verified_data()
        check('Public owned cache and metadata complete roundtrip',replay[0]==complete and replay[3]==tokens)
        cache_before={name:data.digest(CACHE/name) for name in FILES}
        check('Completed prepare is readonly and skips source selection',prepare()==complete
            and cache_before=={name:data.digest(CACHE/name) for name in FILES})
        saved=read(CACHE/'initialized.json');data.atomic_json(CACHE/'initialized.json',dict(saved,source_sha256='0'*64))
        reject('Wrong cache owner rejected',load_verified_data);data.atomic_json(CACHE/'initialized.json',saved)
        saved_tokens=(CACHE/'token-ids.json').read_bytes();(CACHE/'token-ids.json').write_text('[]')
        reject('Truncated or changed token cache rejected',load_verified_data);(CACHE/'token-ids.json').write_bytes(saved_tokens)
        saved_complete=read(OUT/'data-complete.json')
        data.atomic_json(OUT/'data-complete.json',dict(saved_complete,gpu_work_run=True))
        reject('Source-only false flags enforced',load_verified_data);data.atomic_json(OUT/'data-complete.json',saved_complete)
        data.atomic_json(OUT/'data-complete.json',dict(saved_complete,metadata_sha256={}))
        reject('Missing audit binding rejected',load_verified_data);data.atomic_json(OUT/'data-complete.json',saved_complete)
    # Fixed complete geometry without loading real articles or building a huge row table.
    check('Full production input/row namespace geometry',GENERAL*6+PAIRED*5+FULL*5==3360
        and DOCUMENT_START==9_000_000 and SAMPLE_START==9_000_000_000_000)
    check('Conservative exact URL canonicalization',canonical_url('HTTPS://Example.COM:443/a?b=2&a=1#x')=='https://example.com/a?b=2&a=1'
        and canonical_url('https://example.com/A')!=canonical_url('https://example.com/a'))
    check('Formal namespaces never rebound and source unchanged',formal_paths==(str(OUT),str(CACHE),str(SNAPSHOT)) and data.digest(__file__)==source_sha)
    report=dict(source_sha256=source_sha,checks=checks,checks_passed=len(checks),synthetic_documents=len(keys),synthetic_historical_manifests=6,synthetic_original_inventory_counts=[249,249],synthetic_natural_prepared_prompts=4,synthetic_natural_messages=12,
        synthetic_inputs=len(inputs),synthetic_rows=len(rows),real_new_window_bodies_decoded=0,
        real_new_IDs_selected=0,historical_source_text_decoded=0,classifier_probabilities_read=False,
        test_metrics_read=False,gpu_work_run=False,model_generation_run=False,downloads_run=False,
        formal_protocol_declared=False,formal_cache_created=False,real_model_weights_or_feature_banks_read=False,real_tokenizer_assets_loaded=False,real_tokenizer_verified=False,real_history_exclusions_verified=False,synthetic_coverage=proof)
    path=folder/'report.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report_path=str(path),report_sha256=data.digest(path),**report)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true',help='CPU synthetic articles and /tmp only; zero real new-window decodes.')
    action.add_argument('--stage',choices=['declare','prepare','verify'],help='Explicit metadata declaration or root-only source prepare/replay.')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.stage=='declare':declare();print(json.dumps(dict(plan_sha256=data.digest(OUT/'plan.json'))),flush=True)
    elif args.stage=='prepare':
        complete=prepare();print(json.dumps(dict(data_complete_sha256=data.digest(OUT/'data-complete.json'),inputs=complete['inputs'],rows=complete['rows'])),flush=True)
    elif args.stage=='verify':print(json.dumps(verify()),flush=True)
    else:parser.print_help()


if __name__=='__main__':main()
