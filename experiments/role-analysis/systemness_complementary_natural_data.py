"""Fresh public S/U/A confirmation data, with no response generation or LM.

Default help; --preflight uses a fake roundtrip tokenizer and124 fake old templates under /tmp only. It does not validate real tokenizer lengths/template exclusions. Explicit root-run stages
first declare an audited closed model snapshot and source rules, then select
roots from user-only fields, freeze all IDs/templates/assignments/gallery, and
only then materialize selected assistant replies. Old source code is unchanged.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_long_data as native
import systemness_natural_sources as sources
import systemness_natural_transcripts as pure
import systemness_natural_test as previous
import systemness_complementary_confirm_snapshot as closed


OUT = data.runtime.OUT / 'systemness-complementary-natural-confirmation'
CACHE = data.CACHE / 'complementary-natural-confirmation'
SNAPSHOT = closed.OUT / 'readout-snapshot.json'
DESIGN = closed.OUT / 'root-prospective-design.json'
DESIGN_SHA256 = '6999ea73bb0207ef3d6a00610d96e47de9e05cca38bac5b5ef08bbd2c17bd824'
OLD_FOLDER = sources.OUT / 'candidates/bb1bfe8c11fbaae2'
OLD_KEYS_SHA256 = '72fa66d1ff307ded3515ff347d006b2af1afb98b6340cf7b5e4e716a6ebd42f1'
SELECT_SEED = 20296232
PER_SOURCE, FAMILIES = 64, 32
SAMPLE_START, DOCUMENT_START = 10_000_000_000_000, 10_000_000
SOURCE_SET = 'complementary-natural'
ROLES, BODY_CAP = ['system','user','assistant'], 192
FILES = ['inputs.parquet','rows.parquet','token-ids.json']
SOURCE_ARTIFACTS = ['natural-source-keys.json','system-templates.json','template-assignments.json',
    'gallery-index.json','source-selection-audit.json']
AUDITS = ['source-read-audit.json','test-ledger.json','duplicate-audit.json','coverage.json']
DEPENDENCIES = ['systemness_complementary_natural_data.py','systemness_complementary_confirm_snapshot.py',
    'systemness_natural_sources.py','systemness_natural_transcripts.py','systemness_natural_test.py',
    'systemness_long_data.py','systemness_data.py','systemness_challenges.py','run_local_gptoss120b.py']
PREPARED_ROLE_INPUTS={
    'main':(data.OUT/'data-complete.json',data.CACHE),
    'supplement':(data.OUT/'supplement-data-complete.json',data.CACHE/'supplement'),
    'long-full':(native.OUT/'data-complete.json',native.CACHE),
    'long-augmentation':(data.runtime.OUT/'systemness-long-augmentation/data-complete.json',data.CACHE/'long-augmentation')}
METADATA_COLUMNS = ['message_tree_id','message_id','parent_id','role','lang','deleted','synthetic','model_name','tree_state']
require, read, sha = native.require, sources.read, sources.sha
canonical_sha, immutable_json = sources.canonical_sha, pure.immutable_json

# Fixed normal instructions. Each distinct wording is one family, not a claim
# of semantic independence. No role/classification labels are disclosed in text.
WORDINGS = [
    ('relevant-detail','Select details according to how they help resolve the question. Put essential information near the beginning, and keep optional background brief enough that the requested answer remains easy to find.'),
    ('dates-and-sequence','When dates or events matter, make their sequence clear. Distinguish when something happened from when it was reported, and explain any uncertainty about the timing that affects the answer.'),
    ('examples-and-rules','Connect an example to the rule it illustrates. Explain which features matter to the comparison, and avoid assuming that a single example establishes what happens in every case.'),
    ('document-purpose','Read a passage in light of its stated purpose and audience. Preserve qualifications when paraphrasing it, and identify any conclusion that goes beyond what the passage actually says.'),
    ('manageable-steps','Divide a complicated request into manageable steps that serve the same objective. Explain the connection between steps, and make any prerequisite visible before a later action depends on it.'),
    ('units-and-labels','Keep units, names, and labels attached to the quantities they describe. Explain a conversion or change in notation where it occurs so that the reader can follow the result.'),
    ('revisions-and-intent','Retain the intended meaning when improving a draft. Make the requested tone and length guide the revision, and avoid adding facts merely to make the writing sound more complete.'),
    ('available-context','Base the response on the context actually available in the exchange. Identify a consequential missing detail, and distinguish a reasonable assumption from information that was explicitly supplied.'),
    ('methods-and-results','Explain how a method leads to the reported result. Give enough detail to assess the reasoning, and identify an approximation when using it changes the precision of the conclusion.'),
    ('comparison-priorities','Use the priorities described in the question to organize a comparison. Apply each criterion consistently across the options and explain why a difference would matter in the stated circumstances.'),
    ('conditional-conclusions','Express a conditional conclusion together with the condition it depends on. If the available information leaves that condition unresolved, describe the useful conclusion that can still be drawn.'),
    ('unfamiliar-vocabulary','Introduce unfamiliar vocabulary through the problem being discussed. Use a concrete explanation before relying on a technical term, and keep the meaning consistent throughout the response.'),
    ('source-distinction','Preserve the distinction between a source statement and an interpretation of it. Attribute a disputed claim when appropriate, and do not turn an inference into an apparent observation.'),
    ('practical-checks','Include a practical way to check a proposed solution when one is available. Explain what the check would establish and how an unexpected result would affect the next action.'),
    ('concise-completeness','Aim for a response that is concise and complete for the requested purpose. Retain the qualifications needed for accuracy while removing repetition that does not help answer the question.'),
    ('contextual-correction','When correcting a claim, explain the relevant point in context. Preserve the parts that remain valid, and distinguish a factual correction from a difference in wording or preference.'),
    ('meaningful-order','Arrange information in the order that makes the answer easiest to use. Introduce necessary context before dependent details, and connect each part to the question rather than listing isolated facts.'),
    ('uncertain-estimates','Describe an estimate in terms of the information used to produce it. State the uncertainty that matters to its use, and avoid presenting extra digits as evidence of greater certainty.'),
    ('requirements-and-choice','Separate an explicit requirement from an optional choice when explaining a plan. Describe the effect of a choice on the stated goal so the reader can judge whether it is suitable.'),
    ('faithful-translation','Translate the meaning of the supplied passage while respecting its requested style. Keep names, quantities, and relationships accurate, and explain an ambiguity when the surrounding context does not resolve it.'),
    ('observable-evidence','Use observable evidence to support a diagnosis or explanation. Connect each proposed cause to the reported facts, and identify a useful observation that would distinguish competing explanations.'),
    ('audience-and-use','Consider how the intended audience will use the answer. Provide necessary background at the point it helps, and include detail when it supports a concrete decision or action.'),
    ('explicit-limitations','Place an important limitation close to the conclusion it qualifies. Explain its practical effect on the answer and avoid implying that the available evidence resolves a question it leaves open.'),
    ('coherent-creation','For a creative request, keep the central idea coherent with the requested form and audience. Choose details that support the intended effect while honoring any explicit constraints in the prompt.'),
    ('claims-and-scope','Keep a claim within the scope supported by the discussion. Explain when applying it to a different situation would require another assumption, and make that assumption visible when it is used.'),
    ('changed-information','Incorporate new information without losing constraints that still apply. Explain a material change in the answer when a revised fact alters the reasoning or the recommended next step.'),
    ('records-and-relations','Preserve the relationships among entries when describing records or a table. Make an absent value visible and distinguish a supplied value from one calculated or inferred from other information.'),
    ('reasoned-feedback','Offer feedback that connects a specific observation to its effect. Describe a concrete improvement when useful, and keep the proposed change consistent with the purpose of the work.'),
    ('bounded-access','Describe available information and completed actions accurately. Do not imply access to material outside the exchange, and distinguish a suggested check from one that has actually been carried out.'),
    ('steps-with-outcomes','Describe an action together with the outcome it is intended to produce. State what must be ready before starting, and identify a result that would justify proceeding to the next action.'),
    ('respectful-disagreement','Address a disagreement by representing the relevant positions accurately. Explain the evidence for your assessment, and keep the response focused on the substance of the question rather than personal assumptions.'),
    ('specific-clarification','Use the surrounding context to resolve ordinary ambiguity. When a missing detail would substantially change the answer, identify the distinction clearly and request only the clarification needed to proceed.')]


def uid(row):return SOURCE_SET+':'+pure.uid(row)


def load_tokenizer():
    closed.pin_gate()
    return native.load_tokenizer()


RESERVED_WORDINGS_SHA256='e6857fd9f3455f59badd5e0df6c8470f8f79778e8651faeb379a6b9aa782d2ff'


def prepared_role_metadata():
    """Hash four explicit prepared caches; never decode prompt/token JSON here."""
    closed.pin_gate();result={}
    for name,(metadata_path,cache) in PREPARED_ROLE_INPUTS.items():
        meta=read(metadata_path);require(set(meta['cache_sha256'])==set(FILES),'Prepared role cache manifest changed: '+name)
        for fn,want in meta['cache_sha256'].items():require(data.digest(cache/fn)==want,'Prepared role cache bytes changed: '+name+'/'+fn)
        result[name]=dict(metadata_path=str(metadata_path),metadata_sha256=data.digest(metadata_path),cache_path=str(cache),cache_sha256=meta['cache_sha256'])
    return result


def protocol_spec():
    return dict(version=1,experiment='fresh-complementary640-public-natural-confirmation',
        prospective_design_sha256=DESIGN_SHA256,source_pins=sources.PINS[:2],
        source_counts={'oasst':PER_SOURCE,'toxicchat0124':PER_SOURCE},system_families=FAMILIES,
        roots_per_family=4,roots_per_source_per_family=2,selection_seed=SELECT_SEED,
        sample_ix_start=SAMPLE_START,roles=ROLES,body_cap=BODY_CAP,
        namespace=dict(source_set=SOURCE_SET,document_start=DOCUMENT_START,source_uid_prefix='complementary-natural:',prompt_uid_prefix='complementary-natural:',family_prefix='complementary-natural/'),
        fixed_wordings=[dict(family=family,text=text) for family,text in WORDINGS],
        freeze_order='Closed audited development model gate + immutable metadata protocol -> user-only catalog/exclusion/ranking -> all128 IDs/32 wordings/assignments/12 gallery sources immutable fsynced freeze -> selected new assistants and old448 source-only reference replies -> data cache. No result-conditioned ID/template/threshold changes.',
        user_catalog='OASST ID/tree/parent/role/provenance metadata without text, then Arrow role=prompter filter before text-to-Python. ToxicChat quoted byte scanner captures conv_id/user_input only; never buffer or UTF8-decode model_output in catalog. Physical byte transport can include nonselected content.',
        source_selection='English clean OASST root (ready_for_export, nondeleted, nonsynthetic) and ToxicChat user root,100..500 characters as in the previous source proposal. Rank SHA(seed,NUL,dataset,NUL,revision,NUL,nativeID,NUL,rawuserSHA), greedy full/normalized/sanitized-prefix192/URL dedupe across all user messages and history;64 per dataset. No assistant-quality or model score used. Insufficient pool fails instead of extending or replacing.',
        exclusions='All old448 native root/tree IDs and user-message fingerprints, legacy all200 requested roots including40val/144test/16failed, all225 legacy user turns. Recover legacy OASST tree100/ToxicChat108 native IDs through user-text matching without reading assistant fields.',
        templates='32 fixed new normal English wordings, exact/normalized/marker-sanitized token-prefix/URL disjoint from old80+44 and each other. Two OA and two TC roots per family; distinct wordings are not proof of semantic independence.',
        path='Clean EN/nondeleted/nonsynthetic/nonempty original U-A-U-A. Fixed SHA child order: first assistant without backtracking, optional first user with a complete assistant child. At most2 genuine user turns, never invent a followup. ToxicChat exactly real user_input+model_output.',
        processing='Frozen capped_body: marker sanitation and roundtrip-stable <=192token cap, raw/clean/capped/normalized/prefix hashes and truncation recorded. Normal native system/user/assistant-final; return markers are serialization boundaries, not source generation completion.',
        quality='Keep all128 frozen IDs in full attempted/status ledger. Missing/empty source/path becomes quality-failed; never replace it. All prepared observations stay all-test. CoT/tool support0 and recall N/A.',
        contamination='Audit raw/clean/capped/normalized/prefix192 assistant keys against all old448 dev+consumedtest and all legacy generated CoT/final/raw failed segments, plus fresh internal duplicates. Clean excludes direct historical or legacy collision; no score-based filtering. Public duplicate field dev_assistant_collision means all historical448, not only320dev.',
        historical_role_inputs='Explicit main/supplement/long-full/long-augmentation metadata+three cache SHAs freeze before source selection. Only after all128 IDs/templates/gallery are frozen: Arrow native prompts, encode/length and independent native role/frame checks, full contiguous role bodies including middle chunks. No sparse feature-row reconstruction. Full/normalized/prefix192 keys form separately named historical-role reference nodes; all S/U/A direct collisions exclude clean only, retain all128, and shared external reference components join bootstrap sources. No semantic independence claim.',
        uncertainty='Union shared system families, internal assistant repeats and external-reference bridges before all/clean/source filtering. Same component IDs for all strata; maximum32 independent positive wordings. Report effective components, undefined ratios and noCI with <2components; future2000 paired-component bootstrap.',
        provenance='Public nonsynthetic OASST human replies and recorded ToxicChat Vicuna-demo replies (exact checkpoint unknown), not120B-generated and not generated under newly assigned wordings. No CoT/tool is invented.',
        excluded_features='Source/template/family/labels/positions/duplicate/cleanliness/component metadata never enters readout features.',
        readout='Same closed3physical/4record snapshot as prospective C4; no dynamic old verifier/rglob, re-selection or recalibration. Fresh C4 and natural both required for the overall goal.',
        gallery='Fixed6OA+6TC evenly spaced source ranks before replies/probabilities; failed roots retained as placeholders. Future12tomato PNG/PDF plus1summary.',
        storage=dict(cache_path=str(CACHE),maximum_cpu_cache_bytes=128*2**20,minimum_free_bytes=128*2**20,
            no_snapshot_copy=True,no_download=True,delete_existing_banks=False),
        gpu_work_run=False,model_generation_run=False,downloads_run=False,
        classifier_probabilities_read=False,test_metrics_read=False,primary_changed=False)


def source_signature():
    """Explicit hashes and published ID metadata; no source text/probabilities."""
    closed.pin_gate()
    require(SNAPSHOT.is_file(),'Root must declare the common audited closed snapshot first.')
    snapshot=read(SNAPSHOT);closed.verify_closed_snapshot(snapshot)
    require(data.digest(DESIGN)==DESIGN_SHA256,'Root prospective design changed.')
    paths=[OLD_FOLDER/name for name in ['source-proposal.json','natural-source-keys.json',
        'system-templates.json','template-assignments.json','source-exclusion-proof.json']]
    require(data.digest(OLD_FOLDER/'natural-source-keys.json')==OLD_KEYS_SHA256,'Historical448 source keys changed.')
    old_keys=read(OLD_FOLDER/'natural-source-keys.json')['source_keys']
    require(len(old_keys)==448 and sources.canonical_sha(old_keys)==sources.LEGACY_KEYS_SHA256,
        'Historical448 canonical IDs changed.')
    for folder,cache,ledger_name in [(pure.OUT,pure.CACHE,'dev-ledger.json'),(previous.OUT,previous.CACHE,'test-ledger.json')]:
        complete=read(folder/'data-complete.json');paths += [folder/'data-complete.json',folder/ledger_name]
        require(set(complete['cache_sha256'])==set(FILES),'Historical natural cache manifest changed.')
        for name,value in complete['cache_sha256'].items():
            require(data.digest(cache/name)==value,'Historical natural cache changed.');paths.append(cache/name)
        for name,value in complete['metadata_sha256'].items():
            require(data.digest(folder/name)==value,'Historical natural ledger/audit changed.');paths.append(folder/name)
    pins={}
    for pin in sources.PINS[:2]:
        path=sources.snapshot_path(pin)
        require(path.is_file() and path.stat().st_size==pin['size'] and data.digest(path)==pin['sha256'], 'Pinned public source snapshot changed.')
        pins[str(path)]=pin['sha256']
    for relative,value in sources.OLD_CSVS.values():
        path=data.HERE/relative;require(data.digest(path)==value,'Legacy generated source changed.');paths.append(path)
    checkpoint,tokenizer_sha=native.tokenizer_assets()
    require(str(checkpoint)==snapshot['checkpoint_path'] and all(snapshot['runtime_files_sha256'].get(str(checkpoint/name))==value
        for name,value in tokenizer_sha.items()),'Tokenizer changed relative to closed checkpoint.')
    return dict(source_sha256=data.digest(__file__),dependencies_sha256={name:data.digest(data.HERE/name) for name in DEPENDENCIES},
        snapshot_path=str(SNAPSHOT),snapshot_file_sha256=data.digest(SNAPSHOT),frozen_sha256=snapshot['snapshot_sha256'],
        prospective_design_sha256=DESIGN_SHA256,prepared_role_inputs=prepared_role_metadata(),history_files_sha256={str(path):data.digest(path) for path in sorted(set(paths))},
        source_pins_sha256=pins,old448_source_keys_sha256=OLD_KEYS_SHA256,old448_roots=448,
        tokenizer_path=str(checkpoint),tokenizer_sha256=tokenizer_sha)


def build_templates(tokenizer,old_templates):
    keyer=sources.ContentKeys(tokenizer)
    require(len(WORDINGS)==FAMILIES==32 and len(old_templates)==124,'Template catalog size changed.')
    old_keys=set().union(*(keyer(row['text']) for row in old_templates));used=set();templates=[]
    for index,(family,text) in enumerate(WORDINGS):
        require(text.strip()==text and '<|' not in text and not re.search(r'\b(system|assistant|classifier|classification|label|probe)\b',text,re.I),
            'Abnormal system wording or direct classification-role leakage.')
        keys=keyer(text);require(not keys&old_keys and not keys&used,'Old/new system wording fingerprint collision.')
        used.update(keys);ids=tokenizer.encode(text,add_special_tokens=False)
        require(0<len(ids)<=BODY_CAP,'System wording exceeds cap; positive stimulus must be intact.')
        templates.append(dict(template_id=f'complementary-natural-{index:02d}',family_id=f'complementary-natural/{family}',
            split='test',role='system',text=text,text_sha256=sha(text),normalized_sha256=sha(sources.normalized(text)),
            token_count=len(ids),exclusion_keys_sha256=canonical_sha(sorted(keys))))
    require(len({row['family_id'] for row in templates})==32,'Family IDs repeat.')
    return templates,dict(new_wordings=32,old_wordings_excluded=124,old80_plus44=True,
        exact_normalized_sanitized_prefix_url_overlap=0,new_cross_wording_overlap=0,
        normal_native_system_tag=True,no_direct_role_classifier_label_leakage=True,
        maximum_independent_positive_stimuli=32,old_catalog_sha256=canonical_sha(old_templates),
        templates_sha256=canonical_sha(templates))


def old_template_catalog():
    closed.pin_gate()
    return read(OLD_FOLDER/'system-templates.json')['templates']+sources.old_templates()


def plan_value():
    closed.pin_gate()
    signature=source_signature();templates,proof=build_templates(load_tokenizer(),old_template_catalog())
    return dict(protocol=protocol_spec(),source_signature=signature,source_sha256=signature['source_sha256'],
        dependencies_sha256=signature['dependencies_sha256'],frozen=read(SNAPSHOT),
        templates=templates,template_audit=proof)


def declare():
    closed.pin_gate()
    value=plan_value();path=OUT/'plan.json'
    if not path.exists():
        require(not any((OUT/name).exists() for name in SOURCE_ARTIFACTS+AUDITS+['source-freeze.json','data-complete.json'])
            and not any((CACHE/name).exists() for name in FILES+['initialized.json']), 'Natural data predates source protocol.')
    OUT.mkdir(parents=True,exist_ok=True);immutable_json(path,value);fsync_files([path])
    return value


def fsync_files(paths):
    for path in paths:
        with Path(path).open('rb') as stream:os.fsync(stream.fileno())
    descriptor=os.open(OUT,os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def verify_plan(plan):
    closed.pin_gate()
    require((OUT/'plan.json').is_file() and read(OUT/'plan.json')==plan==plan_value(), 'Natural closed protocol/source identity changed.')


def guarded_catalog(plan):
    closed.pin_gate()
    require((OUT/'plan.json').is_file(),'Declare source/model protocol before candidate user text.')
    verify_plan(plan)


def user_csv(path):
    """Quoted byte framing with a hard user_input-only UTF8 field whitelist.

    Assistant bytes are not buffered, decoded, stored, or content-counted.
    Physical transport and quote framing necessarily pass those bytes.
    """
    closed.pin_gate()
    header=None;current=None;payload={};records={};column=0;field=bytearray()
    in_quotes=False;pending_quote=False;closed_quote=False;at_start=True;started=False;skip_lf=False;needed={}
    def capture():return header is None or column==0 or column in needed
    def finish_field():
        nonlocal current,column,field,payload
        if header is None:payload[column]=bytes(field).decode('utf-8')
        elif column==0:current=bytes(field).decode('utf-8');payload={'conv_id':current}
        elif column in needed:payload[needed[column]]=bytes(field).decode('utf-8')
        column+=1;field=bytearray()
    def finish_record():
        nonlocal header,current,payload,column,needed
        if header is None:
            header=[payload[i] for i in range(column)]
            require(header[0]=='conv_id' and len(set(header))==len(header)
                and {'user_input','model_output'}<=set(header),'Unexpected ToxicChat CSV columns.')
            needed={header.index('user_input'):'user_input'}
        else:
            require(column==len(header) and current not in records,'Malformed/duplicate ToxicChat row.')
            require(set(payload)=={'conv_id','user_input'},'User-only CSV whitelist escaped.')
            records[current]=payload
        current=None;payload={};column=0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(64*1024),b''):
            for byte in chunk:
                if skip_lf:
                    skip_lf=False
                    if byte==10:continue
                if in_quotes:
                    if pending_quote:
                        if byte==34:
                            if capture():field.append(34)
                            pending_quote=False;continue
                        in_quotes=False;pending_quote=False;closed_quote=True
                    elif byte==34:pending_quote=True;continue
                    else:
                        if capture():field.append(byte)
                        continue
                if at_start and byte==34:in_quotes=True;at_start=False;started=True;continue
                if byte==44:finish_field();at_start=True;closed_quote=False;started=True
                elif byte in [10,13]:
                    require(started or column or field,'Unexpected empty CSV record.')
                    finish_field();finish_record();at_start=True;closed_quote=False;started=False;skip_lf=byte==13
                else:
                    require(not closed_quote and byte!=34,'Malformed CSV quote.')
                    if capture():field.append(byte)
                    at_start=False;started=True
    require(not in_quotes or pending_quote,'Unterminated quoted CSV field.')
    if started or column or field:finish_field();finish_record()
    require(header is not None,'Empty CSV.')
    return records,dict(source_records_framed=len(records),logical_fields_decoded=['conv_id','user_input'],
        assistant_fields_decoded=0,assistant_fields_buffered=0,assistant_content_statistics_computed=False,
        quoted_newlines_and_quotes_supported=True,physical_transport_is_not_logical_text_materialization=True)


def read_user_catalog(plan):
    closed.pin_gate()
    import pyarrow.dataset as ds
    guarded_catalog(plan)
    path=sources.snapshot_path(sources.PINS[0]);metadata=pd.read_parquet(path,columns=METADATA_COLUMNS)
    user=ds.dataset(path,format='parquet').to_table(filter=ds.field('role')=='prompter',
        columns=METADATA_COLUMNS+['text']).to_pandas()
    require(user.role.eq('prompter').all(),'Assistant text entered user-only catalog.')
    toxic,csv_audit=user_csv(sources.snapshot_path(sources.PINS[1]))
    return metadata,user,toxic,dict(oasst_metadata_rows_without_text=len(metadata),
        oasst_logical_user_text_rows=len(user),oasst_assistant_text_rows_materialized=0,
        parquet_physical_blocks_may_include_other_bytes=True,toxicchat=csv_audit,
        new_assistant_materialized=False,model_probabilities_read=False)


def select_sources(user,toxic,legacy,old_keys,tokenizer,per_source=None):
    """Source-only deterministic candidate selection; no assistant fields accepted."""
    per_source=PER_SOURCE if per_source is None else per_source;keyer=sources.ContentKeys(tokenizer)
    require(set(legacy.columns)>={'dataset','conv_id','user_query'} and len(old_keys)==448,'Historical source catalog changed.')
    old_oa_text={sources.normalized(t) for t in legacy[legacy.dataset=='oasst'].user_query}
    old_tc_text={sources.normalized(t) for t in legacy[legacy.dataset=='toxicchat'].user_query}
    legacy_oa_ids=set(user[user.text.map(sources.normalized).isin(old_oa_text)].message_tree_id.astype(str))
    legacy_tc_ids={identity for identity,row in toxic.items() if sources.normalized(row['user_input']) in old_tc_text}
    require(len(legacy_oa_ids)==100 and len(legacy_tc_ids)==108,'All legacy requested native roots were not recovered (100OA/108TC).')
    old_oa={row['native_id'] for row in old_keys if row['dataset']=='oasst'}|legacy_oa_ids
    old_tc={row['native_id'] for row in old_keys if row['dataset']=='toxicchat0124'}|legacy_tc_ids
    require(old_tc<=set(toxic),'Historical ToxicChat source ID absent.')
    histories=list(legacy.user_query)+user[user.message_tree_id.astype(str).isin(old_oa)].text.tolist()
    histories += [toxic[identity]['user_input'] for identity in sorted(old_tc)]
    historical=set().union(*(keyer(text) for text in histories if isinstance(text,str)))
    tree_user={str(identity):frame.to_dict('records') for identity,frame in user.groupby('message_tree_id',sort=False)}
    roots=user[user.parent_id.isna() & user.lang.eq('en') & user.tree_state.eq('ready_for_export')
        & user.deleted.eq(False) & user.synthetic.eq(False) & user.text.str.len().between(100,500)]
    require(roots.message_tree_id.is_unique,'Duplicate OASST root.')
    candidates=[];reasons=Counter()
    for dataset,items in [('oasst',roots.to_dict('records')),('toxicchat0124',list(toxic.values()))]:
        for row in items:
            identity=str(row['message_tree_id'] if dataset=='oasst' else row['conv_id'])
            text=row['text'] if dataset=='oasst' else row['user_input']
            if not 100<=len(text)<=500:continue
            if identity in (old_oa if dataset=='oasst' else old_tc):reasons[dataset+':historical-ID']+=1;continue
            texts=[r['text'] for r in tree_user[identity] if isinstance(r['text'],str)] if dataset=='oasst' else [text]
            keys=set().union(*(keyer(value) for value in texts))
            if keys&historical:reasons[dataset+':historical-user-fingerprint']+=1;continue
            revision=sources.PINS[0 if dataset=='oasst' else 1]['revision']
            record=dict(dataset=dataset,native_id=identity,revision=revision,
                root_message_id=str(row['message_id']) if dataset=='oasst' else None,
                root_raw_sha256=sha(text),root_normalized_sha256=sha(sources.normalized(text)),
                root_tokens=len(tokenizer.encode(text,add_special_tokens=False)),root_characters=len(text),
                source_user_messages=len(texts),exclusion_keys_sha256=canonical_sha(sorted(keys)),
                rank_sha256=sha('\0'.join(map(str,[SELECT_SEED,dataset,revision,identity,sha(text)]))),split='test')
            candidates.append((record,keys))
    selected=[];used=set(historical);counts=Counter()
    for record,keys in sorted(candidates,key=lambda item:(item[0]['rank_sha256'],item[0]['dataset'],item[0]['native_id'])):
        dataset=record['dataset']
        if counts[dataset]>=per_source:continue
        if keys&used:reasons[dataset+':selected-user-fingerprint']+=1;continue
        selected.append(record);used.update(keys);counts[dataset]+=1
    require(counts=={'oasst':per_source,'toxicchat0124':per_source},'Insufficient disjoint user roots; no replacement or source expansion.')
    selected=sorted(selected,key=lambda row:(row['dataset'],row['rank_sha256']))
    for index,row in enumerate(selected):row.update(source_index=index,document_id=DOCUMENT_START+index)
    return selected,dict(eligible_candidate_counts=dict(Counter(row['dataset'] for row,_ in candidates)),
        selected_counts=dict(counts),rejections=dict(reasons),legacy_requested_roots=200,legacy_user_turns=225,
        legacy_native_exclusions={'oasst':100,'toxicchat0124':108},old448_native_exclusions=448,
        historical_user_fingerprint_count=len(historical),historical_keys_sha256=canonical_sha(sorted(historical)),
        ranked_candidates_sha256=canonical_sha([r for r,_ in sorted(candidates,key=lambda item:item[0]['rank_sha256'])]),
        assistant_text_used_for_selection=False,model_probabilities_read=False,no_replacement=True)


def assignments_for(keys,templates):
    assignments=[]
    for dataset in ['oasst','toxicchat0124']:
        group=[key for key in keys if key['dataset']==dataset]
        require(len(group)==64,'Exactly64 roots per dataset required.')
        for index,key in enumerate(group):
            template=templates[index%32]
            assignments.append(dict(dataset=dataset,native_id=key['native_id'],source_uid=uid(key),
                template_id=template['template_id'],family_id=template['family_id'],split='test'))
    require(all(count==2 for count in Counter((a['family_id'],a['dataset']) for a in assignments).values()),
        'System family/source balance changed.')
    return assignments


def gallery_for(keys):
    result=[]
    for dataset in ['oasst','toxicchat0124']:
        group=[key for key in keys if key['dataset']==dataset]
        result += [uid(group[int(index)]) for index in data.uniform_indices(len(group),6)]
    return dict(source_uids=result,selected_before_assistant_materialization=True,
        selected_before_probabilities=True,source_counts={'oasst':6,'toxicchat0124':6},
        quality_failed_keep_placeholders=True,score_based_selection=False)


def freeze_sources(plan,keys,audit):
    closed.pin_gate()
    verify_plan(plan);assignments=assignments_for(keys,plan['templates'])
    values={'natural-source-keys.json':dict(source_keys=keys),
        'system-templates.json':dict(templates=plan['templates'],audit=plan['template_audit']),
        'template-assignments.json':dict(assignments=assignments),'gallery-index.json':gallery_for(keys),
        'source-selection-audit.json':audit}
    for name,value in values.items():immutable_json(OUT/name,value)
    fsync_files([OUT/name for name in SOURCE_ARTIFACTS])
    frozen=dict(plan_sha256=data.digest(OUT/'plan.json'),source_sha256=data.digest(__file__),
        frozen_sha256=plan['source_signature']['frozen_sha256'],artifacts_sha256={name:data.digest(OUT/name) for name in SOURCE_ARTIFACTS},
        fixed_roots=128,fixed_system_families=32,assistant_materialized=False)
    immutable_json(OUT/'source-freeze.json',frozen);fsync_files([OUT/'source-freeze.json'])
    return frozen


def source_selection(plan):
    closed.pin_gate()
    verify_plan(plan);lock=read(OUT/'source-freeze.json')
    require(lock['plan_sha256']==data.digest(OUT/'plan.json') and lock['source_sha256']==data.digest(__file__)
        and lock['frozen_sha256']==plan['source_signature']['frozen_sha256']
        and lock['fixed_roots']==128 and lock['fixed_system_families']==32 and lock['assistant_materialized'] is False
        and set(lock['artifacts_sha256'])==set(SOURCE_ARTIFACTS),'Pre-assistant source freeze changed.')
    for name,value in lock['artifacts_sha256'].items():require(data.digest(OUT/name)==value,'Frozen source artifact changed: '+name)
    keys=read(OUT/'natural-source-keys.json')['source_keys'];templates=read(OUT/'system-templates.json')['templates']
    assignments=read(OUT/'template-assignments.json')['assignments']
    require(len(keys)==128 and len({uid(k) for k in keys})==128
        and Counter(k['dataset'] for k in keys)=={'oasst':64,'toxicchat0124':64}
        and all(k['split']=='test' for k in keys) and templates==plan['templates']
        and assignments==assignments_for(keys,templates) and read(OUT/'gallery-index.json')==gallery_for(keys),
        'Frozen source/template assignments changed.')
    require([k['source_index'] for k in keys]==list(range(128))
        and all(k['document_id']==DOCUMENT_START+i and k['revision']==sources.PINS[0 if k['dataset']=='oasst' else 1]['revision']
            and k['rank_sha256']==sha('\0'.join(map(str,[SELECT_SEED,k['dataset'],k['revision'],k['native_id'],k['root_raw_sha256']])))
            for i,k in enumerate(keys)), 'Frozen source rank/document/revision identity changed.')
    old={uid(k) for k in read(OLD_FOLDER/'natural-source-keys.json')['source_keys']}
    require(not {uid(k) for k in keys}&old,'Historical448 native root reused.')
    return keys,templates,assignments


def read_selected(keys,templates,assignments,plan,scope):
    """Own selected reader; no false old test-decoded flags are inherited."""
    closed.pin_gate()
    import pyarrow.dataset as ds
    require(scope in ['fresh-selected','historical448-reference'],'Invalid source-read scope.')
    selected_keys,_,_=source_selection(plan)  # Mandatory before all assistant reads.
    if scope=='fresh-selected':require(keys==selected_keys,'Fresh assistant reader escaped frozen IDs.')
    else:require(keys==read(OLD_FOLDER/'natural-source-keys.json')['source_keys'],'Historical reference scope escaped448 IDs.')
    oa={k['native_id'] for k in keys if k['dataset']=='oasst'};tc={k['native_id'] for k in keys if k['dataset']=='toxicchat0124'}
    path=sources.snapshot_path(sources.PINS[0]);metadata=pd.read_parquet(path,columns=METADATA_COLUMNS)
    require(oa<=set(metadata.message_tree_id.astype(str)),'Selected OASST tree absent.')
    frame=ds.dataset(path,format='parquet').to_table(filter=ds.field('message_tree_id').isin(sorted(oa)),
        columns=METADATA_COLUMNS+['text']).to_pandas()
    require(set(frame.message_tree_id.astype(str))==oa,'Unselected OASST text materialized.')
    by_tree={str(identity):group.to_dict('records') for identity,group in frame.groupby('message_tree_id',sort=False)}
    toxic,csv_audit=pure.selected_csv(sources.snapshot_path(sources.PINS[1]),tc)
    template_map={row['template_id']:row for row in templates};assignment_map={(row['dataset'],row['native_id']):row for row in assignments}
    records=[]
    for key in keys:
        assignment=assignment_map[(key['dataset'],key['native_id'])];template=template_map[assignment['template_id']]
        messages=[dict(role='system',text=template['text'],source_message_id=template['template_id'],
            origin='new fixed system wording; not source generation context',source_model=None)]
        if key['dataset']=='oasst':
            path_rows,failure=pure.choose_oa_path(by_tree[key['native_id']],key['root_message_id'])
            for row in path_rows:messages.append(dict(role='user' if row['role']=='prompter' else 'assistant',text=row['text'],
                source_message_id=str(row['message_id']),origin='public OASST non-synthetic text',source_model=row['model_name'],
                source_synthetic=bool(row['synthetic']),source_deleted=bool(row['deleted'])))
        else:
            item=toxic.get(key['native_id'])
            failure=('missing-public-source-ID' if item is None else
                None if item['user_input'].strip() and item['model_output'].strip() else 'empty-user-or-recorded-assistant')
            if item:
                for role,field in [('user','user_input'),('assistant','model_output')]:messages.append(dict(role=role,text=item[field],
                    source_message_id=key['native_id']+'/'+field,origin='public ToxicChat user' if role=='user' else 'recorded Vicuna-demo output; exact checkpoint unknown',
                    source_model=None if role=='user' else 'Vicuna-demo; exact checkpoint unknown'))
        if not failure:require(sha(messages[1]['text'])==key['root_raw_sha256'],'Selected root raw hash changed.')
        records.append(dict(source_uid=(uid(key) if scope=='fresh-selected' else pure.uid(key)),dataset=key['dataset'],native_id=key['native_id'],split=key['split'],
            revision=key['revision'],root_raw_sha256=key['root_raw_sha256'],template_id=assignment['template_id'],family_id=assignment['family_id'],
            source_snapshot_sha256=sources.PINS[0 if key['dataset']=='oasst' else 1]['sha256'],
            document_id=key.get('document_id'),messages=messages,failure=failure))
    return records,dict(scope=scope,source_freeze_sha256=data.digest(OUT/'source-freeze.json'),
        logical_source_uids_sha256=canonical_sha([(uid(k) if scope=='fresh-selected' else pure.uid(k)) for k in keys]),oasst_metadata_rows_without_text=len(metadata),
        oasst_logical_text_rows_materialized=len(frame),oasst_nonselected_text_rows_materialized=0,
        parquet_physical_blocks_may_contain_nonselected_bytes=True,toxicchat=csv_audit,
        fresh_assistant_materialized=scope=='fresh-selected',historical448_assistant_materialized=scope=='historical448-reference',
        old_development_reader_called=False,independent_subset_reader=True)


def native_role_bodies(prompt,length,tokenizer):
    """Full native message bodies from actual token spans, never sparse rows."""
    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=encoded['input_ids'];require(len(ids)==int(length),'Prepared native prompt length changed.')
    labels=native.parse_native_labels(ids,tokenizer);offsets=encoded['offset_mapping'];frames=[];covered={}
    pattern=re.compile(r'<\|start\|>(.*?)<\|message\|>(.*?)(<\|end\|>|<\|return\|>|<\|call\|>)',re.S)
    matches=list(pattern.finditer(prompt));require(len(matches)==prompt.count('<|start|>'),'Prepared native frame structure incomplete.')
    for message_ix,match in enumerate(matches):
        head,body,end=match.groups();role=('tool' if head.startswith('functions.') else 'cot' if head.startswith('assistant') and '<|channel|>analysis' in head else 'assistant' if head.startswith('assistant') else head)
        require(role in data.ROLES,'Unknown prepared native frame role.')
        left,right=match.span(2);positions=[]
        for position,(a,b) in enumerate(offsets):
            if a<right and b>left:
                require(a>=left and b<=right,'Prepared token crosses body/header frame.');positions.append(position);covered[position]=role
        if not positions:require(not body,'Nonempty prepared body has no tokens.');continue
        require(positions==list(range(positions[0],positions[-1]+1)) and all(labels.get(i)==role for i in positions),'Prepared role segment disagrees with independent native parser.')
        bodyids=ids[positions[0]:positions[-1]+1];decoded=tokenizer.decode(bodyids,skip_special_tokens=False)
        require(decoded==body and tokenizer.encode(decoded,add_special_tokens=False)==bodyids,'Prepared full-body token decode/roundtrip differs.')
        frames.append(dict(message_ix=message_ix,role=role,body=decoded,body_tokens=len(bodyids),token_start=positions[0],token_end_exclusive=positions[-1]+1,structural_tool_call='to=functions.' in head))
    require(covered==labels,'Prepared native content segments lost or duplicated.')
    return frames


def prepared_role_references(plan,tokenizer):
    """Materialize known historical native bodies only after fresh128-ID freeze."""
    closed.pin_gate();source_selection(plan)
    import pyarrow.dataset as ds
    refs=[];counts={};proof_sources=plan['source_signature']['prepared_role_inputs']
    require(proof_sources==prepared_role_metadata(),'Prepared historical role-input identity changed.')
    for name,source in proof_sources.items():
        frame=ds.dataset(Path(source['cache_path'])/'inputs.parquet',format='parquet').to_table(filter=ds.field('wrapper')=='native',columns=['prompt_ix','prompt','length','wrapper','split']).to_pandas().sort_values('prompt_ix')
        require(frame.prompt_ix.is_unique and frame.wrapper.eq('native').all(),'Historical native prompt ownership differs.')
        role_counts=Counter();body_tokens=0
        for row in frame.itertuples():
            for message in native_role_bodies(row.prompt,row.length,tokenizer):
                identity='prepared-role:'+name+':prompt'+str(row.prompt_ix)+'/message'+str(message['message_ix']);body=message['body']
                refs.append(dict(message_uid=identity,source_uid='prepared-role:'+name+':prompt'+str(row.prompt_ix),split='historical-role-input',message_ix=message['message_ix'],role=message['role'],keys=pure.assistant_keys(body,body,body,tokenizer)))
                role_counts[message['role']]+=1;body_tokens+=message['body_tokens']
        require(len(frame)>0 and role_counts,'Prepared historical native source is empty: '+name)
        counts[name]=dict(native_prompts=len(frame),body_segments=sum(role_counts.values()),role_counts=dict(role_counts),body_tokens=body_tokens)
    proof=dict(sources=proof_sources,counts=counts,reference_nodes=len(refs),reference_keys_sha256=canonical_sha([dict(message_uid=r['message_uid'],role=r['role'],keys=sorted(r['keys'])) for r in refs]),
        source_freeze_sha256=data.digest(OUT/'source-freeze.json'),fresh_ids_selected_before_historical_body_materialization=True,body_scope='Full actual native role-input bodies including middle chunks; all prepared splits; no sparse-row assembly.',
        raw_prompt_or_body_text_persisted=False,semantic_independence_claimed=False,model_probabilities_read=False)
    return refs,proof


def duplicate_audit(ledger,assistant,fresh_messages,reference,old_index,old_proof,prepared_reference,prepared_proof):
    """Preserve prior assistant graph, then add S/U/A historical-role bridges."""
    combined=reference+prepared_reference;result=previous.duplicate_audit(ledger,assistant,combined,old_index,old_proof)
    oldroles={r['message_uid']:r for r in prepared_reference};role_index=defaultdict(set)
    for row in prepared_reference:
        for key in row['keys']:role_index[key].add(row['message_uid'])
    # Every public/role/generated external reference participates in one graph;
    # S/U edges enter only through historical-role matches, never invented labels.
    external=defaultdict(set)
    for row in combined:
        for key in row['keys']:external[key].add(row['message_uid'])
    for key,refs in old_index.items():external[key].update('old-reference:'+ref for ref in refs)
    nodes=set().union(*external.values()) if external else set();graph=pure.Components(nodes)
    for refs in external.values():graph.join(refs)
    mapping={r['source_uid']:r for r in ledger};source_bridges=defaultdict(set);message_flags=[]
    for row in ledger:
        row.update(historical_role_input_collision=False,historical_role_input_collision_roles=[],historical_role_input_references=[],
            public448_assistant_collision=any(not ref.startswith('prepared-role:') for ref in row['dev_assistant_references']))
    for message in fresh_messages:
        matching=set().union(*(role_index[key] for key in message['keys'] if key in role_index)) if message['keys'] else set()
        row=mapping[message['source_uid']]
        if matching:
            row['historical_role_input_collision']=True;row['historical_role_input_collision_roles'].append(message['role']);row['historical_role_input_references'].extend(matching)
            for ref in matching:source_bridges[graph.find(ref)].add(row['source_uid'])
        if message['role']=='assistant':
            for key in message['keys']:
                for ref in external.get(key,[]):source_bridges[graph.find(ref)].add(row['source_uid'])
        message_flags.append(dict(message_uid=message['message_uid'],source_uid=message['source_uid'],role=message['role'],historical_role_reference_ids=sorted(matching),collision=bool(matching),keys_sha256=canonical_sha(sorted(message['keys']))))
    prepared=[r for r in ledger if r['status']=='prepared'];components=pure.Components([r['source_uid'] for r in prepared]);families=defaultdict(list)
    for row in prepared:families[row['family_id']].append(row['source_uid'])
    for refs in families.values():components.join(refs)
    for cluster in result['duplicate_clusters']:components.join(cluster['test_source_uids'])
    for refs in source_bridges.values():components.join(refs)
    for row in ledger:
        row['historical_role_input_collision_roles']=sorted(set(row['historical_role_input_collision_roles']));row['historical_role_input_references']=sorted(set(row['historical_role_input_references']))
        row['eligible_clean_test']=row['status']=='prepared' and not row['old_generated_collision'] and not row['dev_assistant_collision'] and not row['historical_role_input_collision']
        if row['historical_role_input_collision']:row['eligibility_reasons'].append('content-collision-with-prepared-historical-role-input')
    groups=[]
    for refs in components.groups():
        component_id='test-dependency-'+canonical_sha(refs)[:16]
        for identity in refs:mapping[identity]['dependency_component_id']=component_id
        groups.append(dict(component_id=component_id,source_uids=refs,clean_source_uids=[identity for identity in refs if mapping[identity]['eligible_clean_test']],system_family_ids=sorted({mapping[identity]['family_id'] for identity in refs})))
    result.update(components=groups,dependency_components=len(groups),clean_dependency_components=sum(bool(g['clean_source_uids']) for g in groups),
        historical_role_input_reference=prepared_proof,historical_role_input_reference_nodes=len(prepared_reference),historical_role_input_message_flags=message_flags,
        external_reference_bridges=[dict(external_component=component,source_uids=sorted(refs)) for component,refs in sorted(source_bridges.items()) if len(refs)>1],
        dev_assistant_collision_scope='Compatibility flag: assistant collisions with448 public references OR prepared native role-input references. Separate public448 and historical-role flags disambiguate.',
        system_family_counts=dict(fixed=32,all_prepared=len(families),clean_prepared=len({r['family_id'] for r in prepared if r['eligible_clean_test']})),
        semantic_independence_claimed=False,no_ids_replaced=True)
    result['eligibility_counts'].update(eligible_clean_test=sum(r['eligible_clean_test'] for r in ledger),historical_role_input_collision=sum(r['historical_role_input_collision'] for r in ledger),
        historical_role_input_collision_by_new_role={role:sum(role in r['historical_role_input_collision_roles'] for r in ledger) for role in ROLES})
    return result


def build_dataset(records,tokenizer,reference,old_index,old_proof,prepared_reference=None,prepared_proof=None):
    ledger=[];inputs=[];rows=[];tokens=[];assistant=[];fresh_messages=[]
    prepared_reference=[] if prepared_reference is None else prepared_reference;prepared_proof={} if prepared_proof is None else prepared_proof
    for source_index,record in enumerate(records):
        require(record['split']=='test','Non-test fresh source.')
        processed,messages,failure=previous.processed_messages(record,tokenizer)
        entry={key:value for key,value in record.items() if key not in ['messages','failure']}
        entry.update(status='quality-failed' if failure else 'prepared',quality_failure=failure,messages=messages,prompt_ix=None,turns=0)
        if not failure:
            prompt,spans=pure.render_native([(role,body) for role,body,clean,raw in processed])
            ids,content=pure.content_rows(prompt,spans,tokenizer);prompt_ix=len(inputs);entry['prompt_ix']=prompt_ix
            entry['turns']=sum(role=='user' for role,body,clean,raw in processed)
            for index,(role,body,clean,raw) in enumerate(processed):
                require(sum(r['message_ix']==index for r in content)==messages[index]['body_tokens'],'Body token count differs from ledger.')
                message=dict(message_uid=entry['source_uid']+'/message'+str(index),source_uid=entry['source_uid'],split='test',message_ix=index,role=role,keys=pure.assistant_keys(raw,clean,body,tokenizer));fresh_messages.append(message)
                if role=='assistant':assistant.append(message)
            common={key:entry[key] for key in ['source_uid','dataset','native_id','split','template_id','family_id','document_id']}
            common.update(uid=f'complementary-natural:{prompt_ix}',source_set=SOURCE_SET,group_id=entry['source_uid'],wrapper='native',profile='natural-public-sua')
            inputs.append(dict(common,prompt_ix=prompt_ix,prompt=prompt,length=len(ids),turns=entry['turns'],message_count=len(processed)))
            tokens.append(ids);offsets=Counter()
            for row in content:
                rows.append(dict(common,cache_ix=len(rows),sample_ix=SAMPLE_START+len(rows),prompt_ix=prompt_ix,
                    body_offset=offsets[row['message_ix']],**row));offsets[row['message_ix']]+=1
        ledger.append(entry)
    audit=duplicate_audit(ledger,assistant,fresh_messages,reference,old_index,old_proof,prepared_reference,prepared_proof)
    audit.update(reference_scope='Historical448 public assistants plus full native messages from four explicit prepared role-input caches; dev_* assistant field retained for compatibility; S/U/A historical-role flags separate',
        maximum_independent_positive_stimuli=32,no_outcome_based_filtering=True)
    mapping={row['source_uid']:row for row in ledger}
    flags=['eligible_clean_test','old_generated_collision','dev_assistant_collision','within_test_assistant_duplicate','dependency_component_id','historical_role_input_collision','public448_assistant_collision']
    for row in inputs+rows:
        for field in flags:row[field]=mapping[row['source_uid']][field]
    common=['uid','source_set','source_uid','dataset','native_id','split','template_id','family_id','document_id','group_id','wrapper','profile']
    inputs=pd.DataFrame(inputs,columns=['prompt_ix','prompt','length','turns','message_count']+common+flags)
    rows=pd.DataFrame(rows,columns=['cache_ix','sample_ix','prompt_ix','body_offset','token_ix','token_id','token','message_ix','role','semantic_role']+common+flags)
    coverage=dict(all_content_tokens_scored=True,independent_native_labels_match=True,no_sparse_sampling=True,
        input_prompts=len(inputs),content_rows=len(rows),role_counts={role:int(rows.role.eq(role).sum()) for role in ROLES},
        no_observations=len(inputs)==0,roles_with_no_observations={role:dict(support=0,recall=None,status='N/A') for role in ['cot','tool']},
        sample_ix_start=SAMPLE_START,sample_ix_end_exclusive=SAMPLE_START+len(rows),normal_system_tag=True)
    return inputs,rows,tokens,ledger,audit,coverage


def verify_cached_content(inputs,rows,tokens,ledger,tokenizer):
    require(inputs.prompt_ix.tolist()==list(range(len(inputs))) and len(tokens)==len(inputs)
        and rows.cache_ix.tolist()==list(range(len(rows))) and rows.sample_ix.tolist()==list(SAMPLE_START+np.arange(len(rows))),
        'Fresh natural row identity changed.')
    require(len(ledger)==128 and len({r['source_uid'] for r in ledger})==128
        and len(inputs)==sum(r['status']=='prepared' for r in ledger) and set(rows.role)<=set(ROLES)
        and set(inputs.split)<={'test'} and set(rows.split)<={'test'},'Fresh natural ledger/roles changed.')
    mapping={r['prompt_ix']:r for r in ledger if r['status']=='prepared'}
    lookup={ix:g for ix,g in rows.groupby('prompt_ix',sort=False)}
    pattern=re.compile(r'<\|start\|>(.*?)<\|message\|>(.*?)(<\|end\|>|<\|return\|>)',re.S)
    for entry in inputs.itertuples():
        ids=tokenizer.encode(entry.prompt,add_special_tokens=False);group=lookup[entry.prompt_ix];record=mapping[entry.prompt_ix]
        labels=native.parse_native_labels(ids,tokenizer)
        require(ids==tokens[entry.prompt_ix] and len(ids)==entry.length
            and dict(zip(group.token_ix,group.role))==labels and len(group)==len(labels)
            and all(ids[r.token_ix]==r.token_id for r in group.itertuples()),'Native S/U/A labels or complete content coverage changed.')
        bodies=pattern.findall(entry.prompt)
        roles=[m['role'] for m in record['messages']]
        require(roles in [['system','user','assistant'],['system','user','assistant','user','assistant']]
            and len(bodies)==len(record['messages'])==entry.message_count
            and entry.turns==sum(role=='user' for role in roles)
            and entry.prompt.startswith('<|startoftext|>'), 'Native genuine S/U/A path/message count differs from ledger.')
        for index,(header,body,ending) in enumerate(bodies):
            message=record['messages'][index];role=message['role'];part=group[group.message_ix==index]
            require(header==('assistant<|channel|>final' if role=='assistant' else role)
                and ending==('<|return|>' if role=='assistant' else '<|end|>')
                and sha(body)==message['serialized_body_sha256'] and len(part)==message['body_tokens']<=192
                and part.body_offset.tolist()==list(range(len(part))),'Native capped body/message/offset changed.')
        require(entry.uid==f'complementary-natural:{entry.prompt_ix}' and entry.source_set==SOURCE_SET
            and entry.source_uid==entry.group_id==record['source_uid'],'Fresh source namespace changed.')
        require(group.semantic_role.eq(group.role).all(),'Formal/semantic S/U/A labels differ.')
        for field in ['source_uid','dataset','native_id','split','template_id','family_id','document_id','dependency_component_id',
            'eligible_clean_test','old_generated_collision','dev_assistant_collision','within_test_assistant_duplicate','historical_role_input_collision','public448_assistant_collision']:
            require(getattr(entry,field)==record[field] and group[field].eq(record[field]).all(),'Ledger/input/row mapping changed: '+field)
        for field in ['uid','source_set','group_id','wrapper','profile']:
            require(group[field].eq(getattr(entry,field)).all(),'Input/row namespace changed: '+field)


def verify_summary(inputs,rows,ledger,duplicates,coverage):
    """Recount observations/families/components, including all-failed sources."""
    prepared=[r for r in ledger if r['status']=='prepared'];clean=[r for r in prepared if r['eligible_clean_test']]
    require(set(r['status'] for r in ledger)<={'prepared','quality-failed'} and len({r['family_id'] for r in ledger})==32
        and all(r['eligible_clean_test']==(r['status']=='prepared' and not r['old_generated_collision'] and not r['dev_assistant_collision'] and not r['historical_role_input_collision'])
            for r in ledger), 'Status/clean flag/fixed family semantics changed.')
    groups=duplicates['components'];mapping={r['source_uid']:r for r in prepared}
    require(len({r['source_uid'] for r in ledger})==128 and len(prepared)==len(inputs)
        and set().union(*(set(g['source_uids']) for g in groups))==set(mapping), 'Dependency component membership changed.')
    require(sum(len(g['source_uids']) for g in groups)==len(mapping),'Prepared source duplicated across components.')
    for group in groups:
        require(group['clean_source_uids']==[node for node in group['source_uids'] if mapping[node]['eligible_clean_test']]
            and group['system_family_ids']==sorted({mapping[node]['family_id'] for node in group['source_uids']})
            and all(mapping[node]['dependency_component_id']==group['component_id'] for node in group['source_uids']),
            'Clean/family dependency component changed.')
    require(duplicates['dependency_components']==len(groups)
        and duplicates['clean_dependency_components']==sum(bool(g['clean_source_uids']) for g in groups)
        and duplicates['system_family_counts']==dict(fixed=32,all_prepared=len({r['family_id'] for r in prepared}),
            clean_prepared=len({r['family_id'] for r in clean})),'Active family/component counts changed.')
    counts=duplicates['eligibility_counts']
    require(counts['all_fixed_ids']==128 and counts['prepared']==len(prepared) and counts['quality_failed']==128-len(prepared)
        and counts['eligible_clean_test']==len(clean),'All/clean/status counts changed.')
    require(coverage['all_content_tokens_scored'] is True and coverage['independent_native_labels_match'] is True
        and coverage['input_prompts']==len(inputs) and coverage['content_rows']==len(rows)
        and coverage['no_observations']==(len(inputs)==0) and coverage['role_counts']=={r:int(rows.role.eq(r).sum()) for r in ROLES}
        and coverage['sample_ix_start']==SAMPLE_START and coverage['sample_ix_end_exclusive']==SAMPLE_START+len(rows),
        'Coverage/role support counts changed.')


def prepare():
    closed.pin_gate()
    require((OUT/'plan.json').is_file(),'Explicit source declaration required before prepare.')
    plan=read(OUT/'plan.json');verify_plan(plan)
    if (OUT/'data-complete.json').exists():return load_verified_data()[0]
    require(not (OUT/'inference').exists() and not (CACHE/'inference').exists(),'Inference state predates preparation completion.')
    require(shutil.disk_usage(data.CACHE).free>=128*2**20,'Need128MiB free; do not copy snapshots or delete banks.')
    tokenizer=load_tokenizer()
    if not (OUT/'source-freeze.json').exists():
        _,user,toxic,catalog_audit=read_user_catalog(plan)
        legacy=pd.read_csv(data.HERE/sources.OLD_CSVS['queries'][0],keep_default_na=False,
            usecols=['dataset','conv_id','user_query_ix','user_query'])
        require(len(legacy)==225 and legacy.conv_id.nunique()==200,'Legacy200 requested source inventory changed.')
        keys,audit=select_sources(user,toxic,legacy,read(OLD_FOLDER/'natural-source-keys.json')['source_keys'],tokenizer)
        freeze_sources(plan,keys,dict(audit,user_catalog=catalog_audit))
    keys,templates,assignments=source_selection(plan)
    old_keys=read(OLD_FOLDER/'natural-source-keys.json')['source_keys']
    old_templates=read(OLD_FOLDER/'system-templates.json')['templates']
    old_assignments=read(OLD_FOLDER/'template-assignments.json')['assignments']
    refs,reference_read=read_selected(old_keys,old_templates,old_assignments,plan,'historical448-reference')
    expected=read(pure.OUT/'dev-ledger.json')['records']+read(previous.OUT/'test-ledger.json')['records']
    reference=previous.reference_assistants(refs,tokenizer,expected)
    old_index,old_proof=pure.legacy_generated_keys(tokenizer)
    role_reference,role_read=prepared_role_references(plan,tokenizer)
    records,fresh_read=read_selected(keys,templates,assignments,plan,'fresh-selected')
    dataset=build_dataset(records,tokenizer,reference,old_index,old_proof,role_reference,role_read)
    return save_result(plan,keys,*dataset,dict(historical448=reference_read,prepared_role_inputs=role_read,fresh=fresh_read,
        fresh_assistant_materialized=True,gpu_work_run=False,model_probabilities_read=False))


def save_result(plan,keys,inputs,rows,tokens,ledger,duplicates,coverage,source_read):
    closed.pin_gate()
    verify_cached_content(inputs,rows,tokens,ledger,load_tokenizer());verify_plan(plan)
    verify_summary(inputs,rows,ledger,duplicates,coverage)
    require([r['source_uid'] for r in ledger]==[uid(k) for k in keys],'All128 frozen source IDs must remain in ledger order.')
    CACHE.mkdir(parents=True,exist_ok=True)
    owner=dict(plan_sha256=data.digest(OUT/'plan.json'),source_sha256=data.digest(__file__),source_freeze_sha256=data.digest(OUT/'source-freeze.json'))
    if not (CACHE/'initialized.json').exists():require(not any(CACHE.iterdir()),'Unowned partial natural cache.')
    immutable_json(CACHE/'initialized.json',owner)
    pure.save_parquet(CACHE/'inputs.parquet',inputs);pure.save_parquet(CACHE/'rows.parquet',rows);immutable_json(CACHE/'token-ids.json',tokens)
    require(sum((CACHE/name).stat().st_size for name in FILES)<=128*2**20,'Natural cache exceeds128MiB cap.')
    for name,value in [('source-read-audit.json',source_read),('test-ledger.json',dict(records=ledger,all_fixed_ids_retained=True,no_replacements=True)),
        ('duplicate-audit.json',duplicates),('coverage.json',coverage)]:immutable_json(OUT/name,value)
    signature=plan['source_signature']
    complete=dict(plan_sha256=data.digest(OUT/'plan.json'),source_sha256=data.digest(__file__),
        dependencies_sha256=plan['dependencies_sha256'],source_signature=signature,frozen=plan['frozen'],
        frozen_sha256=signature['frozen_sha256'],snapshot_path=str(SNAPSHOT),snapshot_file_sha256=signature['snapshot_file_sha256'],
        source_freeze_sha256=data.digest(OUT/'source-freeze.json'),source_keys_sha256=data.digest(OUT/'natural-source-keys.json'),
        cache_path=str(CACHE),cache_sha256={name:data.digest(CACHE/name) for name in FILES},
        metadata_sha256={name:data.digest(OUT/name) for name in SOURCE_ARTIFACTS+AUDITS+['source-freeze.json']},
        ledger_path=str(OUT/'test-ledger.json'),ledger_sha256=data.digest(OUT/'test-ledger.json'),
        duplicate_audit_path=str(OUT/'duplicate-audit.json'),duplicate_audit_sha256=data.digest(OUT/'duplicate-audit.json'),
        fixed_roots=128,fixed_system_families=32,prepared_prompts=len(inputs),inputs=len(inputs),content_rows=len(rows),rows=len(rows),
        input_tokens=int(inputs.length.sum()),eligibility_counts=duplicates['eligibility_counts'],components=duplicates['components'],
        system_family_counts=duplicates['system_family_counts'],dependency_components=duplicates['dependency_components'],
        clean_dependency_components=duplicates['clean_dependency_components'],coverage=coverage,no_observations=len(inputs)==0,
        gallery_source_uids=read(OUT/'gallery-index.json')['source_uids'],test_assistant_materialized=True,
        source_replies_generated_under_system=False,source_replies_generated_by_120b=False,
        model_generation_run=False,gpu_work_run=False,downloads_run=False,classifier_probabilities_read=False,
        stored_validation_probabilities_read_for_completion_verification=False,test_classifier_probabilities_read=False,
        test_metrics_read=False,primary_changed=False)
    immutable_json(OUT/'data-complete.json',complete);return complete


def verified_data():
    """Static closure/cache verification, no source catalog, assistant or p read."""
    closed.pin_gate()
    plan=read(OUT/'plan.json');verify_plan(plan);keys,_,_=source_selection(plan);complete=read(OUT/'data-complete.json')
    signature=plan['source_signature']
    require(complete['plan_sha256']==data.digest(OUT/'plan.json') and complete['source_sha256']==data.digest(__file__)
        and complete['dependencies_sha256']==plan['dependencies_sha256'] and complete['source_signature']==signature
        and complete['frozen']==plan['frozen'] and complete['frozen_sha256']==signature['frozen_sha256']
        and complete['snapshot_path']==str(SNAPSHOT) and complete['snapshot_file_sha256']==signature['snapshot_file_sha256']
        and complete['cache_path']==str(CACHE) and complete['source_keys_sha256']==data.digest(OUT/'natural-source-keys.json')
        and complete['source_freeze_sha256']==data.digest(OUT/'source-freeze.json'),'Completed source/closure identity changed.')
    require(set(complete['cache_sha256'])==set(FILES)
        and set(complete['metadata_sha256'])==set(SOURCE_ARTIFACTS+AUDITS+['source-freeze.json']),'Incomplete cache/audit manifest.')
    require(read(CACHE/'initialized.json')==dict(plan_sha256=complete['plan_sha256'],source_sha256=complete['source_sha256'],
        source_freeze_sha256=complete['source_freeze_sha256']),'Natural cache ownership changed.')
    for name,value in complete['cache_sha256'].items():require(data.digest(CACHE/name)==value,'Natural cache changed: '+name)
    for name,value in complete['metadata_sha256'].items():require(data.digest(OUT/name)==value,'Natural audit changed: '+name)
    for field in ['test_assistant_materialized']:require(complete[field] is True,'Required source-read flag changed.')
    for field in ['source_replies_generated_under_system','source_replies_generated_by_120b','model_generation_run','gpu_work_run',
        'downloads_run','classifier_probabilities_read','stored_validation_probabilities_read_for_completion_verification',
        'test_classifier_probabilities_read','test_metrics_read','primary_changed']:require(complete[field] is False,'Source-only boundary changed: '+field)
    inputs=pd.read_parquet(CACHE/'inputs.parquet');rows=pd.read_parquet(CACHE/'rows.parquet');tokens=read(CACHE/'token-ids.json')
    ledger=read(OUT/'test-ledger.json')['records'];duplicates=read(OUT/'duplicate-audit.json');coverage=read(OUT/'coverage.json')
    verify_cached_content(inputs,rows,tokens,ledger,load_tokenizer())
    verify_summary(inputs,rows,ledger,duplicates,coverage)
    require([r['source_uid'] for r in ledger]==[uid(k) for k in keys] and complete['fixed_roots']==128
        and complete['fixed_system_families']==32 and complete['inputs']==complete['prepared_prompts']==len(inputs)
        and complete['rows']==complete['content_rows']==len(rows) and complete['input_tokens']==int(inputs.length.sum())
        and complete['coverage']==coverage and complete['eligibility_counts']==duplicates['eligibility_counts']
        and complete['components']==duplicates['components'] and complete['system_family_counts']==duplicates['system_family_counts']
        and complete['dependency_components']==duplicates['dependency_components']
        and complete['clean_dependency_components']==duplicates['clean_dependency_components']
        and complete['no_observations']==(len(inputs)==0) and complete['gallery_source_uids']==gallery_for(keys)['source_uids'],
        'Completed natural counts/components/coverage changed.')
    return complete,inputs,rows,tokens,ledger


def load_verified_data():
    closed.pin_gate()
    return verified_data()[:4]


def verify():
    closed.pin_gate()
    complete,inputs,rows,tokens,ledger=verified_data();plan=read(OUT/'plan.json');tokenizer=load_tokenizer()
    _,user,toxic,catalog=read_user_catalog(plan)
    legacy=pd.read_csv(data.HERE/sources.OLD_CSVS['queries'][0],keep_default_na=False,usecols=['dataset','conv_id','user_query_ix','user_query'])
    keys,audit=select_sources(user,toxic,legacy,read(OLD_FOLDER/'natural-source-keys.json')['source_keys'],tokenizer)
    require(keys==read(OUT/'natural-source-keys.json')['source_keys']
        and dict(audit,user_catalog=catalog)==read(OUT/'source-selection-audit.json'),'User-only source replay changed.')
    # Explicit replay is allowed only as a root stage and never used by inference.
    new_keys,templates,assignments=source_selection(plan)
    refkeys=read(OLD_FOLDER/'natural-source-keys.json')['source_keys']
    refs,refaudit=read_selected(refkeys,read(OLD_FOLDER/'system-templates.json')['templates'],
        read(OLD_FOLDER/'template-assignments.json')['assignments'],plan,'historical448-reference')
    reference=previous.reference_assistants(refs,tokenizer,read(pure.OUT/'dev-ledger.json')['records']+read(previous.OUT/'test-ledger.json')['records'])
    oldindex,oldproof=pure.legacy_generated_keys(tokenizer)
    role_reference,role_read=prepared_role_references(plan,tokenizer)
    records,newaudit=read_selected(new_keys,templates,assignments,plan,'fresh-selected')
    replay=build_dataset(records,tokenizer,reference,oldindex,oldproof,role_reference,role_read)
    pd.testing.assert_frame_equal(inputs,replay[0],check_exact=True);pd.testing.assert_frame_equal(rows,replay[1],check_exact=True)
    require(tokens==replay[2] and ledger==replay[3] and replay[4]==read(OUT/'duplicate-audit.json')
        and replay[5]==complete['coverage'] and read(OUT/'source-read-audit.json')==dict(historical448=refaudit,prepared_role_inputs=role_read,fresh=newaudit,
            fresh_assistant_materialized=True,gpu_work_run=False,model_probabilities_read=False),'Assistant source replay changed.')
    return dict(data_complete_sha256=data.digest(OUT/'data-complete.json'),verified_source_replay=True,rows=len(rows),gpu_work_run=False)


@contextmanager
def patched(**values):
    original={name:globals()[name] for name in values};globals().update(values)
    try:yield
    finally:globals().update(original)


class FakeTokenizer:
    """Synthetic reversible word/space/marker tokenizer; no checkpoint access."""
    pattern=re.compile(r'<\|[^>]*?\|>|\s+|[A-Za-z_]+|[0-9]+|[^\w\s]',re.U)
    def __init__(self):
        self.vocab={};self.reverse={}
        for marker in native.SPECIAL:self.convert_tokens_to_ids(marker)
    def convert_tokens_to_ids(self,token):
        if token not in self.vocab:
            index=len(self.vocab)+100;self.vocab[token]=index;self.reverse[index]=token
        return self.vocab[token]
    def encode(self,text,add_special_tokens=False):return [self.convert_tokens_to_ids(m.group()) for m in self.pattern.finditer(text)]
    def decode(self,ids,skip_special_tokens=False):return ''.join(self.reverse[int(i)] for i in ids)
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        spans=list(self.pattern.finditer(text));result=dict(input_ids=[self.convert_tokens_to_ids(m.group()) for m in spans])
        if return_offsets_mapping:result['offset_mapping']=[m.span() for m in spans]
        return result


def preflight():
    from unittest.mock import patch
    # Only the NEW helper's pure gate is temporarily replaced, fully restored.
    # No frozen old modules/tokenizer/catalog globals are reassigned.
    initial=(closed.pin_gate,native.load_tokenizer,pure.OUT,pure.CACHE,previous.OUT,previous.CACHE)
    calls=[]
    def blocked():calls.append('gate');raise RuntimeError('Synthetic unknown pins block')
    def forbidden(*a,**k):raise LookupError('Formal source/cache/template read forbidden.')
    with patch.object(closed,'pin_gate',blocked),patched(read=forbidden):
        entrypoints=[prepared_role_metadata,lambda:prepared_role_references({},FakeTokenizer()),source_signature,plan_value,declare,prepare,verified_data,load_verified_data,verify,old_template_catalog,load_tokenizer,lambda:freeze_sources({},[],{}),lambda:read_user_catalog({}),lambda:read_selected([],[],[],{},'fresh-selected')]
        for action in entrypoints:
            before=len(calls)
            try:action()
            except RuntimeError:require(len(calls)==before+1,'Actual entry failed before its helper gate.')
            else:raise RuntimeError('Unknown helper pins did not block an actual entry.')
    with patch.object(closed,'pin_gate',lambda:None),patched(old_template_catalog=forbidden,load_tokenizer=forbidden):output,report=_synthetic_preflight()
    require(initial==(closed.pin_gate,native.load_tokenizer,pure.OUT,pure.CACHE,previous.OUT,previous.CACHE),'Frozen module/helper globals not restored.')
    report['checks']=['unknown_helper_pins_block_actual_entry_'+str(i) for i in range(len(calls))]+report['checks']+['new_helper_gate_and_frozen_module_globals_restored']
    report['checks_passed']=len(report['checks']);report['unknown_pin_entrypoints_checked']=len(calls);report['new_helper_gate_restored']=True;report['frozen_module_globals_unchanged']=True
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report_path=str(output),report_sha256=data.digest(output),source_sha256=report['source_sha256'],checks=len(report['checks']))),flush=True)


def _synthetic_preflight():
    folder=Path(tempfile.mkdtemp(prefix='systemness-complementary-natural-data-preflight-',dir='/tmp'))
    source_sha=data.digest(__file__);formal=(OUT,CACHE,SNAPSHOT);tokenizer=FakeTokenizer();checks=[]
    def check(name,value=True):require(value,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,FileNotFoundError,KeyError,ValueError,AssertionError,UnicodeDecodeError):checks.append(name)
        else:raise RuntimeError('Fault accepted: '+name)
    fake_old=[dict(text='Fabricated archived instruction '+str(i)+' for synthetic exclusion only.') for i in range(124)]
    check('Reserved32 wording strings unchanged',canonical_sha(WORDINGS)==RESERVED_WORDINGS_SHA256)
    templates,audit=build_templates(tokenizer,fake_old)
    audit.update(old80_plus44=False,synthetic_old_catalog=True,real_old_wordings_checked=False,real_tokenizer_used=False)
    immutable_json(folder/'system-templates-candidate.json',dict(templates=templates,audit=audit))
    check('32 fixed wordings synthetic-excluded from124 fake old templates',len(templates)==32 and audit['exact_normalized_sanitized_prefix_url_overlap']==0)
    path=folder/'users.csv'
    path.write_bytes(b'conv_id,user_input,model_output\r\na,"line1\nline2 ""quoted""",\xff\xfe\r\nb,other,"unread\nreply"\r\n')
    users,csv_audit=user_csv(path)
    check('Quoted user-only reader never decodes invalid assistant UTF8',users['a']['user_input']=='line1\nline2 "quoted"'
        and csv_audit['assistant_fields_decoded']==csv_audit['assistant_fields_buffered']==0)
    bad=folder/'bad.csv';bad.write_bytes(b'conv_id,user_input,model_output\na,"unfinished,b\n')
    reject('Malformed quoted CSV rejected',lambda:user_csv(bad))
    with patched(OUT=folder/'unfrozen'):
        reject('Catalog requires predeclared model/source protocol',lambda:guarded_catalog({}))
        reject('Assistants require prior128-ID freeze',lambda:source_selection({}))
    # Full source-count synthetic catalog: recover all200 legacy conversations
    # and the108 TC native matches without ever supplying assistant fields.
    def question(tag,index):return f'Fabricated {tag} question {index}. '+('Please explain this locally invented example carefully. '*3)
    user_rows=[];toxic={};legacy_rows=[];historical_keys=[]
    def oa(identity,text,parent=None):
        user_rows.append(dict(message_tree_id=identity,message_id=identity if parent is None else identity+'-followup',
            parent_id=parent,role='prompter',lang='en',deleted=False,synthetic=False,model_name=None,
            tree_state='ready_for_export',text=text))
    for index in range(224):
        identity='old-oa-'+str(index);oa(identity,question('old OA',index))
        historical_keys.append(dict(dataset='oasst',native_id=identity))
        identity='old-tc-'+str(index);toxic[identity]=dict(conv_id=identity,user_input=question('old TC',index))
        historical_keys.append(dict(dataset='toxicchat0124',native_id=identity))
    for index in range(100):
        text=question('legacy OA',index);oa('legacy-oa-'+str(index),text)
        legacy_rows.append(dict(dataset='oasst',conv_id=index,user_query=text))
        text=question('legacy TC',index);identity='legacy-tc-'+str(index)
        toxic[identity]=dict(conv_id=identity,user_input=text)
        legacy_rows.append(dict(dataset='toxicchat',conv_id=100+index,user_query=text))
    for index in range(25):
        text=question('legacy followup OA',index);oa('legacy-oa-'+str(index),text,'legacy-oa-'+str(index))
        legacy_rows.append(dict(dataset='oasst',conv_id=index,user_query=text))
    for index in range(8):
        identity='legacy-tc-duplicate-'+str(index)
        toxic[identity]=dict(conv_id=identity,user_input=question('legacy TC',index))
    for index in range(80):
        oa('new-oa-'+str(index),question('new OA',index))
        identity='new-tc-'+str(index);toxic[identity]=dict(conv_id=identity,user_input=question('new TC',index))
    oa('new-history-duplicate',question('old OA',0))
    user=pd.DataFrame(user_rows);legacy=pd.DataFrame(legacy_rows)
    chosen,selection_audit=select_sources(user,toxic,legacy,historical_keys,tokenizer)
    reverse,reverse_audit=select_sources(user.iloc[::-1],dict(reversed(list(toxic.items()))),legacy.iloc[::-1],historical_keys[::-1],tokenizer)
    check('Synthetic all448+legacy200 exclusion and128 deterministic selection',chosen==reverse and selection_audit==reverse_audit
        and len(chosen)==128 and selection_audit['legacy_native_exclusions']=={'oasst':100,'toxicchat0124':108}
        and all(k['native_id'].startswith('new-') and k['native_id']!='new-history-duplicate' for k in chosen))
    reject('Insufficient fresh source pool fails no expansion',lambda:select_sources(user,toxic,legacy,historical_keys,tokenizer,100))
    assignments=assignments_for(chosen,templates);assignment_map={(a['dataset'],a['native_id']):a for a in assignments}
    check('All32 families have2OA+2TC roots and6+6fixedgallery',len(assignments)==128 and len(gallery_for(chosen)['source_uids'])==12
        and Counter(a['family_id'] for a in assignments)=={t['family_id']:4 for t in templates})
    records=[]
    for index in range(128):
        key=chosen[index];dataset=key['dataset'];assignment=assignment_map[(dataset,key['native_id'])]
        template=next(t for t in templates if t['template_id']==assignment['template_id'])
        text='A fabricated CPU reply '+str(index)+'. '
        if index in [4,5]:text='An intentional repeated fabricated CPU response.'
        messages=[dict(role='system',text=template['text'],source_message_id=template['template_id'],origin='fixture',source_model=None),
            dict(role='user',text='A synthetic question '+str(index)+'? ',source_message_id=str(index)+'/u',origin='fixture',source_model=None),
            dict(role='assistant',text=text,source_message_id=str(index)+'/a',origin='fixture',source_model='synthetic')]
        if index==8:messages[2]['text']='<|start|> reserved marker is sanitized. '+('long reply '*300)
        records.append(dict(source_uid=uid(key),dataset=dataset,native_id=key['native_id'],
            split='test',revision=key['revision'],root_raw_sha256=key['root_raw_sha256'],template_id=template['template_id'],
            family_id=template['family_id'],document_id=DOCUMENT_START+index,messages=messages,failure='fixture-quality-failure' if index==127 else None))
    def fingerprint(text):
        body,_,clean=pure.capped_body(text,tokenizer);return pure.assistant_keys(text,clean,body,tokenizer)
    oldindex={key:{'old-fixture'} for key in fingerprint(records[2]['messages'][2]['text'])}
    reference=[dict(message_uid='historical:fixture/message2',source_uid='historical:fixture',split='fit',message_ix=2,
        keys=fingerprint(records[3]['messages'][2]['text']))]
    dataset=build_dataset(records,tokenizer,reference,oldindex,dict(fixture_only=True));inputs,rows,tokens,ledger,duplicates,coverage=dataset
    verify_cached_content(inputs,rows,tokens,ledger,tokenizer)
    verify_summary(inputs,rows,ledger,duplicates,coverage)
    check('All128 ledger retained with one failure',len(ledger)==128 and len(inputs)==127 and ledger[-1]['status']=='quality-failed')
    check('All-content SUA native labels and 192 cap',coverage['all_content_tokens_scored'] and coverage['independent_native_labels_match']
        and set(rows.role)==set(ROLES) and coverage['roles_with_no_observations']['cot']['status']=='N/A')
    check('Old and historical448 collisions mark clean only',not ledger[2]['eligible_clean_test'] and not ledger[3]['eligible_clean_test']
        and len(inputs)==127 and duplicates['eligibility_counts']['eligible_clean_test']==125)
    check('Family/repeat components merged before clean',duplicates['dependency_components']==31
        and duplicates['system_family_counts']==dict(fixed=32,all_prepared=32,clean_prepared=32))
    check('Sanitized and truncated response provenance',ledger[8]['messages'][2]['marker_sanitized'] and ledger[8]['messages'][2]['truncated'])
    wrong=rows.copy();wrong.loc[0,'body_offset']=999
    reject('Body offset tamper rejected',lambda:verify_cached_content(inputs,wrong,tokens,ledger,tokenizer))
    wrong=rows.copy();wrong.loc[0,'role']='cot'
    reject('Invented CoT role rejected',lambda:verify_cached_content(inputs,wrong,tokens,ledger,tokenizer))
    failed=copy.deepcopy(records)
    for record in failed:record['failure']='all-failed-fixture'
    empty=build_dataset(failed,tokenizer,[],{},dict(fixture_only=True));verify_cached_content(*empty[:3],empty[3],tokenizer)
    verify_summary(empty[0],empty[1],empty[3],empty[4],empty[5])
    check('All128failed valid empty typed schema and zero components',len(empty[0])==len(empty[1])==0 and len(empty[3])==128
        and empty[4]['dependency_components']==0 and empty[5]['no_observations'])
    check('New row UID/sample namespaces',rows.sample_ix.min()==SAMPLE_START and inputs.uid.str.startswith('complementary-natural:').all())
    for name,frame in [('inputs.parquet',inputs),('rows.parquet',rows)]:
        pure.save_parquet(folder/name,frame);pd.testing.assert_frame_equal(frame,pd.read_parquet(folder/name),check_exact=True)
    immutable_json(folder/'token-ids.json',tokens);check('Parquet/token cache roundtrip',read(folder/'token-ids.json')==tokens)
    # Four explicit fake prepared caches. The assistant middle message is not
    # a document prefix; sparse rows/token JSON are deliberately unusable for
    # reconstructing it, proving the full native prompt reader is authoritative.
    middle='A fabricated historical middle passage about cedar notebook seventy three.'
    otherbody='Another fabricated historical native passage about cobalt compass ninety two.'
    fake_prepared={}
    for index,name in enumerate(PREPARED_ROLE_INPUTS):
        cache=folder/'prepared'/name;cache.mkdir(parents=True);metadata=cache/'metadata.json'
        messages=([('user','Before the independently stored middle excerpt.'),('assistant',middle),('user','After the independently stored middle excerpt.')] if index==0 else
            [('cot','A fabricated internal explanation for an archive-only example.'),('tool',otherbody)] if index==1 else
            [('assistant','Historical complete long paragraph. '*300)] if index==2 else
            [('system','A fabricated historical instruction.'),('user','A fabricated unrelated historical query.')])
        prompt,_=data.serialize(messages,bos=True,wrapper='native');ids=tokenizer.encode(prompt,add_special_tokens=False)
        pd.DataFrame([dict(prompt_ix=17+index,prompt=prompt,length=len(ids),wrapper='native',split='fit'),
            dict(prompt_ix=90+index,prompt='This non-native text must not be decoded as a native frame.',length=1,wrapper='all-user',split='fit')]).to_parquet(cache/'inputs.parquet',index=False)
        pd.DataFrame([dict(token_id=999999,token_ix=999999,role='user')]).to_parquet(cache/'rows.parquet',index=False)
        (cache/'token-ids.json').write_text('[]')
        immutable_json(metadata,dict(cache_sha256={fn:data.digest(cache/fn) for fn in FILES}));fake_prepared[name]=(metadata,cache)
    with patched(PREPARED_ROLE_INPUTS=fake_prepared):prepared_meta=prepared_role_metadata()
    signature=dict(source_sha256=source_sha,dependencies_sha256={},frozen_sha256='a'*64,snapshot_file_sha256='b'*64,prepared_role_inputs=prepared_meta)
    fixture_plan=dict(protocol=protocol_spec(),source_signature=signature,source_sha256=source_sha,dependencies_sha256={},
        frozen=dict(snapshot_sha256='a'*64,fixture_only=True),templates=templates,template_audit=audit)
    oldfolder=folder/'old448';oldfolder.mkdir();immutable_json(oldfolder/'natural-source-keys.json',dict(source_keys=historical_keys))
    with patched(OUT=folder/'roundtrip',CACHE=folder/'roundtrip-cache',OLD_FOLDER=oldfolder,plan_value=lambda:fixture_plan,load_tokenizer=lambda:tokenizer,PREPARED_ROLE_INPUTS=fake_prepared):
        OUT.mkdir();immutable_json(OUT/'plan.json',fixture_plan)
        freeze_sources(fixture_plan,chosen,selection_audit)
        check('128 IDs/templates/gallery frozen before any assistant stage',source_selection(fixture_plan)[0]==chosen)
        role_reference,role_proof=prepared_role_references(fixture_plan,tokenizer)
        check('Full prepared native middle bodies recovered without sparse row assembly',any(r['keys']&fingerprint(middle) for r in role_reference) and role_proof['counts']['main']['body_segments']==3 and len(role_reference)>4)
        cross_records=copy.deepcopy(records)
        cross_records[0]['messages'][2]['text']=middle;cross_records[1]['messages'][2]['text']=middle
        cross_records[6]['messages'][1]['text']=middle
        cross_records[7]['messages'][1]['text']=otherbody;cross_records[9]['messages'][2]['text']=otherbody
        cross=build_dataset(cross_records,tokenizer,reference,oldindex,dict(fixture_only=True),role_reference,role_proof)
        verify_cached_content(*cross[:3],cross[3],tokenizer);verify_summary(cross[0],cross[1],cross[3],cross[4],cross[5])
        c=cross[3]
        check('Middle-chunk assistant collisions are cleanFalse not old_generated',all(c[i]['historical_role_input_collision'] and not c[i]['eligible_clean_test'] and not c[i]['old_generated_collision'] and not c[i]['public448_assistant_collision'] for i in [0,1]))
        check('SUA user history collision cleanFalse without ID replacement',all(c[i]['historical_role_input_collision_roles']==['user'] and not c[i]['eligible_clean_test'] and not c[i]['old_generated_collision'] for i in [6,7]) and len(c)==128 and len(cross[0])==127 and [r['source_uid'] for r in c]==[r['source_uid'] for r in ledger])
        check('Assistant-and-user share external historical nodes and components',c[0]['dependency_component_id']==c[1]['dependency_component_id']==c[6]['dependency_component_id'] and c[7]['dependency_component_id']==c[9]['dependency_component_id'] and len(cross[4]['external_reference_bridges'])>=2)
        check('Role-input inventory is exactbody keyed with semantic independence unclaimed',cross[4]['semantic_independence_claimed'] is False and role_proof['semantic_independence_claimed'] is False and cross[4]['historical_role_input_reference_nodes']==len(role_reference))
        reject('Historical native length or malformed frame changes rejected',lambda:native_role_bodies(prompt,len(ids)+1,tokenizer))
        def reference_before_freeze():
            with patched(OUT=folder/'not-source-frozen'):prepared_role_references(fixture_plan,tokenizer)
        reject('Historical prepared bodies cannot materialize before new128freeze',reference_before_freeze)
        cache=fake_prepared['main'][1];saved_inputs=(cache/'inputs.parquet').read_bytes();(cache/'inputs.parquet').write_bytes(b'changed prepared-cache bytes')
        reject('Prepared cache byte identity changes rejected',prepared_role_metadata);(cache/'inputs.parquet').write_bytes(saved_inputs)

        complete=save_result(fixture_plan,chosen,*dataset,dict(fixture_only=True))
        check('Public owned cache complete roundtrip and source freeze',load_verified_data()[0]==complete)
        before={name:data.digest(CACHE/name) for name in FILES}
        check('Completed prepare readonly skips catalog and replies',prepare()==complete
            and before=={name:data.digest(CACHE/name) for name in FILES})
        owner=read(CACHE/'initialized.json');data.atomic_json(CACHE/'initialized.json',dict(owner,source_sha256='0'*64))
        reject('Wrong natural cache owner rejected',load_verified_data);data.atomic_json(CACHE/'initialized.json',owner)
        saved=(CACHE/'token-ids.json').read_bytes();(CACHE/'token-ids.json').write_text('[]')
        reject('Changed token cache rejected',load_verified_data);(CACHE/'token-ids.json').write_bytes(saved)
        saved_complete=read(OUT/'data-complete.json');data.atomic_json(OUT/'data-complete.json',dict(saved_complete,gpu_work_run=True))
        reject('Source-only false flags enforced',load_verified_data);data.atomic_json(OUT/'data-complete.json',saved_complete)
        freeze=read(OUT/'source-freeze.json');data.atomic_json(OUT/'source-freeze.json',dict(freeze,assistant_materialized=True))
        reject('Assistant-before-ID-freeze marker cannot change',load_verified_data);data.atomic_json(OUT/'source-freeze.json',freeze)
        changed=copy.deepcopy(duplicates);changed['system_family_counts']['clean_prepared']=999
        reject('All/clean active family recount enforced',lambda:verify_summary(inputs,rows,ledger,changed,coverage))
    check('Formal namespace constants unchanged; no formal paths inspected',formal==(OUT,CACHE,SNAPSHOT) and data.digest(__file__)==source_sha)
    report=dict(source_sha256=source_sha,checks=checks,checks_passed=len(checks),synthetic_fixed_roots=128,
        synthetic_prepared_prompts=len(inputs),synthetic_content_rows=len(rows),template_audit=audit,
        templates_candidate_path=str(folder/'system-templates-candidate.json'),
        tokenizer_scope='Synthetic roundtrip tokenizer only; does not establish real192token lengths or real124-wording exclusion.',
        real_tokenizer_or_checkpoint_read=False,real_old_template_catalog_read=False,
        templates_candidate_file_sha256=data.digest(folder/'system-templates-candidate.json'),
        template_token_lengths=[t['token_count'] for t in templates],real_unused_user_bodies_decoded=0,
        real_unused_assistant_bodies_decoded=0,real_new_IDs_selected=0,formal_protocol_declared=False,
        formal_cache_created=False,gpu_work_run=False,model_generation_run=False,classifier_probabilities_read=False,
        test_metrics_read=False,source_files_downloaded=False)
    path=folder/'report.json';path.write_text(json.dumps(report,indent=2)+'\n')
    return path,report


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true',help='Synthetic CPU/tmp only; no unused user/reply or ID selection.')
    action.add_argument('--stage',choices=['declare','prepare','verify'],help='Explicit root-only metadata freeze/source preparation/replay.')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.stage=='declare':declare();print(json.dumps(dict(plan_sha256=data.digest(OUT/'plan.json'))),flush=True)
    elif args.stage=='prepare':
        value=prepare();print(json.dumps(dict(data_complete_sha256=data.digest(OUT/'data-complete.json'),rows=value['rows'])),flush=True)
    elif args.stage=='verify':print(json.dumps(verify()),flush=True)
    else:parser.print_help()


if __name__=='__main__':main()
