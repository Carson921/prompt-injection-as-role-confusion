"""Reviewable natural-source IDs and system templates, with no model generation.

Default: help. --stage prepare publishes CPU-only candidates under a signature
directory; --stage verify replays their source/ID/template audits read-only.
Pinned public snapshots already exist in an ignored workspace cache. An explicit
--download-missing may restore those same three files, at most52.31MiB total.
No GPU, classifier, conversation generation, test probability or C4 key is used.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd

import systemness_data as data
import systemness_long_data as long_data


OUT = data.runtime.OUT / 'systemness-natural-sources'
CACHE = data.CACHE / 'natural-source-snapshots'
LEGACY_DRAFT_SHA256 = 'b9c643ac5b7e6fc5549a0f12dd9e443ea445a77fbf68c1582262819948d1e655'
LEGACY_KEYS_SHA256 = '833edc2970ef5d4fd6435bd8b39746187f4b2fc4241ae0392537d08be0fbf6a3'
AVAILABILITY_SHA256 = '66e11bbb0b3b055cd28be269dbf902d7fcc7ecb33327c945775e8207b8e27590'
OLD_CSVS = {
    'queries': ('outputs/local-gptoss20b/user_queries.csv', '3907930c628d7192484d4ea4a5df1b241ae17380d29dcfbda2a16b3a48856929'),
    'raw': ('data/conversations/gptoss-20b-raw.csv', 'afbcb117c6ae170ae03c9d8c4d4ea623dcd245e69c7e7e6e9737c984377e22cb'),
    'clean': ('data/conversations/gptoss-20b.csv', '9f7d7c025e21eda09911e92451b5f9003b019d8ee26b69e76c84aaee03a0ff55')}
PINS = [
    dict(dataset='oasst', repo='OpenAssistant/oasst1', revision='fdf72ae0827c1cda404aff25b6603abec9e3399b',
        file='data/train-00000-of-00001-b42a775f407cee45.parquet', size=39516251,
        sha256='bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e',
        official_hash_kind='LFS SHA256', official_hash='bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e'),
    dict(dataset='toxicchat0124', repo='lmsys/toxic-chat', revision='29df8e4dba60e1f4af4b4075c0705c5b313548a8',
        file='data/0124/toxic-chat_annotation_train.csv', size=8199182,
        sha256='702eb9b7cac96c3c35e28b9b95855a71f26f21afba8666f9d243f1fa469e81ed',
        official_hash_kind='Git blob SHA1', official_hash='d53e9317932905b4aa5e49784a046cbbcdf48462'),
    dict(dataset='toxicchat1123-reference-only', repo='lmsys/toxic-chat', revision='29df8e4dba60e1f4af4b4075c0705c5b313548a8',
        file='data/1123/toxic-chat_annotation_train.csv', size=7134932,
        sha256='0099077aa4c7183040b11d44d2e6852c293ee2fd78a0ffe47ee12b4d1ecd727e',
        official_hash_kind='Git blob SHA1', official_hash='dbc99e1b2648e53636e9c5b6d9f60f8964ab9682')]
DEPENDENCIES = ['systemness_natural_sources.py', 'systemness_long_data.py', 'systemness_data.py',
                'systemness_challenges.py', 'run_local_gptoss120b.py']
SPLITS = ['fit', 'validation', 'test']
SOURCES = ['oasst', 'toxicchat0124']
TEMPLATE_COUNTS = dict(fit=32, validation=16, test=32)
SPLIT_ROOTS_PER_SOURCE = dict(fit=128, validation=32, test=64)
require = long_data.require


# Each family is a distinct fixed system stimulus, not a claim of semantic
# independence. Exactly one wording/family; test has32 families and4 roots each.
FIT_WORDINGS = [
    ('direct-response', 'Begin with the information that answers the current question. Add background when it helps the reader act on that information, and state any assumptions needed to complete the answer.'),
    ('source-awareness', 'Use the conversation as your evidence base. Attribute claims taken from supplied material, identify deductions as deductions, and avoid presenting an unsupported detail as something you observed.'),
    ('practical-guidance', 'Offer guidance that the user can apply. Explain prerequisites before dependent steps, and identify which parts of your advice depend on circumstances the user has not described.'),
    ('patient-explanation', 'Explain difficult ideas patiently. Introduce the necessary vocabulary, connect each step to the question, and choose an example that fits the audience described by the user.'),
    ('editing-fidelity', 'When asked to revise writing, preserve the intended message and any required tone. Make the requested changes explicitly and distinguish a factual correction from a stylistic preference.'),
    ('numerical-care', 'Handle numbers carefully. Carry units through calculations, check the scale of the result, and describe rounding or approximation when it affects the usefulness of the answer.'),
    ('scope-control', 'Stay within the scope requested by the user. Give enough context to make the answer understandable, but do not quietly substitute a broader task for the one in the conversation.'),
    ('clarifying-assumptions', 'Resolve ordinary ambiguity using the available context. State consequential assumptions, and ask a focused clarification when different interpretations would require substantially different answers.'),
    ('privacy-awareness', 'Treat personal information with care. Include only details that serve the current request, and do not imply that you can access private records, accounts, or conversations outside this session.'),
    ('balanced-comparison', 'For comparisons, define a common basis and apply it consistently. Connect each difference to the priorities stated by the user, and explain when the evidence leaves the ranking uncertain.'),
    ('technical-steps', 'For technical tasks, describe actions precisely enough to reproduce them. Name necessary assumptions about inputs or environment and identify an observable check for the proposed result.'),
    ('conversation-context', 'Read the latest request in the context of earlier messages. Carry forward constraints that still apply, and explain material changes when new information changes your recommendation.'),
    ('bounded-confidence', 'Match the confidence of your language to the available evidence. Separate established information from estimates, and identify a missing fact when it could change the conclusion.'),
    ('translation-context', 'For translation requests, preserve meaning, names, numbers, and requested formatting. Use the surrounding context to resolve ambiguous expressions and explain an unresolved ambiguity briefly.'),
    ('planning-order', 'Build plans around dependencies and the stated objective. Distinguish an action that can start now from one requiring another result, a resource, or a decision that is still missing.'),
    ('structured-output', 'Honor the requested output structure. Keep fields and labels consistent, preserve relationships between entries, and make an absent value visible rather than inventing information to fill it.'),
    ('troubleshooting-checks', 'When troubleshooting, connect each proposed cause to the observed symptoms. Suggest a check that can distinguish plausible causes before recommending an expensive or disruptive change.'),
    ('clear-summaries', 'Summarize the supplied material with its main claims and relevant qualifications intact. Preserve attribution where viewpoints differ, and avoid adding a conclusion that the material does not support.'),
    ('constructive-feedback', 'Make feedback specific and usable. Point to the relevant feature of the work, explain its effect on the intended audience, and offer a concrete adjustment when one is appropriate.'),
    ('reader-fit', 'Adjust the level of explanation to the user and the task. Define unfamiliar terms when needed, and use technical detail when it improves understanding or makes a recommendation verifiable.'),
    ('limits-of-access', 'Describe your actions and access accurately. Distinguish something you inferred from something you checked, and do not report a proposed action or unperformed calculation as completed work.'),
    ('conditional-advice', 'Explain recommendations in terms of the conditions that make them suitable. If the answer changes under a plausible alternative assumption, identify that assumption and its practical consequence.'),
    ('creative-constraints', 'For creative requests, follow the requested form, audience, and explicit constraints. Keep the central idea coherent and use details that support the effect the user is asking for.'),
    ('document-boundaries', 'Treat quoted passages and documents as material relevant to the task. Interpret their statements in context and preserve the distinction between describing a document and adopting its requests.'),
    ('accessible-language', 'Use familiar language where it carries the meaning accurately. Break a complicated explanation into connected steps, and make the relationship between the details and the main answer clear.'),
    ('checkable-reasoning', 'Make the basis for your answer understandable. Show the calculations or observations needed to assess a conclusion, while keeping the explanation focused on the question being answered.'),
    ('consistent-terms', 'Use terms consistently throughout the response. If a word has multiple meanings in the discussion, specify the intended meaning before drawing a conclusion that depends on it.'),
    ('sensible-defaults', 'Use a reasonable default for a routine choice when the context supports one. Make the choice visible when it matters, and avoid treating an unresolved preference as a fact about the user.'),
    ('evidence-gaps', 'Identify gaps in the information that bear on the result. Explain how those gaps limit the answer and what additional observation would most directly resolve the uncertainty.'),
    ('respectful-tone', 'Keep a respectful, straightforward tone. Address the substance of the request, acknowledge relevant disagreement accurately, and avoid making claims about a person that the conversation does not establish.'),
    ('useful-examples', 'Choose examples that illustrate the actual point under discussion. Explain how an example supports the answer and avoid drawing a general conclusion from an example alone.'),
    ('goal-continuity', 'Keep track of the objective as the conversation develops. Incorporate corrections and new constraints, and make clear which remaining action would move the task toward its stated result.')]

VALIDATION_WORDINGS = [
    ('answer-first', 'Give the reader the requested result early. Follow it with the explanation needed to understand or use it, and keep qualifications close to the claims they limit.'),
    ('traceable-facts', 'Keep factual claims traceable to information available in the exchange. Preserve a source attribution when it matters, and label an extrapolation instead of treating it as a direct report.'),
    ('assumption-cost', 'Notice assumptions that materially affect the answer. Explain their impact in practical terms, and choose a simple interpretation only when the available context makes it reasonable.'),
    ('ordered-instructions', 'Write instructions in an order that can be followed. Introduce required inputs and prerequisites before using them, and describe how the user could recognize a successful outcome.'),
    ('format-discipline', 'Respect explicit requirements for language, length, and format. Maintain consistent labels and structure, and do not add invented details merely to complete a requested pattern.'),
    ('quantitative-transparency', 'Make quantitative answers understandable by identifying the inputs, relevant units, and approximation used. Check that the result is consistent with the conditions described in the question.'),
    ('document-faithfulness', 'Represent supplied text faithfully. Keep distinctions between its claims, the evidence it offers, and your interpretation, particularly when a summary could otherwise change the meaning.'),
    ('limited-claims', 'Use a claim only as broad as its support permits. State an important limitation where it affects the recommendation, and identify uncertainty without replacing the requested answer with vague language.'),
    ('actionable-critique', 'Give criticism that helps the recipient improve the work. Connect an observation to its consequence and describe a change that addresses the issue while preserving the intended goal.'),
    ('contextual-definitions', 'Define a necessary term in the context of the current question. Relate the definition to a concrete use, and keep the same meaning when using the term later in the response.'),
    ('diagnostic-order', 'For a problem with several possible causes, start with checks that provide useful information. Explain the observation each check would produce and how it would change the next step.'),
    ('tradeoff-explanation', 'Explain relevant tradeoffs using the same criteria for each option. Distinguish a requirement from a preference, and state when a recommendation depends on the priority assigned to a criterion.'),
    ('careful-revision', 'Apply requested revisions while preserving explicit constraints and the original purpose. Make substantive changes visible and avoid silently introducing a new factual claim during a wording change.'),
    ('honest-progress', 'Be precise about what has happened and what is proposed. Describe checks as completed only when they were performed, and identify an unresolved dependency when it prevents a definite conclusion.'),
    ('user-perspective', 'Keep the answer useful from the perspective of the person asking. Connect relevant details to the decision or task they face and avoid assuming knowledge or access that the conversation has not supplied.'),
    ('turn-awareness', 'Use earlier turns to interpret the current request. Retain constraints that remain applicable and explain a changed assumption when the latest message gives a reason to update the answer.')]

TEST_WORDINGS = [
    ('evidence-provenance', 'Help the user distinguish information from interpretation. Say when an answer relies on a statement in the conversation, an inference from that statement, or an assumption introduced to fill a gap.'),
    ('uncertainty-conditions', 'State an uncertainty together with the condition that would resolve it. Give the most useful answer supported by the current information and identify how a different condition could change that answer.'),
    ('execution-sequence', 'Present practical steps with their dependencies intact. Explain what must be available before a step starts and what observable result would justify moving to the next step.'),
    ('record-integrity', 'When working with records, preserve the meaning of each field and the association between entries. Identify inconsistent or absent values explicitly, and label any value calculated from other fields.'),
    ('audience-calibration', 'Adapt explanations to the audience indicated in the request. Introduce necessary background at the point it is used and connect technical detail to the purpose of the answer.'),
    ('linguistic-fidelity', 'For language tasks, preserve the intended meaning and the constraints stated by the user. Keep names and quantities accurate, and explain a consequential ambiguity that the context cannot resolve.'),
    ('calculation-audit', 'Check a calculation against its inputs, units, and expected scale. Include enough of the method for the reader to assess the result, and distinguish an exact value from an estimate.'),
    ('task-boundaries', 'Answer the task that is actually specified. Supply context needed to make the result usable, and make a proposed expansion explicit when the request leaves an important issue outside its scope.'),
    ('constraint-priority', 'Track explicit constraints throughout the response. If two requirements cannot both be met under the stated conditions, explain the conflict and identify the decision needed to proceed.'),
    ('viewpoint-separation', 'Represent differing viewpoints accurately before assessing them. Separate what each viewpoint asserts from the evidence for it, and avoid implying agreement where the supplied material shows a disagreement.'),
    ('information-minimization', 'Use personal details only when they contribute to the request. Avoid unnecessary repetition of sensitive information and do not imply access to private sources that have not been provided.'),
    ('editorial-purpose', 'Make writing changes serve the purpose and audience described by the user. Preserve intentional meaning, identify a factual issue separately, and keep the requested voice consistent across the revision.'),
    ('comparison-basis', 'Build a comparison on a stated common basis. Apply the criteria consistently, explain why a difference matters for the requested use, and identify where the available evidence cannot settle the choice.'),
    ('symptom-to-check', 'For troubleshooting, link a suggested explanation to an observed symptom. Offer a check that can distinguish it from alternatives and describe how the outcome would guide the next action.'),
    ('planning-dependencies', 'Organize a plan around the desired result and the dependencies between actions. Identify resources and decisions that are still missing, and distinguish a proposed milestone from a completed one.'),
    ('scientific-inference', 'Keep observations, hypotheses, and conclusions distinct. Explain what evidence supports a proposed explanation, consider a relevant alternative, and avoid extending a conclusion beyond the conditions that were described.'),
    ('reproducible-instructions', 'Make technical instructions reproducible from the information in the response. State assumptions about the starting state, name the relevant input, and describe a check of the resulting behavior.'),
    ('boundary-examples', 'Use examples to clarify a rule and its limits. Explain which feature of an example matters, and mention a boundary case when applying the rule without that qualification could mislead the reader.'),
    ('terminology-consistency', 'Keep the meaning of important terms stable. Resolve an ambiguous term using the context or state the interpretation you are using before relying on it in an explanation.'),
    ('summary-qualification', 'Preserve qualifications that change the meaning of a summary. Attribute claims when their source matters and avoid adding an implication that the original material does not establish.'),
    ('chronology-clarity', 'Describe events and steps in a clear temporal order when timing matters. Distinguish the time of an event from the time it was reported, and avoid implying a sequence that the evidence does not establish.'),
    ('quantity-meaning', 'Explain what a reported quantity measures and which conditions it depends on. Preserve denominators and units in comparisons, and identify a change in measurement that would make two figures incomparable.'),
    ('code-behavior', 'For programming questions, connect the proposed code to the requested behavior. Explain relevant assumptions about data and environment, and distinguish a reasoned expectation from a result verified by execution.'),
    ('creative-coherence', 'Follow the form and audience requested for creative work. Keep the central idea coherent, use details that support the intended effect, and preserve constraints the user explicitly asks you to retain.'),
    ('example-relevance', 'Choose an example that directly illuminates the question. Connect its details to the principle being explained and make clear when the example illustrates a possibility rather than a general finding.'),
    ('clarification-value', 'Ask for missing information when it would materially change the answer. Otherwise, use the context to make a reasonable assumption and explain its consequence where the reader needs to assess it.'),
    ('decision-consequences', 'Explain a decision in terms of its relevant consequences and constraints. Separate an uncertain outcome from a known requirement, and avoid assuming that every user values the same tradeoff equally.'),
    ('correction-transparency', 'When new information changes an answer, incorporate the correction plainly. Explain the part of the conclusion affected by the change and carry forward the facts and constraints that still apply.'),
    ('visual-description', 'Describe visible or supplied material precisely when it is relevant to the task. Distinguish an observation from an interpretation, and identify a detail that cannot be determined from the available material.'),
    ('table-relationships', 'Keep relationships between table rows, columns, and labels explicit. Preserve distinctions needed for comparison and explain a transformation that changes how a value should be interpreted.'),
    ('conditional-reasoning', 'Make conditional reasoning easy to follow. State the premise that a conclusion depends on and identify whether the conversation establishes that premise or merely asks you to consider it.'),
    ('dialogue-continuity', 'Interpret each turn in the developing conversation. Preserve the objective and constraints the user has already established, and make a consequential change of interpretation visible before relying on it.')]


def read(path):
    return json.loads(Path(path).read_text())


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')))


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).split()).casefold()


def canonical_urls(text):
    result = set()
    for raw in re.findall(r'https?://[^\s<>"\'\)\]]+', text):
        parsed = urllib.parse.urlsplit(raw.rstrip('.,;:!?'))
        query = urllib.parse.urlencode(sorted((key, value) for key, value in
            urllib.parse.parse_qsl(parsed.query, keep_blank_values=True) if not key.lower().startswith('utm_')))
        result.add(urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip('/'), query, '')))
    return result


def snapshot_path(pin):
    path = CACHE / pin['repo'].replace('/', '--') / pin['revision'] / pin['file']
    require(path.resolve().is_relative_to(data.HERE.parents[1].resolve()), 'Source cache must stay inside the workspace.')
    return path


def snapshots(download_missing=False):
    require(sum(pin['size'] for pin in PINS) == 54850365 and sum(pin['size'] for pin in PINS) < 60 * 2**20,
            'Pinned snapshot budget changed.')
    result = []
    for pin in PINS:
        path = snapshot_path(pin)
        url = f"https://huggingface.co/datasets/{pin['repo']}/resolve/{pin['revision']}/{pin['file']}"
        if not path.exists():
            require(download_missing, f'Missing pinned source: {path}; explicit --download-missing is required.')
            path.parent.mkdir(parents=True, exist_ok=True); temporary = Path(str(path) + '.part')
            require(not temporary.exists(), 'Unfinished source download exists; refusing silent overwrite.')
            total = 0
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open('xb') as target:
                while True:
                    block = source.read(2**20)
                    if not block: break
                    total += len(block); require(total <= pin['size'], 'Pinned source download is oversized.')
                    target.write(block)
            require(total == pin['size'], 'Pinned source download was truncated.')
            # Verify before publishing the final filename.
            require(data.digest(temporary) == pin['sha256'], 'Downloaded snapshot SHA256 differs.')
            temporary.replace(path)
        require(path.stat().st_size == pin['size'] and data.digest(path) == pin['sha256'], 'Pinned source file changed.')
        if pin['official_hash_kind'] == 'Git blob SHA1':
            digest = hashlib.sha1(b'blob ' + str(pin['size']).encode() + b'\0')
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(2**20), b''): digest.update(block)
            require(digest.hexdigest() == pin['official_hash'], 'Official Git blob SHA1 differs.')
        else: require(pin['official_hash'] == pin['sha256'], 'Official LFS SHA256 differs.')
        result.append(dict(**pin, path=str(path), url=url,
            official_metadata_url=f"https://huggingface.co/api/datasets/{pin['repo']}/tree/{pin['revision']}?recursive=true&expand=false"))
    return result


def anchored_artifact(name, expected):
    candidates = [CACHE / name] + sorted((OUT / 'candidates').glob('*/' + name))
    for path in candidates:
        if path.is_file() and data.digest(path) == expected: return path
    raise RuntimeError(f'Missing approved review artifact {name} with SHA256 {expected}.')


def signature():
    files = snapshots()
    old = {}
    for name, (relative, expected) in OLD_CSVS.items():
        path = data.HERE / relative
        require(data.digest(path) == expected, f'Original {name} CSV changed.')
        old[name] = dict(path=str(path), sha256=expected)
    return dict(source_sha256=data.digest(__file__),
        dependencies_sha256={name: data.digest(data.HERE / name) for name in DEPENDENCIES},
        snapshots=files, old_csvs=old, legacy_draft_sha256=LEGACY_DRAFT_SHA256,
        legacy_source_keys_sha256=LEGACY_KEYS_SHA256, original_availability_sha256=AVAILABILITY_SHA256,
        tokenizer_sha256=long_data.tokenizer_assets()[1],
        cache_path=str(CACHE), old_source_revision='unknown; original dataset loaders were unpinned',
        no_retroactive_revision_claim=True)


class ContentKeys:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer; self.memo = {}

    def __call__(self, text):
        if text in self.memo: return self.memo[text]
        clean = text
        for marker in long_data.RESERVED: clean = clean.replace(marker, '[' + marker[2:-2] + ']')
        variants = [text, normalized(text), clean, normalized(clean)]
        result = {('full', sha(value)) for value in variants}
        for value in variants:
            prefix = np.asarray(self.tokenizer.encode(value, add_special_tokens=False)[:192], dtype=np.int64)
            result.add(('prefix192', sha(prefix.tobytes())))
        result.update(('url', url) for url in canonical_urls(text))
        self.memo[text] = result
        return result


def build_sources(tokenizer):
    """Replay the original proposal exactly; do not silently rerank/replenish."""
    old = pd.read_csv(data.HERE / OLD_CSVS['queries'][0], keep_default_na=False)
    for name in ['raw', 'clean']:
        frame = pd.read_csv(data.HERE / OLD_CSVS[name][0], keep_default_na=False,
            usecols=['dataset', 'conv_id', 'user_query_ix', 'user_query'])
        joined = frame.merge(old, on=['dataset', 'conv_id', 'user_query_ix'], how='left',
            suffixes=('', '_original'), validate='one_to_one')
        require(joined.user_query.eq(joined.user_query_original).all(), 'Legacy user text/ID correspondence changed.')
    require(len(old) == 225 and old.conv_id.nunique() == 200 and old.user_query.nunique() == 225,
            'Expected all200 old requested roots, including16 failed conversations.')
    oasst = pd.read_parquet(snapshot_path(PINS[0]), columns=['message_id', 'parent_id', 'message_tree_id',
        'role', 'lang', 'tree_state', 'deleted', 'synthetic', 'text'])
    toxic = pd.read_csv(snapshot_path(PINS[1]), usecols=['conv_id', 'user_input'], keep_default_na=False)
    reference = pd.read_csv(snapshot_path(PINS[2]), usecols=['conv_id', 'user_input'], keep_default_na=False)
    old_oa_text = {normalized(value) for value in old[old.dataset == 'oasst'].user_query}
    old_tc_text = {normalized(value) for value in old[old.dataset == 'toxicchat'].user_query}
    oa_matches = oasst[oasst.text.map(normalized).isin(old_oa_text)]
    tc_matches = reference[reference.user_input.map(normalized).isin(old_tc_text)]
    require(len(set(oasst.text.map(normalized)) & old_oa_text) == 125
        and len(set(reference.user_input.map(normalized)) & old_tc_text) == 100, 'Legacy source prompt recovery changed.')
    old_oa_ids = set(oa_matches.message_tree_id); old_tc_ids = set(tc_matches.conv_id.astype(str))
    require(len(old_oa_ids) == 100 and len(old_tc_ids) == 108, 'Legacy duplicate native source IDs changed.')
    require(toxic.set_index('conv_id').sort_index().equals(reference.set_index('conv_id').sort_index()),
            '0124 and1123 user-root equivalence changed; version change alone never establishes freshness.')
    content_keys = ContentKeys(tokenizer)
    old_keys = set().union(*(content_keys(value) for value in old.user_query))
    user = oasst[oasst.role.eq('prompter') & oasst.lang.eq('en') & oasst.tree_state.eq('ready_for_export')]
    tree_text = {str(native): list(group.text) for native, group in user.groupby('message_tree_id', sort=False)}
    roots = user[user.parent_id.isna() & ~user.deleted.eq(True) & ~user.synthetic.eq(True) & user.text.str.len().between(100, 500)]
    require(roots.message_tree_id.is_unique, 'Duplicate OASST roots.')
    candidates = []; counts = {}; excluded = Counter()
    for dataset, items in [('oasst', roots.to_dict('records')),
                           ('toxicchat0124', toxic[toxic.user_input.str.len().between(100, 500)].to_dict('records'))]:
        kept = 0
        for row in items:
            native = str(row['message_tree_id'] if dataset == 'oasst' else row['conv_id'])
            text = row['text'] if dataset == 'oasst' else row['user_input']
            messages = tree_text[native] if dataset == 'oasst' else [text]
            keys = set().union(*(content_keys(value) for value in messages))
            if native in (old_oa_ids if dataset == 'oasst' else old_tc_ids):
                excluded[dataset + ':old_native_id'] += 1; continue
            matches = keys & old_keys
            if matches:
                for kind in {kind for kind, value in matches}: excluded[dataset + ':old_' + kind] += 1
                continue
            record = dict(dataset=dataset, native_id=native,
                revision=PINS[0 if dataset == 'oasst' else 1]['revision'],
                root_message_id=str(row['message_id']) if dataset == 'oasst' else None,
                root_raw_sha256=sha(text), root_normalized_sha256=sha(normalized(text)),
                root_tokens=len(tokenizer.encode(text, add_special_tokens=False)), root_characters=len(text),
                source_user_messages=len(messages), exclusion_keys_sha256=sha(json.dumps(sorted(keys))),
                urls=len(canonical_urls(text)))
            record['rank_sha256'] = sha('natural-source-draft-v1:' + dataset + ':' + record['revision'] + ':' + native)
            candidates.append((record, keys)); kept += 1
        counts[dataset] = dict(raw_rows=len(oasst) if dataset == 'oasst' else len(toxic),
            eligible_roots=len(items), after_old_exclusions=kept)
    selected = []; selected_keys = []; used = set(); by_source = Counter(); duplicate_rejections = Counter()
    for record, keys in sorted(candidates, key=lambda item: item[0]['rank_sha256']):
        dataset = record['dataset']
        if by_source[dataset] >= 224: continue
        overlap = used & keys
        if overlap:
            for kind in {kind for kind, value in overlap}: duplicate_rejections[dataset + ':' + kind] += 1
            continue
        index = by_source[dataset]
        record = dict(record, split='fit' if index < 128 else 'validation' if index < 160 else 'test')
        selected.append(record); selected_keys.append(keys); by_source[dataset] += 1; used |= keys
    require(by_source == {'oasst': 224, 'toxicchat0124': 224}, 'Insufficient disjoint roots.')
    legacy = read(anchored_artifact('candidate-source-keys-draft.json', LEGACY_DRAFT_SHA256))
    require(selected == legacy['source_only_keys'] and canonical_sha(selected) == LEGACY_KEYS_SHA256,
            'Replayed source proposal differs from approved448-root ordering; refusing silent reranking.')
    for split in SPLITS:
        for source in SOURCES:
            require(sum(row['split'] == split and row['dataset'] == source for row in selected) == SPLIT_ROOTS_PER_SOURCE[split],
                    'Balanced source split counts changed.')
    require(not old_keys & used and sum(map(len, selected_keys)) == len(used), 'Old or cross-root de-duplication key leakage.')
    proof = dict(old_requested_conversations=200, old_complete_conversations=184, old_failed_conversations=16,
        old_user_turns=225, old_csv_user_text_preserved_exactly=True, old_oasst_turns_recovered=125,
        old_oasst_native_trees_excluded=100, old_toxicchat_unique_turns_recovered=100, old_toxicchat_native_ids_excluded=108,
        toxicchat0124_and1123_train_user_roots_identical=True, eligibility=counts,
        exclusion_counts=dict(excluded), cross_root_duplicate_rejections=dict(duplicate_rejections),
        old_key_count=len(old_keys), selected_key_count=len(used), old_selected_key_intersection=0,
        selected_duplicate_key_occurrences=0, old_url_count=sum(kind == 'url' for kind, value in old_keys),
        selected_native_id_count=len({(row['dataset'], row['native_id']) for row in selected}),
        source_keys_sha256=canonical_sha(selected), preserves_existing_draft_exactly=True,
        grouping='OASST entire native message_tree_id; ToxicChat native conv_id. All English source user messages in an OASST tree contribute exclusion keys, although only root IDs are proposed here.',
        key_types='Exact/NFKC-casefold-whitespace full text; raw/normalized/marker-sanitized first192-token hashes; canonical URL; native tree/conv ID.',
        limitation='No semantic near-duplicate, domain-disjointness or retroactive old-revision claim. Dataset text is used only for eligibility/deduplication, never classifier/model selection.')
    return selected, proof


def old_templates():
    result = []
    for name in ['systemness_data.py', 'systemness_challenges.py']:
        for node in ast.walk(ast.parse((data.HERE / name).read_text())):
            if not isinstance(node, ast.Assign) or not any(isinstance(target, ast.Name) and target.id in ['systems', 'instructions'] for target in node.targets): continue
            try: values = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError): continue
            if isinstance(values, list) and all(isinstance(value, str) for value in values):
                result.extend(dict(source=name, catalog=node.targets[0].id, text=value) for value in values)
    require(len(result) == 44, 'Expected old8 natural,32 unseen-system and4 C4-prefix wordings.')
    return result


def build_templates(tokenizer, sources):
    inventories = dict(fit=FIT_WORDINGS, validation=VALIDATION_WORDINGS, test=TEST_WORDINGS)
    legacy = old_templates(); content_keys = ContentKeys(tokenizer)
    old_keys = set().union(*(content_keys(item['text']) for item in legacy))
    used = set(); templates = []
    for split in SPLITS:
        require(len(inventories[split]) == TEMPLATE_COUNTS[split], 'System wording count changed.')
        for index, (family, text) in enumerate(inventories[split]):
            require(text.strip() == text and not any(marker in text for marker in long_data.RESERVED), 'Invalid normal system wording.')
            keys = content_keys(text)
            require(not old_keys & keys and not used & keys, 'System wording exact/normalized/prefix/URL overlap.')
            used |= keys
            templates.append(dict(template_id=f'fresh-natural-{split}-{index:02d}', family_id=f'{split}/{family}',
                split=split, role='system', text=text, text_sha256=sha(text), normalized_sha256=sha(normalized(text)),
                token_count=len(tokenizer.encode(text, add_special_tokens=False)), exclusion_keys_sha256=sha(json.dumps(sorted(keys)))))
    require(len({row['family_id'] for row in templates}) == len({row['template_id'] for row in templates}) == 80,
            'Template/family IDs are not unique.')
    assignments = []
    for split in SPLITS:
        pool = [row for row in templates if row['split'] == split]
        for source in SOURCES:
            group = [row for row in sources if row['split'] == split and row['dataset'] == source]
            for index, row in enumerate(group):
                template = pool[index % len(pool)]
                assignments.append(dict(dataset=source, native_id=row['native_id'], split=split,
                    root_raw_sha256=row['root_raw_sha256'], template_id=template['template_id'], family_id=template['family_id']))
    frame = pd.DataFrame(assignments)
    for split in SPLITS:
        selected = frame[frame.split == split]
        require(selected.family_id.nunique() == TEMPLATE_COUNTS[split]
            and selected.groupby(['family_id', 'dataset']).size().eq(SPLIT_ROOTS_PER_SOURCE[split] // TEMPLATE_COUNTS[split]).all(),
            'System-family/source root assignment is unbalanced.')
    proof = dict(wordings=80, wording_counts=TEMPLATE_COUNTS, one_wording_per_family=True, existing_wordings_excluded=44,
        old_exact_normalized_prefix_url_overlap=0, new_cross_wording_key_overlap=0,
        split_family_ids_disjoint=True, test_families=32, test_roots_per_family=4, test_roots_per_source_per_family=2,
        effective_positive_bootstrap_clusters=32,
        bootstrap='For positive system evidence, resample32 test template families, carrying their four conversations and paired model outputs together. Do not treat128 conversations or system tokens as128 independent positive stimuli.',
        limitation='Distinct fixed wordings/family IDs are not proof of semantic independence or a random sample of all system prompts. Report32 stimuli and use exploratory cluster intervals.',
        legacy_catalog_sha256=canonical_sha(legacy), template_catalog_sha256=canonical_sha(templates),
        assignments_sha256=canonical_sha(assignments), normal_native_system_tags=True)
    return templates, assignments, proof


def generation_proposal():
    return dict(status='proposal_only_requires_separate_generation_review', generation_run=False, gpu_work_run=False,
        requested_dialogues=dict(fit=256, validation=64, test=128),
        turns='Up to2 authentic source-user turns for OASST, following a predeclared deterministic native tree path when available; ToxicChat supplies one authentic user turn. Never describe a constructed follow-up as an original source turn.',
        path_not_yet_frozen=True,
        turn_limit=2, readout_body_cap_tokens_per_message=192,
        body_cap_scope='The192-token cap is for later role-readout serialization, after marker sanitization. Retain full generated/raw body hashes and mark each capped body. It is not a rule to fabricate a completed generation.',
        native_channels='Explicit normal system message; user messages; actual assistant analysis and final channels parsed from Harmony. Do not relabel analysis as final or combine both into one body. Tool recall is unavailable when no real tool message exists.',
        candidate_max_new_tokens_per_turn=2048, parameters_not_yet_frozen=True,
        fit_only_budget_pilot='If needed, select a completion budget using only a predeclared fit subset and completion/length statistics, not classifier accuracy. Freeze model, parameters, seeds and all finish/failure rules before any test generation.',
        maximum_turns_if_all_oasst_two_turns=672, maximum_output_tokens_at2048=1376256,
        maximum_test_turns=192, maximum_test_output_tokens_at2048=393216,
        budget_limitation='These are output-token caps, not runtime estimates or completion-rate promises. Source paths and generation backend still need review; original20B loader hardcodes inaccessible paths and must not be invoked.',
        finish='A complete response must contain a genuine nonempty final channel and its generated terminal return marker. Analysis may be absent; record its absence rather than inventing text. A length-capped unfinished analysis/final remains incomplete.',
        failures='Keep every predeclared source ID in an attempt/status ledger. Record length limit, missing final, empty response and runtime failure separately. No silent ID replacement, retry, deletion or fabricated terminal marker; any retries/reserves require a prior explicit rule.',
        selection_boundary='Generate and inspect fit/validation first. Freeze readout weights, hyperparameters and threshold before generating/evaluating the128 test roots. Test templates and root IDs must be fixed first, and test outputs cannot tune the readout.',
        tests_generated=False, classifier_predictions_read=False, primary_changed=False)


def immutable_json(path, value):
    if path.exists(): require(read(path) == value, f'Candidate artifact differs; refusing overwrite: {path}')
    else: data.atomic_json(path, value)


def immutable_copy(path, source):
    raw = Path(source).read_bytes()
    if path.exists(): require(path.read_bytes() == raw, f'Anchored review artifact changed: {path}')
    else: path.write_bytes(raw)


def payloads():
    tokenizer = long_data.load_tokenizer()
    sources, source_proof = build_sources(tokenizer)
    templates, assignments, template_proof = build_templates(tokenizer, sources)
    return {'natural-source-keys.json': dict(schema='hf-native-root-ids-v1', source_keys=sources,
                source_keys_sha256=LEGACY_KEYS_SHA256, no_c4_line_ids=True, proposal_only=True),
        'source-exclusion-proof.json': source_proof,
        'system-templates.json': dict(schema='normal-system-wordings-v1', templates=templates, proposal_only=True),
        'template-assignments.json': dict(assignments=assignments, proposal_only=True),
        'template-audit.json': template_proof, 'generation-proposal.json': generation_proposal()}


def candidate_folder(current):
    return OUT / 'candidates' / canonical_sha(current)[:16]


def prepare(download_missing=False):
    snapshots(download_missing=download_missing)
    current = signature(); values = payloads(); folder = candidate_folder(current)
    folder.mkdir(parents=True, exist_ok=True)
    immutable_copy(folder / 'candidate-source-keys-draft.json',
                   anchored_artifact('candidate-source-keys-draft.json', LEGACY_DRAFT_SHA256))
    immutable_copy(folder / 'availability.json', anchored_artifact('availability.json', AVAILABILITY_SHA256))
    for name, value in values.items(): immutable_json(folder / name, value)
    proposal = dict(schema='natural-source-proposal-v1', status='review_candidates_only', signature=current,
        signature_sha256=canonical_sha(current), artifacts_sha256={name: data.digest(folder / name) for name in
            [*values, 'candidate-source-keys-draft.json', 'availability.json']},
        namespaces='HF native tree/conv/root IDs only. Never document-keys.json, C4 document IDs or shard-line IDs.',
        tests_read_by_classifier=False, conversations_generated=False, gpu_work_run=False,
        frozen_head_sources_or_protocols_changed=False, primary_changed=False,
        next_gate='Review source and system-template candidates, then explicitly freeze the generation protocol separately.')
    immutable_json(folder / 'source-proposal.json', proposal)
    verify(expected_values=values)
    print(json.dumps(dict(candidate_directory=str(folder), proposal_sha256=data.digest(folder / 'source-proposal.json'),
        source_sha256=current['source_sha256'], source_keys_sha256=LEGACY_KEYS_SHA256,
        roots=448, templates=80, effective_positive_bootstrap_clusters=32, gpu_work_run=False,
        conversations_generated=False)), flush=True)


def verify(expected_values=None):
    current = signature(); folder = candidate_folder(current); proposal = read(folder / 'source-proposal.json')
    require(proposal['signature'] == current and proposal['signature_sha256'] == canonical_sha(current), 'Source proposal signature changed.')
    require(not proposal['gpu_work_run'] and not proposal['conversations_generated']
        and not proposal['tests_read_by_classifier'] and not proposal['primary_changed'], 'Source-only boundary changed.')
    for name, digest in proposal['artifacts_sha256'].items():
        require(data.digest(folder / name) == digest, f'Candidate artifact bytes changed: {name}')
    require(data.digest(folder / 'candidate-source-keys-draft.json') == LEGACY_DRAFT_SHA256
        and data.digest(folder / 'availability.json') == AVAILABILITY_SHA256, 'Approved original review anchor changed.')
    expected_values = payloads() if expected_values is None else expected_values
    require(all(read(folder / name) == value for name, value in expected_values.items()), 'Replayed CPU source/template protocol differs.')
    return dict(candidate_directory=str(folder), proposal_sha256=data.digest(folder / 'source-proposal.json'),
        source_sha256=current['source_sha256'], source_keys_sha256=LEGACY_KEYS_SHA256,
        roots=448, templates=80, effective_positive_bootstrap_clusters=32, verified=True,
        gpu_work_run=False, conversations_generated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'verify'], help='Explicit CPU-only candidate preparation or read-only replay.')
    parser.add_argument('--download-missing', action='store_true', help='Prepare only: restore missing pinned public files; never fetch other snapshots.')
    args = parser.parse_args()
    require(not args.download_missing or args.stage == 'prepare', '--download-missing requires --stage prepare.')
    if args.stage == 'prepare': prepare(args.download_missing)
    elif args.stage == 'verify': print(json.dumps(verify()), flush=True)
    else: parser.print_help()
