"""CPU-only preparation of the exact historical AD6 and IPI12 panels for DeepSeek.

Builds separate datasets; does not start a scorer, model, API client or optimization.
Payloads remain in dataset artifacts as required for replay and are never printed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
EXPERIMENT = Path(__file__).resolve().parents[2]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))

network_attempts = []


def no_network(*args, **kwargs):
    network_attempts.append(True)
    raise RuntimeError('Network access is disabled during CPU-only protocol preparation')


socket.create_connection = no_network
socket.socket.connect = no_network

from ds_common import Codec, MODEL, read_json, sha, write_json
import ds_agentdojo
import ds_contexts
import ds_ipi


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    old_ad_run = ROOT / 'experiments/autodan-role-probe/outputs/richevo-20261105'
    old_ad_path = ROOT / 'experiments/autodan-role-probe/outputs/diverse8h-20261105/dataset.json'
    old_ipi_run = ROOT / 'experiments/ipi-arena-richevo/outputs/ipievo-20261108'
    old_ipi_path = old_ipi_run / 'ipi_dataset.json'
    baseline = EXPERIMENT / 'outputs/baseline-20261118'
    joint_path = EXPERIMENT / 'outputs/dsevo-20261118/dataset.json'
    old_ad, old_ipi, joint = map(read_json, (old_ad_path, old_ipi_path, joint_path))
    old_ad_protocol = read_json(old_ad_run / 'score_protocol.json')
    old_ipi_protocol = read_json(old_ipi_run / 'score_protocol.json')
    assert sha(old_ad_path) == old_ad_protocol['dataset_sha256']
    assert sha(old_ipi_path) == old_ipi_protocol['dataset_sha256']
    assert str(old_ad_path.relative_to(ROOT)) in (old_ad_run / 'run.sh').read_text()
    assert str(old_ipi_path.relative_to(ROOT)) in (old_ipi_run / 'run.sh').read_text()
    old_ad_ids = [c['pair_id'] for c in old_ad['contexts']]
    current_ad_ids = [c['case_id'] for c in joint['contexts']]
    assert len(old_ad_ids) == len(set(old_ad_ids)) == 6
    assert Counter(c['suite'] for c in old_ad['contexts']) == dict(banking=2, slack=2, travel=2)
    assert old_ad['sample_manifest']['selection_uses_outcomes'] is False
    assert old_ad['sample_manifest']['selection_uses_exposure'] is False
    assert [r['pair_id'] for r in old_ad['sample_manifest']['selected_pairs']] == old_ad_ids
    template = read_json(baseline / 'conditions_agentdojo.json')['original']['template']
    codec = Codec(MODEL)
    deps = ds_agentdojo.dependencies()
    suites = {name: deps['get_suite'](ds_agentdojo.BENCHMARK_VERSION, name)
              for name in ('banking', 'slack', 'travel')}
    contexts, ad_checks, missing = [], [], []
    for old_context in old_ad['contexts']:
        case_id = old_context['pair_id']
        case_path = baseline / 'agentdojo/cases/original' / (case_id + '.json')
        if not case_path.exists():
            missing.append(dict(benchmark='agentdojo', case_id=case_id, reason='missing DeepSeek baseline case'))
            continue
        record = read_json(case_path)
        context = ds_contexts.context_from_record(record)
        if record['status'] != 'complete' or context is None:
            missing.append(dict(benchmark='agentdojo', case_id=case_id, reason='baseline has no completed injection exposure'))
            continue
        suite = suites[context['suite']]
        assert context['goal'] == old_context['actual_goal'] == suite.injection_tasks[context['injection_task_id']].GOAL
        assert context['user_task_id'] in suite.user_tasks
        context.update(source_template=template, source_case_sha256=sha(case_path),
                       source_case_path=str(case_path), historical_pair_id=case_id)
        replayed, _ = ds_contexts.replay(deps, suite, context, template)
        assert replayed == context['messages'], case_id
        location = ds_contexts.locate(codec, replayed, context['tools'], template, context['goal'])
        source_call = record['calls'][context['exposure_call']]
        assert len(location['input_ids']) == source_call['prompt_tokens'], case_id
        assert location['goal_token_indices'] and set(location['goal_token_indices']) <= set(location['payload_token_indices'])
        context.update(prompt_tokens=len(location['input_ids']), goal_tokens=len(location['goal_token_indices']))
        contexts.append(context)
        historical_request = Path(old_context['source_request'])
        historical_request_verified = historical_request.exists() and sha(historical_request) == old_context['source_request_sha256']
        assert historical_request_verified
        ad_checks.append(dict(case_id=case_id, suite=context['suite'],
                             historical_case_id=old_context['case_id'],
                             historical_source_request=old_context['source_request'],
                             historical_source_request_sha256=old_context['source_request_sha256'],
                             historical_source_request_sha256_verified=historical_request_verified,
                             historical_source_call_id=old_context['source_call_id'],
                             historical_source_phase=old_context['source_phase'],
                             deepseek_source_case=str(case_path), deepseek_source_case_sha256=sha(case_path),
                             exposure_call=context['exposure_call'], historical_goal_matches=True,
                             replay_messages_exact=True, recorded_prompt_length_matches=True,
                             rendered_prompt_tokens=len(location['input_ids']), goal_tokens=len(location['goal_token_indices']),
                             rendered_token_ids_sha256=text_hash(json.dumps(location['input_ids'], separators=(',', ':')))))
    ad_dataset = dict(schema='deepseek-agentdojo-contexts-v1', optimization_mode='agentdojo_only',
                      suites=['banking', 'slack', 'travel'], contexts=contexts, ipi_cases=[], ipi_goals={},
                      source_run=str(baseline / 'agentdojo'), source_condition='original', source_template=template,
                      per_suite=2, historical_dataset=str(old_ad_path), historical_dataset_sha256=sha(old_ad_path),
                      objective='equal mean over the six historical AgentDojo pair ids, using native DeepSeek baseline exposures',
                      provenance_note='Exact historical pair identities and goals; native DeepSeek trajectories, not byte-identical historical GPT-OSS requests.')
    if len(contexts) == 6:
        write_json(HERE / 'agentdojo6.dataset.json', ad_dataset)

    behavior_module, common, goals_module, prepare_fill_text = ds_ipi.ipi_modules()
    behaviors = goals_module.load_behaviors()
    goals, _ = goals_module.build_goal_manifest(behaviors)
    panel = list(old_ipi['panel'])
    assert len(panel) == len(set(panel)) == 12
    categories = Counter(behaviors[c]['category'] for c in panel)
    assert categories == dict(browser=4, coding=4, tool=4)
    protocol = read_json(baseline / 'ipi/protocol.json')
    protocol_rows = {(r['condition'], r['case_id']): r for r in protocol['rows']}
    ipi_checks = []
    for case_id in panel:
        assert goals[case_id]['primary'] and old_ipi['goals'][case_id] == goals[case_id]['goal']
        assert sha(json.dumps(behaviors[case_id], sort_keys=True)) == old_ipi['data_sha256'][case_id]
        row = protocol_rows[('original', case_id)]
        request_path = baseline / 'ipi' / row['request']
        request = read_json(request_path)
        assert sha(request_path) == row['request_sha256']
        encoded = codec.encode(ds_ipi.encoder_messages(request['messages']), request['tools'])
        assert encoded['input_ids'] == request['prompt_token_ids']
        prepared = prepare_fill_text(request['payload'], request['fill_config'])
        goal_text = goals[case_id]['goal']
        prepared_goal = json.dumps(goal_text, ensure_ascii=False)[1:-1] if request['fill_config'].get('json_escape') else goal_text
        located = ds_ipi.locate(encoded['prompt'], encoded['offsets'], prepared, prepared_goal)
        indices = located[request['scored_occurrence']]['goal_token_indices']
        assert indices == request['goal_token_indices'] and indices
        ipi_checks.append(dict(case_id=case_id, category=behaviors[case_id]['category'], primary=True,
                              historical_goal_matches=True, historical_fixture_hash_matches=True,
                              deepseek_source_request=str(request_path), deepseek_source_request_sha256=sha(request_path),
                              rendered_token_ids_exact=True, goal_indices_exact=True,
                              rendered_prompt_tokens=len(encoded['input_ids']), goal_tokens=len(indices),
                              rendered_token_ids_sha256=text_hash(json.dumps(encoded['input_ids'], separators=(',', ':')))))
    ipi_dataset = dict(schema='deepseek-agentdojo-contexts-v1', optimization_mode='ipi_only',
                       suites=['browser', 'coding', 'tool'], contexts=[], ipi_cases=panel,
                       ipi_goals={c: goals[c]['goal'] for c in panel},
                       source_run=str(baseline / 'ipi'), source_condition='original',
                       historical_dataset=str(old_ipi_path), historical_dataset_sha256=sha(old_ipi_path),
                       objective='equal mean over the exact twelve historical IPI panel behavior ids',
                       provenance_note='Historical IPI panel and goals are unchanged; prompts use the native DeepSeek encoder and existing text-only fixture adaptation.')
    write_json(HERE / 'ipi12.dataset.json', ipi_dataset)
    assert not network_attempts
    manifest = dict(schema='deepseek-independent-protocol-preparation-v1',
                    status='ready' if len(contexts) == 6 and len(ipi_checks) == 12 else 'incomplete',
                    cpu_only=dict(model_loaded=False, scorer_constructed=False, cuda_visible_devices='',
                                  network_connect_attempts=len(network_attempts), optimization_started=False),
                    codec=codec.binding, benchmark_version=ds_agentdojo.BENCHMARK_VERSION,
                    historical_method=dict(agentdojo='AD-only six contexts; transferred templates were also tested on IPI',
                                           ipi='separate IPI-only twelve-behavior panel, followed by full 36-task primary evaluation',
                                           current='AD6+IPI6 equal-weight joint search; a different objective and panel'),
                    sources={str(p): sha(p) for p in [old_ad_path, old_ad_run / 'run.sh', old_ad_run / 'score_protocol.json',
                                                     old_ipi_path, old_ipi_run / 'run.sh', old_ipi_run / 'score_protocol.json',
                                                     ROOT / 'experiments/autodan-role-probe/prepare.py',
                                                     ROOT / 'experiments/ipi-arena-richevo/ipi_scorer.py', joint_path,
                                                     HERE / 'prepare_protocol.py', EXPERIMENT / 'ds_common.py',
                                                     EXPERIMENT / 'ds_agentdojo.py', EXPERIMENT / 'ds_contexts.py', EXPERIMENT / 'ds_ipi.py']},
                    agentdojo=dict(historical_ids=old_ad_ids, current_joint_ids=current_ad_ids,
                                   overlap=sorted(set(old_ad_ids) & set(current_ad_ids)),
                                   suite_counts=dict(Counter(c['suite'] for c in old_ad['contexts'])),
                                   historical_selection=old_ad['sample_manifest'],
                                   historical_context_provenance=old_ad['provenance'], mappings=ad_checks),
                    ipi=dict(historical_ids=panel, current_joint_ids=joint['ipi_cases'],
                             overlap=sorted(set(panel) & set(joint['ipi_cases'])), category_counts=dict(categories),
                             panel_source='ipi_scorer.py PANEL constant, corroborated by frozen ipi_dataset.json and score_protocol dataset hash',
                             mappings=ipi_checks),
                    missing=missing,
                    limitations=['Same historical task identities and goals do not imply identical model-specific trajectories or rendered prompts.',
                                 'Historical AD contexts reused frozen GPT-OSS calls/tool outputs; the prepared AD dataset replays native DeepSeek original-baseline calls.',
                                 'CPU replay validates native messages, local encoding, recorded prompt lengths and goal positions; it does not reproduce model probabilities or ASR.',
                                 'Current joint-specific selection/controller wrappers are not an independent-run launcher; preparation does not authorize or launch new experiments.'],
                    datasets={p.name: dict(path=str(p), sha256=sha(p)) for p in [HERE / 'agentdojo6.dataset.json', HERE / 'ipi12.dataset.json'] if p.exists()})
    write_json(HERE / 'manifest.json', manifest)
    print(json.dumps(dict(status=manifest['status'], agentdojo_contexts=len(contexts), ipi_behaviors=len(ipi_checks),
                          agentdojo_old_current_overlap=len(manifest['agentdojo']['overlap']),
                          agentdojo_suite_counts=manifest['agentdojo']['suite_counts'], ipi_category_counts=dict(categories),
                          missing=missing, datasets=manifest['datasets']), ensure_ascii=False))


if __name__ == '__main__':
    main()
