"""Client for ds_probe_server.py plus request builders shared by measurements and the fast scorer."""
from __future__ import annotations

import json
import urllib.request

from ds_common import SLOT


def call(op, port=8791, timeout=3600, **args):
    request = urllib.request.Request(f'http://127.0.0.1:{port}/op', data=json.dumps(dict(op=op, args=args)).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        payload = json.loads(error.read())
    if not payload.get('ok'):
        raise RuntimeError(payload.get('error', 'probe server error') + '\n' + payload.get('traceback', ''))
    return payload['result'], payload['seconds']


class Builder:
    """Render (context, template) into token ids and payload / goal token indices (CPU only)."""

    def __init__(self, dataset):
        import ds_agentdojo
        import ds_contexts
        import ds_ipi
        from ds_common import Codec, MODEL
        self.contexts_module, self.ipi = ds_contexts, ds_ipi
        self.codec = Codec(MODEL)
        self.deps = ds_agentdojo.dependencies()
        self.contexts = list(dataset['contexts'])
        self.suites = {name: self.deps['get_suite'](ds_agentdojo.BENCHMARK_VERSION, name)
                       for name in sorted({c['suite'] for c in self.contexts})}
        self.ipi_behavior, self.ipi_common, goals_module, self.ipi_fill = ds_ipi.ipi_modules()
        behaviors = goals_module.load_behaviors()
        goals, _ = goals_module.build_goal_manifest(behaviors)
        self.ipi_all = {k: dict(case_id=k, suite=behaviors[k]['category'], behavior=behaviors[k], goal=goals[k]['goal'],
                                primary=goals[k]['primary'], tools=self.ipi_behavior.build_tools(behaviors[k])) for k in goals}
        self.ipi_cases = [self.ipi_all[k] for k in dataset.get('ipi_cases', [])]

    def cases(self):
        return self.contexts + self.ipi_cases

    def render(self, case, template):
        """-> dict(input_ids, payload_token_indices, goal_token_indices)."""
        if 'behavior' in case:
            payload = template.replace(SLOT, case['goal'])
            fill_config = case['behavior'].get('fill_config', {})
            messages, _ = self.ipi_common.build_text_messages(case['behavior'], payload)
            encoded = self.codec.encode(self.ipi.encoder_messages(messages), case['tools'])
            prepared_goal = json.dumps(case['goal'], ensure_ascii=False)[1:-1] if fill_config.get('json_escape') else case['goal']
            occurrence = self.ipi.locate(encoded['prompt'], encoded['offsets'], self.ipi_fill(payload, fill_config), prepared_goal)[0]
            return dict(input_ids=encoded['input_ids'], payload_token_indices=occurrence['token_indices'],
                        goal_token_indices=occurrence['goal_token_indices'])
        messages, _ = self.contexts_module.replay(self.deps, self.suites[case['suite']], case, template)
        location = self.contexts_module.locate(self.codec, messages, case['tools'], template, case['goal'])
        return dict(input_ids=location['input_ids'], payload_token_indices=location['payload_token_indices'],
                    goal_token_indices=location['goal_token_indices'])

    def render_true_user(self, case, reply='Done.'):
        """Reference: the same goal text as a genuine user turn after the (uninjected) tool result."""
        if 'behavior' in case:
            messages, _ = self.ipi_common.build_text_messages(case['behavior'], '')
            messages, tools = self.ipi.encoder_messages(messages), case['tools']
        else:
            suite = self.suites[case['suite']]
            neutral = dict(case, goal='')  # empty payload: tool results without any injected text
            messages, _ = self.contexts_module.replay(self.deps, suite, neutral, SLOT)
            tools = case['tools']
        messages = messages + [dict(role='assistant', content=reply), dict(role='user', content=case['goal'])]
        encoded = self.codec.encode(messages, tools)
        start = encoded['prompt'].rfind(case['goal'])
        end = start + len(case['goal'])
        indices = [i for i, (a, b) in enumerate(encoded['offsets']) if b > a and a < end and b > start]
        return dict(input_ids=encoded['input_ids'], payload_token_indices=indices, goal_token_indices=indices)
