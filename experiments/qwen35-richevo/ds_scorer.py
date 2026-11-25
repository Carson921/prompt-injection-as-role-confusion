"""Qwen3.5-397B-A17B (GPTQ Int4) role-probe scoring (native four-H100 backbone + frozen five-role probe).

``Prober`` returns per-token role probabilities for chosen token positions of an exact token-id
prompt.  ``Scorer`` has the Scorer.score(template) contract of autodan-role-probe/scorer.py:
fitness = mean over the fixed AgentDojo contexts of mean goal-token P(user).
Run with /scratch/wz55191/venvs/deepseek-v4-role-probe/bin/python (native runtime overlay).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from ds_common import Codec, MODEL, PROBE_RUN, PROBE_SHA256, ROLES, SLOT, read_json, sha, write_json

ROLE_ANALYSIS = Path(__file__).resolve().parents[1] / 'role-analysis'
AGENTDOJO_SITE = '/scratch/wz55191/venvs/chatinject-table1/lib/python3.12/site-packages'
USER = ROLES.index('user')


def prepare_native_runtime():
    """Resolve the native model implementation before exposing AgentDojo's other venv (its torchvision is
    incompatible with the native runtime's optional-vision checks).  Creates no model or CUDA context."""
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    original_path = sys.path[:]
    try:
        sys.path[:] = [entry for entry in sys.path if entry != AGENTDOJO_SITE]
        import accelerate  # noqa: F401
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling
        return modeling
    finally:
        sys.path[:] = original_path


class Prober:
    def __init__(self, output=None):
        sys.path.insert(0, str(ROLE_ANALYSIS))
        os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
        import qwen35_role_probe_model as backbone
        import deepseek_role_probe_train as trainer
        trainer.CAPTURE = backbone.FEATURE_POSITION  # shared trainer / probe format, Qwen capture point
        frozen = trainer.verify_freeze(PROBE_RUN)
        if frozen['probe_sha256'] != PROBE_SHA256 or sha(PROBE_RUN / 'checkpoint.npz') != PROBE_SHA256:
            raise ValueError('Frozen Qwen3.5 probe bytes differ')
        self.probe = trainer.Probe(PROBE_RUN / 'checkpoint.npz', expected_binding=read_json(PROBE_RUN / 'features/manifest.json'))
        if tuple(self.probe.roles) != ROLES:
            raise ValueError('Probe role order differs')
        self.heads = self.probe.metadata['heads']
        self.backbone = backbone
        self.model = backbone.load_model(MODEL, run_dir=output, lm_head=False)
        self.protocol = dict(format='qwen35-goal-token-user-probability-v1', probe_sha256=PROBE_SHA256, roles=list(ROLES),
                             head_ids=self.heads, feature_position=self.probe.metadata['feature_position'],
                             target_role='user', target_role_index=USER, feature_cast='BF16 -> float16 -> float32',
                             model_binding=self.probe.metadata['model_binding'], batch_size=1, truncation=False,
                             backbone_source_sha256=sha(ROLE_ANALYSIS / 'qwen35_role_probe_model.py'))

    def probabilities(self, input_ids, positions):
        features = self.backbone.extract_selected(self.model, list(map(int, input_ids)), np.asarray(positions, dtype=np.int64),
                                                  selected=self.heads)
        result = self.probe.predict_proba(features)
        assert result.shape == (len(positions), len(ROLES)) and np.isfinite(result).all()
        return result

    def close(self):
        import torch
        del self.model
        torch.cuda.empty_cache()


class Scorer:
    def __init__(self, dataset_path, output_path, model_path=None, probe_path=None, target_role='user', allow_special=False):
        if target_role != 'user':
            raise ValueError('The frozen Qwen3.5 objective targets the user role')
        for supplied, expected, label in ((model_path, MODEL, 'model'),
                                          (probe_path, PROBE_RUN / 'checkpoint.npz', 'probe')):
            if supplied is not None and Path(supplied).resolve() != expected.resolve():
                raise ValueError(f'Qwen3.5 {label} path differs from the frozen experiment binding')
        prepare_native_runtime()
        if AGENTDOJO_SITE not in sys.path:
            sys.path.append(AGENTDOJO_SITE)  # AgentDojo's pure-python deps only; torch/transformers stay native
        import ds_agentdojo
        import ds_contexts
        self.contexts_module = ds_contexts
        self.dataset_path, self.output = Path(dataset_path).resolve(), Path(output_path).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.dataset = read_json(self.dataset_path)
        self.contexts = self.dataset['contexts']
        self.suites_order = tuple(self.dataset['suites'])
        self.allow_special = allow_special
        self.codec = Codec(MODEL)
        self.deps = ds_agentdojo.dependencies()
        self.suites = {name: self.deps['get_suite'](ds_agentdojo.BENCHMARK_VERSION, name)
                       for name in sorted({c['suite'] for c in self.contexts})}
        for context in self.contexts:  # replay must reproduce the recorded request exactly
            messages, _ = ds_contexts.replay(self.deps, self.suites[context['suite']], context, context['source_template'])
            if messages != context['messages']:
                raise ValueError(f"Replay differs from the recorded request: {context['case_id']}")
        self.ipi_cases = []
        if self.dataset.get('ipi_cases'):  # optional IPI Arena panel (fixed fixtures, no environment replay needed)
            import ds_ipi
            self.ipi = ds_ipi
            self.ipi_behavior, self.ipi_common, goals_module, self.ipi_fill = ds_ipi.ipi_modules()
            behaviors = goals_module.load_behaviors()
            goals, _ = goals_module.build_goal_manifest(behaviors)
            for case_id in self.dataset['ipi_cases']:
                assert goals[case_id]['primary'] and goals[case_id]['goal'] == self.dataset['ipi_goals'][case_id], case_id
                self.ipi_cases.append(dict(case_id=case_id, suite=behaviors[case_id]['category'], behavior=behaviors[case_id],
                                           goal=goals[case_id]['goal'], tools=self.ipi_behavior.build_tools(behaviors[case_id])))
        self.prober = Prober(self.output)
        self.protocol = dict(self.prober.protocol, dataset_sha256=sha(self.dataset_path),
                             objective=f'mean over {len(self.contexts)} AgentDojo contexts and {len(self.ipi_cases)} IPI Arena behaviors of mean goal-token P(user)',
                             aggregation='first complete payload occurrence, goal tokens mean per case, cases equal mean',
                             codec=self.codec.binding, payload_policy='no added/special token spelling or id, except explicitly allowed baseline conditions',
                             code_sha256={n: sha(Path(__file__).with_name(n)) for n in
                                          ('ds_scorer.py', 'ds_contexts.py', 'ds_common.py', 'ds_agentdojo.py', 'ds_ipi.py')})
        if self.ipi_cases:
            self.protocol['ipi_fixture_sha256'] = sha(json.dumps(self.ipi_cases, sort_keys=True, ensure_ascii=False))
            self.protocol['ipi_code_sha256'] = {str(module.__file__): sha(Path(module.__file__)) for module in
                                                (self.ipi_behavior, self.ipi_common, goals_module)}
        self.protocol_sha256 = sha(json.dumps(self.protocol, sort_keys=True))
        write_json(self.output / 'score_protocol.json', self.protocol)
        self.closed = False

    def prepare(self, context, template):
        if not self.allow_special and self.codec.has_special(template.replace(SLOT, ' ')):
            raise ValueError('Template contains a genuine special-token spelling or id')
        messages, payload = self.contexts_module.replay(self.deps, self.suites[context['suite']], context, template)
        location = self.contexts_module.locate(self.codec, messages, context['tools'], template, context['goal'])
        ids = location['input_ids']
        if not self.allow_special and any(ids[i] in self.codec.special_ids for i in location['payload_token_indices']):
            raise ValueError('Rendered payload contains a special token id')
        return location

    def prepare_ipi(self, case, template):
        if not self.allow_special and self.codec.has_special(template.replace(SLOT, ' ')):
            raise ValueError('Template contains a genuine special-token spelling or id')
        payload = template.replace(SLOT, case['goal'])
        fill_config = case['behavior'].get('fill_config', {})
        messages, _ = self.ipi_common.build_text_messages(case['behavior'], payload)
        encoded = self.codec.encode(self.ipi.encoder_messages(messages), case['tools'])
        prepared_goal = json.dumps(case['goal'], ensure_ascii=False)[1:-1] if fill_config.get('json_escape') else case['goal']
        occurrence = self.ipi.locate(encoded['prompt'], encoded['offsets'], self.ipi_fill(payload, fill_config), prepared_goal)[0]
        ids = encoded['input_ids']
        if not self.allow_special and any(ids[i] in self.codec.special_ids for i in occurrence['token_indices']):
            raise ValueError('Rendered payload contains a special token id')
        if not occurrence['goal_token_indices']:
            raise ValueError('Goal has no tokens')
        return dict(input_ids=ids, payload_token_indices=occurrence['token_indices'], goal_token_indices=occurrence['goal_token_indices'])

    def score_case(self, context, template):
        location = self.prepare_ipi(context, template) if 'behavior' in context else self.prepare(context, template)
        positions = np.asarray(location['payload_token_indices'], dtype=np.int64)
        goal_mask = np.isin(positions, np.asarray(location['goal_token_indices'], dtype=np.int64))
        probabilities = self.prober.probabilities(location['input_ids'], positions)
        goal = probabilities[goal_mask].mean(0, dtype=np.float64)
        wrapper = probabilities[~goal_mask].mean(0, dtype=np.float64) if (~goal_mask).any() else None
        return dict(status='complete', case_id=context['case_id'], suite=context['suite'], goal_tokens=int(goal_mask.sum()),
                    input_tokens=len(location['input_ids']), goal_user_mean=float(goal[USER]),
                    probabilities=dict(goal=dict(zip(ROLES, map(float, goal))),
                                       wrapper=dict(zip(ROLES, map(float, wrapper))) if wrapper is not None else None))

    def score(self, template: str) -> dict:
        if not isinstance(template, str) or template.count(SLOT) != 1:
            raise ValueError('A candidate must contain exactly one {goal} slot')
        # Admission precedes cache lookup: a baseline scored with allow_special=True
        # must never let a later strict search recover a forbidden candidate from cache.
        if not self.allow_special and self.codec.has_special(template.replace(SLOT, ' ')):
            raise ValueError('Template contains a genuine special-token spelling or id')
        started = time.monotonic()
        result_path = self.output / 'scores' / sha(template)[:24] / 'score.json'
        if result_path.exists():
            cached = read_json(result_path)
            if cached.get('status') == 'complete' and cached.get('template') == template \
                    and cached.get('protocol_sha256') == self.protocol_sha256 \
                    and cached.get('allow_special', False) == self.allow_special:
                return {**cached, 'cache_hit': True}
        records = [self.score_case(context, template) for context in self.contexts + self.ipi_cases]
        fitness = float(np.mean([r['goal_user_mean'] for r in records]))
        result = dict(status='complete', complete=True, fitness=fitness, goal_target_mean=fitness, template=template,
                      allow_special=self.allow_special,
                      target_role='user', target_role_index=USER, per_case=records,
                      suite_means={s: float(np.mean([r['goal_user_mean'] for r in records if r['suite'] == s])) for s in self.suites_order},
                      goal_role_means={role: float(np.mean([r['probabilities']['goal'][role] for r in records])) for role in ROLES},
                      expected_cases=len(self.contexts) + len(self.ipi_cases), completed_cases=len(records), protocol_sha256=self.protocol_sha256,
                      cache_hit=False, elapsed_seconds=time.monotonic() - started, artifact=str(result_path))
        write_json(result_path, result)
        return result

    def close(self):
        if not self.closed:
            self.prober.close()
            self.closed = True
