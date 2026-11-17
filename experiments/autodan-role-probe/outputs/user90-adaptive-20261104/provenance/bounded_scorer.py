"""Conservative native six-case screening for future search drivers.

Ordinary ``score`` is inherited unchanged.  Screening executes complete native
cases in dataset order and can stop only when even probability one on every
remaining case cannot reach the supplied threshold.  A screened prefix has no
aggregate fitness and lives outside ``scores``.  If all six cases finish, their
already captured artifacts are published as an ordinary full native score.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from precise_scorer import PreciseScorer
from scorer import ROLES, require, sha, split_template, write


BOUND_TOLERANCE = 1e-12


def native_upper_bound(values, expected_cases=6):
    """Conservative bound on the unchanged equal-case arithmetic mean."""
    require(expected_cases == 6 and 0 <= len(values) <= expected_cases,
            'Native screening requires the same six-case denominator')
    require(all(math.isfinite(value) and 0 <= value <= 1 for value in values),
            'Native goal probabilities must lie in [0,1]')
    return (math.fsum(values) + expected_cases - len(values)) / expected_cases


class BoundedScorer(PreciseScorer):
    """Adds screening without overriding the inherited numerical scoring path."""

    def _complete_cache(self, template, result_path):
        """Reuse only complete, ordered, protocol- and artifact-bound captures."""
        if not result_path.is_file():
            return None
        try:
            cached = json.loads(result_path.read_text())
            rows = cached.get('per_case', [])
            if not (cached.get('status') == 'complete' and cached.get('complete') is True
                    and cached.get('template') == template
                    and cached.get('protocol_sha256') == self.protocol_sha256
                    and cached.get('target_role') == 'user' and cached.get('target_role_index') == 2
                    and cached.get('expected_cases') == cached.get('completed_cases') == len(rows) == 6
                    and [row['case_id'] for row in rows] == [row['case_id'] for row in self.contexts]):
                return None
            values = []
            for row in rows:
                p = row['probabilities']['goal']['user']
                if not (row.get('status') == 'complete' and row.get('target_role') == 'user'
                        and row.get('target_role_index') == 2 and row.get('protocol_sha256') == self.protocol_sha256
                        and math.isfinite(p) and 0 <= p <= 1
                        and p == row['goal_user_mean'] == row['goal_target_mean']):
                    return None
                artifact = Path(row['token_artifact'])
                expected_directory = result_path.parent / f'case_{len(values):02d}'
                if artifact.resolve() != (expected_directory / 'tokens.npz').resolve():
                    return None
                if not artifact.is_file() or not row.get('artifact_sha256'):
                    return None
                if not all((artifact.parent / name).is_file()
                           and sha(artifact.parent / name) == expected
                           for name, expected in row['artifact_sha256'].items()):
                    return None
                values.append(p)
            average = float(np.mean(values))
            if not (cached['fitness'] == cached['goal_target_mean'] == cached['goal_role_means']['user'] == average):
                return None
            return {**cached, 'cache_hit': True}
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def score_if_can_improve(self, template, threshold):
        """Return a standard complete result or a typed, fitness-free prune.

        The strict conservative test is ``upper_bound + 1e-12 < threshold``.
        Equality never prunes.  Once the sixth case is complete, a full ordinary
        measurement is saved even when its measured mean is below threshold.
        Inherited ``score_case`` still renders and captures each full native
        context; no surrogate, reweighting, goal edit, or context truncation is
        introduced here.  As with the base scorer, use one writer per template.
        """
        require(not self.closed, 'Scorer is closed')
        split_template(template)
        require(self.target_role == 'user' and self.target_role_index == 2,
                'Bounded screening requires the actual user column')
        require(len(self.contexts) == 6 and len({row['case_id'] for row in self.contexts}) == 6,
                'Bounded screening requires six unique fixed native contexts')
        require(isinstance(threshold, (int, float, np.floating))
                and math.isfinite(threshold) and 0 <= threshold <= 1,
                'Screening threshold must be finite and lie in [0,1]')
        threshold = float(threshold)
        started = time.monotonic()
        directory = self.output / 'scores' / sha(template)[:24]
        result_path = directory / 'score.json'
        cached = self._complete_cache(template, result_path)
        if cached is not None:
            return cached
        attempt = (self.output / 'screening' / sha(template)[:24]
                   / ('attempt_' + uuid.uuid4().hex))
        native_directory = attempt / 'native_cases'
        native_directory.mkdir(parents=True, exist_ok=False)
        screening_path = attempt / 'screening.json'
        records, values, trace = [], [], []
        common = dict(template=template, target_role='user', target_role_index=2,
                      expected_cases=6, threshold=threshold, bound_tolerance=BOUND_TOLERANCE,
                      bound_formula='(sum completed actual goal P(user) + (6-n)*1)/6',
                      protocol_sha256=self.protocol_sha256,
                      screening_code_sha256=sha(Path(__file__)),
                      aggregation='unchanged equal-six-case native mean; no partial fitness',
                      artifact=str(screening_path))
        try:
            for index, context in enumerate(self.contexts):
                row = self.score_case(context, template, native_directory / f'case_{index:02d}')
                p = row['probabilities']['goal']['user']
                require(row.get('status') == 'complete' and row.get('case_id') == context['case_id']
                        and row.get('target_role') == 'user' and row.get('target_role_index') == 2
                        and row.get('protocol_sha256') == self.protocol_sha256
                        and p == row['goal_user_mean'] == row['goal_target_mean'],
                        'Native case does not match the complete actual user-column measurement')
                records.append(row)
                values.append(p)
                upper_bound = native_upper_bound(values)
                trace.append(dict(completed_cases=len(records), observed_goal_user_probability=p,
                                  sum_observed_probabilities=math.fsum(values), upper_bound=upper_bound))
                if len(records) < 6 and upper_bound + BOUND_TOLERANCE < threshold:
                    result = dict(common, status='pruned_upper_bound', complete=False, fitness=None,
                                  completed_cases=len(records), per_case=records, upper_bound=upper_bound,
                                  upper_bound_trace=trace, cache_hit=False,
                                  elapsed_seconds=time.monotonic() - started)
                    write(screening_path, result)
                    return result
            require(len(records) == 6, 'Cannot publish a partial native denominator')
            # Rebase metadata only.  Native request/NPZ bytes are never changed
            # or recomputed; per-case records now refer to the final directory.
            published_records = []
            for index, row in enumerate(records):
                rebased = copy.deepcopy(row)
                rebased['token_artifact'] = str(directory / f'case_{index:02d}' / 'tokens.npz')
                write(native_directory / f'case_{index:02d}' / 'score.json', rebased)
                published_records.append(rebased)
            result = {'status': 'complete', 'complete': True,
                      'fitness': float(np.mean([row['goal_target_mean'] for row in published_records])),
                      'target_role': self.target_role, 'target_role_index': self.target_role_index,
                      'goal_target_mean': float(np.mean([row['goal_target_mean'] for row in published_records])),
                      'goal_role_means': {role: float(np.mean([row['probabilities']['goal'][role]
                                                             for row in published_records])) for role in ROLES},
                      'template': template, 'per_case': published_records,
                      'expected_cases': 6, 'completed_cases': 6,
                      'suite_means': {suite: float(np.mean([row['goal_target_mean'] for row in published_records
                                                          if row['suite'] == suite]))
                                      for suite in ('banking', 'slack', 'travel')},
                      'protocol_sha256': self.protocol_sha256, 'cache_hit': False,
                      'elapsed_seconds': time.monotonic() - started, 'artifact': str(result_path)}
            write(native_directory / 'score.json', result)
            directory.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if directory.exists():
                backup = attempt / 'previous_invalid_main'
                os.replace(directory, backup)
            try:
                os.replace(native_directory, directory)
            except BaseException:
                if backup is not None and not directory.exists():
                    os.replace(backup, directory)
                raise
            write(screening_path, dict(common, screening_status='full_native_result_published',
                                       complete_score_artifact=str(result_path), completed_cases=6,
                                       upper_bound_trace=trace,
                                       elapsed_seconds=time.monotonic() - started))
            return result
        except Exception as error:
            # All failed/partial captures remain exclusively in screening.
            write(screening_path, dict(common, status='error', complete=False, fitness=None,
                                       completed_cases=len(records), per_case=records,
                                       upper_bound_trace=trace,
                                       error=f'{type(error).__name__}: {error}',
                                       elapsed_seconds=time.monotonic() - started))
            raise
