# coding=utf-8
# Copyright 2022 The ML Fairness Gym Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Tests for lending.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import test_util
from environments import lending
import numpy as np
from six.moves import range


class GroupsLoanTest(parameterized.TestCase):

  @parameterized.parameters((lending.SimpleLoans,),
                            (lending.DifferentialExpressionEnv,),
                            (lending.DelayedImpactEnv,))
  def test_simulations_can_run_with_defaults(self, env_class):
    test_util.run_test_simulation(env=env_class())

  @parameterized.parameters((lending.SimpleLoans,),
                            (lending.DifferentialExpressionEnv,),
                            (lending.DelayedImpactEnv,))
  def test_some_candidates_are_successful_and_some_fail(self, env_class):
    env = env_class()
    env.seed(100)
    defaults = []
    state = env._get_state()
    for _ in range(10):
      env._applicant_updater.update(state, action=None)
      defaults.append(state.will_default)

    # Some candidates will succeed.
    self.assertTrue(any([not default for default in defaults]))
    # Some candidates will default.
    self.assertTrue(any([default for default in defaults]))

  # SimpleLoans is not included in this test since all applicants are drawn
  # from the same group in that environment.
  @parameterized.parameters((lending.DifferentialExpressionEnv,),
                            (lending.DelayedImpactEnv,))
  def test_candidates_are_from_different_groups(self, env_class):
    env = env_class()
    env.seed(100)
    groups = []
    state = env._get_state()
    for _ in range(10):
      env._applicant_updater.update(state, action=None)
      groups.append(state.group)
    # Some candidates are from a different group than the first candidate.
    self.assertTrue(np.any([group != groups[0] for group in groups]))

  def test_credit_score_distributions_change(self):
    env = lending.DelayedImpactEnv()
    state = env._get_state()

    # Candidate is in group 0
    state.group = [1, 0]
    state.group_id = 0
    # Candidate will default.
    state.will_default = 1
    # Should move probability mass from clusters 3 to 2.
    state.applicant_features = [0]*7
    state.applicant_features[3] = 1

    lending._CreditShift().update(state, lending.LoanDecision.ACCEPT)

    def get_cluster_probs(params):
      return params.applicant_distribution.components[0].weights

    self.assertLess(
        get_cluster_probs(state.params)[3],
        get_cluster_probs(env.initial_params)[3])
    self.assertGreater(
        get_cluster_probs(state.params)[2],
        get_cluster_probs(env.initial_params)[2])

  def test_delayed_impact_env_has_multinomial_observation_space(self):
    env = lending.DelayedImpactEnv()
    for _ in range(10):
      features = env.observation_space.sample()['applicant_features']
      self.assertEqual(features.sum(), 1)
      self.assertSameElements(features, {0, 1})

  def test_higher_credit_scores_default_less(self):
    env = lending.DelayedImpactEnv()
    high_scores = []
    low_scores = []
    rng = np.random.RandomState()
    rng.seed(100)
    for _ in range(1000):
      applicant = env.initial_params.applicant_distribution.sample(rng)
      if np.argmax(applicant.features) > 4:
        high_scores.append(applicant)
      else:
        low_scores.append(applicant)

    self.assertNotEmpty(high_scores)
    self.assertNotEmpty(low_scores)

    self.assertLess(
        np.mean([applicant.will_default for applicant in high_scores]),
        np.mean([applicant.will_default for applicant in low_scores]))

  @parameterized.parameters((lending.SimpleLoans,),
                            (lending.DifferentialExpressionEnv,))
  def test_render_succeeds_for_two_dimensional_environments(self, env_class):
    env = env_class()
    test_util.run_test_simulation(env=env)
    env.render()

  def test_render_fails_for_high_dimensional_environments(self):
    env = lending.DelayedImpactEnv()
    test_util.run_test_simulation(env=env)
    with self.assertRaises(NotImplementedError):
      env.render()

  def test_unsupported_render_modes_fail(self):
    env = lending.SimpleLoans()
    test_util.run_test_simulation(env=env)
    for mode in ['rgb_array', 'ansi', 'something_else']:
      with self.assertRaises(NotImplementedError):
        env.render(mode)


if __name__ == '__main__':
  absltest.main()
