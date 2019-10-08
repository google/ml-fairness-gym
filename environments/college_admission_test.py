# coding=utf-8
# Copyright 2019 The ML Fairness Gym Authors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import core
import params
import test_util
from agents import random_agents
from environments import college_admission
import numpy as np


class CollegeAdmissionsTest(absltest.TestCase):

  def _return_individual_burden(self, env, agent):
    # pylint: disable=g-complex-comprehension
    return [
        env.state.params.group_cost[group_id] *
        (agent.default_action['threshold'] - score)
        if score < agent.default_action['threshold'] else 0
        for group_id, score in zip(env.state.applicant_groups,
                                   env.state.test_scores_x)
    ]

  def test_parties_can_interact_gaming(self):
    """Test stackelberg simulation."""
    env = college_admission.CollegeAdmissionsEnv(user_params={'gaming': True})
    agent = random_agents.RandomAgent(env.action_space, None,
                                      env.observation_space)
    test_util.run_test_simulation(agent=agent, env=env, stackelberg=True)

  def test_parties_can_interact_no_gaming(self):
    """Test stackelberg simulation with no gaming."""
    env = college_admission.CollegeAdmissionsEnv(user_params={'gaming': False})
    agent = random_agents.RandomAgent(env.action_space, None,
                                      env.observation_space)
    test_util.run_test_simulation(agent=agent, env=env, stackelberg=True)

  def test_manipulate_features_no_gaming(self):
    """Test features are not manipulated when gaming is off."""
    env = college_admission.CollegeAdmissionsEnv(user_params={
        'num_applicants': 6,
        'gaming': False,
        'group_cost': {
            0: 3,
            1: 4
        }
    })
    agent = random_agents.RandomAgent(
        env.action_space,
        None,
        env.observation_space,
        default_action={
            'threshold': np.array(0.8),
            'epsilon_prob': np.array(0)
        })
    action = agent.initial_action()
    _, _, _, _ = env.step(action)
    env.state.test_scores_x = [0.1, 0.3, 0.6, 0.7, 0.7, 0.9]
    env.state.applicant_groups = [0, 1, 1, 1, 0, 0]
    env.state.true_eligible = [0, 0, 1, 1, 0, 1]
    expected_changed_scores = [0.1, 0.3, 0.6, 0.7, 0.7, 0.9]
    expected_individual_burden = [0] * env.state.params.num_applicants
    changed_scores, individual_burden = env._manipulate_features(
        env.state, action)
    self.assertTrue(
        np.all(np.isclose(expected_changed_scores, changed_scores, atol=1e-4)))
    self.assertTrue(
        np.all(
            np.isclose(
                individual_burden, expected_individual_burden, atol=1e-4)))

  def test_manipulate_features_no_max_control(self):
    """Tests that features are manipulated as expected no gaming control."""
    env = college_admission.CollegeAdmissionsEnv(
        user_params={
            'num_applicants': 6,
            'gaming': True,
            'gaming_control': np.inf,
            'noise_params': params.BoundedGaussian(max=0, mu=0, min=0, sigma=0),
            'group_cost': {
                0: 3,
                1: 4
            }
        })
    agent = random_agents.RandomAgent(
        env.action_space,
        None,
        env.observation_space,
        default_action={
            'threshold': np.array(0.8),
            'epsilon_prob': np.array(0)
        })
    env.set_scalar_reward(agent.reward_fn)
    action = agent.initial_action()
    env.step(action)
    env.state.test_scores_x = [0.1, 0.3, 0.6, 0.7, 0.7, 0.9]
    env.state.applicant_groups = [0, 1, 1, 1, 0, 0]
    env.state.true_eligible = [0, 0, 1, 1, 0, 1]
    expected_changed_scores = [0.1, 0.3, 0.8, 0.8, 0.8, 0.9]
    expected_individual_burden = self._return_individual_burden(env, agent)
    changed_scores, individual_burden = env._manipulate_features(
        env.state, action)
    self.assertTrue(
        np.all(np.isclose(expected_changed_scores, changed_scores, atol=1e-4)))
    self.assertTrue(
        np.all(
            np.isclose(
                individual_burden, expected_individual_burden, atol=1e-4)))

  def test_manipulate_features_with_max_control(self):
    """Tests that features are manipulated as expected given max gaming."""
    env = college_admission.CollegeAdmissionsEnv(
        user_params={
            'num_applicants': 6,
            'noise_params': params.BoundedGaussian(max=0, mu=0, min=0, sigma=0),
            'gaming': True,
            'gaming_control': 0.1,
            'group_cost': {
                0: 3,
                1: 4
            }
        })
    agent = random_agents.RandomAgent(
        env.action_space,
        None,
        env.observation_space,
        default_action={
            'threshold': np.array(0.8),
            'epsilon_prob': np.array(0)
        })
    env.set_scalar_reward(agent.reward_fn)
    action = agent.initial_action()
    _, _, _, _ = env.step(action)
    env.state.test_scores_x = [0.1, 0.3, 0.6, 0.7, 0.7, 0.9]
    env.state.applicant_groups = [0, 1, 1, 1, 0, 0]
    env.state.true_eligible = [0, 0, 1, 1, 0, 1]
    expected_changed_scores = [0.1, 0.3, 0.6, 0.8, 0.8, 0.9]
    expected_individual_burden = self._return_individual_burden(env, agent)
    changed_scores, individual_burden = env._manipulate_features(
        env.state, action)
    self.assertTrue(
        np.all(np.isclose(expected_changed_scores, changed_scores, atol=1e-4)))
    self.assertTrue(
        np.all(
            np.isclose(
                individual_burden, expected_individual_burden, atol=1e-4)))

  def test_cost_fn_subsidies_cost_for_group_1_with_subsidy(self):
    """Test for groupwise cost function with and without subsidies."""
    env = college_admission.CollegeAdmissionsEnv(user_params={
        'subsidize': True,
        'group_cost': {
            0: 3,
            1: 4
        },
        'subsidy_beta': 0.6
    })
    group_0_cost = env._cost_function(0.8, 0)
    group_1_cost = env._cost_function(0.8, 1)
    self.assertEqual(group_0_cost, 0.8 * 3)
    self.assertEqual(group_1_cost, 0.8 * 0.6 * 4)

  def test_cost_fn_does_not_subsidize_cost_for_group_1_with_no_subsidy(self):
    env = college_admission.CollegeAdmissionsEnv(user_params={
        'subsidize': False,
        'group_cost': {
            0: 3,
            1: 4
        }
    })
    group_1_cost = env._cost_function(0.8, 1)
    group_0_cost = env._cost_function(0.8, 0)
    self.assertEqual(group_0_cost, 0.8 * 3)
    self.assertEqual(group_1_cost, 0.8 * 4)

  def test_select_candidates(self):
    """Tests predictions by jury, given modified scores are as expected."""
    env = college_admission.CollegeAdmissionsEnv(
        user_params={'num_applicants': 4})
    agent = random_agents.RandomAgent(
        env.action_space,
        None,
        env.observation_space,
        default_action={
            'threshold': np.array(0.8),
            'epsilon_prob': np.array(0)
        })
    env.set_scalar_reward(agent.reward_fn)
    action = agent.initial_action()
    _ = env.step(action)
    env.state.test_scores_y = [0.1, 0.9, 0.8, 0.79]
    env.state.true_eligible = [0, 1, 0, 1]
    predictions, selected_ground_truth = env._select_candidates(
        env.state, action)
    self.assertEqual(list(predictions), [0, 1, 1, 0])
    self.assertEqual(list(selected_ground_truth), [2, 1, 0, 2])

  def test_one_sided_noise_generated_correctly(self):
    env = college_admission.CollegeAdmissionsEnv(
        user_params={
            'num_applicants':
                4,
            'noise_params':
                params.BoundedGaussian(min=0, max=0.3, mu=0.2, sigma=0.00001)
        })
    noise = env._add_noise(env.state.rng)
    self.assertTrue(np.isclose(0.2, noise, atol=1e-3))

  def feature_noise_propagates_to_labels(self):
    env = college_admission.CollegeAdmissionsEnv(
        user_params={
            'num_applicants':
                10,
            'noise_params':
                params.BoundedGaussian(min=0.5, max=0.5, mu=0, sigma=1)
        })
    env.state.rng = np.random.RandomState(seed=100)
    env._sample_next_state_vars(env.state)
    scores = np.array(env.state.test_scores_x)
    eligible = np.array(env.state.true_eligible)
    # Check that at least one "eligible" candidate has a lower score than an
    # ineligible one.
    self.assertLess(
        np.min(scores[eligible == 1]), np.max(scores[eligible == 0]))

  def error_raised_when_noise_params_wrong(self):
    env = college_admission.CollegeAdmissionsEnv(
        user_params={
            'noise_params':
                params.BoundedGaussian(min=0, max=0.3, mu=0, sigma=0.00001),
        })
    with self.assertRaises(ValueError):
      env._add_noise()

  def test_is_done_when_max_steps_reached(self):
    env = college_admission.CollegeAdmissionsEnv(user_params={
        'num_applicants': 4,
        'max_steps': 8
    })
    agent = random_agents.RandomAgent(
        env.action_space,
        None,
        env.observation_space,
        default_action={
            'threshold': np.array(0.8),
            'epsilon_prob': np.array(0)
        })
    with self.assertRaises(core.EpisodeDoneError):
      test_util.run_test_simulation(agent=agent, env=env, stackelberg=True)
    self.assertEqual(env.state.steps, 9)

  def test_candidates_less_than_threshold_allowed_epsilon_selection(self):
    env = college_admission.CollegeAdmissionsEnv(user_params={'gaming': False})
    env.state.test_scores_y = [0.7] * env.initial_params.num_applicants
    action = {'threshold': np.array(0.8), 'epsilon_prob': np.array(0.5)}
    selected_candidates, _ = env._select_candidates(env.state, action)
    self.assertGreater(sum(selected_candidates), 0)

  def test_candidates_less_than_threshold_not_allowed_non_epsilon_selection(
      self):
    env = college_admission.CollegeAdmissionsEnv(user_params={'gaming': False})
    env.state.test_scores_y = [0.7] * env.initial_params.num_applicants
    action = {'threshold': np.array(0.8), 'epsilon_prob': np.array(0)}
    selected_candidates, _ = env._select_candidates(env.state, action)
    self.assertEqual(sum(selected_candidates), 0)

  def test_unmanipualted_features_are_noisified_when_noisy_features_on(self):
    env = college_admission.CollegeAdmissionsEnv(user_params={
        'gaming': False,
        'noisy_features': True
    })
    agent = random_agents.RandomAgent(
        env.action_space,
        None,
        env.observation_space,
        default_action={
            'threshold': np.array(0.8),
            'epsilon_prob': np.array(0)
        })
    action = agent.initial_action()
    env.step(action)
    self.assertFalse((np.array(env.state.original_test_scores) -
                      np.array(env.state.test_scores_x) == 0).all())

  def test_unmanipualted_features_not_noisified_when_noisy_features_off(self):
    env = college_admission.CollegeAdmissionsEnv(user_params={
        'gaming': False,
        'noisy_features': False
    })
    agent = random_agents.RandomAgent(
        env.action_space,
        None,
        env.observation_space,
        default_action={
            'threshold': np.array(0.8),
            'epsilon_prob': np.array(0)
        })
    action = agent.initial_action()
    env.step(action)
    self.assertTrue((np.array(env.state.original_test_scores) -
                     np.array(env.state.test_scores_x) == 0).all())

  def test_invalid_gaming_control_raises_error(self):
    with self.assertRaises(ValueError):
      college_admission.CollegeAdmissionsEnv(user_params={'gaming_control': 2})

  def test_invalid_noise_dist_raises_error(self):
    with self.assertRaises(ValueError):
      college_admission.CollegeAdmissionsEnv(
          user_params={'noise_dist': 'random'})


if __name__ == '__main__':
  absltest.main()
