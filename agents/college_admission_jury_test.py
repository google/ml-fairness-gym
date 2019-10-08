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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import core
import params
import test_util
from agents import college_admission_jury
from environments import college_admission
import numpy as np


class FixedJuryTest(absltest.TestCase):

  def test_fixed_agent_simulation_runs_successfully(self):
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.FixedJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.7)
    test_util.run_test_simulation(env=env, agent=agent, stackelberg=True)

  def test_agent_raises_episode_done_error(self):
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.FixedJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.7)
    with self.assertRaises(core.EpisodeDoneError):
      agent.act(
          observation={
              'threshold': np.array(0.5),
              'epsilon_prob': np.array(0)
          },
          done=True)

  def test_agent_raises_invalid_observation_error(self):
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.FixedJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.7)
    with self.assertRaises(core.InvalidObservationError):
      agent.act(observation={0: 'Invalid Observation'}, done=False)

  def test_agent_produces_zero_no_epsilon_greedy(self):
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.FixedJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.7,
        epsilon_greedy=False)
    epsilon_probs = [agent.initial_action()['epsilon_prob'] for _ in range(10)]
    self.assertEqual(epsilon_probs, [0] * 10)

  def test_agent_produces_different_epsilon_with_epsilon_greedy(self):
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.FixedJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.7,
        epsilon_greedy=True)
    obs, _, done, _ = env.step(agent.initial_action())
    epsilon_probs = [float(agent.initial_action()['epsilon_prob'])]
    epsilon_probs.extend(
        [float(agent.act(obs, done)['epsilon_prob']) for _ in range(10)])
    self.assertGreater(len(set(epsilon_probs)), 1)

  def test_epsilon_prob_decays_as_expected(self):
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.FixedJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.7,
        epsilon_greedy=True,
        initial_epsilon_prob=0.3,
        decay_steps=5,
        epsilon_prob_decay_rate=0.001)
    obs, _, done, _ = env.step(agent.initial_action())
    epsilon_probs = [float(agent.initial_action()['epsilon_prob'])]
    epsilon_probs.extend(
        [float(agent.act(obs, done)['epsilon_prob']) for _ in range(2)])
    self.assertTrue(
        np.all(np.isclose(epsilon_probs, [0.3, 0.0753, 0.0189], atol=1e-2)))


class NaiveJuryTest(absltest.TestCase):

  def test_jury_successfully_initializes(self):
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.NaiveJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.7)
    self.assertEqual(agent.initial_action()['threshold'], 0.7)
    self.assertEqual(agent.initial_action()['epsilon_prob'], 0)

  def test_simple_classifier_simulation_runs_successfully(self):
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.NaiveJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.7)
    test_util.run_test_simulation(env=env, agent=agent, stackelberg=True)

  def test_get_default_features_returns_same_features(self):
    """Checks that the feature selection fn works as expected."""
    observations = {
        'test_scores_y': [0.2, 0.3, 0.4, 0.5, 0.6],
        'selected_ground_truth': [1, 0, 2, 1, 2],
        'selected_applicants': [1, 1, 0, 1, 0]
    }
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.NaiveJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.7)
    features = agent._get_default_features(observations)
    self.assertListEqual(features, [0.2, 0.3, 0.5])

  def test_label_fn_returns_correct_labels(self):
    """Checks that the label function works as expected."""
    observations = {
        'test_scores_y': [0.2, 0.3, 0.4, 0.5, 0.6],
        'selected_ground_truth': [1, 0, 2, 1, 2],
        'selected_applicants': [1, 1, 0, 1, 0]
    }
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.NaiveJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.7)
    labels = agent._label_fn(observations)
    self.assertListEqual(labels, [1, 0, 1])

  def test_agent_returns_same_threshold_till_burnin_and_then_change(self):
    """Tests that agent returns same threshold till burnin without freezing."""
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.NaiveJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.3,
        burnin=4,
        freeze_classifier_after_burnin=False)
    test_util.run_test_simulation(
        env=env, agent=agent, num_steps=10, stackelberg=True)
    actions = [float(action['threshold']) for _, action in env.history]
    self.assertEqual(set(actions[:4]), {0.3})
    self.assertGreater(len(set(actions)), 4)

  def test_agent_returns_same_threshold_till_burnin_learns_and_freezes(self):
    """Tests that agent returns same threshold till burnin and freezes after."""
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.NaiveJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0.3,
        burnin=4,
        freeze_classifier_after_burnin=True)
    test_util.run_test_simulation(
        env=env, agent=agent, num_steps=10, stackelberg=True)
    actions = [float(action['threshold']) for _, action in env.history]
    self.assertEqual(set(actions[:4]), {0.3})
    self.assertLen(set(actions), 3)

  def test_agent_returns_correct_threshold(self):
    env = college_admission.CollegeAdmissionsEnv(
        user_params={
            'gaming':
                False,
            'subsidize':
                False,
            'noise_params':
                params.BoundedGaussian(max=0.3, min=0, sigma=0, mu=0.1),
            'feature_params': params.GMM(mix_weight=[0.5, 0.5], mu=[0.5, 0.5],
                                         sigma=[0.1, 0.1])
        })
    agent = college_admission_jury.NaiveJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        threshold=0,
        burnin=9,
        freeze_classifier_after_burnin=True)
    test_util.run_test_simulation(
        env=env, agent=agent, num_steps=10, stackelberg=True)
    learned_threshold = env.history[-1].action['threshold']
    self.assertTrue(np.isclose(learned_threshold, 0.55, atol=1e-2))


class RobustJuryTest(absltest.TestCase):

  def test_robust_classifier_simulation_runs_successfully(self):
    env = college_admission.CollegeAdmissionsEnv()
    agent = college_admission_jury.RobustJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        group_cost=env.initial_params.group_cost,
        burnin=10)
    test_util.run_test_simulation(env=env, agent=agent, stackelberg=True)

  def test_correct_max_score_change_calculated_no_subsidy(self):
    """Tests that the max gaming steps gives output as expected."""
    env = college_admission.CollegeAdmissionsEnv(
        user_params={
            'group_cost': {
                0: 2,
                1: 4
            },
            'subsidize': False,
            'subsidy_beta': 0.6,
            'gaming_control': np.inf
        })
    agent = college_admission_jury.RobustJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        group_cost=env.initial_params.group_cost,
        subsidize=env.initial_params.subsidize,
        subsidy_beta=env.initial_params.subsidy_beta,
        gaming_control=env.initial_params.gaming_control)
    obs, _, _, _ = env.step(agent.initial_action())
    max_change = agent._get_max_allowed_score_change(obs)
    self.assertEqual(max_change, [0.5, 0.25])

  def test_correct_max_score_change_calculated_with_subsidy(self):
    """Tests that the max gaming steps gives output as expected."""
    env = college_admission.CollegeAdmissionsEnv(
        user_params={
            'group_cost': {
                0: 2,
                1: 4
            },
            'subsidize': True,
            'subsidy_beta': 0.8,
            'gaming_control': np.inf
        })
    agent = college_admission_jury.RobustJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        group_cost=env.initial_params.group_cost,
        subsidize=env.initial_params.subsidize,
        subsidy_beta=env.initial_params.subsidy_beta,
        gaming_control=env.initial_params.gaming_control)
    obs, _, _, _ = env.step(agent.initial_action())
    max_change = agent._get_max_allowed_score_change(obs)
    self.assertEqual(max_change, [0.5, 0.3125])

  def test_correct_robust_threshold_returned(self):
    env = college_admission.CollegeAdmissionsEnv()

    agent = college_admission_jury.RobustJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        group_cost=env.initial_params.group_cost)
    agent._features = [0.1, 0.2, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8]
    agent._labels = [0, 0, 1, 0, 0, 1, 1, 1]
    agent._train_model()
    self.assertEqual(agent._threshold, 0.6)

  def test_features_manipulated_to_maximum_limit_with_no_control(self):
    env = college_admission.CollegeAdmissionsEnv(user_params={
        'num_applicants': 5,
        'gaming_control': np.inf,
        'group_cost': {
            0: 2,
            1: 4
        }
    })
    agent = college_admission_jury.RobustJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        group_cost=env.initial_params.group_cost)
    observations = {
        'test_scores_y': np.asarray([0.2, 0.3, 0.4, 0.5, 0.4]),
        'selected_applicants': np.asarray([0, 1, 0, 1, 1]),
        'selected_ground_truth': np.asarray([2, 0, 2, 1, 1]),
        'applicant_groups': np.asarray([0, 1, 1, 0, 1])
    }
    agent.act(observations, done=False)
    self.assertTrue(
        np.all(
            np.isclose(
                agent._get_maximum_manipulated_features(observations),
                [0.55, 1.0, 0.65],
                atol=1e-4)))
    self.assertEqual(agent._features,
                     agent._get_maximum_manipulated_features(observations))

  def test_features_manipulated_to_maximum_limit_with_gaming_control(self):
    env = college_admission.CollegeAdmissionsEnv(user_params={
        'num_applicants': 5,
        'gaming_control': 0.3,
        'group_cost': {
            0: 2,
            1: 4,
        }
    })
    agent = college_admission_jury.RobustJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        group_cost=env.initial_params.group_cost,
        gaming_control=env.initial_params.gaming_control)
    observations = {
        'test_scores_y': np.asarray([0.2, 0.3, 0.4, 0.5, 0.4]),
        'selected_applicants': np.asarray([0, 1, 0, 1, 1]),
        'selected_ground_truth': np.asarray([2, 0, 2, 1, 1]),
        'applicant_groups': np.asarray([0, 1, 1, 0, 1])
    }
    self.assertTrue(
        np.all(
            np.isclose(
                agent._get_maximum_manipulated_features(observations),
                [0.55, 0.8, 0.65],
                atol=1e-4)))

  def test_features_manipulated_to_maximum_limit_with_control_epsilon_greedy(
      self):
    env = college_admission.CollegeAdmissionsEnv(user_params={
        'num_applicants': 5,
        'gaming_control': 0.3,
        'group_cost': {
            0: 2,
            1: 4,
        }
    })
    agent = college_admission_jury.RobustJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        group_cost=env.initial_params.group_cost,
        gaming_control=env.initial_params.gaming_control,
        epsilon_greedy=True,
        initial_epsilon_prob=0.2)
    observations = {
        'test_scores_y': np.asarray([0.2, 0.3, 0.4, 0.5, 0.4]),
        'selected_applicants': np.asarray([0, 1, 0, 1, 1]),
        'selected_ground_truth': np.asarray([2, 0, 2, 1, 1]),
        'applicant_groups': np.asarray([0, 1, 1, 0, 1])
    }
    self.assertTrue(
        np.all(
            np.isclose(
                agent._get_maximum_manipulated_features(observations),
                [0.5, 0.8, 0.6],
                atol=1e-4)))

  def test_features_manipulated_to_maximum_limit_no_control_epsilon_greedy(
      self):
    env = college_admission.CollegeAdmissionsEnv(user_params={
        'num_applicants': 5,
        'gaming_control': np.inf,
        'group_cost': {
            0: 2,
            1: 4,
        }
    })
    agent = college_admission_jury.RobustJury(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=(lambda x: 0),
        group_cost=env.initial_params.group_cost,
        gaming_control=env.initial_params.gaming_control,
        epsilon_greedy=True,
        initial_epsilon_prob=0.2)
    observations = {
        'test_scores_y': np.asarray([0.2, 0.3, 0.4, 0.5, 0.4]),
        'selected_applicants': np.asarray([0, 1, 0, 1, 1]),
        'selected_ground_truth': np.asarray([2, 0, 2, 1, 1]),
        'applicant_groups': np.asarray([0, 1, 1, 0, 1])
    }
    self.assertTrue(
        np.all(
            np.isclose(
                agent._get_maximum_manipulated_features(observations),
                [0.5, 0.9, 0.6],
                atol=1e-4)))

  def test_assertion_raised_when_burnin_less_than_2(self):
    env = college_admission.CollegeAdmissionsEnv()

    with self.assertRaises(ValueError):
      college_admission_jury.RobustJury(
          action_space=env.action_space,
          observation_space=env.observation_space,
          reward_fn=(lambda x: 0),
          group_cost=env.initial_params.group_cost,
          burnin=1)


if __name__ == '__main__':
  absltest.main()
