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
"""Tests for classifier_agents.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import absltest
import rewards
import test_util
from agents import classifier_agents
from agents import threshold_policies
from spaces import multinomial
import gym
import numpy as np


def _one_hot(idx, vecsize=10):
  vec = np.zeros(vecsize)
  vec[idx] = 1
  return vec


def _example_builder():
  return classifier_agents.TrainingExample(
      observation={'my_feature': 'a'},
      features='a',
      label=0,
      action=1,
      weight=3.14)


class TrainingCorpusTest(absltest.TestCase):

  def test_training_example_is_labeled_is_correct(self):
    example = _example_builder()

    self.assertTrue(example.is_labeled())
    example.label = None
    self.assertFalse(example.is_labeled())

  def test_filter_unlabeled(self):
    corpus = classifier_agents.TrainingCorpus()
    for idx in range(10):
      example = _example_builder()
      if idx % 2:
        example.label = None
      corpus.add(example)

    filtered_corpus = corpus.remove_unlabeled()

    # Check that the original corpus has not been filtered.
    self.assertGreater(len(corpus.examples), len(filtered_corpus.examples))

    # Check that all examples are labeled
    for example in filtered_corpus.examples:
      self.assertTrue(example.is_labeled())

  def test_get_weights(self):
    corpus = classifier_agents.TrainingCorpus()
    for idx in range(10):
      example = _example_builder()
      if idx % 2:
        example.label = None
      corpus.add(example)
    self.assertSameElements(corpus.get_weights(), {3.14})
    for weights in corpus.get_weights(stratify_by='my_feature').values():
      self.assertSameElements(weights, {3.14})


class ThresholdAgentTest(absltest.TestCase):

  def test_agent_raises_with_improper_number_of_features(self):
    env = test_util.DummyEnv()

    single_feature_params = classifier_agents.ScoringAgentParams(
        default_action_fn=env.action_space.sample, feature_keys=['x'])

    many_feature_params = classifier_agents.ScoringAgentParams(
        default_action_fn=env.action_space.sample, feature_keys=['x', 'y'])

    no_feature_params = classifier_agents.ScoringAgentParams(
        default_action_fn=env.action_space.sample, feature_keys=[])

    initialize = functools.partial(
        classifier_agents.ThresholdAgent,
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'))

    agent = initialize(params=single_feature_params)
    # This should succeed.
    agent.act(env.observation_space.sample(), done=False)

    agent = initialize(params=many_feature_params)
    with self.assertRaises(ValueError):
      agent.act(env.observation_space.sample(), done=False)

    agent = initialize(params=no_feature_params)
    with self.assertRaises(ValueError):
      agent.act(env.observation_space.sample(), done=False)

  def test_agent_trains(self):
    env = test_util.DummyEnv()
    params = classifier_agents.ScoringAgentParams(
        burnin=200,
        default_action_fn=env.action_space.sample,
        feature_keys=['x'])

    agent = classifier_agents.ThresholdAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params)

    # Train with points that are nearly separable but have some overlap between
    # 0.3 and 0.4.
    for observation in np.linspace(0, 0.4, 100):
      agent._act_impl({'x': np.array([observation])}, reward=0, done=False)

    for observation in np.linspace(0.3, 0.8, 100):
      agent._act_impl({'x': np.array([observation])}, reward=1, done=False)

    # Add a negative point at the top of the range so that the training labels
    # are not fit perfectly by a threshold.
    agent._act_impl({'x': np.array([0.9])}, reward=0, done=False)

    agent.frozen = True
    actions = [
        agent.act({'x': np.array([obs])}, done=False)
        for obs in np.linspace(0, 0.95, 100)
    ]

    # Assert some actions are 0 and some are 1.
    self.assertSameElements(actions, {0, 1})
    # Assert actions are sorted - i.e., 0s followed by 1s.
    self.assertSequenceEqual(actions, sorted(actions))

    self.assertGreater(agent.global_threshold, 0)
    self.assertFalse(agent.group_specific_thresholds)

  def test_agent_can_learn_different_thresholds(self):

    observation_space = gym.spaces.Dict({
        'x': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        'group': gym.spaces.Discrete(2)
    })

    params = classifier_agents.ScoringAgentParams(
        default_action_fn=lambda: 0,
        feature_keys=['x'],
        group_key='group',
        threshold_policy=threshold_policies.ThresholdPolicy.EQUALIZE_OPPORTUNITY
    )

    rng = np.random.RandomState(100)

    agent = classifier_agents.ThresholdAgent(
        observation_space=observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params,
        rng=rng)

    # Train over the whole range of observations. Expect slightly different
    # thresholds to be learned.
    for observation in rng.rand(100):
      for group in [0, 1]:
        agent._act_impl(
            {
                'x': np.array([observation]),
                'group': np.array([group])
            },
            reward=observation > 0.5 + 0.1 * group,
            done=False)

    agent.frozen = True

    actions = {}
    for group in [0, 1]:
      actions[group] = []
      for observation in np.linspace(0, 1, 1000):
        actions[group].append(
            agent.act({
                'x': np.array([observation]),
                'group': np.array([group])
            },
                      done=False))

    # The two groups are classified with different policies so they are not
    # exactly equal.
    self.assertNotEqual(actions[0], actions[1])
    self.assertLen(agent.group_specific_thresholds, 2)

  def test_one_hot_conversion(self):
    observation_space = gym.spaces.Dict({'x': multinomial.Multinomial(10, 1)})

    params = classifier_agents.ScoringAgentParams(
        default_action_fn=lambda: 0,
        feature_keys=['x'],
        convert_one_hot_to_integer=True,
        threshold_policy=threshold_policies.ThresholdPolicy.SINGLE_THRESHOLD)

    agent = classifier_agents.ThresholdAgent(
        observation_space=observation_space,
        reward_fn=rewards.NullReward(),
        params=params)

    self.assertEqual(agent._get_features({'x': _one_hot(5)}), [5])

  def test_agent_on_one_hot_vectors(self):

    # Space of 1-hot vectors of length 10.
    observation_space = gym.spaces.Dict({'x': multinomial.Multinomial(10, 1)})

    params = classifier_agents.ScoringAgentParams(
        default_action_fn=lambda: 0,
        feature_keys=['x'],
        convert_one_hot_to_integer=True,
        burnin=999,
        threshold_policy=threshold_policies.ThresholdPolicy.SINGLE_THRESHOLD)

    agent = classifier_agents.ThresholdAgent(
        observation_space=observation_space,
        reward_fn=rewards.NullReward(),
        params=params)

    observation_space.seed(100)
    # Train a boundary at 3 using 1-hot vectors.
    observation = observation_space.sample()
    agent._act_impl(observation, reward=None, done=False)
    for _ in range(1000):
      last_observation = observation
      observation = observation_space.sample()
      agent._act_impl(
          observation,
          reward=int(np.argmax(last_observation['x']) >= 3),
          done=False)
      if agent._training_corpus.examples:
        assert int(agent._training_corpus.examples[-1].features[0] >= 3
                  ) == agent._training_corpus.examples[-1].label

    agent.frozen = True

    self.assertTrue(agent.act({'x': _one_hot(3)}, done=False))
    self.assertFalse(agent.act({'x': _one_hot(2)}, done=False))

  def test_frozen_classifier_never_trains(self):
    env = test_util.DummyEnv()
    params = classifier_agents.ScoringAgentParams(
        burnin=0, default_action_fn=env.action_space.sample, feature_keys=['x'])

    agent = classifier_agents.ThresholdAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params,
        frozen=True)
    # Initialize global_threshold with a distinctive value.
    agent.global_threshold = 0.123

    # Run for some number of steps, global_threshold should not change.
    for _ in range(10):
      agent.act(env.observation_space.sample(), False)
    self.assertEqual(agent.global_threshold, 0.123)

  def test_threshold_history_is_recorded(self):
    observation_space = gym.spaces.Dict({
        'x': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        'group': gym.spaces.MultiDiscrete([1])
    })
    observation_space.seed(100)

    params = classifier_agents.ScoringAgentParams(
        default_action_fn=lambda: 0,
        feature_keys=['x'],
        group_key='group',
        burnin=0,
        threshold_policy=threshold_policies.ThresholdPolicy.EQUALIZE_OPPORTUNITY
    )

    agent = classifier_agents.ThresholdAgent(
        observation_space=observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params)

    for _ in range(10):
      agent.act(observation_space.sample(), False)

    self.assertLen(agent.global_threshold_history, 10)
    self.assertTrue(agent.group_specific_threshold_history)
    for _, history in agent.group_specific_threshold_history.items():
      # Takes 2 extra steps (one to observe features and one to observe label)
      # before any learned group-specific threshold is available.
      self.assertLen(history, 8)

  def test_freeze_after_burnin(self):
    env = test_util.DummyEnv()
    burnin = 10
    params = classifier_agents.ScoringAgentParams(
        burnin=burnin,
        freeze_classifier_after_burnin=True,
        default_action_fn=env.action_space.sample,
        feature_keys=['x'])

    agent = classifier_agents.ThresholdAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params)

    for _ in range(burnin + 1):
      self.assertFalse(agent.frozen)
      _ = agent.act(env.observation_space.sample(), False)

    self.assertTrue(agent.frozen)
    self.assertTrue(agent.global_threshold)  # Agent has learned something.

  def test_skip_retraining_fn(self):
    env = test_util.DummyEnv()
    burnin = 10

    def _skip_retraining(action, observation):
      """Always skip retraining."""
      del action, observation
      return True

    params = classifier_agents.ScoringAgentParams(
        burnin=burnin,
        freeze_classifier_after_burnin=False,
        default_action_fn=env.action_space.sample,
        feature_keys=['x'],
        skip_retraining_fn=_skip_retraining)

    agent = classifier_agents.ThresholdAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params)

    for _ in range(burnin + 1):
      self.assertFalse(agent.frozen)
      _ = agent.act(env.observation_space.sample(), False)

    self.assertFalse(agent.frozen)  # Agent is not frozen.
    self.assertFalse(agent.global_threshold)  # Agent has not learned.

  def test_agent_seed(self):
    env = test_util.DummyEnv()

    params = classifier_agents.ScoringAgentParams(
        burnin=10,
        freeze_classifier_after_burnin=False,
        default_action_fn=env.action_space.sample,
        feature_keys=['x'])

    agent = classifier_agents.ThresholdAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params)

    agent.seed(100)
    a = agent.rng.randint(0, 1000)
    agent.seed(100)
    b = agent.rng.randint(0, 1000)
    self.assertEqual(a, b)

  def test_interact_with_env_replicable(self):
    env = test_util.DummyEnv()
    params = classifier_agents.ScoringAgentParams(
        burnin=10,
        freeze_classifier_after_burnin=False,
        default_action_fn=env.action_space.sample,
        feature_keys=['x'])

    agent = classifier_agents.ThresholdAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params)
    test_util.run_test_simulation(env=env, agent=agent)


class ClassifierAgentTest(absltest.TestCase):

  def test_agent_trains(self):
    env = test_util.DummyEnv()
    params = classifier_agents.ScoringAgentParams(
        default_action_fn=env.action_space.sample,
        feature_keys=['x'],
        burnin=200)

    agent = classifier_agents.ClassifierAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params)

    # Train with points that are nearly separable but have some overlap between
    # 0.3 and 0.4 with 1s in the lower region and 0s on the higher region.
    # A linear transform of x -> -x is expected to be learned so that a
    # threshold classifier can be successful.
    for observation in np.linspace(0, 0.4, 100):
      agent._act_impl({'x': np.array([observation])}, reward=1, done=False)

    for observation in np.linspace(0.3, 0.8, 100):
      agent._act_impl({'x': np.array([observation])}, reward=0, done=False)

    # Add a positive point at the top of the range so that the training labels
    # are not fit perfectly by a threshold.
    agent._act_impl({'x': np.array([0.9])}, reward=1, done=False)

    agent.frozen = True
    actions = [
        agent.act({'x': np.array([obs])}, done=False)
        for obs in np.linspace(0, 0.95, 100)
    ]

    # Assert some actions are 0 and some are 1.
    self.assertSameElements(actions, {0, 1})
    # Assert actions are reverse-sorted - i.e., 1s followed by 0s.
    self.assertSequenceEqual(actions, sorted(actions, reverse=True))

  def test_agent_trains_with_two_features(self):
    params = classifier_agents.ScoringAgentParams(
        default_action_fn=lambda: 0, feature_keys=['x', 'y'], burnin=200)

    agent = classifier_agents.ClassifierAgent(
        action_space=gym.spaces.Discrete(2),
        observation_space=gym.spaces.Dict({
            'x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=[1]),
            'y': gym.spaces.Box(low=-np.inf, high=np.inf, shape=[1])
        }),
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params)

    # Train with points that are nearly separable but have some overlap between
    # 0.3 and 0.4 with 1s in the lower region and 0s on the higher region.
    # A linear transform of x -> -x is expected to be learned so that a
    # threshold classifier can be successful.
    # `y` is the relevant feature. `x` is a constant.
    const = np.array([1])

    for observation in np.linspace(0, 0.4, 100):
      agent._act_impl({
          'y': np.array([observation]),
          'x': const
      },
                      reward=1,
                      done=False)

    for observation in np.linspace(0.3, 0.8, 100):
      agent._act_impl({
          'y': np.array([observation]),
          'x': const
      },
                      reward=0,
                      done=False)

    # Add a positive point at the top of the range so that the training labels
    # are not fit perfectly by a threshold.
    agent._act_impl({'y': np.array([0.9]), 'x': const}, reward=1, done=False)

    agent.frozen = True
    actions = []
    for obs in np.linspace(0, 0.95, 100):
      actions.append(agent.act({'y': np.array([obs]), 'x': const}, done=False))

    # Assert some actions are 0 and some are 1.
    self.assertSameElements(actions, {0, 1})
    # Assert actions are reverse-sorted - i.e., 1s followed by 0s.
    self.assertSequenceEqual(actions, sorted(actions, reverse=True))

  def test_insufficient_burnin_raises(self):
    env = test_util.DummyEnv()
    burnin = 5
    params = classifier_agents.ScoringAgentParams(
        default_action_fn=env.action_space.sample,
        feature_keys=['x'],
        burnin=burnin)

    agent = classifier_agents.ClassifierAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params)

    # Only give positive points to train.
    for _ in range(burnin):
      agent._act_impl(env.observation_space.sample(), reward=1, done=False)

    # Should raise a ValueError since the burnin has passed and the classifier
    # cannot train to make a decision.
    with self.assertRaises(ValueError):
      agent._act_impl(env.observation_space.sample(), reward=1, done=False)

  def test_interact_with_env_replicable(self):
    env = test_util.DummyEnv()
    params = classifier_agents.ScoringAgentParams(
        default_action_fn=env.action_space.sample, feature_keys=['x'], burnin=5)

    agent = classifier_agents.ClassifierAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('x'),
        params=params)
    test_util.run_test_simulation(env=env, agent=agent)


if __name__ == '__main__':
  absltest.main()
