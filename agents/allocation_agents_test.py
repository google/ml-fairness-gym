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
"""Tests for naive_probability_matching_allocator.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import core
import rewards
import test_util
from agents import allocation_agents
from environments import attention_allocation
import gym
import numpy as np
from six.moves import range


class NaiveProbabilityMatchingAgentTest(absltest.TestCase):

  def test_update_counts(self):
    """Check that counts are updated correctly given an observation."""
    env = attention_allocation.LocationAllocationEnv()
    agent_params = allocation_agents.NaiveProbabilityMatchingAgentParams()
    agent_params.decay_prob = 0
    agent = allocation_agents.NaiveProbabilityMatchingAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=None,
        params=agent_params)
    counts = [3, 6, 8]
    observation = np.array([1, 2, 0])
    updated_counts = agent._update_beliefs(observation, counts)
    self.assertTrue(np.all(np.equal(updated_counts, [4, 8, 8])))

  def test__allocate_by_counts(self):
    """Check allocation proportions match probabilities from counts."""
    env = attention_allocation.LocationAllocationEnv()
    agent = allocation_agents.NaiveProbabilityMatchingAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=None)
    counts = [3, 6, 8]
    n_resource = 20
    n_samples = 100
    samples = [agent._allocate(n_resource, counts) for _ in range(n_samples)]
    counts_normalized = [(count / float(np.sum(counts))) for count in counts]
    samples_normalized = [
        (count / float(np.sum(samples))) for count in np.sum(samples, axis=0)
    ]
    self.assertTrue(
        np.all(np.isclose(counts_normalized, samples_normalized, atol=0.05)))

  def test_allocate_by_counts_zero(self):
    """Check allocations are even when counts are zero."""
    env = attention_allocation.LocationAllocationEnv()
    agent = allocation_agents.NaiveProbabilityMatchingAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=None)
    counts = [0, 0, 0]
    n_resource = 15
    n_samples = 100
    samples = [agent._allocate(n_resource, counts) for _ in range(n_samples)]
    mean_samples = np.sum(samples, axis=0) / float(n_samples)
    expected_mean = n_resource / float(len(counts))
    std_dev = np.std(samples)
    means_close = [
        np.abs(mean - expected_mean) < std_dev for mean in mean_samples
    ]
    self.assertTrue(np.all(means_close))

  def test_can_interact_with_attention_env(self):
    env = attention_allocation.LocationAllocationEnv()
    agent = allocation_agents.NaiveProbabilityMatchingAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=None)
    test_util.run_test_simulation(env=env, agent=agent)

  def test_get_added_vector_features(self):
    action_space_len = 2
    observation = {
        'incidents_seen': np.array([0, 1]),
        'incidents_reported': np.array([3, 1])
    }
    features = allocation_agents._get_added_vector_features(
        observation, action_space_len)
    expected = [3.0, 2.0]
    self.assertSequenceAlmostEqual(features.tolist(), expected)
    features = allocation_agents._get_added_vector_features(
        observation, action_space_len, keys=['incidents_reported'])
    expected = [3.0, 1.0]
    self.assertSequenceAlmostEqual(features.tolist(), expected)

  def test_episode_done_raises_error(self):
    env = attention_allocation.LocationAllocationEnv()
    agent = allocation_agents.NaiveProbabilityMatchingAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=None)
    observation = env.reset()
    with self.assertRaises(core.EpisodeDoneError):
      agent.act(observation, done=True)


class MLEProbabilityMatchingAgentTest(absltest.TestCase):

  def test_can_interact_with_attention_env(self):
    env = attention_allocation.LocationAllocationEnv()
    agent = allocation_agents.MLEProbabilityMatchingAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=None,
        params=None)
    test_util.run_test_simulation(env=env, agent=agent)

  def test_MLE_rate_estimation(self):
    env_params = attention_allocation.Params()
    env_params.prior_incident_counts = (500, 500)
    env_params.n_attention_units = 5

    # pylint: disable=g-long-lambda
    agent_params = allocation_agents.MLEProbabilityMatchingAgentParams()

    agent_params.feature_selection_fn = lambda obs: allocation_agents._get_added_vector_features(
        obs, env_params.n_locations, keys=['incidents_seen'])
    agent_params.interval = 200
    agent_params.epsilon = 0

    env = attention_allocation.LocationAllocationEnv(env_params)
    agent = allocation_agents.MLEProbabilityMatchingAgent(
        action_space=env.action_space,
        reward_fn=lambda x: None,
        observation_space=env.observation_space,
        params=agent_params)
    seed = 0
    agent.rng.seed(seed)
    env.seed(seed)
    observation = env.reset()
    done = False
    steps = 200
    for _ in range(steps):
      action = agent.act(observation, done)
      observation, _, done, _ = env.step(action)

    self.assertTrue(
        np.all(
            np.isclose(
                list(agent.beliefs), list(env_params.incident_rates),
                atol=0.5)))


class MLEGreedyAgentTest(absltest.TestCase):

  def test_can_interact_with_attention_env(self):
    env = attention_allocation.LocationAllocationEnv()
    agent = allocation_agents.MLEGreedyAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=None)
    test_util.run_test_simulation(env=env, agent=agent)

  def test_allocate_beliefs_fair_unsatisfiable(self):
    env_params = attention_allocation.Params(
        n_locations=4,
        prior_incident_counts=(10, 10, 10, 10),
        n_attention_units=5,
        incident_rates=[0, 0, 0, 0])
    env = attention_allocation.LocationAllocationEnv(params=env_params)
    agent_params = allocation_agents.MLEGreedyAgentParams(
        epsilon=0.0, alpha=0.25)
    agent = allocation_agents.MLEGreedyAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.VectorSumReward('incidents_seen'),
        params=agent_params)
    with self.assertRaises(gym.error.InvalidAction):
      agent._allocate(5, [5, 2, 1, 1])

  def test_allocate_beliefs_fair(self):
    env_params = attention_allocation.Params(
        n_locations=4,
        prior_incident_counts=(10, 10, 10, 10),
        n_attention_units=6,
        incident_rates=[0, 0, 0, 0])
    env = attention_allocation.LocationAllocationEnv(params=env_params)
    agent_params = allocation_agents.MLEGreedyAgentParams(
        epsilon=0.0, alpha=0.25)
    agent = allocation_agents.MLEGreedyAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.VectorSumReward('incidents_seen'),
        params=agent_params)
    allocation = agent._allocate(6, [5, 2, 1, 1])
    self.assertTrue(np.all(np.equal(allocation, [3, 1, 1, 1])))

  def test_allocate_beliefs_greedy(self):
    env_params = attention_allocation.Params(
        n_locations=4,
        prior_incident_counts=(10, 10, 10, 10),
        n_attention_units=5,
        incident_rates=[0, 0, 0, 0])
    env = attention_allocation.LocationAllocationEnv(params=env_params)
    agent_params = allocation_agents.MLEGreedyAgentParams(epsilon=0.0)
    agent = allocation_agents.MLEGreedyAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.VectorSumReward('incidents_seen'),
        params=agent_params)
    allocation = agent._allocate(5, [5, 2, 1, 1])
    self.assertTrue(np.all(np.equal(allocation, [4, 1, 0, 0])))


if __name__ == '__main__':
  absltest.main()
