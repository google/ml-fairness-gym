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
import test_util
from agents import random_agents
from environments import attention_allocation
from metrics import distribution_comparison_metrics
import numpy as np
from six.moves import range


class DistributionComparisonMetricsTest(absltest.TestCase):

  def test_measure(self):
    params = attention_allocation.Params()
    params.incident_rates = [4.0, 2.0]
    params.attention_replacement = True
    env = attention_allocation.LocationAllocationEnv(params)
    env = attention_allocation.LocationAllocationEnv(params)
    env.seed(100)
    env.action_space.seed(100)
    env.observation_space.seed(100)
    agent = random_agents.RandomAgent(env.action_space, None,
                                      env.observation_space)

    observation = env.reset()
    done = False
    for _ in range(250):
      action = agent.act(observation, done)
      observation, _, done, _ = env.step(action)

    metric = distribution_comparison_metrics.DistributionComparisonMetric(
        env, "incidents_seen", 250)
    state_dist, action_dist, distance = metric.measure(env)

    expected_state_dist = env.state.params.incident_rates / np.sum(
        env.state.params.incident_rates)
    # Expected action distribution is uniform because RandomAgent is random.
    expected_action_dist = [0.5, 0.5]
    expected_distance = np.linalg.norm(expected_state_dist -
                                       expected_action_dist)

    self.assertTrue(
        np.all(np.isclose(state_dist, expected_state_dist, atol=0.05)))
    self.assertTrue(
        np.all(np.isclose(action_dist, expected_action_dist, atol=0.05)))
    self.assertTrue(np.isclose(distance, expected_distance, atol=0.1))

  def test_error_on_scalar(self):
    """Test confirms an error is raised when an actions are scalars."""
    env = test_util.DummyEnv()
    env.seed(100)
    env.action_space.seed(100)
    env.observation_space.seed(100)
    agent = random_agents.RandomAgent(env.action_space, None,
                                      env.observation_space)

    observation = env.reset()
    done = False
    for _ in range(2):
      action = agent.act(observation, done)
      observation, _, done, _ = env.step(action)

    metric = distribution_comparison_metrics.DistributionComparisonMetric(
        env, "x", 100)
    with self.assertRaises(ValueError):
      metric.measure(env)


if __name__ == "__main__":
  absltest.main()
