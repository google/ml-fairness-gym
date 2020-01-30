# coding=utf-8
# Copyright 2020 The ML Fairness Gym Authors.
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
"""Tests for attention_allocation environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import test_util
from agents import random_agents
from environments import attention_allocation
import numpy as np
from six.moves import range


class LocationAllocationTest(absltest.TestCase):

  def test_sample_incidents_centered_on_incident_rates(self):
    """Check sample_incidents return follows expected distribution.

    This test verifies the incident rates across locations returned by
    _sample_incidents are centered on the underling incident_rates.
    """
    n_trials = 100
    rng = np.random.RandomState()
    rng.seed(0)
    params = attention_allocation.Params()
    # _sample_incidents returns occurred incidents and reported incidents.
    # They are generated identically so we are testing on ocurred incidents,
    # which is at index 0.
    samples = [
        attention_allocation._sample_incidents(rng, params)[0]
        for _ in range(n_trials)
    ]
    means = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    errors_in_tolerance = [(np.abs(means[i] - params.incident_rates[i]) <
                            (std[i] / 3.0)) for i in range(params.n_locations)]
    self.assertTrue(np.all(errors_in_tolerance))

  def test_update_state(self):
    """Check that state is correctly updated with incidents_seen.

    This tests checks that numbers of incidents_seen are no more than the
    incidents  generated and the attention deployed as specified in the action,
    if allocating without attention replacement.
    """
    env = attention_allocation.LocationAllocationEnv()
    env.seed(0)
    agent = random_agents.RandomAgent(env.action_space, None,
                                      env.observation_space)
    observation = env.reset()
    action = agent.act(observation, False)
    crimes, reported_incidents = attention_allocation._sample_incidents(
        env.state.rng, env.state.params)
    attention_allocation._update_state(env.state, crimes, reported_incidents,
                                       action)
    incidents_seen = env.state.incidents_seen
    self.assertTrue((incidents_seen <= crimes).all())
    if not env.state.params.attention_replacement:
      self.assertTrue((incidents_seen <= action).all())

  def test_parties_can_interact(self):
    test_util.run_test_simulation(
        env=attention_allocation.LocationAllocationEnv())

  def test_dynamic_rate_change(self):
    params = attention_allocation.Params()
    params.dynamic_rate = 0.1
    params.incident_rates = [4.0, 2.0]
    params.n_attention_units = 2
    env = attention_allocation.LocationAllocationEnv(params=params)
    env.seed(0)
    env.step(action=np.array([2, 0]))
    new_rates = env.state.params.incident_rates
    expected_rates = [3.8, 2.1]
    self.assertEqual(expected_rates, new_rates)

  def test_features_centered_correctly_no_noise(self):
    params = attention_allocation.Params()
    params.n_locations = 3
    params.prior_incident_counts = (500, 500, 500)
    params.incident_rates = [1., 1., 1.]
    params.miss_incident_prob = (0.2, 0.2, 0.2)
    params.extra_incident_prob = (0., 0., 0.)
    params.feature_covariances = [[0, 0], [0, 0]]
    rng = np.random.RandomState()
    rng.seed(0)

    expected_features_shape = (params.n_locations, len(params.feature_means))
    features = attention_allocation._get_location_features(
        params, rng, [1, 1, 1])
    self.assertEqual(features.shape, expected_features_shape)
    expected_feautres = np.array([[1., 2.], [1., 2.], [1., 2.]])
    self.assertTrue(np.array_equal(expected_feautres, features))

  def test_features_centered_correctly(self):
    params = attention_allocation.Params()
    params.n_locations = 3
    params.prior_incident_counts = (500, 500, 500)
    params.incident_rates = [1., 1., 1.]
    params.miss_incident_prob = (0.2, 0.2, 0.2)
    params.extra_incident_prob = (0., 0., 0.)
    rng = np.random.RandomState()
    rng.seed(0)

    n_samples = 200
    expected_features_shape = (params.n_locations, len(params.feature_means))
    features = np.zeros(expected_features_shape)
    for _ in range(n_samples):
      new_features = attention_allocation._get_location_features(
          params, rng, [1, 1, 1])
      self.assertEqual(new_features.shape, expected_features_shape)
      features = features + new_features

    expected_feature_average = np.array([[1., 2.], [1., 2.], [1., 2.]])
    self.assertTrue(
        np.all(
            np.isclose(
                features / n_samples, expected_feature_average, atol=0.1)))


if __name__ == '__main__':
  absltest.main()
