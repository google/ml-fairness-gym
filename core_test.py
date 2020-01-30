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
"""Tests for fairness_gym.core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from absl.testing import absltest
from absl.testing import parameterized
import attr
import core
import test_util
from agents import random_agents
from environments import attention_allocation
import gym
import numpy as np
from six.moves import range


@attr.s
class CoreTestParams(core.Params):
  a = attr.ib(default=1)
  b = attr.ib(default=2)
  c = attr.ib(default=3)


# This class defines a state that is compatible with DummyEnv in test_util.py
@attr.s(cmp=False)
class CoreTestState(core.State):
  x = attr.ib(default=0.)
  params = attr.ib(default=None)
  rng = attr.ib(factory=np.random.RandomState)


class CoreApiTest(parameterized.TestCase):

  def test_interactions(self):
    # With no arguments tests dummy implementations defined in test_util.
    test_util.run_test_simulation()

  def test_invalid_env_interactions(self):
    env = test_util.DummyEnv()
    with self.assertRaises(gym.error.InvalidAction):
      env.step('not a real action')

    # Succeeds.
    env.step(0)

  def test_metric_multiple(self):
    env = attention_allocation.LocationAllocationEnv()
    agent = random_agents.RandomAgent(env.action_space, None,
                                      env.observation_space)

    env.seed(100)
    observation = env.reset()
    done = False

    for _ in range(2):
      action = agent.act(observation, done)
      observation, _, done, _ = env.step(action)

    metric1 = core.Metric(env)
    metric2 = core.Metric(env)

    history1 = metric1._extract_history(env)
    history2 = metric2._extract_history(env)
    self.assertEqual(history1, history2)

  def test_episode_done_raises_error(self):
    env = test_util.DummyEnv()
    agent = random_agents.RandomAgent(env.action_space, None,
                                      env.observation_space)
    obs = env.reset()
    with self.assertRaises(core.EpisodeDoneError):
      agent.act(obs, done=True)

  def test_metric_realigns_history(self):
    env = test_util.DummyEnv()
    agent = random_agents.RandomAgent(env.action_space, None,
                                      env.observation_space)
    env.set_scalar_reward(agent.reward_fn)

    def realign_fn(history):
      return [(1, action) for _, action in history]

    metric = test_util.DummyMetric(env, realign_fn=realign_fn)
    _ = test_util.run_test_simulation(env, agent, metric)
    history = metric._extract_history(env)
    self.assertCountEqual([1] * 10, [state for state, _ in history])

  def test_state_deepcopy_maintains_equality(self):
    state = CoreTestState(x=0., params=None, rng=np.random.RandomState())
    copied_state = copy.deepcopy(state)
    self.assertIsInstance(copied_state, CoreTestState)
    self.assertEqual(state, copied_state)

  def test_state_with_nested_numpy_serializes(self):

    @attr.s
    class _TestState(core.State):
      x = attr.ib()

    state = _TestState(x={'a': np.zeros(2, dtype=int)})
    self.assertEqual(state.to_json(), '{"x": {"a": [0, 0]}}')

  def test_base_state_updater_raises(self):
    env = test_util.DummyEnv()
    state = env._get_state()
    with self.assertRaises(NotImplementedError):
      core.StateUpdater().update(state, env.action_space.sample())

  def test_noop_state_updater_does_nothing(self):
    env = test_util.DummyEnv()
    state = env._get_state()
    before = copy.deepcopy(state)
    core.NoUpdate().update(state, env.action_space.sample())
    self.assertEqual(state, before)

  def test_json_encode_function(self):

    def my_function(x):
      return x

    self.assertIn('my_function',
                  core.to_json({'params': {
                      'function': my_function
                  }}))

  def test_to_json_with_indent(self):
    self.assertNotIn('\n', core.to_json({'a': 5, 'b': [1, 2, 3]}))
    self.assertIn('\n', core.to_json({'a': 5, 'b': [1, 2, 3]}, indent=4))

if __name__ == '__main__':
  absltest.main()
