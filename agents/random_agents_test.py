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

# Lint as: python3
"""Tests for fairness_gym.agents.random_agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import test_util
from agents import random_agents


class RandomAgentsTest(absltest.TestCase):

  def test_can_run_with_env(self):
    env = test_util.DummyEnv()
    agent = random_agents.RandomAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=None)
    test_util.run_test_simulation(env=env, agent=agent)


if __name__ == '__main__':
  absltest.main()
