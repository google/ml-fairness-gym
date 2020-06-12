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
"""Tests for safe_rl_recs.rnn_agent."""
from absl import flags
from absl.testing import absltest
import.agents.recommenders.rnn_agent as rnn_agent
import.agents.recommenders.rnn_cvar_agent as rnn_cvar_agent
import.environments.recommenders.restaurant_toy_recsim as restaurant_toy_recsim

FLAGS = flags.FLAGS


def get_test_env():
  """Returns an environment with pre-defined user dynamics."""
  config = restaurant_toy_recsim.EnvConfig()
  env = restaurant_toy_recsim.build_restaurant_recs_recsim_env(config)
  env.reset()
  return env


class RNNAgentTest(absltest.TestCase):
  """Tests for RNN agents."""

  def setUp(self):
    super(RNNAgentTest, self).setUp()
    self.env = get_test_env()
    self.agent = rnn_agent.RNNAgent(self.env.observation_space,
                                    self.env.action_space,
                                    max_episode_length=5)

  def test_forward_pass(self):
    """Tests whether the forward pass works."""
    # collect data does a bunch of forward passes
    self.agent.simulate(self.env, 3, eval_mode=True)

  def test_backward_pass(self):
    self.agent.simulate(self.env, 3, eval_mode=False)

  def test_eval_mode(self):
    self.agent.simulate(self.env, 3, eval_mode=True)
    # Right now the test is doing nothing specific to check if the eval_mode
    # flag is indeed effective.

  def test_deterministic_forward_pass(self):
    """Tests whether the deterministic=True mode for the RNNAgent works."""
    self.agent.empty_buffer()
    self.agent.simulate(self.env, 2, eval_mode=True, deterministic=True)
    rec_history = self.agent.replay_buffer['recommendation_seqs']
    # Check whether the first recommendation made by the agent is the same
    # for two user trajectories. After t=1, the trajectories may be different
    # because the users might transition states according the environment.
    self.assertEqual(rec_history[0][1], rec_history[1][1])

  def test_recommendation_in_action_space(self):
    self.env.reset()
    observation = self.env.observation_space.sample()
    slate = self.agent.step(0, observation)
    self.assertIn(slate, self.env.action_space)


class SafeRNNAgentTest(RNNAgentTest):
  """Tests for SafeRNN agents."""

  def setUp(self):
    """Inherits all the setup and tests from RNNAgentTest."""
    super(SafeRNNAgentTest, self).setUp()
    self.agent = rnn_cvar_agent.SafeRNNAgent(
        self.env.observation_space,
        self.env.action_space,
        max_episode_length=5)

  def test_deterministic_forward_pass(self):
    """Tests whether the deterministic=True mode for the SafeRNNAgent works."""
    self.agent.empty_buffer()
    self.agent.simulate(self.env, 2, eval_mode=True, deterministic=True)
    rec_history = self.agent.replay_buffer['recommendation_seqs']
    self.assertEqual(rec_history[0][1], rec_history[1][1])

  def test_recommendation_in_action_space(self):
    self.env.reset()
    observation = self.env.observation_space.sample()
    slate = self.agent.step(0, observation)
    self.assertIn(slate, self.env.action_space)

  def test_multiple_episodes_per_step_works(self):
    self.agent.empty_buffer()
    self.agent.simulate(self.env, 2, eval_mode=True, deterministic=True)
    self.agent.model_update()


if __name__ == '__main__':
  absltest.main()
