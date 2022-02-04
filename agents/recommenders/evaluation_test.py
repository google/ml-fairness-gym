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

# Lint as: python3
"""Tests for evaluation.py."""

import os
import tempfile
import types
from absl import flags
from absl.testing import absltest
import file_util
from agents.recommenders import evaluation
from agents.recommenders import rnn_cvar_agent
from environments.recommenders import restaurant_toy_recsim
import numpy as np


FLAGS = flags.FLAGS


def get_test_env(user_config):
  """Returns an environment with pre-defined user dynamics."""
  config = restaurant_toy_recsim.EnvConfig(user_config=user_config)
  env = restaurant_toy_recsim.build_restaurant_recs_recsim_env(config)
  env.reset()
  return env


class EvaluationTest(absltest.TestCase):

  def setUp(self):
    super(EvaluationTest, self).setUp()
    transition_matrix = restaurant_toy_recsim.TransitionMatrix.RandomMatrix(
        num_states=1, num_actions=2)
    self.mdp_config = restaurant_toy_recsim.UserConfig(
        user_states_names=['Neutral'],
        state_transition_matrix=transition_matrix,
        reward_matrix=np.array([[1.0, 0.0]]))
    self.config = {
        'max_episode_length': 20,
        'initial_lambda': 10.0,
        'beta': 0.5,
        'alpha': 0.95,
        'lambda_learning_rate': 0.0,
        'var_learning_rate': 0.01,
        'learning_rate': None,
        'num_users_eval': 1,
        'num_users_eval_final': 100,
        'num_episodes_per_update': 1,
        'optimizer_name': 'Adam',
        'num_updates': 1000,
        'mdp': self.mdp_config,
        'eval_deterministic': False
    }
    self.config = types.SimpleNamespace(**self.config)
    self.env = get_test_env(self.mdp_config)
    self.agent = rnn_cvar_agent.SafeRNNAgent(
        self.env.observation_space,
        self.env.action_space,
        max_episode_length=self.config.max_episode_length)

  def test_reward_health_metrics_in_range(self):
    """Tests whether the reward and health metric are within range."""
    results = evaluation.evaluate_agent(
        self.agent, self.env, self.config.alpha, num_users=10,
        risk_score_extractor=evaluation.health_risk)
    self.assertBetween(results['rewards'],
                       np.min(self.mdp_config.reward_matrix),
                       np.max(self.mdp_config.reward_matrix))
    self.assertBetween(results['health'], 0, 1)
    self.assertBetween(results['var'], 0, 1)

    # Since cvar is the mean of the values beyond var, cvar>= var
    self.assertBetween(results['cvar'], results['var'], 1)

  def test_plotting_works(self):
    """Tests whether the plotting feature in the evaluation function works."""
    with tempfile.TemporaryDirectory(dir=FLAGS.test_tmpdir) as tmpdirname:
      figure_file_path = os.path.join(tmpdirname, 'test_plot.png')
      with file_util.open(figure_file_path, 'wb') as figure_file_obj:
        evaluation.evaluate_agent(
            self.agent,
            self.env,
            alpha=self.config.alpha,
            num_users=10,
            scatter_plot_trajectories=True,
            figure_file_obj=figure_file_obj,
            risk_score_extractor=evaluation.health_risk)
      filecontents = file_util.open(figure_file_path, 'rb').read()
      self.assertNotEmpty(filecontents)


if __name__ == '__main__':
  absltest.main()
