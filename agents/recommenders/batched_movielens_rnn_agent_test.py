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
"""Tests for batched_movielens_rnn_agent."""

import os
import tempfile
from absl import flags
from absl.testing import absltest
import file_util
from agents.recommenders import batched_movielens_rnn_agent
from environments.recommenders import movie_lens_dynamic

FLAGS = flags.FLAGS


class MovielensRnnAgentTest(absltest.TestCase):

  def setUp(self):
    super(MovielensRnnAgentTest, self).setUp()
    self.data_dir = os.path.join(FLAGS.test_srcdir,
                                 os.path.split(os.path.abspath(__file__))[0],
                                 '../../environments/recommenders/test_data')
    self.env_config = movie_lens_dynamic.EnvConfig(
        seeds=movie_lens_dynamic.Seeds(0, 0),
        data_dir=self.data_dir,
        train_eval_test=[0.6, 0.2, 0.2],
        embeddings_path=os.path.join(self.data_dir, 'embeddings.json'))

    self.workdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)

  def tearDown(self):
    super(MovielensRnnAgentTest, self).tearDown()
    file_util.delete_recursively(self.workdir)

  def test_interaction(self):
    env = movie_lens_dynamic.create_gym_environment(self.env_config)
    agent = batched_movielens_rnn_agent.MovieLensRNNAgent(
        env.observation_space, env.action_space, max_episode_length=None)
    for _ in range(3):
      for _ in range(2):
        reward = 0
        observation = env.reset()
        for _ in range(2):
          slate = agent.step(reward, observation)
          observation, reward, _, _ = env.step(slate)
        agent.end_episode(reward, observation, eval_mode=True)
      agent.model_update(
          learning_rate=0.1, lambda_learning_rate=0.1, var_learning_rate=0.1)
      agent.empty_buffer()

  def test_stateful_interaction(self):
    env = movie_lens_dynamic.create_gym_environment(self.env_config)
    agent = batched_movielens_rnn_agent.MovieLensRNNAgent(
        env.observation_space,
        env.action_space,
        stateful=True,
        batch_size=1,
        max_episode_length=None)
    for _ in range(3):
      # The agent and environment simulate one episode at a time.
      agent.set_batch_size(1)
      for _ in range(7):
        reward = 0
        observation = env.reset()
        for _ in range(2):
          slate = agent.step(reward, observation)
          observation, reward, _, _ = env.step(slate)
        agent.end_episode(reward, observation, eval_mode=True)
      # There are 7 episodes in every batch used to update the model.
      agent.set_batch_size(7)
      agent.model_update(
          learning_rate=0.1, lambda_learning_rate=0.1, var_learning_rate=0.1)
      agent.empty_buffer()

  def test_no_user_id(self):
    env = movie_lens_dynamic.create_gym_environment(self.env_config)
    agent = batched_movielens_rnn_agent.MovieLensRNNAgent(
        env.observation_space,
        env.action_space,
        stateful=True,
        batch_size=1,
        user_id_input=False,
        user_embedding_size=0,
        max_episode_length=None)
    for _ in range(3):
      for _ in range(7):
        reward = 0
        observation = env.reset()
        for _ in range(2):
          slate = agent.step(reward, observation)
          observation, reward, _, _ = env.step(slate)
        agent.end_episode(reward, observation, eval_mode=True)
      # There are 7 episodes in every batch used to update the model.
      agent.set_batch_size(7)
      agent.model_update(
          learning_rate=0.1, lambda_learning_rate=0.1, var_learning_rate=0.1)
      agent.empty_buffer()
      # The agent and environment simulate one episode at a time.
      agent.set_batch_size(1)

  def test_batch_interaction(self):
    envs = [
        movie_lens_dynamic.create_gym_environment(self.env_config)
        for _ in range(5)
    ]
    agent = batched_movielens_rnn_agent.MovieLensRNNAgent(
        envs[0].observation_space,
        envs[0].action_space,
        max_episode_length=None)
    for _ in range(3):
      rewards = [0 for _ in envs]
      observations = [env.reset() for env in envs]
      for _ in range(2):
        slates = agent.step(rewards, observations)
        observations, rewards, _, _ = zip(
            *[env.step(slate) for env, slate in zip(envs, slates)])
      agent.end_episode(rewards, observations, eval_mode=True)
      agent.model_update(
          learning_rate=0.1, lambda_learning_rate=0.1, var_learning_rate=0.1)
      agent.empty_buffer()

if __name__ == '__main__':
  absltest.main()
