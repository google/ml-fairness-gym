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
"""Tests for movie_lens."""
import copy
import functools
import os
import tempfile
from absl import flags
from absl.testing import absltest
import attr
import file_util
import test_util
from environments.recommenders import movie_lens_dynamic as movie_lens
from environments.recommenders import movie_lens_utils
from environments.recommenders import recsim_samplers
from environments.recommenders import recsim_wrapper
import numpy as np
from recsim.simulator import recsim_gym

FLAGS = flags.FLAGS


class MovieLensTest(absltest.TestCase):

  def _initialize_from_config(self, env_config):
    self.working_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)

    self.initial_embeddings = movie_lens_utils.load_embeddings(env_config)

    user_ctor = functools.partial(movie_lens.User,
                                  **attr.asdict(env_config.user_config))
    self.dataset = movie_lens_utils.Dataset(
        env_config.data_dir,
        user_ctor=user_ctor,
        movie_ctor=movie_lens.Movie,
        embeddings=self.initial_embeddings)

    self.document_sampler = recsim_samplers.SingletonSampler(
        self.dataset.get_movies(), movie_lens.Movie)

    self.user_sampler = recsim_samplers.UserPoolSampler(
        seed=env_config.seeds.user_sampler,
        users=self.dataset.get_users(),
        user_ctor=user_ctor)

    self.user_model = movie_lens.UserModel(
        user_sampler=self.user_sampler,
        seed=env_config.seeds.user_model,
    )

    env = movie_lens.MovieLensEnvironment(
        self.user_model,
        self.document_sampler,
        num_candidates=self.document_sampler.size(),
        slate_size=1,
        resample_documents=False)
    env.reset()

    reward_aggregator = functools.partial(
        movie_lens.multiobjective_reward,
        lambda_non_violent=env_config.lambda_non_violent)
    self.env = recsim_gym.RecSimGymEnv(env, reward_aggregator)

  def setUp(self):
    super(MovieLensTest, self).setUp()
    self.data_dir = os.path.join(FLAGS.test_srcdir,
                                 os.path.split(os.path.abspath(__file__))[0],
                                 'test_data')
    self.env_config = movie_lens.EnvConfig(
        seeds=movie_lens.Seeds(0, 0),
        data_dir=self.data_dir,
        embeddings_path=os.path.join(self.data_dir, 'embeddings.json'))
    self._initialize_from_config(self.env_config)

  def tearDown(self):
    file_util.delete_recursively(self.working_dir)
    super(MovieLensTest, self).tearDown()

  def test_document_observation_space_matches(self):
    for doc in self.dataset.get_movies():
      self.assertIn(doc.create_observation(), doc.observation_space())

  def test_user_observation_space_matches(self):
    user = self.user_sampler.sample_user()
    self.assertIn(user.create_observation(), user.observation_space())

  def test_observations_in_observation_space(self):
    for slate in [[0], [1], [2]]:
      observation, _, _, _ = self.env.step(slate)
      for field in ['doc', 'response', 'user']:
        self.assertIn(observation[field],
                      self.env.observation_space.spaces[field])

  def test_user_can_score_document(self):
    user = self.user_sampler.get_user(1)
    for doc in self.dataset.get_movies():
      self.assertBetween(
          user.score_document(doc), user.MIN_SCORE, user.MAX_SCORE)

  def test_environment_can_advance_by_steps(self):
    # Recommend some manual slates.
    for slate in [[0], [1], [3]]:
      # Tests that env.step completes successfully.
      self.env.step(slate)

  def test_environment_observation_space_is_as_expected(self):
    for slate in [[0], [1], [2]]:
      observation, _, _, _ = self.env.step(slate)
      for field in ['doc', 'response', 'user']:
        self.assertIn(observation[field],
                      self.env.observation_space.spaces[field])

  def test_gym_environment_builder(self):
    env = movie_lens.create_gym_environment(self.env_config)
    env.seed(100)
    env.reset()

    # Recommend some manual slates and check that the observations are as
    # expected.
    for slate in [[0], [0], [2]]:
      observation, _, _, _ = env.step(slate)
      for field in ['doc', 'response', 'user']:
        self.assertIn(observation[field], env.observation_space.spaces[field])

  def test_if_user_state_resets(self):
    observation = self.env.reset()
    curr_user_id = observation['user']['user_id']
    ta_vec = np.copy(self.env._environment.user_model._user_sampler
                     ._users[curr_user_id].topic_affinity)
    for i in range(3):
      self.env.step([i])
    self.env.reset()
    ta_new = self.env._environment.user_model._user_sampler._users[
        curr_user_id].topic_affinity
    self.assertTrue(np.all(ta_new == ta_vec))

  def test_user_order_is_shuffled(self):
    """Tests that user order does not follow a fixed pattern.

    We test this by checking that the list is not perioc for periods between
    0-10. Since there are only 5 unique users, this is enough to show that
    it's not following a simple pattern.
    """
    self.env.seed(100)

    user_list = []
    for _ in range(100):
      observation = self.env.reset()
      user_list.append(observation['user']['user_id'])

    def _is_periodic(my_list, period):
      for idx, val in enumerate(my_list[:-period]):
        if val != my_list[idx + period]:
          return False
      return True

    for period in range(1, 10):
      self.assertFalse(_is_periodic(user_list, period))

  def test_user_order_is_consistent(self):
    self.env.reset_sampler()
    first_list = []
    for _ in range(100):
      observation = self.env.reset()
      first_list.append(observation['user']['user_id'])

    self.env.reset_sampler()
    other_list = []
    for _ in range(100):
      observation = self.env.reset()
      other_list.append(observation['user']['user_id'])

    self.assertEqual(first_list, other_list)

    # Also check that changing the seed creates a new ordering.
    config = copy.deepcopy(self.env_config)
    config.seeds.user_sampler += 1
    env = movie_lens.create_gym_environment(config)
    other_list = []
    for _ in range(100):
      observation = env.reset()
      other_list.append(observation['user']['user_id'])
    self.assertNotEqual(first_list, other_list)

  def test_ml_fairness_gym_environment_can_run(self):
    ml_fairness_env = recsim_wrapper.wrap(self.env)
    test_util.run_test_simulation(env=ml_fairness_env, stackelberg=True)


if __name__ == '__main__':
  absltest.main()
