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
"""Tests for movielens_recs."""

import hashlib
import os
import pickle
import tempfile
from absl import flags
from absl.testing import absltest
import file_util
from environments.recommenders import movie_lens_dynamic
from experiments import movielens_recs
import numpy as np


FLAGS = flags.FLAGS


class CvarExperimentMovielensTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()
    self.test_data_dir = os.path.join(
        FLAGS.test_srcdir,
        os.path.split(os.path.abspath(__file__))[0],
        '../environments/recommenders/test_data')
    embedding_dim = 55

    user_emb = np.zeros((5, embedding_dim))
    movie_emb = np.zeros((5, embedding_dim))

    # Range of the dot product is [-2, 2]
    user_emb[:, 1] = (np.random.rand(5) - 0.5) * 4
    movie_emb[:, 1] = (np.random.rand(5) - 0.5) * 4

    # Add a bias term of 3.0 as dim 0.
    user_emb[:, 0] = 1.0
    movie_emb[:, 0] = 3.0
    initial_embeddings = {'users': user_emb, 'movies': movie_emb}
    pickle_file = os.path.join(self.tempdir, 'embeddings.pkl')

    with file_util.open(pickle_file, 'wb') as outfile:
      pickle.dump(initial_embeddings, outfile)
    self.env_config = movie_lens_dynamic.EnvConfig(
        seeds=movie_lens_dynamic.Seeds(0, 0),
        data_dir=self.test_data_dir,
        train_eval_test=[0.6, 0.2, 0.2],
        embeddings_path=pickle_file)

    self.config = {
        'results_dir': self.tempdir,
        'max_episode_length': 2,
        'initial_lambda': 0,
        'beta': 0.5,
        'alpha': 0.95,
        'experiment_suffix': 'my_test_expriment',
        'lambda_learning_rate': 0.1,
        'var_learning_rate': 0.1,
        'learning_rate': 0.1,
        'embedding_size': 10,
        'user_embedding_size': 0,
        'hidden_size': 4,
        'num_hidden_layers': 4,
        'num_users_eval': 4,
        'num_users_eval_final': 4,
        'num_episodes_per_update': 8,
        'optimizer_name': 'Adam',
        'num_updates': 2,
        'eval_deterministic': True,
        'gamma': 0.95,
        'baseline_value': 0.1,
        'momentum': 0.1,
        'clipnorm': 0.1,
        'clipval': 0.1,
        'checkpoint_every': 1,
        'lr_scheduler': 1,
        'eval_every': 1,
        'regularization_coeff': 0.5,
        'agent_seed': 103,
        'initial_agent_model': None,
        'activity_regularization_coeff': 0.1,
        'dropout': 0.1,
        'stateful_rnn': True,
    }
    self.config['user_id_input'] = self.config['user_embedding_size'] > 0
    self.config['experiment_name'] = 'id_' + hashlib.sha1(
        repr(sorted(self.config.items())).encode()).hexdigest()
    self.config['env_config'] = self.env_config

  def tearDown(self):
    super().tearDown()
    file_util.delete_recursively(self.tempdir)

  def test_do_training(self):
    movielens_recs.train(self.config)


if __name__ == '__main__':
  absltest.main()
