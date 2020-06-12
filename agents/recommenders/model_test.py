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
"""Tests for safe_rl_recs.agents.model."""

from absl.testing import absltest
from agents.recommenders import batched_movielens_rnn_agent
from agents.recommenders import model


class ModelTest(absltest.TestCase):

  def test_sequence_model_outputs_can_vary_in_size(self):
    vocab_size = 64
    num_users = 16

    my_model = model.create_model(
        max_episode_length=None,
        action_space_size=vocab_size,
        embedding_size=16,
        hidden_size=8,
        batch_size=None,
        user_id_input=True,
        num_users=num_users,
        user_embedding_size=4,
        repeat_recs_in_episode=False,
        genre_vector_input=False,
        genre_vec_size=3)
    start_token = vocab_size

    model_input = batched_movielens_rnn_agent.Sequence(
        vocab_size=vocab_size, mask_previous_recs=True, start_token=start_token)
    observation = {'user': {'user_id': 4}, 'response': [{'violence_score': 0}]}

    # `vocab_size` is used as a start token as it's out of the vocabulary.
    start_token = vocab_size
    model_input.update(
        last_recommendation=start_token, reward=3, observation=observation)
    model_input.update(last_recommendation=2, reward=0, observation=observation)
    model_input.update(last_recommendation=3, reward=1, observation=observation)

    input_ = model_input.build_prediction_input(
        ['recommendations', 'rewards', 'users', 'final_mask'])
    output = my_model.predict(input_)
    self.assertEqual(output.shape, (1, 3, 64))

    model_input.update(last_recommendation=4, reward=1, observation=observation)
    input_ = model_input.build_prediction_input(
        ['recommendations', 'rewards', 'users', 'final_mask'])
    output = my_model.predict(input_)
    self.assertEqual(output.shape, (1, 4, 64))

  def test_batch_model_input(self):
    vocab_size = 64
    num_users = 16
    start_token = vocab_size

    my_model = model.create_model(
        max_episode_length=None,
        action_space_size=vocab_size,
        embedding_size=16,
        hidden_size=8,
        batch_size=None,
        user_id_input=True,
        num_users=num_users,
        user_embedding_size=4,
        repeat_recs_in_episode=False,
        genre_vector_input=False,
        genre_vec_size=3)

    model_input = batched_movielens_rnn_agent.Sequence(
        vocab_size=vocab_size, mask_previous_recs=True, start_token=start_token)

    observation = {'user': {'user_id': 4}, 'response': [{'violence_score': 0}]}
    model_input.update(
        last_recommendation=1,
        reward=3,
        observation=observation,
        batch_position=0)
    model_input.update(
        last_recommendation=2,
        reward=0,
        observation=observation,
        batch_position=0)
    model_input.update(
        last_recommendation=3,
        reward=1,
        observation=observation,
        batch_position=0)

    observation = {'user': {'user_id': 2}, 'response': [{'violence_score': 0}]}
    model_input.update(
        last_recommendation=3,
        reward=3,
        observation=observation,
        batch_position=1)
    model_input.update(
        last_recommendation=2,
        reward=0,
        observation=observation,
        batch_position=1)
    model_input.update(
        last_recommendation=1,
        reward=1,
        observation=observation,
        batch_position=1)

    input_ = model_input.build_prediction_input(
        ['recommendations', 'rewards', 'users', 'final_mask'])
    output = my_model.predict(input_)
    self.assertEqual(output.shape, (2, 3, 64))


if __name__ == '__main__':
  absltest.main()
