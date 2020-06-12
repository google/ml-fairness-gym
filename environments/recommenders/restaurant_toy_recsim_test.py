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
"""Tests for restaurant_toy_recsim."""


import itertools
from absl import flags
from absl.testing import absltest
import test_util
from environments.recommenders import recsim_wrapper
from environments.recommenders import restaurant_toy_recsim
import numpy as np
from recsim.simulator import environment
from recsim.simulator import recsim_gym

FLAGS = flags.FLAGS


def _always_moving_transition_matrix(num_states, num_actions):
  """Returns a transition matrix that moves deterministically at every step."""
  transition_matrix = restaurant_toy_recsim.TransitionMatrix(
      num_states, num_actions)
  for state, action in itertools.product(range(num_states), range(num_actions)):
    row = np.zeros(num_states, dtype=float)
    row[(state + 1) % num_states] = 1.0
    transition_matrix.add_row(state=state, action=action, row=row)
  return transition_matrix


def _build_components(deterministic_transitions=False):
  """Returns recsim components."""
  rec_types = [
      restaurant_toy_recsim.RestaurantType.JUNK,
      restaurant_toy_recsim.RestaurantType.HEALTHY
  ]
  user_states = ['Neutral', 'UnhealthySt', 'HealthySt']
  num_states = len(user_states)
  num_actions = len(rec_types)

  transition_matrix_constructor = (
      _always_moving_transition_matrix if deterministic_transitions else
      restaurant_toy_recsim.TransitionMatrix.RandomMatrix)

  user_config = restaurant_toy_recsim.UserConfig(
      user_states_names=user_states,
      state_transition_matrix=transition_matrix_constructor(
          num_states, num_actions),
      reward_matrix=np.random.rand(num_states, num_actions))

  seeds = restaurant_toy_recsim.SimulationSeeds(2, 5)
  config = restaurant_toy_recsim.EnvConfig(user_config, rec_types, seeds)
  user_sampler, user_model = restaurant_toy_recsim.build_user_components(config)
  restaurants, document_sampler = restaurant_toy_recsim.build_document_components(
      config)

  env = environment.Environment(
      user_model,
      document_sampler,
      num_candidates=num_actions,
      slate_size=1,
      resample_documents=False)

  recsim_env = recsim_gym.RecSimGymEnv(env, restaurant_toy_recsim.rating_reward)
  return (config, user_sampler, user_model, restaurants, document_sampler,
          recsim_env)


class RestaurantToyExampleTest(absltest.TestCase):

  def setUp(self):
    super(RestaurantToyExampleTest, self).setUp()
    (self.config, self.usersampler, self.user_model, self.restaurants,
     self.document_sampler,
     self.env) = _build_components(deterministic_transitions=False)

  def set_up_deterministic(self):
    (self.config, self.usersampler, self.user_model, self.restaurants,
     self.document_sampler,
     self.env) = _build_components(deterministic_transitions=True)

  def test_document_observation_space_matches(self):
    for doc in self.restaurants:
      self.assertIn(doc.create_observation(), doc.observation_space())

  def test_user_observation_space_matches(self):
    user = self.usersampler.sample_user()
    self.assertIn(user.create_observation(), user.observation_space())

  def test_environment_can_advance_by_steps(self):
    self.env.reset()
    for slate in [[0]]*10:
      self.env.step(slate)
      # Tests that the env.step() completes successfully

  def test_different_users_are_sampled_as_steps_progress(self):
    unique_users = set()
    for _ in range(5):
      self.env.reset()
      for slate in [[0], [0], [0], [0]]:
        observation, _, _, _ = self.env.step(slate)
        unique_users.add(observation['user']['user_id'])
    self.assertLen(unique_users, 5)

  def test_update_state_changes_state(self):
    """Tests if the user transitions state when step is called.

    The test uses an off diagonal deterministic transition matrix to make sure
    the user moves for sure.
    """
    self.set_up_deterministic()
    self.env.reset()
    state = self.env._environment.user_model._user_state.curr_state
    for slate in [[0]]*10:
      self.env.step(slate)
      next_state = self.env._environment.user_model._user_state.curr_state
      self.assertNotEqual(state, next_state)
      state = next_state

  def test_ml_fairness_gym_environment_can_run(self):
    self.set_up_deterministic()
    ml_fairness_env = recsim_wrapper.wrap(self.env)
    test_util.run_test_simulation(env=ml_fairness_env, stackelberg=True)


if __name__ == '__main__':
  absltest.main()
