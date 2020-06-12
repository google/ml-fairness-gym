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
"""Recsim environment for toy restaurant MDP problem.

The environment simulates interactions between a user and recommender that
recommends one of two restaurants: one that serves healthy food and one
that serves junk food. The user's preferences for healthy and junk food evolve
over time following a Markov Decision Process in which their current preference
depends on their previous preferences and the most recent restaurant that they
have attended.

The parameters of the MDP are configured when constructing the environment.
"""
import enum
import functools
import itertools
from absl import flags
from absl import logging
import attr
from environments.recommenders import recsim_samplers
from gym import spaces
import numpy as np
from recsim import document
from recsim import user
from recsim.simulator import environment
from recsim.simulator import recsim_gym

FLAGS = flags.FLAGS


MAXINT = np.iinfo(np.int32).max


def unit_scalar_space():
  """Returns a space describing floats in [0, 1]."""
  return spaces.Box(0.0, 1.0, tuple(), np.float32)


class RestaurantType(enum.IntEnum):
  """Enum representing Restaurant types."""
  JUNK = 0
  HEALTHY = 1


@attr.s
class JsonableConfig(object):
  """Base class for configs that can be encoded as json dictionaries."""

  def to_jsonable(self):
    return attr.asdict(self)


@attr.s
class UserConfig(JsonableConfig):
  """Configuration for Users."""
  user_states_names = attr.ib(factory=lambda: ['Unhealthy', 'Healthy'])
  state_transition_matrix = attr.ib(
      factory=lambda: TransitionMatrix.RandomMatrix(2, 2))
  reward_matrix = attr.ib(factory=lambda: np.random.rand(2, 2))


@attr.s
class SimulationSeeds(JsonableConfig):
  """Random seeds to be used in simulation."""
  user_sampler = attr.ib(default=None)
  user_model = attr.ib(default=None)


@attr.s
class EnvConfig(JsonableConfig):
  """Configuration for a full environment."""
  user_config = attr.ib(factory=UserConfig)
  restaurant_types = attr.ib(factory=lambda: [r.value for r in RestaurantType])
  seeds = attr.ib(factory=SimulationSeeds)
  reward_weights = attr.ib(factory=lambda: [1.0, 0.0])


class InvalidTransitionMatrixError(Exception):
  """Raised if the MDP transition matrix is invalid."""


def _check_simplex(x, exception=ValueError):
  """Raises exception if x is not in a simplex."""
  x = np.array(x)  # Convert sequence to array.
  if not np.isclose(np.sum(x), 1):
    raise exception('x does not add to 1.')
  if not np.all(x >= 0):
    raise exception('x contains negative values.')


class TransitionMatrix(object):
  """Encodes transition probabilities for the MDP."""

  def __init__(self, num_states, num_actions):
    # Matrix[i, j, k] holds the probability of transitioning from state i to
    # state k as a result of action j.
    self._matrix = np.zeros((num_states, num_actions, num_states))
    self._num_states = num_states
    self._num_actions = num_actions

  def add_row(self, state, action, row):
    """Add a row to the transition matrix."""
    _check_simplex(row)
    self._matrix[state, action] = row
    return self

  def to_jsonable(self):
    """Convert object to something that can be json encoded."""
    return self._matrix

  @property
  def shape(self):
    return self._matrix.shape

  def transition(self, state, action, rng):
    """Return the next state."""
    return rng.choice(self._num_states, p=self._matrix[state, action])

  @classmethod
  def RandomMatrix(cls, num_states, num_actions, seed=None):  # pylint: disable=invalid-name
    """Returns a random transition matrix."""
    rng = np.random.RandomState(seed)
    transition_matrix = cls(num_states, num_actions)
    for state, action in itertools.product(
        range(num_states), range(num_actions)):
      random_transition = rng.dirichlet([1]*num_states)
      transition_matrix.add_row(
          state=state, action=action, row=random_transition)
    return transition_matrix

  @classmethod
  def FromArray(cls, array):  # pylint: disable=invalid-name
    """Constructs a TransitionMatrix from a numpy array."""
    array = np.array(array)  # Convert to a numpy array from the start.
    num_states, num_actions, _ = array.shape
    transition_matrix = cls(num_states, num_actions)
    for state, action in itertools.product(
        range(num_states), range(num_actions)):
      transition_matrix.add_row(
          state=state, action=action, row=array[state, action, :])
    return transition_matrix


class Restaurant(document.AbstractDocument):
  """Class to represent a Restaurant.

  Restaurants are the "documents" that the recommender can choose from.

  Attributes:
    _doc_id: A unique integer (accessed with the doc_id() method).
    restaurant_type: a `RestaurantType` enum value.
  """

  def __init__(self, restaurant_id, restaurant_type):
    super(Restaurant, self).__init__(int(restaurant_id))
    self.restaurant_type = restaurant_type

  def create_observation(self):
    """Returns a description of the restaurant."""
    return {'restaurant_id': self.doc_id(),
            'restaurant_type': np.int(self.restaurant_type)}

  @classmethod
  def observation_space(cls):
    """Returns a gym.Space describing observations."""
    return spaces.Dict({
        'restaurant_id': spaces.Discrete(MAXINT),
        'restaurant_type': spaces.Discrete(len(RestaurantType))
    })


class User(user.AbstractUserState):
  """Class to represent a user.

  Users receive recommendations from a recommender and respond to them.

  The user's behavior is governed by a Markov Decision Process (MDP). It has
  an internal state, which affects how it responds to restaurant
  recommendations. The state also changes based on the most recent
  recommendation.
  """

  def __init__(self, user_id, reward_matrix, state_transition_matrix,
               initial_state_id=0, user_states_names=None, seed=None):
    """Initializes a new user."""
    self.user_id = user_id
    self.initial_state_id = initial_state_id
    self.curr_state = initial_state_id
    self.num_states = state_transition_matrix.shape[0]
    if user_states_names is None:
      self.user_states_names = [str(i) for i in range(self.num_states)]
    else:
      self.user_states_names = user_states_names
    self.state_transition_matrix = state_transition_matrix
    self.reward_matrix = reward_matrix
    self._rng = np.random.RandomState(seed)

  def reward_recs(self, rec):
    """Returns the user's reward to the recommendation in the current state."""
    return self.reward_matrix[self.curr_state, rec]

  def score_document(self, doc):
    """Returns the user's affinity to the document."""
    return self.reward_matrix[self.curr_state, doc]

  def update_state(self, rec):
    """Updates the state of the user by sampling from the transition matrix."""
    self.curr_state = self.state_transition_matrix.transition(
        self.curr_state, rec, self._rng)

  def create_observation(self):
    """Returns a user observation.

    Only the user's ID is visible. Their current state is not.
    """
    return {'user_id': self.user_id}

  def reset_state(self):
    """Resets user to their initial state."""
    self.curr_state = self.initial_state_id

  @classmethod
  def observation_space(cls):
    return spaces.Dict({'user_id': spaces.Discrete(MAXINT)})


@attr.s
class Response(user.AbstractResponse):
  """Class to represent a user's response to a restaurant."""
  health_score = attr.ib()
  rating = attr.ib()

  def create_observation(self):
    return {
        'health_score': np.array(self.health_score),
        'rating': np.array(self.rating)
    }

  @classmethod
  def response_space(cls):
    return spaces.Dict({
        'rating': unit_scalar_space(),
        'health_score': unit_scalar_space(),
    })


class UserModel(user.AbstractUserModel):
  """Responsible for generating `Responses` for documents."""

  def __init__(self, user_sampler, seed=0):
    super(UserModel, self).__init__(
        response_model_ctor=Response, user_sampler=user_sampler, slate_size=1)

  def update_state(self, slate_documents, responses):
    del responses  # Unused.
    # Only one document is recommended in each interaction.
    rec = slate_documents[0].doc_id()
    self._user_state.update_state(rec)

  def _rate(self, doc):
    """Returns a rating for a document."""
    return self._user_state.reward_recs(doc.doc_id())

  def simulate_response(self, documents):
    """Returns responses for each document."""
    return [Response(health_score=doc.restaurant_type, rating=self._rate(doc))
            for doc in documents]

  def is_terminal(self):
    """Returns a boolean indicating if the session is over."""
    return False


def build_user_components(config):
  """Returns the user components of the simulator."""
  user_constuctor = functools.partial(User, **attr.asdict(config.user_config))
  user_sampler = recsim_samplers.ConstructionSampler(user_constuctor,
                                                     config.seeds.user_sampler)
  user_model = UserModel(
      user_sampler=user_sampler, seed=config.seeds.user_model)
  return user_sampler, user_model


def build_document_components(config):
  """Returns the document components of the simulator."""
  # Create a finite pool of restaurants as candidates.
  restaurants = [
      Restaurant(idx, type_)
      for idx, type_ in enumerate(config.restaurant_types)
  ]
  # There is no need to sample restaurants repeatedly. We just want to use
  # the same set over the course of the simulation. The SingletonSampler is
  # useful for this.
  return restaurants, recsim_samplers.SingletonSampler(restaurants, Restaurant)


def weighted_reward(responses, weights):
  r"""Aggregates the ratings and health scores from a sequence of Responses.

  The aggregation is \sum weights[0]*rating + weights[1] * health.

  Args:
    responses: A list of Responses.
    weights: A array with two weights.

  Returns:
    A scalar aggregation of scores.
  """
  _check_simplex(weights)
  return sum([
      np.dot(weights, [response.rating, response.health_score])
      for response in responses
  ])


def rating_reward(responses):
  """Returns a the sum of the ratings in a sequence of Responses."""
  return weighted_reward(responses, [1., 0.])


def build_restaurant_recs_recsim_env(config):
  """Returns a recsim_gym environment object."""
  _, user_model = build_user_components(config)
  restaurants, restaurant_sampler = build_document_components(config)
  env = environment.Environment(
      user_model=user_model,
      document_sampler=restaurant_sampler,
      num_candidates=len(restaurants),
      slate_size=1,
      resample_documents=False)
  reward_aggregator = functools.partial(weighted_reward,
                                        weights=config.reward_weights)
  return recsim_gym.RecSimGymEnv(env, reward_aggregator)


def run_simulation(env, num_iters=5, slate_size=10):
  """Collects rewards over random recommendation sequences from a given env."""
  rewards = []
  slate_seqs = []
  for _ in range(num_iters):
    logging.info('New User')
    env.reset()
    total = 0.0
    slate_seq = np.random.randint(2, size=slate_size)
    slate_seqs.append(slate_seq)
    for slate in slate_seq:
      observation, reward, _, _ = env.step(slate)
      curr_user_id = observation['user']['user_id']
      curr_reward = reward
      logging.info('User: %d, Response: %f', curr_user_id, curr_reward)
      total += observation['response'][0]['reward']
    logging.info('Total Reward: %f', total)
    rewards.append(total)
  mean_rewards = sum(rewards)/5.0
  logging.info('Average Reward = %f', mean_rewards)
  return slate_seqs, rewards
