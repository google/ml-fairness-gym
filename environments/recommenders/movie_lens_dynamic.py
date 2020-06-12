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
"""Dynamic user simulation using movielens data.

A user's rating of a movie is given by a dot-product between a user vector and
a movie vector, clipped between [0, 5].

In this simulation, users become more interested in movies from a genre as they
watch more movies from that genre. For example, a user that watches more
`Adventure` movies, rates future `Adventure` movies more highly.

Movies are annotated with a violence score, derived from the movie genome
project. While considering the quality of recommendations, an agent can also
consider how much exposure to violence the user is being exposed to.

The download_movielens.py script can be used to download the movielens 1M
dataset and format it for use in this environment.
"""
import collections
import functools
from absl import flags
import attr
import core
from environments.recommenders import movie_lens_utils
from environments.recommenders import recsim_samplers
from gym import spaces
import numpy as np
from recsim import document
from recsim import user
from recsim.simulator import environment as recsim_environment
from recsim.simulator import recsim_gym


FLAGS = flags.FLAGS


@attr.s
class UserConfig(core.Params):
  affinity_update_delta = attr.ib(default=1.0)
  topic_affinity_update_threshold = attr.ib(default=3.0)


@attr.s
class Seeds(core.Params):
  user_sampler = attr.ib(default=None)
  user_model = attr.ib(default=None)
  train_eval_test = attr.ib(default=None)


@attr.s
class EnvConfig(core.Params):
  """Config object for MovieLensEnvironment."""
  data_dir = attr.ib()
  # Path to a json or pickle file with a dict containing user and movie
  # embeddings
  embeddings_path = attr.ib()
  # Dictionary key to access the movie embeddings.
  embedding_movie_key = attr.ib(default='movies')
  # Dictionary key to access the user embeddings.
  embedding_user_key = attr.ib(default='users')
  train_eval_test = attr.ib(factory=lambda: [0.7, 0.15, 0.15])
  # Amount of weight to assign to non-violence in a multireward objective.
  lambda_non_violent = attr.ib(default=0)
  user_config = attr.ib(factory=UserConfig)
  seeds = attr.ib(factory=Seeds)


class Movie(document.AbstractDocument):
  """Class to represent a MovieLens Movie.

  Attributes:
    _doc_id: A unique integer (accessed with the doc_id() method).
    title: A string.
    genres: A list of ints between 0 and len(GENRES).
    genre_vec: A binary vector of size len(GENRES).
    movie_vec: An embedding vector for this movie.
    violence: A float indicating how much violence is in the movie.
  """

  def __init__(self, doc_id, title, genres, vec, violence):
    super(Movie, self).__init__(int(doc_id))
    self.title = title
    self.genres = genres
    self.genre_vec = np.zeros(len(movie_lens_utils.GENRES), dtype=np.int)
    self.genre_vec[self.genres] = 1
    self.movie_vec = vec
    self.violence = violence

  def create_observation(self):
    """Returns an observation dictionary."""
    return {'genres': self.genre_vec, 'doc_id': self.doc_id()}

  @classmethod
  def observation_space(cls):
    """Returns a gym.Space describing observations."""
    return spaces.Dict({
        'genres': spaces.MultiBinary(len(movie_lens_utils.GENRES)),
        'doc_id': spaces.Discrete(movie_lens_utils.NUM_MOVIES)
    })


class User(user.AbstractUserState):
  """Class to represent a movielens user."""

  MIN_SCORE = 1
  MAX_SCORE = 5

  def __init__(self,
               user_id,
               affinity_update_delta=1.0,
               topic_affinity_update_threshold=3.0):
    """Initializes the dynamic user.

    Args:
      user_id: Integer identifier of the user.
      affinity_update_delta: Delta for updating user's preference for genres
        whose movies are rated >= topic_affinity_update_threshold.
      topic_affinity_update_threshold: Rating threshold above which user's
        preferences for the genre's is updated.
    """
    self.user_id = user_id
    self.affinity_update_delta = affinity_update_delta
    self.topic_affinity_update_threshold = topic_affinity_update_threshold
    # The topic_affinity vector is None until the _populate_embeddings call
    # assigns a vector to it.
    self.topic_affinity = None
    # Also store a pristine initial value so that the user can be reset after
    # topic affinity changes during the simulation.
    self.initial_topic_affinity = None

  def score_document(self, doc):
    """Returns the user's affinity to the document."""
    return np.clip(
        np.dot(doc.movie_vec, self.topic_affinity), self.MIN_SCORE,
        self.MAX_SCORE)

  def create_observation(self):
    """Returns a user observation."""
    return {'user_id': self.user_id}

  def _update_affinity_vector(self, doc):
    embedding_dim = len(self.topic_affinity)
    assert embedding_dim >= len(movie_lens_utils.GENRES)
    offset_index = embedding_dim - len(movie_lens_utils.GENRES)
    genre_indices_to_update = [
        genre_id + offset_index for genre_id in doc.genres
    ]
    self.topic_affinity[genre_indices_to_update] *= (1 +
                                                     self.affinity_update_delta)

  def update_state(self, doc, response):
    if response.rating >= self.topic_affinity_update_threshold:
      self._update_affinity_vector(doc)

  def reset_state(self):
    self.topic_affinity = np.copy(self.initial_topic_affinity)

  @classmethod
  def observation_space(cls):
    return spaces.Dict({'user_id': spaces.Discrete(movie_lens_utils.NUM_USERS)})


class Response(user.AbstractResponse):
  """Class to represent a user's response to a document."""

  def __init__(self, rating=0, violence_score=0):
    self.rating = rating
    self.violence_score = violence_score

  def create_observation(self):
    # Ratings are cast into numpy floats to be consistent with the space
    # described by `spaces.Box` (see the response_space description below).
    return {
        'rating': np.float_(self.rating),
        'violence_score': np.float_(self.violence_score),
    }

  @classmethod
  def response_space(cls):
    return spaces.Dict({
        'rating': spaces.Box(0, 5, tuple(), np.float32),
        'violence_score': spaces.Box(0, 1, tuple(), np.float32),
    })


class UserModel(user.AbstractUserModel):
  """Dynamic Model of a user responsible for generating responses."""

  def __init__(self,
               user_sampler,
               seed=None,
               affinity_update_delta=1.0,
               topic_affinity_update_threshold=3.0):
    """Defines the dynamic user model.

    Args:
      user_sampler: Object of Class UserSampler.
      seed: Random seed for the user model.
      affinity_update_delta: Delta for updating user's preference for genres
        whose movies are rated >= topic_affinity_update_threshold.
      topic_affinity_update_threshold: Rating threshold above which user's
        preferences for the genre's is updated.
    """
    super().__init__(
        slate_size=1,
        user_sampler=user_sampler,
        response_model_ctor=Response)
    self._response_model_ctor = Response
    self.affinity_update_delta = affinity_update_delta
    self.topic_affinity_update_threshold = topic_affinity_update_threshold
    self._rng = np.random.RandomState(seed)

  def update_state(self, slate_documents, responses):
    """Updates the user state for the current user.

    Updates the topic_affinity vector for the current user based on responses.

    Args:
      slate_documents: List of documents in the slate recommended to the user.
      responses: List of response objects.
    Updates:
      The user's topic affinity in self._user_state.topic_affinity
    """
    for doc, response in zip(slate_documents, responses):
      self._user_state.update_state(doc, response)

  def simulate_response(self, documents):
    """Simulates the user's response to a slate of documents.

    Args:
      documents: a list of Movie objects in the slate.

    Returns:
      responses: a list of Response objects, one for each document.
    """

    return [
        self._response_model_ctor(
            rating=self._user_state.score_document(doc),
            violence_score=doc.violence) for doc in documents
    ]

  def is_terminal(self):
    """Returns a boolean indicating if the session is over."""
    return False

  def reset(self):
    """Resets the current user to their initial state and samples a new user."""
    self._user_state.reset_state()
    self._user_state = self._user_sampler.sample_user()


class MovieLensEnvironment(recsim_environment.Environment):
  """MovieLensEnvironment with some modifications to recsim.Environment."""

  USER_POOLS = {'train': 0, 'eval': 1, 'test': 2}

  def reset(self):
    self._user_model.reset()
    user_obs = self._user_model.create_observation()
    if self._resample_documents:
      self._do_resample_documents()
    if self._resample_documents or not hasattr(self, '_current_documents'):
      # Since _candidate_set.create_observation() is an expensive operation
      # for large candidate sets, this step only needs to be done when the
      # _current_documents attribute hasn't been defined yet or
      # _resample_documents is set to True.
      self._current_documents = collections.OrderedDict(
          self._candidate_set.create_observation())
    return (user_obs, self._current_documents)

  def step(self, slate):
    """Executes the action, returns next state observation and reward.

    Args:
      slate: An integer array of size slate_size, where each element is an index
        into the set of current_documents presented

    Returns:
      user_obs: A gym observation representing the user's next state
      doc_obs: A list of observations of the documents
      responses: A list of AbstractResponse objects for each item in the slate
      done: A boolean indicating whether the episode has terminated
    """
    assert (len(slate) <= self._slate_size
           ), 'Received unexpectedly large slate size: expecting %s, got %s' % (
               self._slate_size, len(slate))

    # Get the documents associated with the slate
    doc_ids = list(self._current_documents)  # pytype: disable=attribute-error
    mapped_slate = [doc_ids[x] for x in slate]
    documents = self._candidate_set.get_documents(mapped_slate)
    # Simulate the user's response
    responses = self._user_model.simulate_response(documents)

    # Update the user's state.
    self._user_model.update_state(documents, responses)

    # Update the documents' state.
    self._document_sampler.update_state(documents, responses)

    # Obtain next user state observation.
    user_obs = self._user_model.create_observation()

    # Check if reaches a terminal state and return.
    done = self._user_model.is_terminal()

    # Optionally, recreate the candidate set to simulate candidate
    # generators for the next query.
    if self._resample_documents:
      self._do_resample_documents()

      # Create observation of candidate set.
      # Compared to the original recsim environment code, _current_documents
      # needs to be done only for each step when resample_docuemnts is set to
      # True.
      self._current_documents = collections.OrderedDict(
          self._candidate_set.create_observation())

    return (user_obs, self._current_documents, responses, done)

  def set_active_pool(self, pool_name):
    self._user_model._user_sampler.set_active_pool(self.USER_POOLS[pool_name])  # pylint: disable=protected-access


def average_ratings_reward(responses):
  """Calculates the average rating for the slate from a list of responses."""
  if not responses:
    raise ValueError('Empty response list')
  return np.mean([response.rating for response in responses])


def multiobjective_reward(responses, lambda_non_violent=0.0):
  """Calculates the reward for the multi-objective setting.

  Reward aggregator for recsim environment to calcualte reward of a traejectory.

  Usage: Build a lambda function for a given value of lambda

  Args:
    responses: A list of Response objects from the environment.
    lambda_non_violent: Weight to the non_violence score.
      Ratings get (1-lambda) weight.

  Returns:
    A scalar reward.
  """
  if not responses:
    raise ValueError('Empty response list.')
  return np.mean([
      lambda_non_violent * (1-response.violence_score) +
      (1 - lambda_non_violent) * response.rating for response in responses
  ])


def create_gym_environment(env_config):
  """Returns a RecSimGymEnv with specified environment parameters.

  Args:
    env_config: an `EnvConfig` object.
  Returns:
    A RecSimGymEnv object.
  """

  user_ctor = functools.partial(User, **attr.asdict(env_config.user_config))

  initial_embeddings = movie_lens_utils.load_embeddings(
      env_config)

  dataset = movie_lens_utils.Dataset(
      env_config.data_dir,
      user_ctor=user_ctor,
      movie_ctor=Movie,
      embeddings=initial_embeddings)

  document_sampler = recsim_samplers.SingletonSampler(dataset.get_movies(),
                                                      Movie)

  user_sampler = recsim_samplers.UserPoolSampler(
      seed=env_config.seeds.user_sampler, users=dataset.get_users(),
      user_ctor=user_ctor, partitions=env_config.train_eval_test,
      partition_seed=env_config.seeds.train_eval_test)

  user_model = UserModel(
      user_sampler=user_sampler,
      seed=env_config.seeds.user_model,
  )

  env = MovieLensEnvironment(
      user_model,
      document_sampler,
      num_candidates=document_sampler.size(),
      slate_size=1,
      resample_documents=False)

  reward_aggregator = functools.partial(
      multiobjective_reward, lambda_non_violent=env_config.lambda_non_violent)

  return recsim_gym.RecSimGymEnv(env, reward_aggregator)
