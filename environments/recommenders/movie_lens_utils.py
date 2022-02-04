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
"""Utilities for movielens simulation."""

import json
import os
import pathlib
import pickle
import types
from absl import logging
import file_util
import numpy as np
import pandas as pd

GENRES = [
    'Other', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
GENRE_MAP = {genre: idx for idx, genre in enumerate(GENRES)}
OTHER_GENRE_IDX = GENRE_MAP['Other']

NUM_MOVIES = 6041
NUM_USERS = 3884


def load_embeddings(env_config):
  """Attempts to loads user and movie embeddings from a json or pickle file."""
  path = env_config.embeddings_path
  suffix = pathlib.Path(path).suffix
  if suffix == '.json':
    loader = json
    logging.info('Reading a json file. %s', path)
  elif suffix in ('.pkl', '.pickle'):
    loader = pickle
    logging.info('Reading a pickle file. %s', path)
  else:
    raise ValueError('Unrecognized file type! %s' % path)

  embedding_dict = loader.load(file_util.open(path, 'rb'))
  return types.SimpleNamespace(
      movies=np.array(embedding_dict[env_config.embedding_movie_key]),
      users=np.array(embedding_dict[env_config.embedding_user_key]))


class Dataset(object):
  """Class to represent all of the movielens data together."""

  def __init__(self, data_dir, embeddings, user_ctor, movie_ctor):
    """Initializes data from the data directory.

    Args:
      data_dir: Path to directory with {movies, users}.csv files.
      embeddings: An object containing embeddings with attributes `users` and
        `movies`.
      user_ctor: User constructor.
      movie_ctor: Movie constructor.
    """
    self._user_ctor = user_ctor
    self._movie_ctor = movie_ctor
    self._movies = self._read_movies(os.path.join(data_dir, 'movies.csv'))
    self._users = self._read_users(os.path.join(data_dir, 'users.csv'))
    self._populate_embeddings(embeddings)

  def _read_movies(self, path):
    """Returns a dict of Movie objects."""
    movies = {}
    movie_df = pd.read_csv(file_util.open(path))

    for _, row in movie_df.iterrows():
      genres = [
          GENRE_MAP.get(genre, OTHER_GENRE_IDX)
          for genre in row.genres.split('|')
      ]
      assert isinstance(row.movieId, int)
      movie_id = row.movieId
      # `movie_vec` is left as None, and will be filled in later in the init
      # of this Dataset.
      movies[movie_id] = self._movie_ctor(
          movie_id,
          row.title,
          genres,
          vec=None,
          violence=row.violence_tag_relevance)
    return movies

  def _read_users(self, path):
    """Returns a dict of User objects."""
    users = {}
    for _, row in pd.read_csv(file_util.open(path)).iterrows():
      users[row.userId] = self._user_ctor(user_id=row.userId)
    return users

  def _populate_embeddings(self, initial_embeddings):
    """Modifies stored Users and Movies with learned vectors."""

    for movie_ in self.get_movies():
      movie_.movie_vec = initial_embeddings.movies[movie_.doc_id()]
      assert (len(movie_.movie_vec) > len(GENRES)
             ), 'The movie embeddings must include genre dimensions.'
    for user_ in self.get_users():
      user_.topic_affinity = np.copy(initial_embeddings.users[user_.user_id])
      # Since users' topic affinities can change over time, store the initial
      # value as well.
      user_.initial_topic_affinity = np.copy(
          initial_embeddings.users[user_.user_id])

  def get_movies(self):
    """Returns an iterator over movies."""
    return list(self._movies.values())

  def get_users(self):
    """Returns an iterator over users."""
    return list(self._users.values())
