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
"""Samplers for Recsim simulations."""
import numpy as np
from recsim import document
from recsim import user

MAXINT = np.iinfo(np.int32).max


class SequentialSampler(document.AbstractDocumentSampler):
  """Iterates over a sequence of candidate documents."""

  def __init__(self, documents, doc_ctor, repeat=True):
    self._corpus = documents
    self._idx = 0
    self._doc_ctor = doc_ctor
    self.repeat = repeat

  def reset_sampler(self):
    self._idx = 0

  def size(self):
    return len(self._corpus)

  def sample_document(self):
    """Returns the next document.

    If the sampler is a repeating sampler (constructed with repeat=True),
    it will back to the start if the corpus is exhausted.

    Raises:
      IndexError: if self.repeat is False and the corpus is exhausted.
    """
    if self._idx >= len(self._corpus):
      if not self.repeat:
        raise IndexError('Attempting to sample more items than available.')
      self.reset_sampler()
    doc = self._corpus[self._idx]
    self._idx += 1
    return doc


class SingletonSampler(SequentialSampler):
  """Iterates over a sequence of candidate documents only once."""

  def __init__(self, documents, doc_ctor):
    super(SingletonSampler, self).__init__(documents, doc_ctor, repeat=False)


class ConstructionSampler(user.AbstractUserSampler):
  """Constructs a new user with a unique user id for each sample."""

  def __init__(self, user_ctor, seed):
    """Initializes the ConstructionSampler.

    Args:
      user_ctor: A User constructor with two arguments: (user_id, seed)
      seed: Random seed for the sampler.
    """
    super(ConstructionSampler, self).__init__(user_ctor=user_ctor, seed=seed)
    self.user_id = -1

  def sample_user(self):
    """Generates a new user with a unique user id.."""
    self.user_id += 1
    return self._user_ctor(self.user_id, seed=self._rng.randint(0, MAXINT))


class UserPoolSampler(user.AbstractUserSampler):
  """Samples users from a fixed pool read in at initialization."""

  def __init__(self,
               users,
               user_ctor,
               seed=None,
               partitions=None,
               partition_seed=100):
    """Initializes the UserPoolSampler.

    Args:
      users: A list of `AbstractUsers`.
      user_ctor: Constructor for the user class.
      seed: Random seed for the pool sampler.
      partitions: A list of floats that describe how to partition the users.
        For example: [0.3, 0.3, 0.4] would create 3 partitions, with 30%, 30%
        and 40% of the users, respectively.
      partition_seed: Used to control how users are randomly allocated to
        partitions.
    """
    super(UserPoolSampler, self).__init__(seed=seed,
                                          user_ctor=user_ctor)
    self._users = {user.user_id: user for user in users}
    self._partitions = [np.array(list(self._users.keys()))]
    self._active_pool = 0

    if partitions is not None and not np.isclose(np.sum(partitions), 1.0):
      raise ValueError('Partitions must sum to 1.')

    # Shuffle the keys to create a random partition.
    partition_rng = np.random.RandomState(partition_seed)
    partition_rng.shuffle(self._partitions[0])
    if partitions is not None:
      cutpoints = (np.cumsum(partitions)*len(self._users)).astype(np.int32)
      # The final cutpoint at len does not need to be specified.
      self._partitions = np.split(self._partitions[0], cutpoints[:-1])

    for partition in self._partitions:
      assert partition.size, (
          'Empty partition! Used cutpoints %s to cut a list of len %d.' %
          (cutpoints, len(self._users.keys())))

  def size(self):
    return len(self._users)

  def sample_user(self):
    # Random choice over keys from the current partition of users.
    user_id = self._rng.choice(list(self._partitions[self._active_pool]))
    return self.get_user(user_id)

  def get_user(self, user_id):
    return self._users[user_id]

  def set_active_pool(self, pool):
    if pool > len(self._partitions):
      raise ValueError('Trying to select pool %d but there are only %d pools.' %
                       (pool, len(self._partitions)))
    self._active_pool = pool
