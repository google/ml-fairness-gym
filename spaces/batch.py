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

# Lint as: python2, python3
"""Space that contains a variable-sized batch of observations."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Any, Iterable, List
import gym
from six.moves import range


class Batch(gym.Space):
  """A batch of samples from a base observation space.

  Example usage:
  self.observation_space = Batch(Tuple((Discrete(2), Discrete(3))))
  """

  def __init__(self, space):
    """Initialize Batch space.

    Args:
      space: A gym.Space that contains each individual observation in the batch.
    """
    self.space = space
    gym.Space.__init__(self, None, None)

  def sample(self):
    return [self.space.sample() for _ in range(10)]

  def contains(self, batch):
    return all(self.space.contains(element) for element in batch)

  def __repr__(self):
    return "Batch(" + self.space + ")"

  def to_jsonable(self, sample_n):
    return [self.space.to_jsonable(sample) for sample in sample_n]

  def from_jsonable(self, sample_n):
    return [self.space.from_jsonable(sample) for sample in sample_n]
