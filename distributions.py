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

"""Classes for building distributions."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import logging
import attr
import numpy as np
from typing import Sequence


@attr.s
class Distribution(object):
  """Base distribution class.

  Inheriting classes should fill in the sample method and initialize dim.
  """
  dim = attr.ib(init=False)

  def sample(self, rng):
    raise NotImplementedError


def _check_sum_to_one(instance, attribute, value):
  """Raises ValueError if the value does not sum to one."""
  del instance, attribute  # Unused.
  value = np.array(value)
  if not np.isclose(np.sum(value), 1):
    raise ValueError("Array must sum to one. Got %s." % np.sum(value))


def _check_nonnegative(instance, attribute, value):
  """Raises ValueError if the value elements are negative."""
  del instance, attribute  # Unused.
  value = np.array(value)
  if np.any(value < 0):
    raise ValueError("Array must be nonnegative. Got %s." % value)


def _check_in_zero_one_range(instance, attribute, value):
  """Raises ValueError if value is not in [0, 1]."""
  del instance, attribute  # Unused.
  value = np.array(value)
  if np.any(value < 0) or np.any(value > 1):
    raise ValueError("Value must be in [0, 1]. Got %s." % value)


@attr.s
class Mixture(Distribution):
  """A mixture distribution."""
  components = attr.ib(factory=list)  # type: Sequence[Distribution]
  weights = attr.ib(
      factory=list, validator=[_check_sum_to_one,
                               _check_nonnegative])  # type: Sequence[float]

  def sample(self, rng):
    logging.debug("Sampling from a mixture with %d components. Weights: %s",
                  len(self.components), self.weights)
    component = rng.choice(self.components, p=self.weights)
    return component.sample(rng)

  def __attrs_post_init__(self):
    for component in self.components:
      if component.dim != self.components[0].dim:
        raise ValueError("Components do not have the same dimensionality.")
    self.dim = self.components[0].dim


@attr.s
class Gaussian(Distribution):
  """A Gaussian Distribution."""
  mean = attr.ib()
  std = attr.ib()

  def __attrs_post_init__(self):
    self.dim = len(self.mean)

  def sample(self, rng):
    return rng.normal(self.mean, self.std)


@attr.s
class Bernoulli(Distribution):
  """A Bernoulli Distribution."""

  p = attr.ib(validator=[_check_in_zero_one_range])

  def __attrs_post_init__(self):
    self.dim = 1

  def sample(self, rng):
    return rng.rand() < self.p


@attr.s
class Constant(Distribution):
  """A Constant Distribution."""

  mean = attr.ib()

  def __attrs_post_init__(self):
    self.dim = len(self.mean)

  def sample(self, rng):
    del rng  # Unused.
    return self.mean
