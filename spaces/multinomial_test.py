# coding=utf-8
# Copyright 2019 The ML Fairness Gym Authors.
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
"""Tests for multinomial_spaces.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from spaces import multinomial
import numpy as np
from six.moves import range


class MultinomialTest(absltest.TestCase):

  def setUp(self):
    self.n = 15  # number of trials
    self.k = 6  # number of categories
    self.multinomial_space = multinomial.Multinomial(self.k, self.n)
    self.multinomial_space.seed(0)
    super(MultinomialTest, self).setUp()

  def test_sample_sum(self):
    n_trials = 100
    samples = [self.multinomial_space.sample() for _ in range(n_trials)]
    sums_to_n = [np.sum(sample) == self.n for sample in samples]
    self.assertTrue(np.all(sums_to_n))

  def test_sample_distribution(self):
    n_trials = 100
    samples = [self.multinomial_space.sample() for _ in range(n_trials)]
    # check roughly uniform distribution by checking means for each category
    # are within 3*std dev of the expected mean
    expected_mean = float(self.n) / self.k
    means = np.mean(samples, axis=0)
    std = np.std(means)
    near_mean = np.asarray(
        [np.abs(mean - expected_mean) < 3.0 * std for mean in means])
    self.assertTrue(np.all(near_mean))

  def test_contains_correct_n_in_vector(self):
    # check a vector is contained even if it has n as one of its values.
    n = 1  # number of trials
    k = 2  # number of categories
    multinomial_space = multinomial.Multinomial(k, n)
    is_contained_vector = np.asarray([1, 0], dtype=np.uint32)
    self.assertTrue(multinomial_space.contains(is_contained_vector))

  def test_contains_correct(self):
    is_contained_vector = np.asarray([2, 3, 2, 3, 3, 2], dtype=np.uint32)
    self.assertTrue(self.multinomial_space.contains(is_contained_vector))

  def test_contains_incorrect_length(self):
    # check vector with incorrect length is not contained
    not_contained_vector = np.asarray([3, 3, 3, 3, 3], dtype=np.uint32)
    self.assertFalse(self.multinomial_space.contains(not_contained_vector))

  def test_contains_incorrect_sum(self):
    # check vector with incorrect sum is not contained
    not_contained_vector = np.asarray([3, 3, 3, 3, 3, 3], dtype=np.uint32)
    self.assertFalse(self.multinomial_space.contains(not_contained_vector))

  def test_contains_incorrect_dtype(self):
    # check vector with wrong dtype is not contained
    not_contained_vector = np.asarray([2.0, 3.0, 2.0, 3.0, 3.5, 1.5])
    self.assertFalse(self.multinomial_space.contains(not_contained_vector))

  def test_contains_samples(self):
    n_trials = 100
    samples = [self.multinomial_space.sample() for _ in range(n_trials)]
    contains_samples = [
        self.multinomial_space.contains(sample) for sample in samples
    ]
    self.assertTrue(np.all(contains_samples))


if __name__ == '__main__':
  absltest.main()
