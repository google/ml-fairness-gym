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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import distributions
import numpy as np


class DistributionsTest(absltest.TestCase):

  def test_mixture_returns_components(self):
    my_distribution = distributions.Mixture(
        components=[distributions.Constant((0,)),
                    distributions.Constant((1,))],
        weights=[0.1, 0.9])
    rng = np.random.RandomState(seed=100)
    samples = [my_distribution.sample(rng) for _ in range(1000)]
    self.assertSetEqual(set(samples), {(0,), (1,)})
    self.assertAlmostEqual(np.mean(samples), 0.9, delta=0.1)

  def test_bernoulli_returns_proportionally(self):
    my_distribution = distributions.Bernoulli(p=0.9)
    rng = np.random.RandomState(seed=100)
    samples = [my_distribution.sample(rng) for _ in range(1000)]
    self.assertAlmostEqual(np.mean(samples), 0.9, delta=0.1)

  def test_constant_returns_the_same_thing(self):
    my_distribution = distributions.Constant(mean=(0, 1, 2))
    rng = np.random.RandomState(seed=100)
    unique_samples = {my_distribution.sample(rng) for _ in range(1000)}
    self.assertEqual(unique_samples, {(0, 1, 2)})

  def test_gaussian_has_right_mean_std(self):
    my_distribution = distributions.Gaussian(mean=[0, 0, 1], std=0.1)
    rng = np.random.RandomState(seed=100)
    samples = [my_distribution.sample(rng) for _ in range(1000)]
    self.assertLess(
        np.linalg.norm(np.mean(samples, 0) - np.array([0, 0, 1])), 0.1)
    self.assertLess(
        np.linalg.norm(np.std(samples, 0) - np.array([0.1, 0.1, 0.1])), 0.1)

  def test_improper_distributions_raise_errors(self):
    for p in [-10, -0.9, 1.3]:
      with self.assertRaises(ValueError):
        _ = distributions.Bernoulli(p=p)

    for vec in [
        [0.1, 0.3, 0.5],  # Does not sum to one.
        [0.5, 0.9, -0.4],  # Has negative values.
    ]:
      with self.assertRaises(ValueError):
        _ = distributions.Mixture(
            weights=vec,
            components=[distributions.Constant(mean=(0,))] * len(vec))


if __name__ == '__main__':
  absltest.main()
