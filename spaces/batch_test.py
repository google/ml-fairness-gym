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
"""Tests for fairness_gym.spaces.batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from spaces import batch
from gym import spaces
import numpy as np
from six.moves import range


class BatchTest(absltest.TestCase):

  def test_batches_are_contained_in_space(self):
    base_space = spaces.Tuple(
        (spaces.Discrete(2),
         spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)))

    batch_space = batch.Batch(base_space)
    # Check that many different batch sizes (including empty) are contained.
    for batch_size in [0, 1, 5, 20]:
      self.assertTrue(
          batch_space.contains([base_space.sample() for _ in range(batch_size)
                               ]))


if __name__ == '__main__':
  absltest.main()
