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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import rewards
import numpy as np


class RewardsTest(absltest.TestCase):

  def test_scalar_delta_reward(self):
    reward = rewards.ScalarDeltaReward('x', baseline=0)
    # Variable goes up from baseline.
    self.assertEqual(reward({'x': 1}), 1)
    # Variable goes up again.
    self.assertEqual(reward({'x': 5}), 4)

  def test_invalid_vector_in_scalar_delta_reward(self):
    observation = {'x': np.array([1, 1])}
    reward = rewards.ScalarDeltaReward('x', baseline=0)
    with self.assertRaises(TypeError):
      reward(observation)

  def test_binarized_scalar_delta_reward(self):
    observation = {'x': 1}
    reward = rewards.BinarizedScalarDeltaReward('x', baseline=0)
    # Variable goes up from baseline.
    self.assertEqual(reward(observation), 1)
    # Variable goes up.
    observation['x'] = 5
    self.assertEqual(reward(observation), 1)
    # Variable stays constant.
    self.assertIsNone(reward(observation))
    # Variable goes down.
    observation['x'] = 4
    self.assertEqual(reward(observation), 0)

  def test_vector_sum_reward(self):
    reward = rewards.VectorSumReward('x')
    self.assertEqual(reward({'x': [1, 5, 2, 4, 7]}), 19)


if __name__ == '__main__':
  absltest.main()
