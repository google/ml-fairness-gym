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

# Lint as: python2, python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from spaces import multi_discrete_with_none
from gym.spaces import multi_discrete


class MultiDiscreteWithNoneTest(absltest.TestCase):

  def test_space_contains_none(self):
    """The space should contain None."""
    space = multi_discrete_with_none.MultiDiscreteWithNone([1, 2, 3, 4])
    self.assertTrue(space.contains(None))

  def test_base_and_with_none_agree(self):
    """MultiDiscrete and MultiDiscreteWithNone should agree about non-None."""
    nvec = [4, 4, 4]
    multi_discrete_space = multi_discrete.MultiDiscrete(nvec)
    multi_discrete_with_none_space = (
        multi_discrete_with_none.MultiDiscreteWithNone(nvec))
    for test_vec in (
        [-1, 1, 2],
        [2, 1, 0],
        [1, 2, 3]):
      self.assertEqual(multi_discrete_space.contains(test_vec),
                       multi_discrete_with_none_space.contains(test_vec))

  def test_none_can_be_sampled(self):
    space = multi_discrete_with_none.MultiDiscreteWithNone(
        nvec=[1, 2, 3, 4], none_probability=1)
    self.assertIsNone(space.sample())


if __name__ == '__main__':
  absltest.main()
