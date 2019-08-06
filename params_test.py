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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import params
import numpy as np


class ParamsTest(absltest.TestCase):

  def test_as_array(self):
    array = params.CostMatrix(tp=1, tn=10, fp=3, fn=100).as_array()
    self.assertAlmostEqual(
        np.linalg.norm(array - np.array([[10, 3],
                                         [100, 1]])), 0)


if __name__ == '__main__':
  absltest.main()
