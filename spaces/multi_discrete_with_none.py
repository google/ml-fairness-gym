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
"""A MultiDiscrete space that contains None.

None can be used to represent special actions like 'do nothing this step.'
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from gym.spaces import multi_discrete


class MultiDiscreteWithNone(multi_discrete.MultiDiscrete):
  """A MultiDiscrete space that contains None."""

  def __init__(self, nvec, none_probability=0.5):
    self._none_probability = none_probability
    super(MultiDiscreteWithNone, self).__init__(nvec)

  def contains(self, x):
    if x is None:
      return True
    return super(MultiDiscreteWithNone, self).contains(x)

  def sample(self):
    u = self.np_random.rand()

    if u <= self._none_probability:
      return None

    return super(MultiDiscreteWithNone, self).sample()
