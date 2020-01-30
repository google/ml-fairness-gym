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

"""A space class for the gym for representing multinomial action spaces.

Allows for a space encompasses counts for k discrete categories that sum to n.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from gym import spaces
import numpy as np


class Multinomial(spaces.MultiDiscrete):
  """Class representing a multinomial space."""

  def __init__(self, veclength, n):
    """Initialization function.

    Args:
      veclength: length of vector for the counts of categorical variables
        (number of catgeories).
      n: limit on sum of vector counts
    """
    # +1 because this represents a non-inclusive upper bound.
    nvec = [n+1] * veclength
    self.n = n
    super(Multinomial, self).__init__(nvec)

  def sample(self):
    return self.np_random.multinomial(
        self.n, [1 / float(self.nvec.size)] * self.nvec.size, 1).astype(
            self.dtype).flatten()

  def contains(self, x):
    return (len(x) == len(self.nvec)) and (np.sum(x) == self.n) and super(
        Multinomial, self).contains(x) and x.dtype.kind in "ui"

  def __repr__(self):
    return "Multinomial({},{})".format(self.nvec, self.n)
