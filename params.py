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

"""Helper objects to store parameters."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import attr
import numpy as np
from typing import List


@attr.s
class CostMatrix(object):
  tp = attr.ib()  #  type: float
  fp = attr.ib()  #  type: float
  fn = attr.ib()  #  type: float
  tn = attr.ib()  #  type: float

  def as_array(self):
    return np.array([[self.tn, self.fp], [self.fn, self.tp]])


@attr.s
class MinMax(object):
  """Defines min, max param values."""
  min = attr.ib(default=0)  # type: float
  max = attr.ib(default=1)  # type : float


@attr.s
class PosNeg(object):
  """Defines positive, negative change param values."""
  pos = attr.ib()  # type: float
  neg = attr.ib()  # type : float


@attr.s
class GMM(object):
  """Defines GMM Params."""
  mix_weight = attr.ib()  # type: List[float]
  mu = attr.ib()  # type : List[float]
  sigma = attr.ib()  # type : List[float]


@attr.s
class BoundedGaussian(object):
  """Defines Gaussian Params with min max bounds."""
  mu = attr.ib()  # type : float
  sigma = attr.ib()  # type : float
  min = attr.ib()  # type : float
  max = attr.ib()  # type : float
