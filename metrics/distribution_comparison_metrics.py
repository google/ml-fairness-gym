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
"""Fairness metric that compares distribution of actions to state variable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Optional, Text, Tuple
import core
import numpy as np
from six.moves import zip


class DistributionComparisonMetric(core.Metric):
  """Compare distribution of actions to distribution of some state variable.

  This metric is designed to measure environments whose actions are allocation
  actions, where values are allocated across different bins, represented by an
  array.
  """

  def __init__(self,
               environment,
               state_variable,
               window = 100,
               realign_fn = None):
    """Initializes a DistributionComparisonMetric.

    Args:
      environment: A 'core.Environment' object to measure.
      state_variable: A string defining a variable in environment.state to be
        measured.
      window: An integer of how far back in the history to include in the
        measurement.
      realign_fn: Optional. If not None, defines how to realign hsitory for use
        by the metric.
    """
    self.window = window
    self.state_variable = state_variable
    super(DistributionComparisonMetric, self).__init__(environment, realign_fn)

  def measure(self,
              env):
    """Returns a measurement of distributions for given enviornment.

    Audit consists of the distribution of actions and the distribution of the
    state variable defined by self.state_variable.

    Assumes state_variable and actions in history are both 1D arrays. The
    distribution of the values over the indices of the arrays is compared.

    Args:
      env: A 'core.Environment' object to measure.

    Returns:
      Tuple of arrays and a scalar where the first array is the normalized
      distribution over the window of the state variable. The second array is
      the normalized distribution of the actions over the window. Third is a
      scalar representing the distance between the arrays.
    """
    history = self._extract_history(env)

    states, actions = list(zip(*history[-self.window:]))
    relevant_state_items = [
        getattr(state, self.state_variable) for state in states
    ]

    if not np.all([np.array(action).ndim == 1 for action in actions]):
      raise ValueError('At least one action is not dimension 1 as required.')
    if not np.all([
        np.array(state_item).ndim == 1 for state_item in relevant_state_items
    ]):
      raise ValueError(
          'At least one instance of State variable %s is not of dimension 1 as required.'
          % self.state_variable)

    action_distribution = np.sum(actions, axis=0) / float(self.window)
    action_distribution_normalized = action_distribution / np.sum(
        action_distribution)

    state_distribution = np.sum(
        relevant_state_items, axis=0) / float(self.window)
    state_distribution_normalized = state_distribution / np.sum(
        state_distribution)

    distribution_distance = np.linalg.norm(state_distribution_normalized -
                                           action_distribution_normalized)

    return (state_distribution_normalized, action_distribution_normalized,
            distribution_distance)
