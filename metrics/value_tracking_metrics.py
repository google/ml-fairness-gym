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
"""Metrics that apply a group-wise aggregation function to state variable(s)."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from typing import Callable, Dict, Optional, Text, Any
import core
import gin
import numpy as np
from six.moves import zip


@gin.configurable
class SummingMetric(core.Metric):
  """Metric that sums a state variable.

  A simple sum-specific implementation.
  """

  def __init__(self,
               env,
               selection_fn,
               realign_fn = None):
    """Initialize SummingMetric.

    Args:
      env: A `core.FairnessEnv`.
      selection_fn: Returns a state variable which will be summed. Can be a
        float or np.ndarray.
      realign_fn: Optional. If not None, defines how to realign history for use
        by a metric.
    """
    super(SummingMetric, self).__init__(env, realign_fn)
    self.selection_fn = selection_fn

  def measure(self, env):
    """Returns the sum of the state variable from the selection_fn.

    Args:
     env: A `core.FairnessEnv`.

    Returns:
     np.ndarray of the sum of the state variable returned by self.selection_fn.

    """
    history = self._extract_history(env)
    return np.sum([self.selection_fn(history_item) for history_item in history],
                  0)


@gin.configurable
class AggregatorMetric(core.Metric):
  """Metric that modifies and aggregates an env state variable.

  This metric can be used to calculate any value that needs to be aggregated in
  sum or mean over the entire history by applying some modifications to the
  env state variable based on group-id.

  For instance, to calculate costs for each group, we might have different cost
  functions per group. Thus the selection_fn would return the state variable
  that is used to calculate cost and the modifier function will return the group
  specific cost function applied to this state variable, which will then be
  aggregated over entire history for each group.
  """

  def __init__(
      self,
      env,
      selection_fn,
      stratify_fn = None,
      modifier_fn = None,
      realign_fn = None,
      calc_mean = False):
    """Initializes the metric.

    Args:
      env: A `core.FairnessEnv`.
      selection_fn: Returns a state variable which needs to be modified and
        aggregated.
      stratify_fn: A function that takes a (state, action) pair and returns a
        stratum-id to collect together pairs. By default (None), all examples
        are in a single stratum.
      modifier_fn: A function that takes the state variable key returned by the
        selection_fn, the stratum-id returned by stratify_fn, and an environment
        instance, and applies a transformation to selected variable based on
        stratum-id and env params.
      realign_fn: Optional. If not None, defines how to realign history for use
        by a metric.
      calc_mean: Bool. If use mean aggregator else use sum aggregator.
    """
    super(AggregatorMetric, self).__init__(env, realign_fn)
    self.selection_fn = selection_fn
    self.stratify_fn = stratify_fn or (lambda x: 1)
    self.modifier_fn = modifier_fn or (lambda x, y, z: x)
    self.calc_mean = calc_mean

  def measure(self, env):
    """Returns an aggregated value per group.

    Args:
     env: A `core.FairnessEnv`.

    Returns:
     A dict with keys as strata (could be group-ids) and values as scalar
     values obtained by summing over value returned by modifier fn. each step.
    """
    sum_aggregate_result = collections.defaultdict(int)
    group_count_result = collections.defaultdict(int)
    history = self._extract_history(env)
    for step in history:
      stratification = self.stratify_fn(step)
      selections = self.selection_fn(step)
      if not isinstance(stratification, collections.Sequence):
        stratification = [stratification]
        selections = [selections]

      for strata, selection in zip(stratification, selections):
        sum_aggregate_result[strata] += self.modifier_fn(selection, strata, env)
        group_count_result[strata] += 1

    if self.calc_mean:
      return {
          strata: (sum_aggregate_result[strata] / group_count_result[strata])
          for strata in sum_aggregate_result.keys()
      }  # Note: group_count_result[strata] is always > 0.
    return sum_aggregate_result


@gin.configurable
class ValueChange(core.Metric):
  """Metric that returns how much a value has changed over the experiment."""

  def __init__(self,
               env,
               state_var,
               normalize_by_steps = True,
               realign_fn = None):
    """Initializes the ValueChange metric.

    Args:
      env: A `core.FairnessEnv`.
      state_var: string name of a state variable to track.
      normalize_by_steps: Whether to divide by number of steps to get an average
        change.
      realign_fn: Optional. If not None, defines how to realign history for use
        by a metric.
    """
    super(ValueChange, self).__init__(env, realign_fn)
    self.state_var = state_var
    self.normalize_by_steps = normalize_by_steps

  def measure(self, env):
    """Returns the value difference between the first and last history item."""
    history = self._extract_history(env)
    initial_state = history[0].state
    final_state = history[-1].state
    delta = (
        getattr(final_state, self.state_var) -
        getattr(initial_state, self.state_var))
    if self.normalize_by_steps:
      delta /= (len(history) - 1)
    return delta


@gin.configurable
class FinalValueMetric(core.Metric):
  """Metric that returns the final value of a `State` variable."""

  def __init__(self,
               env,
               state_var,
               realign_fn = None):
    """Initialize the metric.

    Args:
      env: A `core.FairnessEnv`.
      state_var: Variable name whose final value needs to be reported.
      realign_fn: Optional. If not None, defines how to realign history for use
        by a metric.
    """
    super(FinalValueMetric, self).__init__(env, realign_fn)
    self.state_var = state_var

  def measure(self, env):
    """Returns the final value of a state variable.

    Args:
     env: A `core.FairnessEnv`.

    Returns:
     The final value of a state variable.
    """
    history = self._extract_history(env)
    final_state = history[-1].state
    return getattr(final_state, self.state_var)
