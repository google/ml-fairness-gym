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

"""Metrics for the infectious disease environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import core
from environments import infectious_disease
import gin
from typing import Callable, Optional


def num_in_health_state(step, health_state):
  """Returns the number of people in health_state during a step."""
  state = step.state  # type: infectious_disease.State  # pytype: disable=annotation-type-mismatch
  return sum(s == health_state for s in state.health_states)


@gin.configurable
class PersonStepsInHealthState(core.Metric):
  """Gives the total number of person-steps spent in a health state."""

  def __init__(
      self,
      environment,
      health_state,
      realign_fn = None):
    super(PersonStepsInHealthState, self).__init__(environment, realign_fn)
    self.health_state = health_state

  def measure(self, env):
    """Measures the total number of person-steps spent in a healthy state."""
    history = self._extract_history(env)

    num_person_steps = 0
    for step in history:
      num_person_steps += num_in_health_state(step, self.health_state)
    return num_person_steps


class DiseasePrevalence(core.Metric):
  """Gives the disease prevalence in the most recent step."""

  def measure(self, env):
    """Measures disease prevelance at the most recent point in history."""
    history = self._extract_history(env)
    population_size = env.state.population_graph.number_of_nodes()
    infectious_state = env.initial_params.infectious_index

    return (
        float(num_in_health_state(history[-1], infectious_state))
        / population_size)
