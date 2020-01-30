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

# Lint as: python2, python3
"""Experiment class for running college admission experiments."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy

import core
import gin
import more_itertools
import numpy as np


@gin.configurable
def realign_history(history):
  """"Realigns history so as to be compatible with auditors.

  Since the true applicants groups, unmanipulated test scores and true_eligible
  are generated before the agent's action, they are in the previous state, so we
  push them one step ahead in history and ignore the first step.

  Args:
    history: A list of tuples of state, action pairs.

  Returns:
    A realigned history with changed state, action pairs.
  """
  realign_variables = [
      'test_scores_x', 'applicant_groups', 'true_eligible', 'params'
  ]
  realigned_history = []
  for (state, _), (next_state,
                   next_action) in more_itertools.pairwise(history):
    new_history_point = core.HistoryItem(
        state=copy.deepcopy(next_state), action=copy.deepcopy(next_action))
    for variable in realign_variables:
      setattr(new_history_point.state, variable, getattr(state, variable))
    realigned_history.append(new_history_point)
  return realigned_history


@gin.configurable
def stratify_by_group(step):
  return list(step.state.applicant_groups)


@gin.configurable
def accuracy_nr_fn(step):
  """Returns if predictions are correct."""
  return [
      pred == groundtruth for pred, groundtruth in zip(
          step.state.selected_applicants, step.state.true_eligible)
  ]


@gin.configurable
def selection_fn_social_burden_eligible_auditor(step):
  """This returns individual burden of only eligible candidates."""
  return np.array(step.state.true_eligible) * np.array(
      step.state.individual_burden)


@gin.configurable
def stratify_to_one_group(step):
  return [1 for _ in range(step.state.params.num_applicants)]
