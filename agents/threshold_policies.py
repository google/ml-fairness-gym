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
"""Helper functions for finding appropriate thresholds.

Many agents use classifiers to calculate continuous scores and then use a
threshold to transform those scores into decisions that optimize some reward.
The helper functions in this module are intended to aid with choosing those
thresholds.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import enum

import numpy as np
from six.moves import zip
from sklearn import metrics as sklearn_metrics


class ThresholdPolicy(enum.Enum):
  SINGLE_THRESHOLD = "single_threshold"
  MAXIMIZE_REWARD = "maximize_reward"
  EQUALIZE_OPPORTUNITY = "equalize_opportunity"


def _reward(fpr, tpr, num_positive, num_negative, cost_matrix):
  """Reward at a (false positive rate, true positive rate) pair.

  Args:
    fpr: False positive rate.
    tpr: True positive rate.
    num_positive: Number of positively labeled examples.
    num_negative: Number of negatively labeled examples.
    cost_matrix: A CostMatrix
  Returns:
    A scalar reward.
  """
  tn = (1 - fpr) * num_negative
  tp = tpr * num_positive
  fn = (1 - tpr) * num_positive
  fp = fpr * num_negative
  confusion_counts = np.array([tn, fp, fn, tp]).reshape((2, 2))
  # Elementwise multiplication.
  return np.multiply(confusion_counts, cost_matrix.as_array()).sum()


def _reward_at_tpr(roc, tpr_target, num_positive, num_negative,
                   cost_matrix):
  """Calculate the reward at a fixed true positive rate.

  Args:
    roc: A tuple of lists (fprs, tprs, thresholds) that are the output of
      sklearn_metrics.roc_curve.
    tpr_target: True positive rate target to match.
    num_positive: Number of positively labeled examples.
    num_negative: Number of negatively labeled examples.
    cost_matrix: A CostMatrix.
  Returns:
    A scalar reward.
  """
  fpr, tpr, _ = roc
  idx = bisect.bisect_left(tpr, tpr_target)
  return _reward(fpr[idx], tpr[idx], num_positive, num_negative, cost_matrix)


def single_threshold(predictions, labels, weights, cost_matrix):
  """Finds a single threshold that maximizes reward.

  Args:
    predictions: A list of float predictions.
    labels: A list of binary labels.
    weights: A list of instance weights.
    cost_matrix: A CostMatrix.


  Returns:
    A single threshold that maximizes reward.
  """
  return equality_of_opportunity_thresholds({"dummy": predictions},
                                            {"dummy": labels},
                                            {"dummy": weights},
                                            cost_matrix)["dummy"]


def equality_of_opportunity_thresholds(group_predictions, group_labels,
                                       group_weights, cost_matrix):
  """Finds thresholds that equalize opportunity while maximizing reward.

  Using the definition from "Equality of Opportunity in Supervised Learning" by
  Hardt et al., equality of opportunity constraints require that the classifier
  have equal true-positive rate for all groups and can be enforced as a
  post-processing step on a threshold-based binary classifier by creating
  group-specific thresholds.

  Since there are many different thresholds where equality of opportunity
  constraints can hold, we simultaneously maximize reward described by a reward
  matrix.

  Args:
    group_predictions: A dict mapping from group identifiers to predictions for
      instances from that group.
    group_labels: A dict mapping from group identifiers to labels for instances
      from that group.
    group_weights: A dict mapping from group identifiers to weights for
      instances from that group.
    cost_matrix: A CostMatrix.


  Returns:
    A dict mapping from group identifiers to thresholds such that recall is
    equal for all groups.

  Raises:
    ValueError if the keys of group_predictions and group_labels are not the
      same.
  """

  if set(group_predictions.keys()) != set(group_labels.keys()):
    raise ValueError("group_predictions and group_labels have mismatched keys.")

  groups = set(group_predictions.keys())
  roc = {}
  thresholds = {}

  # num_positive, num_negative store the number of examples with ground_truth
  # positive, negative labels per group.
  num_positive = {}
  num_negative = {}
  feasible_tpr = []
  for group in groups:
    labels = group_labels[group]

    if group_weights is None or group_weights[group] is None:
      # If weights is unspecified, use equal weights.
      weights = [1 for _ in labels]
    else:
      weights = group_weights[group]

    num_positive[group] = sum(
        weight for weight, label in zip(weights, labels) if label)
    num_negative[group] = sum(
        weight for weight, label in zip(weights, labels) if not label)
    predictions = group_predictions[group]
    fpr, tpr, thresh = sklearn_metrics.roc_curve(labels, predictions,
                                                 sample_weight=weights)
    roc[group] = (fpr, tpr, thresh)
    feasible_tpr.extend(tpr)

  # Compute a reward for each potential tpr value.
  reward = {}
  for tpr_target in feasible_tpr:
    reward[tpr_target] = sum(
        _reward_at_tpr(roc[group], tpr_target, num_positive[group],
                       num_negative[group], cost_matrix)
        for group in groups)

  best_tpr, _ = max(list(reward.items()), key=lambda x: x[1])

  for group, (_, tpr, thresh) in roc.items():
    idx = bisect.bisect_left(tpr, best_tpr)
    thresholds[group] = thresh[idx]
  return thresholds
