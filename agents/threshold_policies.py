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

import attr
import numpy as np
import scipy.optimize
from six.moves import zip
from sklearn import metrics as sklearn_metrics


class ThresholdPolicy(enum.Enum):
  SINGLE_THRESHOLD = "single_threshold"
  MAXIMIZE_REWARD = "maximize_reward"
  EQUALIZE_OPPORTUNITY = "equalize_opportunity"


@attr.s
class RandomizedThreshold(object):
  """Represents a distribution over decision thresholds."""
  values = attr.ib(factory=lambda: [0.])
  weights = attr.ib(factory=lambda: [1.])
  rng = attr.ib(factory=np.random.RandomState)

  def smoothed_value(self):
    # If one weight is small, this is probably an optimization artifact.
    # Snap to a single threshold.
    if len(self.weights) == 2 and min(self.weights) < 1e-4:
      return self.values[np.argmax(self.weights)]
    return np.dot(self.weights, self.values)

  def sample(self):
    return self.rng.choice(self.values, p=self.weights)

  def iteritems(self):
    return zip(self.weights, self.values)


def _threshold_from_tpr(roc, tpr_target, rng):
  """Returns a `RandomizedThreshold` that achieves `tpr_target`.

  For an arbitrary value of tpr_target in [0, 1], there may not be a single
  threshold that achieves that tpr_value on our data. In this case, we
  interpolate between the two closest achievable points on the discrete ROC
  curve.

  See e.g., Theorem 1 of Scott et al (1998)
  "Maximum realisable performance: a principled method for enhancing
  performance by using multiple classifiers in variable cost problem domains"
  http://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/Scott_tr320.pdf

  Args:
    roc: A tuple (fpr, tpr, thresholds) as returned by sklearn's roc_curve
      function.
    tpr_target: A float between [0, 1], the target value of TPR that we would
      like to achieve.
    rng: A `np.RandomState` object that will be used in the returned
      RandomizedThreshold.

  Return:
    A RandomizedThreshold that achieves the target TPR value.
  """
  _, tpr_list, thresh_list = roc
  idx = bisect.bisect_left(tpr_list, tpr_target)

  # TPR target is larger than any of the TPR values in the list. In this case,
  # take the highest threshold possible.
  if idx == len(tpr_list):
    return RandomizedThreshold(weights=[1], values=[thresh_list[-1]], rng=rng)

  # TPR target is exactly achievable by an existing threshold. In this case,
  # do not randomize between two different thresholds. Use a single threshold
  # with probability 1.
  if tpr_list[idx] == tpr_target:
    return RandomizedThreshold(weights=[1], values=[thresh_list[idx]], rng=rng)

  # Interpolate between adjacent thresholds.
  # TODO() May be better to search over all pairs using a convex
  # hull formulation.
  alpha = _interpolate(x=tpr_target, low=tpr_list[idx - 1], high=tpr_list[idx])
  return RandomizedThreshold(
      weights=[alpha, 1 - alpha],
      values=[thresh_list[idx - 1], thresh_list[idx]],
      rng=rng)


def _interpolate(x, low, high):
  """returns a such that a*low + (1-a)*high = x."""
  assert low <= x <= high, ("x is not between [low, high]: Expected %s <= %s <="
                            " %s") % (low, x, high)
  alpha = 1 - ((x - low) / (high - low))
  assert np.abs(alpha * low + (1 - alpha) * high - x) < 1e-6
  return alpha


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
  threshold = equality_of_opportunity_thresholds({"dummy": predictions},
                                                 {"dummy": labels},
                                                 {"dummy": weights},
                                                 cost_matrix)["dummy"]
  return threshold.smoothed_value()


def equality_of_opportunity_thresholds(group_predictions,
                                       group_labels,
                                       group_weights,
                                       cost_matrix,
                                       rng=None):
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
    rng: A `np.random.RandomState`.

  Returns:
    A dict mapping from group identifiers to thresholds such that recall is
    equal for all groups.

  Raises:
    ValueError if the keys of group_predictions and group_labels are not the
      same.
  """

  if set(group_predictions.keys()) != set(group_labels.keys()):
    raise ValueError("group_predictions and group_labels have mismatched keys.")

  if rng is None:
    rng = np.random.RandomState()

  groups = sorted(group_predictions.keys())
  roc = {}

  if group_weights is None:
    group_weights = {}

  for group in groups:
    if group not in group_weights or group_weights[group] is None:
      # If weights is unspecified, use equal weights.
      group_weights[group] = [1 for _ in group_labels[group]]

    assert len(group_labels[group]) == len(group_weights[group]) == len(
        group_predictions[group])

    fprs, tprs, thresholds = sklearn_metrics.roc_curve(
        y_true=group_labels[group],
        y_score=group_predictions[group],
        sample_weight=group_weights[group])

    roc[group] = (fprs, np.nan_to_num(tprs), thresholds)
    # roc[group] = (np.nan_to_num(fprs), np.nan_to_num(tprs), thresholds)

  def negative_reward(tpr_target):
    """Returns negative reward suitable for optimization by minimization."""

    my_reward = 0
    for group in groups:
      weights_ = []
      predictions_ = []
      labels_ = []
      for thresh_prob, threshold in _threshold_from_tpr(
          roc[group], tpr_target, rng=rng).iteritems():
        labels_.extend(group_labels[group])
        for weight, prediction in zip(group_weights[group],
                                      group_predictions[group]):
          weights_.append(weight * thresh_prob)
          predictions_.append(prediction >= threshold)
      confusion_matrix = sklearn_metrics.confusion_matrix(
          labels_, predictions_, sample_weight=weights_)

      my_reward += np.multiply(confusion_matrix, cost_matrix.as_array()).sum()
    return -my_reward

  opt = scipy.optimize.minimize_scalar(
      negative_reward,
      bounds=[0, 1],
      method="bounded",
      options={"maxiter": 100})
  return ({
      group: _threshold_from_tpr(roc[group], opt.x, rng=rng) for group in groups
  })
