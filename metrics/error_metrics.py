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
"""Auditors that measure classification errors."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from typing import Any, Callable, Dict, Optional, Text, List, Union
import attr
import core
import params
import numpy as np
from six.moves import zip


@attr.s
class ConfusionMatrix(object):
  """Confusion Matrix object for storing counts of binary corrects/errors."""
  tp = attr.ib(default=0.)  # type: float
  tn = attr.ib(default=0.)  # type: float
  fp = attr.ib(default=0.)  # type: float
  fn = attr.ib(default=0.)  # type: float

  def _convert_pred_truth_to_string(self, prediction, truth):
    """Returns one of {tp, fn, fp, fn}."""
    builder = []
    builder.append('t' if prediction == truth else 'f')
    builder.append('p' if prediction else 'n')
    return ''.join(builder)

  def update(self, prediction, truth, weight=1.0):
    """Update counts with a new prediction, truth pair."""
    lookup = self._convert_pred_truth_to_string(prediction, truth)
    new_val = getattr(self, lookup) + weight
    setattr(self, lookup, new_val)

  def compute_cost(self, cost_matrix):
    """Compute cost with a CostMatrix object."""
    return np.sum(np.multiply(self.as_array(), cost_matrix.as_array()))

  def as_array(self):
    """Convert to a numpy array."""
    return np.array([[self.tn, self.fp],
                     [self.fn, self.tp]])

  def to_jsonable(self):
    return {'___CONFUSION_MATRIX___': attr.asdict(self)}


class AccuracyMetric(core.Metric):
  """Metric that returns a report of an agent's classification accuracy."""

  def __init__(self,
               env,
               numerator_fn,
               denominator_fn = None,
               stratify_fn = None,
               realign_fn = None):
    """Initializes AccuracyMetric.

    Args:
      env: A `core.FairnessEnv`.
      numerator_fn: A function that takes a (state, action) pair and returns
        1 or a list of 1's and 0's if that action is the "correct" action to
        take in that state. This function is allowed to access both observable
        and hidden variables in the state.
      denominator_fn: A function that takes a (state, action) pair and returns
        1 or a list of 1's or 0's if that instance should be considered in
        computing the metric. By default (None), all examples are in scope.
      stratify_fn: A function that takes a (state, action) pair and returns a
        stratum-id to collect together pairs. By default (None), all examples
        are in a single stratum.
      realign_fn: Optional. If not None, defines how to realign hsitory for use
        by a metric.
    """
    super(AccuracyMetric, self).__init__(env, realign_fn)
    self.numerator_fn = numerator_fn
    self.denominator_fn = denominator_fn or (lambda x: 1)
    self.stratify_fn = stratify_fn or (lambda x: 1)

  def measure(self, env):
    """Returns the rate the agent made the correct decision.

    Args:
      env: A `core.FairnessEnv`.

    Returns:
      A dict of correct rates for each stratum. If the denominator is 0 for a
      stratum, that rate is set to None.
    """
    history = self._extract_history(env)
    strata = collections.defaultdict(list)
    for step in history:
      correct_predictions = self.numerator_fn(step)
      stratification = self.stratify_fn(step)
      predictions_to_keep = self.denominator_fn(step)

      if not isinstance(stratification, collections.Sequence):
        stratification = [stratification]
      if not isinstance(correct_predictions, collections.Sequence):
        correct_predictions = [correct_predictions]
      if not isinstance(predictions_to_keep, collections.Sequence):
        predictions_to_keep = [predictions_to_keep] * len(correct_predictions)

      assert (
          len(correct_predictions) == len(predictions_to_keep) ==
          len(stratification)
      ), ('Expected stratification, correct_predictions and predictions_to_keep'
          ' to have the same length, but found %d, %d, %d respectively.' %
          (len(stratification), len(correct_predictions),
           len(predictions_to_keep)))
      for correct_prediction, to_keep, stratum in zip(correct_predictions,
                                                      predictions_to_keep,
                                                      stratification):
        if to_keep:
          strata[stratum].append(correct_prediction)
    for stratum in strata:
      for value in strata[stratum]:
        assert value in (0, 1), ('Found unexpected value %d in stratum %s.' %
                                 (value, stratum))

    return {
        stratum: np.mean(responses) if responses else None
        for stratum, responses in strata.items()  # pytype: disable=bad-return-type
    }


class ConfusionMetric(core.Metric):
  """Metric that returns a group-stratified confusion matrix."""

  def __init__(self,
               env,
               prediction_fn,
               ground_truth_fn,
               stratify_fn = None,
               realign_fn = None):
    """Initializes ConfusionMetric.

    Args:
      env: A `core.FairnessEnv`.
      prediction_fn: A function that takes a (state, action) pair and returns a
        value (int or List) representing the prediction(s).
      ground_truth_fn: A function that takes a (state, action) pair and returns
        a value representing ground truth.
      stratify_fn: A function that takes a (state, action) pair and returns a
        stratum-id to collect together pairs. By default (None), all examples
        are in a single stratum.
      realign_fn: Optional. If not None, defines how to realign hsitory for use
        by a metric.
    """
    super(ConfusionMetric, self).__init__(env, realign_fn)
    self.prediction_fn = prediction_fn
    self.ground_truth_fn = ground_truth_fn
    self.stratify_fn = stratify_fn or (lambda x: 1)

  def measure(self, env):
    """Returns group-wise confusion matrix or a cost matrix measurement.

    Args:
      env: An environment.

    Returns:
      Returns a dict with keys as group-ids and values as confusion matrix for
      that group.
    """
    history = self._extract_history(env)
    confusion = collections.defaultdict(ConfusionMatrix)

    for step in history:
      stratification = self.stratify_fn(step)
      predictions = self.prediction_fn(step)
      ground_truth = self.ground_truth_fn(step)
      # Convert atomic input to sequential input.
      if not isinstance(predictions, collections.Sequence):
        stratification = [stratification]
        predictions = [predictions]
        ground_truth = [ground_truth]
      # Compute confusion.
      assert (len(predictions) == len(ground_truth) == len(stratification)), (
          'Expected stratification, predictions, and ground truth to '
          'have the same length, but found %d, %d, %d respectively.' %
          (len(stratification), len(predictions), len(ground_truth)))
      for strat, pred, truth in zip(stratification, predictions, ground_truth):
        confusion[strat].update(prediction=pred, truth=truth, weight=1.0)
    return confusion


class CostedConfusionMetric(ConfusionMetric):
  """Metric that returns a group-stratified cost."""

  def __init__(self,
               env,
               prediction_fn,
               ground_truth_fn,
               stratify_fn = None,
               realign_fn = None,
               cost_matrix = None):
    """Initializes CostedConfusionMetric.

    Args:
      env: A `core.FairnessEnv`.
      prediction_fn: A function that takes a (state, action) pair and returns a
        value representing the prediction.
      ground_truth_fn: A function that takes a (state, action) pair and returns
        a value representing ground truth.
      stratify_fn: A function that takes a (state, action) pair and returns a
        stratum-id to collect together pairs. By default (None), all examples
        are in a single stratum.
      realign_fn: Optional. If not None, defines how to realign hsitory for use
        by a metric.
      cost_matrix: A dict that has keys ['TP', 'TN', 'FN', 'FP'] which define
        the costs for each cell in the confusion matrix. Default value is None.
    """
    super(CostedConfusionMetric,
          self).__init__(env, prediction_fn, ground_truth_fn, stratify_fn,
                         realign_fn)
    self.cost_matrix = cost_matrix

  def measure(self, env):
    """Returns group-wise confusion matrix or a cost matrix measurement.

    Args:
      env: An environment.

    Returns:
      Returns a dict with keys as group-ids and values as the cost for that
      group.
    """
    confusion = super(CostedConfusionMetric, self).measure(env)
    return {
        stratum: confusion_matrix.compute_cost(self.cost_matrix)
        for stratum, confusion_matrix in confusion.items()
    }


class RecallMetric(ConfusionMetric):
  """Computes recall."""

  def measure(self, env):
    """Returns recall: tp / (tp + fn).

    Args:
      env: An environment.

    Returns:
      Stratified recall.
    """
    result = super(RecallMetric, self).measure(env)
    return {
        stratum: confusion.tp / (confusion.tp + confusion.fn)
        for stratum, confusion in result.items()
    }


class PrecisionMetric(ConfusionMetric):
  """Computes precision."""

  def measure(self, env):
    """Returns recall: tp / (tp + fp).

    Args:
      env: An environment.

    Returns:
      Stratified recall.
    """
    result = super(PrecisionMetric, self).measure(env)
    return {
        stratum: confusion.tp / (confusion.tp + confusion.fp)
        for stratum, confusion in result.items()
    }
