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
"""Tests for fairness_gym.agents.threshold_policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import params
from agents import threshold_policies
import numpy as np
from six.moves import range
from six.moves import zip
from sklearn import metrics as sklearn_metrics

COST_MATRIX = params.CostMatrix(tp=1.5, fp=-1.0, fn=-0.3, tn=2.0)


EPSILON = 1e-6


class ThresholdPoliciesTest(absltest.TestCase):

  def test_reward(self):
    np.random.seed(100)

    predictions = np.random.choice(2, 1000, p=[0.1, 0.9])
    labels = np.random.choice(2, 1000)
    fpr, tpr, _ = sklearn_metrics.roc_curve(labels, predictions)

    # These should be len 3 with the first and last values corresponding to end
    # points of the ROC curve. The middle value corresponds to the performance
    # of the binary classifeir.
    self.assertLen(fpr, 3)
    self.assertLen(tpr, 3)

    fpr = fpr[1]
    tpr = tpr[1]

    num_positive = sum(labels)
    num_negative = len(labels) - num_positive

    reward_a = threshold_policies._reward(fpr, tpr, num_positive, num_negative,
                                          COST_MATRIX)

    reward_b = (sklearn_metrics.confusion_matrix(labels, predictions) *
                COST_MATRIX.as_array()).sum()
    self.assertAlmostEqual(reward_a, reward_b)

  def test_reward_at_tpr(self):
    np.random.seed(100)

    predictions = np.random.rand(1000)
    labels = np.random.choice(2, 1000)

    num_positive = sum(labels)
    num_negative = len(labels) - num_positive

    roc = sklearn_metrics.roc_curve(labels, predictions)

    _, tpr, _ = roc
    # Use epsilon to ensure that there is no exact match to a tpr_target.
    tpr_target = np.random.choice(tpr) - EPSILON
    reward_a = threshold_policies._reward_at_tpr(roc, tpr_target, num_positive,
                                                 num_negative, COST_MATRIX)

    # Find a threshold that gives tpr > tpr_target.
    target_threshold = 0
    for _, tpr, thresh in zip(*roc):
      if tpr > tpr_target:
        target_threshold = thresh
        # Check that tpr is close to tpr_target.
        assert np.abs(tpr - tpr_target) < 1.5*EPSILON
        assert tpr_target not in roc[1]
        break

    binarized_predictions = [
        prediction >= target_threshold for prediction in predictions
    ]

    self.assertAlmostEqual(
        tpr_target,
        sklearn_metrics.recall_score(labels, binarized_predictions),
        places=5)

    reward_b = (
        sklearn_metrics.confusion_matrix(labels, binarized_predictions) *
        COST_MATRIX.as_array()).sum()

    self.assertAlmostEqual(reward_a, reward_b)

  def test_equality_of_opportunity_holds(self):
    np.random.seed(100)
    group_a_predictions = np.random.rand(100000)
    group_a_labels = np.random.choice([0, 1], p=[0.5, 0.5], size=100000)

    group_b_predictions = np.random.normal(size=100000)
    group_b_labels = np.random.choice([0, 1], p=[0.2, 0.8], size=100000)

    thresholds = threshold_policies.equality_of_opportunity_thresholds(
        group_predictions={
            'a': group_a_predictions,
            'b': group_b_predictions
        },
        group_labels={
            'a': group_a_labels,
            'b': group_b_labels
        },
        group_weights=None,
        cost_matrix=COST_MATRIX)

    group_a_recall = ((group_a_predictions > thresholds['a']) *
                      group_a_labels).sum() / group_a_labels.sum()
    group_b_recall = ((group_b_predictions > thresholds['b']) *
                      group_b_labels).sum() / group_b_labels.sum()

    self.assertLess(np.abs(group_a_recall - group_b_recall), 1e-3)

  def test_reward_is_maximized(self):
    np.random.seed(100)
    predictions = np.random.rand(10000)
    labels = np.random.choice([0, 1], p=[0.5, 0.5], size=10000)

    # With only one group, this should return a reward-maximizing threshold.
    thresholds = threshold_policies.equality_of_opportunity_thresholds(
        {'a': predictions}, {'a': labels},
        group_weights=None,
        cost_matrix=COST_MATRIX)

    optimal_threshold = thresholds['a']
    confusion = sklearn_metrics.confusion_matrix(
        labels, predictions > optimal_threshold)
    optimal_reward = (confusion * COST_MATRIX.as_array()).sum()

    for _ in range(20):
      # Optimal threshold should be better than or equal to all challengers.
      challenge_threshold = np.random.rand()
      confusion = sklearn_metrics.confusion_matrix(
          labels, predictions > challenge_threshold)
      reward = (confusion * COST_MATRIX.as_array()).sum()
      self.assertLessEqual(reward, optimal_reward)

  def test_weighted_single_threshold(self):
    # Weighted labels/predictions are perfectly calibrated.
    # With a vanilla cost matrix, this should result in threshold > 0.5
    predictions = np.array([0.25, 0.25, 0.5, 0.5, 0.75, 0.75])
    labels = np.array([0, 1, 0, 1, 0, 1])
    weights = np.array([3., 1., 1., 1., 1., 3.])
    weights = weights / sum(weights)

    vanilla_cost_matrix = params.CostMatrix(tn=1., fp=-1., fn=-1., tp=1.)

    weighted_threshold = threshold_policies.single_threshold(
        predictions, labels, weights, vanilla_cost_matrix)
    self.assertEqual(weighted_threshold, 0.75)


if __name__ == '__main__':
  absltest.main()
