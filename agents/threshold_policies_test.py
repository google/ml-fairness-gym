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
"""Tests for fairness_gym.agents.threshold_policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import params
from agents import threshold_policies
import numpy as np
from six.moves import range
from sklearn import metrics as sklearn_metrics

COST_MATRIX = params.CostMatrix(tp=1.5, fp=-1.0, fn=-0.3, tn=2.0)


EPSILON = 1e-6


class ThresholdPoliciesTest(absltest.TestCase):

  def test_equality_of_opportunity_holds(self):
    rng = np.random.RandomState(100)
    group_a_predictions = rng.rand(100000)
    group_a_labels = rng.choice([0, 1], p=[0.5, 0.5], size=100000)

    group_b_predictions = rng.normal(size=100000)
    group_b_labels = rng.choice([0, 1], p=[0.2, 0.8], size=100000)

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
        cost_matrix=COST_MATRIX,
        rng=rng)

    group_a_recall = ((group_a_predictions > thresholds['a'].sample()) *
                      group_a_labels).sum() / group_a_labels.sum()
    group_b_recall = ((group_b_predictions > thresholds['b'].sample()) *
                      group_b_labels).sum() / group_b_labels.sum()

    self.assertLess(np.abs(group_a_recall - group_b_recall), 1e-3)

  def test_reward_is_maximized(self):
    rng = np.random.RandomState(100)
    predictions = rng.rand(10000)
    labels = rng.choice([0, 1], p=[0.5, 0.5], size=10000)

    # With only one group, this should return a reward-maximizing threshold.
    thresholds = threshold_policies.equality_of_opportunity_thresholds(
        {'a': predictions}, {'a': labels},
        group_weights=None,
        cost_matrix=COST_MATRIX,
        rng=rng)

    optimal_threshold = thresholds['a']
    confusion = sklearn_metrics.confusion_matrix(
        labels, predictions > optimal_threshold.sample())
    optimal_reward = (confusion * COST_MATRIX.as_array()).sum()

    for _ in range(20):
      # Optimal threshold should be better than or equal to all challengers.
      challenge_threshold = rng.rand()
      confusion = sklearn_metrics.confusion_matrix(
          labels, predictions >= challenge_threshold)
      reward = (confusion * COST_MATRIX.as_array()).sum()
      self.assertLessEqual(reward, optimal_reward)

  def test_weighted_single_threshold(self):
    # With a vanilla cost matrix, this should result in accepting
    # only predictions >= 0.75
    predictions = np.array([0.25, 0.25, 0.33, 0.33, 0.75, 0.75])
    labels = np.array([0, 1, 0, 1, 0, 1])
    weights = np.array([3., 1., 2., 1., 1., 3.])
    weights = weights / sum(weights)

    vanilla_cost_matrix = params.CostMatrix(tn=0., fp=-1., fn=0., tp=1.)

    weighted_threshold = threshold_policies.single_threshold(
        predictions, labels, weights, vanilla_cost_matrix)
    self.assertAlmostEqual(weighted_threshold, 0.75)

  def test_convex_hull_roc(self):

    # The ROC curve for the test looks like this.
    # 'o' marks the points that should be deleted.
    #
    #             /
    #            /
    #   ____o   /
    #  /     \o/
    # /
    ##################

    fpr_tpr = [
        (0.0, 0.0),
        (0.2, 0.5),
        (0.3, 0.5),  # This point should be removed: Not in convex hull.
        (0.4, 0.1),  # This point should be removed: Below random-guessing line.
        (0.5, 0.7),
        (1.0, 1.0),
    ]

    expected_fpr_tpr = [
        (0.0, 0.0),
        (0.2, 0.5),
        (0.5, 0.7),
        (1.0, 1.0),
    ]

    fpr, tpr = zip(*fpr_tpr)
    expected_fpr, expected_tpr = zip(*expected_fpr_tpr)
    np.random.seed(100)
    # Thresholds don't really matter here, so they can be random.
    thresholds = sorted([np.random.rand() for _ in fpr])
    result_fpr, result_tpr, _ = threshold_policies.convex_hull_roc(
        (list(fpr), list(tpr), thresholds))
    self.assertListEqual(list(expected_fpr), result_fpr)
    self.assertListEqual(list(expected_tpr), result_tpr)


if __name__ == '__main__':
  absltest.main()
