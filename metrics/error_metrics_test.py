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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl.testing import absltest
import core
import params
import rewards
import test_util
from metrics import error_metrics
import numpy as np
from six.moves import range


class ErrorMetricTest(absltest.TestCase):

  def test_accuracy_metric_can_interact_with_dummy(self):
    def _is_zero(history_item):
      _, action = history_item
      return int(action == 0)

    env = test_util.DummyEnv()
    env.set_scalar_reward(rewards.NullReward())
    metric = error_metrics.AccuracyMetric(env=env, numerator_fn=_is_zero)
    test_util.run_test_simulation(env=env, metric=metric)

  def test_stratified_accuracy_metric_correct_atomic_prediction(self):
    """Check correctness when stratifying into (wrong, right) bins."""

    def _x_select(history_item):
      state, _ = history_item
      return int(state.x[0] == 1)

    def _x_stratify(history_item):
      state, _ = history_item
      return state.x[0]

    env = test_util.DeterministicDummyEnv()
    env.set_scalar_reward(rewards.NullReward())
    metric = error_metrics.AccuracyMetric(
        env=env, numerator_fn=_x_select, stratify_fn=_x_stratify)

    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric)

    logging.info('Measurement: %s.', measurement)

    self.assertEqual(measurement[0], 0)
    self.assertEqual(measurement[1], 1)

  def test_stratified_accuracy_metric_correct_sequence_prediction(self):
    """Check correctness when stratifying into (wrong, right) bins."""

    def _x_select(history_item):
      return [i == 1 for i in history_item.state.x]

    def _x_stratify(history_item):
      return history_item.state.x

    env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=10))
    env.set_scalar_reward(rewards.NullReward())
    metric = error_metrics.AccuracyMetric(
        env=env, numerator_fn=_x_select, stratify_fn=_x_stratify)

    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric)

    logging.info('Measurement: %s.', measurement)

    self.assertEqual(measurement[0], 0)
    self.assertEqual(measurement[1], 1)

  def test_confusion_metric_correct_for_atomic_prediction_rule(self):

    def _ground_truth_fn(history_item):
      state, _ = history_item
      return state.x[0]

    env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=1))
    env.set_scalar_reward(rewards.NullReward())
    # Always predict 1.
    metric = error_metrics.ConfusionMetric(
        env=env,
        prediction_fn=lambda x: 1,
        ground_truth_fn=_ground_truth_fn,
        stratify_fn=lambda x: 1)

    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric)

    logging.info('Measurement: %s.', measurement)

    # The keys in measurement are given by group membership, which in this case
    # is defined to always be 1.
    self.assertEqual(measurement[1].fp, 5)
    self.assertEqual(measurement[1].tp, 5)
    self.assertNotIn(0, measurement)

  def test_recall_metric_correct_for_atomic_prediction_rule(self):
    def _ground_truth_fn(history_item):
      state, _ = history_item
      return state.x[0]

    env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=1))
    env.set_scalar_reward(rewards.NullReward())
    # Always predict 1.
    metric = error_metrics.RecallMetric(
        env=env,
        prediction_fn=lambda x: 1,
        ground_truth_fn=_ground_truth_fn,
        stratify_fn=lambda x: 1)

    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric, num_steps=50)

    logging.info('Measurement: %s.', measurement)
    self.assertEqual({1: 1}, measurement)

  def test_recall_with_zero_denominator(self):
    env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=1))
    env.set_scalar_reward(rewards.NullReward())
    # Ground truth is always 0, recall will have a zero denominator.
    metric = error_metrics.RecallMetric(
        env=env,
        prediction_fn=lambda x: 0,
        ground_truth_fn=lambda x: 0,
        stratify_fn=lambda x: 1)

    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric, num_steps=50)
    self.assertEqual({1: 0}, measurement)

  def test_precision_metric_correct_for_atomic_prediction_rule(self):
    def _ground_truth_fn(history_item):
      state, _ = history_item
      return state.x[0]

    env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=1))
    env.set_scalar_reward(rewards.NullReward())
    # Always predict 1.
    metric = error_metrics.PrecisionMetric(
        env=env,
        prediction_fn=lambda x: 1,
        ground_truth_fn=_ground_truth_fn,
        stratify_fn=lambda x: 1)

    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric, num_steps=50)

    self.assertEqual({1: 0.5}, measurement)

  def test_precision_with_zero_denominator(self):
    def _ground_truth_fn(history_item):
      state, _ = history_item
      return state.x[0]

    env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=1))
    env.set_scalar_reward(rewards.NullReward())
    # Always predict 0, precision will have a zero denominator.
    metric = error_metrics.PrecisionMetric(
        env=env,
        prediction_fn=lambda x: 0,
        ground_truth_fn=_ground_truth_fn,
        stratify_fn=lambda x: 1)

    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric, num_steps=50)

    self.assertEqual({1: 0}, measurement)

  def test_confusion_metric_correct_for_sequence_prediction_rule(self):
    dim = 10
    def _ground_truth_fn(history_item):
      state, _ = history_item
      return state.x

    env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=dim))
    env.set_scalar_reward(rewards.NullReward())
    # Always predict a sequence of 1s.
    metric = error_metrics.ConfusionMetric(
        env=env,
        prediction_fn=lambda x: [1 for _ in range(dim)],
        ground_truth_fn=_ground_truth_fn,
        stratify_fn=lambda x: [1 for _ in range(dim)])

    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric)

    logging.info('Measurement: %s.', measurement)

    self.assertEqual(measurement[1].fp, 50)
    self.assertEqual(measurement[1].tp, 50)
    self.assertNotIn(0, measurement)

  def test_cost_metric_correct_for_atomic_prediction_rule(self):

    def _ground_truth_fn(history_item):
      state, _ = history_item
      return state.x[0]

    env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=1))
    env.set_scalar_reward(rewards.NullReward())
    cost_metric = error_metrics.CostedConfusionMetric(
        env=env,
        prediction_fn=lambda x: 1,
        ground_truth_fn=_ground_truth_fn,
        stratify_fn=lambda x: 1,
        cost_matrix=params.CostMatrix(tp=1, fp=-2, tn=-1, fn=-1))
    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=cost_metric)

    logging.info('Cost measurement: %s.', measurement)

    self.assertEqual(measurement[1], -5)
    self.assertNotIn(0, measurement)

  def test_cost_metric_correct_for_sequence_prediction_rule(self):
    dim = 10

    def _ground_truth_fn(history_item):
      state, _ = history_item
      return state.x

    env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=dim))
    env.set_scalar_reward(rewards.NullReward())
    cost_metric = error_metrics.CostedConfusionMetric(
        env=env,
        prediction_fn=lambda x: [1 for _ in range(dim)],
        ground_truth_fn=_ground_truth_fn,
        stratify_fn=lambda x: [1 for _ in range(dim)],
        cost_matrix=params.CostMatrix(tp=1, fp=-2, tn=-1, fn=-1))
    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=cost_metric)

    logging.info('Cost measurement: %s.', measurement)

    self.assertEqual(measurement[1], -50)
    self.assertNotIn(0, measurement)

  def test_confusion_as_array_and_cost_as_array_line_up(self):
    cost_matrix = params.CostMatrix(tp=1, tn=2, fp=3, fn=4).as_array()
    confusion_matrix = error_metrics.ConfusionMatrix(
        tp=1, tn=2, fp=3, fn=4).as_array()
    self.assertAlmostEqual(np.linalg.norm(cost_matrix - confusion_matrix), 0)

  def test_confusion_matrix_updates_correctly(self):
    confusion_matrix = error_metrics.ConfusionMatrix()
    confusion_matrix.update(prediction=1, truth=0)  # Add one false positive
    confusion_matrix.update(prediction=0, truth=0)  # Add one true negative
    self.assertEqual(confusion_matrix.fp, 1)
    self.assertEqual(confusion_matrix.tn, 1)

  def test_confusion_matrix_serializes(self):
    confusion_matrix = error_metrics.ConfusionMatrix()
    confusion_matrix.update(prediction=1, truth=0)  # Add one false positive
    confusion_matrix.update(prediction=0, truth=0)  # Add one true negative
    core.to_json(confusion_matrix)


if __name__ == '__main__':
  absltest.main()
