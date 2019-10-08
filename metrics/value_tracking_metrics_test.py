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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl.testing import absltest

import core
import rewards
import test_util
from environments import attention_allocation
from environments import lending
from metrics import value_tracking_metrics
from gym import spaces
import numpy as np


def _modifier_fn(selected_variable, group_id, env):
  del env  # Unused argument
  if group_id == 0:
    return -0.2 * (1 + selected_variable)
  else:
    return 0.3 * (1 + selected_variable)


def _selection_fn(history_step):
  return history_step.state.x


def _stratify_fn(history_step):
  return [1 - x for x in history_step.state.x]


def _setup_test_simulation(dim=1, calc_mean=False, modifier_fn=_modifier_fn):
  env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=dim))
  env.set_scalar_reward(rewards.NullReward())
  metric = value_tracking_metrics.AggregatorMetric(
      env=env,
      modifier_fn=modifier_fn,
      selection_fn=_selection_fn,
      stratify_fn=_stratify_fn,
      calc_mean=calc_mean)
  return env, metric


class AggregatingMetricsTest(absltest.TestCase):
  """Tests for aggregating metrics.

  These tests use the DeterministicDummyEnv, which alternates between 1 and 0
  or a list of 0's and 1's with length given by dim.
  To values for tests are calculated for each group as:
  sum/mean over the values passed by the modifier_fn. The modifier_fn receives
  values from the selection function.

  For example:
  For getting the sum from a list with dim=10 and over 10 steps, with
  modifier function as defined, we would expect values to be:
    group 0 = sum([-0.2 * (1 + 0)] * 10 for 5 steps) = -20
    group 1 = sum([0.3 * (1 + 1)] * 10 for 5 steps) = 15

  Similarly, without modifier function for a list it would be:
    group 0 = sum([0] * 10 for 5 steps)  = 0
    group 1 = sum([1] * 10 for 5 steps)  = 50
  """

  def test_aggregate_metric_give_correct_sum_value_for_list(self):
    """Test aggregate metric with sum for a list.

    Expected values:
      group 0 = sum([-0.2 * (1 + 0)] * 10 for 5 steps) = -20
      group 1 = sum([0.3 * (1 + 1)] * 10 for 5 steps) = 15
    """
    env, metric = _setup_test_simulation(dim=10, calc_mean=False)
    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric, num_steps=10)
    logging.info('Measurement result: %s.', measurement)
    self.assertSequenceAlmostEqual(
        sorted(measurement.values()), [-20, 15], delta=1e-4)

  def test_aggregate_metric_give_correct_sum_value_for_atomic_value(self):
    """Test aggregate metric with sum for a atomic values.

    Expected values:
      group 0 = sum([-0.2 * (1 + 0)] * 1 for 5 steps) = -2
      group 1 = sum([0.3 * (1 + 1)] * 1 for 5 steps) = 1.5
    """
    env, metric = _setup_test_simulation(dim=1, calc_mean=False)
    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric, num_steps=10)
    logging.info('Measurement result: %s.', measurement)
    self.assertSequenceAlmostEqual(
        sorted(measurement.values()), [-2, 1.5], delta=1e-4)

  def test_aggregate_metric_give_correct_mean_value_for_list(self):
    """Test aggregate metric with mean for a list.

    Expected values:
      group 0 = mean([-0.2 * (1 + 0)] * 10 for 5 steps) = -0.4
      group 1 = mean([0.3 * (1 + 1)] * 10 for 5 steps) = 0.3
    """
    env, metric = _setup_test_simulation(dim=10, calc_mean=True)
    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric, num_steps=10)
    logging.info('Measurement result: %s.', measurement)
    self.assertSequenceAlmostEqual(
        sorted(measurement.values()), [-0.4, 0.3], delta=1e-4)

  def test_aggregate_metric_give_correct_mean_value_for_atomic_value(self):
    """Test aggregate metric with mean for a atomic values.

    Expected values:
      group 0 = mean([-0.2 * (1 + 0)] * 10 for 5 steps) = -0.4
      group 1 = mean([0.3 * (1 + 1)] * 10 for 5 steps) = 0.3
    """
    env, metric = _setup_test_simulation(dim=1, calc_mean=True)
    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric, num_steps=10)
    logging.info('Measurement result: %s.', measurement)
    self.assertSequenceAlmostEqual(
        sorted(measurement.values()), [-0.4, 0.3], delta=1e-4)

  def test_aggregate_metric_give_correct_result_for_list_no_modifier(self):
    """Test aggregate metric with mean for a list with no modifier function.

    Expected values:
      group 0 = sum([0] * 10 for 5 steps) = 0
      group 1 = sum([1] * 10 for 5 steps) = 50
    """
    env, metric = _setup_test_simulation(
        dim=10, calc_mean=False, modifier_fn=None)
    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric, num_steps=10)
    logging.info('Measurement result: %s.', measurement)
    self.assertSequenceAlmostEqual(
        sorted(measurement.values()), [0, 50], delta=1e-4)

  def test_aggregate_metric_give_correct_result_for_atomic_value_no_modifier(
      self):
    """Test aggregate metric with sum for a atomic values with no modifier fn.

    Expected values:
      group 0 = sum([0] * 1 for 5 steps) = 0
      group 1 = sum([1] * 1 for 5 steps) = 5
    """
    env, metric = _setup_test_simulation(
        dim=1, calc_mean=False, modifier_fn=None)
    measurement = test_util.run_test_simulation(
        env=env, agent=None, metric=metric, num_steps=10)
    logging.info('Measurement result: %s.', measurement)
    self.assertSequenceAlmostEqual(
        sorted(measurement.values()), [0, 5], delta=1e-4)


class SummingMetricsTest(absltest.TestCase):

  def test_summing_metric_give_correct_sum_dummy_env(self):
    env = test_util.DeterministicDummyEnv(test_util.DummyParams(dim=1))
    env.set_scalar_reward(rewards.NullReward())

    metric = value_tracking_metrics.SummingMetric(
        env=env, selection_fn=_selection_fn)
    measurement = test_util.run_test_simulation(
        env, agent=None, metric=metric, seed=0)

    self.assertTrue(np.all(np.equal(measurement, [5])))

  def test_summing_metric_give_correct_sum_alloc_env(self):
    env = attention_allocation.LocationAllocationEnv()

    def _attn_alloc_selection_fn(step):
      state, _ = step
      return state.incidents_seen

    metric = value_tracking_metrics.SummingMetric(
        env=env, selection_fn=_attn_alloc_selection_fn)
    measurement = test_util.run_test_simulation(
        env, agent=None, metric=metric, seed=0)

    self.assertTrue(np.all(np.equal(measurement, [4, 5])))


class _XState(core.State):
  """State with a single variable x."""

  def __init__(self):
    self.x = 0


class IncreasingEnv(core.FairnessEnv):
  """Environment with a single state variable that increase at each step."""

  def __init__(self):
    self.action_space = spaces.Discrete(1)
    super(IncreasingEnv, self).__init__()
    self.state = _XState()

  def _step_impl(self, state, action):
    """Increase state.x by 1."""
    del action
    state.x += 1
    return state


class ValueChangeTest(absltest.TestCase):

  def test_value_change_measures_correctly_unnormalized(self):
    env = IncreasingEnv()
    metric = value_tracking_metrics.ValueChange(
        env, 'x', normalize_by_steps=False)
    # Running step 11 times records 10 steps in history because the 11th is
    # stored in current state.
    for _ in range(11):
      env.step(env.action_space.sample())
    self.assertEqual(metric.measure(env), 10)

  def test_value_change_measures_correctly_normalized(self):
    env = IncreasingEnv()
    metric = value_tracking_metrics.ValueChange(
        env, 'x', normalize_by_steps=True)
    for _ in range(17):
      env.step(action=env.action_space.sample())
    self.assertAlmostEqual(metric.measure(env), 1.0)

  def test_metric_can_interact_with_lending(self):
    env = lending.DelayedImpactEnv()
    metric = value_tracking_metrics.ValueChange(env, 'bank_cash')
    test_util.run_test_simulation(env=env, metric=metric)


class FinalValueMetricTest(absltest.TestCase):

  def test_returns_correct_final_value(self):
    env = IncreasingEnv()
    metric = value_tracking_metrics.FinalValueMetric(env, 'x')
    for _ in range(5):
      env.step(env.action_space.sample())
    self.assertEqual(metric.measure(env), 4)

  def test_returns_correct_final_value_with_realign(self):
    env = IncreasingEnv()
    metric = value_tracking_metrics.FinalValueMetric(
        env, 'x', realign_fn=lambda h: h[1:] + [h[0]])
    for _ in range(5):
      env.step(env.action_space.sample())
    self.assertEqual(metric.measure(env), 0)


if __name__ == '__main__':
  absltest.main()
