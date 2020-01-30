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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import distributions
import rewards
import test_util
from environments import lending
from environments import lending_params
from metrics import lending_metrics
import numpy as np


class CreditDistributionTest(absltest.TestCase):

  def test_final_credit_distribution_metric_can_interact_with_lending(self):
    env = lending.DelayedImpactEnv()
    env.set_scalar_reward(rewards.NullReward())
    # Use step=-1 to get the final credit distribution.
    final_distribution = lending_metrics.CreditDistribution(env, step=-1)
    initial_distribution = lending_metrics.CreditDistribution(env, step=0)
    test_util.run_test_simulation(
        env=env, metric=[final_distribution, initial_distribution])

  def test_measure_distribution_change_measurement(self):

    # The lower cluster has a 100% success rate and the upper cluster has a 0%
    # success rate. This causes applicants to move constantly between clusters.
    clusters = distributions.Mixture(
        components=[
            lending_params._credit_cluster_builder(
                group_membership=[1, 0],
                cluster_probs=[0.1, 0.9],
                success_probs=[1., 0.])(),
            lending_params._credit_cluster_builder(
                group_membership=[0, 1],
                cluster_probs=[0.8, 0.2],
                success_probs=[1., 0.])(),
        ],
        weights=(0.5, 0.5))

    env = lending.DelayedImpactEnv(
        lending_params.DelayedImpactParams(applicant_distribution=clusters))
    initial_distribution = lending_metrics.CreditDistribution(env, 0)
    final_distribution = lending_metrics.CreditDistribution(env, -1)

    # Giving a loan should change the distribution.
    env.step(np.asarray(1))
    # Take another step to move current state into history. This step does not
    # change the distribution because the loan is rejected.
    env.step(np.asarray(0))

    self.assertEqual({
        '0': [0.1, 0.9],
        '1': [0.8, 0.2]
    }, initial_distribution.measure(env))
    self.assertNotEqual({
        '0': [0.1, 0.9],
        '1': [0.8, 0.2]
    }, final_distribution.measure(env))


class CumulativeLoansTest(absltest.TestCase):

  def test_cumulative_count(self):
    env = lending.DelayedImpactEnv()
    metric = lending_metrics.CumulativeLoans(env)

    env.seed(100)
    _ = env.reset()
    for _ in range(10):
      env.step(np.asarray(1))

    result = metric.measure(env)
    self.assertEqual(result.shape, (2, 10))

    # On the first step, the combined number of loans given out should be 1.
    self.assertEqual(result[:, 0].sum(), 1)

    # On the last step, the combined number of loans given out should be 10.
    self.assertEqual(result[:, -1].sum(), 10)

  def test_no_loans_to_group_zero(self):
    env = lending.DelayedImpactEnv()
    metric = lending_metrics.CumulativeLoans(env)

    env.seed(100)
    obs = env.reset()
    for _ in range(10):
      # action is 0 for group 0 and 1 for group 1.
      action = np.argmax(obs['group'])
      obs, _, _, _ = env.step(action)

    result = metric.measure(env)
    self.assertEqual(result.shape, (2, 10))

    # Group 0 gets no loans.
    self.assertEqual(result[0, -1], 0)

    # Group 1 gets at least 1 loan.
    self.assertGreater(result[1, -1], 0)


if __name__ == '__main__':
  absltest.main()
