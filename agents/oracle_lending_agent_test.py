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

"""Tests for oracle_lending_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import params
import rewards
import test_util
from agents import classifier_agents
from agents import oracle_lending_agent
from agents import threshold_policies
from environments import lending


class OracleLendingAgentTest(absltest.TestCase):

  def test_oracle_lending_agent_interacts(self):
    env = lending.DelayedImpactEnv()

    agent_params = classifier_agents.ScoringAgentParams(
        feature_keys=['applicant_features'],
        group_key='group',
        default_action_fn=(lambda: 1),
        burnin=1,
        convert_one_hot_to_integer=True,
        cost_matrix=params.CostMatrix(
            fn=0, fp=-1, tp=env.initial_params.interest_rate, tn=0))

    agent = oracle_lending_agent.OracleThresholdAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('bank_cash'),
        params=agent_params,
        env=env)

    test_util.run_test_simulation(env=env, agent=agent)

  def test_oracle_maxutil_classifier_is_stable(self):
    env = lending.DelayedImpactEnv()

    agent_params = classifier_agents.ScoringAgentParams(
        feature_keys=['applicant_features'],
        group_key='group',
        default_action_fn=(lambda: 1),
        burnin=1,
        threshold_policy=threshold_policies.ThresholdPolicy.SINGLE_THRESHOLD,
        convert_one_hot_to_integer=True,
        cost_matrix=params.CostMatrix(
            fn=0, fp=-1, tp=env.initial_params.interest_rate, tn=0))

    agent = oracle_lending_agent.OracleThresholdAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        reward_fn=rewards.BinarizedScalarDeltaReward('bank_cash'),
        params=agent_params,
        env=env)

    test_util.run_test_simulation(env=env, agent=agent)
    # Drop 0 threshold associated with burn-in.
    first_nonzero_threshold = None
    for thresh in agent.global_threshold_history:
      if thresh > 0:
        if first_nonzero_threshold is None:
          first_nonzero_threshold = thresh
        self.assertAlmostEqual(first_nonzero_threshold, thresh)
    # Make sure there is at least one non-zero threshold.
    self.assertIsNotNone(first_nonzero_threshold)


if __name__ == '__main__':
  absltest.main()
