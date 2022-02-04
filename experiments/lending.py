# coding=utf-8
# Copyright 2022 The ML Fairness Gym Authors.
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
"""Code to run lending experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
import core
import params
import rewards
import run_util
from agents import classifier_agents
from agents import oracle_lending_agent
from agents import threshold_policies
from environments import lending
from environments import lending_params
from metrics import error_metrics
from metrics import lending_metrics
from metrics import value_tracking_metrics

MAXIMIZE_REWARD = threshold_policies.ThresholdPolicy.MAXIMIZE_REWARD
EQUALIZE_OPPORTUNITY = threshold_policies.ThresholdPolicy.EQUALIZE_OPPORTUNITY


@attr.s
class Experiment(core.Params):
  """Specifies the parameters of an experiment to run."""
  ######################
  # Environment params #
  ######################

  group_0_prob = attr.ib(default=0.5)
  bank_starting_cash = attr.ib(default=100)
  interest_rate = attr.ib(default=1.0)
  cluster_shift_increment = attr.ib(default=0.01)
  cluster_probabilities = attr.ib(
      default=lending_params.DELAYED_IMPACT_CLUSTER_PROBS)

  ################
  # Agent params #
  ################

  # Policy the agent uses when setting thresholds.
  threshold_policy = attr.ib(default=MAXIMIZE_REWARD)
  # Number of steps before applying the threshold policy.
  burnin = attr.ib(default=50)

  ##############
  # Run params #
  ##############

  seed = attr.ib(default=27)  # Random seed.
  num_steps = attr.ib(default=10000)  # Number of steps in the experiment.
  return_json = attr.ib(default=True)  # Return the results as a json string.
  include_cumulative_loans = attr.ib(default=False)

  def scenario_builder(self):
    """Returns an agent and environment pair."""
    env_params = lending_params.DelayedImpactParams(
        applicant_distribution=lending_params.two_group_credit_clusters(
            cluster_probabilities=self.cluster_probabilities,
            group_likelihoods=[self.group_0_prob, 1 - self.group_0_prob]),
        bank_starting_cash=self.bank_starting_cash,
        interest_rate=self.interest_rate,
        cluster_shift_increment=self.cluster_shift_increment,
    )
    env = lending.DelayedImpactEnv(env_params)

    agent_params = classifier_agents.ScoringAgentParams(
        feature_keys=['applicant_features'],
        group_key='group',
        default_action_fn=(lambda: 1),
        burnin=self.burnin,
        convert_one_hot_to_integer=True,
        threshold_policy=self.threshold_policy,
        skip_retraining_fn=lambda action, observation: action == 0,
        cost_matrix=params.CostMatrix(
            fn=0, fp=-1, tp=env_params.interest_rate, tn=0))

    agent = oracle_lending_agent.OracleThresholdAgent(
        action_space=env.action_space,
        reward_fn=rewards.BinarizedScalarDeltaReward(
            'bank_cash', baseline=env.initial_params.bank_starting_cash),
        observation_space=env.observation_space,
        params=agent_params,
        env=env)
    agent.seed(100)
    return env, agent

  def run(self):
    """Run a lending experiment.

    Returns:
      A json encoding of the experiment result.
    """

    env, agent = self.scenario_builder()
    metrics = {
        'initial_credit_distribution':
            lending_metrics.CreditDistribution(env, step=0),
        'final_credit_distributions':
            lending_metrics.CreditDistribution(env, step=-1),
        'recall':
            error_metrics.RecallMetric(
                env,
                prediction_fn=lambda x: x.action,
                ground_truth_fn=lambda x: not x.state.will_default,
                stratify_fn=lambda x: str(x.state.group_id)),
        'precision':
            error_metrics.PrecisionMetric(
                env,
                prediction_fn=lambda x: x.action,
                ground_truth_fn=lambda x: not x.state.will_default,
                stratify_fn=lambda x: str(x.state.group_id)),
        'profit rate':
            value_tracking_metrics.ValueChange(env, state_var='bank_cash'),
    }

    if self.include_cumulative_loans:
      metrics['cumulative_loans'] = lending_metrics.CumulativeLoans(env)
      metrics['cumulative_recall'] = lending_metrics.CumulativeRecall(env)

    metric_results = run_util.run_simulation(env, agent, metrics,
                                             self.num_steps, self.seed)
    report = {
        'environment': {
            'name': env.__class__.__name__,
            'params': env.initial_params,
            'history': env.history,
        },
        'agent': {
            'name': agent.__class__.__name__,
            'params': agent.params,
            'debug_string': agent.debug_string(),
            'threshold_history': agent.group_specific_threshold_history,
            'tpr_targets': agent.target_recall_history,
        },
        'experiment_params': self,
        'metric_results': metric_results,
    }
    if self.return_json:
      return core.to_json(report, indent=4)
    return report
