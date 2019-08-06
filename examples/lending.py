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
"""Code to run lending experiments."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import bisect
import os

from absl import logging
import attr
import core
import params
import rewards
import run_util
from agents import classifier_agents
from agents import threshold_policies
from environments import lending
from environments import lending_params
from metrics import error_metrics
from metrics import lending_metrics
from metrics import value_tracking_metrics
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg

MAXIMIZE_REWARD = threshold_policies.ThresholdPolicy.MAXIMIZE_REWARD
EQUALIZE_OPPORTUNITY = threshold_policies.ThresholdPolicy.EQUALIZE_OPPORTUNITY
SUCCESS_PROBABILITIES = lending_params.DELAYED_IMPACT_SUCCESS_PROBS


def _write(path):
  """Write a plot to a path."""
  if path:
    logging.info('Writing plot to %s', path)
    plt.savefig(path)


def _build_transition_matrix(success_probabilities):
  transition_matrix = np.zeros(
      (len(success_probabilities), len(success_probabilities)))
  transition_matrix += np.diag(success_probabilities[:-1], 1)
  transition_matrix += np.diag([1 - s for s in success_probabilities[1:]], -1)
  transition_matrix[0, 0] = 1 - success_probabilities[0]
  transition_matrix[-1, -1] = success_probabilities[-1]
  return transition_matrix.T


def plot_steady_state_distribution(basedir):
  """Plot the steady state distribution of credit scores."""
  eigs = linalg.eig(_build_transition_matrix(SUCCESS_PROBABILITIES))
  for i in range(len(SUCCESS_PROBABILITIES)):
    if np.abs(eigs[0][i] - 1) < 1e-6:
      plt.plot(eigs[1][:, i] / sum(eigs[1][:, i]), '--o', linewidth=3)
      plt.gca().grid(False)
      for loc in ['bottom', 'top', 'right', 'left']:
        plt.gca().spines[loc].set_color('gray')
      plt.gca().set_facecolor('white')
      plt.xticks([])
      plt.yticks([])
      plt.ylabel('% applicants \n in population', fontsize=24)
      plt.xlabel('Credit score', fontsize=24)
      plt.title('Permissive lending\nPopulation steady state', fontsize=24)
  path = os.path.join(basedir, 'steady_state.pdf')
  _write(path)


def plot_bars(recall, title, path, ylabel='Recall', figure=None):
  """Create a bar plot with two bars."""
  if figure is None:
    plt.figure()
  plt.title(title, fontsize=16)
  plt.bar([0, 1], [recall['0'], recall['1']])
  plt.xticks([0, 1], ['group 1', 'group 2'], fontsize=16, rotation=90)
  plt.ylabel(ylabel, fontsize=16)
  plt.gca().set_facecolor('white')
  plt.grid(color='k', linewidth=0.5, axis='y')
  plt.tight_layout()
  _write(path)


def plot_credit_distribution(credit_distributions,
                             title,
                             path,
                             include_median=False,
                             figure=None):
  """Plot a distribution over credit scores."""
  if figure is None:
    plt.figure(figsize=(3, 3))
  plt.plot(credit_distributions['0'], 'bo--', label='Group 1', linewidth=3)
  plt.plot(credit_distributions['1'], 'go--', label='Group 2', linewidth=3)
  if include_median:
    median = bisect.bisect_left(np.cumsum(credit_distributions['0']), 0.5)
    plt.plot([median], [credit_distributions['0'][median]],
             'b^--',
             markersize=16)
    median = bisect.bisect_left(np.cumsum(credit_distributions['1']), 0.5)
    plt.plot([median], [credit_distributions['1'][median]],
             'g^--',
             markersize=16)
  plt.ylim([0, 1])
  plt.gca().set_facecolor('white')
  plt.grid(color='k', linewidth=0.5, axis='y')
  plt.title(title, fontsize=16)
  plt.xlabel('Credit score', fontsize=16)
  plt.ylabel('% applicants', fontsize=16)
  plt.tight_layout()
  legend = plt.legend(numpoints=1)
  frame = legend.get_frame()
  frame.set_color('white')
  _write(path)


def plot_cumulative_loans(cumulative_loans, path, figure=None, normalize=None):
  """Plot cumulative loans over steps."""
  if figure is None:
    plt.figure(figsize=(8, 3))
  plt.title('Cumulative loans', fontsize=16)
  colors = ['b', 'g']
  if normalize is None:
    normalize = [1.0, 1.0]
  for group in [0, 1]:
    for title, loans in cumulative_loans.items():
      style = '-'
      if 'opp' in title.lower():
        style = '--'
      plt.plot(
          [val / normalize[group] for val in loans[group]],
          style + colors[group],
          label='Group %d-%s' % (
              (group + 1),  # Use 1-indexed group names.
              title),
          linewidth=3)

  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.ylabel('# Loans', fontsize=16)
  plt.xlabel('# Steps', fontsize=16)
  legend = plt.legend(loc='upper left', fontsize=12)
  frame = legend.get_frame()
  frame.set_color('white')
  plt.gca().set_facecolor('white')
  plt.grid(color='k', linewidth=0.5, axis='y')
  plt.tight_layout()
  _write(path)


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

    agent = classifier_agents.ThresholdAgent(
        action_space=env.action_space,
        reward_fn=rewards.BinarizedScalarDeltaReward(
            'bank_cash', baseline=env.initial_params.bank_starting_cash),
        observation_space=env.observation_space,
        params=agent_params)

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
        # 'mu':
        #     lending_metrics.RepaymentProb(env),
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

    metric_results = run_util.run_simulation(env, agent, metrics,
                                             self.num_steps, self.seed)
    report = {
        'environment': {
            'name': env.__class__.__name__,
            'params': env.initial_params
        },
        'agent': {
            'name': agent.__class__.__name__,
            'params': agent.params,
            'debug_string': agent.debug_string()
        },
        'experiment_params': self,
        'metric_results': metric_results,
    }
    if self.return_json:
      return core.to_json(report, indent=4)
    return report
