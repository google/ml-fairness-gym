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
"""Helper functions to create plots for lending experiments."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import bisect
import enum
import os

import file_util
from environments import lending_params
import matplotlib as mpl
# pylint: disable=g-import-not-at-top
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
# pylint: enable=g-import-not-at-top

mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.facecolor'] = 'white'

MAX_UTIL_TITLE = 'Max Utility'
EQ_OPP_TITLE = 'Eq. Opportunity'
SUCCESS_PROBABILITIES = lending_params.DELAYED_IMPACT_SUCCESS_PROBS


class PlotTypes(enum.Enum):
  CREDIT_DISTRIBUTIONS = 1
  CUMULATIVE_LOANS = 2
  THRESHOLD_HISTORY = 3
  MEAN_CREDIT_OVER_TIME = 4
  CUMULATIVE_RECALLS = 5


def _write(path):
  """Write a plot to a path."""
  if path:
    plt.savefig(file_util.open(path, 'w'), format='pdf')


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
      for loc in ['bottom', 'top', 'right', 'left']:
        plt.gca().spines[loc].set_color('gray')
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
  plt.grid(color='k', linewidth=0.5, axis='y')
  plt.title(title, fontsize=16)
  plt.xlabel('Credit score', fontsize=16)
  plt.ylabel('% applicants', fontsize=16)
  plt.tight_layout()
  plt.legend(numpoints=1)
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
  plt.legend(loc='upper left', fontsize=12)
  plt.grid(color='k', linewidth=0.5, axis='y')
  plt.tight_layout()
  _write(path)


def plot_threshold_history(threshold_history, path):
  """Creates a line plot of threshold values over time."""
  plt.figure(figsize=(8, 3))
  plt.title('EO agent decision thresholds', fontsize=16)
  colors = ['b', 'g']

  for group, history in sorted(
      threshold_history.items(), key=lambda x: np.argmax(x[0])):
    group_id = np.argmax(group)
    plt.plot(
        [val.smoothed_value()  for val in history],
        colors[group_id],
        label='Group %d' % (group_id + 1),
        linewidth=3)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.ylabel('Threshold', fontsize=16)
  plt.xlabel('# Steps', fontsize=16)
  plt.legend(loc='upper left', fontsize=12)
  plt.grid(color='k', linewidth=0.5, axis='y')
  plt.tight_layout()
  _write(path)


def _mu(step, group_id):
  """Computes average probability of paying back a loan."""
  dist = step.state.params.applicant_distribution.components[group_id]
  return np.dot(dist.weights,
                [1 - component.will_default.p for component in dist.components])


def plot_mu(histories, path):
  """Plots average probabilities of paying back a loan over time."""
  plt.figure(figsize=(8, 3))
  plt.title('Population credit', fontsize=16)
  colors = ['b', 'g']

  for group in [0, 1]:
    for title, history in histories.items():
      style = '-'
      if 'opp' in title.lower():
        style = '--'
      plt.plot(
          [_mu(step, group) for step in history],
          style + colors[group],
          label='Group %d-%s' % (
              (group + 1),  # Use 1-indexed group names.
              title),
          linewidth=3)

  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.ylabel('P(will repay)', fontsize=16)
  plt.xlabel('# Steps', fontsize=16)
  plt.legend(loc='upper left', fontsize=12)
  plt.grid(color='k', linewidth=0.5, axis='y')
  plt.tight_layout()
  _write(path)


def plot_cumulative_recall_differences(cumulative_recalls, path):
  """Plot differences in cumulative recall between groups up to time T."""
  plt.figure(figsize=(8, 3))
  style = {'dynamic': '-', 'static': '--'}

  for setting, recalls in cumulative_recalls.items():
    abs_array = np.mean(np.abs(recalls[0::2, :] - recalls[1::2, :]), axis=0)

    plt.plot(abs_array, style[setting], label=setting)

  plt.title(
      'Recall gap for EO agent in dynamic vs static environments', fontsize=16)
  plt.yscale('log')
  plt.xscale('log')
  plt.ylabel('TPR gap', fontsize=16)
  plt.xlabel('# steps', fontsize=16)
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  _write(path)


def plot_recall_targets(recall_targets, path):
  """Plot instantaneous recall targets."""
  plt.figure(figsize=(8, 3))
  plt.plot(recall_targets)
  plt.ylabel('Target TPR', fontsize=16)
  plt.xlabel('# Steps', fontsize=16)
  plt.tight_layout()
  _write(path)


def do_plotting(maximize_reward_result,
                equality_of_opportunity_result,
                static_equality_of_opportunity_result,
                plotting_dir,
                options=None):
  """Creates plots and writes them to a directory.

  Args:
    maximize_reward_result: The results from an experiment with a max-util
      agent.
    equality_of_opportunity_result: The results from an experiment with an
      agent constrained by equality of opportunity.
    static_equality_of_opportunity_result: The results from an experiment with
      an agent constrained by equality of opportunity without long-term credit
      dynamics.
    plotting_dir: A directory to write the plots.
    options: A set of PlotType enums that indicate which plots to create.
      If None, create everything.
  """

  if options is None:
    options = set(PlotTypes)

  if PlotTypes.CREDIT_DISTRIBUTIONS in options:
    plot_credit_distribution(
        maximize_reward_result['metric_results']['initial_credit_distribution'],
        'Initial',
        path=os.path.join(plotting_dir, 'initial.pdf'))
    plot_credit_distribution(
        maximize_reward_result['metric_results']['final_credit_distributions'],
        title=MAX_UTIL_TITLE,
        path=os.path.join(plotting_dir, 'max_utility.pdf'))
    plot_credit_distribution(
        equality_of_opportunity_result['metric_results']
        ['final_credit_distributions'],
        title=EQ_OPP_TITLE,
        path=os.path.join(plotting_dir, 'equalize_opportunity.pdf'))

  if PlotTypes.CUMULATIVE_LOANS in options:
    cumulative_loans = {
        'max reward':
            maximize_reward_result['metric_results']['cumulative_loans'],
        'equal-opp':
            equality_of_opportunity_result['metric_results']['cumulative_loans']
    }
    plot_cumulative_loans(
        cumulative_loans, os.path.join(plotting_dir, 'cumulative_loans.pdf'))

  if PlotTypes.THRESHOLD_HISTORY in options:
    plot_threshold_history(
        equality_of_opportunity_result['agent']['threshold_history'],
        os.path.join(plotting_dir, 'threshold_history.pdf'))

  if PlotTypes.MEAN_CREDIT_OVER_TIME in options:
    histories = {
        'max reward': maximize_reward_result['environment']['history'],
        'equal-opp': equality_of_opportunity_result['environment']['history']
    }

    plot_mu(histories, os.path.join(plotting_dir, 'mu.pdf'))

  if PlotTypes.CUMULATIVE_RECALLS in options:

    with file_util.open(
        os.path.join(plotting_dir, 'cumulative_recall_dynamic.txt'),
        'w') as outfile:
      np.savetxt(
          outfile,
          equality_of_opportunity_result['metric_results']['cumulative_recall'])
    with file_util.open(
        os.path.join(plotting_dir, 'target_recall_dynamic.txt'),
        'w') as outfile:
      np.savetxt(
          outfile,
          list(equality_of_opportunity_result['agent']['tpr_targets'].values()))

    plot_recall_targets(
        equality_of_opportunity_result['agent']['tpr_targets'][(0, 1)],
        os.path.join(plotting_dir, 'target_recall_dynamic.pdf'))

    with file_util.open(
        os.path.join(plotting_dir, 'cumulative_recall_static.txt'),
        'w') as outfile:
      np.savetxt(
          outfile, static_equality_of_opportunity_result['metric_results']
          ['cumulative_recall'])
