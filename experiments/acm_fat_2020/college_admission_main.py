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
"""Main file to run college admission experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os

from absl import app
from absl import flags
from absl import logging
import attr
import core
import params
from experiments import college_admission
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_steps', 750, 'Number of steps to run the episode.')

flags.DEFINE_integer('burnin', 10, 'Number of steps for burnin.')

flags.DEFINE_bool('epsilon_greedy', False, 'Float defines by how much to'
                  'control gaming.')

flags.DEFINE_float('initial_epsilon_prob', 0.1,
                   'Epsilon probability for epsilon greedy agent.')
flags.DEFINE_string('output_dir', '/tmp/',
                    'Output directory to write results to.')
flags.DEFINE_bool('verbose', 'False', 'If true, print results to screen.')
flags.DEFINE_enum('noisy_dist', 'gaussian', ['gaussian', 'beta'],
                  'Type of noise distribution to use.')
flags.DEFINE_bool('noisy_features', False, 'Exp with noisy features.')
flags.DEFINE_bool('noisy_threshold', False, 'Exp with noisy thresholds.')
flags.DEFINE_list('feature_mu', [0.5, 0.5], 'Feature means for groups.')

ENV_PARAMS = {
    'num_applicants':
        20,
    'feature_params':
        params.GMM(mix_weight=[0.5, 0.5], mu=[0.5, 0.5], sigma=[0.1, 0.1]),
    'noise_params':
        params.BoundedGaussian(max=0.0, min=-0, mu=0, sigma=0),
    # Scalar multiplier for cost (of a feature) per group.
    # Note: We codify B's disadvantage by requiring Cost A < Cost B.
    'group_cost': {
        0: 5,
        1: 10
    },
    'gaming':
        True,
    'gaming_control':
        np.inf
}

FIXED_AGENT_NUMSTEPS = 30
THRESHOLD_SPACING = 0.01
STDEV_RANGE_DEFAULTS = [0, 0.001, 0.01, 0.1, 0.15, 0.2, 0.3]


def log_results(exp_result):
  logging.info('------------------------------------------------')
  logging.info(
      'Agent: %s, Initial Threshold: %f, Burnin %d, Epsilon_Greedy %s, Epsilon_Prob %f',
      exp_result['exp_params']['agent_type'],
      exp_result['exp_params']['agent_threshold'],
      exp_result['exp_params']['burnin'],
      exp_result['exp_params']['epsilon_greedy'],
      exp_result['exp_params']['initial_epsilon_prob'])
  logging.info(exp_result['metric_results'])


#--------------------------------------------------
# Plotting Functions
#--------------------------------------------------


def plot_thresholds(reslist, thresholds, name=0):
  """Plots thresholds vs metric and equilibrium thresholds of various agents."""
  fig = plt.figure(figsize=(15, 6), facecolor='w', edgecolor='k')
  colors = ['m', 'g', 'c']
  agents = ['static', 'continuous', 'robust']
  metrics = ['accuracy', 'social_burden']
  final_thresholds = [
      reslist[agent][0.0]['metric_results']['final_threshold']
      for agent in agents
  ]
  for num, metric in enumerate(metrics):
    ax = fig.add_subplot(1, 2, num + 1)
    group_0 = [
        reslist['fixed'][tt]['metric_results'][metric]['0'] for tt in thresholds
    ]
    group_1 = [
        reslist['fixed'][tt]['metric_results'][metric]['1'] for tt in thresholds
    ]
    all_groups = [
        reslist['fixed'][tt]['metric_results']['overall_' + metric]['1']
        for tt in thresholds
    ]
    for final_threshold, color, agent in zip(final_thresholds, colors, agents):
      ax.plot([final_threshold, final_threshold], [0, 1],
              color,
              linewidth=2,
              label=agent)
      ax.text(final_threshold, 1, agent, rotation=30, fontsize=16)
      ax.text(
          final_threshold,
          0,
          np.around(final_threshold, 2),
          fontsize=11,
          rotation=10)
    ax.plot(thresholds, group_0, 'b--o', linewidth=3, label='group 0')
    ax.plot(thresholds, group_1, 'r--o', linewidth=3, label='group 1')
    ax.plot(thresholds, all_groups, 'g--o', linewidth=3, label='all groups')
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1, 0.1))
    ax.set_xlabel('thresholds', fontsize=16)
    ax.set_ylabel(metric, fontsize=16)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(metric, fontsize=16)
    ax.legend()
  plt.tight_layout()
  plt.show()
  plt.savefig(
      os.path.join(FLAGS.output_dir, 'thresholds_' + str(name) + '.pdf'))


def plot_threshold_pair(reslists, reslist_labels, thresholds, name=0):
  """Plots to compare metrics and agent thresholds across environments."""
  # TODO(): Remove redundancy with `plot_thresholds`.
  fig = plt.figure(figsize=(15, 5), facecolor='w', edgecolor='k')
  pal = sns.color_palette('Dark2', 6)
  colors = pal[3:]
  group_colors = pal[:3]
  alphas = [1, 1]
  linestyles = [':', '-', '--']
  agents = ['static', 'continuous', 'robust']
  agent_labels = ['Static', 'Continuous', 'Robust']
  metrics = ['accuracy', 'social_burden']
  metric_labels = ['Accuracy', 'Social Burden']
  ymaxs = [1.0, 1.2]

  for num, metric in enumerate(metrics):
    for resnum, reslist in enumerate(reslists):
      final_thresholds = [
          reslist[agent][0.0]['metric_results']['final_threshold']
          for agent in agents
      ]

      ax = fig.add_subplot(2, 2, 2 * num + resnum + 1)
      group_0 = [
          reslist['fixed'][tt]['metric_results'][metric]['0']
          for tt in thresholds
      ]
      group_1 = [
          reslist['fixed'][tt]['metric_results'][metric]['1']
          for tt in thresholds
      ]
      all_groups = [
          reslist['fixed'][tt]['metric_results']['overall_' + metric]['1']
          for tt in thresholds
      ]

      if resnum == 0 or True:
        for final_threshold, color, linestyle, agent_label in zip(
            final_thresholds, colors, linestyles, agent_labels):
          ax.axvline(
              final_threshold,
              c=color,
              ls=linestyle,
              linewidth=4,
              label=agent_label)

      ax.plot(
          thresholds,
          group_0,
          c=group_colors[0],
          alpha=alphas[resnum],
          linewidth=3,
          label='Disadv. Group')
      ax.plot(
          thresholds,
          group_1,
          c=group_colors[1],
          alpha=alphas[resnum],
          linewidth=3,
          label='Adv. Group')
      ax.plot(
          thresholds,
          all_groups,
          c=group_colors[2],
          alpha=alphas[resnum],
          linewidth=3,
          label='All Groups')
      ax.set_ylim(0, ymaxs[num])

      ax.set_xticks(np.arange(0, 1, 0.1))
      ax.set_yticks(np.arange(0, ymaxs[num], 0.2))
      if num == 1:
        ax.set_xlabel('Score Decision Thresholds', fontsize=16)
      ax.set_ylabel(metric_labels[num], fontsize=16)
      ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
      ax.xaxis.tick_bottom()
      ax.grid(axis='y', linewidth=1)
      if num == 0:
        ax.set_title(reslist_labels[resnum], fontsize=16)

      if num == 0 and resnum == 0:
        ax.legend(loc='upper left')

  plt.tight_layout()
  # Add space between plots corresponding to different environment conditions.
  plt.subplots_adjust(wspace=0.25)
  plt.show()
  plt.savefig(
      os.path.join(FLAGS.output_dir, 'threshold_pair' + str(name) + '.pdf'))


def plot_metrics(reslist, thresholds, name=0):
  """Plots metrics for different agents."""
  fig = plt.figure(figsize=(15, 6), facecolor='w', edgecolor='k')
  agents = ['static', 'continuous', 'robust']
  metrics = ['accuracy', 'social_burden']
  indices = [
      int(np.where(np.isclose(thresholds, t, rtol=1e-2))[0])
      for t in [0.5, 0.6, 0.7, 0.8, 0.9]
  ]
  fixed_thresholds = thresholds[indices]
  x_ticks = ['fixed_'+ str(t) for t in fixed_thresholds] + agents
  for num, metric in enumerate(metrics):
    ax = fig.add_subplot(1, 2, num + 1)
    group_0 = [
        reslist['fixed'][tt]['metric_results'][metric]['0']
        for tt in fixed_thresholds
    ] + [
        reslist[agent][0.0]['metric_results'][metric]['0'] for agent in agents
    ]
    group_1 = [
        reslist['fixed'][tt]['metric_results'][metric]['1']
        for tt in fixed_thresholds
    ] + [
        reslist[agent][0.0]['metric_results'][metric]['1'] for agent in agents
    ]
    all_groups = [
        reslist['fixed'][tt]['metric_results']['overall_' + metric]['1']
        for tt in fixed_thresholds
    ] + [
        reslist[agent][0.0]['metric_results']['overall_' + metric]['1']
        for agent in agents
    ]

    ax.plot(group_0, 'b--o', linewidth=1, markersize=12, label='group 0')
    ax.plot(group_1, 'r--o', linewidth=1, markersize=12, label='group 1')
    ax.plot(all_groups, 'g--o', linewidth=1, markersize=12, label='all groups')

    ax.set_yticks(np.arange(0, 1, 0.1))
    ax.set_xlabel('agents', fontsize=16)
    ax.set_ylabel(metric, fontsize=16)
    ax.set_xticklabels(x_ticks)
    ax.set_title(metric, fontsize=16)
    ax.legend()
  plt.tight_layout()
  plt.show()
  plt.savefig(os.path.join(FLAGS.output_dir, 'metrics_' + str(name) + '.pdf'))


def plot_deltas(deltas):
  """Plots the differences in final threshold of continuous & static agent."""
  plt.figure(figsize=(15, 6), facecolor='w', edgecolor='k')
  x = [sd for sd in deltas.keys()]
  y = [delta for delta in deltas.values()]
  plt.plot(
      x,
      y,
  )
  plt.ylabel('Delta [Continuous- Static] Final Threshold')
  plt.xlabel('Noise Standard Deviation')
  plt.tight_layout()
  plt.show()
  plt.savefig(os.path.join(FLAGS.output_dir, 'deltas.pdf'))


#--------------------------------------------------
# Main experiment runs.
#--------------------------------------------------
def run_baseline_experiment(feature_mu=None):
  """Run baseline experiment without noise."""
  results = {}
  agent_types = ['fixed', 'static', 'continuous', 'robust']
  thresholds = np.arange(0, 1, THRESHOLD_SPACING)
  if feature_mu is None:
    feature_mu = [0.5, 0.5]
  if len(feature_mu) != 2:
    raise ValueError('Expected feature_mu to be of length 2.')
  env_config_params = copy.deepcopy(ENV_PARAMS)

  env_config_params.update({
      'feature_params':
          params.GMM(mix_weight=[0.5, 0.5], mu=feature_mu,
                     sigma=[0.1, 0.1])
  })

  for agent_type in agent_types:
    results[agent_type] = {}
    for threshold in thresholds:
      results[agent_type][threshold] = {}
      if agent_type != 'fixed' and threshold > 0:
        continue
      num_steps = FIXED_AGENT_NUMSTEPS if agent_type == 'fixed' else FLAGS.num_steps
      college_experiment = college_admission.CollegeExperiment(
          num_steps=num_steps,
          env_config=env_config_params,
          agent_type=agent_type,
          agent_threshold=threshold,
          burnin=FLAGS.burnin,
          epsilon_greedy=FLAGS.epsilon_greedy,
          initial_epsilon_prob=FLAGS.initial_epsilon_prob)
      json_dump = college_experiment.run_experiment()
      exp_result = json.loads(json_dump)
      exp_params = copy.deepcopy(attr.asdict(college_experiment))
      exp_result.update({'exp_params': exp_params})
      if FLAGS.verbose:
        log_results(exp_result)
      with open(
          os.path.join(FLAGS.output_dir, 'experiment_results.json'), 'a+') as f:
        core.to_json(exp_result, f)
        f.write('\n---------------------------------------\n')
      results[agent_type][threshold] = exp_result
  return results, thresholds


def run_noisy_experiment(noise_dist='gaussian',
                         noisy_features=False,
                         noisy_threshold=False,
                         feature_mu=None,
                         stdevs=None):
  """Noisy experiment runs."""
  results = {}
  deltas = {}
  agent_types = ['fixed', 'static', 'continuous', 'robust']
  thresholds = np.arange(0, 1, THRESHOLD_SPACING)
  if noise_dist == 'beta':
    logging.info('Using Beta Noise Distribution.')
    stdevs = np.arange(2, 9, 1)
    mu = 2
    max_value = 0.7
    min_value = 0
  else:
    logging.info('Using Gaussian Noise Distribution.')
    mu = 0
    max_value = 0.35
    min_value = -0.35
  if feature_mu is None:
    feature_mu = [0.5, 0.5]
  if len(feature_mu) != 2:
    raise ValueError('Expected feature_mu to be of length 2.')
  if stdevs is None:
    stdevs = STDEV_RANGE_DEFAULTS
  for sd in stdevs:
    env_config_params = copy.deepcopy(ENV_PARAMS)
    env_config_params.update({
        'noise_dist':
            noise_dist,
        'noise_params':
            params.BoundedGaussian(
                max=max_value, min=min_value, mu=mu, sigma=sd),
        'noisy_features':
            noisy_features,
        'noisy_threshold':
            noisy_threshold,
        'feature_params':
            params.GMM(mix_weight=[0.5, 0.5], mu=feature_mu, sigma=[0.1, 0.1]),
    })
    logging.info('Stdev %f', sd)
    results[sd] = {}
    for agent_type in agent_types:
      results[sd][agent_type] = {}
      for threshold in thresholds:
        results[sd][agent_type][threshold] = {}
        if agent_type != 'fixed' and threshold > 0:
          continue
        num_steps = FIXED_AGENT_NUMSTEPS if agent_type == 'fixed' else FLAGS.num_steps
        college_experiment = college_admission.CollegeExperiment(
            num_steps=num_steps,
            env_config=env_config_params,
            agent_type=agent_type,
            agent_threshold=threshold,
            burnin=FLAGS.burnin,
            epsilon_greedy=FLAGS.epsilon_greedy,
            initial_epsilon_prob=FLAGS.initial_epsilon_prob)
        json_dump = college_experiment.run_experiment()
        exp_result = json.loads(json_dump)
        exp_params = copy.deepcopy(attr.asdict(college_experiment))
        exp_result.update({'exp_params': exp_params})
        if FLAGS.verbose:
          log_results(exp_result)
        with open(
            os.path.join(FLAGS.output_dir, 'experiment_results.json'),
            'w+') as f:
          core.to_json(exp_result, f)
          f.write('\n---------------------------------------\n')
        results[sd][agent_type][threshold] = exp_result
    deltas[sd] = (
        results[sd]['continuous'][0.0]['metric_results']['final_threshold'] -
        results[sd]['static'][0.0]['metric_results']['final_threshold'])
  return results, thresholds, deltas, stdevs


def main(argv):
  # TODO(): add tests to check correct plots generated.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  deltas = None

  # Casting to float since these are sometimes passed through flags as strings.
  feature_mu = [float(mu) for mu in FLAGS.feature_mu]
  if FLAGS.noisy_features or FLAGS.noisy_threshold:
    results, thresholds, deltas, stdevs = run_noisy_experiment(
        noise_dist=FLAGS.noisy_dist,
        noisy_features=FLAGS.noisy_features,
        noisy_threshold=FLAGS.noisy_threshold,
        feature_mu=feature_mu)
    plot_deltas(deltas)
    for sd in stdevs:
      plot_thresholds(results[sd], thresholds, name=sd)
      plot_metrics(results[sd], thresholds, name=sd)

  else:
    results, thresholds = run_baseline_experiment(FLAGS.feature_mu)
    results_noise, _, _, _ = run_noisy_experiment(
        noisy_features=True, stdevs=[0.1])
    plot_threshold_pair([results, results_noise[0.1]],
                        ['Noiseless Features', 'Noisy Features'], thresholds)


if __name__ == '__main__':
  app.run(main)
