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

"""Main file to run attention allocation experiments.

This file replicates experiments done for the KDD workshop paper
"Fairness is Not Static".

Note this file can take a significant amount of time to run all experiments
since experiments are being repeated multiple times and the results averaged.
To run experiments fewer times, change the experiments.num_runs parameter to
10.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
import test_util
from agents import probability_matching_agents
from environments import attention_allocation
from examples import attention_allocation_experiment
from examples import attention_allocation_experiment_plotting
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', '/tmp/',
                    'Output directory to write results to.')


def _get_base_env_params():
  return attention_allocation.Params(
      n_locations=5,
      prior_incident_counts=(500, 500, 500, 500, 500),
      incident_rates=[2.3, 1.1, 1.8, .6, .3],
      n_attention_units=4,
      miss_incident_prob=(0., 0., 0., 0., 0.),
      extra_incident_prob=(0., 0., 0., 0., 0.),
      dynamic_rate=0.0)


def _setup_experiment():
  return attention_allocation_experiment.Experiment(
      num_runs=50,
      num_steps=1000,
      num_workers=25,
      seed=0,
      env_class=attention_allocation.LocationAllocationEnv,
      env_params=_get_base_env_params())


def _print_discovered_missed_incidents_report(value, report):
  discovered_incidents = np.array(report['metrics']['discovered_incidents'])
  discovered_total = np.sum(discovered_incidents)
  missed_incidents = np.array(
      report['metrics']['occurred_incidents']) - np.array(
          report['metrics']['discovered_incidents'])
  missed_total = np.sum(missed_incidents)

  print(
      'REPORT dynamic_value: {}\ndiscovered_total: {}\nmissed_total: {}\ndiscovered_locations: {}\nmissed_locations: {}\n'
      .format(value, discovered_total, missed_total, discovered_incidents,
              missed_incidents))


def uniform_agent_resource_all_dynamics():
  """Run experiments on a uniform agent across dynamic rates."""

  dynamic_values_to_test = [0.0, 0.01, 0.05, 0.1, 0.15]
  experiment = _setup_experiment()
  experiment.agent_class = test_util.DummyAgent

  reports_dict = {}

  for value in dynamic_values_to_test:
    print('Running an experiment...')
    experiment.env_params.dynamic_rate = value
    json_report = attention_allocation_experiment.run(experiment)
    report = json.loads(json_report)

    print('\n\nUniform Random Agent, 4 attention units')
    _print_discovered_missed_incidents_report(value, report)
    output_filename = 'uniform_4units_%f.json' % value
    with open(os.path.join(FLAGS.output_dir, output_filename), 'w') as f:
      json.dump(report, f)

    reports_dict[value] = json_report
  return reports_dict


def mle_agent_epsilon_1_resource_all_dynamics():
  """Run experiments on a greedy-epsilon mle agent, epsilon=0.1, across dynamics."""
  dynamic_values_to_test = [0.0, 0.01, 0.05, 0.1, 0.15]
  experiment = _setup_experiment()
  experiment.agent_class = probability_matching_agents.MLEProbabilityMatchingAgent
  experiment.agent_params = probability_matching_agents.MLEProbabilityMatchingAgentParams(
  )
  experiment.agent_params.burn_steps = 25
  experiment.agent_params.window = 100

  reports_dict = {}

  for value in dynamic_values_to_test:
    print('Running an experiment...')
    experiment.env_params.dynamic_rate = value
    json_report = attention_allocation_experiment.run(experiment)
    report = json.loads(json_report)

    print('\n\nMLE Agent, 4 attention units, epsilon=0.1')
    _print_discovered_missed_incidents_report(value, report)
    output_filename = 'mle_epsilon.1_4units_%f.json' % value
    with open(os.path.join(FLAGS.output_dir, output_filename), 'w') as f:
      json.dump(report, f)

    reports_dict[value] = json_report
  return reports_dict


def mle_agent_epsilon_5_resource_all_dynamics():
  """Run experiments on a greedy-epsilon mle agent, epsilon=0.6, across dynamics."""
  dynamic_values_to_test = [0.0, 0.01, 0.05, 0.1, 0.15]
  experiment = _setup_experiment()
  experiment.agent_class = probability_matching_agents.MLEProbabilityMatchingAgent
  experiment.agent_params = probability_matching_agents.MLEProbabilityMatchingAgentParams(
  )
  experiment.agent_params.burn_steps = 25
  experiment.agent_params.epsilon = 0.5
  experiment.agent_params.window = 100

  reports_dict = {}

  for value in dynamic_values_to_test:
    experiment.env_params.dynamic_rate = value
    json_report = attention_allocation_experiment.run(experiment)
    report = json.loads(json_report)

    print('\n\nMLE Agent, 4 attention units, epsilon=0.5')
    _print_discovered_missed_incidents_report(value, report)
    output_filename = 'mle_epsilon.5_4units_%f.json' % value
    with open(os.path.join(FLAGS.output_dir, output_filename), 'w') as f:
      json.dump(report, f)

    reports_dict[value] = json_report
  return reports_dict


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  uniform_reports = uniform_agent_resource_all_dynamics()
  mle1_reports = mle_agent_epsilon_1_resource_all_dynamics()
  mle5_reports = mle_agent_epsilon_5_resource_all_dynamics()

  agent_names = [
      'uniform', 'proportional epsilon=0.1', 'proportional epsilon=0.5'
  ]
  dataframe = attention_allocation_experiment_plotting.create_dataframe_from_results(
      agent_names, [uniform_reports, mle1_reports, mle5_reports])

  attention_allocation_experiment_plotting.plot_total_miss_discovered(
      dataframe, os.path.join(FLAGS.output_dir, 'dynamic_rate_across_agents'))
  attention_allocation_experiment_plotting.plot_occurence_action_single_dynamic(
      json.loads(uniform_reports[0.1]),
      os.path.join(FLAGS.output_dir, 'uniform_incidents_actions_over_time'))
  attention_allocation_experiment_plotting.plot_occurence_action_single_dynamic(
      json.loads(mle1_reports[0.1]),
      os.path.join(FLAGS.output_dir,
                   'proportional_incidents_actions_over_time'))


if __name__ == '__main__':
  app.run(main)
