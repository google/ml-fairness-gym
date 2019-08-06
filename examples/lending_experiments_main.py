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

"""Code to recreate plots for the KDD workshop paper, Fairness is not Static.

This code creates the plots for the lending section of the paper.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
import os

from absl import app
from absl import flags
from absl import logging
from examples import lending


flags.DEFINE_string('plotting_dir', '/tmp/plots',
                    'Path location to store plots.')
flags.DEFINE_integer('num_steps', 20000, 'Number of steps to run experiments.')
flags.DEFINE_integer('seed', 200, 'Random seed.')

FLAGS = flags.FLAGS


MAX_UTIL_TITLE = 'Max Utility'
EQ_OPP_TITLE = 'Eq. Opportunity'


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  plotting_dir = os.path.join(FLAGS.plotting_dir, '%d' % FLAGS.seed)
  os.makedirs(plotting_dir)
  lending.plot_steady_state_distribution(plotting_dir)

  experiment_fn = functools.partial(
      lending.Experiment,
      group_0_prob=0.5,
      interest_rate=1.0,
      bank_starting_cash=10000,
      seed=FLAGS.seed,
      num_steps=FLAGS.num_steps,
      burnin=200,
      cluster_shift_increment=0.01,
      include_cumulative_loans=True,
      return_json=False)

  logging.info(
      'experiment params %s %s',
      experiment_fn(threshold_policy=lending.EQUALIZE_OPPORTUNITY).asdict(),
      experiment_fn(threshold_policy=lending.MAXIMIZE_REWARD).asdict())
  env, agent = experiment_fn(
      threshold_policy=lending.EQUALIZE_OPPORTUNITY).scenario_builder()
  logging.info('environment params %s', env.initial_params.asdict())
  logging.info('agent params %s', agent.params.asdict())

  # Run two long-running experiments and then use the results for all further
  # plotting.
  equality_of_opportunity_result = experiment_fn(
      threshold_policy=lending.EQUALIZE_OPPORTUNITY).run()
  maximize_reward_result = experiment_fn(
      threshold_policy=lending.MAXIMIZE_REWARD).run()

  lending.plot_credit_distribution(
      maximize_reward_result['metric_results']['initial_credit_distribution'],
      'Initial',
      path=os.path.join(plotting_dir, 'initial.pdf'))
  lending.plot_credit_distribution(
      maximize_reward_result['metric_results']['final_credit_distributions'],
      title=MAX_UTIL_TITLE,
      path=os.path.join(plotting_dir, 'max_utility.pdf'))
  lending.plot_credit_distribution(
      equality_of_opportunity_result['metric_results']
      ['final_credit_distributions'],
      title=EQ_OPP_TITLE,
      path=os.path.join(plotting_dir, 'equalize_opportunity.pdf'))

  cumulative_loans = {
      'max reward':
          maximize_reward_result['metric_results']['cumulative_loans'],
      'equal-opp':
          equality_of_opportunity_result['metric_results']['cumulative_loans']
  }
  lending.plot_cumulative_loans(
      cumulative_loans, os.path.join(plotting_dir, 'cumulative_loans.pdf'))
  logging.info('Maxutil agent: %s', maximize_reward_result['agent'])
  logging.info('Equalized opportunity agent: %s',
               equality_of_opportunity_result['agent'])

  logging.info('Equality of opportunity recall: %s',
               equality_of_opportunity_result['metric_results']['recall'])
  logging.info('Maxutil recall: %s',
               maximize_reward_result['metric_results']['recall'])
  logging.info('Equality of opportunity profit rate: %s',
               equality_of_opportunity_result['metric_results']['profit rate'])
  logging.info('Maxutil profit rate: %s',
               maximize_reward_result['metric_results']['profit rate'])


if __name__ == '__main__':
  app.run(main)
