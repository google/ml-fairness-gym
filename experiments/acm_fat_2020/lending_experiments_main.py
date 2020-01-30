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

"""Code to recreate plots for lending experiments.

Code to recreate the recall-gap plot can be found in
aggregate_lending_recall_values.py
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
import os

from absl import app
from absl import flags
from absl import logging
import file_util
from experiments import lending
from experiments import lending_plots


flags.DEFINE_string('plotting_dir', '/tmp/fairness_gym/lending_plots',
                    'Path location to store plots.')
flags.DEFINE_integer('num_steps', 20000, 'Number of steps to run experiments.')
flags.DEFINE_integer('seed', 200, 'Random seed.')
flags.DEFINE_float('cluster_shift_increment', 0.01, 'Inverse population size.')


FLAGS = flags.FLAGS


DELAYED_IMPACT_CLUSTER_PROBS = (
    (0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0),
    (0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0),
)

ACM_FAT_2020_PLOTS = frozenset({
    lending_plots.PlotTypes.CREDIT_DISTRIBUTIONS,
    lending_plots.PlotTypes.CUMULATIVE_LOANS,
    lending_plots.PlotTypes.THRESHOLD_HISTORY,
    lending_plots.PlotTypes.MEAN_CREDIT_OVER_TIME,
    lending_plots.PlotTypes.CUMULATIVE_RECALLS,
})


DELAYED_IMPACT_CLUSTER_PROBS = (
    (0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0),
    (0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0),
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  plotting_dir = os.path.join(FLAGS.plotting_dir, '%d' % FLAGS.seed)
  file_util.makedirs(plotting_dir)
  lending_plots.plot_steady_state_distribution(plotting_dir)

  experiment_fn = functools.partial(
      lending.Experiment,
      cluster_probabilities=DELAYED_IMPACT_CLUSTER_PROBS,
      group_0_prob=0.5,
      interest_rate=1.0,
      bank_starting_cash=10000,
      seed=FLAGS.seed,
      num_steps=FLAGS.num_steps,
      # No need for burn in since the agent is an oracle agent.
      burnin=1,
      cluster_shift_increment=FLAGS.cluster_shift_increment,
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
  # The static experiment removes the delayed effects of defaulting on future
  # credit scores.
  static_equality_of_opportunity_result = experiment_fn(
      threshold_policy=lending.EQUALIZE_OPPORTUNITY,
      cluster_shift_increment=0).run()
  maximize_reward_result = experiment_fn(
      threshold_policy=lending.MAXIMIZE_REWARD).run()

  lending_plots.do_plotting(
      maximize_reward_result,
      equality_of_opportunity_result,
      static_equality_of_opportunity_result,
      plotting_dir,
      options=ACM_FAT_2020_PLOTS)


if __name__ == '__main__':
  app.run(main)
