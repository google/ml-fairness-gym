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

# Lint as: python3
"""Code to aggregate multiple runs for cumulative-recall measurements.

To recreate the recall-gap plot from the lending experiments in the paper,
first run:

for seed in {0..99}
  experiments/amc_fat_2020/lending_experiments_main.py --seed=${seed}
done

Then run:

experiments/amc_fat_2020/aggregate_lending_recall_values.py
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
from absl import app
from absl import flags
import file_util
from experiments import lending_plots
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('source_path', '/tmp/fairness_gym/lending_plots',
                    'Root directory where results are stored. Under the root '
                    'directory there should be subdirectories for each random '
                    'seed containing run statistics.')


flags.DEFINE_string('plotting_dir', '/tmp/fairness_gym/lending_plots',
                    'Path to write plots.')

flags.DEFINE_integer('num_trials', 100, 'Number of trials to aggregate.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  recall_values = {}
  for setting in ['static', 'dynamic']:
    recall = []
    for seed in range(FLAGS.num_trials):
      with file_util.open(
          os.path.join(FLAGS.source_path, str(seed), 'cumulative_recall_%s.txt')
          % setting, 'r') as infile:
        recall.append(np.loadtxt(infile))
      recall_values[setting] = np.vstack(recall)

  lending_plots.plot_cumulative_recall_differences(
      recall_values,
      path=os.path.join(FLAGS.plotting_dir, 'combined_simpsons_paradox.pdf'))


if __name__ == '__main__':
  app.run(main)
