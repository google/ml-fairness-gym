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
"""Tests for lending.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from examples import lending
import simplejson as json


class LendingExampleTest(absltest.TestCase):

  def test_experiment_runs_with_default_parameters(self):
    # Tests that the experiment can run.
    result = lending.Experiment(num_steps=100).run()
    # Tests that the result is a valid json string.
    result = json.loads(result)

  def test_short_run_recall_is_perfect(self):
    # Run for fewer steps than the burnin - this should give 100% recall
    # since during the burnin period, all loans are accepted.
    result = lending.Experiment(num_steps=10).run()
    result = json.loads(result)
    self.assertEqual(result['metric_results']['recall'], {'0': 1.0, '1': 1.0})


if __name__ == '__main__':
  absltest.main()
