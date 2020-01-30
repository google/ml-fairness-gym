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

# Lint as: python2 python3
"""Tests for infectious_disease."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from experiments import infectious_disease


class InfectiousDiseaseTest(absltest.TestCase):

  def test_run_experiment(self):
    experiment = infectious_disease.Experiment(
        graph_name='karate',
        infection_probability=0,
        infected_exit_probability=0,
        agent_constructor=infectious_disease.NullAgent)
    run_result = experiment.run()
    metrics = run_result['metric_results']
    for measurement in metrics['sick-days']:
      # There is exactly one sick person every day.
      self.assertEqual(measurement, 1)

    for measurement in metrics['states']:
      # `states` should hold a health state for every individual. There are 34
      # members in the karate club graph.
      self.assertLen(measurement, 34)

      # Everyone should be at state 0 except for one individual who is 1.
      self.assertSameElements(measurement, {0, 1})
      self.assertEqual(sum(measurement), 1)


if __name__ == '__main__':
  absltest.main()
