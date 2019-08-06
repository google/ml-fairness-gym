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

"""Tests for attention_allocation_experiment.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import test_util
from environments import attention_allocation
from examples import attention_allocation_experiment
import simplejson as json


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
      num_runs=1,
      num_steps=10,
      seed=0,
      env_class=attention_allocation.LocationAllocationEnv,
      env_params=_get_base_env_params(),
      agent_class=test_util.DummyAgent)


class AttentionAllocationExperimentTest(absltest.TestCase):

  def test_report_valid_json(self):
    # Tests that the experiment can run.
    experiment = _setup_experiment()
    result = attention_allocation_experiment.run(experiment)
    # Tests that the result is a valid json string.
    result = json.loads(result)

  def test_report_is_replicable(self):
    experiment = _setup_experiment()
    json_report = attention_allocation_experiment.run(experiment)
    json_report_second = attention_allocation_experiment.run(experiment)
    self.assertEqual(json_report, json_report_second)

  def test_multiprocessing_works(self):
    experiment = _setup_experiment()
    experiment.num_workers = 5
    experiment.num_steps = 10
    experiment.num_runs = 5
    result = attention_allocation_experiment.run(experiment)
    result = json.loads(result)


if __name__ == '__main__':
  absltest.main()
