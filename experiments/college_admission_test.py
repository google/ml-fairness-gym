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
"""Tests for fairness_gym.experiments.college_admission."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl.testing import absltest
import params
from experiments import college_admission

import numpy as np


TEST_PARAMS = {
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


class CollegeAdmissionTest(absltest.TestCase):

  def test_experiment_runs_with_default_parameters(self):
    # Tests that the experiment can run.
    test_experiment = college_admission.CollegeExperiment(
        agent_type='fixed', num_steps=10)
    result = test_experiment.run_experiment()
    # Tests that the result is a valid json string.
    result = json.loads(result)

  def test_short_run_final_threshold(self):
    # Accuracy with static agent should be around 0.55 (true threshold)
    # with TEST_PARAMS at equilibrium.
    test_experiment = college_admission.CollegeExperiment(
        num_steps=30, burnin=25, agent_type='static', env_config=TEST_PARAMS)
    result = test_experiment.run_experiment()
    result = json.loads(result)
    self.assertTrue(
        np.isclose(
            result['metric_results']['final_threshold'], 0.55, atol=1e-2))


if __name__ == '__main__':
  absltest.main()
