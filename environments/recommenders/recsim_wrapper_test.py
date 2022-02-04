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

# Lint as: python3
"""Tests for recsim.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import test_util
from environments.recommenders import recsim_wrapper
from recsim.environments import interest_exploration


class RecommenderTest(absltest.TestCase):

  def test_interest_exploration_can_run(self):
    env_config = {
        'num_candidates': 5,
        'slate_size': 2,
        'resample_documents': False,
        'seed': 100,
    }
    params = recsim_wrapper.Params(
        recsim_env=interest_exploration.create_environment(env_config))
    env = recsim_wrapper.RecsimWrapper(params)
    test_util.run_test_simulation(env=env, stackelberg=True)

  def test_interest_exploration_can_run_with_resampling(self):
    env_config = {
        'num_candidates': 5,
        'slate_size': 2,
        'resample_documents': True,
        'seed': 100,
    }
    params = recsim_wrapper.Params(
        recsim_env=interest_exploration.create_environment(env_config))
    env = recsim_wrapper.RecsimWrapper(params)
    test_util.run_test_simulation(env=env, stackelberg=True)


if __name__ == '__main__':
  absltest.main()
