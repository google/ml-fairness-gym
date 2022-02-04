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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import core  # pylint: disable=unused-import
import runner_lib
import test_util  # pylint: disable=unused-import
from agents import random_agents  # pylint: disable=unused-import
import gin


class RunnerLibTest(absltest.TestCase):

  def setUp(self):
    super(RunnerLibTest, self).setUp()
    gin.clear_config()

  def test_configured_runner_takes_correct_number_of_steps(self):
    gin.parse_config("""
      runner_lib.Runner.env_class = @test_util.DummyEnv
      runner_lib.Runner.agent_class = @random_agents.RandomAgent
      runner_lib.Runner.num_steps = 10
      runner_lib.Runner.seed = 1337
      runner_lib.Runner.metric_classes = {"num_steps": @test_util.DummyMetric}
    """)
    runner = runner_lib.Runner()
    results = runner.run()
    self.assertEqual(10, results['metrics']['num_steps'])

  def test_configured_parametrized_runner_takes_correct_number_of_steps(self):
    gin.parse_config("""
      runner_lib.Runner.env_class = @test_util.DummyEnv
      runner_lib.Runner.env_params_class = @core.Params
      runner_lib.Runner.agent_class = @random_agents.RandomAgent
      runner_lib.Runner.num_steps = 10
      runner_lib.Runner.seed = 1234
      runner_lib.Runner.metric_classes = {"num_steps": @test_util.DummyMetric}
    """)
    runner = runner_lib.Runner()
    results = runner.run()
    self.assertEqual(10, results['metrics']['num_steps'])

  def test_environment_underspecification_raises(self):
    gin.parse_config("""
      runner_lib.Runner.env_class = None
      runner_lib.Runner.env_callable = None
    """)
    with self.assertRaises(TypeError):
      runner = runner_lib.Runner()
      runner.run()


if __name__ == '__main__':
  absltest.main()
