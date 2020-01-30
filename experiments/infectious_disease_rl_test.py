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
"""Tests for infectious_disease_rl."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
from absl.testing import absltest
from experiments import infectious_disease_rl


class InfectiousDiseaseRlTest(absltest.TestCase):

  def test_dopamine_train_eval(self):
    """Tests that both train and eval can execute without raising errors."""
    tmpdir = tempfile.mkdtemp()
    runner = infectious_disease_rl.dopamine_train(
        base_dir=tmpdir,
        hidden_layer_size=10,
        gamma=0.5,
        learning_rate=0.1,
        num_train_steps=10,
        network='chain')
    infectious_disease_rl.dopamine_eval(runner, patient0=0)

  def test_negative_delta_percent_sick(self):
    reward_fn = infectious_disease_rl.NegativeDeltaPercentSick(base=0.25)
    observation = {'health_states': [0, 1, 2, 1]}
    # 50% are infected. The base is 25%, so the negative delta is -0.25
    self.assertEqual(reward_fn(observation), -0.25)
    # Using the same observation a second time. Now the percent infected has not
    # changed, so negative delta should be 0.
    self.assertEqual(reward_fn(observation), 0)


if __name__ == '__main__':
  absltest.main()
