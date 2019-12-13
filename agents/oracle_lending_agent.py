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

"""Lending agent with access to ground truth distributions of applicants."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from agents import classifier_agents


class OracleThresholdAgent(classifier_agents.ThresholdAgent):
  """Threshold agent with oracle access to distributional data."""

  def __init__(self, action_space, reward_fn, observation_space, params, env):
    super(OracleThresholdAgent, self).__init__(
        action_space=action_space,
        reward_fn=reward_fn,
        observation_space=observation_space,
        params=params)
    self.env = env

  def _record_training_example(self, observation, action, reward):
    self._training_corpus = classifier_agents.TrainingCorpus()
    applicant_distribution = self.env.state.params.applicant_distribution
    for component in applicant_distribution.components:
      for cluster, weight in zip(component.components, component.weights):
        prob_default = cluster.will_default.p
        observation = {
            "bank_cash": 1.0,
            "applicant_features": cluster.features.mean,
            "group": cluster.group_membership.mean
        }
        self._training_corpus.add(
            classifier_agents.TrainingExample(
                observation=observation,
                action=1,
                label=0,
                weight=prob_default * weight + 1e-6,
                features=self._get_features(observation)))
        self._training_corpus.add(
            classifier_agents.TrainingExample(
                observation=observation,
                action=1,
                label=1,
                weight=(1 - prob_default) * weight + 1e-6,
                features=self._get_features(observation)))
