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
"""Experiment class for running college admission experiments."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy

import attr
import core
import run_util
from agents import college_admission_jury
from environments import college_admission
from metrics import error_metrics
from metrics import value_tracking_metrics
import more_itertools
import numpy as np


@attr.s
class CollegeExperiment(object):
  """Main Experiment class to run college admission experiments."""
  #===================================================
  # Experiment Params.
  #===================================================

  agent_type = attr.ib()
  # Experiment Params
  # The number of steps to simulate.
  num_steps = attr.ib(default=100)
  # Random seed.
  seed = attr.ib(default=1)
  # Environment Params
  env_config = attr.ib(factory=dict)

  # Tunable Agent Params

  agent_threshold = attr.ib(default=0.5)
  epsilon_greedy = attr.ib(default=False)
  initial_epsilon_prob = attr.ib(default=0.1)
  epsilon_prob_decay_rate = attr.ib(default=0.02)
  burnin = attr.ib(default=20)
  decay_steps = attr.ib(default=20)

  #===================================================
  # Metrics Related Functions.
  #===================================================

  def realign_history(self, history):
    """"Realigns history so as to be compatible with auditors.

    Since the true applicants groups, unmanipulated test scores and
    true_eligible
    are generated before the agent's action, they are in the previous state, so
    we
    push them one step ahead in history and ignore the first step.

    Args:
      history: A list of tuples of state, action pairs.

    Returns:
      A realigned history with changed state, action pairs.
    """
    realign_variables = [
        'test_scores_x', 'applicant_groups', 'true_eligible', 'params'
    ]
    realigned_history = []
    for (state, _), (next_state,
                     next_action) in more_itertools.pairwise(history):
      new_history_point = core.HistoryItem(
          state=copy.deepcopy(next_state), action=copy.deepcopy(next_action))
      for variable in realign_variables:
        setattr(new_history_point.state, variable, getattr(state, variable))
      realigned_history.append(new_history_point)
    return realigned_history

  def stratify_by_group(self, step):
    return list(step.state.applicant_groups)

  def accuracy_nr_fn(self, step):
    """Returns if predictions are correct."""
    return [
        pred == groundtruth for pred, groundtruth in zip(
            step.state.selected_applicants, step.state.true_eligible)
    ]

  def selection_fn_social_burden_eligible_auditor(self, step):
    """This returns individual burden of only eligible candidates."""
    return np.array(step.state.true_eligible) * np.array(
        step.state.individual_burden)

  #===================================================
  # Build up environment and agent.
  #===================================================

  def build_scenario(self):
    """Returns agent and env according to provided params."""
    env = college_admission.CollegeAdmissionsEnv(user_params=self.env_config)

    if self.agent_type == 'robust':
      agent = college_admission_jury.RobustJury(
          action_space=env.action_space,
          reward_fn=(lambda x: 0),
          observation_space=env.observation_space,
          group_cost=env.initial_params.group_cost,
          subsidize=env.initial_params.subsidize,
          subsidy_beta=env.initial_params.subsidy_beta,
          gaming_control=env.initial_params.gaming_control,
          epsilon_greedy=self.epsilon_greedy,
          initial_epsilon_prob=self.initial_epsilon_prob,
          decay_steps=self.decay_steps,
          epsilon_prob_decay_rate=self.epsilon_prob_decay_rate,
          burnin=self.burnin)

    elif self.agent_type == 'static':
      agent = college_admission_jury.NaiveJury(
          action_space=env.action_space,
          reward_fn=(lambda x: 0),
          observation_space=env.observation_space,
          threshold=0,
          epsilon_greedy=self.epsilon_greedy,
          initial_epsilon_prob=self.initial_epsilon_prob,
          epsilon_prob_decay_rate=self.epsilon_prob_decay_rate,
          decay_steps=self.decay_steps,
          freeze_classifier_after_burnin=True,
          burnin=self.burnin)

    elif self.agent_type == 'continuous':
      agent = college_admission_jury.NaiveJury(
          action_space=env.action_space,
          reward_fn=(lambda x: 0),
          observation_space=env.observation_space,
          threshold=0,
          epsilon_greedy=self.epsilon_greedy,
          initial_epsilon_prob=self.initial_epsilon_prob,
          epsilon_prob_decay_rate=self.epsilon_prob_decay_rate,
          freeze_classifier_after_burnin=False,
          decay_steps=self.decay_steps,
          burnin=self.burnin)
    else:
      agent = college_admission_jury.FixedJury(
          action_space=env.action_space,
          reward_fn=(lambda x: 0),
          observation_space=env.observation_space,
          threshold=self.agent_threshold,
          epsilon_greedy=self.epsilon_greedy,
          decay_steps=self.decay_steps,
          initial_epsilon_prob=self.initial_epsilon_prob,
          epsilon_prob_decay_rate=self.epsilon_prob_decay_rate)

    return env, agent

  #===================================================
  # Run Experiment.
  #===================================================
  def run_experiment(self):
    """Main experiment runner."""
    env, agent = self.build_scenario()

    social_burden = value_tracking_metrics.AggregatorMetric(
        env=env,
        selection_fn=self.selection_fn_social_burden_eligible_auditor,
        modifier_fn=None,
        stratify_fn=self.stratify_by_group,
        realign_fn=self.realign_history,
        calc_mean=True)
    accuracy = error_metrics.AccuracyMetric(
        env=env,
        numerator_fn=self.accuracy_nr_fn,
        denominator_fn=None,
        stratify_fn=self.stratify_by_group,
        realign_fn=self.realign_history)
    overall_accuracy = error_metrics.AccuracyMetric(
        env=env,
        numerator_fn=self.accuracy_nr_fn,
        denominator_fn=None,
        # pylint: disable=g-long-lambda
        stratify_fn=lambda x:
        [1 for _ in range(env.initial_params.num_applicants)],
        realign_fn=self.realign_history)
    overall_social_burden = value_tracking_metrics.AggregatorMetric(
        env=env,
        selection_fn=self.selection_fn_social_burden_eligible_auditor,
        modifier_fn=None,
        # pylint: disable=g-long-lambda
        stratify_fn=lambda x:
        [1 for _ in range(env.initial_params.num_applicants)],
        realign_fn=self.realign_history,
        calc_mean=True)
    final_threshold = value_tracking_metrics.FinalValueMetric(
        env=env,
        state_var='decision_threshold',
        realign_fn=self.realign_history)

    metrics = [
        social_burden, accuracy, overall_accuracy, overall_social_burden,
        final_threshold
    ]
    metric_names = [
        'social_burden', 'accuracy', 'overall_accuracy',
        'overall_social_burden', 'final_threshold'
    ]
    metric_results = run_util.run_stackelberg_simulation(
        env, agent, metrics, self.num_steps, self.seed)
    return core.to_json({
        'metric_results': dict(zip(metric_names, metric_results)),
    })
