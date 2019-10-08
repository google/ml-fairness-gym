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

"""Agents for college admissions environments."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import core
import params
import rewards
from agents import threshold_policies
import gin
from gym import spaces
import numpy as np
from typing import Any, List, Mapping, Text, Optional, Callable

_SCORE_MAX = 1
_SCORE_MIN = 0
_UNSELECTED_INDICATOR = 2


@gin.configurable
class FixedJury(core.Agent):
  """Baseline college admission agent that uses a fixed threshold.

  It also has the option to be epsilon greedy, so it also emits the epsilon
  probability for each round.
  Thus, according to this policy, if the epsilon value is non-zero,
  each candidate is allowed in with a prob epsilon_prob else candidates
  whose scores are above or equal to threshold are allowed in with prob
  1- epsilon_prob. If epsilon_prob is zero, all candidates above the threshold
  only are allowed.
  """

  def __init__(
      self,
      action_space,
      reward_fn,
      observation_space,
      threshold = 0.5,
      epsilon_greedy = False,
      initial_epsilon_prob = 0.7,
      decay_steps = 10,
      epsilon_prob_decay_rate = 0.02,
  ):
    """Initializes the agent.

    Args:
     action_space: a `gym.Space` that contains valid actions.
     reward_fn: a `RewardFn` object.
     observation_space: a `gym.Space` that contains valid observations.
     threshold: Fixed threshold.
     epsilon_greedy: Bool. Whether we want this agent to follow an epsilon
       greedy policy.
     initial_epsilon_prob: Float. Initial value of probablity for an epsilon
       greedy agent.
     decay_steps: A positive integer.
     epsilon_prob_decay_rate: A positive float.
    """
    if reward_fn is None:
      reward_fn = rewards.NullReward()
    super(FixedJury, self).__init__(action_space, reward_fn, observation_space)
    self._threshold = threshold
    self._epsilon_greedy = epsilon_greedy
    self._initial_epsilon_prob = initial_epsilon_prob
    self._decay_rate = epsilon_prob_decay_rate
    self._decay_steps = decay_steps
    self._steps = 0
    self.rng = np.random.RandomState()

  def _get_epsilon_prob(self):
    """Returns epsilon_prob if epsilon_greedy is True, else 0 (default)."""
    if self._epsilon_greedy:
      return self._initial_epsilon_prob * self._decay_rate**(
          self._steps / self._decay_steps)
    return 0

  def initial_action(self):
    return {
        'threshold': np.array(self._threshold),
        'epsilon_prob': np.array(self._get_epsilon_prob())
    }

  def _act_impl(self, observation, reward,
                done):
    """Returns a fixed threshold.

    Args:
      observation: An observation in self.observation_space.
      reward: A scalar value that can be used as a supervising signal.
      done: A boolean indicating whether the episode is over.

    Raises:
      core.EpisodeDoneError if `done` is True.
      core.InvalidObservationError if observation is not in
        `self.observation_space`.
    """
    if done:
      raise core.EpisodeDoneError('Called act on a done episode.')

    if not self.observation_space.contains(observation):
      raise core.InvalidObservationError('Invalid observation: %s' %
                                         observation)
    self._steps += 1
    return self.initial_action()


@gin.configurable
class NaiveJury(FixedJury):
  """College admission scenario that simulates a naive jury in Stackelberg game.

  The jury here publishes its decision threshold at each round, which is then
  used by the applicants to decide how they will change their scores.
  This jury picks the threshold that maximizes reward on manipulated scores.
  """

  def __init__(self,
               action_space,
               reward_fn,
               observation_space,
               feature_selection_fn = None,
               label_fn = None,
               freeze_classifier_after_burnin = False,
               threshold = 0.5,
               burnin = -1,
               cost_matrix = None,
               epsilon_greedy = False,
               initial_epsilon_prob = 0.7,
               decay_steps = 10,
               epsilon_prob_decay_rate = 0.02):
    """Initializes the jury.

    Args:
      action_space: a `gym.Space` that contains valid actions.
      reward_fn: a `RewardFn` object.
      observation_space: a `gym.Space` that contains valid observations.
      feature_selection_fn: Function that returns a feature vector suitable for
        training from observations.
      label_fn: Function that returns a label from observations and reward.
      freeze_classifier_after_burnin: If True, the classifier will freeze
        classifier after learning a model after burnin steps.
      threshold: Initial threshold.
      burnin: Number of steps before using a learned policy.
      cost_matrix: a fairness_policies.CostMatrix object.
      epsilon_greedy: Bool. Whether we want to have an epsilon greedy agent.
      initial_epsilon_prob: Float. Initial value of probablity for an epsilon
        greedy agent.
      decay_steps: A positive integer.
      epsilon_prob_decay_rate: A positive float.
    """
    super(NaiveJury, self).__init__(
        action_space,
        reward_fn,
        observation_space,
        threshold=threshold,
        epsilon_greedy=epsilon_greedy,
        initial_epsilon_prob=initial_epsilon_prob,
        decay_steps=decay_steps,
        epsilon_prob_decay_rate=epsilon_prob_decay_rate)
    self._initial_threshold = threshold
    self._features = []
    self._labels = []
    self._burnin = burnin
    self._freeze_classifier_after_burnin = freeze_classifier_after_burnin
    self._feature_selection_fn = feature_selection_fn or self._get_default_features
    self._label_fn = label_fn or self._label_fn

    if not cost_matrix:
      self._cost_matrix = params.CostMatrix(tp=1, tn=1, fp=-1, fn=-1)
    else:
      self._cost_matrix = cost_matrix

  def _act_impl(self, observation, reward,
                done):
    """Returns an action from `self.action_space`.

    Args:
      observation: An observation in self.observation_space.
      reward: A scalar value that can be used as a supervising signal.
      done: A boolean indicating whether the episode is over.

    Raises:
      core.EpisodeDoneError if `done` is True.
      core.InvalidObservationError if observation is not in
        `self.observation_space`.
    """
    if done:
      raise core.EpisodeDoneError('Called act on a done episode.')

    if not self.observation_space.contains(observation):
      raise core.InvalidObservationError('Invalid observation: %s' %
                                         observation)

    features = self._feature_selection_fn(observation)
    labels = self._label_fn(observation)

    # No need to build up a dataset of observations when the model is frozen.
    if not self._freeze_classifier():
      # Update dataset and fit new model
      self._update_dataset(features, labels)
      self._train_model()
    # Since this method is called afetr initial action in the first step.
    self._steps += 1
    if self._steps < self._burnin:
      return {
          'threshold': np.asarray(self._initial_threshold),
          'epsilon_prob': np.asarray(self._get_epsilon_prob())
      }

    return {
        'threshold': np.asarray(self._threshold),
        'epsilon_prob': np.asarray(self._get_epsilon_prob())
    }

  def _freeze_classifier(self):
    return self._steps > self._burnin and self._freeze_classifier_after_burnin

  def _update_dataset(self, features, labels):
    """Stores the most recent labeled examples.

    Args:
      features: A 1-d numpy array features denoting features of N candidates.
        The features are test scores in [0, 1].
      labels: A 1-d numpy array denoting the corresponding labels.

    Raises:
      ValueError if features and labels are not the same shape.
    """
    if len(features) != len(labels):
      raise ValueError('Features and labels inconsistent')
    self._features.extend(features)
    self._labels.extend(labels)

  def _train_model(self):
    """Trains the internal model using all features seen so far."""
    if not self._can_train():
      return
    self._threshold = np.clip(
        threshold_policies.single_threshold(
            predictions=self._features,
            labels=self._labels,
            weights=None,
            cost_matrix=self._cost_matrix), _SCORE_MIN, _SCORE_MAX)

  def _can_train(self):
    return set(self._labels) == {0, 1} and self._features

  def _get_default_features(self, observations):
    """Returns a list of features from observations."""
    return [
        score for score, eligible in zip(observations['test_scores_y'],
                                         observations['selected_ground_truth'])
        if eligible != _UNSELECTED_INDICATOR
    ]

  def _label_fn(self, observations):
    """Returns labels (a list) from observations."""
    return [
        label for label in observations['selected_ground_truth']
        if label != _UNSELECTED_INDICATOR
    ]


@gin.configurable
class RobustJury(NaiveJury):
  """College admission scenario, implements a robust jury in Stackelberg game.

  This agent assumes a burnin period with 0 threshold, so it is able to see
  unmanipulated scores in order to learn a good decision threshold after the
  burnin period, which is then fixed for rounds after that.
  At each round during the burnin period, the jury adds the maximum allowable
  gaming to the scores it sees. After burnin, it learns a threshold, that
  minimizes overall error as described in Algorithm 1 in Hardt 2015
  https://arxiv.org/abs/1506.06980.

  """

  def __init__(self,
               action_space,
               reward_fn,
               observation_space,
               group_cost,
               subsidize = False,
               subsidy_beta = 0.8,
               gaming_control = np.inf,
               label_fn = None,
               burnin = 10,
               epsilon_greedy = False,
               initial_epsilon_prob = 0.7,
               decay_steps = 10,
               epsilon_prob_decay_rate = 0.02):
    super(RobustJury, self).__init__(
        action_space,
        reward_fn,
        observation_space,
        threshold=0,
        feature_selection_fn=self._get_maximum_manipulated_features,
        label_fn=label_fn,
        cost_matrix=params.CostMatrix(tp=1, tn=1, fp=-1, fn=-1),
        freeze_classifier_after_burnin=True,
        burnin=burnin,
        epsilon_greedy=epsilon_greedy,
        initial_epsilon_prob=initial_epsilon_prob,
        decay_steps=decay_steps,
        epsilon_prob_decay_rate=epsilon_prob_decay_rate)

    self._group_cost = group_cost
    self._subsidize = subsidize
    self._subsidy_beta = subsidy_beta
    self._gaming_control = gaming_control

    if burnin < 2:
      raise ValueError(
          'This agent expects a longer burnin period, to work as expected.')

  # TODO (): Retain original observed scores in self._features.

  def _get_maximum_manipulated_features(
      self, observations):
    """Returns scores that have been increased by max gaming ability."""
    max_allowed_change = self._get_max_allowed_score_change(observations)
    robust_features = []
    for score, eligible, group in zip(observations['test_scores_y'],
                                      observations['selected_ground_truth'],
                                      observations['applicant_groups']):
      if eligible != _UNSELECTED_INDICATOR:
        robust_features.append(
            np.clip(score + max_allowed_change[group], _SCORE_MIN, _SCORE_MAX))
    return robust_features

  def _get_max_allowed_score_change(
      self, observations):
    """Calculate the max allowed change in score given costs and subsidy.

    Args:
      observations: An observation in self.observation_space.

    Returns:
      The max number of steps that each group can change given group costs and
      subsidies.
    """
    # Change in utility / cost = max score change
    # Change in utility = (1 - epsilon) * 1
    cost_a = self._group_cost[0]
    cost_b = self._group_cost[1]
    eps = self._get_epsilon_prob()
    if self._subsidize:
      cost_b *= self._subsidy_beta
    if self._gaming_control == np.inf:
      return [(1 - eps) / cost_a, (1 - eps) / cost_b]
    else:
      return [
          np.min(((1 - eps) / cost_a, self._gaming_control)),
          np.min(((1 - eps) / cost_b, self._gaming_control))
      ]
