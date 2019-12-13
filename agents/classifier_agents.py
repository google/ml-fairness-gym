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
"""Agents whose actions are determined by the output of a binary classifier."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import logging
from typing import Any, Callable, List, Mapping, Optional, Text, Union

import attr
import core
import params
from agents import threshold_policies
import gym
import numpy as np
from sklearn import linear_model


@attr.s
class TrainingExample(object):
  observation = attr.ib()  # type: Mapping[Text, np.ndarray]
  features = attr.ib()  # type: Any
  label = attr.ib()  # type: Optional[int]
  action = attr.ib()  # type: Optional[int]
  weight = attr.ib(default=1.0)  # type: float

  def is_labeled(self):
    return self.label is not None


class TrainingCorpus(object):
  """Class to hold a collection of TrainingExamples."""

  def __init__(self, examples=None):
    self.examples = []  # type: List[TrainingExample]
    if examples is not None:
      self.examples = list(examples)

  def remove_unlabeled(self):
    return TrainingCorpus(
        examples=[example for example in self.examples if example.is_labeled()])

  def add(self, example):
    self.examples.append(example)

  def get_features(
      self,
      stratify_by=None):
    """Returns features of the training examples.

    Args:
      stratify_by: observation key to stratify by.

    Returns:
      If stratify is None, returns a list of features. Otherwise a dictionary
      of lists of features where the keys are the values of the stratify_by key.
    """
    if stratify_by is None:
      return [example.features for example in self.examples]
    stratified_features = collections.defaultdict(list)
    for example in self.examples:
      stratified_features[tuple(example.observation.get(stratify_by))].append(
          example.features)
    return stratified_features

  def get_labels(
      self,
      stratify_by=None):
    """Returns labels of the training examples.

    Args:
      stratify_by: observation key to stratify by.

    Returns:
      If stratify is None, returns a list of labels. Otherwise a dictionary
      of lists of labels where the keys are the values of the stratify_by key.
    """
    if stratify_by is None:
      return [example.label for example in self.examples]
    stratified_labels = collections.defaultdict(list)
    for example in self.examples:
      stratified_labels[tuple(example.observation.get(stratify_by))].append(
          example.label)
    return stratified_labels

  def get_weights(
      self,
      stratify_by=None):
    """Returns weights of the training examples.

    Args:
      stratify_by: observation key to stratify by.

    Returns:
      If stratify is None, returns a list of weights. Otherwise a dictionary
      of lists of weights where the keys are the values of the stratify_by key.
    """
    if stratify_by is None:
      return [example.weight for example in self.examples]
    stratified_weights = collections.defaultdict(list)
    for example in self.examples:
      stratified_weights[tuple(example.observation.get(stratify_by))].append(
          example.weight)
    return stratified_weights


@attr.s
class ScoringAgentParams(core.Params):
  """Parameter class for ScoringAgents."""
  default_action_fn = attr.ib()  #  type: Callable[[], Any]
  feature_keys = attr.ib(factory=list)  # type: List[Text]

  # Some environment use features which are one-hot and can be "thresholded"
  # by converting the one-hot vector to an integer and applying the threshold
  # in that way.
  convert_one_hot_to_integer = attr.ib(default=False)

  # Whether to continue training. Classifiers will still accumulate labeled
  # data while they are frozen.
  freeze_classifier_after_burnin = attr.ib(default=False)
  cost_matrix = attr.ib(default=params.CostMatrix(tp=1, tn=1, fp=-1, fn=-1))
  burnin = attr.ib(default=-1)
  threshold_policy = attr.ib(
      default=threshold_policies.ThresholdPolicy.SINGLE_THRESHOLD)
  use_propensity_score_weighting = attr.ib(default=False)
  group_key = attr.ib(default="")

  # A function which takes last action and last observation as inputs and
  # determines whether to skip training the classifier on this step.
  skip_retraining_fn = attr.ib(
      default=None)  # Optional[Callable[[Any, Dict[Text, Any]], bool]]


@attr.s
class ScoringAgent(core.Agent):
  """Abstract base class of an agent that acts based on a thresholded score.

  Inheriting classes must implement _get_features, _score_transform, and
  _score_transform_update.
  """
  observation_space = attr.ib()  # type: gym.Space
  reward_fn = attr.ib()  # type: core.RewardFn
  params = attr.ib()  # type: ScoringAgentParams
  frozen = attr.ib(default=False)  # type: bool
  action_space = attr.ib(
      factory=lambda: gym.spaces.Discrete(2))  # type: gym.Space
  rng = attr.ib(factory=np.random.RandomState)  # type: np.random.RandomState

  global_threshold = attr.ib(default=0.)
  group_specific_thresholds = attr.ib(factory=dict)
  global_threshold_history = attr.ib(factory=list)
  group_specific_threshold_history = attr.ib(
      factory=lambda: collections.defaultdict(list))
  target_recall_history = attr.ib(
      factory=lambda: collections.defaultdict(list))

  _step = attr.ib(default=0)
  _last_action = attr.ib(default=None)
  _training_corpus = attr.ib(factory=TrainingCorpus)

  # Maintain a global threshold if group-specific thresholds are not available.

  _last_observation = attr.ib(default=None)
  _last_action = attr.ib(default=None)

  def _act_impl(self, observation, reward, done):
    self.global_threshold_history.append(self.global_threshold)
    for group, thresh in self.group_specific_thresholds.items():
      self.group_specific_threshold_history[group].append(thresh)
      self.target_recall_history[group].append(thresh.tpr_target)

    self._record_training_example(self._last_observation, self._last_action,
                                  reward)

    if self._step < self.params.burnin:
      action = self.params.default_action_fn()
      # No reason to train during burnin. Train on the first non-burnin step.
    else:
      self._train()
      if self.params.freeze_classifier_after_burnin:
        self.frozen = True

      group_id = observation.get(self.params.group_key)
      if group_id is not None:
        # Convert np.array to a hashable form.
        group_id = tuple(group_id)
      features = self._get_features(observation)
      score = self._score_transform([features])[0]
      action = int(score >= self._get_threshold(group_id))

    self._last_observation = observation
    self._last_action = action
    self._step += 1
    return action

  def _train(self):

    if self.frozen:
      return

    if self.params.use_propensity_score_weighting:
      # TODO(): Implement propensity score weighting.
      raise NotImplementedError(
          "propensity score weighting training is not implemented YET!")

    training_corpus = self._training_corpus.remove_unlabeled()

    # Don't train if there's nothing to train on.
    if not training_corpus.examples:
      return

    if self.params.skip_retraining_fn and self.params.skip_retraining_fn(
        self._last_action, self._last_observation):
      return

    self._score_transform_update(training_corpus)
    self._set_thresholds(training_corpus)

  def _set_thresholds(self, training_corpus):
    self.global_threshold = threshold_policies.single_threshold(
        predictions=self._score_transform(training_corpus.get_features()),
        labels=training_corpus.get_labels(),
        weights=training_corpus.get_weights(),
        cost_matrix=self.params.cost_matrix)

    if self.params.threshold_policy == threshold_policies.ThresholdPolicy.EQUALIZE_OPPORTUNITY:
      self.group_specific_thresholds = (
          threshold_policies.equality_of_opportunity_thresholds(
              group_predictions=self._recursively_apply_score_transform(
                  training_corpus.get_features(
                      stratify_by=self.params.group_key)),
              group_labels=training_corpus.get_labels(
                  stratify_by=self.params.group_key),
              group_weights=training_corpus.get_weights(
                  stratify_by=self.params.group_key),
              cost_matrix=self.params.cost_matrix,
              rng=self.rng))

  def _get_threshold(self, group_id):
    # Try to get a group specific threshold but fall back to the global
    # threshold if not available.
    if group_id in self.group_specific_thresholds:
      return self.group_specific_thresholds[group_id].sample()
    return self.global_threshold

  def _recursively_apply_score_transform(self, features):
    if isinstance(features, dict):
      return {
          key: self._recursively_apply_score_transform(value)
          for key, value in features.items()
      }
    return self._score_transform(features)

  def _record_training_example(self, observation, action, reward):
    if action is None or observation is None:
      return

    self._training_corpus.add(
        TrainingExample(
            observation=observation,
            action=action,
            label=reward,
            features=self._get_features(observation)))

  def _get_features(self, observation):
    raise NotImplementedError

  def _score_transform(self, features):
    raise NotImplementedError

  def _score_transform_update(self, training_corpus):
    raise NotImplementedError

  def debug_string(self):
    return "My thresholds are %s and %s" % (self.global_threshold,
                                            self.group_specific_thresholds)


class ThresholdAgent(ScoringAgent):
  """Agent that learns thresholds for a single 1D feature."""

  def _get_features(self, observation):
    if len(self.params.feature_keys) != 1:
      raise ValueError(
          "Threshold agent can only have a single feature key. Got %s" %
          self.params.feature_keys)
    feature = observation.get(self.params.feature_keys[0])
    if self.params.convert_one_hot_to_integer:
      return [np.argmax(feature)]
    return [feature]

  def _score_transform(self, features):
    return [feat[0] for feat in features]

  def _score_transform_update(self, training_corpus):
    pass


@attr.s
class ClassifierAgent(ScoringAgent):
  """Agent that learns to transform features and apply a threshold."""

  _classifier = attr.ib(
      factory=lambda: linear_model.LogisticRegression(solver="lbfgs"))

  def _score_transform_update(self, training_corpus):
    try:
      self._classifier.fit(training_corpus.get_features(),
                           training_corpus.get_labels())
    except ValueError:
      logging.warning(
          "Could not fit the classifier at step %d. This may be because there is  "
          "not enough data. Consider using a longer burn-in period to ensure "
          "that sufficient data is collected. See the exception for more "
          "details on why it was raised.", self._step
      )
      raise

  def _score_transform(self, features):
    return self._classifier.predict_proba(features)[:, 1]

  def _get_features(self, observation):
    return np.concatenate([
        observation.get(feature_key) for feature_key in self.params.feature_keys
    ], 0).ravel()
