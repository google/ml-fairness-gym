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

"""Environment for testing scenarios with college admissions.

  This implements an environment and its contestants in a two player strategic
  classification game (Stackelberg competition).
  It simulates two groups: A (advantaged group) and B (disadvantaged group),
  which present as 1D features (test scores) in the current implementation.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
import attr
import core
import params
import gin
from gym import spaces
import numpy as np

from typing import Any, Dict, Optional, Text, List, Tuple, Mapping, Union

_EPSILON_VALUE = 1e-6
_UNSELECTED_INDICATOR = 2


@gin.configurable
@attr.s
class Params(core.Params):
  """College Admissions Params."""
  # Number of applicants that appear each round.
  num_applicants = attr.ib(default=20)  # type: int
  # Min and Max values of the test score for each applicant.
  score_params = attr.ib(default=params.MinMax(max=1,
                                               min=0))  # type: params.MinMax
  # Stopping params.
  # Max steps to run the interaction for.
  max_steps = attr.ib(default=100000)  # type: int

  # Tunable Params.
  # Parameters of the population feature distribution.
  feature_params = attr.ib(
      default=params.GMM(
          mix_weight=[0.6, 0.4], mu=[0.8, 0.6],
          sigma=[0.1, 0.3]))  # type: params.GMM
  # To calculate true thresholds, as mu + factor * mu (factor can be -ve).
  true_threshold_factor = attr.ib(default=0.1)  # type: float
  # Params for the noise ditribution for noise added to threshold.
  noise_params = attr.ib(
      default=params.BoundedGaussian(
          max=0.3, min=-0.3, mu=0.1,
          sigma=0.01))  # type: params.BoundedGaussian

  # Scalar multiplier for cost (of a feature) per group.
  # Note: We codify B's disadvantage by requiring Cost A < Cost B.
  # pylint: disable=g-long-lambda
  group_cost = attr.ib(factory=lambda: {
      0: 0.8,
      1: 1.2
  })  # type: Mapping[int, float]
  # pylint: enable=g-long-lambda
  gaming = attr.ib(default=True)  # type: bool
  # Control amount of gaming. If inf, then allow maximum allowable by cost fn.
  # else if a value in [0, 1], applicant can only change by that many points.
  gaming_control = attr.ib(default=np.inf)  # type: float
  # Whether to subsidize
  subsidize = attr.ib(default=False)  # type: bool
  # Group 2 is subsidized by only allowing it to bear subsidy_beta cost.
  # subsidy_beta can be a value in [0, 1]. Percentage of cost B actually pays,
  # higher this is higher is the cost paid by B.
  subsidy_beta = attr.ib(default=0.8)  # type: float
  # Distribution of noise. One of beta or gaussian.
  noise_dist = attr.ib(default='gaussian')  # type: Text
  # Add noise to unmanipulated features.
  noisy_features = attr.ib(default=False)  # type: bool
  # Add noise to observed threshold.
  noisy_threshold = attr.ib(default=False)  # type: bool


@gin.configurable
@attr.s(cmp=False)
class State(core.State):
  """College Admissions State."""
  # Random state.
  rng = attr.ib(factory=np.random.RandomState)  # type: np.random.RandomState

  # Parameters.
  params = attr.ib(factory=Params)  # type: Params

  steps = attr.ib(default=0)  # type: int
  applicant_groups = attr.ib(factory=list)  # type: List[int]
  true_eligible = attr.ib(factory=list)  # type: List[int]
  true_thresholds = attr.ib(factory=dict)  # type: Mapping[int, float]
  original_test_scores = attr.ib(factory=list)  # type: List[float]
  test_scores_x = attr.ib(factory=list)  # type: List[float]
  test_scores_y = attr.ib(factory=list)  # type: List[float]
  individual_burden = attr.ib(factory=list)  # type: List[float]
  decision_threshold = attr.ib(default=0)  # type: Optional[float]
  selected_applicants = attr.ib(factory=list)  # type: np.ndarray
  selected_ground_truth = attr.ib(factory=list)  # type: np.ndarray
  num_manipulated = attr.ib(default=0)  # type: int


@gin.configurable
class CollegeAdmissionsEnv(core.FairnessEnv):
  """College Admissions Environment.


  This implements an environment that can sample N candidates for college
  admission, to simulate a two-player Stackelberg game.

  The candidates present as test_scores in [0,1] with latent group-ids {0,1}.

  The two-player Stackelberg game proceeds as follows:
    1. The environment generates N applicants each round with ground truth
      eligibility to be admitted. The applicants belong to two groups and have
      different Gaussian distributions and base rates of eligibility.
    2. The learner/jury publishes its classifier [one dimensional case], a
      threshold in [0, 1], indicating threshold of test scores and is group
      agnostic. The agent, if following an epsilon greedy policy also publishes
      its epsilon probability, which by default is 0.
    3. The candidates can then manipulate their scores based on the published
      threshold. The candidates will only manipulate if they can flip from being
      ineligible to eligible (if candidates are already eligible no change in
      scores takes place) and when the cost of doing so is below a maximum cost.
    4. All candidates (after score manipulation) who exceed the threshold are
      accepted.
    5. The learner updates its model using the observations (manipulated scores)
      and the true results of which candidates were eligible.
  """

  def __init__(self, user_params = None):
    """Initializes the College Admissions environment with initial params.

    Args:
      user_params: Dict. Any params not passed will take default values in
        Params.
    Raise:
      ValueError: If provided params not as expected.
    """
    # TODO(): make parameter handling consistent across environments.
    # Called env_params unlike in other environments because this environment
    # incorporates params with the default to get the comprehensive environment
    # params.
    env_params = Params()
    if user_params is not None:
      env_params = Params(**user_params)
    # The jury's action is a dict containing the threshold which specifies a 1D
    # threshold on scores above which applicants will be admitted and an epsilon
    # probability value, which specifies the probability value for an
    # epsilon greedy agent and is 0 by default.
    self.action_space = spaces.Dict({
        'threshold':
            spaces.Box(
                low=env_params.score_params.min,
                high=env_params.score_params.max,
                dtype=np.float32,
                shape=()),
        'epsilon_prob':
            spaces.Box(low=0, high=1, dtype=np.float32, shape=())
    })  # type: spaces.Space

    # The observations include test scores, [0, 1], eligibility of selected
    # applicants, ground truth for selected candidates and applicant group ids.
    self.observable_state_vars = {
        'test_scores_y':
            spaces.Box(
                low=env_params.score_params.min,
                high=env_params.score_params.max,
                dtype=np.float32,
                shape=(env_params.num_applicants,)),
        'selected_applicants':
            spaces.MultiBinary(env_params.num_applicants),
        'selected_ground_truth':
            spaces.MultiDiscrete([3] * env_params.num_applicants),
        'applicant_groups':
            spaces.MultiBinary(env_params.num_applicants)
    }  # type: Dict[Text, spaces.Space]
    super(CollegeAdmissionsEnv, self).__init__(env_params)
    if env_params.gaming_control != np.inf and (env_params.gaming_control > 1 or
                                                env_params.gaming_control < 0):
      raise ValueError('Gaming control needs to be in [0, 1]')

    if env_params.noise_dist not in ['gaussian', 'beta']:
      raise ValueError('Undefined noise distribution.')
    self._state_init()

  def _state_init(self, rng = None):
    state = State(
        rng=rng or np.random.RandomState(),
        params=copy.deepcopy(self.initial_params),
        true_thresholds=self._get_true_thresholds(),
        selected_applicants=np.array([0] * self.initial_params.num_applicants),
        selected_ground_truth=np.array([_UNSELECTED_INDICATOR] *
                                       self.initial_params.num_applicants))
    self.state = self._sample_next_state_vars(state)

  def _get_true_thresholds(self):
    env_params = self.initial_params
    mu = env_params.feature_params.mu
    return {
        group_id:
        mu[group_id] + (env_params.true_threshold_factor * mu[group_id])
        for group_id in range(2)
    }

  def _step_impl(self, state, action):
    """Run one timestep of environment's dynamics."""
    state.steps += 1
    state.decision_threshold = action['threshold']
    # Change scores according to jury threshold and calculate social burden.
    state.test_scores_y, state.individual_burden = (
        self._manipulate_features(state, action))
    # Select based on modified scores.
    state.selected_applicants, state.selected_ground_truth = (
        self._select_candidates(state, action))
    # Sample next state features and groundtruth labels.
    state = self._sample_next_state_vars(state)
    return state

  def reset(self):
    self._state_init(self.state.rng)
    return super(CollegeAdmissionsEnv, self).reset()

  def _is_done(self):
    return self.state.steps > self.state.params.max_steps

  def _manipulate_features(
      self, state,
      action):
    """Returns manipulated features and the individual burden.

    Args:
      state: State. Which represents the current state of the environment.
      action: An action from action space containing agent threshold and epsilon
        probability.

    Returns:
      manipulated_features: List of values in [0, 1] representing features that
       may have been manipulated given threshold.
      individual_burden: List of float values representing the cost of changing
       features to make it a favourable decision, even if this is above the
       maximum allowable cost and thus ineligible to change.
    """
    env_params = state.params
    individual_burden = []
    manipulated_features = []
    state.num_manipulated = 0
    if env_params.gaming:
      for score, group_id in zip(state.test_scores_x, state.applicant_groups):
        if score < action['threshold']:
          noise = 0
          if env_params.noisy_threshold:
            noise = self._add_noise(state.rng)
          changed_score = np.clip(action['threshold'] + noise,
                                  env_params.score_params.min,
                                  env_params.score_params.max)
          change_cost = self._change_cost(score, changed_score, group_id)
          # Saving all individual burdens even if not eligible to change.
          individual_burden.append(change_cost)
          if np.isinf(env_params.gaming_control):
            # Can change to anywhere in [0, 1].
            max_change_cost = 100
          else:
            # Can only change by max_amount_of_gaming points.
            max_change_cost = self._change_cost(
                score, score + env_params.gaming_control, group_id)
          # If epsilon-greedy agent, fewer candidates likely to game, since
          # there is an epsilon prob they can get good outcome without gaming.
          # Thus candidates will game if:
          # cost_of_manipulation > cost_of_no_maipulation
          # with change in outcome from 0 to 1. Basically :
          # or,  1>= (1 - epsilon)(new_outcome) + epsilon(1) - change_cost >=
          # (1- epsilon) (old_outcome) + epsilon
          # which reduces to 1 >= 1 - epsilon - change_cost >= 0
          cost_of_manipulation = (
              self._outcome(action['threshold'], changed_score) - change_cost -
              action['epsilon_prob'])

          if (0 <= cost_of_manipulation <=
              1) and (change_cost < max_change_cost + _EPSILON_VALUE):
            # Can manipulate & get positive decision.
            manipulated_features.append(changed_score)
            state.num_manipulated += 1
          else:
            manipulated_features.append(score)
        else:
          manipulated_features.append(score)
          individual_burden.append(0)
      return manipulated_features, individual_burden
    else:
      return state.test_scores_x, [0] * env_params.num_applicants

  def _select_candidates(
      self, state,
      action):
    """Returns predictions and groundtruth, given manipulated scores.

    Args:
      state: State. Representing the current state of the environment.
      action: An action from action space. Contains dict with agent's threshold
        and epsilon probability.

    Returns:
      predictions: Array of floats representing manipulated scores.
      selected_ground_truth: Array indicating whether a selected candidate was
        actually eligible(1) or ineligible (0) and 2 if not selected.
    """
    predictions = []
    for changed_score in state.test_scores_y:
      predictions.append(
          self._epsilon_outcome(
              rng=state.rng, action=action, feature=changed_score))
    if len(predictions) != len(state.true_eligible):
      raise ValueError('Shape of predictions and labels is inconsistent')
    selected_ground_truth = np.array([
        eligible if selected == 1 else _UNSELECTED_INDICATOR
        for eligible, selected in zip(state.true_eligible, predictions)
    ])
    return np.array(predictions), selected_ground_truth

  def _sample_next_state_vars(self, state):
    """Updates state with features and groundtruth labels for next state."""
    score_params = state.params.score_params
    state.original_test_scores, state.applicant_groups = (
        self._sample_applicants(state.rng))
    if state.params.noisy_features:
      state.test_scores_x = [
          np.clip(score + self._add_noise(state.rng), score_params.min,
                  score_params.max) for score in state.original_test_scores
      ]
    else:
      state.test_scores_x = copy.deepcopy(state.original_test_scores)

    state.true_eligible = [
        self._outcome(state.true_thresholds[group_id],
                      score) for group_id, score in zip(
                          state.applicant_groups, state.original_test_scores)
    ]
    return state

  def _sample_applicants(self, rng):
    """Samples test scores and group ids for applicants at each round.

    Args:
      rng: random number generator.

    Returns:
      Lists of test_scores and group_id with len as num_applicants.
    """
    env_params = self.initial_params
    original_test_scores = []
    group_ids = []
    mixture_ids = rng.multinomial(env_params.num_applicants,
                                  env_params.feature_params.mix_weight)
    for group_id, count in enumerate(mixture_ids):
      group_ids.extend([group_id] * count)
      original_test_scores.extend(
          np.clip(
              rng.normal(env_params.feature_params.mu[group_id],
                         env_params.feature_params.sigma[group_id], count),
              env_params.score_params.min, env_params.score_params.max))
    return original_test_scores, group_ids

  def _outcome(self, threshold,
               feature):
    """Returns the outcome (eligible/not eligible).

    Args:
      threshold: Float. Scalar value of threshold.
      feature: Float. Scalar value of applicant test score.

    Returns:
      1 if eligible 0 otherwise.
    """
    return int(feature >= threshold)

  def _epsilon_outcome(self, rng, action,
                       feature):
    """Returns the outcome (eligible/not eligible) with epsilon prob random.

    Implements an epsilon greedy selection mechanism where it randomly allows
    candidates with probability p and uses threshold for selection with prob
    1 - p.

    Args:
      rng: random number generator.
      action: An action from action space containing agent threshold and epsilon
        probablity.
      feature: Float. Scalar value of applicant test score.

    Returns:
      1 if eligible 0 otherwise.
    """
    if rng.rand() < action['epsilon_prob']:
      return 1
    return int(feature >= action['threshold'])

  def _cost_function(self, applicant_feature, group_id):
    """Returns group specific cost, and applies subsidies if required."""
    env_params = self.initial_params
    cost = applicant_feature * env_params.group_cost[group_id]
    if env_params.subsidize and group_id == 1:  # subsidize disdavantaged group.
      return env_params.subsidy_beta * cost
    else:
      return cost

  def _change_cost(self, original_score, changed_score,
                   group_id):
    return (self._cost_function(changed_score, group_id) -
            self._cost_function(original_score, group_id))

  def _add_noise(self, rng):
    """Adds noise to threshold.

    Args:
      rng: Random number generator.

    Returns:
      float in [noise_params.min, noise_params.max] to be added to the
      manipulated score.
    Raises:
      ValueError: If noise_params.min > noise_params.mu.
    """
    env_params = self.initial_params
    noise_params = env_params.noise_params
    if noise_params.sigma > 0:
      multiplier = 1
      if env_params.noise_dist == 'gaussian':
        if (noise_params.min >= noise_params.mu) or (noise_params.mu >=
                                                     noise_params.max):
          raise ValueError(
              'Invalid Noise Params. Min: %f, Max:%f, Mu: %f' %
              (noise_params.min, noise_params.max, noise_params.mu))

        return np.clip(noise_params.sigma * rng.randn() + noise_params.mu,
                       noise_params.min, noise_params.max)
      else:
        # noise_params.mu=beta, noise_params.sigma=alpha.
        assert noise_params.mu > 0
        multiplier = -1 if rng.random_sample() < 0.5 else 1
        return multiplier * rng.beta(noise_params.sigma, noise_params.mu)
    else:
      # no noise added if sigma <=0.
      return 0
