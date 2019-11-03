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
"""Agent who allocates resource with a probability in proportion to belief distribution."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
from typing import Any, Callable, List, Mapping, Optional, Text, TypeVar

from absl import logging
import attr
import core
import rewards
from spaces import multinomial
import gym
import numpy as np
from scipy import stats
from six.moves import range
from statsmodels.base import model


def _get_added_vector_features(observation,
                               n,
                               keys = None):
  """Returns  combined specified observation values that are 1D of length n.

  This function takes all observation values specified by keys (all values if
  keys==None). It then adds element-wise all values that are length n and 1D.

  Args:
    observation: Dictionary of observations.
    n: target length of features array.
    keys: optionally the subset of observations to consider.
  """
  if keys is None:
    keys = list(observation.keys())
  features = np.zeros(n)
  for key in keys:
    if observation[key].ndim == 1 and len(observation[key]) == n:
      features = features + observation[key]
  return features


def _allocate_proportional_to_beliefs(rng,
                                      n_resource,
                                      beliefs):
  """Returns array of attention allocations proportional to beliefs.

  Args:
    rng: a
    n_resource: An int defining the amount of attention that can be allocated -
      the returned array has sum equal to n_resource.
    beliefs: An array of integers reflecting the belief of the current incident
      rate across groups.

  Returns:
    An array with length equal to length of input counts that sums to
      n_resource.
  """
  beliefs_total = float(np.sum(beliefs))
  if beliefs_total > 0:
    probabilities = [(belief / beliefs_total) for belief in beliefs]
  else:
    # If counts are all 0, we want equal probabilities.
    probabilities = [(1. / len(beliefs)) for _ in beliefs]
  return rng.multinomial(n_resource, probabilities)


@attr.s
class AllocationAgentParams(core.Params):
  feature_selection_fn = attr.ib(
      default=None
  )  # type: Optional[Callable[[Mapping[Text, Any], int, Optional[List[Text]]], np.ndarray]]
  observation_adjustment_fn = attr.ib(
      default=None
  )  # type: Optional[Callable[[np.random.RandomState, List[float], Mapping[Text, Any]], Mapping[Text, Any]]]


_AllocationAgentParamsBound = TypeVar(
    "_AllocationAgentParamsBound", bound=AllocationAgentParams)


class AllocationAgent(core.Agent):
  """Base class for allocating agents.

  AllocationAgents maintain a belief about the underlying rates of incidents
  for each location and allocate based on those beliefs according to some
  scheme.

  Main API method of this class is:

    act: Take an observation and reward and returns an allocation as an action.

  Subclasses must overwrite and implement _update_beliefs(features, beliefs)
  and _allocate(n_resource, beliefs).
  """

  def __init__(self,
               action_space,
               reward_fn,
               observation_space,
               params = None):

    super(AllocationAgent, self).__init__(action_space, reward_fn,
                                          observation_space)
    if reward_fn is None:
      reward_fn = rewards.NullReward()
    if params is None:
      params = AllocationAgentParams()
    self.params = params
    self._n_bins = len(action_space.nvec)

    self.rng = np.random.RandomState()
    self._n_resource = self.action_space.n

    self.beliefs = np.zeros(self._n_bins).tolist()
    self.feature_selection_fn = params.feature_selection_fn or (
        lambda obs: _get_added_vector_features(obs, self._n_bins)
    )  # type: Callable

  def _act_impl(self, observation, reward,
                done):
    """Returns an action from 'self.action_space'.

    Args:
      observation: An observation in self.observation_space.
      reward: A scalar value that can be used as a supervising signal.
      done: A boolean indicating whether the episode is over.

    Returns:
      An action from self.action space.

    Raises:
      core.EpisodeDoneError if `done` is True.
      core.InvalidObservationError if observation is not contained in
        'self.observation_space'.
      gym.error.InvalidAction if the generated action to return is not contained
        in 'self.action_space'.
    """
    if done:
      raise core.EpisodeDoneError("Called act on a done episode.")

    if not self.observation_space.contains(observation):
      raise core.InvalidObservationError("Invalid ovservation: %s" %
                                         observation)
    if self.params.observation_adjustment_fn:
      observation = self.params.observation_adjustment_fn(
          self.rng, self.beliefs, observation)

    features = self.feature_selection_fn(observation)
    self.beliefs = self._update_beliefs(features, self.beliefs)
    action = self._allocate(self._n_resource, self.beliefs)

    if not self.action_space.contains(action):
      raise gym.error.InvalidAction("Invalid action: %s" % action)

    return action

  ####################################################################
  # Method to be overridden by each proportional allocator agent.    #
  ####################################################################

  def _allocate(self, n_resource, beliefs):
    """Returns an array of attention allocations across groups.

    Args:
      n_resource: An int defining the amount of attention that can be allocated
        - the returned array has sum equal to n_resource.
      beliefs: An array of integers reflecting the current belief of
        distribution of target reward across groups.

    Returns:
      An array with length equal to length of input counts that sums to
        n_resource.
    """
    raise NotImplementedError

  def _update_beliefs(self, features,
                      beliefs):
    """Returns an updated beliefs vector reflecting the new belief distribution.

    Args:
      features: An numpy array of input features for computing new belief.
      beliefs: A 1-D numpy vector containing the current belief distribution.

    Returns:
      An array with same shape as input beliefs that has been updated based on
      input features.
    """
    raise NotImplementedError


@attr.s
class NaiveProbabilityMatchingAgentParams(AllocationAgentParams):
  decay_prob = attr.ib(default=0.01)  # type: float


class NaiveProbabilityMatchingAgent(AllocationAgent):
  """Naive probability matching agent whose beliefs are sums of counts of features."""

  def __init__(self,
               action_space,
               reward_fn,
               observation_space,
               params = None):

    if params is None:
      params = NaiveProbabilityMatchingAgentParams()

    super(NaiveProbabilityMatchingAgent,
          self).__init__(action_space, reward_fn, observation_space, params)

    self.beliefs = np.zeros(self._n_bins, dtype=np.uint32)

  def _allocate(self, n_resource, beliefs):
    return _allocate_proportional_to_beliefs(self.rng, n_resource, beliefs)

  def _update_beliefs(self, features,
                      beliefs):
    """Returns updated belief reflecting the agent's belief distribution.

    Updates the belief counts with the given observation and a decay.
    The decay is computed as a binomial draw of the belief counts where the
    success probability in the binomial distribution is set to decay_prob.

    Args:
      features: A feature in 'observation_space'. Must be to be a 1-D array
        where each item is the observed counts for the bin represented by that
        index of the array.
      beliefs: An array of integers reflecting the current belief of
        distribution of target reward across groups.

    Returns:
      An array with same length as counts that has been updated with the
        observation and decay factor.

    Raises:
      BadFeatureFnError: if the input features are not expected 1-D array form.
    """
    if (len(features) != len(beliefs) or features.ndim != 1):
      raise core.BadFeatureFnError()

    assert len(features) == len(beliefs)
    decay = self.rng.binomial(beliefs, self.params.decay_prob)
    updated_beliefs = [
        beliefs[i] + features[i] - decay[i] for i in range(len(beliefs))
    ]
    return updated_beliefs

  def _linear_rejection_sampling(self, rng, beliefs, observation, keys=None):
    """Returns observation subject to rejection sampling the provided keys fields.

    If keys is None, as by default, function returns observations unmodified.

    Args:
      rng: A numpy.random.RandomState() object.
      beliefs: Vector of numbers used to determine rejection thresholds.
      observation: Dict observation to be subsampled.
      keys: Optional list of keys for observation fields that should be
        subsampled. None by default. keys should correspond to fields that are
        1D vectors of length equal to beliefs length.

    Returns:
      Observation: Copy of the observation dict with the given key fields
        subsampled.

    Raises:
      KeyError: If keys provided in keys input do not correspond to fields in
        observation that are vectors of length equal to beliefs.
    """
    if not keys:
      return observation

    # Deep copy so observation is not mutated in case it needs to be reused.
    observation_copy = copy.deepcopy(observation)

    total = np.sum(beliefs)
    contributions = np.array([
        (total - belief) / float(total) for belief in beliefs
    ])
    max_contribution = np.max(contributions)
    normalized_contributions = contributions / max_contribution

    for index in range(len(normalized_contributions)):
      if rng.random_sample() >= normalized_contributions[index]:
        for key in keys:
          if len(normalized_contributions) != len(
              observation_copy[key]) or observation_copy[key].ndim != 1:
            raise KeyError(
                "Key %s field is not a 1D vector with length equal to len(beliefs)."
                % key)
          observation_copy[key][index] = 0

    return observation_copy


class _CensoredPoisson(model.GenericLikelihoodModel):
  """Class modeling a censored poisson likelihood model.

  For use by MLEProbabilityMatchingAgent.
  """

  def __init__(self, endog, exog=None, **kwds):
    """Initializes the model.

    Args:
      endog : array-like dependent variable.
      exog : array-like independent variables.
      **kwds: other kwds.
    """
    if exog is None:
      exog = np.zeros_like(endog)

    super(_CensoredPoisson, self).__init__(endog, exog, **kwds)
    # Setting xnames manually so model has correct number of parameters.
    self.data.xnames = ["x1"]

  def nloglikeobs(self, params):
    """Return the negative loglikelihood of endog given the params for the model.

    Args:
      params: Vector containing parameters for the likelihood model.

    Returns:
      Negative loglikelihood of self.endog computed with given params.
    """
    lambda_ = params[0]

    ll_output = self._LL(self.endog, rate=lambda_)

    return -np.log(ll_output)

  def fit(self, start_params=None, maxiter=10000, **kwds):
    """Override fit to call super's fit with desired start params."""
    if start_params is None:
      lambda_start = self.endog[:, 0].mean()
      start_params = np.array([lambda_start])

    return super(_CensoredPoisson, self).fit(
        start_params=start_params, maxiter=maxiter, **kwds)

  def _LL(self, data, rate):
    """Return likelihood of the given data with the given rate as the poisson parameter."""
    observed_count = np.array(data[:, 0])
    allocated_count = np.array(data[:, 1])
    output_array = np.zeros(len(observed_count))

    assert len(observed_count) == len(allocated_count)

    output_array += (observed_count < allocated_count) * stats.poisson.pmf(
        observed_count, rate)
    # The probability of observing a count equal to the allocated count is
    # the tail of the poisson pmf from the observed_count value.
    # Summing the tail is equivalent to 1-the sum of the head up to
    # observed_count which is the value of the cdf at the observed_count-1.
    output_array += (observed_count == allocated_count) * (
        1.0 - stats.poisson.cdf(observed_count - 1, rate))

    return output_array


@attr.s
class MLEProbabilityMatchingAgentParams(AllocationAgentParams):
  burn_steps = attr.ib(default=20)  # type: int
  interval = attr.ib(default=10)  # type: int
  window = attr.ib(default=0)  # type: int
  epsilon = attr.ib(default=0.1)  # type: float


_MLEProbabilityMatchingAgentParamsBound = TypeVar(
    "_MLEProbabilityMatchingAgentParamsBound",
    bound=MLEProbabilityMatchingAgentParams)


class MLEProbabilityMatchingAgent(AllocationAgent):
  """Probability matching agent that estimates poisson parameter for each bin.

  The agent then allocates with probability in proportion to the
  parameters.

  Assumes specific likelihood model defined by _TruncatedPoisson() as the
  likelihood model for the observations.
  """

  def __init__(self, action_space,
               reward_fn, observation_space,
               params):

    if params is None:
      params = MLEProbabilityMatchingAgentParams()

    super(MLEProbabilityMatchingAgent, self).__init__(action_space, reward_fn,
                                                      observation_space, params)

    self.data = [[] for _ in range(self._n_bins)]
    self.last_allocation = None
    self.n_steps = 0

  def _allocate(self, n_resource, beliefs):
    # With probability epsilon allocate with uniform probability.
    # With probability 1-epsilon, allocate according to belief.
    if self.rng.random_sample() < self.params.epsilon:

      self.last_allocation = self.action_space.sample()
    else:
      self.last_allocation = _allocate_proportional_to_beliefs(
          self.rng, n_resource, beliefs)
    return self.last_allocation

  def _update_beliefs(self, features, beliefs):
    """Returns updated belief reflecting the agent's estimated belief distribution.

    Updates the belief every self.interval number of steps using MLE to estimate
    the parameters underlying the given features for each bin and uses those
    parameters as its new belief.

    Args:
      features: A feature in 'observation_space'. Must be to be a 1-D array
        where each item is the observed counts for the bin represented by that
        index of the array.
      beliefs: An array of integers reflecting the current belief of
        distribution of target reward across groups.

    Returns:
      An array with same length as beliefs that has been updated with new
      parameter estimates if the current self.n_steps is a multiple of
      self.interval.
      Otherwise, returns unchanged beliefs.

    Raises:
      BadFeatureFnError: if the input features are not expected 1-D array form.
    """
    self.n_steps += 1
    if self.last_allocation is None:
      return beliefs
    for i_bin in range(self._n_bins):
      self.data[i_bin].append((features[i_bin], self.last_allocation[i_bin]))
      if self.params.burn_steps <= self.n_steps and self.n_steps % self.params.interval == 0:
        ll_model = _CensoredPoisson(
            np.array(self.data[i_bin][-self.params.window:]))
        results = ll_model.fit(disp=0)
        beliefs[i_bin] = results.params[0]
    return beliefs


@attr.s
class MLEGreedyAgentParams(MLEProbabilityMatchingAgentParams):
  # Alpha is the constraint on equality of incident discovery across bins.
  # Default of 1 means there is no constraint.
  alpha = attr.ib(default=1.0)  # type: float

  poisson_truncation_val = attr.ib(default=50)  # type: int


class MLEGreedyAgent(MLEProbabilityMatchingAgent):
  """Greedy agent that allocates to maximize yield under fairness constraint.

  Assumes specific likelihood model defined by _CensoredPoisson() as the
  likelihood model for the observations.

  This agent is a version of the agent described in Elzayn et al.'s paper
  "Fair Algorithms for Learning in Allocation Problems".

  The agent allocates to maximize likelihood of each allocation discovering
  another incident, while constraining allocations to approximately equalize
  candidate discovery probability.
  The candidate discovery probability is defined as
  f_i(v_i) E{c_i~C_i}[min(v_i, c_i)/c_i], where c_i are the incidents occurred
  in bin i (as drawn from distribution with lambda=C_i) and v_i are the units of
  attention allocated to bin i.
  The fairness constraint, alpha forces allocations to satisfy
  |f_i(v_i)-f_j(v_j)] <= alpha for all pairs of bins i and j.
  """

  def __init__(self,
               action_space,
               reward_fn,
               observation_space,
               params = None):

    if params is None:
      params = MLEGreedyAgentParams()

    super(MLEGreedyAgent, self).__init__(action_space, reward_fn,
                                         observation_space, params)

    self.data = [[] for _ in range(self._n_bins)]
    self.last_allocation = None
    self.n_steps = 0

  def _calculate_tail_probability(self, x, rate):
    """Calculates the probability c>=x for all c for a poisson(rate)."""
    return 1 - stats.poisson.cdf(x - 1, rate)

  def _construct_approx_fi_table(self, n_bins, rates, n_resource):
    """Constructs a n_bins by n_resource+1 matrix of f_i values.

    Args:
      n_bins: int number of bins.
      rates: rate for each bin.
      n_resource: int number of resources available to allocate.

    Returns:
      np.ndarray of f_i(v_i) for every allocation, bin pair.
    """
    c_values = np.array(
        range(1, self.params.poisson_truncation_val), dtype=float).reshape(
            (self.params.poisson_truncation_val - 1, 1))
    c_values_with0 = np.array(
        range(self.params.poisson_truncation_val), dtype=float).reshape(
            (self.params.poisson_truncation_val, 1))
    alloc_vals = np.array(
        range(1, n_resource), dtype=float).reshape((n_resource - 1, 1))
    poissons = stats.poisson(mu=np.array(rates))
    pmf_values = poissons.pmf(c_values_with0)

    minimums = np.minimum(alloc_vals.T, c_values)
    mins_over_c = np.divide(minimums, c_values)
    # defining 0/0 to be 1, can change to np.zeros if 0/0 should be zero.
    mins_over_c = np.concatenate((np.ones((1, n_resource - 1)), mins_over_c),
                                 axis=0)
    # can also switch the order of these concatenates depending on if min(v,c)/c
    # where v=0, c=0 should be 0 or one.
    mins_over_c = np.concatenate((np.zeros(
        (self.params.poisson_truncation_val, 1)), mins_over_c),
                                 axis=1)
    fi = np.matmul(pmf_values.T, mins_over_c)
    return fi

  def _allocate(self, n_resource, beliefs):
    """Returns an array of attention allocations across groups.

    Args:
      n_resource: An int defining the amount of attention that can be allocated
        - the returned array has sum equal to n_resource.
      beliefs: An array of integers reflecting the current belief of
        distribution of target reward across groups.

    Returns:
      An array with length equal to length of input counts that sums to
        n_resource.
    """
    # With probability epsilon allocate with uniform probability.
    # With probability 1-epsilon, allocate according to belief.
    if self.rng.binomial(1, self.params.epsilon):
      self.last_allocation = self.action_space.sample()
    else:
      optimal_allocation = None
      max_expected_yield = 0

      # Construct entire fi table, and corresponding min and max fi tables.
      # The fi table is a table of the expected probability that a incident
      # in a bin is discovered by an attention unit, for each bin and each
      # possible allocation amount for that bin.
      fi_table = self._construct_approx_fi_table(self._n_bins, beliefs,
                                                 self._n_resource + 1)
      min_fi_table = np.maximum(fi_table - self.params.alpha, 0)
      max_fi_table = min_fi_table + self.params.alpha

      # For every bin.
      for bin_i in range(self._n_bins):
        current_allocation = np.zeros(
            self._n_bins, dtype=self.action_space.dtype)
        alloc_upperbound = np.zeros(self._n_bins, dtype=self.action_space.dtype)

        # Get all upper and lower bounds with bin_i as starting bin.
        rows = np.array([i for i in range(self._n_bins) if i != bin_i])
        broadcast_shape = (self._n_resource + 1, len(rows),
                           self._n_resource + 1)
        lower_bounds = np.argmax(
            (np.broadcast_to(fi_table[rows, :], broadcast_shape).T >=
             min_fi_table[bin_i]).T,
            axis=2)
        upper_bounds = np.argmin(
            (np.broadcast_to(fi_table[rows, :], broadcast_shape).T <=
             max_fi_table[bin_i]).T,
            axis=2) - 1
        upper_bounds[upper_bounds == -1] = self._n_resource

        # For every possible allocation to that bin.
        for alloc_to_i in range(self._n_resource + 1):
          current_allocation = np.zeros(
              self._n_bins, dtype=self.action_space.dtype)
          current_allocation[bin_i] = alloc_to_i
          alloc_upperbound[rows] = upper_bounds[alloc_to_i]
          # Set current allocation values to lower bounds.
          current_allocation[rows] = lower_bounds[alloc_to_i]
          alloc_upperbound[bin_i] = alloc_to_i

          if np.sum(current_allocation) > self._n_resource or np.any(
              current_allocation > alloc_upperbound):
            # This allocation scheme requires more resource than available.
            # Move on to next possible allocation scheme.
            continue
          remaining_resource = self._n_resource - np.sum(current_allocation)

          # Now greedily allocate remaining resources to bins that have maximal
          # marginal probability of making another discovery.
          for _ in range(remaining_resource):
            marginal_probs = []
            for j in range(self._n_bins):
              if current_allocation[j] < alloc_upperbound[j]:
                marginal_probs.append(
                    ((self._calculate_tail_probability(
                        current_allocation[j] + 1, beliefs[j]) -
                      self._calculate_tail_probability(current_allocation[j],
                                                       beliefs[j])), j))
            if not marginal_probs:
              # Allocation cannot make full use of resources and satisfy
              # fairness constraint go to next allocation.
              break
            next_bin = max(marginal_probs, key=lambda i: i[0])[1]
            current_allocation[next_bin] += 1
          if np.sum(current_allocation) < self._n_resource or np.any(
              current_allocation > alloc_upperbound):
            # This allocation scheme requires more resource than available
            # or doesn't make full use of resources.
            # Move on to next possible allocation scheme.
            continue

          # If current_allocation has the highest expected yield, store it as
          # the optimal allocation.
          # pylint: disable=g-complex-comprehension
          expected_yield = np.sum([
              np.sum([
                  self._calculate_tail_probability(
                      np.array(range(1, current_allocation[i] + 1)), beliefs[i])
              ]) for i in range(self._n_bins)
          ])
          # pylint: enable=g-complex-comprehension

          if expected_yield >= max_expected_yield:
            max_expected_yield = expected_yield
            optimal_allocation = current_allocation

      if optimal_allocation is None:
        print("No allocation found for this alpha: %f" % self.params.alpha)
        logging.warning("No allocation found for this alpha: %f",
                        self.params.alpha)
        optimal_allocation = np.zeros(
            self._n_bins, dtype=self.action_space.dtype)
        raise gym.error.InvalidAction("Invalid action: %s with alpha %f" %
                                      (optimal_allocation, self.params.alpha))

      self.last_allocation = optimal_allocation

    return self.last_allocation
