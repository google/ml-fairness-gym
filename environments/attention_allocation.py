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

# Lint as: python2, python3
"""ML Fairness gym location-based attention allocation environment.

This environment is meant to be a general but simple location-based
attention allocation environment.

Situations that could be modeled by this environment are pest-control, or
allocation of social interventions like mobile STD testing clinics.
This is  not a good environment to model predictive policing in part due to the
concerns raised in "Dirty Data, Bad Predictions: How Civil Rights Violations
Impact Police Data, Predictive Policing Systems, and Justice", Richardson et al.
(https://www.nyulawreview.org/wp-content/uploads/2019/04/NYULawReview-94-Richardson-Schultz-Crawford.pdf)

The environment contains k locations. At each timestep, some number of incidents
occur (such as rat infestations) in each location, and the agent has to allocate
some of its attention to each location. Each attention unit may then "see" or
discover some number of the incidents.

Incidents are drawn from a Poisson centered on an underlying incident rate.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
from typing import List, Optional, Tuple
import attr
import core
from spaces import multinomial
from gym import spaces
import numpy as np
from six.moves import range


@attr.s
class Params(core.Params):
  """Attention Allocation parameters."""

  # Number of locations represented in the environment.
  n_locations = attr.ib(default=2)  # type: int

  # Prior counts of incidents for each location.
  prior_incident_counts = attr.ib(default=(650, 250))  # type: Tuple[int, ...]
  # Incident rates for each location.
  incident_rates = attr.ib(factory=lambda: [4., 3.])  # type: List[float]
  # Discovered incidents weight - weights how many incidents are discovered
  # incidents, for which attention needs to be allocated to be discovered
  discovered_incident_weight = attr.ib(default=1.0)  # type: float
  # Reported inicdents weight - weights how many incidents are reported, which
  # are observed from a location without attention being allocated there.
  reported_incident_weight = attr.ib(default=0.0)  # type: float

  # Number of attention units that are available to be allocated.
  n_attention_units = attr.ib(default=1)  # type: int
  # If True, each unit of attention can discover more than one incident.
  # If False, each unit of attention can discover at most one incident.
  attention_replacement = attr.ib(default=False)  # type: bool
  # Probability an incident is missed by an attention unit for each location.'

  miss_incident_prob = attr.ib(default=(0.2, 0.2))  # type: Tuple[float, ...]
  # Probability an incident is falsely reported without being discovered.
  extra_incident_prob = attr.ib(default=(0.0, 0.0))  # type: Tuple[float, ...]
  # The rate at which the incident_rates change in response to allocation
  # of attention units.
  dynamic_rate = attr.ib(default=0.0)  # type: float

  # Location feature parameters.
  # Means and covariances of the multivariate gaussians for the features.
  feature_means = attr.ib(factory=lambda: [1., 1.])
  feature_covariances = attr.ib(factory=lambda: [[0.8, 0.0], [0.0, 0.7]])
  # Vector with coefficients to control the correlation between features and
  # underlying incident rates.
  feature_coefficients = attr.ib(default=(0, 1))


@attr.s(cmp=False)
class State(core.State):
  """Attention Allocation state."""

  # Parameters.
  params = attr.ib()  # type: Params

  # A ndarray of integers representing the incidents seen at each location
  incidents_seen = attr.ib()  # type: np.ndarray

  # A ndarray of integers representing the incidents reported for each location.
  incidents_reported = attr.ib()  # type: np.ndarray

  # A ndarray of integers representing the incidents reported for each location.
  incidents_occurred = attr.ib()  # type: np.ndarray

  # A ndarray of floats representing features for each location.
  location_features = attr.ib()  # type: np.ndarray

  # Random state.
  rng = attr.ib(factory=np.random.RandomState)  # type: np.random.RandomState


def _sample_incidents(rng, params):
  """Generates new crimeincident occurrences across locations.

  Args:
    rng: A numpy RandomState() object acting as a random number generator.
    params: A Params instance for this environment.

  Returns:
    incidents_occurred: a list of integers of number of incidents for each
    location.
    that could be discovered by attention.
    reported_incidents: a list of integers of a number of incidents reported
    directly.
  """
  # pylint: disable=g-complex-comprehension
  crimes = [
      rng.poisson([
          params.incident_rates[i] * params.discovered_incident_weight,
          params.incident_rates[i] * params.reported_incident_weight
      ]) for i in range(params.n_locations)
  ]
  incidents_occurred, reported_incidents = np.hsplit(np.asarray(crimes), 2)
  return incidents_occurred.flatten(), reported_incidents.flatten()


def _get_location_features(params, rng, incidents_occurred):
  """Returns a matrix of float features for each location.

  Calculates new feature means based on incidents occurred and draws features
  from a multivariate gaussian distribution using the parameter defined means
  and covariances.

  Args:
    params: A Params instance for this environment.
    rng: A numpy RandomState() object acting as a random number generator.
    incidents_occurred: A list of integers of number of incidents for each
      location that occurred.

  Returns:
    A numpy array of n_locations by number of features.
  """
  # Move feature means based on incidents that occurred to make m by k matrix
  # where each row is the means for the features for location k at this step.
  shifted_feature_means = params.feature_means + np.outer(
      incidents_occurred, params.feature_coefficients)

  feature_noise = rng.multivariate_normal(
      np.zeros_like(params.feature_means),
      params.feature_covariances,
      size=params.n_locations)

  return shifted_feature_means + feature_noise


def _update_state(state, incidents_occurred, incidents_reported, action):
  """Updates the state given the agents' action.

  This function simulates attention discovering incidents in order to determine
  and populate the number of seen incidents in the state.

  Args:
    state: a 'State' object with the state to be updated.
    incidents_occurred: a vector of length equal to n_locations in state.param
      that contains integer counts of incidents that occurred for each location.
    incidents_reported: a vector of length equal to n_locations in state.param
      that contains integer counts of incidents that are reported for each
      location.
    action: an action in the action space of LocationAllocationEnv that is a
      vector of integer counts of attention allocated to each location.
  """
  params = state.params
  if params.attention_replacement:
    discover_probability = 1 - (np.power(params.miss_incident_prob, action))
    incidents_seen = [
        state.rng.binomial(incidents_occurred[i], discover_probability[i])
        for i in range(params.n_locations)
    ]
  else:
    # Attention units are without replacement, so each units can only catch 1
    # crime.
    incidents_seen = [0] * params.n_locations
    for location_ind in range(params.n_locations):
      unused_attention = action[location_ind]
      # Iterate over crime incidents and determine if each one is "caught".
      for _ in range(incidents_occurred[location_ind]):
        incidents_discovered = state.rng.binomial(
            1, 1 - (np.power(params.miss_incident_prob[location_ind],
                             unused_attention)))
        unused_attention -= incidents_discovered
        incidents_seen[location_ind] += incidents_discovered
        if unused_attention <= 0:
          # Terminate for loop early because there are no attention left.
          break
        # If there are unused individuals have them generate false incidents.
        for _ in range(unused_attention):
          incidents_seen[location_ind] += state.rng.binomial(
              1, params.extra_incident_prob[location_ind])

  # Handle dynamics.
  for location_ind in range(params.n_locations):
    attention = action[location_ind]
    if attention == 0:
      params.incident_rates[location_ind] += params.dynamic_rate
    else:
      params.incident_rates[location_ind] = max(
          0.0, params.incident_rates[location_ind] -
          (params.dynamic_rate * attention))

  state.location_features = _get_location_features(params, state.rng,
                                                   incidents_occurred)
  state.incidents_occurred = np.asarray(incidents_occurred)
  state.incidents_seen = np.asarray(incidents_seen)
  state.incidents_reported = np.asarray(incidents_reported)


class LocationAllocationEnv(core.FairnessEnv):
  """Location based allocation environment.

  In each step, agent allocates attention across locations. Environment then
  simulates seen incidents based on incidents that occurred and attention
  distribution.
  Incidents are generated from a poisson distribution of underlying incidents
  rates for each location.
  """

  def __init__(self, params = None):
    if params is None:
      params = Params()

    self.action_space = multinomial.Multinomial(params.n_locations,
                                                params.n_attention_units)

    assert (params.n_locations == len(params.prior_incident_counts) and
            params.n_locations == len(params.incident_rates))

    # Define the observation space.
    # Crimes seen is multidiscrete because it may not sum to n_attention_units.
    # MultiDiscrete uses dtype=np.int32.
    if params.attention_replacement:
      # If there is attention replacement, the number of attention doesn't bound
      # the incidents_seen.
      incidents_seen_space = spaces.MultiDiscrete([np.iinfo(np.int32).max] *
                                                  params.n_locations)
    else:
      incidents_seen_space = spaces.MultiDiscrete(
          [params.n_attention_units + 1] * params.n_locations)

    incidents_reported_space = spaces.MultiDiscrete([np.iinfo(np.int32).max] *
                                                    params.n_locations)

    n_features = len(params.feature_means)
    location_features_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(params.n_locations, n_features),
        dtype=np.float32)

    # The first observation from this state is not necessarily contained by this
    # observation space. It conveys a prior of the initial incident counts.
    self.observable_state_vars = {
        'incidents_seen': incidents_seen_space,
        'incidents_reported': incidents_reported_space,
        'location_features': location_features_space
    }

    super(LocationAllocationEnv, self).__init__(params)
    self._state_init()

  def _state_init(self, rng=None):
    n_locations = self.initial_params.n_locations
    self.state = State(
        rng=rng or np.random.RandomState(),
        params=copy.deepcopy(self.initial_params),
        incidents_seen=np.zeros(n_locations, dtype='int64'),
        incidents_reported=np.zeros(n_locations, dtype='int64'),
        incidents_occurred=np.zeros(n_locations, dtype='int64'),
        location_features=np.zeros(
            (n_locations, len(self.initial_params.feature_means))))

  def reset(self):
    """Resets the environment."""
    self._state_init(self.state.rng)
    return super(LocationAllocationEnv, self).reset()

  def _is_done(self):
    """Never returns true because there is no end case to this environment."""
    return False

  def _step_impl(self, state, action):
    """Run one timestep of the environment's dynamics.

    In a step, the agent allocates attention across disctricts. The environement
    then returns incidents seen as an observation based off the actual hidden
    incident occurrences and attention allocation.

    Args:
      state: A 'State' object containing the current state.
      action: An action in 'action space'.

    Returns:
      A 'State' object containing the updated state.
    """
    incidents_occurred, reported_incidents = _sample_incidents(
        state.rng, state.params)
    _update_state(state, incidents_occurred, reported_incidents, action)
    return state
