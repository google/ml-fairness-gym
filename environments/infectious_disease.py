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
"""Infectious disease environments for the Fairness Gym.

Environments in this module are built on the Susceptible-Infected (SI) family
of models and their extensions.  An infectious disease environment instance
contains a population of individuals and represents the dynamics of disease
within that population.

Agents that interact with infectious disease environments allocate treatment to
individuals in order to improve their health and reduce the spread of disease.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
from typing import Dict, List, Optional, Text, Union

import attr
import core
from spaces import graph
from spaces import multi_discrete_with_none
import gin
from gym import spaces
import matplotlib  # pylint: disable=unused-import
import networkx as nx
import numpy as np
from six.moves import range


@attr.s(cmp=False)
class Params(core.Params):
  """Infectious disease parameters."""

  # A Markov transition matrix.  transition_matrix[i, j] gives the probability
  # of transitioning from i to j.  Note that transitions probabilities from
  # transition_matrix[healthy_index, :] are ignored because infection is
  # governed by infection_probability.
  transition_matrix = attr.ib()  # type: np.ndarray

  # Treatment transition matrix.  treatment_transtion_matrix[i, j] gives the
  # probability of transitioning from i to j if given treatment.
  treatment_transition_matrix = attr.ib()  # type: np.ndarray

  # A list of strings that give the name and index of each state.
  state_names = attr.ib()  # type: List[Text]

  # The index of the healthy state.
  healthy_index = attr.ib()  # type: int

  # The index of the infectious state.  This is a special state that allows
  # individuals to infect others.
  infectious_index = attr.ib()  # type: int

  # The index of the state that healthy people transition to due to contact
  # with infected people.
  healthy_exit_index = attr.ib()  # type: int

  # The probability that contact with an infected individual leads to a new
  # infection during one step.
  infection_probability = attr.ib()  # type: float

  # The contact network over which disease spreads.
  population_graph = attr.ib(factory=nx.karate_club_graph)  # type: nx.Graph

  # Specifies the initial health state of the population. If None, the
  # population will be initialized uniformly using initial_health_state_seed.
  initial_health_state = attr.ib(default=None)  # type: Optional[List[int]]
  initial_health_state_seed = attr.ib(default=100)  # type: int

  # num_treatments is the number of treatments that will actually be given out,
  # and max_treatments provides an upper bound on that.
  num_treatments = attr.ib(default=1)  # type: int
  max_treatments = attr.ib(default=10)  # type: int

  # Allow the epidemic to progress without intervention for burn_in steps. Note
  # that the burned-in period will not be included in the environment's history.
  burn_in = attr.ib(default=0)  # type: int

  def __attrs_post_init__(self):
    # Validate parameters.
    if self.transition_matrix.shape != (
        len(self.state_names), len(self.state_names)):
      raise ValueError(
          'Transition matrix has shape %s, expected shape %s' % (
              self.transition_matrix.shape,
              (len(self.state_names), len(self.state_names))))
    if self.transition_matrix.shape != self.treatment_transition_matrix.shape:
      raise ValueError(
          'transition_matrix and treatment_transition_matrix must have the '
          'same shape.')
    for dim in range(self.transition_matrix.shape[0]):
      if dim != self.healthy_index and self.transition_matrix[dim].sum() != 1:
        raise ValueError('Transition matrix rows must sum to one.')
    for index in (
        self.healthy_index, self.healthy_exit_index, self.infectious_index):
      if not 0 <= index < len(self.state_names):
        raise ValueError('Index %d out of range.' % len(self.state_names))

    # Set rng and health state.
    rng = np.random.RandomState(self.initial_health_state_seed)
    if self.initial_health_state is None:
      self.initial_health_state = rng.choice(
          [self.healthy_index, self.infectious_index],
          p=[0.9, 0.1],
          size=self.population_graph.number_of_nodes()).tolist()

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return False

    # We know these dicts have the same keys because 'other' is also a params
    # instance.
    self_dict = self.asdict()
    other_dict = other.asdict()

    for key in self_dict:
      if key == 'population_graph':
        if not nx.is_isomorphic(
            self_dict['population_graph'], other_dict['population_graph']):
          return False
      if (isinstance(self_dict[key], np.ndarray)
          and isinstance(other_dict[key], np.ndarray)):
        return (self_dict[key] == other_dict[key]).all()
      else:
        if self_dict[key] != other_dict[key]:
          return False
    return True

  def __neq__(self, other):
    return not self.__eq__(other)


@attr.s(cmp=False)
class State(core.State):
  """Infectious disease state."""
  # Parameters.
  params = attr.ib()  # type: Params

  # Random state.
  rng = attr.ib(factory=np.random.RandomState)  # type: np.random.RandomState

  # A list of integers representing the health states of members of the
  # population.
  health_states = attr.ib(factory=list)  # type: List[int]

  # The contact graph of the population.
  population_graph = attr.ib(factory=nx.Graph)  # type: nx.Graph


class InfectiousDiseaseEnv(core.FairnessEnv):
  """An infectious disease environment.

  At each time step, the disease progresses due to contact between susceptible
  and infected individuals.  After this occurs, individuals recieve
  state-changing treatment.

  State transitions due to contact are determined as follows.  Let j be the
  index of an individual in the population, B be the probability of infection,
  n_j be the number of neighbors of individual j in the infected_state, and I_j
  be the random variable representing the number of times that individual j is
  infected during this step.

  Because each possible infection of individual j is independent,
  I ~ Binomial(n_j, B).

  Individual j remains in the healthy_state if I = 0. This happens with
  probability P(I = 0) = (1 - B) ** n_j.

  Individual j transitions out of the healthy_state if I > 0. This happens with
  probability P(I > 0) = 1 - P(I = 0) = 1 - (1 - B) ** n_j.

  Attributes:
    action_space: A MultiDiscrete space from which treatment indices are drawn.
    observable_state_vars: A dictionary mapping the names of observable state
      variables to their space.
    state: A State instance containing the environment state that can be read
      by other entities.
    state_name_to_index: A dictionary that maps state name to their integer
      values.
  """

  def __init__(self, params):
    population_size = params.population_graph.number_of_nodes()

    # The action space is a population_size vector where each element takes on
    # values in [0, population_size).  Each element in the vector represents a
    # treatment (of which at most max_treatments can be given out at any one
    # timestep), and the value represents the index of the person who receives
    # the treatment.
    #
    # If None is passed instead of a vector, no treatment is administered.
    self.action_space = multi_discrete_with_none.MultiDiscreteWithNone([
        population_size
        for _ in range(params.max_treatments)
    ])  # type: spaces.Space

    # Define the spaces of observable state variables.
    self.observable_state_vars = {
        'health_states': spaces.MultiDiscrete(
            [len(params.state_names) for _ in range(population_size)]),
        'population_graph': graph.GraphSpace(population_size,
                                             directed=False),
    }  # type: Dict[Text, spaces.Space]

    # Map state names to indices.
    self.state_name_to_index = {
        state: i for i, state in enumerate(params.state_names)}

    super(InfectiousDiseaseEnv, self).__init__(params)
    self.state = self._create_initial_state()

  def _create_initial_state(self, rng=None):
    """Creates and returns a new State instance."""
    # Copy so self.initial_params remains pristine if state.params are
    # mutated.
    state = State(params=copy.deepcopy(self.initial_params))

    state.rng = rng or np.random.RandomState()
    state.population_graph = state.params.population_graph

    params = state.params
    population_size = state.params.population_graph.number_of_nodes()

    assert len(params.initial_health_state) == population_size, (
        'params.initial_health_state has length %d, expected %d.' % (
            len(params.initial_health_state), population_size))
    state.health_states = params.initial_health_state

    return state

  def render(self,
             color_map,
             mode='human',
             **kwargs):
    if mode != 'human':
      raise ValueError('Unsupported mode \'%s\'.' % mode)
    colors = [
        color_map[health_state] for health_state in self.state.health_states]
    nx.draw(self.state.population_graph, node_color=colors, **kwargs)

  def _get_observable_state(self):
    """Extracts observable state from `self.state`.

    This method is overridden in order to handle the population contact graph
    correctly.

    Returns:
      A dict that maps variable names to variable values. The value is first
      cast to a numpy array unless the variable name is 'population_graph,' in
      which case it is passed through as-is.
    """
    observable_state = {}
    for var_name in self.observable_state_vars:
      if var_name == 'population_graph':
        observable_state[var_name] = getattr(self.state, var_name)
      else:
        observable_state[var_name] = np.asarray(getattr(self.state, var_name))

    return observable_state

  def _step_impl(self, state, action):
    """Moves forward one timestep.

    First, the agent allocates treatment.
    Next, health state changes of the population are computed and applied.

    Args:
      state: A `State` object containing the current state.
      action: An action in `action_space`.

    Returns:
      A `State` object containing the updated state.
    """
    params = state.params

    # Apply treatment.
    if action is None:
      action = np.array([])

    assert len(action) in {0, state.params.max_treatments}, (
        'Got a vector with length %d while taking a step, but expected a '
        'vector with length max_treatments (%d) or zero.') % (
            len(action), state.params.max_treatments)
    # action is a list of patients indices to treat.
    for idx in action[:min(state.params.num_treatments, len(action))]:
      health_state = state.health_states[idx]
      transition_probs = state.params.treatment_transition_matrix[
          health_state, :]
      state.health_states[idx] = state.rng.choice(
          len(params.state_names), p=transition_probs)

    # Progress disease by tracking state transitions then applying them.
    transitions = []  # Tracks new states.
    for index, health_state in enumerate(state.health_states):
      transition_probs = state.params.transition_matrix[health_state, :]

      # Handle transitions from the healthy state as a special case.  See the
      # class-level docstring for a description of this process.
      if health_state == state.params.healthy_index:
        num_infected_neighbors = sum(
            state.health_states[neighbor] == state.params.infectious_index
            for neighbor in state.population_graph.neighbors(index))

        transition_probs = np.zeros(len(state.params.state_names))
        transition_probs[state.params.healthy_index] = (
            (1. - state.params.infection_probability) ** num_infected_neighbors)
        transition_probs[state.params.healthy_exit_index] = (
            1 - transition_probs[state.params.healthy_index])

      # Choose the new state and append it to transitions.
      new_state = state.rng.choice(
          len(params.state_names), 1, p=transition_probs)[0]
      transitions.append(new_state)

    # Apply transitions (some of which maintain the same state).
    for i, new_state in enumerate(transitions):
      state.health_states[i] = new_state

    return state

  def reset(self):
    """Resets the environment."""
    self.state = self._create_initial_state(self.state.rng)
    # Note that the burned-in period will not be included in the environment's
    # history.
    for _ in range(self.initial_params.burn_in):
      self.step(None)
    return super(InfectiousDiseaseEnv, self).reset()

  def set_initial_health_state(self, initial_health_state):
    """Set the initial health state that will be used on next reset."""
    self.initial_params.initial_health_state = initial_health_state


@gin.configurable
def build_si_model(
    population_graph,
    infection_probability,
    num_treatments,
    max_treatments=None,
    treatment_transition_matrix=None,
    initial_health_state=None,
    initial_health_state_seed=100,
    burn_in=0):
  """Builds a Susceptible-Infected environment."""

  if max_treatments is None:
    max_treatments = population_graph.number_of_nodes()

  state_names = ['susceptible', 'infected']
  healthy_index = 0
  infectious_index = 1

  transition_matrix = np.array([[0, 0],
                                [0, 1]])
  if treatment_transition_matrix is None:
    treatment_transition_matrix = np.array([[1, 0], [1, 0]])

  params = Params(
      population_graph=population_graph,
      transition_matrix=transition_matrix,
      treatment_transition_matrix=treatment_transition_matrix,
      state_names=state_names,
      healthy_index=healthy_index,
      infectious_index=infectious_index,
      healthy_exit_index=infectious_index,
      infection_probability=infection_probability,
      initial_health_state=initial_health_state,
      initial_health_state_seed=initial_health_state_seed,
      num_treatments=num_treatments,
      max_treatments=max_treatments,
      burn_in=burn_in)

  return InfectiousDiseaseEnv(params)


@gin.configurable
def build_sir_model(
    population_graph,
    infection_probability,
    infected_exit_probability,
    num_treatments,
    max_treatments=None,
    treatment_transition_matrix=None,
    initial_health_state=None,
    initial_health_state_seed=100,
    burn_in=0):
  """Builds a Susceptible-Infected-Recovered environment."""
  if max_treatments is None:
    max_treatments = population_graph.number_of_nodes()

  state_names = ['susceptible', 'infected', 'recovered']
  healthy_index = 0
  healthy_exit_index = 1
  infectious_index = 1

  transition_matrix = np.array([
      [0, 0, 0],
      [0, 1 - infected_exit_probability, infected_exit_probability],
      [0, 0, 1]])

  if treatment_transition_matrix is None:
    # By default, treatment moves people from any initial state into
    # 'recovered.'
    treatment_transition_matrix = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]])

  params = Params(
      population_graph=population_graph,
      transition_matrix=transition_matrix,
      treatment_transition_matrix=treatment_transition_matrix,
      state_names=state_names,
      healthy_index=healthy_index,
      infectious_index=infectious_index,
      healthy_exit_index=healthy_exit_index,
      infection_probability=infection_probability,
      initial_health_state=initial_health_state,
      initial_health_state_seed=initial_health_state_seed,
      num_treatments=num_treatments,
      max_treatments=max_treatments,
      burn_in=burn_in)

  return InfectiousDiseaseEnv(params)


@gin.configurable
def build_seir_model(
    population_graph,
    infection_probability,
    exposed_exit_probability,
    infected_exit_probability,
    num_treatments,
    max_treatments=None,
    treatment_transition_matrix=None,
    initial_health_state=None,
    initial_health_state_seed=100,
    burn_in=0):
  """Builds a Susceptible-Exposed-Infected-Recovered environment."""
  if max_treatments is None:
    max_treatments = population_graph.number_of_nodes()

  state_names = ['susceptible', 'exposed', 'infected', 'recovered']
  healthy_index = 0
  healthy_exit_index = 1
  infectious_index = 2

  transition_matrix = np.zeros((4, 4))
  transition_matrix[healthy_exit_index, :] = [
      0, 1 - exposed_exit_probability, exposed_exit_probability, 0]
  transition_matrix[infectious_index, :] = [
      0, 0, 1 - infected_exit_probability, infected_exit_probability]
  transition_matrix[3, :] = [0, 0, 0, 1]

  if treatment_transition_matrix is None:
    # By default, treatment moves people from any initial state into
    # 'recovered.'
    treatment_transition_matrix = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]])

  params = Params(
      population_graph=population_graph,
      transition_matrix=transition_matrix,
      treatment_transition_matrix=treatment_transition_matrix,
      state_names=state_names,
      healthy_index=healthy_index,
      infectious_index=infectious_index,
      healthy_exit_index=healthy_exit_index,
      infection_probability=infection_probability,
      initial_health_state=initial_health_state,
      initial_health_state_seed=initial_health_state_seed,
      num_treatments=num_treatments,
      max_treatments=max_treatments,
      burn_in=burn_in)

  return InfectiousDiseaseEnv(params)
