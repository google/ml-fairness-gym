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

"""Agents for infectious disease environments."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
import attr
import core
import rewards
import gin
from gym import spaces
import networkx as nx
import numpy as np

from typing import Any, List, Mapping, Sequence, Text


def _infection_indicator(
    health_states, infected_state_index):
  """Returns a binary vector that indicates whether individuals are infected."""
  return [int(state == infected_state_index) for state in health_states]


@gin.configurable
@attr.s
class Params(core.Params):
  """Infectious disease agent parameters."""
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

  # The number of treatments that can be given out in one step.
  num_treatments = attr.ib()  # type: int


def env_to_agent_params(env_params):
  return Params(
      state_names=env_params.state_names,
      healthy_index=env_params.healthy_index,
      infectious_index=env_params.infectious_index,
      healthy_exit_index=env_params.healthy_exit_index,
      num_treatments=env_params.num_treatments)


class _BaseAgent(core.Agent):
  """Base class for infectious disease agents.

  Infectious disease agents triage individuals at each timestep, and treatments
  are given out according to that triage order until they are exhausted.

  Subclasses must override _triage.  Otherwise, a NotImplementedError will be
  raised when act is called.
  """

  def __init__(self,
               action_space,
               reward_fn,
               observation_space,
               params):
    self.initial_params = copy.deepcopy(params)
    if reward_fn is None:
      reward_fn = rewards.NullReward()
    super(_BaseAgent, self).__init__(
        action_space, reward_fn, observation_space)

    self.rng = np.random.RandomState()

  def _triage(self, observation):
    """Returns person indices ordered from first to last person to treat."""
    raise NotImplementedError

  def _act_impl(self,
                observation,
                reward,
                done):
    """Returns a treatment action.

    Args:
      observation: An observation in Dict Space with 'population' and
        'population_graph' keys.
      reward: A scalar float reward value.
      done: A boolean indicating whether the simulation has finished.

    Returns:
      A numpy ndarray containing population indices that represents a treatment
      action.
    """
    if done:
      raise core.EpisodeDoneError('Called act on a done episode.')
    if not self.observation_space.contains(observation):
      raise core.InvalidObservationError(
          'Invalid observation: %s.' % observation)

    return self._triage(observation)


@gin.configurable
class CentralityAgent(_BaseAgent):
  """An agent that triages based on graph centrality."""

  def _triage(self, observation):
    """Returns person indices ordered from first to last person to treat.

    Infected people are prioritized above non-infected people, and infected
    people are ranked according to their centrality in the contact graph.

    Args:
      observation: An observation from a Dict Space with 'health_states' and
        'population_graph' keys.  The 'health_states' observation contains the
        health states of the population, and the 'population_graph' observation
        contains the contact graph over which disease spreads.

    Returns:
      A numpy array of population indices representing the triage order.
    """
    infections = _infection_indicator(
        observation['health_states'], self.initial_params.infectious_index)
    centrality = nx.eigenvector_centrality(observation['population_graph'])

    # Negate because lower scores are treated first. Note that centrality is a
    # dict that maps from node-keys to centrality values, and it happens
    # that the node keys are zero-counted contiguouos integers, which is
    # required for the following enumeration-based indexing to work out.  This
    # condition is checked by the assertion immediately below.
    assert list(
        observation['population_graph'].nodes()) == list(
            range(observation['population_graph'].number_of_nodes()))
    triage_scores = np.array([
        -infection * centrality[i] for i, infection in enumerate(infections)])

    max_treatments = len(self.action_space.nvec)
    return np.argsort(triage_scores)[:max_treatments]


class RandomAgent(_BaseAgent):
  """An agent that treats infected people chosen at random."""

  def _triage(self, observation):
    """Returns person indices ordered from first to last person to treat.

    Infected people are prioritized above non-infected people, and the triage
    ordering among infected people is chosen randomly.

    Args:
      observation: An observation from a Dict Space with 'health_states' and
        'population_graph' keys.  The 'health_states' observation contains the
        health states of the population, and the 'population_graph' observation
        contains the contact graph over which disease spreads.

    Returns:
      A numpy array of population indices representing the triage order.
    """
    infections = _infection_indicator(
        observation['health_states'], self.initial_params.infectious_index)
    # This line has the effect of prioritizing infected over uninfected people,
    # ensuring that the order of infected people is random, and padding the
    # actions taken with uninfected people.
    triage_scores = np.array([
        -infection * self.rng.rand() for infection in infections])
    max_treatments = len(self.action_space.nvec)
    return np.argsort(triage_scores)[:max_treatments]


class InteractiveAgent(_BaseAgent):
  """An agent that asks a user for input on who to treat."""

  def _triage(self, observation):
    infectious_index = self.initial_params.infectious_index
    infected_indices = [
        i for i, health_state in enumerate(observation['health_states'])
        if health_state == infectious_index]

    while True:
      print('Infected indices: %s.' % infected_indices)
      num_treatments = self.initial_params.num_treatments
      print('You have %d treatments to distribute' % num_treatments)
      user_input = input(
          'Please provide space-separated ids of people to treat: ')
      to_treat = [int(x) for x in user_input.split(' ')]
      if len(to_treat) != num_treatments:
        print(
            ('You have %d treatments to give out, but selected %d people to '
             'treat.') % (num_treatments, len(to_treat)))
      else:
        break

    max_treatments = len(self.action_space.nvec)

    treatment = np.zeros(max_treatments, dtype='int64')

    for i, index in enumerate(to_treat):
      treatment[i] = index

    return treatment



