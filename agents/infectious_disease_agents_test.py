# coding=utf-8
# Copyright 2022 The ML Fairness Gym Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from absl.testing import absltest
from absl.testing import parameterized
import core
import rewards
import test_util
from agents import infectious_disease_agents
from environments import infectious_disease
import networkx as nx


@contextlib.contextmanager
def mock_input(result):
  """Creates a context where input() calls return a predetermined result."""
  input_fn = __builtins__.input
  __builtins__.input = lambda _: result
  yield
  __builtins__.input = input_fn


def build_empty_graph(num_nodes):
  """Returns a graph with no edges."""
  graph = nx.Graph()
  graph.add_nodes_from(range(num_nodes))
  return graph


def instantiate_environment_and_agent(
    agent_class,
    population_graph,
    initial_health_state,
    infection_probability=0.5,
    num_treatments=5,
    max_treatments=10,
    seed=100,
    agent_seed=50):
  env = infectious_disease.build_si_model(
      population_graph=population_graph,
      infection_probability=infection_probability,
      num_treatments=num_treatments,
      initial_health_state=initial_health_state,
      max_treatments=max_treatments)
  agent = agent_class(
      env.action_space,
      rewards.NullReward(),
      env.observation_space,
      infectious_disease_agents.env_to_agent_params(env.initial_params))
  env.seed(seed)
  agent.seed(agent_seed)
  _ = env.reset()
  return env, agent


def set_up_and_observe(
    population_graph,
    initial_health_state,
    agent_class=infectious_disease_agents.CentralityAgent,
    infection_probability=0.5,
    num_treatments=5,
    max_treatments=10):
  env, agent = instantiate_environment_and_agent(
      agent_class,
      population_graph,
      initial_health_state,
      infection_probability=infection_probability,
      num_treatments=num_treatments,
      max_treatments=max_treatments)
  observation = {
      name: space.sample()
      for name, space in env.observable_state_vars.items()}
  return env, agent, observation


class InfectiousDiseaseAgentsTest(parameterized.TestCase):

  def test_interact_with_env_replicable_randomagent(self):
    graph = nx.karate_club_graph()
    centrality = nx.eigenvector_centrality(graph)
    sorted_nodes = sorted(
        centrality.keys(), key=lambda k: centrality[k], reverse=True)
    # Infect the 3rd through 5th most central people.
    initial_health_state = [
        1 if index in sorted_nodes[3:6] else 0
        for index in range(len(sorted_nodes))
    ]
    env, agent, _ = set_up_and_observe(
        population_graph=graph,
        initial_health_state=initial_health_state,
        agent_class=infectious_disease_agents.RandomAgent)
    test_util.run_test_simulation(env=env, agent=agent)

  def test_interact_with_env_replicable_centralityagent(self):
    graph = nx.karate_club_graph()
    centrality = nx.eigenvector_centrality(graph)
    sorted_nodes = sorted(
        centrality.keys(), key=lambda k: centrality[k], reverse=True)
    # Infect the 3rd through 5th most central people.
    initial_health_state = [
        1 if index in sorted_nodes[3:6] else 0
        for index in range(len(sorted_nodes))
    ]
    env, agent, _ = set_up_and_observe(
        population_graph=graph,
        initial_health_state=initial_health_state,
        agent_class=infectious_disease_agents.CentralityAgent)
    test_util.run_test_simulation(env=env, agent=agent)

  def test_infection_indicator_indicates_infection(self):
    self.assertCountEqual(
        [1, 0, 1],
        infectious_disease_agents._infection_indicator([2, 1, 2], 2))

  def test_base_class_act_raises(self):
    graph = build_empty_graph(50)
    initial_health_state = [0 for _ in range(50)]
    _, agent, observation = set_up_and_observe(
        population_graph=graph,
        initial_health_state=initial_health_state,
        agent_class=infectious_disease_agents._BaseAgent)
    with self.assertRaises(NotImplementedError):
      agent.act(observation, False)

  def test_act_on_done_raises(self):
    graph = build_empty_graph(50)
    initial_health_state = [0 for _ in range(50)]
    _, agent, observation = set_up_and_observe(
        population_graph=graph,
        initial_health_state=initial_health_state)
    with self.assertRaises(core.EpisodeDoneError):
      agent.act(observation, True)

  def test_invalid_observation_raises(self):
    graph = build_empty_graph(50)
    initial_health_state = [0 for _ in range(50)]
    _, agent, good_observation = set_up_and_observe(
        population_graph=graph,
        initial_health_state=initial_health_state)

    # Check that missing the 'health_states' key raises.
    bad_observation = {
        key: val for key, val in good_observation.items()
        if key != 'health_states'}
    with self.assertRaises(core.InvalidObservationError):
      agent.act(bad_observation, False)

    # Check that an observation of the wrong type raises.
    bad_observation = 'this is a bad observation'
    with self.assertRaises(core.InvalidObservationError):
      agent.act(bad_observation, False)

  def test_centrality_treatment_ordering_correct(self):
    # Initialize a small example graph and sort nodes by their centrality.
    graph = nx.karate_club_graph()
    centrality = nx.eigenvector_centrality(graph)
    sorted_nodes = sorted(
        centrality.keys(),
        key=lambda k: centrality[k],
        reverse=True)

    # Infect the 3rd through 5th most central people.  We expect these people to
    # be the 1st through 3rd people to receive treatment.
    initial_health_state = [
        1 if index in sorted_nodes[3:6] else 0
        for index in range(len(sorted_nodes))]

    # Initialize an environment with that initial health state and a centrality
    # agent.
    env, agent = instantiate_environment_and_agent(
        agent_class=infectious_disease_agents.CentralityAgent,
        population_graph=graph,
        initial_health_state=initial_health_state)

    # Confirm that the infected people are sorted by centrality in the agent's
    # action.  We expect 3rd the through 5th most central people to be the 1st
    # through 3rd people to receive treatment.
    observation = env._get_observable_state()
    action = agent.act(observation, False)
    self.assertEqual(sorted_nodes[3:6], action[:3].tolist())

  @parameterized.parameters(
      (infectious_disease_agents.CentralityAgent,),
      (infectious_disease_agents.RandomAgent,))
  def test_agents_choose_infected_people(self, agent_class):
    population_size = 50

    graph = build_empty_graph(population_size)

    # Choose infected population members and validate the choices.
    infected_indices = [1, 6, 33, 46]
    for index in infected_indices:
      self.assertLess(index, population_size)
    initial_health_state = [
        1 if index in infected_indices else 0
        for index in range(population_size)]

    # Initialize an environment and agent.
    env, agent = instantiate_environment_and_agent(
        agent_class=agent_class,
        population_graph=graph,
        initial_health_state=initial_health_state)

    # Confirm that the infected people are triaged first.
    observation = env._get_observable_state()
    action = agent.act(observation, False)
    self.assertCountEqual(
        infected_indices,
        action[:len(infected_indices)].tolist())

  def test_interactive_agent_treats_chosen_people(self):
    population_size = 50

    graph = build_empty_graph(population_size)

    # Choose infected population members and validate the choices.
    infected_indices = [1, 6, 33, 46]
    for index in infected_indices:
      self.assertLess(index, population_size)
    initial_health_state = [
        0 if index in infected_indices else 1
        for index in range(population_size)]

    env, agent = instantiate_environment_and_agent(
        agent_class=infectious_disease_agents.InteractiveAgent,
        population_graph=graph,
        initial_health_state=initial_health_state,
        num_treatments=3,
        max_treatments=10)

    observation = env._get_observable_state()

    with mock_input('1 6 33'):
      action = agent.act(observation, False)

    self.assertSameElements([1, 6, 33], action[:3].tolist())

  def test_environment_understands_interactive_agent_action(self):
    population_size = 50

    graph = build_empty_graph(population_size)

    initial_health_state = [1 for _ in range(population_size)]

    env, agent = instantiate_environment_and_agent(
        agent_class=infectious_disease_agents.InteractiveAgent,
        population_graph=graph,
        initial_health_state=initial_health_state,
        num_treatments=3,
        max_treatments=population_size)

    observation = env._get_observable_state()

    with mock_input('11 5 32'):
      action = agent.act(observation, False)

    # This will fail if the environment does not understand the agent's action.
    env.step(action)


if __name__ == '__main__':
  absltest.main()
