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

import copy
from absl.testing import absltest
import test_util
from agents import random_agents
from environments import infectious_disease
from spaces import graph as graph_space
import networkx as nx
import numpy as np


def complete_graph_edge_list(size):
  edges = []
  for a in range(size):
    for b in range(size):
      if a != b:
        edges.append((a, b))
  return edges


def get_population_health_state(state):
  """Gets the list that maps index to health state."""
  return copy.deepcopy(state.health_states)


def num_in_health_state(state, health_state):
  """Returns the number of people in health_state given an environment state."""
  states = get_population_health_state(state)
  return sum(1 for x in states if x == health_state)


class InfectiousDiseaseTest(absltest.TestCase):

  def test_disease_does_not_progress_without_contact(self):
    num_steps = 10

    # Set up an environment with some infected people but no contact between
    # members of the population.
    graph = nx.Graph()
    graph.add_nodes_from(range(50))
    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=0.5,
        num_treatments=0,
        max_treatments=10,
        initial_health_state=[0 for _ in graph])
    agent = random_agents.RandomAgent(
        env.action_space, lambda x: 0, env.observation_space)
    initial_health_state = get_population_health_state(env.state)

    # Run the simulation and ensure that the population's health state does
    # change (because there's no opportunity for disease to spread due to
    # the abscence of contact between people).
    test_util.run_test_simulation(env=env, agent=agent, num_steps=num_steps)
    final_health_state = get_population_health_state(env.state)
    self.assertEqual(
        initial_health_state,
        final_health_state)

  def test_disease_progresses_with_contact_si(self):
    num_steps = 10
    population_size = 5

    # Set up a population that is well-connected (here, totally connected).
    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    graph.add_edges_from(complete_graph_edge_list(population_size))
    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=1.0,
        num_treatments=0,
        max_treatments=10,
        initial_health_state=[
            0 if i % 2 == 0 else 1
            for i in range(graph.number_of_nodes())])
    agent = random_agents.RandomAgent(
        env.action_space, lambda x: 0, env.observation_space)
    initial_state = copy.deepcopy(env.state)

    # Ensure that there are more infected people after running the simulation
    # for some time.
    test_util.run_test_simulation(env=env, agent=agent, num_steps=num_steps)
    self.assertGreater(
        num_in_health_state(env.state, env.state_name_to_index['infected']),
        num_in_health_state(initial_state, env.state_name_to_index['infected']))

  def test_disease_progresses_with_contact_sir(self):
    num_steps = 10
    population_size = 5

    # Set up a population that is well-connected (here, totally connected).
    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    graph.add_edges_from(complete_graph_edge_list(population_size))
    env = infectious_disease.build_sir_model(
        population_graph=graph,
        infection_probability=1.0,
        infected_exit_probability=0.0,
        num_treatments=0,
        max_treatments=10,
        initial_health_state=[
            0 if i % 2 == 0 else 1
            for i in range(graph.number_of_nodes())])
    agent = random_agents.RandomAgent(
        env.action_space, lambda x: 0, env.observation_space)
    initial_state = copy.deepcopy(env.state)

    # Ensure that there are more infected people after running the simulation
    # for some time.
    test_util.run_test_simulation(env=env, agent=agent, num_steps=num_steps)
    self.assertGreater(
        num_in_health_state(env.state, env.state_name_to_index['infected']),
        num_in_health_state(initial_state, env.state_name_to_index['infected']))

  def test_disease_progresses_with_contact_during_burn_in(self):
    population_size = 10
    burn_in = 5

    # Construct a chain graph.
    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    for i in range(1, population_size):
      graph.add_edge(i-1, i)

    # Construct a SI environment where the left-most node is infected.
    initial_health_state = [1] + [0 for _ in range(population_size - 1)]
    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=1.0,
        num_treatments=0,
        max_treatments=population_size,
        initial_health_state=initial_health_state,
        burn_in=burn_in)
    # Before burn in, the initial health state should remain unchanged.
    self.assertCountEqual(
        [1] + [0 for _ in range(population_size - 1)],
        env.state.health_states)

    # After burn-in, the infection frontier should have moved by burn_in steps.
    env.reset()
    self.assertEqual(
        [1 for _ in range(burn_in + 1)] +
        [0 for _ in range(population_size - burn_in - 1)],
        env.state.health_states)

  def test_disease_permanently_eradicated_if_whole_population_treated(self):
    population_size = 50
    seed = 1
    infected_indices = {4, 6, 30, 44}

    initial_health_state = [
        1 if i in infected_indices else 0
        for i in range(population_size)]

    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    graph.add_edges_from(complete_graph_edge_list(population_size))

    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=1.0,
        num_treatments=population_size,
        max_treatments=population_size,
        initial_health_state=initial_health_state)
    env.set_scalar_reward(lambda x: 0)
    env.seed(seed)
    self.assertGreater(
        num_in_health_state(env.state,
                            env.state_name_to_index['infected']), 0)

    # Treat everyone and ensure the disease is gone.
    total_treatment = np.arange(population_size)
    env.step(total_treatment)
    self.assertEqual(
        0, num_in_health_state(env.state,
                               env.state_name_to_index['infected']))

    # Once the disease has been eradicated, the population should remain
    # disease-free.
    env.state.num_treatments = 0  # Stop treatment.
    for _ in range(50):
      env.step(np.arange(population_size))
      self.assertEqual(
          0, num_in_health_state(
              env.state, env.state_name_to_index['infected']))

  def test_last_population_used_for_infection_propagation(self):
    population_size = 3
    seed = 1

    # Set up a graph I - S - S with a 1.0 infection rate. If we strictly use
    # the 'frozen' population state to propagate disease, only the middle node
    # should become infected, which is the desired behavior. If we use the
    # evolving population and iterate from left to right, we should also expect
    # the rightmost node to become infected, which is not the desired behavior.
    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    graph.add_edges_from([(0, 1), (1, 2)])

    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=1.0,
        num_treatments=0,
        max_treatments=population_size,
        initial_health_state=[1, 0, 0])
    env.set_scalar_reward(lambda x: 0)
    env.seed(seed)

    # The rightmost person should be SUSCEPTIBLE at t=0.
    self.assertEqual(
        env.state_name_to_index['susceptible'],
        env.state.health_states[2])

    # The rightmost person should remain SUSCEPTIBLE at t=1.
    env.step(np.arange(population_size))  # Treatment has no effect.
    self.assertEqual(
        env.state_name_to_index['susceptible'],
        env.state.health_states[2])

  def test_one_transition_per_step_in_seir(self):
    population_size = 10
    seed = 1
    initially_infected_nodes = {1, 3, 5}

    initial_health_state = [
        2 if i in initially_infected_nodes else 0
        for i in range(population_size)]

    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    graph.add_edges_from(complete_graph_edge_list(population_size))

    env = infectious_disease.build_seir_model(
        population_graph=graph,
        infection_probability=1.0,
        exposed_exit_probability=1.0,
        infected_exit_probability=1.0,
        num_treatments=0,
        max_treatments=population_size,
        initial_health_state=initial_health_state)

    env.set_scalar_reward(lambda x: 0)
    env.seed(seed)

    # Some people need to be infected at the outset.
    self.assertGreater(
        num_in_health_state(env.state, env.state_name_to_index['infected']), 0)

    # After one step, susceptible -> exposed and infected -> recovered.
    env.step(np.arange(population_size))
    self.assertCountEqual(
        [env.state_name_to_index['exposed'] for i in range(population_size)
         if i not in initially_infected_nodes],
        [env.state.health_states[i] for i in range(population_size)
         if i not in initially_infected_nodes])
    self.assertCountEqual(
        [env.state_name_to_index['recovered'] for i in range(population_size)
         if i in initially_infected_nodes],
        [env.state.health_states[i] for i in range(population_size)
         if i in initially_infected_nodes])

    # After two steps, exposed -> infected.
    env.step(np.arange(population_size))
    self.assertCountEqual(
        [env.state_name_to_index['infected'] for i in range(population_size)
         if i not in initially_infected_nodes],
        [env.state.health_states[i] for i in range(population_size)
         if i not in initially_infected_nodes])

    # After three steps, infected -> recovered.
    env.step(np.arange(population_size))
    self.assertCountEqual(
        [env.state_name_to_index['recovered'] for i in range(population_size)
         if i not in initially_infected_nodes],
        [env.state.health_states[i] for i in range(population_size)
         if i not in initially_infected_nodes])

  def test_graph_space_contains_contact_network(self):
    population_size = 50

    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    graph.add_edges_from(complete_graph_edge_list(population_size))
    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=1.0,
        num_treatments=0,
        max_treatments=10)

    space = graph_space.GraphSpace(population_size, directed=False)

    self.assertTrue(space.contains(env.state.population_graph))

  def test_render_does_not_raise(self):
    graph = nx.Graph()
    graph.add_nodes_from(range(50))
    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=0.5,
        num_treatments=0,
        max_treatments=10,
        initial_health_state=[0 for _ in graph])
    env.render({0: 'blue', 1: 'red'})

  def test_render_fails_with_invalid_mode(self):
    graph = nx.Graph()
    graph.add_nodes_from(range(50))
    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=0.5,
        num_treatments=0,
        max_treatments=10,
        initial_health_state=['susceptible' for _ in graph])
    with self.assertRaises(ValueError):
      env.render({0: 'blue',
                  1: 'red'},
                 mode='an invalid mode')

  def test_dimension_mismatch_in_params_raises(self):
    graph = nx.Graph()
    graph.add_nodes_from(range(50))

    state_names = ['susceptible', 'infected', 'extra state']

    transition_matrix = np.zeros((2, 2))
    treatment_transition_matrix = np.zeros((2, 2))

    with self.assertRaises(ValueError):
      _ = infectious_disease.Params(
          population_graph=graph,
          transition_matrix=transition_matrix,
          treatment_transition_matrix=treatment_transition_matrix,
          state_names=state_names,
          healthy_index=0,
          healthy_exit_index=1,
          infectious_index=1,
          infection_probability=0.1,
          num_treatments=0,
          max_treatments=0)

  def test_unnormalized_transition_in_params_raises(self):
    graph = nx.Graph()
    graph.add_nodes_from(range(50))

    state_names = ['susceptible', 'infected']

    transition_matrix = np.ones((2, 2))
    treatment_transition_matrix = np.ones((2, 2))

    with self.assertRaises(ValueError):
      _ = infectious_disease.Params(
          population_graph=graph,
          transition_matrix=transition_matrix,
          treatment_transition_matrix=treatment_transition_matrix,
          state_names=state_names,
          healthy_index=0,
          healthy_exit_index=1,
          infectious_index=1,
          infection_probability=0.1,
          num_treatments=0,
          max_treatments=0)

  def test_param_equality_correct(self):
    graph = nx.Graph()
    graph.add_nodes_from(range(50))

    state_names = ['susceptible', 'infected']

    transition_matrix = np.ones((2, 2)) * 0.5
    treatment_transition_matrix = np.ones((2, 2)) * 0.5

    num_treatments = max_treatments = 0

    p1 = infectious_disease.Params(
        population_graph=graph,
        transition_matrix=transition_matrix,
        treatment_transition_matrix=treatment_transition_matrix,
        state_names=state_names,
        healthy_index=0,
        healthy_exit_index=1,
        infectious_index=1,
        infection_probability=0.1,
        num_treatments=num_treatments,
        max_treatments=max_treatments)

    p2 = infectious_disease.Params(
        population_graph=graph,
        transition_matrix=transition_matrix,
        treatment_transition_matrix=treatment_transition_matrix,
        state_names=state_names,
        healthy_index=0,
        healthy_exit_index=1,
        infectious_index=1,
        infection_probability=0.1,
        num_treatments=num_treatments,
        max_treatments=max_treatments)

    self.assertEqual(p1, p2)

  def test_rng_initialized_correctly(self):
    # Set up an environment with some infected people but no contact between
    # members of the population.
    graph = nx.Graph()
    graph.add_nodes_from(range(50))
    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=0.5,
        num_treatments=0,
        max_treatments=10,
        initial_health_state=[0 for _ in graph])
    env.state.rng.rand()  # This will fail if rng is not initialized correctly.

  def test_population_transforming_treatments_seir(self):
    """Test that treatment can set the population to a uniform state."""
    seed = 1
    initial_health_state = [
        0, 0, 0, 1, 1, 1, 2, 2, 3, 3,]

    population_size = len(initial_health_state)

    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    graph.add_edges_from(complete_graph_edge_list(population_size))

    for i in range(4):
      treatment_transition_matrix = np.zeros((4, 4))
      treatment_transition_matrix[:, i] = 1
      env = infectious_disease.build_seir_model(
          population_graph=graph,
          # No natural transitions in this setting.
          infection_probability=0,
          exposed_exit_probability=0,
          infected_exit_probability=0,
          num_treatments=population_size,
          max_treatments=population_size,
          treatment_transition_matrix=treatment_transition_matrix,
          initial_health_state=initial_health_state)
      env.set_scalar_reward(lambda x: 0)
      env.seed(seed)

      env.step(np.arange(population_size))
      self.assertCountEqual(
          [i for _ in range(population_size)],
          env.state.health_states)

  def test_transition_matrix_shape_checking_correct(self):
    graph = nx.Graph()
    graph.add_nodes_from(range(50))

    state_names = ['susceptible', 'infected']

    # Same shape shouldn't raise.
    transition_matrix = np.array([[0, 0], [0, 1]])
    treatment_transition_matrix = np.array([[1, 0], [1, 0]])
    _ = infectious_disease.Params(
        population_graph=graph,
        transition_matrix=transition_matrix,
        treatment_transition_matrix=treatment_transition_matrix,
        state_names=state_names,
        healthy_index=0,
        healthy_exit_index=1,
        infectious_index=1,
        infection_probability=0.1,
        num_treatments=0,
        max_treatments=0)

    # Different shape should raise.
    treatment_transition_matrix = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    with self.assertRaises(ValueError):
      _ = infectious_disease.Params(
          population_graph=graph,
          transition_matrix=transition_matrix,
          treatment_transition_matrix=treatment_transition_matrix,
          state_names=state_names,
          healthy_index=0,
          healthy_exit_index=1,
          infectious_index=1,
          infection_probability=0.1,
          num_treatments=0,
          max_treatments=0)

  def test_none_action_leads_to_no_treatment(self):
    seed = 1
    population_size = 25

    # Start everyone in 'susceptible.'
    initial_health_state = [0 for _ in range(population_size)]

    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))

    # Create an environment with no state natural transitions but where
    # treatment allocation always leads to a detectable state change.
    treatment_transition_matrix = np.zeros((4, 4))
    treatment_transition_matrix[:, -1] = 1
    env = infectious_disease.build_seir_model(
        population_graph=graph,
        infection_probability=0.0,
        exposed_exit_probability=0.0,
        infected_exit_probability=0.0,
        num_treatments=population_size,
        max_treatments=population_size,
        treatment_transition_matrix=treatment_transition_matrix,
        initial_health_state=initial_health_state)
    env.set_scalar_reward(lambda x: 0)
    env.seed(seed)

    # Because treatment causes state changes every time it is applied, and,
    # additionally, there are no other sources of state change, maintaining
    # health state between steps implies that no treatment was applied (and
    # thus the None action is working correctly).
    env.step(None)
    self.assertCountEqual(initial_health_state, env.state.health_states)

  def test_observation_is_up_to_date(self):
    """Tests that the observation reflects the population that will be treated.

    Checks that the treatment is applied BEFORE disease spread at every step,
    not after.
    """
    seed = 1
    population_size = 25

    # Start everyone in 'susceptible' except for patient 0 who is infected.
    initial_health_state = [0 for _ in range(population_size)]
    initial_health_state[0] = 1

    # Fully connected contact graph.
    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    graph.add_edges_from(complete_graph_edge_list(population_size))

    # Treatments make infected people recover,
    treatment_transition_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]])

    env = infectious_disease.build_sir_model(
        population_graph=graph,
        infection_probability=1.0,
        infected_exit_probability=0.0,
        treatment_transition_matrix=treatment_transition_matrix,
        num_treatments=1,
        max_treatments=1,
        initial_health_state=initial_health_state)

    env.set_scalar_reward(lambda x: 0)
    env.seed(seed)
    _ = env.reset()
    observation, _, _, _ = env.step([0])

    # Check that the treatment got there before the disease spread. If it got
    # there afterward, many more people would be sick.
    expected_health = [2] + [0] * (population_size - 1)
    self.assertEqual(observation['health_states'].tolist(), expected_health)


if __name__ == '__main__':
  absltest.main()
