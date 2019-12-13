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
"""General library for experiments with the infectious disease environment."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import attr
import core
import rewards
from agents import infectious_disease_agents
from environments import infectious_disease
import networkx as nx
import numpy as np
import tqdm

GRAPHS = {'karate': nx.karate_club_graph(),
          'chain': nx.generators.path_graph(12)}


class DayTracker(core.Metric):
  """Counts the number of individuals in a health state every step."""

  def __init__(self, env,
               health_state):
    self.health_state_idx = health_state
    super(DayTracker, self).__init__(env)

  def measure(self, env):
    history = self._extract_history(env)
    return [
        np.count_nonzero(
            np.array(step.state.asdict()['health_states']) ==
            self.health_state_idx) for step in history
    ]


class StateTracker(core.Metric):
  """Returns the full health state of the population for every step."""

  def measure(self, env):
    history = self._extract_history(env)
    return [step.state.asdict()['health_states'] for step in history]


class RandomAgent(infectious_disease_agents._BaseAgent):  # pylint: disable=protected-access
  """An agent that treats people chosen at random."""

  def _triage(self, observation):
    max_treatments = len(self.action_space.nvec)
    triage_scores = np.array(
        [self.rng.rand() for _ in observation['health_states']])
    return np.argsort(triage_scores)[:max_treatments]


class NullAgent(infectious_disease_agents._BaseAgent):  # pylint: disable=protected-access
  """Agent that does not treat anyone."""

  def _triage(self, observation):
    """Treats nobody by returning None."""
    pass


# TODO(): Move to new experiment runner.
@attr.s
class Experiment(core.Params):
  """Specifies the parameters of an experiment to run."""
  ######################
  # Environment params #
  ######################

  # Extremely infectious settings.
  infection_probability = attr.ib(default=0.25)
  infected_exit_probability = attr.ib(default=0.01)
  num_treatments = attr.ib(default=1)
  burn_in = attr.ib(default=3)
  graph_name = attr.ib(default='chain')

  ################
  # Agent params #
  ################
  agent_constructor = attr.ib(default=RandomAgent)

  ##############
  # Run params #
  ##############

  env_seed = attr.ib(default=27)  # Random seed.
  seed = attr.ib(default=100)
  num_steps = attr.ib(default=100)  # Number of steps in the experiment.

  def scenario_builder(self):
    """Returns an agent and environment pair."""
    graph = GRAPHS[self.graph_name]

    env = infectious_disease.build_sir_model(
        population_graph=graph,
        infection_probability=self.infection_probability,
        infected_exit_probability=self.infected_exit_probability,
        num_treatments=self.num_treatments,
        max_treatments=1,
        burn_in=self.burn_in,
        # Treatments turn susceptible people into recovered without having them
        # get sick.
        treatment_transition_matrix=np.array([[0, 0, 1],
                                              [0, 1, 0],
                                              [0, 0, 1]]),
        # Everybody starts out healthy.
        initial_health_state=[0] * graph.number_of_nodes(),
        initial_health_state_seed=self.env_seed)

    agent = self.agent_constructor(
        env.action_space,
        rewards.NullReward(),
        env.observation_space,
        params=infectious_disease_agents.env_to_agent_params(
            env.initial_params))

    return env, agent

  def run(self):
    """Runs a single simulation and returns measured metrics."""
    env, agent = self.scenario_builder()

    # Choose a single patient at random to be sick.
    initial_health_state = [0 for _ in env.initial_params.initial_health_state]
    patient0 = env.state.rng.choice(
        len(env.initial_params.initial_health_state))
    initial_health_state[patient0] = 1
    env.set_initial_health_state(initial_health_state)

    metrics = {'states': StateTracker(env),
               'sick-days': DayTracker(env, health_state=1)}

    env.seed(self.seed)
    # The reset function also runs `burn_in` number of steps.
    observation = env.reset()

    # Run the main simulation.
    done = False
    for _ in tqdm.trange(self.num_steps):
      action = agent.act(observation, done)
      observation, _, done, _ = env.step(action)
      assert not done, ('The environment ended the simulation before the '
                        'experiment was over.')
    metric_results = {
        name: metric.measure(env) for name, metric in metrics.items()
    }
    return {'metric_results': metric_results}
