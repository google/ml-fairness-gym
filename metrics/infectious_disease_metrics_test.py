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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from environments import infectious_disease
from metrics import infectious_disease_metrics
import networkx as nx
import numpy as np


class InfectiousDiseaseMetricsTest(absltest.TestCase):

  def test_healthy_population_counted_correctly(self):
    num_steps = 4
    population_size = 25
    healthy_state = 0

    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=0.0,
        num_treatments=0,
        max_treatments=population_size,
        initial_health_state=[healthy_state for _ in graph])

    for _ in range(num_steps):
      env.step(np.arange(population_size))

    metric = infectious_disease_metrics.PersonStepsInHealthState(
        env, healthy_state)
    self.assertEqual(metric.measure(env), num_steps * population_size)

  def test_disease_prevalence_correct(self):
    num_steps = 4
    population_size = 40
    healthy_state = 0
    infectious_state = 1

    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    env = infectious_disease.build_si_model(
        population_graph=graph,
        infection_probability=0.0,
        num_treatments=0,
        max_treatments=population_size,
        initial_health_state=[
            healthy_state if i % 2 == 0 else infectious_state
            for i in range(graph.number_of_nodes())])

    # Infection rates shouldn't change, so the most recent infection rate should
    # be the same at each step.
    metric = infectious_disease_metrics.DiseasePrevalence(env)
    for _ in range(num_steps):
      env.step(np.arange(population_size))
      self.assertEqual(0.5, metric.measure(env))


if __name__ == '__main__':
  absltest.main()
