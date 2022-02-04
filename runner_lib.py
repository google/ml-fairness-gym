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

# Lint as: python2, python3
"""A gin-configurable experiment runner for the fairness gym.

For usage, please see runner.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Callable, Dict, Optional, Text, Type

import attr
import core
import run_util
import gin


@gin.configurable
def default_report(
    env,
    agent,
    metric_results):
  return {
      'environment': {
          'name': env.__class__.__name__,
          'params': env.initial_params
      },
      'agent': {
          'name': agent.__class__.__name__,
      },
      'metrics': metric_results,
  }


@gin.configurable
def run_simulation(env, agent, metrics, num_steps):
  """Runs a single simulation and returns metrics."""
  return run_util.run_simulation(env, agent, metrics, num_steps)


@gin.configurable
def run_stackelberg_simulation(env, agent, metrics, num_steps):
  """Runs a single Stackelberg simulation and returns metrics."""

  return run_util.run_stackelberg_simulation(env, agent, metrics, num_steps)


# The type of a simulation function.  This is used to annotate the simulation_fn
# attr of Runner.
_SimulationFnType = Callable[
    [core.FairnessEnv, core.Agent, Dict[Text, core.Metric], int],
    Dict[Text, Any]]

# The type of a report function.  This is used to annotate the report_fn attr
# of Runner.
_ReportFnType = Callable[
    [core.FairnessEnv, core.Agent, Dict[Text, Any]],
    Dict[Text, Any]]


@gin.configurable
@attr.s
class Runner(object):
  """A gin-configurable class for running experiments."""

  # The agent class to use in this experiment.
  agent_class = attr.ib()  # type: Type[core.Agent]

  # A dictionary that maps metric name strings to metric classes that will
  # be used in this experiment.
  metric_classes = attr.ib()  # type: Dict[Text, Type[core.Metric]]

  # The number of steps to take in this experiment.
  num_steps = attr.ib()  # type: int

  # The random seed to use with this experiment.
  seed = attr.ib()  # type: int

  # TODO(): Once agent seeding capabilities have been added, add
  # an agent seed attribute.

  # The environment class to use in this experiment.  If None, the environment
  # is set through env_callable instead.
  env_class = attr.ib(default=None)  # type: Optional[Type[core.FairnessEnv]]

  # The parameter class that will be used to parameterize the environment.  If
  # None (default), the environment will be instatiated without parameters
  # being passed.
  # This attribute is ignored if env_class is None.
  env_params_class = attr.ib(default=None)  # type: Optional[Type[core.Params]]

  # A callable that returns an instantiated environment.  This is only used if
  # env_class is None.  If both env_class is None and env_callable is None, an
  # exception is raised.
  env_callable = attr.ib(
      default=None)  # type: Optional[Callable[[], core.FairnessEnv]]

  # The function that will be used to perform the experiment.  This function
  # manages the interaction between agent and environment and the progression
  # of the experiment's simulation.
  simulation_fn = attr.ib(default=run_simulation)  # type: _SimulationFnType

  # The function that will be used to report the results of the experiment.
  report_fn = attr.ib(default=default_report)  # type: _ReportFnType

  # TODO(): Add a parallelized parameter-exploring run method.
  def run(self):
    """Runs an experiment and returns results."""

    # Instantiate environment.
    if self.env_class is not None:
      if self.env_params_class is not None:
        env = self.env_class(self.env_params_class())
      else:
        env = self.env_class()
    elif self.env_callable is not None:
      env = self.env_callable()
    else:
      raise ValueError(
          'Both env_class and env_callable are None, so no environment could '
          'be instantiated.')
    env.seed(self.seed)

    # Instantiate metrics.
    metrics = {
        name: metric_class(env)
        for name, metric_class in self.metric_classes.items()}

    # Instantiate agent.
    agent = self.agent_class(
        env.action_space,
        None,
        env.observation_space)

    # Run the simulation and gather metric results.
    metric_results = self.simulation_fn(env, agent, metrics, self.num_steps)

    # Return a report on the simulation.
    return self.report_fn(env, agent, metric_results)
