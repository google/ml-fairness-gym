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
"""Testing utilities for ML fairness gym."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import attr
import core
import run_util
from agents import random_agents
from spaces import batch
import gin
import gym
import numpy as np
import simplejson as json
from six.moves import range


@attr.s(cmp=False)
class DummyState(core.State):
  x = attr.ib()
  rng = attr.ib()
  params = attr.ib()


@gin.configurable
class DummyEnv(core.FairnessEnv):
  """Simple Dummy Environment used for testing."""
  hidden_state_vars = ['rng']

  observable_state_vars = {
      'x': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
  }

  def __init__(self, params=None):
    self.action_space = gym.spaces.Discrete(2)
    if params is None:
      params = core.Params()  # Empty params.
    super(DummyEnv, self).__init__(params)
    self._state_init()

  def _state_init(self, rng=None):
    self.state = DummyState(
        rng=rng or np.random.RandomState(),
        x=np.array([0.5]),
        params=self.initial_params)

  def reset(self):
    self._state_init(self.state.rng)
    return super(DummyEnv, self).reset()

  def _step_impl(self, state, action):
    """Run one timestep of the environment's dynamics.

    At each timestep, x is resampled from a uniform distribution.

    Args:
      state: A `State` object containing the current state.
      action: An action in `action_space`.

    Returns:
      A `State` object containing the updated state.
    """
    del action  # Unused.
    state.x = state.rng.rand(1)
    return state


@attr.s
class DummyParams(core.Params):
  dim = attr.ib(default=1)


class DeterministicDummyEnv(core.FairnessEnv):
  """Simple Dummy Environment with alternating binary state used for testing."""

  observable_state_vars = {'x': batch.Batch(gym.spaces.Discrete(2))}

  def __init__(self, params=None):
    if params is None:
      params = DummyParams()
    self.action_space = gym.spaces.Discrete(2)
    super(DeterministicDummyEnv, self).__init__(params)
    self._state_init()

  def _state_init(self):
    self.state = DummyState(
        params=copy.deepcopy(self.initial_params),
        rng=None,
        x=[0 for _ in range(self.initial_params.dim)])

  def _step_impl(self, state, action):
    """Run one timestep of the environment's dynamics.

    At each timestep, x is flipped from zero to one or one to zero.

    Args:
      state: A `State` object containing the current state.
      action: An action in `action_space`.

    Returns:
      A `State` object containing the updated state.
    """
    del action  # Unused.
    state.x = [1 - x for x in state.x]
    return state


# TODO(): There isn't actually anything to configure in DummyMetric,
# but we mark it as configurable so that we can refer to it on the
# right-hand-side of expressions in gin configurations.  Find out whether
# there's a better way of indicating that than gin.configurable.
@gin.configurable
class DummyMetric(core.Metric):
  """Simple metric for testing.

  Measurement returns the length of the history.
  """

  def measure(self, env):
    """Returns the length of history."""
    history = self._extract_history(env)
    return len(history)


def setup_test_simulation(env=None, agent=None, metric=None, return_copy=False):
  """Create an environment, agent, and metric for testing purposes.

  Arguments that are left as None will be replaced by dummy versions defined
  in this file.

  Args:
    env: A `core.FairnessEnv` or None.
    agent: A `core.Agent` or None.
    metric: A `core.Metric` or None.
    return_copy: If True, copies of the environment, agent, and auditors are
      returned rather than the originals.

  Returns:
    An (environment, agent, metric) tuple.
  """
  if env is None:
    env = DummyEnv()

  if agent is None:
    agent = random_agents.RandomAgent(env.action_space, None,
                                      env.observation_space)

  if metric is None:
    metric = DummyMetric(env)

  if return_copy:
    return copy.deepcopy(env), copy.deepcopy(agent), copy.deepcopy(metric)

  return env, agent, metric


def run_test_simulation(env=None,
                        agent=None,
                        metric=None,
                        num_steps=10,
                        seed=100,
                        stackelberg=False,
                        check_reproducibility=True):
  """Perform a simple test simulation and return a measurement.

  Arguments that are left as None will be replaced by dummy versions defined
  in this file.

  Args:
    env: A `core.FairnessEnv` or None.
    agent: A `core.Agent` or None.
    metric: A `core.Metric` or None.
    num_steps: An integer indicating the number of steps to simulate.
    seed: An integer indicating a random seed.
    stackelberg: Bool. if true, run a two player stackelberg game else run the
      default simulation.
    check_reproducibility: Bool. If true, run the simulation twice and check
      that the same histories are produced.

  Raises:
    core.NotReproducibleError if the histories of multiple runs do not match.

  Returns:
    A measurement result.
  """
  env, agent, metric = setup_test_simulation(
      env=env, agent=agent, metric=metric)

  # Create the clones before any simulation is run.
  if check_reproducibility:
    # Env doesn't need to be cloned because run_simulation will re-seed and
    # reset the environment.
    clones = copy.deepcopy((agent, metric))

  simulator = (
      run_util.run_stackelberg_simulation
      if stackelberg else run_util.run_simulation)

  result = simulator(env, agent, metric, num_steps, seed=seed, agent_seed=seed)

  if check_reproducibility:
    base_history = env.serialize_history()
    cloned_agent, cloned_metric = clones
    simulator(
        env, cloned_agent, cloned_metric, num_steps, seed=seed, agent_seed=seed)
    cloned_history = env.serialize_history()

    # Check reproducibility by comparing histories of the cloned run with the.
    # original. They should be identical.
    base_history = json.loads(base_history)['history']
    cloned_history = json.loads(cloned_history)['history']
    for step, ((state_a, action_a),
               (state_b,
                action_b)) in enumerate(zip(base_history, cloned_history)):
      if state_a != state_b:
        raise core.NotReproducibleError('Step %d. State mismatch: %s vs %s' %
                                        (step, state_a, state_b))
      if action_a != action_b:
        raise core.NotReproducibleError('Step %d. Action mismatch: %s vs %s' %
                                        (step, action_a, action_b))

  return result

# In keeping with the style of DummyEnv and DummyMetric, alias DummyAgent as
# well.
# pylint: disable=invalid-name
DummyAgent = gin.external_configurable(
    random_agents.RandomAgent, name='test_util.DummyAgent')
