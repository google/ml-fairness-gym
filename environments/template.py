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
"""Template environment definition."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
from typing import Optional
import attr
import core
from gym import spaces
import numpy as np


@attr.s
class Params(core.Params):
  """Params object for my environment."""
  # Use this to define any parameters you want to pass in at initialization.
  # TODO(): Fill this section in with relevant parameters.
  foo = attr.ib(default=0.0)


# `cmp` must be set to False here to use core.State's equality methods.
@attr.s(cmp=False)
class State(core.State):
  """State object for my environments."""
  # Use this to define any state variables of the environment.
  rng = attr.ib()  # type: np.random.RandomState
  params = attr.ib()  # type: Params
  # TODO(): Fill this section in with relevant state variables.
  bar = attr.ib(default=0.5)
  bat = attr.ib(default=1.0)


class ExampleEnv(core.FairnessEnv):
  """Example Environment."""

  metadata = {'render.modes': ['human']}

  def __init__(self, params = None):
    if params is None:
      params = Params()

    # TODO(): Fill this section in with action_space.
    # Use a gym.Space to describe the action space. In the example below,
    # the action space has two discrete actions.
    self.action_space = spaces.Discrete(2)  # Two possible discrete actions.
    # TODO(): Fill this in with a dict mapping from state variable names to
    # gym.Spaces.
    # In the example below, 'bar' is an observable float between [-10., 10].
    # `bat` is a state variable but it is not observable, so it is not in this
    # dictionary.
    self.observable_state_vars = {
        'bar': spaces.Box(np.array(-10.), np.array(10.), dtype=float)
    }
    # This call sets up env.initial_params and history.
    super(ExampleEnv, self).__init__(params)
    self._state_init()

  def _state_init(self, rng=None):
    """Initialize the environment's state."""
    self.state = State(
        rng=rng or np.random.RandomState(),
        params=copy.deepcopy(self.initial_params),
        # TODO(): Initialize any other state variables here.
    )

  def _step_impl(self, state, action):
    """Run one timestep of the environment's dynamics.

    Args:
      state: A `State` object containing the current state.
      action: An action in `action_space`.

    Returns:
      A `State` object containing the updated state.
    """
    # TODO(): Change the state in response to the input action here.
    # In the example below, `bar` is increased by `foo`.
    # `bat` is decreased by `foo` if action is 0.
    state.bar += state.params.foo
    if not action:
      state.bat -= state.params.foo
    return state

  def reset(self):
    """Resets the environment."""
    self._state_init(self.state.rng)
    return super(ExampleEnv, self).reset()
