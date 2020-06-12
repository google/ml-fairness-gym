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

# Lint as: python3
"""Wrapper code for a recsim environment."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Any
import attr
import core


@attr.s
class Params(core.Params):
  """Params object for recsim wrapper."""
  recsim_env = attr.ib()


@attr.s(cmp=False)
class State(core.State):
  """State object for recsim wrapper."""
  recsim_env = attr.ib()
  observation = attr.ib(default=None)
  is_done = attr.ib(default=False)


class RecsimWrapper(core.FairnessEnv):
  """Wraps a recsim environment as a FairnessEnv."""

  def __init__(self, params=None):
    """Initializes RecsimWrapper."""
    super(RecsimWrapper, self).__init__(
        params, initialize_observation_space=False)
    self.state = State(recsim_env=params.recsim_env)

  # The use of @property here is intentional. RecsimGym objects have
  # action_space and observation_space as properties because they are updated
  # over the course of the simulation. In order to keep up to date with the
  # current spaces, this wrapper must do the same thing.
  @property
  def action_space(self):
    return self.state.recsim_env.action_space

  @property
  def observation_space(self):
    return self.state.recsim_env.observation_space

  def _step_impl(self, state, action):
    obs, _, done, _ = state.recsim_env.step(action)
    state.observation = obs
    state.is_done = done
    return state

  def _get_observable_state(self):
    return self.state.observation

  def _is_done(self):
    return self.state.is_done

  def reset(self):
    """Resets the environment."""
    observation = self.state.recsim_env.reset()
    self.state.observation = observation
    return observation


def wrap(environment):
  """Wrap a Recsim Environment to be an ML Fairness Gym Environment."""
  return RecsimWrapper(Params(environment))
