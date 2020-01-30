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
"""Agent whose actions are randomly sampled from the action space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import core
import rewards
import gin
import gym


@gin.configurable
class RandomAgent(core.Agent):
  """Simple agent that takes random actions."""

  def __init__(self,
               action_space,
               reward_fn,
               observation_space,
               default_action=None):
    """Initializes the random agent, which takes randomly sampled actions.

    Args:
      action_space: A gym.space defining the space of possible actions.
      reward_fn: A function that takes an observation and calculates the agents'
        reward.
      observation_space: A gym.space defining the space of possible
        observations.
      default_action: The first action of the agent when no observation is
        given.
    """
    if reward_fn is None:
      reward_fn = rewards.NullReward()
    super(RandomAgent, self).__init__(action_space, reward_fn,
                                      observation_space)
    self.default_action = default_action

  def initial_action(self):
    """Describes default action of the agent when no observation is given."""
    if self.default_action is not None:
      action = self.default_action
    else:
      action = self.sample_from(self.action_space)
    if not self.action_space.contains(action):
      raise gym.error.InvalidAction('Invalid action: %s' % action)
    return action

  def _act_impl(self, observation, reward, done):
    """Returns an action from `self.action_space`.

    Args:
      observation: An observation in self.observation_space.
      reward: A scalar value that can be used as a supervising signal.
      done: A boolean indicating whether the episode is over.

    Raises:
      core.EpisodeDoneError if `done` is True.
      core.InvalidObservationError if observation is not in
        `self.observation_space`.
      core.InvalidRewardError if reward is not a scalar or None.
    """
    if done:
      raise core.EpisodeDoneError('Called act on a done episode.')

    if not self.observation_space.contains(observation):
      raise core.InvalidObservationError('Invalid observation: %s' %
                                         observation)

    core.validate_reward(reward)
    # Use `_sample_from` so that the randomness comes from the agent's random
    # state rather than the action_space's random_state may be changed by other
    # parties.
    return self.sample_from(self.action_space)
