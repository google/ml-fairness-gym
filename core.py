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
"""Fairness environment base classes."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
import enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Text, Tuple, TypeVar, Union

from absl import flags
from absl import logging
import attr
import gin
import gym
from gym.utils import seeding
import gym.utils.json_utils
import more_itertools
import networkx as nx
import numpy as np
from recsim.simulator import recsim_gym
import simplejson as json


# Values with associated with this key within dictionaries are given
# special treatment as RandomState internals during JSON serialization /
# deserialization.  This works around an issue where RandomState itself fails
# to serialize.
RANDOM_STATE_KEY = '__random_state__'

flags.DEFINE_bool(
    'validate_history', False,
    'If True, metrics check the validity of the history when measuring. '
    'Can be turned off to save computation.')


class NotInitializedError(Exception):
  """Object is not fully initialized."""
  pass


class InvalidObservationError(Exception):
  """Observation is not valid."""
  pass


class InvalidRewardError(Exception):
  """Reward is not valid."""
  pass


class BadFeatureFnError(Exception):
  """Featurization is not valid."""
  pass


class InvalidHistoryError(Exception):
  """History is not valid."""
  pass


class EpisodeDoneError(Exception):
  """Called act on a done episode."""
  pass


class NotReproducibleError(Exception):
  """Simulation was run in a non-reproducible way."""
  pass


def validate_reward(reward):
  """Raises InvalidRewardError if reward is not None or a scalar."""
  if reward is None:
    return True
  try:
    float(reward)
  except TypeError:
    raise InvalidRewardError


class GymEncoder(json.JSONEncoder):
  """Encoder to handle common gym and numpy objects."""

  def default(self, obj):
    # First check if the object has a to_jsonable() method which converts it to
    # a representation that can be json encoded.
    try:
      return obj.to_jsonable()
    except AttributeError:
      pass

    if callable(obj):
      return {'callable': obj.__name__}

    if isinstance(obj, (bool, np.bool_)):
      return int(obj)

    if isinstance(obj, enum.Enum):
      return {'__enum__': str(obj)}

    if isinstance(obj, recsim_gym.RecSimGymEnv):
      # TODO(): We cannot serialize a full RecSimGymEnv but for now
      # we can note its existence.
      return 'RecSimGym'

    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
      return int(obj)
    if isinstance(obj, (bool, np.bool_)):
      return str(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
      return float(obj)
    if isinstance(obj, nx.Graph):
      return nx.readwrite.json_graph.node_link_data(obj)
    if isinstance(obj, np.random.RandomState):
      state = obj.get_state()
      return {
          RANDOM_STATE_KEY:
              (state[0], state[1].tolist(), state[2], state[3], state[4])
      }
    if isinstance(obj, Params) or isinstance(obj, State):
      return obj.asdict()
    return json.JSONEncoder.default(self, obj)


def to_json(dictionary, sort_keys=True, **kw):
  return json.dumps(dictionary, cls=GymEncoder, sort_keys=sort_keys, **kw)


@attr.s(cmp=False)
class State(object):
  """Simple mutable storage class for state variables."""

  asdict = attr.asdict

  def to_json(self):
    return to_json(self)

  def __eq__(self, other):
    return self.to_json() == other.to_json()

  def __ne__(self, other):
    return self.to_json() != other.to_json()


# TODO(): Find a better type for actions than Any.
ActionType = Any  # pylint: disable=invalid-name


@attr.s
class HistoryItem(object):
  """Data class for state, action pairs that make up a history."""
  state = attr.ib()  # type: State
  action = attr.ib()  # type: ActionType

  def to_jsonable(self):
    return attr.astuple(self)

  # Allow HistoryItems to act like tuples for unpacking.
  def __iter__(self):
    return iter(attr.astuple(self, recurse=False))


HistoryType = List[HistoryItem]


@gin.configurable
@attr.s
class Params(object):
  """Simple mutable storage class for parameter variables."""

  asdict = attr.asdict


ParamsType = TypeVar('ParamsType', bound=Params)


class RewardFn(object):
  """Base reward function.

  A reward function describes how to extract a scalar reward from state or
  changes in state.

  Subclasses should override the __call__ function.
  """

  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    raise NotImplementedError


DEFAULT_GROUP = np.ones(1)
NO_GROUP = np.zeros(1)
DEFAULT_GROUP_SPACE = gym.spaces.MultiBinary(1)


class StateUpdater(object):
  """An object used to update state."""

  def update(self, state, action):
    raise NotImplementedError


class NoUpdate(StateUpdater):
  """Applies no update."""

  def update(self, state, action):
    """Does nothing."""
    del state, action  # Unused.


class FairnessEnv(gym.Env):
  """ML-fairness-gym Environment.

  An ML-fairness-gym environment is an environment that additionally reports to
  an oracle that can determine the potential outcomes for each action that the
  agent takes.

  The main API methods that users of this class need to know are:

  Inherited from gym.Env (see gym/core.py for more documentation):
      step
      reset
      render
      close
      seed
  # TODO(): Add methods to save/restore state.

  Extends gym.Env:
      set_scalar_reward: Allows an agent to specify how the environment should
        translate state or changes in state to a scalar reward.

  Observations returned immediately after reset (initial observations) may not
  be in the observation space. They can be used to establish some prior.
  Subsequent observations are checked at each step to ensure they are contained.

  When implementing a FairnessEnv, override `_step_impl` instead of overriding
  the `step` method.
  """

  observable_state_vars = {}  # type: Mapping[Text, gym.Space]

  # Should inherit from gym.Space
  action_space = None  # type: Optional[gym.Space]

  # group_membership_var should be the name of an observable state variable.
  group_membership_var = None  # type: Optional[Text]
  assert (not group_membership_var or
          (group_membership_var in observable_state_vars))

  def __init__(self,
               params = None,
               initialize_observation_space = True):
    self.history = []  # type: HistoryType
    self.state = None  # type: Optional[State]
    self.reward_fn = None  # type: Optional[RewardFn]

    if initialize_observation_space:
      self.observation_space = gym.spaces.Dict(self.observable_state_vars)
    # Copy params so if environment mutates params it is contained to this
    # environment instance.
    self.initial_params = copy.deepcopy(params)

    def get_group_identifier(observation):
      return observation.get(self.group_membership_var, DEFAULT_GROUP)

    self.group_identifier_fn = get_group_identifier  # type: Callable

  def step(
      self,
      action):
    """Run one timestep of the environment's dynamics.

    This is part of the openAI gym interface and should not be overridden.
    When writing a new ML fairness gym environment, users should override the
    `_step_impl` method.

    Args:
        action: An action provided by the agent. A member of `action_space`.

    Returns:
        observation: Agent's observation of the current environment. A member
          of `observation_space`.
        reward: Scalar reward returned after previous action. This should be the
          output of a `RewardFn` provided by the agent.
        done: Whether the episode has ended, in which case further step() calls
          will return undefined results.
        info: A dictionary with auxiliary diagnostic information.

    Raises:
      NotInitializedError: If called before first reset().
      gym.error.InvalidAction: If `action` is not in `self.action_space`.
    """
    if self.state is None:
      raise NotInitializedError(
          'State is None. State must be initialized before taking a step.'
          'If using core.FairnessEnv, subclass and implement necessary methods.'
      )

    if not self.action_space.contains(action):
      raise gym.error.InvalidAction('Invalid action: %s' % action)

    self._update_history(self.state, action)
    self.state = self._step_impl(self.state, action)
    observation = self._get_observable_state()

    logging.debug('Observation: %s.', observation)
    logging.debug('Observation space: %s.', self.observation_space)

    assert self.observation_space.contains(
        observation
    ), 'Observation %s is not contained in self.observation_space' % observation

    # TODO(): Remove this completely.
    # For compatibility, compute a reward_fn if one is given.
    reward = self.reward_fn(observation) if self.reward_fn is not None else 0
    return observation, reward, self._is_done(), {}

  def seed(self, seed = None):
    """Sets the seed for this env's random number generator."""
    rng, seed = seeding.np_random(seed)
    self.state.rng = rng
    return [seed]

  def reset(self):
    """Resets the state of the environment and returns an initial observation.

    Returns:
      observation: The observable features for the first interaction.
    """
    self._reset_history()
    return self._get_observable_state()

  # TODO(): Remove this.
  def set_scalar_reward(self, reward_fn):
    """Sets the environment's reward_fn.

    `reward_fn` describes how to extract a scalar reward from the environment's
    state or changes in state.
    The agent interacting with the environment is expected to call this function
    if it intends to use the environment's reward response.

    Args:
      reward_fn: A `RewardFn` object.
    """
    self.reward_fn = reward_fn

  def serialize_history(self):
    """Serialize history to JSON.

    Returns:
      A string containing a serialized JSON representation of the environment's
      history.
    """
    # Sanitize history by handling non-json-serializable state.
    sanitized_history = [(json.loads(history_item.state.to_json()),
                          history_item.action) for history_item in self.history]
    return json.dumps(
        {
            'environment': repr(self.__class__),
            'history': sanitized_history
        },
        cls=GymEncoder,
        sort_keys=True)

  ####################################################################
  # Methods to be overridden by each fairness environment.           #
  ####################################################################

  def _step_impl(self, state, action):
    """Run one timestep of the environment's dynamics.

    This should be implemented when creating a new enviornment.

    Args:
        state: A `State` object.
        action: An action provided by the agent. A member of `action_space`.

    Returns:
        An updated `State` object.
    """
    raise NotImplementedError

  def _get_observable_state(self):
    """Extracts observable state from `self.state`.

    Returns:
      A dictionary mapping variable name to a numpy array with that variable's
      value.
    """
    return {
        var_name: np.array(getattr(self.state, var_name))
        for var_name in self.observable_state_vars
    }

  def _get_reward(self):
    """Extracts a scalar reward from `self.state`."""
    return

  def _is_done(self):
    """Extracts whether the episode is done from `self.state`."""
    return False

  #####################
  # Metric interface #
  #####################

  def _get_history(self):
    """This function should only be called by a Metric."""
    return self.history

  def _get_state(self):
    """This function should only be called by a Metric."""
    return copy.deepcopy(self.state)

  #################################
  # Private convenience functions #
  #################################

  def _update_history(self, state, action):
    """Adds state and action to the environment's history."""
    self.history.append(HistoryItem(state=copy.deepcopy(state), action=action))

  def _set_history(self, history):
    self.history = history

  def _reset_history(self):
    """Resets the environment's history."""
    self.history = []

  def _set_state(self, state):
    """Sets the environment's state."""
    self.state = state
    return self


class Metric(object):
  """Base metric class.

  A metric processes the history of interactions between an agent and an
  environment and evaluates some measure of fairness of those interactions.

  The main API methods that users of this class need to know is:

      measure: Takes a FairnessEnv as input and outputs an measure report. The
        type of the measure report is not specified in the base class, but may
        be specified for subclasses.
  """

  def __init__(self,
               environment,
               realign_fn = None):
    # A copy of the environment is used so that simulations do not affect
    # the history of the environment being measured.
    self._environment = copy.deepcopy(environment)
    self._environment_setter = self._environment._set_state  # pylint: disable=protected-access
    self._realign_fn = realign_fn

  def _simulate(self, state, action):
    """Simulates the effect of `action` on `state`.

    Args:
      state: A `State` object.
      action: An action that is in the action space of `self.environment`.

    Returns:
      A new state.
    """
    env = self._environment_setter(state)
    env.step(action)
    simulated_state = env._get_state()  # pylint: disable=protected-access
    return simulated_state

  def _validate_history(self, history):
    """Checks that a history can be replayed using the metric's simulation.

    Args:
      history: an iterable of (state, action) pairs.

    Raises:
      ValueError if the metric's simulation and the history do not match.
    """
    history = copy.deepcopy(history)
    for idx, (step, next_step) in enumerate(more_itertools.pairwise(history)):
      simulated_state = self._simulate(step.state, step.action)
      if simulated_state != next_step.state:
        raise ValueError('Invalid history at step %d %s != %s' %
                         (idx, step, next_step))

  def _extract_history(self, env):
    """Gets and validates a history from an environment."""
    history = env._get_history()  # pylint: disable=protected-access
    if flags.FLAGS.validate_history:
      self._validate_history(history)
    if self._realign_fn is not None:
      return self._realign_fn(history)
    return history

  def measure(self, env):
    """Measures an agent's history of interactions with an environment."""
    raise NotImplementedError


class Agent(object):
  """Base Agent class.

  The main API methods that users of this class need to know is:

      act: Takes (observation, reward, done) from the environment and returns
        an action in the action space of the environment.

  """

  def __init__(self, action_space, reward_fn,
               observation_space):
    """Initializes an Agent.

    Args:
      action_space: a `gym.Space` that contains valid actions.
      reward_fn: a `RewardFn` object.
      observation_space: a `gym.Space` that contains valid observations.
    """
    self.action_space = action_space
    self.reward_fn = reward_fn
    self.observation_space = observation_space
    self.rng = np.random.RandomState()

  def initial_action(self):
    """Returns an action in action_space that is the initial default action."""
    raise NotImplementedError

  # TODO(): Find a better type for observations than Any.
  def act(self, observation, done):
    """Returns an action in the action_space specified in the constructor.

    Do not override this method. When implementing act for a child class,
    override the _act_impl method instead.

    Args:
      observation: An observation in `self.observation_space`.
      done: Boolean indicating whether the simulation has terminated.
    """
    reward = self.reward_fn(observation)
    return self._act_impl(observation, reward, done)

  # TODO(): Find a better type for observations than Any.
  def _act_impl(self, observation, reward, done):
    """The implementation of the agent's act method.

    This should be overridden by any class inheriting from Agent. When calling
    this function, the agent has already replaced the environment's reward
    value with its own.

    Args:
      observation: An observation in `self.observation_space`.
      reward: A scalar reward function that the agent has computed from
        observation.
      done: Boolean indicating whether the simulation has terminated.
    """
    raise NotImplementedError

  # TODO(): Find a better type for observations than Any.
  def flatten_features(self, observation):
    """Flattens observation in `observation_space` into a vector for training.

    Args:
     observation: An observation in `observation_space`.

    Returns:
     A 1-d numpy array containing the values from the observation.
    """
    return np.concatenate([
        np.array(feat).reshape((-1,)) for _, feat in sorted(observation.items())
    ])

  def seed(self, value):
    rng, seed = seeding.np_random(value)
    self.rng = rng
    return [seed]

  def sample_from(self, space):
    """Sample from a space using the agent's own state."""
    space = copy.deepcopy(space)
    space.np_random = self.rng
    return space.sample()
