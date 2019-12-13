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

"""Library functions for training with dopamine."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import functools

from absl import flags
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment
import core
from experiments import infectious_disease as infectious_disease_exp
import numpy as np
import tensorflow as tf

flags.DEFINE_string(
    'output_dir',
    '/tmp/ml-fairness-gym/infection_rl',
    'directory to write files.')

FLAGS = flags.FLAGS


class _DopamineWrapper(gym_lib.GymPreprocessing):
  """Wraps the infectious disease environment to be compatible with Dopamine."""

  def format_observation(self, obs, padding=2):
    """Formats health state observations into a numpy array.

    The health-states are one-hot encoded as row vectors, and then stacked
    together vertically to create a |population| x |health states| array.

    The population is padded on top and bottom with "recovered" indivduals,
    which don't affect the disease spread but make convolutions simpler.

    Args:
      obs: An observation dictionary.
      padding: An integer indicating how many people to use for padding.

    Returns:
      A numpy array suitable for passing to a DQN agent.
    """
    vecs = []
    initial_params = self.environment.initial_params
    num_states = len(initial_params.state_names)
    recovered_state = initial_params.state_names.index('recovered')
    for state in obs['health_states']:
      vecs.append(np.zeros((num_states, 1), dtype=float))
      vecs[-1][state] = 1.0
    pad = np.zeros((num_states, padding))
    pad[recovered_state, :] = 1.0
    return np.hstack([pad] + vecs + [pad]).T

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def observation_shape(self):
    return self.format_observation(
        self.environment.observation_space.sample()).shape

  @property
  def initial_params(self):
    return self.environment.initial_params

  def reset(self):
    """Resets the environment and chooses an initial patient to infect."""
    observation = self.environment.reset()
    self.environment.set_scalar_reward(
        NegativeDeltaPercentSick(_percent_sick(observation)))
    return self.format_observation(observation)

  def step(self, action):
    """Play the environment one step forward."""
    action = np.array([action])
    observation, reward, game_over, info = self.environment.step(action)
    self.game_over = game_over
    return self.format_observation(observation), reward, game_over, info

  def set_initial_health_state(self, initial_health_state):
    return self.environment.set_initial_health_state(initial_health_state)


def _percent_sick(observation):
  return np.mean(
      [health_state == 1 for health_state in observation['health_states']])


class NegativeDeltaPercentSick(core.RewardFn):
  """Reward function that penalizes newly sick days."""

  def __init__(self, base=0):
    super(NegativeDeltaPercentSick, self).__init__()
    self.base = base

  def __call__(self, observation):
    percent_sick = _percent_sick(observation)
    delta = percent_sick - self.base
    self.base = percent_sick
    return -delta


def _create_environment(seed=100, network='chain'):
  """Returns a Dopamine-compatible version of the infectious disease env."""
  experiment = infectious_disease_exp.Experiment(graph_name=network)
  env, _ = experiment.scenario_builder()
  env.seed(seed)
  env.reset()
  env.set_scalar_reward(NegativeDeltaPercentSick())
  return _DopamineWrapper(env)


DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])


class _SimpleDQNNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's Q-values."""

  def __init__(self, num_actions, hidden_layer_size=64, name=None):
    """Creates the layers used for calculating Q-values.

    Args:
      num_actions: int, number of actions.
      hidden_layer_size: int, number of hidden units.
      name: str, used to create scope for network parameters.
    """
    super(_SimpleDQNNetwork, self).__init__(name=name)

    self.num_actions = num_actions
    activation_fn = tf.keras.activations.relu
    # Set names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv1D(
        32,
        5,
        strides=1,
        padding='valid',
        activation=activation_fn,
        name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        hidden_layer_size, activation=activation_fn, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Parameters created here will have scope according to the `name` argument
    given at `.__init__()` call.
    Args:
      state: Tensor, input tensor.

    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    # Fully connected network. No convolutions or graph convolutions here.
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    return DQNNetworkType(x)


def _create_agent(sess,
                  environment,
                  summary_writer,
                  gamma=0.95,
                  hidden_layer_size=32,
                  learning_rate=0.00025):
  """Returns a DQN agent."""
  return dqn_agent.DQNAgent(
      sess,
      network=functools.partial(
          _SimpleDQNNetwork, hidden_layer_size=hidden_layer_size),
      num_actions=int(environment.action_space.nvec[0]),
      observation_shape=environment.observation_shape,
      observation_dtype=tf.int32,
      gamma=gamma,
      stack_size=1,
      epsilon_train=0.5,
      min_replay_history=1000,
      summary_writer=summary_writer,
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate))


def dopamine_train(base_dir,
                   hidden_layer_size,
                   gamma,
                   learning_rate,
                   num_train_steps,
                   network='chain'):
  """Train an agent using dopamine."""
  runner = run_experiment.Runner(
      base_dir,
      functools.partial(
          _create_agent,
          hidden_layer_size=hidden_layer_size,
          gamma=gamma,
          learning_rate=learning_rate),
      functools.partial(_create_environment, network=network),
      num_iterations=num_train_steps,
      training_steps=500,
      evaluation_steps=100,
      max_steps_per_episode=20)
  runner.run_experiment()
  return runner


def dopamine_eval(runner, patient0, seed=100):
  """Evaluate an agent."""

  base_env = runner._environment.environment  # pylint: disable=protected-access
  initial_health_state = np.zeros_like(
      base_env.initial_params.initial_health_state)
  initial_health_state[patient0] = 1
  base_env.set_initial_health_state(initial_health_state)
  base_env.seed(seed)
  base_env.reset()

  metrics = {
      'state_tracker': infectious_disease_exp.StateTracker(base_env),
      'sick-days': infectious_disease_exp.DayTracker(base_env, 1)
  }
  runner._agent.eval_mode = True  # pylint: disable=protected-access
  runner._run_one_episode()  # pylint: disable=protected-access
  retval = {name: metric.measure(base_env) for name, metric in metrics.items()}
  retval['actions'] = [step.action for step in base_env.history]
  return retval
