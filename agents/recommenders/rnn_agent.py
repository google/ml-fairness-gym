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
"""Implements a Recsim agent that uses an RNN model for making recommendations.

The RNNAgent class implements an RNN agent that can receive reward feedback
from a Recsim environment and update the model.
"""

from absl import flags
from absl import logging
from agents.recommenders import model
from agents.recommenders import utils
import numpy as np
import recsim


FLAGS = flags.FLAGS


class RNNAgent(recsim.agent.AbstractEpisodicRecommenderAgent):
  """Defines an RNN that stores and recommends user recommendations."""

  def __init__(self, observation_space, action_space, max_episode_length,
               embedding_size=64,
               hidden_size=64, optimizer_name='Adam', gamma=0.99, epsilon=0.0,
               replay_buffer_size=100, constant_baseline=0.0,
               load_from_checkpoint=None, regularization_coeff=0.0,
               random_seed=None):
    """RNN Agent that makes one recommendation at a time.

    Args:
      observation_space: Environment.observation_space object.
      action_space: Environment.action_space object.
      max_episode_length: maximum length of a user's episode.
      embedding_size: Previous recommendation feature embedding size.
      hidden_size: Size of the LSTM hidden layer.
      optimizer_name: Name of the keras optimizer. Supports 'Adam', 'SGD'.
      gamma: Gamma for discounting future reward (traditional RL meaning).
      epsilon: Epsilon for the epsilon-greedy exploration.
      replay_buffer_size: Number of trajectories stored in the buffer before
        performing an update on the model.
      constant_baseline: Constant baseline value to subtract from reward to
        reduce variance.
      load_from_checkpoint: File name for the model file to load from.
      regularization_coeff: L2 regularization coefficient for all layers.
      random_seed: Seed to be used for the RandomState of the agent.
    """
    self.observation_space = observation_space
    self.action_space = action_space
    self.action_space_size = action_space.nvec[0]
    self._rng = np.random.RandomState(random_seed)

    self.padding_token = int(self.action_space_size + 1)
    self.start_token = int(self.action_space_size)
    self.max_episode_length = max_episode_length

    self.reset_current_episode_logs()

    self.gamma = gamma
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.optimizer_name = optimizer_name
    self.regularization_coeff = regularization_coeff
    if load_from_checkpoint:
      # TODO(): Check if the model is according to arguments.
      self.model = utils.load_model(load_from_checkpoint, self.optimizer_name)
    else:
      self.build_model()
    self.empty_buffer()
    self.epsilon = epsilon
    if self.epsilon < 0.0 or self.epsilon > 1.0:
      raise ValueError(
          f'Epsilon should be between 0 and 1 but it is {self.epsilon}')
    self.constant_baseline = constant_baseline

  def build_model(self):
    self.model = model.create_model(
        max_episode_length=self.max_episode_length,
        action_space_size=self.action_space_size,
        embedding_size=self.embedding_size,
        hidden_size=self.hidden_size,
        batch_size=None,
        optimizer_name=self.optimizer_name,
        regularization_coeff=self.regularization_coeff)

  def reset_current_episode_logs(self):
    self.curr_recommendation_list = np.ones(
        self.max_episode_length+1, dtype=np.int) * self.padding_token
    self.curr_recommendation_list[0] = self.start_token
    self.curr_reward_list = np.zeros(self.max_episode_length+1)
    self.curr_trajectory_length = 0

  def empty_buffer(self):
    """Clears the history stored by the agent."""
    self.replay_buffer = {'recommendation_seqs': [], 'reward_seqs': []}

  def get_model_prediction(self):
    """Returns the Softmax layer for the last time step."""
    curr_len = self.curr_trajectory_length
    softmax_all_layers = self.model.predict(
        # Offset by one to use (recs, rewards) for the input.
        [np.array([self.curr_recommendation_list[:-1]]),
         np.array([self.curr_reward_list[:-1]])])
    return softmax_all_layers[:, curr_len - 1]

  def end_episode(self, reward, observation, eval_mode=False):
    """Stores the last reward, updates the model. No recommendation returned."""
    self.curr_reward_list[self.curr_trajectory_length] = reward
    self.curr_trajectory_length += 1
    self.replay_buffer['recommendation_seqs'].append(
        self.curr_recommendation_list)
    self.replay_buffer['reward_seqs'].append(self.curr_reward_list)
    if not eval_mode:
      self.model_update()
      self.empty_buffer()  # Empty the buffer after updating the model.
    self.reset_current_episode_logs()

  def step(self,
           reward,
           observation,
           eval_mode=False,
           deterministic=False):
    """Update the model using the reward, and recommends the next slate."""
    self.curr_reward_list[self.curr_trajectory_length] = reward
    self.curr_trajectory_length += 1

    # make next recommendation
    softmax_probs = self.get_model_prediction()[0]
    rec = self._choose_rec_from_softmax(softmax_probs, deterministic)
    self.curr_recommendation_list[self.curr_trajectory_length] = rec
    return [rec]  # return the slate

  def _choose_rec_from_softmax(self, softmax_probs, deterministic):
    if deterministic:
      rec = np.argmax(softmax_probs)
    else:
      # Fix the probability vector to avoid np.random.choice exception.
      softmax_probs = np.nan_to_num(softmax_probs)
      softmax_probs += 1e-10
      if not np.any(softmax_probs):
        logging.warn('All zeros in the softmax prediction.')
      softmax_probs = softmax_probs / np.sum(softmax_probs)
      # TODO(): Use epsilon for exploration at the model level.
      rec = self._rng.choice(self.action_space_size, p=softmax_probs)
    return rec

  # TODO(): Move the simulation function to a runner class.
  def simulate(self, env, num_episodes=100,
               episode_length=None, eval_mode=False,
               deterministic=False, initial_buffer_size_before_training=0):
    """Run a number of iterations between the agent and env.

    Args:
      env: RecSimGymEnv environment that supplies the rewards.
      num_episodes: Number of episodes/users to iterate through.
      episode_length: The length of trajectory for each user.
      eval_mode: Set to true to not learn after each episode.
      deterministic: Whether to choose the argmax from the softmax rather than
        sampling.
      initial_buffer_size_before_training: Number of episodes in the beginning
        to collect before starting to train.
    """
    if deterministic and not eval_mode:
      logging.warning(
          'The simulation is set to use a deterministic policy, '
          'with eval_mode set to False. The policy might not learn anything.'
      )
    for episode_number in range(num_episodes):
      if episode_number < initial_buffer_size_before_training:
        # Do not update the model for initial_buffer_size_before_training
        # number of episodes
        curr_eval_mode = True
      else:
        curr_eval_mode = eval_mode
      reward = 0
      observation = env.reset()
      if episode_length is None:
        episode_length = self.max_episode_length
      for _ in range(episode_length):
        slate = self.step(reward, observation, eval_mode=curr_eval_mode,
                          deterministic=deterministic)
        observation, reward, _, _ = env.step(slate)
      self.end_episode(reward, observation,
                       eval_mode=curr_eval_mode)

  def model_update(self):
    """Updates the agent's model and returns the training history.

    The model takes num_epochs number of gradient steps on the current replay
    buffer.

    Returns:
      Object returned by keras.fit that contains the history of losses and
      other logged metrics during training.
    """
    formatted_data = utils.format_data(
        self.replay_buffer, self.gamma, self.constant_baseline)
    loss_value = self.model.train_on_batch(
        formatted_data['input'],
        formatted_data['output'],
        sample_weight=formatted_data['sample_weights_temporal']
        )
    return loss_value

  def change_model_lr(self, learning_rate):
    """Changes the model's learning rate."""
    model.change_optimizer_lr(self.model, learning_rate)
