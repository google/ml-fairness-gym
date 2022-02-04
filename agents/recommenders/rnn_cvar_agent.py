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

# Lint as: python3
"""Implements a RNN RecSim agent that constrains CVaR and optimizes reward."""

from absl import flags
from agents.recommenders import rnn_agent
from agents.recommenders import utils
import numpy as np

FLAGS = flags.FLAGS


class SafeRNNAgent(rnn_agent.RNNAgent):
  """Defines an RNN with CVaR variables that recommends user recommendations."""

  def __init__(self, observation_space, action_space, max_episode_length,
               embedding_size=64,
               hidden_size=64,
               optimizer_name='Adam',
               gamma=0.99, epsilon=0.0,
               replay_buffer_size=100,
               initial_lambda=0.1,
               alpha=0.95,
               beta=0.3,
               max_cost=1.0,
               min_cost=0.0,
               load_from_checkpoint=None,
               regularization_coeff=0.0,
               random_seed=None):
    """RNN Agent with safety constraints.

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
      initial_lambda: Initial Value of lambda for the CVaR optimization.
      alpha: Alpha in the definition of CVaR.
      beta: The upper bound for CVaR.
      max_cost: Maximum safety cost value.
      min_cost: Minimum safety cost value.
      load_from_checkpoint: Filepath to load the model from.
      regularization_coeff: Regularization coefficient for all the layers.
      random_seed: Random seed for the agent's selection from the softmax.
    """
    super(SafeRNNAgent, self).__init__(
        observation_space,
        action_space,
        max_episode_length,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        optimizer_name=optimizer_name,
        gamma=gamma,
        epsilon=epsilon,
        replay_buffer_size=replay_buffer_size,
        load_from_checkpoint=load_from_checkpoint,
        regularization_coeff=regularization_coeff,
        random_seed=random_seed)
    self.curr_safety_costs = []
    self.policy_var = None  # uninitialized
    self.lmbda = initial_lambda
    self.alpha = alpha
    self.beta = beta
    self.min_cost = min_cost
    self.max_cost = max_cost
    # Intiialize var to something close to initial VaR of the agent
    # Since the agent gives healthy recommendations half the time,
    # VaR(Health)~ 0.5 for all alpha since the distribution is expected to be
    # highly peaked around 0.5.
    self.var = 0.5 * (self.max_cost - self.min_cost)
    self.constant_baseline = 0.0
    self.empty_buffer()

  def empty_buffer(self):
    """Clears the history stored by the agent."""
    self.replay_buffer = {
        'recommendation_seqs': [],
        'reward_seqs': [],
        'safety_costs': []
    }

  def _get_safety_cost(self, observation):
    return 1-observation['response'][0]['health_score']

  def end_episode(self, reward, observation, eval_mode=False):
    """Stores the last reward, updates the model. No recommendation returned."""

    self.curr_safety_costs.append(self._get_safety_cost(observation))
    self.replay_buffer['safety_costs'].append(
        np.mean(self.curr_safety_costs[1:]))  # Skipping index 0 as it is None.
    self.curr_safety_costs = []
    super(SafeRNNAgent, self).end_episode(reward, observation, eval_mode)
    return

  def step(self,
           reward,
           observation,
           eval_mode=False,
           deterministic=False):
    """Update the model using the reward, and recommends the next slate."""
    if observation['response']:  # Response is None for the first step.
      self.curr_safety_costs.append(self._get_safety_cost(observation))
    else:
      self.curr_safety_costs.append(None)  # first step in the user's episode.
    return super(SafeRNNAgent, self).step(reward, observation, eval_mode,
                                          deterministic)

  def model_update(self, num_epochs=1, batch_size=None,
                   learning_rate=None,
                   var_learning_rate=0.1,
                   lambda_learning_rate=0.01):
    """Updates the agent's model and returns the training history.

    The model takes num_epochs number of gradient steps on the current replay
    buffer.

    Args:
      num_epochs: Number of epochs for training. Defaults to 1 i.e. every
        trajectory passes through the model once making it online-only trianing.
      batch_size: Mini batch size.
      learning_rate: Learning rate for the RNN model.
      var_learning_rate: Learning rate for the Value-at-Risk updates.
      lambda_learning_rate: Learning rate for the lambda update.

    Returns:
      Object returned by keras.fit that contains the history of losses and
      other logged metrics during training, and updated values of VaR and
      lambda.
    """
    if learning_rate:
      self.change_model_lr(learning_rate)
    if batch_size is None:
      batch_size = len(self.replay_buffer['reward_seqs'])
    curr_var = self.var
    training_history = self._update_params()
    self._update_var(var_learning_rate)
    self._update_lambda(curr_var, lambda_learning_rate)
    return training_history, self.var, self.lmbda

  def _update_params(self):
    formatted_data = utils.format_data_safe_rl(self.replay_buffer, self.gamma,
                                               self.constant_baseline)
    loss_value = self.model.train_on_batch(
        formatted_data['input'],
        formatted_data['output'],
        sample_weight=self._calculate_weights_for_reinforce(
            formatted_data['reward_weights'],
            formatted_data['trajectory_costs']))
    return loss_value

  def _update_var(self, var_learning_rate):
    """Update rule for VaR maintained by the agent."""
    curr_var = self.var
    cost_greater_eq_var = np.array(
        self.replay_buffer['safety_costs']) >= self.var
    self.var = curr_var - var_learning_rate * self.lmbda * (
        1 - (np.mean(cost_greater_eq_var) / (1 - self.alpha)))
    self.var = np.clip(self.var, self.min_cost, self.max_cost)

  def _update_lambda(self, current_var, lambda_learning_rate):
    """Update rule for that lambda (langrangian parameter)."""
    margin_over_var = np.array(self.replay_buffer['safety_costs']) - current_var
    mean_margin_over_var = np.mean(np.clip(margin_over_var, 0, None))
    self.lmbda = self.lmbda + lambda_learning_rate * (
        current_var - self.beta + (1.0 /
                                   (1.0 - self.alpha)) * mean_margin_over_var)
    self.lmbda = np.clip(self.lmbda, 0.0, None)

  def _calculate_weights_for_reinforce(self, reward_weights, trajectory_costs):
    """Calculates weights for REINFORCE using rewards and trajectory costs.

    Args:
      reward_weights: rewards for each timestep in the trajectory
      trajectory_costs: costs for the entire trajectory
    Returns:
      Weights for each softmax output layer in the RNN.
    """
    return np.array(reward_weights) - self._calculate_safe_rl_correction(
        trajectory_costs)[:, np.newaxis]

  def _calculate_safe_rl_correction(self, cost_trajectory):
    """Returns the correction vector to subtract from reward at each timestep.

    Args:
      cost_trajectory: numpy array of safety costs for a single trajectory,
      or a number of trajectories.
    """
    return (self.lmbda /
            (1 - self.alpha)) * np.clip(cost_trajectory - self.var, 0, None)
