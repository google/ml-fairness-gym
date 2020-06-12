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
"""Utilities for RecSim agent.

Defines a few functions used by the RecSim RNNAgent.
"""


import itertools
import os
import tempfile
from absl import flags
import file_util
from agents.recommenders import model
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def accumulate_rewards(rewards, gamma):
  """Computes the discounted reward for the entire episode."""
  reversed_rewards = rewards[::-1]  # list reversal
  acc = list(itertools.accumulate(reversed_rewards, lambda x, y: x*gamma + y))
  return np.array(acc[::-1])


def format_data(data_history, gamma, constant_baseline=0.0):
  """The function formats the data into input, output format for keras."""
  inp_rec_seq, inp_reward_seq, output_recs, reward_weights = [], [], [], []
  for curr_recs, curr_rewards in zip(data_history['recommendation_seqs'],
                                     data_history['reward_seqs']):
    inp_rec_seq.append(curr_recs[:-1])
    inp_reward_seq.append(curr_rewards[:-1])
    output_recs.append(np.expand_dims(curr_recs[1:], axis=-1))
    output_rewards = accumulate_rewards(curr_rewards[1:] - constant_baseline,
                                        gamma)

    reward_weights.append(output_rewards)
  return {'input': [np.array(inp_rec_seq), np.array(inp_reward_seq)],
          'output': np.array(output_recs),
          'sample_weights_temporal': np.array(reward_weights)}


def format_data_safe_rl(data_history, gamma, constant_baseline=0.0):
  """The function formats the data into input, output format for keras.

  This function is specific to the implementation of CVaR safety constraint.
  See https://braintex.goog/read/zyprpgsjbtww for more details.

  Args:
    data_history: dict with recommendation_seqs, reward_seqs, safety_costs
    fields.
    gamma: Gamma for reward accumulation over the time horizon.
    constant_baseline: Baseline to subtract from each reward to reduce variance.

  Returns:
    A dictionary with input, output and sample weights_temporal fields
  that are input into a keras model.
  """
  inp_rec_seq, inp_reward_seq, output_recs, reward_weights = [], [], [], []
  trajectories_cost = []
  for curr_recs, curr_rewards, curr_safety_costs in zip(
      data_history['recommendation_seqs'],
      data_history['reward_seqs'],
      data_history['safety_costs']):
    inp_rec_seq.append(np.array(curr_recs[:-1]))
    inp_reward_seq.append(np.array(curr_rewards[:-1]))
    output_recs.append(np.expand_dims(np.array(curr_recs[1:]), axis=-1))
    output_rewards = accumulate_rewards(curr_rewards[1:] - constant_baseline,
                                        gamma)
    reward_weights.append(output_rewards)
    cost_trajectory = np.mean(curr_safety_costs)
    trajectories_cost.append(cost_trajectory)
  return {
      'input': [np.array(inp_rec_seq),
                np.array(inp_reward_seq)],
      'output': np.array(output_recs),
      'reward_weights': np.array(reward_weights),
      'trajectory_costs': np.array(trajectories_cost)
  }


def format_data_movielens(data_history, gamma, constant_baseline=0.0,
                          mask_already_recommended=False, user_id_input=True,
                          **kwargs):
  """Format data for movielens RNN agent update step."""
  inp_rec_seq, inp_reward_seq, output_recs, reward_weights = [], [], [], []
  user_id_seq = []
  trajectories_cost = []
  if mask_already_recommended:
    # TODO(): Change argument to repeat_movies to be consistent.
    masks_for_softmax = []
  for user_id, curr_recs, curr_rewards, curr_safety_costs in zip(
      data_history['user_id'],
      data_history['recommendation_seqs'],
      data_history['reward_seqs'],
      data_history['safety_costs']):
    inp_rec_seq.append(np.array(curr_recs[:-1]))
    inp_reward_seq.append(np.array(curr_rewards[:-1]))
    output_recs.append(np.expand_dims(np.array(curr_recs[1:]), axis=-1))
    output_rewards = accumulate_rewards(curr_rewards[1:] - constant_baseline,
                                        gamma)
    user_id_seq.append(np.array([user_id] * len(curr_recs[:-1])))
    reward_weights.append(output_rewards)
    cost_trajectory = np.mean(curr_safety_costs)
    trajectories_cost.append(cost_trajectory)
    masks_for_softmax.append(get_mask_for_softmax(curr_recs[1:-1],
                                                  kwargs['action_space_size']))
  input_list = [np.array(inp_rec_seq),
                np.array(inp_reward_seq)]
  if user_id_input:
    input_list.append(np.array(user_id_seq))
  if mask_already_recommended:
    input_list.append(np.array(masks_for_softmax))
  return {
      'input': input_list,
      'output': np.array(output_recs),
      'reward_weights': np.array(reward_weights),
      'trajectory_costs': np.array(trajectories_cost)
  }


def format_data_batch_movielens(data_history,
                                gamma,
                                constant_baseline=0.0,
                                mask_already_recommended=False,
                                user_id_input=True,
                                **kwargs):
  """Format data for movielens RNN agent update step."""
  inp_rec_seq, inp_reward_seq, output_recs, reward_weights = [], [], [], []
  user_id_seq = []
  trajectories_cost = []
  if mask_already_recommended:
    # TODO(): Change argument to repeat_movies to be consistent.
    masks_for_softmax = []
  for user_id, curr_recs, curr_rewards, curr_safety_costs in zip(
      data_history['users'], data_history['recommendations'],
      data_history['rewards'], data_history['safety_costs']):
    inp_rec_seq.append(np.array(curr_recs[:-1]))
    inp_reward_seq.append(np.array(curr_rewards[:-1]))
    output_recs.append(np.expand_dims(np.array(curr_recs[1:]), axis=-1))
    output_rewards = accumulate_rewards(curr_rewards[1:] - constant_baseline,
                                        gamma)
    user_id_seq.append(user_id[:-1])
    reward_weights.append(output_rewards)
    cost_trajectory = np.mean(curr_safety_costs)
    trajectories_cost.append(cost_trajectory)
    masks_for_softmax.append(
        get_mask_for_softmax(curr_recs[1:-1], kwargs['action_space_size']))
  input_list = [
      np.array(inp_rec_seq),
      np.array(inp_reward_seq),
  ]
  if user_id_input:
    input_list.append(np.array(user_id_seq))

  if mask_already_recommended:
    input_list.append(np.array(masks_for_softmax))

  return {
      'input': input_list,
      'output': np.array(output_recs),
      'reward_weights': np.array(reward_weights),
      'trajectory_costs': np.array(trajectories_cost)
  }


def get_mask_for_softmax(current_recommendations, action_space_size):
  mask = np.ones((len(current_recommendations) + 1, action_space_size),
                 dtype=np.int)
  for i in range(len(current_recommendations)):
    mask[i+1, current_recommendations[:i+1]] = 0
  # TODO(): Add a test to test whether the mask works as expected.
  return mask


def load_model(filepath,
               optimizer_name,
               learning_rate=None,
               momentum=None,
               gradient_clip_value=None,
               gradient_clip_norm=None):
  """Loads RNNAgent model from the path."""
  # Since keras model.load requires a file path and filepath could be a CNS
  # directory, we will first copy the model into a tempfile and then read it
  # from there.
  tmp_model_file_path = os.path.join(tempfile.gettempdir(), 'tmp_model.h5')
  file_util.copy(filepath, tmp_model_file_path, overwrite=True)
  loaded_model = tf.keras.models.load_model(tmp_model_file_path)
  file_util.remove(tmp_model_file_path)
  optimizer = model.construct_optimizer(optimizer_name, learning_rate, momentum,
                                        gradient_clip_value, gradient_clip_norm)
  loaded_model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=optimizer,
      sample_weight_mode='temporal')
  return loaded_model
