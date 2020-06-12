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
"""Module for evaluating an RNN agent.

Defines functions evaluate_agent to run a simulation for a provided agent and
environment to calculate the average reward and safety costs for the agent.
"""

from absl import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


def violence_risk(observation):
  return observation['response'][0]['violence_score']


def health_risk(observation):
  return 1-observation['response'][0]['health_score']


def evaluate_agent(agent, env, alpha, num_users=100, deterministic=False,
                   scatter_plot_trajectories=False, figure_file_obj=None,
                   risk_score_extractor=violence_risk):
  """Runs an agent-env simulation to evaluate average reward and safety costs.

  Args:
    agent: rnn_cvar_agent.SafeRNNAgent object.
    env: Recsim environment that returns responses with reward and health score.
    alpha: The alpha used as the level for VaR/CVaR.
    num_users: Number of users to sample for the evaluation.
    deterministic: Whether the agent chooses the argmax action instead of
      sampling.
    scatter_plot_trajectories: Whether to evaluate
    figure_file_obj: File object to store the plot.
    risk_score_extractor: A function which takes an observation and returns a
      risk score.

  Returns:
    Dictionary with average reward, health score, cvar, var for num_users
    sampled.
  """
  results = {}
  if hasattr(env._environment, 'set_active_pool'):  # pylint: disable=protected-access
    pools = ['train', 'eval', 'test']
  else:
    pools = ['all']

  for pool in pools:
    tf.keras.backend.set_learning_phase(0)
    if hasattr(env._environment, 'set_active_pool'):  # pylint: disable=protected-access
      env._environment.set_active_pool(pool)  # pylint: disable=protected-access
    else:
      assert pool == 'all'

    rewards = []
    health = []
    ratings = []
    max_episode_length = agent.max_episode_length
    agent.epsilon = 0.0  # Turn off any exploration.
    # Set the learning phase to 0 i.e. evaluation to not use dropout.
    # Generate num_users trajectories.
    for _ in range(num_users):
      # TODO(): Clean the logged variables by making a data class.
      curr_user_reward = 0.0
      curr_user_health = 0.0
      curr_user_rating = 0.0

      reward = 0
      observation = env.reset()
      for _ in range(max_episode_length):
        slate = agent.step(reward, observation, eval_mode=True,
                           deterministic=deterministic)
        observation, reward, _, _ = env.step(slate)
        curr_user_reward += reward
        curr_user_health += 1-risk_score_extractor(observation)
        if 'rating' in observation['response'][0]:
          curr_user_rating += observation['response'][0]['rating']
      agent.end_episode(reward, observation, eval_mode=True)
      rewards.append(curr_user_reward/float(max_episode_length))
      health.append(curr_user_health/float(max_episode_length))
      ratings.append(curr_user_rating/float(max_episode_length))
    agent.empty_buffer()
    health_risks = 1-np.array(health)
    var = np.percentile(health_risks, 100*alpha)
    cvar = compute_cvar(health_risks, var)
    logging.info('Average Reward = %f, Average Health = %f, '
                 'Average Ratings = %f,VaR = %f, CVaR = %f',
                 np.mean(rewards), np.mean(health), np.mean(ratings), var, cvar)
    # Set the learning phase back to 1.
    tf.keras.backend.set_learning_phase(1)
    if scatter_plot_trajectories:
      plot_trajectories(ratings, health, figure_file_obj)
    results[pool] = {
        'rewards': np.mean(rewards),
        'health': np.mean(health),
        'ratings': np.mean(ratings),
        'var': var,
        'cvar': cvar
    }

  if len(results) == 1:  # No train/eval/test split, just return one value.
    return results['all']

  # Promote the eval results to the top-level dictionary.
  results.update(results['eval'])
  return results


def plot_trajectories(rewards, health, figure_file_obj):
  plt.figure()
  g = sns.jointplot(x=rewards, y=health, kind='kde')
  g.plot_joint(plt.scatter, c='grey', s=30, linewidth=1, marker='+')
  g.ax_joint.collections[0].set_alpha(0)
  g.set_axis_labels('$Reward$', '$Health$')
  if figure_file_obj:
    plt.savefig(figure_file_obj, format='png')
  else:
    plt.show()


def compute_cvar(health_risks, var):
  """Returns CVaR for the provided health_risks array."""
  return np.mean([risk for risk in health_risks if risk >= var])
