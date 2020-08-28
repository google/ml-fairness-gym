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

Modifies evaluation.py/evaluate_agent to run a simulation for a provided agent and
environment to calculate the average reward and safety costs for the agent.
"""

from absl import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from collections import Counter
import sys


def violence_risk(observation):
    return observation['response'][0]['violence_score']


def health_risk(observation):
    return 1-observation['response'][0]['health_score']


def plot_recs_hists(recs_histogram, pool):
    plt.bar(np.arange(len(recs_histogram))+1,
            sorted(recs_histogram.values(), reverse=True))

    print("Most common 10 recommendations are:",
          recs_histogram.most_common(10))
    plt.ylabel('Freq of rec')
    plt.xlabel('Movie index (sorted by frequency of recommendation)')
    plt.title('Recommendation frequency {}.'.format(pool))
    plt.show()


def evaluate_agent(agent, env, alpha, num_users=100, deterministic=False,
                   softmax_temperature=1.0,
                   scatter_plot_trajectories=False, figure_file_obj=None,
                   risk_score_extractor=violence_risk, plot_histogram=False,
                   plot_trajectories=True,
                   stepwise_plot=False, only_evaluate_pool=None,
                   reward_health_distribution_plot=False, debug_log=False):
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
      plot_histogram: Plots the histogram of recommendation frequency of movies.
      plot_trajectories: Plots trajectories of all the users.
      stepwise_plot: Plots the average rating and health by each step in the trajectory. 
      only_evaluate_pool: Specify the name of the pool (str) that you want to evaluate.
      reward_health_distribution: Plots a reward (x-axis) vs health (y-axis) plot for each
        user in the pool.
      debug_log: returns a list of tuples of the form (user_id, ((movie_id, rating, health), ...))

    Returns:
      Dictionary with average reward, health score, cvar, var for num_users
      sampled.
    """
    results = {}
    if hasattr(env._environment, 'set_active_pool'):  # pylint: disable=protected-access
        pools = ['train', 'eval', 'test']
        if only_evaluate_pool:
            pools = [only_evaluate_pool]
    else:
        pools = ['all']

    for pool in pools:
        tf.keras.backend.set_learning_phase(0)
        if hasattr(env._environment._user_model._user_sampler, 'set_active_pool'):  # pylint: disable=protected-access
            env._environment.set_active_pool(
                pool)  # pylint: disable=protected-access
        else:
            assert pool == 'all' or only_evaluate_pool
        if plot_histogram or plot_trajectories:
            recs_histogram = Counter({})
            recs_histogram_keys_list = {}
        if debug_log:
            user_rec_log = []
        ratings = []
        ratings_health_user_map = {}
        health = []
        rewards = []
        max_episode_length = agent.max_episode_length
        if stepwise_plot:
            stepwise_ratings = [[] for _ in range(max_episode_length)]
            stepwise_healths = [[] for _ in range(max_episode_length)]

        agent.epsilon = 0.0  # Turn off any exploration.
        env._environment._user_model._user_sampler.reset_sampler()
        # Set the learning phase to 0 i.e. evaluation to not use dropout.
        # Generate num_users trajectories.
        for _ in range(num_users):
            # TODO(): Clean the logged variables by making a data class.
            curr_user_reward = 0.0
            curr_user_health = 0.0
            curr_user_rating = 0.0
            if plot_histogram or plot_trajectories:
                current_trajectory = []
            reward = 0
            observation = env.reset()
            curr_user_vector = env.environment.user_model._user_state.topic_affinity
            user_id = observation['user']['user_id']
            if debug_log:
                user_rec_log.append((user_id, []))
            for step_number in range(max_episode_length):
                slate = agent.step(reward, observation, eval_mode=True,
                                   deterministic=deterministic, temperature=softmax_temperature)
                observation, reward, _, _ = env.step(slate)
                rating = observation['response'][0]['rating']
                if plot_histogram or plot_trajectories:
                    current_trajectory.append(slate[0])
                    if slate[0] in recs_histogram:
                        recs_histogram[slate[0]] = recs_histogram[slate[0]] + 1
                    else:
                        recs_histogram[slate[0]] = 1
                        recs_histogram_keys_list[slate[0]] = len(
                            recs_histogram.keys())
                if stepwise_plot:
                    # print(reward, risk_score_extractor(observation))
                    stepwise_ratings[step_number].append(rating)
                    stepwise_healths[step_number].append(
                        1-risk_score_extractor(observation))
                curr_user_rating += rating
                curr_user_reward += reward
                curr_user_health += 1-risk_score_extractor(observation)
                if debug_log:
                    user_rec_log[-1][1].append((slate[0], rating, 1-risk_score_extractor(observation), reward))
            agent.end_episode(reward, observation, eval_mode=True)
            ratings.append(curr_user_rating/float(max_episode_length))
            health.append(curr_user_health/float(max_episode_length))
            ratings_health_user_map[str(curr_user_vector)] = (ratings[-1], health[-1])
            rewards.append(curr_user_reward/float(max_episode_length))
            if plot_trajectories:
                plot_current_trajectory(
                    current_trajectory, observation, recs_histogram_keys_list)
        plt.show()
        agent.empty_buffer()
        health_risks = 1-np.array(health)
        var = np.percentile(health_risks, 100*alpha)
        cvar = compute_cvar(health_risks, var)
        logging.info('Average Reward = %f, Average Health = %f, '
                     'Average Ratings = %f,VaR = %f, CVaR = %f',
                     np.mean(rewards), np.mean(health), np.mean(ratings), var, cvar)
        if plot_histogram:
            plot_recs_hists(recs_histogram, pool)
            plt.show()
        if stepwise_plot:
            plot_stepwise_ratings(stepwise_ratings, stepwise_healths)
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
        if plot_histogram:
            results[pool]['unique_recs'] = len(recs_histogram.keys())
        if reward_health_distribution_plot:
            results[pool]['ratings_health_user_map'] = ratings_health_user_map
            plot_reward_vs_health_distribution(ratings, health)
        if debug_log:
            save_user_rec_log(user_rec_log)
            results[pool]['user_rec_log'] = user_rec_log

    if len(results) == 1:  # No train/eval/test split, just return one value.
        return results[only_evaluate_pool] if only_evaluate_pool else results['all']

    # Promote the eval results to the top-level dictionary.
    results.update(results['eval'])
    return results

def save_user_rec_log(user_rec_log, directory='saved_models/evaluations/'):
  os.makedirs(directory, exist_ok=True)
  pkl.dump(user_rec_log, directory)

def plot_current_trajectory(current_trajectory, observation, recs_histogram_keys_list):
    if len(np.unique(current_trajectory)) != len(current_trajectory):
        raise ValueError(
            'Non-unique recommendations found for user %s.' % observation['user']['user_id'])
    plt.plot([recs_histogram_keys_list[key] for key in current_trajectory],
             label=str(observation['user']['user_id']), marker='.')
    plt.xlabel('Steps')
    plt.ylabel('Document Id')


def plot_stepwise_ratings(stepwise_ratings, stepwise_healths):
    stepwise_reward_means = [np.mean(rews)
                             for rews in stepwise_ratings]
    stepwise_health_means = [np.mean(rews)
                             for rews in stepwise_healths]
    _, axs = plt.subplots(1, 2)
    axs[0].plot(stepwise_reward_means, label='Reward Mean')
    axs[1].plot(stepwise_health_means, label='Health Mean')
    axs[0].set_xlabel('Steps')
    axs[1].set_xlabel('Steps')
    axs[0].legend()
    axs[1].legend()
    plt.show()
    stepwise_ratings_per_user = [[stepwise_ratings[i][user_num] for i in range(
        len(stepwise_ratings))] for user_num in range(len(stepwise_ratings[0]))]
    plt.plot(np.array(stepwise_ratings_per_user).transpose())
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.show()


def plot_trajectories(ratings, health, figure_file_obj):
    plt.figure()
    g = sns.jointplot(x=ratings, y=health, kind='kde')
    g.plot_joint(plt.scatter, c='grey', s=30, linewidth=1, marker='+')
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels('$Rating$', '$Health$')
    if figure_file_obj:
        plt.savefig(figure_file_obj, format='png')
    else:
        plt.show()


def plot_reward_vs_health_distribution(average_ratings, average_healths):
    h = sns.jointplot(x=average_healths, y=average_ratings)
    h.set_axis_labels('Health', 'Ratings', fontsize=16)
    plt.tight_layout()
    plt.show()


def compute_cvar(health_risks, var):
    """Returns CVaR for the provided health_risks array."""
    return np.mean([risk for risk in health_risks if risk >= var])
