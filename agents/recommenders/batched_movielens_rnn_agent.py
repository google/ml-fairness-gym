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
"""Extends the RNNAgent for the movielens setup.
"""
import collections
from absl import logging
import attr
from agents.recommenders import model
from agents.recommenders import rnn_cvar_agent
from agents.recommenders import utils
import numpy as np
import tensorflow as tf


@attr.s
class Sequence(object):
  """Data class to hold model inputs MovieLens RNN."""
  vocab_size = attr.ib()
  mask_previous_recs = attr.ib()
  start_token = attr.ib()
  recommendations = attr.ib(factory=lambda: collections.defaultdict(list))
  rewards = attr.ib(factory=lambda: collections.defaultdict(list))
  safety_cost = attr.ib(factory=lambda: collections.defaultdict(list))
  masks = attr.ib(factory=dict)

  @property
  def batch_size(self):
    assert (len(self.recommendations) == len(self.rewards) == len(
        self.safety_cost))
    return len(self.recommendations)

  def update(self,
             last_recommendation,
             reward,
             observation,
             batch_position=0):
    """Updates the model input with the latest step information."""
    if last_recommendation is None:
      last_recommendation = self.start_token
    uid = (batch_position, observation['user']['user_id'])
    self.recommendations[uid].append(last_recommendation)
    self.rewards[uid].append(reward)
    if uid not in self.masks:
      self.masks[uid] = np.ones(self.vocab_size)
    if last_recommendation < self.vocab_size:  # Ignore OOV.
      self.masks[uid][last_recommendation] = 0

    if observation['response'] is None:
      self.safety_cost[uid].append(None)
    else:
      # Assume single recommendation/response setting.
      assert len(observation['response']) == 1
      for response in observation['response']:
        self.safety_cost[uid].append(response['violence_score'])

  def batch_update(self, last_recommendations, rewards, observations):
    """Perform an update from a batch of observations and rewards."""
    batch_size_before_update = self.batch_size
    is_first_update = (batch_size_before_update == 0)

    assert is_first_update == (last_recommendations is None)

    # Special case: If this is the first update, use a start token for last
    # recommendations.
    if is_first_update:
      last_recommendations = [self.start_token for _ in rewards]

    if not len(last_recommendations) == len(rewards) == len(observations):
      raise ValueError(
          'Lengths must match for a batch update. %d %d %d' %
          (len(last_recommendations), len(rewards), len(observations)))
    for idx, rec in enumerate(last_recommendations):
      self.update(rec, rewards[idx], observations[idx], batch_position=idx)

    assert is_first_update or (self.batch_size == batch_size_before_update), (
        'Batch size changed after an update. Either new users were '
        'introduced or the order of the batch was changed!')

  def build_prediction_input(self, input_args):
    input_dict = self.as_dict()
    return [input_dict[key] for key in input_args]

  def as_dict(self):
    """Returns a dict of sequence data.

     Keys of the dict are:
       - users: (batch, len_sequence)
       - recommendations: (batch, len_sequence)
       - rewards: (batch, len_sequence)
       - safety_costs: (batch, 1, 1)
       - masks: (batch, len_sequence, vocab_size)
       - final_mask: (batch, 1, vocab_size)

    `final_mask` is the mask corresponding to the end of the sequence. This is
    what is used to mask the predictions for the next round.

    A batch is made up of multiple users, where each row of the batch
    corresponds to a different user.
    """
    batch = []
    # The primary key of this sort is the batch_position which is the first
    # element of uid. This way, the batch retains the order that it was received
    # as input.
    for uid, rewards in sorted(self.rewards.items()):
      pos, user_id = uid
      user_vec = [user_id] * len(rewards)
      mask = self.masks[uid]
      recs = self.recommendations[uid]
      costs = np.mean(self.safety_cost[uid][1:])
      batch.append((pos, user_vec, recs, rewards, mask, costs))
    positions, users, recommendations, rewards, masks, costs = zip(*batch)

    assert (list(positions) == sorted(range(
        len(users)))), 'Positions should be %s. Got %s' % (sorted(
            range(len(users))), list(positions))

    return {
        'users': users,
        'recommendations': recommendations,
        'rewards': rewards,
        'safety_costs': np.expand_dims(costs, -1),
        'final_mask': np.expand_dims(masks, 1)
    }


class ReplayBuffer(object):
  """Data class to an agent's replay buffer."""

  KEYS = [
      'recommendations',
      'rewards',
      'safety_costs',
      'users',
      'final_mask',
  ]

  # Parent code may have slightly different names for fields.
  # Aliases help translate one to the other.
  ALIASES = {'reward_seqs': 'rewards'}

  def __init__(self):
    self._buffer = {key: [] for key in self.KEYS}

  def append(self, update_dict):
    if set(update_dict.keys()) != set(self._buffer.keys()):
      raise ValueError('Key mismatch! Expected keys %s, got %s' %
                       (set(self._buffer.keys()), set(update_dict.keys())))
    for key, value in update_dict.items():
      self._buffer[key].append(value)

  def __getitem__(self, key):
    if key in self.ALIASES:
      key = self.ALIASES[key]
    return np.vstack(self._buffer[key])


class MovieLensRNNAgent(rnn_cvar_agent.SafeRNNAgent):
  """Defines an RNN agent for movielens setup with safety."""

  def __init__(self, observation_space, action_space, max_episode_length=None,
               embedding_size=32,
               hidden_size=32,
               optimizer_name='Adam',
               gamma=0.99, epsilon=0.0,
               replay_buffer_size=100,
               initial_lambda=0.0,
               alpha=0.95,
               beta=0.3,
               max_cost=1.0,
               min_cost=0.0, user_embedding_size=32, constant_baseline=3.0,
               learning_rate=None,
               gradient_clip_norm=1.0, gradient_clip_value=None, momentum=0.9,
               num_hidden_layers=1,
               repeat_movies_in_episode=False,
               load_from_checkpoint=None, genre_vec_as_input=False,
               genre_vec_size=0, regularization_coeff=0.0,
               activity_regularization=0.0,
               dropout=0.0, user_id_input=True,
               random_seed=None,
               stateful=False,
               batch_size=None):
    self.user_embedding_size = user_embedding_size
    self.num_users = observation_space['user']['user_id'].n
    self.padding_user_id_token = self.num_users
    self.constant_baseline = constant_baseline
    self.learning_rate = learning_rate
    self.gradient_clip_norm = gradient_clip_norm
    self.gradient_clip_value = gradient_clip_value
    self.momentum = momentum
    self.repeat_movies_in_episode = repeat_movies_in_episode
    self.genre_vec_as_input = genre_vec_as_input
    self.genre_vec_size = genre_vec_size  # Used when genre_vec_as_input is True
    self.num_hidden_layers = num_hidden_layers
    self.activity_regularization = activity_regularization
    self.dropout = dropout
    self.user_id_input = user_id_input
    self.stateful = stateful
    self.batch_size = batch_size
    self._last_rec = None

    super(MovieLensRNNAgent, self).__init__(
        observation_space,
        action_space,
        max_episode_length,
        embedding_size,
        hidden_size,
        optimizer_name,
        gamma,
        epsilon,
        replay_buffer_size,
        initial_lambda,
        alpha,
        beta,
        max_cost,
        min_cost,
        load_from_checkpoint=load_from_checkpoint,
        regularization_coeff=regularization_coeff,
        random_seed=random_seed)
    self._last_rec = None
    self._sequence = Sequence(
        self.action_space_size,
        mask_previous_recs=not repeat_movies_in_episode,
        start_token=self.start_token)
    if (not repeat_movies_in_episode and
        self.max_episode_length is not None and
        self.action_space_size <= self.max_episode_length):
      raise ValueError('The agent is set to not repeat recommendations in an '
                       'episode, but the action space size ({}) is smaller '
                       'than the length of the episode ({})'.format(
                           self.action_space_size, self.max_episode_length))
    # If the model was loaded from a checkpoint, it can be useful to set the
    # batch size again explicitly, just in case.
    self.set_batch_size(self.batch_size)
    tf.keras.backend.set_learning_phase(1)

  def build_model(self):
    self.model = model.create_model(
        max_episode_length=None,  # Allows for variable-length sequence inputs.
        action_space_size=self.action_space_size,
        embedding_size=self.embedding_size,
        hidden_size=self.hidden_size,
        learning_rate=self.learning_rate,
        batch_size=self.batch_size,
        optimizer_name=self.optimizer_name,
        user_id_input=self.user_id_input,
        num_users=self.num_users,
        user_embedding_size=self.user_embedding_size,
        gradient_clip_norm=self.gradient_clip_norm,
        gradient_clip_value=self.gradient_clip_value,
        momentum=self.momentum,
        repeat_recs_in_episode=self.repeat_movies_in_episode,
        genre_vector_input=self.genre_vec_as_input,
        genre_vec_size=self.genre_vec_size,
        regularization_coeff=self.regularization_coeff,
        activity_regularization=self.activity_regularization,
        dropout=self.dropout,
        num_hidden_layers=self.num_hidden_layers,
        stateful=self.stateful)
    tf.keras.backend.set_learning_phase(1)

  def set_batch_size(self, new_batch_size):
    logging.info('Setting batch size')
    self.batch_size = new_batch_size
    # Store the weights from the previous model.
    weights = self.model.get_weights()
    # Clear out the old model to avoid build-up of unused graph components.
    # This will fail when run inside a with graph.as_default() context.
    del self.model
    try:
      tf.keras.backend.clear_session()
    except AssertionError as e:
      raise AssertionError(
          'Keras backend clear_session() cannot be run within a '
          'graph.as_default() context. Are you sure you need to be running in '
          'that context?'
      ) from e
    # Build a new model with the new batch size.
    self.build_model()
    # Copy over the weights.
    self.model.set_weights(weights)
    logging.info('Done setting batch size')

  def step(self, reward, observation, eval_mode=False, deterministic=False):
    """Recommend the next slate."""
    self._update_sequence(self._last_rec, reward, observation)
    softmax_probs = self.get_model_prediction()
    self._last_rec = self._choose_rec_from_softmax(softmax_probs, deterministic)
    return np.expand_dims(self._last_rec, -1)

  def empty_buffer(self):
    """Clears the history stored by the agent."""
    self.replay_buffer = ReplayBuffer()

  def reset_current_episode_logs(self):
    """Clears the current episode's data."""
    self._sequence = Sequence(self.action_space_size,
                              self.repeat_movies_in_episode,
                              self.start_token)
    self._last_rec = None

  def get_model_prediction(self):
    """Returns the Softmax layer for the last time step."""
    input_args = ['recommendations', 'rewards']

    if self.user_id_input:
      input_args.append('users')

    # TODO() Remove this once all models can accept masks.
    if not self.repeat_movies_in_episode:
      input_args.append('final_mask')

    input_to_model = self._sequence.build_prediction_input(input_args)

    # If stateful, just use the _last_ timestep.
    if self.stateful:
      input_to_model = [np.array(input_)[:, -1:] for input_ in input_to_model]

    softmax_all_layers = self.model.predict(input_to_model)

    # Get the prediction for the last step.
    return softmax_all_layers[:, -1, :]

  def end_episode(self, reward, observation, eval_mode=False):
    """Stores the last reward, updates the model. No recommendation returned."""
    self._update_sequence(self._last_rec, reward, observation)
    self.replay_buffer.append(self._sequence.as_dict())
    if not eval_mode:
      self.model_update()
      self.empty_buffer()  # Empty the buffer after updating the model.
    self.reset_current_episode_logs()
    self.model.reset_states()

  def change_model_lr(self, learning_rate):
    """Changes the model's learning rate."""
    model.change_optimizer_lr(self.model, learning_rate)

  def _update_sequence(self, last_rec, reward, observation):
    if isinstance(observation, (list, tuple)):
      self._sequence.batch_update(last_rec, reward, observation)
    else:
      self._sequence.update(last_rec, reward, observation)

  def _update_params(self):
    """This function is called by self.model_update() for the gradient step."""
    formatted_data = utils.format_data_batch_movielens(
        self.replay_buffer,
        self.gamma,
        self.constant_baseline,
        mask_already_recommended=not self.repeat_movies_in_episode,
        action_space_size=self.action_space_size,
        genre_vec_input=self.genre_vec_as_input,
        genre_vec_size=self.genre_vec_size,
        user_id_input=self.user_id_input)
    training_history = self.model.train_on_batch(
        formatted_data['input'],
        formatted_data['output'],
        sample_weight=self._calculate_weights_for_reinforce(
            formatted_data['reward_weights'],
            formatted_data['trajectory_costs']))
    if self.stateful:
      self.model.reset_states()
    return training_history

  def _choose_rec_from_softmax(self, softmax_probs, deterministic):
    """Batched version of _choose_rec_from_softmax."""
    recommendations = []
    for p_vec in softmax_probs:
      recommendations.append(
          super(MovieLensRNNAgent,
                self)._choose_rec_from_softmax(p_vec, deterministic))

    # If not a multi-user batch, demote recommendations into a scalar.
    if len(recommendations) == 1:
      recommendations = recommendations[0]
    return recommendations
