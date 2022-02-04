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
"""Implements a function to return a keras model.

The function create_model returns a keras.Model that is an RNN that outputs a
softmax distribution for each input of past recommendation and corresponding
reward while maintaining a hidden state.
"""
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS


def multihot(indices, n):
  vec = np.zeros((1, n))
  for idx in indices:
    vec[0, idx] = 1
  return vec


def create_model(max_episode_length,
                 action_space_size,
                 embedding_size,
                 hidden_size,
                 learning_rate=None,
                 batch_size=None,
                 optimizer_name='Adam', user_id_input=False,
                 genre_vector_input=False,
                 num_users=None, user_embedding_size=None,
                 user_feats_input_size=0, momentum=None,
                 num_hidden_layers=1,
                 gradient_clip_norm=None,
                 gradient_clip_value=None,
                 repeat_recs_in_episode=True,
                 genre_vec_size=0,
                 regularization_coeff=0.0,
                 activity_regularization=0.0,
                 dropout=0.0,
                 stateful=False):
  """Returns an RNN model for sequential recommendation.

  Currently uses an Adam optimizer.

  Args:
    max_episode_length: Maximum length of the user trajectory. This will be the
      maximum input sequence to the RNN agent.
    action_space_size: Size of the recommendation action space to choose from.
    embedding_size: Size of the embedding for each recommendation at the input.
    hidden_size: Number of hidden nodes in the LSTM.
    learning_rate: Learning Rate for the optimizer.
    batch_size: Batch size while training.
    optimizer_name: Name of the optimizer (current choices: Adam, SGD, Adagrad).
    user_id_input: Whether user_id is input into the model or not
      (default: False).
    genre_vector_input: True if the model should expect a genre vector to be
      passed as an input.
    num_users: Number of users for the size of the user's embedding (default:0).
    user_embedding_size: Size of the user embedding
      (only used when user_id_input=True).
    user_feats_input_size: Size of the user features input to the model
      (default=0).
    momentum: Momentum for SGD optimizer (Note: only works with SGD).
    num_hidden_layers: Number of hidden layers in the model. This is the number
      of layers between LSTM and the softmax.
    gradient_clip_norm: Clip the norm of the gradient to this value.
    gradient_clip_value: Clip the value of the gradient to lie in [-val, val].
    repeat_recs_in_episode: Whether the agent is allowed to repeat
      recommendations.
    genre_vec_size: Size of the genre input vector. Only used when
      genre_vector_input is True.
    regularization_coeff: L2 regularization coefficient for all the layers.
    activity_regularization: Activity regularization on the softmax output.
    dropout: Dropout for the dense layers.
    stateful: If True, set the LSTM layer to stateful.
  """
  tf.disable_eager_execution()
  tf.experimental.output_all_intermediates(True)
  regularizer_obj = tf.keras.regularizers.l2(regularization_coeff)
  activity_regularizer_obj = tf.keras.regularizers.l2(activity_regularization)

  if num_hidden_layers < 1:
    raise ValueError(
        'There should be at least one hidden layer, used {}.'.format(
            num_hidden_layers))

  # Two inputs: Previous recommendations consumed by the user and corresponding
  # rewards.
  rec_input = tf.keras.layers.Input(
      batch_shape=(batch_size, max_episode_length), name='consumed_previous')
  reward_input = tf.keras.layers.Input(
      batch_shape=(batch_size, max_episode_length), name='reward_previous')

  if user_id_input:
    user_id_input_layer = tf.keras.layers.Input(
        batch_shape=(batch_size, max_episode_length), name='user_id')
    user_embeddings = tf.keras.layers.Embedding(
        # Embedding space has one additional tokens for padding_token.
        input_dim=num_users + 1,
        output_dim=user_embedding_size,
        mask_zero=False,
        name='user_embedding_layer',
        embeddings_regularizer=regularizer_obj)(user_id_input_layer)
  if user_feats_input_size > 0:
    user_feats_input = tf.keras.layers.Input(
        batch_shape=(batch_size, user_feats_input_size,
                     max_episode_length), name='user_feats_input')

  rec_embeddings = tf.keras.layers.Embedding(
      # Embedding space has two additional tokens: one for padding_token and
      # one for start_token.
      input_dim=action_space_size + 2,
      output_dim=embedding_size,
      mask_zero=False,
      name='recommendation_embedding_layer',
      embeddings_regularizer=regularizer_obj)(
          rec_input)
  if genre_vector_input:
    genre_vector_input_layer = tf.keras.layers.Input(
        batch_shape=(batch_size, genre_vec_size, max_episode_length),
        name='genre_input_consumed_previous')
    rec_embeddings = tf.keras.layers.Concatenate(
        axis=-1,
        name='concat_rec_genres')([genre_vector_input_layer, rec_embeddings])

  merged_embeddings = tf.keras.layers.Concatenate(
      axis=-1, name='concat_embedding_layer')(
          [rec_embeddings,
           tf.expand_dims(reward_input, -1)])
  if user_id_input:
    merged_embeddings = tf.keras.layers.Concatenate(
        axis=-1,
        name='concat_embedding_layer_3')([merged_embeddings, user_embeddings])

  if user_feats_input_size > 0:
    merged_embeddings = tf.keras.layers.Concatenate(
        axis=-1,
        name='concat_embedding_layer_2')([merged_embeddings, user_feats_input])
  hidden_layer = tf.keras.layers.LSTM(
      units=hidden_size, return_sequences=True, name='LSTM',
      kernel_regularizer=regularizer_obj, stateful=stateful)(
          merged_embeddings)

  for i in range(num_hidden_layers - 1):
    # The output layer to softmax is considered one hidden layer.
    hidden_layer = tf.keras.layers.Dense(
        units=hidden_size,
        activation='relu',
        kernel_regularizer=regularizer_obj,
        name='hidden_layer_{}'.format(i + 1))(
            hidden_layer)
    hidden_layer = tf.keras.layers.Dropout(dropout)(hidden_layer)

  output_layer = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Dense(
          units=action_space_size,
          activation='softmax',
          kernel_regularizer=regularizer_obj,
          activity_regularizer=activity_regularizer_obj),
      name='softmax_output')(
          hidden_layer)
  if dropout > 0:
    output_layer = tf.keras.layers.Dropout(dropout)(output_layer)
    output_layer = output_layer / tf.keras.backend.sum(
        output_layer, axis=-1, keepdims=True)

  if not repeat_recs_in_episode:
    # TODO(): Remove the need for this argument by providing mask
    # with all inputs.
    softmax_mask = tf.keras.layers.Input(
        batch_shape=(batch_size, max_episode_length, action_space_size),
        name='softmax_mask_input')
    masked_output = tf.keras.layers.multiply([softmax_mask, output_layer])
    # Renormalize
    output_layer = masked_output / tf.keras.backend.sum(
        masked_output, axis=-1, keepdims=True)

  # Setup what layers the model should expect in input arrays.
  inputs = [rec_input, reward_input]
  if user_id_input:
    inputs.append(user_id_input_layer)
  if user_feats_input_size > 0:
    inputs.append(user_feats_input)
  if not repeat_recs_in_episode:
    inputs.append(softmax_mask)

  model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)

  optimizer = construct_optimizer(
      optimizer_name,
      learning_rate=learning_rate,
      momentum=momentum,
      gradient_clip_value=gradient_clip_value,
      gradient_clip_norm=gradient_clip_norm)
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=optimizer,
      sample_weight_mode='temporal')
  return model


def construct_optimizer(optimizer_name,
                        learning_rate=None,
                        momentum=None,
                        gradient_clip_value=None,
                        gradient_clip_norm=None):
  """Returns an optimizer for the given optimizer_name or raises ValueError."""
  kwargs = {}
  if learning_rate:
    kwargs['learning_rate'] = learning_rate
  if gradient_clip_value:
    kwargs['clipvalue'] = gradient_clip_value
  if gradient_clip_norm:
    kwargs['clipnorm'] = gradient_clip_norm
  if optimizer_name == 'SGD' and momentum:
    kwargs['momentum'] = momentum
  if optimizer_name == 'Adam':
    optimizer = tf.keras.optimizers.Adam(**kwargs)
  elif optimizer_name == 'SGD':
    optimizer = tf.keras.optimizers.SGD(**kwargs)
  elif optimizer_name == 'Adagrad':
    optimizer = tf.keras.optimizers.Adagrad(**kwargs)
  else:
    raise ValueError(
        "Use optimizer_name as one out 'Adam', 'Adagrad' and 'SGD'.")
  return optimizer


def change_optimizer_lr(model, learning_rate):
  """Change a model's learning rate after it has been already set."""
  tf.keras.backend.set_value(model.optimizer.lr, learning_rate)
