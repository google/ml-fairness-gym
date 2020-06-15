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
"""Run a CVaR-constrained Safe RL learning experiments."""

import copy
import hashlib
import inspect
import os
import tempfile
import types
from absl import flags
from absl import logging
import attr
import core as fg_core
import file_util
from agents.recommenders import batched_movielens_rnn_agent
from agents.recommenders import evaluation
from environments.recommenders import movie_lens_dynamic
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
flags.DEFINE_integer('ep_length', 10, 'Maximum Episode Length.')
flags.DEFINE_float('train_ratio', 0.7, '')
flags.DEFINE_float('eval_ratio', 0.15, '')
flags.DEFINE_float('initial_lambda', 0.0, 'Initial Value of Lambda.')
flags.DEFINE_float('beta', 0.5, 'Beta: Upper bound on CVaR.')
flags.DEFINE_float('alpha', 0.95,
                   'Alpha: Percentile risk for definition of VaR.')
flags.DEFINE_float('lambda_lr', 0.0, 'Learning Rate for Lambda.')
flags.DEFINE_float('var_lr', 0.0, 'Learning Rate for VaR.')
flags.DEFINE_float('lr', 0.0001, 'Learning Rate for the model parameters.')
flags.DEFINE_float('gamma', 0.1, 'Gamma for reward accumulation.')
flags.DEFINE_float('baseline', 3, 'Baseline value for variance reduction.')
flags.DEFINE_integer(
    'num_users_eval', 100, 'Number of users sampled to evaluate'
    'during the training.')
flags.DEFINE_integer('num_users_eval_final', 1000,
                     'Number of users sampled to evaluate the final agent.')
flags.DEFINE_integer('eval_every', 100,
                     'Evaluate the agent after these many steps.')
flags.DEFINE_integer('num_ep_per_update', 64,
                     'Number of episodes to generate for each training step.')
flags.DEFINE_enum('optimizer_name', 'Adam', ['Adam', 'SGD', 'Adagrad'],
                  'Name of the optimizer (choices: Adam, SGD)')
flags.DEFINE_integer('num_updates', 25000,
                     'Number of gradient updates for the model.')
flags.DEFINE_float('clipnorm', None, 'Norm to clip the gradient to.')
flags.DEFINE_float('clipvalue', None, 'Value to clip the gradient to.')
flags.DEFINE_float('momentum', None, 'Momentum for SGD optimizer.')
flags.DEFINE_integer('embedding_size', 32, 'Size of the item embedding.')
flags.DEFINE_integer('user_embedding_size', 0, 'Size of the user embedding.')
flags.DEFINE_integer('hidden_size', 128, 'Size of the LSTM hidden layer.')
flags.DEFINE_float('topic_affinity_update_threshold', 3.0,
                   'Topic affinity update threshold.')
flags.DEFINE_float('affinity_update_delta', 0.5,
                   'Topic affinitiy update delta.')
flags.DEFINE_integer(
    'checkpoint_every', 1000, 'Number of iterations after'
    'which the model is checkpointed.')
flags.DEFINE_float(
    'multiobjective_lambda', 0.0,
    'Weight to the health score in the optimized reward.'
    'Reward(trajectory) = (1-lambda)*avg_rating + lambda*health_score')
flags.DEFINE_boolean(
    'eval_deterministic', False,
    'Whether to evaluate the model using a deterministic '
    '(argmax) policy instead of sampling from Softmax')

flags.DEFINE_integer('num_hidden_layers', 1,
                     'Number of hidden layers in the agent model.')
flags.DEFINE_string('expt_name_suffix', None,
                    'Suffix appended to the experiment name.')
flags.DEFINE_float(
    'regularization_coeff', 0.0,
    'L2 regularization coefficient for all layers in RNN agent.')
flags.DEFINE_float(
    'activity_reg', 0.0,
    'Activity regularization coefficient for softmax layers in RNN agent.')
flags.DEFINE_float('dropout', 0.3,
                   'Dropout for dense layers in RNN agent during training.')
flags.DEFINE_string('initial_model', None,
                    'Path for the initial model file for the agent.')
flags.DEFINE_boolean('stateful_rnn', True,
                     'Whether to use a stateful RNN in the agent.')

DEFAULT_EMBEDDING_PATH = None
DEFAULT_OUTPUT_DIRECTORY = None
DEFAULT_DATA_DIRECTORY = None


flags.DEFINE_string(
    'embedding_path', DEFAULT_EMBEDDING_PATH,
    'Path to store user and movie embeddings as a pickle or json file.')

flags.DEFINE_string('results_dir', DEFAULT_OUTPUT_DIRECTORY,
                    'Results directory.')

flags.DEFINE_string('movielens_data_directory', DEFAULT_DATA_DIRECTORY,
                    'Directory containing movielens data.')


tf.enable_eager_execution()

# Mapping from config items to arguments in agent constructors.
CONFIG_NAME_TO_AGENT_ARG = {
    'baseline_value': 'constant_baseline',
    'activity_regularization_coeff': 'activity_regularization',
    'agent_seed': 'random_seed',
    'num_episodes_per_update': 'batch_size',
    'stateful_rnn': 'stateful',
}


def _rename_keys(my_dict, key_mapper):
  for old_name, new_name in key_mapper.items():
    my_dict[new_name] = my_dict.pop(old_name)
  return my_dict


@attr.s
class LearningRateConfig(object):
  theta = attr.ib()
  var = attr.ib()
  lambda_ = attr.ib()


@attr.s
class WarmStartConfig(object):
  initial_batch = attr.ib()
  filename = attr.ib()


def _warm_start(experiment_name, results_dir):
  """Infers the number of batches that have already been checkpointed to resume from there."""
  checkpointed_models_batch_numbers = []
  filenames = []
  filename_pattern = os.path.join(results_dir,
                                  'agent_model_' + experiment_name + '_*.h5')
  logging.info('Looking for %s', filename_pattern)
  for filename in file_util.glob(filename_pattern):
    logging.info('Found %s', filename)
    # check if the file is not empty
    checkpointed_models_batch_numbers.append(
        int(filename.split('_')[-1].split('.')[0]))
    filenames.append(filename)
  if not checkpointed_models_batch_numbers:
    logging.info('No checkpoint found in the result directory.')
    return WarmStartConfig(initial_batch=0, filename=None)
  max_batch_number = max(checkpointed_models_batch_numbers)
  fname = filenames[checkpointed_models_batch_numbers.index(max_batch_number)]
  logging.info('Checkpoint found for batch number %d. %s', max_batch_number,
               fname)
  return WarmStartConfig(initial_batch=max_batch_number, filename=fname)


def _envs_builder(config, num_envs):
  """Returns a list of environments."""
  # Make the first environment.
  envs = [movie_lens_dynamic.create_gym_environment(config['env_config'])]

  # All subsequent environments are copies with different user sampler seeds.
  for _ in range(1, num_envs):
    logging.info('Build env')
    envs.append(copy.deepcopy(envs[0]))
    # Unseed the envirnment user samplers. Go crazy!
    envs[-1]._environment._user_model._user_sampler._seed = None  # pylint: disable=protected-access
    envs[-1]._environment._user_model.reset_sampler()  # pylint: disable=protected-access
  return envs


def _agent_builder(env, config, agent_ctor=None):
  """Returns a fully configured agent."""
  if agent_ctor is None:
    agent_ctor = batched_movielens_rnn_agent.MovieLensRNNAgent

  config.warm_start = _warm_start(config.experiment_name, config.results_dir)

  # Using vars builtin to convert SimpleNamespace config to dict.
  config_args = _rename_keys(
      copy.deepcopy(vars(config)), CONFIG_NAME_TO_AGENT_ARG)

  config_args.update({'load_from_checkpoint': config.warm_start.filename})

  # Filter out config values that are not agent constructor arguments.
  ctor_args = set(inspect.getfullargspec(agent_ctor).args)
  for key in list(config_args):
    if key not in ctor_args:
      del config_args[key]

  return agent_ctor(env.observation_space, env.action_space, **config_args)


def _run_one_parallel_batch(envs, agent, config):
  """Simulate one batch of training interactions in parallel."""
  rewards = [0 for _ in envs]
  observations = [env.reset() for env in envs]
  for _ in range(config.max_episode_length):
    logging.debug('starting agent step')
    slates = agent.step(rewards, observations)
    logging.debug('starting envs step')
    observations, rewards, _, _ = zip(
        *[env.step(slate) for slate, env in zip(slates, envs)])
    logging.debug('done envs step')
    assert (len({obs['user']['user_id'] for obs in observations}) > 1 or
            len(observations) == 1
           ), 'In a parallel batch there should be many different users!'
  agent.end_episode(rewards, observations, eval_mode=True)


def _get_learning_rate(batch, config):
  del batch  # Unused.
  return LearningRateConfig(
      theta=config.learning_rate,
      var=config.var_learning_rate,
      lambda_=config.lambda_learning_rate)


def _update_model(batch_number, agent, config):
  learning_rates = _get_learning_rate(batch_number, config)
  train_loss_val, _, _ = agent.model_update(
      learning_rate=learning_rates.theta,
      lambda_learning_rate=learning_rates.lambda_,
      var_learning_rate=learning_rates.var)
  agent.empty_buffer()
  if batch_number % 100 == 0:
    logging.info('Batch: %d, Training loss:%f', batch_number, train_loss_val)


def _maybe_checkpoint(batch_number, agent, config):
  """Checkpoints the model into the specified directory."""
  if batch_number % config.checkpoint_every:
    return None

  tmp_model_file_path = os.path.join(tempfile.gettempdir(), 'tmp_model.h5')
  agent.model.save(tmp_model_file_path)
  model_file_path = os.path.join(
      config.results_dir,
      f'agent_model_{config.experiment_name}_{batch_number}.h5')
  file_util.copy(tmp_model_file_path, model_file_path, overwrite=True)
  logging.info('Model saved at %s', model_file_path)
  file_util.remove(tmp_model_file_path)
  return model_file_path


def _training_loop(env, agent, config):
  """Runs training and returns most recent checkpoint."""
  for env_ in env:
    env_._environment.set_active_pool('train')  # pylint: disable=protected-access
  batch_number = config.warm_start.initial_batch
  last_checkpoint = None
  while batch_number < config.num_updates:
    batch_number += 1
    _run_one_parallel_batch(env, agent, config)
    _update_model(batch_number, agent, config)
    checkpoint = _maybe_checkpoint(batch_number, agent, config)
    if checkpoint:
      last_checkpoint = checkpoint
    if batch_number % config.eval_every == 0:
      break

  config.warm_start.initial_batch = batch_number  # Restart here next time.
  return last_checkpoint


def _setup_directories(config):
  file_util.makedirs(config['results_dir'])
  with file_util.open(
      os.path.join(config['results_dir'],
                   config['experiment_name'] + '_info.txt'), 'w') as outfile:
    outfile.write(fg_core.to_json(config))


def train(config):
  """Trains and returns an Safe RNN agent."""
  _set_experiment_name(config)
  logging.info('Launching experiment id: %s', config['experiment_name'])
  _setup_directories(config)
  envs = _envs_builder(config, config['num_episodes_per_update'])
  config = types.SimpleNamespace(**config)
  agent = _agent_builder(envs[0], config)

  while config.warm_start.initial_batch < config.num_updates:
    last_checkpoint = _training_loop(envs, agent, config)
    agent.set_batch_size(1)
    metrics = evaluation.evaluate_agent(
        agent,
        envs[0],
        alpha=config.alpha,
        num_users=config.num_users_eval,
        deterministic=config.eval_deterministic)
    agent.set_batch_size(len(envs))
    step = config.warm_start.initial_batch * config.num_episodes_per_update
    yield step, last_checkpoint, metrics

  # Do one final eval at the end.
  agent.set_batch_size(1)
  metrics = evaluation.evaluate_agent(
      agent,
      envs[0],
      alpha=config.alpha,
      num_users=config.num_users_eval_final,
      deterministic=config.eval_deterministic)
  agent.set_batch_size(len(envs))


def _configure_environment_from_flags():
  """Returns an environment configuration from flag values."""
  test_ratio = 1 - (FLAGS.train_ratio + FLAGS.eval_ratio)
  user_config = movie_lens_dynamic.UserConfig(
      topic_affinity_update_threshold=FLAGS.topic_affinity_update_threshold,
      affinity_update_delta=FLAGS.affinity_update_delta)
  return movie_lens_dynamic.EnvConfig(
      seeds=movie_lens_dynamic.Seeds(None, None, None),
      data_dir=FLAGS.movielens_data_directory,
      user_config=user_config,
      train_eval_test=[FLAGS.train_ratio, FLAGS.eval_ratio, test_ratio],
      embeddings_path=FLAGS.embedding_path,
      embedding_movie_key='movie_emb',
      embedding_user_key='user_emb',
      lambda_non_violent=FLAGS.multiobjective_lambda)


def configure_expt_from_flags():
  """Returns an experiment configuration dictionary populated by flag values."""
  config = _directly_configure_from_flags([
      'initial_lambda', 'beta', 'alpha', 'embedding_size',
      'user_embedding_size', 'hidden_size', 'num_hidden_layers',
      'num_users_eval', 'num_users_eval_final', 'optimizer_name', 'num_updates',
      'eval_deterministic', 'gamma', 'clipnorm', 'checkpoint_every',
      'regularization_coeff', 'dropout', 'results_dir',
      'momentum', 'eval_every', 'stateful_rnn'
  ])
  # Need a little bit of translating in the names.
  config.update({
      'max_episode_length': FLAGS.ep_length,
      'lambda_learning_rate': FLAGS.lambda_lr,
      'var_learning_rate': FLAGS.var_lr,
      'clipval': FLAGS.clipvalue,
      'learning_rate': FLAGS.lr,
      'num_episodes_per_update': FLAGS.num_ep_per_update,
      'baseline_value': FLAGS.baseline,
      'agent_seed': None,
      'initial_agent_model': FLAGS.initial_model,
      'activity_regularization_coeff': FLAGS.activity_reg,
  })
  config['user_id_input'] = config['user_embedding_size'] > 0
  config['env_config'] = _configure_environment_from_flags()
  return config


def _directly_configure_from_flags(fields):
  """Helper function that translate flag values to a dict."""
  return {field: getattr(FLAGS, field) for field in fields}


def _set_experiment_name(config):
  experiment_name = 'id_' + hashlib.sha1(repr(sorted(
      config.items())).encode()).hexdigest()
  if FLAGS.expt_name_suffix:
    experiment_name += '_' + FLAGS.expt_name_suffix
  config['experiment_name'] = experiment_name


def main(_):
  config = configure_expt_from_flags()
  train(config)
