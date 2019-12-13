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

# Lint as: python2, python3
r"""A gin-configurable experiment runner for the fairness gym.

Example usage:
bazel run -c opt third_party/py/fairness_gym:runner -- \
--alsologtostderr \
--gin_config_path=\
third_party/py/fairness_gym/experiments/config/experiments_config.gin \
--output_path=/tmp/output.json

After that finishes, /tmp/output.json should look like this:
{"agent": {"name": "DummyAgent"},
 "environment": {"name": "DummyEnv", "params": {}},
 "metrics": {"num_steps": 10}}
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import core
import runner_lib
import gin

flags.DEFINE_string(
    'gin_config_path',
    '/tmp/config.gin',
    'Path to the gin configuration that specifies this experiment.')

flags.DEFINE_string(
    'output_path',
    '/tmp/output.json',
    'Path where output JSON will be written.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_file(FLAGS.gin_config_path)
  runner = runner_lib.Runner()

  results = runner.run()
  logging.info('Results: %s', results)

  with open(FLAGS.output_path, 'w') as f:
    f.write(core.to_json(results))


if __name__ == '__main__':
  app.run(main)
