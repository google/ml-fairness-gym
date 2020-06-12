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
"""Helper functions for working with different file systems."""
import os
import shutil
import tensorflow.compat.v1 as tf


open = tf.gfile.GFile  # pylint: disable=redefined-builtin
copy = tf.gfile.Copy
remove = tf.gfile.Remove
makedirs = os.makedirs
exists = os.path.exists
list_files = os.listdir
delete_recursively = shutil.rmtree


