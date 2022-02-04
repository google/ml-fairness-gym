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

"""Tests for download_movielens."""

from absl import flags
from absl.testing import absltest
import pandas as pd
import six

FLAGS = flags.FLAGS


class DownloadMovielensTest(absltest.TestCase):

  def test_pandas_bytes_behavior(self):
    """Check that pandas treats BytesIO stream as a string."""
    f = six.BytesIO(b'movieId::title::genres\n1::Toy Story (1995)::Animation')
    _ = pd.read_csv(f, sep='::', encoding='iso-8859-1')


if __name__ == '__main__':
  absltest.main()
