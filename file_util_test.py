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
"""Tests for file_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import file_util


class UtilitiesTest(absltest.TestCase):

  def test_makedirs(self):
    test_root = self.create_tempdir().full_path
    file_util.makedirs('%s/my/multilevel/directory' % test_root)
    self.assertTrue(file_util.exists('%s/my/multilevel/directory' % test_root))

  def test_delete_recursively(self):
    test_root = self.create_tempdir().full_path
    file_util.makedirs('%s/my/favorite/multilevel/directory' % test_root)
    file_util.delete_recursively('%s/my/favorite' % test_root)
    self.assertTrue(file_util.exists('%s/my' % test_root))
    self.assertFalse(file_util.exists('%s/my/favorite' % test_root))

  def test_read_write_files(self):
    test_root = self.create_tempdir().full_path
    with file_util.open('%s/hello.txt' % test_root, 'w') as outfile:
      outfile.write('hello!')

    with file_util.open('%s/hello.txt' % test_root, 'r') as readfile:
      self.assertEqual(readfile.read(), 'hello!')


if __name__ == '__main__':
  absltest.main()
