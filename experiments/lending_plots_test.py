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

# Lint as: python2, python3
"""Tests for lending_plots.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import file_util
from experiments import lending
from experiments import lending_plots


class LendingPlottingTest(absltest.TestCase):

  def assert_contains_pdfs(self, directory):
    """Helper function that asserts a directory contains pdf files."""

    self.assertNotEmpty([
        fname for fname in file_util.list_files(directory)
        if fname.endswith('.pdf')
    ])

  def assert_contains_no_pdfs(self, directory):
    """Helper function that asserts a directory does NOT contain pdf files."""
    self.assertEmpty([
        fname for fname in file_util.list_files(directory)
        if fname.endswith('.pdf')
    ])

  def test_plotting(self):
    plotting_dir = self.create_tempdir().full_path

    self.assert_contains_no_pdfs(plotting_dir)

    # Run a single experiment to create the results.
    result = lending.Experiment(
        num_steps=100, include_cumulative_loans=True, return_json=False).run()
    # Tests that all_plots can execute.
    lending_plots.do_plotting(result, result, result, plotting_dir)
    # There should be at least one .pdf file created.
    self.assert_contains_pdfs(plotting_dir)


if __name__ == '__main__':
  absltest.main()
