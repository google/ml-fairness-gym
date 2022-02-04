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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest import mock
from absl.testing import absltest
import runner_lib
from experiments import college_admission_util
import gin


class CollegeAdmissionUtilTest(absltest.TestCase):

  def test_stratify_to_one_group_stratifies_to_one_group(self):
    step = mock.Mock()
    step.state.params.num_applicants = 10
    self.assertLen(
        set(college_admission_util.stratify_to_one_group(step)), 1)

  def test_stratify_by_group_returns_correct_groups(self):
    applicant_groups = [1, 2, 1, 2, 1, 1, 2]
    valid_groups = set(applicant_groups)

    step = mock.Mock()
    step.state.applicant_groups = applicant_groups

    for group in college_admission_util.stratify_by_group(step):
      self.assertIn(group, valid_groups)

  def test_accuracy_nr_fn_returns_whether_predictions_correct(self):
    predictions = [1, 1, 2]
    groundtruth = [1, 2, 2]
    expected = [True, False, True]

    step = mock.Mock()
    step.state.selected_applicants = predictions
    step.state.true_eligible = groundtruth

    self.assertCountEqual(expected, college_admission_util.accuracy_nr_fn(step))

  def test_social_burden_eligible_auditor_selects_eligible(self):
    eligible = [1, 0, 1]
    burden = [1, 2, 3]
    expected = [1, 0, 3]

    step = mock.Mock()
    step.state.true_eligible = eligible
    step.state.individual_burden = burden

    self.assertCountEqual(
        expected,
        college_admission_util.selection_fn_social_burden_eligible_auditor(
            step))



if __name__ == '__main__':
  absltest.main()
