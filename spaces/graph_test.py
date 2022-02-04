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

# Lint as: python2, python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from spaces import graph


class GraphTest(absltest.TestCase):

  def test_sampled_graphs_contain_correct_number_of_nodes(self):
    num_trials = 10
    num_nodes = 100
    p = 0.05
    space = graph.GraphSpace(num_nodes, directed=False, p=p)

    for _ in range(num_trials):
      g = space.sample()
      self.assertEqual(g.number_of_nodes(), num_nodes)

  def test_undirected_space_not_equal_to_directed_space(self):
    num_nodes = 100
    p = 0.05

    directed_space = graph.GraphSpace(num_nodes, directed=True, p=p)
    undirected_space = graph.GraphSpace(num_nodes, directed=False, p=p)

    self.assertNotEqual(directed_space, undirected_space)


if __name__ == '__main__':
  absltest.main()
