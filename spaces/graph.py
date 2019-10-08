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
"""Graph spaces for the ML fairness gym."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Any
import gym
import networkx as nx


class GraphSpace(gym.Space):
  """The space of random NetworkX graphs with a given number of nodes.

  Graphs sampled from this space are drawn from the Erdos-Renyi random graph
  model where each pair of nodes shares an edge with probability p.

  Graphs can be directed or undirected.

  Two graph spaces are considered equivalent if they have the same number of
  nodes, the same edge probability, and the same directedness.
  """

  def __init__(self, num_nodes, directed = False, p = 0.05):
    """Initialize a GraphSpace instance.

    Args:
      num_nodes: A positive integer indicating the number of nodes of graphs
        that are contained in this space.
      directed: A boolean indicating whether this space contains directed or
        undirected graphs.
      p: A float in [0, 1] that gives the probability that any two nodes are
        connected by an edge in graphs that are sampled from this space.
    """
    self.num_nodes = num_nodes
    self.directed = directed
    self.p = p

  def contains(self, item):
    return (isinstance(item, nx.Graph)
            and item.number_of_nodes() == self.num_nodes)

  def sample(self):
    return nx.fast_gnp_random_graph(
        self.num_nodes, self.p, directed=self.directed)

  def __repr__(self):
    return 'Graph (%d, %.4f, %s)' % (
        self.num_nodes, self.p, self.directed)

  def __eq__(self, other):
    return (
        isinstance(other, self.__class__)
        and self.num_nodes == other.num_nodes
        and self.p == other.p
        and self.directed == other.directed)


