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

"""Parameters for lending environment.

Lending environments are mainly parametrized by a distribution over applicants.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
import core
import distributions
import numpy as np
from typing import Callable, Sequence


@attr.s
class Applicant(object):
  """Data class to store applicant information."""
  features = attr.ib()  # type: np.ndarray
  group = attr.ib()  # type: Sequence[int]
  will_default = attr.ib()  # type: bool

  def __attrs_post_init__(self):
    self.dim = len(self.features)


@attr.s
class ApplicantDistribution(distributions.Distribution):
  """Distribution to sample applicants."""
  features = attr.ib()  #  type: distributions.Distribution
  group_membership = attr.ib()  #  type: distributions.Distribution
  will_default = attr.ib()  #  type: distributions.Distribution

  def __attrs_post_init__(self):
    self.dim = self.features.dim

  def sample(self, rng):
    return Applicant(
        features=self.features.sample(rng),
        group=self.group_membership.sample(rng),
        will_default=self.will_default.sample(rng))


def _rotate(x, degrees):
  """Returns x rotated around the origin by degrees."""
  theta = np.deg2rad(degrees)
  cos, sin = np.cos(theta), np.sin(theta)
  rotation_matrix = np.array([[cos, -sin], [sin, cos]])
  return np.matmul(rotation_matrix, x)


def _gmm_applicant_builder(mean,
                           intercluster_vec,
                           default_likelihoods,
                           group
                          ):
  """Returns a distribution builder for the SimpleLoans environment.

  The SimpleLoans environment has two clusters of applicants, one that tends
  to default and one that tends to pay back. Each of the clusters have features
  that are normally distributed around some mean.

  Args:
    mean: mean of the failing cluster.
    intercluster_vec: vector between the means of the two clusters.
    default_likelihoods: list containing the likelihood of default for each
      cluster.
    group: A constant that indicates group membership in this setting.

  Returns:
    A function that returns a distribution.
  """

  def _single_gmm():
    """Returns a mixture of gaussian applicant distributions."""
    return distributions.Mixture(
        components=[
            ApplicantDistribution(
                features=distributions.Gaussian(mean=mean, std=0.5),
                group_membership=distributions.Constant(group),
                will_default=distributions.Bernoulli(p=default_likelihoods[0])),
            ApplicantDistribution(
                features=distributions.Gaussian(
                    mean=np.array(mean) + np.array(intercluster_vec), std=0.5),
                group_membership=distributions.Constant(group),
                will_default=distributions.Bernoulli(p=default_likelihoods[1]))
        ],
        weights=[0.3, 0.7])

  return _single_gmm


def _rotated_gmm_applicant_builder(mean,
                                   intercluster_vec,
                                   default_likelihoods,
                                   degrees,
                                   group_likelihoods
                                  ):
  """Returns a distribution builder for DifferentialExpressionEnv.

  The DifferentialExpressionEnv environment has two groups of applicants.
  Each group divides up into two clusters, one that tends to default and one
  that tends to pay back.

  Each of the clusters have features that are normally distributed around some
  mean, but the second group is a rotated version of the first group.

  Args:
    mean: mean of the failing cluster in group 0.
    intercluster_vec: vector between the means of the two clusters of group 0.
    default_likelihoods: list containing the likelihood of default for each
      cluster.
    degrees: degrees of rotation between the two groups.
    group_likelihoods: likelihood of drawing from each of the two groups.

  Returns:
    A function that returns a distribution.
  """

  def _rotated_gmm():
    """Returns a mixture of applicant distributions.

    The second applicant distribution is a rotated version of the first.
    """
    return distributions.Mixture(
        components=[
            _gmm_applicant_builder(
                mean=mean,
                intercluster_vec=intercluster_vec,
                default_likelihoods=default_likelihoods,
                group=[1, 0])(),
            _gmm_applicant_builder(
                mean=_rotate(mean, degrees),
                intercluster_vec=_rotate(intercluster_vec, degrees),
                default_likelihoods=default_likelihoods,
                group=[0, 1])(),
        ],
        weights=group_likelihoods)

  return _rotated_gmm


def _credit_cluster_builder(group_membership,
                            cluster_probs,
                            success_probs
                           ):
  """Returns a distribution builder for DelayedImpactEnv.

  The DelayedImpactEnv environment applicants distributed between some number
  of discrete credict scores. The credit score determines the likelihood of
  default for applicants with that score.

  Args:
    group_membership: A constant vector for group membership.
    cluster_probs: vector of size num_clusters with likelihood of drawing from
      each of the credit clusters.
    success_probs: vector of size num_clusters with likelihood of successful
      payback given that an applicant is drawn from that cluster.

  Returns:
    A function that returns a distribution.
  """
  feature_dim = len(success_probs)

  def _single_credit_cluster(idx):
    vec = np.zeros(feature_dim, dtype=np.int32)
    vec[idx] = 1
    return ApplicantDistribution(
        features=distributions.Constant(mean=vec),
        group_membership=distributions.Constant(group_membership),
        will_default=distributions.Bernoulli(1-success_probs[idx]))

  def _credit_clusters():
    return distributions.Mixture(
        components=[_single_credit_cluster(idx) for idx in range(feature_dim)],
        weights=cluster_probs)

  return _credit_clusters


# Likelihoods of credit score given group_id.
DELAYED_IMPACT_CLUSTER_PROBS = (
    (0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0),
    (0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0),
)
# Likelihoods of loan repayment given credit score.
DELAYED_IMPACT_SUCCESS_PROBS = (0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7)


def two_group_credit_clusters(
    group_likelihoods=(0.5, 0.5),
    cluster_probabilities=DELAYED_IMPACT_CLUSTER_PROBS,
    success_probabilities=DELAYED_IMPACT_SUCCESS_PROBS):
  """Returns a mixture of two credit cluster distributions.

  Args:
    group_likelihoods: Probabilities of choosing from each group.
      Should be non-negative and sum to one.
    cluster_probabilities: cluster_probabilities[i][j] gives the probability of
      being drawn from credit cluster j conditioned on being from group i.
    success_probabilities: success_probabilities[j] gives the probability of
      successfully paying back a loan conditioned on being from being from
      credit cluster j.
  """
  components = []
  for idx in [0, 1]:
    # Use a one-hot encoding of group-id.
    group_vec = np.zeros(2)
    group_vec[idx] = 1
    components.append(
        _credit_cluster_builder(
            group_membership=tuple(group_vec),
            cluster_probs=tuple(cluster_probabilities[idx]),
            success_probs=tuple(success_probabilities))())
  return distributions.Mixture(components=components, weights=group_likelihoods)


@attr.s
class Params(core.Params):
  """Parameter class for the SimpleLoan environment."""
  applicant_distribution = attr.ib(
      factory=_gmm_applicant_builder(
          # Mean of the first cluster.
          mean=[0, 1],
          # Vector to the second cluster.
          intercluster_vec=[1, -1],
          default_likelihoods=[0.8, 0.2],
          # Only one group is present here.
          group=[1])
  )  # type: distributions.Distribution

  num_groups = attr.ib(default=1)  # type: int
  min_observation = attr.ib(default=-2.)  # type: float
  max_observation = attr.ib(default=2.)  # type: float

  # Loan constants.
  loan_amount = attr.ib(default=1.)  # type: float
  interest_rate = attr.ib(default=0.30)  # type: float
  bank_starting_cash = attr.ib(default=1000.)  # type: float
  max_cash = attr.ib(default=1e20)  # type: float


@attr.s
class DifferentialExpressionParams(Params):
  """Parameter class for the DifferentialExpressionEnv environment."""

  num_groups = attr.ib(default=2)  # type: int
  applicant_distribution = attr.ib(
      factory=_rotated_gmm_applicant_builder(
          mean=[0, 1],  # Mean of the first cluster in group 0
          intercluster_vec=[1, -1],  # Vector to the second cluster in group 0/
          default_likelihoods=[0.8, 0.2],
          degrees=30,  # Rotation around the origin between the two groups.
          group_likelihoods=[0.5, 0.5]))  # type: distributions.Distribution


@attr.s
class DelayedImpactParams(Params):
  """Parameters class for the DelayedImpactEnv environment.

  These parameters are intended to simulate the dynamics of Liu et al's Delayed
  Impact of Fair Machine Learning: https://arxiv.org/abs/1803.04383.
  """
  applicant_distribution = attr.ib(
      factory=two_group_credit_clusters)  # type: distributions.Distribution
  num_groups = attr.ib(default=2)  # type: int

  min_observation = attr.ib(default=0)  # type: float
  max_observation = attr.ib(default=1)  # type: float

  # Probability mass that shifts away from a cluster due to a raise or
  # lowering of credit score in response to an accepted application.
  cluster_shift_increment = attr.ib(default=0.01)  # type: float
