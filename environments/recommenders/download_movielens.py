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
"""Script to download data from grouplens.org and write csv files."""

import os
import urllib.request
import zipfile
from absl import app
from absl import flags
from absl import logging
import file_util
import numpy as np
import pandas as pd
import six

flags.DEFINE_string('movielens_url',
                    'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                    'Url to download movielens dataset.')

flags.DEFINE_string(
    'tag_genome_url',
    'http://files.grouplens.org/datasets/tag-genome/tag-genome.zip',
    'Url to download tag genome dataset.')
flags.DEFINE_string('output_directory', None, 'location to write CSV files to.')

FLAGS = flags.FLAGS


def _no_gaps(sequence):
  """Returns True if a sequence has all values between 0..N with no gaps."""
  return set(sequence) == set(range(len(sequence)))


def reindex(dataframes):
  """Returns dataframes that have been reindexed to remove gaps."""
  movies, users, ratings = dataframes
  index_dict = pd.Series(
      np.arange(movies.shape[0]), index=movies['movieId'].values).to_dict()
  movies['movieId'] = np.arange(movies.shape[0])
  ratings['movieId'] = [index_dict[iden] for iden in ratings['movieId']]
  ratings['userId'] -= 1
  users['userId'] -= 1

  assert _no_gaps(movies['movieId'])
  assert _no_gaps(users['userId'])

  return movies, users, ratings


def rename_field(dataframes, original, new):
  """Consistently renames fields across data frames."""
  for df in dataframes:
    if original in df:
      df.rename(columns={original: new}, inplace=True)
  return dataframes


def read_movielens_data(url):
  """Reads movielens data and returns pandas dataframes."""

  data = urllib.request.urlopen(url).read()
  downloaded_zip = zipfile.ZipFile(six.BytesIO(data))
  logging.info('Downloaded zip file containing: %s', downloaded_zip.namelist())
  movies_df = pd.read_csv(
      downloaded_zip.open('ml-1m/movies.dat', 'r'),
      sep='::',
      names=['movieId', 'title', 'genres'],
      encoding='iso-8859-1')

  users_df = pd.read_csv(
      downloaded_zip.open('ml-1m/users.dat', 'r'),
      sep='::',
      names=['userId', 'sex', 'age', 'occupation', 'zip_code'],
      encoding='iso-8859-1')

  ratings_df = pd.read_csv(
      downloaded_zip.open('ml-1m/ratings.dat', 'r'),
      sep='::',
      names=['userId', 'movieId', 'rating', 'timestamp'],
      encoding='iso-8859-1')
  return movies_df, users_df, ratings_df


def merge_with_genome_data(url, target_tag, dataframes):
  """Add in tag relevance to movies dataframe for a single tag."""
  data = urllib.request.urlopen(url).read()
  downloaded_zip = zipfile.ZipFile(six.BytesIO(data))
  logging.info('Downloaded zip file containing: %s', downloaded_zip.namelist())
  tags_df = pd.read_csv(
      downloaded_zip.open('tag-genome/tags.dat', 'r'),
      sep='\t',
      names=['tag_id', 'tag', 'tag_popularity'],
      encoding='iso-8859-1')

  target_tag_id = tags_df[tags_df.tag == target_tag].tag_id.values[0]
  logging.info('%s corresponds to tag %d', target_tag, target_tag_id)

  tag_relevance_df = pd.read_csv(
      downloaded_zip.open('tag-genome/tag_relevance.dat', 'r'),
      sep='\t',
      names=['movieId', 'tag_id', 'relevance'],
      encoding='iso-8859-1')

  # Filter for rows that contain the target tag id.
  tag_relevance_df = tag_relevance_df[tag_relevance_df.tag_id == target_tag_id]

  # Merge tag relevance values on to the movies dataframe.
  movies_df, users_df, ratings_df = dataframes
  movies_df = movies_df.merge(tag_relevance_df,
                              on='movieId',
                              how='left').fillna(0)
  movies_df.rename(
      columns={'relevance': '%s_tag_relevance' % target_tag}, inplace=True)

  logging.info('Movies df has keys %s', list(movies_df.keys()))
  logging.info('Movies df now looks like this %s', movies_df.head().to_string())
  return movies_df, users_df, ratings_df


def write_csv_output(dataframes, directory):
  """Write csv file outputs."""
  movies, users, ratings = dataframes
  file_util.makedirs(directory)

  del movies['tag_id']  # This column isn't necessary.

  users.to_csv(
      file_util.open(os.path.join(directory, 'users.csv'), 'w'),
      index=False,
      columns=['userId'])
  movies.to_csv(
      file_util.open(os.path.join(directory, 'movies.csv'), 'w'),
      index=False)
  ratings.to_csv(
      file_util.open(os.path.join(directory, 'ratings.csv'), 'w'),
      index=False)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataframes = read_movielens_data(FLAGS.movielens_url)
  dataframes = merge_with_genome_data(FLAGS.tag_genome_url, 'violence',
                                      dataframes)
  dataframes = reindex(dataframes)
  write_csv_output(dataframes, FLAGS.output_directory)


if __name__ == '__main__':
  app.run(main)
