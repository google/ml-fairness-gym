# Recommendation environments

The recommender system environments in ml-fairness-gym are currently under
development. This is a work in progress.

## Background on RecSim

The recommendation environments in ml-fairness-gym are written using RecSim,
a configurable platform for authoring simulation environments for 
recommender systems (RSs) that naturally supports sequential interaction with users. 
For more information on RecSim, please see the [github repository]((https://github.com/google-research/recsim), 
[white paper](https://arxiv.org/abs/1909.04847), and [colab tutorials](https://github.com/google-research/recsim/blob/master/recsim/colab/RecSim_Overview.ipynb).

## Available environments

### Restaurant recommendations

This environment simulates interactions between a simulated user and a recommender that
recommends one of two restaurants: one that serves healthy food and one
that serves junk food. The simulated user's preferences for healthy and junk food evolve
over time following a Markov Decision Process in which their current preference
depends on their previous preferences and the most recent restaurant that they
have attended. The parameters of the MDP are configured when constructing the 
environment. 

Using the notation from RecSim, the “documents” are two restaurants, one healthy and one unhealthy. The environment consists of a simulated user who receives recommendations and responds with a rating of how much they enjoyed the restaurant. At each step, the recommender agent recommends one or the other, and observes the user’s rating from the last recommendation. 

It is intended to explore tradeoffs between simulated user preferences and long term
healthiness. 


### Movielens recommendations

Similar to the Restaurant recommendation environment, but on a larger
scale, This environment simulates interactions between many different simulated viewers and a
recommender that recommends movies from the [Movielens 1M corpus](https://grouplens.org/datasets/movielens/1m/). The viewer preferences are simulated using a low-dimensional matrix factorization model from the Movielens 1M rating  data. Optionally, viewers can be configured to become more interested in a movie genre
as they watch movies from that genre, simulating an addictive dynamic, or less
interested, simulating boredom.

Here the “documents” are the movies in the movielens 1M corpus. The environment consists of a number of simulated viewers who receive recommendations and respond with a simulated rating of how much they “enjoyed” the movie. At each step, the recommender agent recommends one or the other, and observes the simulated viewer’s rating from the last recommendation. On reset, a new simulated viewer  is sampled and the interaction repeats again with a new viewer. 

Movies are also annotated with a violence score, derived from the [Movie Genome
Project](https://grouplens.org/datasets/movielens/tag-genome/). 
While considering the quality of recommendations, an agent can consider both
the simulated viewer’s feedback in the form of ratings, but also how much representations of
violence the simulated viewer is being exposed to.

The Movielens 1M and Movie Genome datasets are not distributed with ml-fairness-gym.
The `download_movielens.py` script provided in this directory downloads the data
from the grouplens.org website. Please read the relevant license and README files
associated with the data. 
The matrix factorization of user/movie preferences accompanying the simulation is trained on Movielens 1M and released with permission of the Grouplens research group at University of Minnesota.


### Wrapping other RecSim Environments

A number of other environments are also available in the [RecSim repository](https://github.com/google-research/recsim/tree/master/recsim/environments). The `recsim_wrapper.py` library
is intended to make it easy to convert any RecSim environment to an ml-fairness-gym
environment.
