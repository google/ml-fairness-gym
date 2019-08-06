# Using fairness-gym in your research

ML-fairness-gym brings reinforcement learning-style evaluations to
fairness in machine learning research. Here is a suggested pattern for using
the ML-fairness-gym as part of the research process. Others may be added here as
we continue to grow.

## Evaluating a proposed ML algorithm

Here are suggested steps when evaluating a proposed new fair ML algorthm:

*   Choose a [simulation environment](../environments/README.md).
*   Decide on [metrics](../metrics) that
    you would like to measure for that environment.
*   Choose baseline [agents](../agents)
    and choose what reward functions they will optimize.
*   Write an agent that uses your new algorithm.
*   Compare metrics between your baseline agents and your fair agent. Some
    utilities for building experiments are provided in run_util.py. For example,
    `run_simulation`is a simple function that runs an experiment and returns
    metric measurements.
*   Explore parameter settings in your simulation environment - are there
    different regimes?

We provide some implementations of environments, agents, and metrics, but they
are by no means comprehensive. Feel free to implement your own and contribute
to ML-fairness-gym!

