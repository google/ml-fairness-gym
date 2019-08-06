# What is ML-fairness-gym?

ML-fairness-gym is a set of components for building simple simulations that explore the potential long-run impacts of deploying machine learning-based decision systems in social environments. As the importance of machine learning fairness has become increasingly apparent, recent research has focused on potentially surprising long term behaviors of enforcing measures of fairness that were originally defined in a static setting. Key findings have shown that under specific assumptions in simplified dynamic simulations, long term effects may in fact counteract the desired goals. Achieving a deeper understanding of such long term effects is thus a critical direction for ML fairness research. ML-fairness-gym implements a generalized framework for studying and probing long term fairness effects in carefully constructed simulation scenarios where a learning agent interacts with an environment over time. This work fits into a larger push in the fair machine learning literature to design decision systems that induce fair outcomes in the long run, and to understand how these systems might differ from those designed to enforce fairness on a one-shot basis.

This initial version of the ML-fairness-gym (v 0.1.0) focuses on reproducing and generalizing environments that have [previously been discussed in research papers](docs/FAQ.md:#What-research-results-have-been-replicated-with-ML_fairness_gym).

ML-fairness-gym environments implement the environment API from [OpenAI Gym](https://github.com/openai/gym).

This is not an officially supported Google product.


# Contents

* [Installation instructions](docs/installation.md)
* [Quick start guide](docs/quickstart.md)
* [Frequently asked questions](docs/FAQ.md)
* [Using ML-fairness-gym](docs/using_ml_fairness_gym.md)
* Environments
    * [College Admission](examples/docs/college_admission_example.md)
    * [Lending](examples/docs/lending_example.md)
    * [Attention Allocation](examples/docs/attention_allocation_example.md)
* [Examples](examples/README.md)


# Contact us

The ML fairness gym project discussion group is:
ml-fairness-gym-discuss@google.com.

# Versions
v0.1.0: Initial release.
