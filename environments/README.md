# Environments

An environment is code that simulates a scenario in which fairness considerations may arise. The environments themselves don’t assess fairness, they simply simulate the repercussions of different decisions. By analyzing different aspects of how the scenario plays out, the user can consider the outcomes from different angles to understand whether the algorithm is indeed being fair by the criteria that they deem important.

**Some of the environments implemented here are named after some motivating application domains (e.g., lending, college admissions), but they are stylized toy examples.**

These environments are not meant to be realistic representations of the domain in question. While the insights one gains by working with an environment may be useful for designing systems for each of these domains, good performance in a gym environment should not be considered in any way sufficient for certifying performance in any real-world scenario.

For background on the environments see the following papers:

[Fairness is not Static: Deeper Understanding of Long Term Fairness via Simulation Studies](../papers/acm_fat_2020_fairness_is_not_static.pdf)

* College admissions
* Lending
* Attention allocation

[Fair treatment allocations in social networks](../papers/fairmlforhealth2019_fair_treatment_allocations_in_social_networks.pdf)

* Infectious disease

