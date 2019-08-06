# Environments

An environment is code that simulates a scenario in which fairness considerations may arise. The environments themselves don’t assess fairness, they simply simulate the repercussions of different decisions. By analyzing different aspects of how the scenario plays out, the user can consider the outcomes from different angles to understand whether the algorithm is indeed being fair by the criteria that they deem important.

**Some of the environments implemented here are named after some motivating application domains (e.g., lending, college admissions), but they are stylized toy examples.**

These environments are not meant to be realistic representations of the domain in question. While the insights one gains by working with an environment may be useful for designing systems for each of these domains, good performance in a gym environment should not be considered in any way sufficient for certifying performance in any real-world scenario.

Background on environments:

* [College Admission](../examples/docs/college_admission_example.md)
* [Lending](../examples/docs/lending_example.md)
* [Attention Allocation](../examples/docs/attention_allocation_example.md)
