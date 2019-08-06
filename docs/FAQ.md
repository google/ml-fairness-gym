# Frequently Asked Questions (FAQ)

## What do you mean by "ml-fairness"?

Broadly, we are referring to
[machine-learning fairness](https://developers.google.com/machine-learning/fairness-overview/);
developing the benefits of machine learning for everyone.

In ML-fairness-gym, we aim to examine the implications of different decisions
made when training algorithms and make it easier to see intended and unintended
consequences of those decisions.

This version of ML-fairness-gym documentation assumes base knowledge about some
common ML concepts.

## Why do you call this a gym?

The “gym” name is a reference to [OpenAI Gym](https://gym.openai.com/) which is
the basis for the environments in ML-fairness-gym. OpenAI’s Gym has been
incredibly influential in popularizing simulation-based machine learning
challenges and encouraging reproducible research in reinforcement learning. Many
[third party extensions of OpenAI’s Gym](https://github.com/openai/gym/blob/master/docs/environments.md#third-party-environments)
refer to gym somewhere in their names.

## What can I do with ML-fairness-gym?

Take a look at our [quick-start guide](quickstart.md) and
[how to use ML-fairness-gym in your research](using_ml_fairness_gym.md) for some
initial ideas. More examples can be found in the
[examples directory](../examples/)

Simulated environments are very useful for exploration. What seems like a policy
that gives good outcomes with some parameters settings and initial conditions
may in fact play out very differently with other settings.

Some of the most surprising results (e.g., [1,2,3,4]) in the research literature
on fairness issues in machine learning have involved the implications of long
term effects of various decision making rules, agents, or policies.
ML-fairness-gym is intended to allow researchers to explore settings like this,
where long term effects of decisions influence ML fairness outcomes, in
simulated worlds that allow careful study, including the ability to examine the
effects of varying key elements, examine counterfactual results, and perform
easily replicable research in this fascinating area.

*Experiments with simulated data are meant to augment experiments and tests with
real data, not to replace them.*

## How is working with ML-fairness-gym different than working with traditional ML data sets?

The environments in ML-fairness-gym are dynamic. That means that decisions made
by your algorithm at step $$t$$ affect the next decision it will be asked to
make at time $$t+1$$.

This has a few implications: Data collection is part of the agent’s job and is
not limited by how the dataset creators happened to collect their data. You can
design agents that are more or less effective purely based on how they handle
data collection. Metrics that assess every decision independently, like
precision and recall, do not tell the full story because decisions affect each
other.

In traditional datasets which involve decisions, you can only see one outcome
based on the decision that was made at the time (e.g., no data on whether an
applicant would have paid back a loan if they were not offered one). With
simulations, this data can be calculated.

Another difference is that environments do not have a pre-specified “label” or a
“goal”, rather it is up to the agents to decide what they will optimize for and
how they will use the information.

When working with ML-fairness-gym, many of the decisions that are implicitly
made in framing the problem as a classification problem (see e.g., discussions
in [Mitchell et al. Prediction-based decisions and fairness: A catalogue of
choices, assumptions, and definitions](https://arxiv.org/abs/1811.07867)) are
made more explicit.

A drawback of working with simulations is that they are not “real data” and do
not have the full distributional or dynamic complexity of real data. We do not
try to simulate every effect of a decision in the real world, rather to use
simple stylized examples to highlight the challenges that real dynamics could
pose.

## What research results have been replicated with ML-fairness-gym?

ML-fairness-gym environments currently replicate (and generalize) dynamics
proposed in the following papers.

**Lending**
[Liu et al, Delayed Impact of Fair Machine Learning](https://arxiv.org/abs/1803.04383)

**Attention Allocation**
[Ensign et al, Runaway Feedback Loops in Predictive Policing](https://arxiv.org/abs/1706.09847);
[Elzayn et al, Fair Algorithms for Learning in Allocation Problems](https://arxiv.org/abs/1808.10549)

**Strategic Manipulation**
[Hu et al, The disparate effects of strategic manipulation](https://arxiv.org/abs/1808.08646);
[Milli et al, The Social Cost of Strategic Classification](https://arxiv.org/abs/1808.08460)

## What kinds of things can I learn from using ML-fairness-gym?

ML-fairness-gym is for implementing stylized examples of scenarios where
fairness concerns can arise. Although these examples are usually too simple to
capture the full complexity of real deployment scenarios, they provide a number
of unique opportunities to explore the properties of fair machine learning
methods in practice. For example,

*   Simulations allow researchers to explore how the moving parts in a machine
    learning deployment can interact to generate long term outcomes. These
    include censoring in the observation mechanism, errors from the learning
    algorithm, and interactions between the decision policy and environment
    dynamics.
*   Simulations allow researchers to explore how well agents and external
    auditors can assess the fairness characteristics of certain decision
    policies based on observed data alone. These examples can be useful for
    motivating auxiliary data collection policies.
*   Simulations can be used in concert with reinforcement learning algorithms to
    derive new policies with potentially novel fairness properties.

## Are results from ML-fairness-gym applicable to the real world?

Yes and no. ML-fairness-gym environments can reveal interesting dynamics that
apply in real-world scenarios, but these environments are stylized examples that
do not capture the full complexity of real deployment scenarios. We recommend
using ML-fairness-gym to characterize patterns or templates that you might then
look for in real-world scenarios. We caution against taking the application that
is used to motivate each environment (e.g., lending) too seriously - since these
are small-scale abstracted simulations, they make simplifications and
assumptions about reality. While the results from ML-fairness-gym can help
provide intution about agents in environments, and expose certain patterns of
performance, they may differ from results obtained in the real world. For this
reason we recommend the gym as a companion to evaulation with real world data
and field tests.

## What are best practices for stratifying performance metrics by group identifiers?

Many ML-fairness-gym metrics can take, as input, a stratification function which
takes a step of the history consisting of a state and action pair and returns
stratification across the state variable of interest. The stratification
function is left general enough to account for not only categorical group
identification, but intersections as well.

(As always, in stratifying results, consider how sample size is affected, and
interpret results with care, especially when determining which level of
aggregation is appropriate. Cf.
[Simpson’s paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox).)

## How do I test or benchmark a new ML algorithm inside ML-fairness-gym?

See [Using ml-fairness-gym](using_ml_fairness_gym.md).

## I have a new environment I'd like to add; how do I do that?

To create a new environment, start with the
[environments/template.py](../environments/template.py) file. The template
provides an outline to a FairnessEnv subclass, and describes what to fill in
with TODO comments. For examples of already implemented environments, see the
[environments](../environments) folder for:
[college_admission](../environments/college_admission.py)
[lending](../environments/lending.py)
[attention_allocation](../environments/attention_allocation.py).

If you would like to add your developed environment back to the ML-fairness-gym
repo, please see the [contributing doc](../CONTRIBUTING.md).

## How do I add a new agent?

The base agent class is in [core.py](../core.py). The agent class initializes
with a action_space and observation_space defined and provided by the
environment, it also takes an optional reward. To make your own agent, subclass
core.Agent and override the `_act_impl()` function. The only requirement on an
agent is that `_act_impl()` returns an action in action_space, and receives
observations in the observation_space.

For examples of agents, see the [agents](../agents/) directory.

If you would like to add your developed agent back to the ML-fairness-gym repo,
please see the [contributing doc](../CONTRIBUTING.md).

## How do I add a different metric for assessing fairness or other qualities?

The base metric class is in [core.py](../core.py). To make your own metric,
subclass core.Metric and override the `measure(env)` function. For a simple
example of a metric that calculates the sum of a state variable over the
environment’s history, see `SummingMetric` in
[value_tracking_metrics.py](../metrics/value_tracking_metrics.py).

If you would like to add your developed metric back to the ML-fairness-gym repo,
please see the [contributing doc](../CONTRIBUTING.md).

## How do I assess counterfactual results?

When implementing metrics, we look at the environment’s *state*, which is not
fully visible to the agent during the training. This allows us to assess some
counterfactuals like would the loan applicant have paid if they were given the
loan.

The metric class also has a `_simulate` method which allows them to test at any
time point, what would have happened if another action had been taken.

This is one of the key benefits of ML-fairness-gym environments over more
typical test / train splits that do not allow us to see the effect of taking a
different path than happened to be chosen in the historical snapshot.

See core.py for more details.

## What if there are multiple ways to assess ML fairness?

ML-fairness-gym allows for the use of multiple metrics to evaluate a simulation
and assess different notions of fairness. We encourage the use of multiple
metrics to get a deeper understanding of the effect a policy has in a
simulation.

`run_simluation(env, agent, metrics, num_steps, seed)` in
[run_util.py](../run_util.py) accepts a list of metrics and evaluates all the
metrics at the end of a simulation. All example experiments in the
[examples](../examples) directory involve multiple metrics and can provide
examples of analysis with multiple metrics.

## How should I report results using ML-fairness-gym?

Since simulations presented or run as part of ML-fairness-gym can have different
parameters, agents and metrics, the recommended way to report results is to
provide a main file that can reproduce plots and results from your experiments.
Examples of such main files can be found in the
[examples directory](../examples).

Providing the parameter settings for agent, environment, metrics, and the random
seed(s) used is sufficient to replicate the results and allows others to
meaningfully compare their own policies with the reported results.

## Will there be leaderboards for different ML fairness problems?

There will not be leaderboards for different ML fairness problems. It is often
difficult to compare different notions of fairness and success in regards to
them. The goal of ML-fairness-gym is to provide a framework to allow individuals
to explore problems via simulation to gain a deeper understanding of the effects
of policies on different scenarios and their various implications for fairness
over the long term.

## How should I cite ML-fairness-gym?

Please cite:

Alexander D’Amour, Yoni Halpern, Hansa Srinivasan, Pallavi Baljekar, James
Atwood, D. Sculley. Fairness is not Static: Deeper understanding of long term
fairness via agents and environments. KDD workshop on Explainable AI (XAI) 2019.

## How do I report bugs?

Feel free to report bugs by creating github issues.

## How do I contribute to the project?

See [CONTRIBUTING.md](../CONTRIBUTING.md).

Some of the python code looks a little bit strange. Are you using type
annotations? In many parts of the code base, we are using type annotations. You
can
[read more about typing in python here](https://docs.python.org/3/library/typing.html).
Another library we have found helpful for data classes and may be unfamiliar is
the [attrs library](http://www.attrs.org/en/stable/).

## What's coming next for ML-fairness-gym?

This is the initial version of the ML-fairness-gym (v 0.1.0) which focuses on
recreating environments that have previously been discussed in research papers.
The next version will have more environments and experiments.

## References

[1] Lydia T Liu, Sarah Dean, Esther Rolf, Max Simchowitz, and Moritz Hardt.
2018. Delayed Impact of Fair Machine Learning. In Proceedings of the 35th
International Conference on Machine Learning.

[2] Lily Hu, Nicole Immorlica, and Jennifer Wortman Vaughan. 2019. The disparate
effects of strategic manipulation. In Proceedings of the Conference on Fairness,
Accountability, and Transparency. ACM, 259–268.

[3] Smitha Milli, John Miller, Anca D Dragan, and Moritz Hardt. 2019. The Social
Cost of Strategic Classification. In Proceedings of the Conference on Fairness,
Accountability, and Transparency. ACM, 230–239.

[4] Danielle Ensign, Sorelle A Friedler, Scott Neville, Carlos Scheidegger, and
SureshVenkatasubramanian. 2018. Runaway Feedback Loops in Predictive Policing.
In Conference on Fairness, Accountability and Transparency. ACM, 160–171.
