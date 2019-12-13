# The configurable fairness gym runner
The fairness gym comes equipped with a configurable experiment runner that can
be used to run experiments in a consistent, reproducible way across
environments, agents, and metrics.

## Design

Evironment, agent, and metric definitions within the fairness gym can be
complex, and there is not a unified interface.  Rather than impose an interface,
the configurable runner uses dependency injection to partially define
environments, agents, and metrics such that they are compatible with a simple
experimental framework.  This has the benefit of allowing for rich definitions
of fairness gym elements while still providing a consistent method for running
experiments.

The experimental framework is given in the Runner class in the runner_lib
module.  Runner defines an experiment where an environment, agent, and some
number of metrics are instantiated, seeded, and interact with each other for
some specified number of steps.  After that interaction is complete, a report
containing the metric results is generated.

The configurable runner uses the [Gin library](https://github.com/google/gin-config)
for configuration through dependency injection.

## Usage

Experiments are first defined by creating a configuration file and making relevant definitions visible to Gin through decoration.  They are then run using the runner.py module.

### Experiment definition
Experiment definitions are [Gin configuration files](https://github.com/google/gin-config#2-configuring-default-values-with-gin-ginconfigurable-and-bindings)
that register an environment, an agent, and some number of metrics with the
Runner class, configure those classes in such a way that they are compatible with Runner, and configure the Runner itself by setting things like the number of steps to run for and the random seed.

The simplest example configuration is given in experiments/config/example.gcl.
This configuration registers a DummyEnvironment, DummyAgent, and DummyMetric
and runs a simulation for ten steps:

    import fairness_gym.core
    import fairness_gym.test_util

    Runner.env_class = @test_util.DummyEnv
    Runner.env_params_class = @core.Params
    Runner.agent_class = @test_util.DummyAgent
    Runner.metric_classes = {'num_steps': @test_util.DummyMetric}
    Runner.num_steps = 10
    Runner.seed = 4321

Note that classes on the right hand side are prefixed with an '@' symbol and
their modules are imported in the frontmatter.  Importantly, each of these
classes decorated with a gin.configurable statement in the code.  In order to
refer to classes and functions in new configurations, this decoration must be
present in the code.

A more complex example is found in
experiments/config/college_admission_config.gin:

    import fairness_gym.agents.college_admission_jury
    import fairness_gym.environments.college_admission
    import fairness_gym.experiments.college_admission_util
    import fairness_gym.metrics.error_metrics
    import fairness_gym.metrics.value_tracking_metrics
    import fairness_gym.runner_lib

    # Configure the runner.
    Runner.num_steps = 3000
    Runner.seed = 1
    Runner.env_class = @college_admission.CollegeAdmissionsEnv
    Runner.agent_class = @college_admission_jury.FixedJury
    Runner.simulation_fn = @runner_lib.run_stackelberg_simulation

    # Configure the agent.
    college_admission_jury.FixedJury.threshold = 0.5
    college_admission_jury.FixedJury.epsilon_greedy = False
    college_admission_jury.FixedJury.decay_steps = 20
     college_admission_jury.FixedJury.initial_epsilon_prob = 0.1
    college_admission_jury.FixedJury.epsilon_prob_decay_rate = 0.02

    # Specify metrics.
    Runner.metric_classes = {
    "social_burden": @social_burden/value_tracking_metrics.AggregatorMetric,
    "accuracy": @accuracy/error_metrics.AccuracyMetric,
    "overall_accuracy": @overall_accuracy/error_metrics.AccuracyMetric,
    "overall_social_burden": @overall_social_burden/value_tracking_metrics.AggregatorMetric,
    "final_threshold": @final_threshold/value_tracking_metrics.FinalValueMetric,
    }

    # Configure metrics.
    social_burden/value_tracking_metrics.AggregatorMetric.selection_fn = @college_admission_util.selection_fn_social_burden_eligible_auditor
    social_burden/value_tracking_metrics.AggregatorMetric.modifier_fn = None
    social_burden/value_tracking_metrics.AggregatorMetric.stratify_fn= @college_admission_util.stratify_by_group
    social_burden/value_tracking_metrics.AggregatorMetric.realign_fn = @college_admission_util.realign_history
    social_burden/value_tracking_metrics.AggregatorMetric.calc_mean = True

    accuracy/error_metrics.AccuracyMetric.numerator_fn = @college_admission_util.accuracy_nr_fn
    accuracy/error_metrics.AccuracyMetric.denominator_fn = None
    accuracy/error_metrics.AccuracyMetric.stratify_fn = @college_admission_util.stratify_by_group
    accuracy/error_metrics.AccuracyMetric.realign_fn = @college_admission_util.realign_history

    overall_accuracy/error_metrics.AccuracyMetric.numerator_fn = @college_admission_util.accuracy_nr_fn
    overall_accuracy/error_metrics.AccuracyMetric.denominator_fn = None
    overall_accuracy/error_metrics.AccuracyMetric.stratify_fn = @college_admission_util.stratify_to_one_group
    overall_accuracy/error_metrics.AccuracyMetric.realign_fn = @college_admission_util.realign_history

    overall_social_burden/value_tracking_metrics.AggregatorMetric.selection_fn = @college_admission_util.selection_fn_social_burden_eligible_auditor
    overall_social_burden/value_tracking_metrics.AggregatorMetric.modifier_fn = None
    overall_social_burden/value_tracking_metrics.AggregatorMetric.stratify_fn= @college_admission_util.stratify_to_one_group
    overall_social_burden/value_tracking_metrics.AggregatorMetric.realign_fn = @college_admission_util.realign_history
    overall_social_burden/value_tracking_metrics.AggregatorMetric.calc_mean = True

    final_threshold/value_tracking_metrics.FinalValueMetric.state_var = "decision_threshold"
    final_threshold/value_tracking_metrics.FinalValueMetric.realign_fn = @college_admission_util.realign_history

In this example, the agent and metrics require further configuration.  Note that
the agent_class is set to @college_admission_jury.FixedJury, and that arguments
to FixedJury are further configured below.

Note also that several metrics are defined and configured.  Because individual
metric classes are used more than once, they are prepended with namespace/
identifiers (like social_burden/ and accuracy/) that allow the classes to be
configured in different ways.

### Running experiments
To run the experiment, pass your config to the runner:

    python -m runner \
      --gin_config_path=experiments/config/college_admission_config.gin \
      --output_path=/tmp/output.json
