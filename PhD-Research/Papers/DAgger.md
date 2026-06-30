---
title: "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
authors: [Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell]
conference: AISTATS
year: 2011
tags: [imitation-learning, online-learning, structured-prediction, no-regret]
link: https://arxiv.org/abs/1011.0686
aliases: [DAgger, Dataset Aggregation]
---
# Research Question
How can we imitation-learn a single stationary deterministic policy whose error grows linearly, not quadratically, with the task horizon $T$?

# Motivation
Standard supervised imitation breaks the i.i.d. assumption because the learner's own mistakes shift the state distribution, compounding errors up to $T^2\epsilon$.

# Method
**DAgger** iteratively runs the current policy to collect states, labels them with the expert's actions, aggregates them into one growing dataset, and retrains—reducing imitation to no-regret online learning.

# Limitation

# Summary
By training on the state distribution the learner actually induces (rather than the expert's), DAgger turns imitation into a no-regret online learning problem and guarantees performance that degrades only linearly in $T$.

# Connections
- **SMILe / SEARN** — builds-on (earlier iterative reductions DAgger improves over by learning a single deterministic policy)
- **Behavior Cloning** — is-baseline-for (the naive supervised approach DAgger fixes)
