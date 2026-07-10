---
title: "Sugar: Learning Generalizable Humanoid Loco-Manipulation from Diverse Human Videos"
authors: Tianshu Wu, Xiangqi Kong, Yue Chen, Qize Yu, Hang Ye, Jia Li, Yizhou Wang, Hao Dong
conference: arXiv
year: 2026
tags:
  - humanoid
  - loco-manipulation
  - learning-from-video
  - reinforcement-learning
  - sim-to-real
link: https://arxiv.org/abs/2605.20373
aliases:
  - Sugar
---
# Research Question
Can diverse, unannotated human videos be turned into deployable, reference-free whole-body loco-manipulation skills for a real humanoid robot?

# Motivation
Task-specific RL, reference-motion replay, and teleoperation all scale poorly, while abundant human videos yield motion priors too noisy (occlusion, contact artifacts, retargeting errors) for direct imitation.

# Method
A three-stage pipeline that automatically extracts human-object trajectories and VLM-labeled contacts from raw videos, refines them into physically feasible skills via a privileged RL policy with a unified mimic reward and progressive state pool, and distills them into a hierarchical policy (diffusion-based command generator + whole-body command tracker).

# Limitation

no visual input, not VLA, not interested.
error recover is just with mocap system. it knows position of all objects.

# Summary
Noisy video-extracted motion data is unusable for direct imitation but captures the complete task logic — so physics-based refinement in simulation, not better reconstruction, is the key that unlocks scalable, reference-free humanoid loco-manipulation that improves with more video data and transfers zero-shot to real hardware.

# Connections
- [[HDMI]] (HDMI: learning interactive humanoid whole-body control from human videos) — is-baseline-for: reference-based replay of video HOI trajectories that Sugar outperforms and generalizes beyond.
- [[ResMimic]] (ResMimic: from general motion tracking to humanoid whole-body loco-manipulation) — is-baseline-for: residual-learning reference tracker compared against on all six tasks.
- [[HumanX]] (HumanX: agile and generalizable humanoid interaction skills from human videos) — competes-with: concurrent work compiling videos into skills, but relies on manually defined anchor points rather than large-scale multi-trajectory HOI data.
