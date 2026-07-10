---
title: "ZeroWBC: Teleoperation-Free Humanoid Whole-Body Interaction from Human Egocentric Videos"
authors: Haoran Yang, Jiacheng Bao, Yucheng Xin, Haoming Song, Yuyang Tian, Bin Zhao, Dong Wang, Xuelong Li
conference: arXiv
year: 2026
tags:
  - humanoid
  - whole-body-control
  - motion-generation
  - motion-tracking
link: https://arxiv.org/abs/2603.09170
aliases:
  - ZeroWBC
---
# Research Question
Can a humanoid learn from human egocentric videos with synchronized motion and text, without any robot teleoperation data?
# Motivation
Whole-body teleoperation is expensive (two operators, ~100 demos in 8h), while one human with a MoCap suit and chest camera collects ~300 egocentric demos in 2h.
# Method
A fine-tuned Qwen2.5-VL generates VQ-VAE motion tokens from one egocentric image and a text instruction, which are decoded, retargeted, and executed open-loop by an RL tracker whose reward prioritizes global root and wrist/foot trajectories over joint-space accuracy.
# Limitation

The object interactive with, is not moveable.

# Summary
Interaction precision survives human-to-robot transfer by tracking the task-critical world-space points (root, wrists, feet) rather than joints, letting cheap human egocentric data replace teleoperation for static-scene tasks (box moving 16%→64% over a general tracker).
# Connections
- [[SONIC]] — is-baseline-for: used as the general-tracker baseline; ZeroWBC trades slightly worse MPJPE for better interactive body-part tracking (MPIPE/MPIVE).
- [[MotionGPT]] — builds-on: same motion-tokens-as-language-tokens recipe, extended with egocentric image conditioning.
- [[Twist2]] — competes-with: also holistic whole-body control, but depends on costly hardware-specific teleoperation data that ZeroWBC avoids.
