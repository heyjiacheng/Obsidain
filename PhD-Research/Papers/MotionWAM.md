---
title: "MotionWAM: A Real-Time World Action Model for Whole-Body Humanoid Loco-Manipulation"
authors: Jia Zheng, Teli Ma, Yudong Fan, Zifan Wang, Shuo Yang, Junwei Liang
conference: arXiv preprint
year: 2026
tags:
  - world-action-model
  - humanoid
  - loco-manipulation
  - video-diffusion
  - whole-body-control
link: https://arxiv.org/abs/2606.09215
aliases:
  - MotionWAM
---
# Research Question
Can the rich dynamics prior of a video world model be run in real time, in a single unified action space, to drive whole-body humanoid loco-manipulation?

# Motivation
Existing World Action Models are too slow (iterative denoising) for real-time control, and dominant hierarchical humanoids split control into upper-body joint targets and coarse lower-body base commands.

# Method
A dual-DiT model conditions a Motion DiT on the *intermediate denoising features* of a Cosmos-based Video DiT in a single forward pass (one-shot imagination, no full video denoising), predicting unified whole-body "motion tokens" (built on SONIC, use rounding to fit FSQ) trained via a three-stage recipe: egocentric video pretraining → cross-embodiment action post-training → whole-body teleoperation fine-tuning on Unitree G1.

# Limitation
Same structure as psi0

# Summary
Reading off a video world model's hidden states in one pass—rather than fully denoising future frames—makes WAMs fast enough for closed-loop humanoid control (7× faster than Cosmos Policy), and a single unified motion latent lets the legs actively perform tasks (kicking, pedal-stepping) that decoupled upper-lower policies cannot, beating the strongest VLA baseline by >32% absolute on nine real-world tasks.

# Connections
- [[Sonic]] — *builds-on*: provides the universal whole-body controller and quantized motion-token latent that MotionWAM predicts and decodes into joint commands.
- DiT4DiT — *builds-on*: the dual-DiT video-motion architecture and joint flow-matching objective MotionWAM instantiates for humanoids.
- Cosmos Policy — *competes-with*: another world-model policy, but denoises the full future video (0.7 Hz) where MotionWAM reads intermediate features (4.9 Hz).
- GR00T-N1.7 / π0.5 — *is-baseline-for*: VLM-backbone VLAs finetuned on the same demos that MotionWAM outperforms on every task.
- [[PsiZero]] - *same-structure*
