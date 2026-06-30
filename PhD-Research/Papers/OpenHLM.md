---
title: "OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation"
authors: Yingdong Hu, Haodong Zhu, Boyuan Zheng, Yihang Hu, Tong Zhang, Zunhao Chen, Junming Zhao, Ruiqian Nai, Yang Gao
conference: arXiv preprint
year: 2026
tags:
  - humanoid
  - VLA
  - loco-manipulation
  - whole-body-control
  - co-training
link: https://arxiv.org/abs/2606.22174
aliases:
  - OpenHLM
---
# Research Question
What does it take to build a whole-body *native* VLA that maps language and pixels directly to **all** of a humanoid's degrees of freedom, instead of decoupling the upper and lower body into separate controllers?

# Motivation
Most humanoid systems drive arms with IK and legs with a separate RL controller, reducing the robot to a wheeled dual-arm platform and excluding behaviors that recruit the lower body as a manipulator (e.g. squatting to a low shelf, pressing a pedal with the foot).

# Method
A systematic one-variable-at-a-time empirical study over three phases on the **HLM-12** benchmark: (1) controller & teleop — joint-based whole-body teleoperation (mocap retargeted online to robot joints, 0.2 s preview latency); (2) VLA design — adapt a [[Pi05|π0.5]]-initialized backbone via weight-surgery action projection, absolute joint targets, proprioception input, multi-step flow matching; (3) heterogeneous co-training — mix in cheap stationary teleop and [[HuMI]] (humanoid analog of UMI) to extend coverage without more whole-body teleop.

# Limitation

# Summary
Getting the *design details* right — teleop interface, VLA adaptation, and cheap-data co-training — matters more than scaling humanoid data or model size: a π0.5-backbone trained with **zero** humanoid pretraining data beats GR00T N1.6 and Ψ0 (which both include humanoid data in pretraining) on a long-horizon task at less than half the demonstration time, and action MSE is a poor proxy for real-world task progress.

# Connections
- [[Sonic]] — *builds-on*: SONIC motion-tracking controller is used as the low-level whole-body tracker for joint-based teleop.
- [[Pi05|π0.5]] — *builds-on / is-baseline-for*: the VLA backbone; its non-humanoid robot pretraining transfers surprisingly well to the humanoid's full action space.
- GR00T N1.6 & Ψ0 — *competes-with*: state-of-the-art humanoid VLAs that decouple control and add humanoid data to pretraining; OpenHLM outperforms both ([[Active perception on psi0|Ψ0 idea]]).
