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
What does it take to build a whole-body *native* VLA control humanoid's degrees of freedom?

- VR Control Signals? Decoupled; Joint-based; SMPL
- robot-data pretraining?
- action format?
- 1 step denoising?
- whole-body teleop? HuMI?

# Motivation
Need humanoid VLA, systematically research.

# Method
A systematic one-variable-at-a-time empirical study over three phases on the **HLM-12** benchmark: (1) controller & teleop — joint-based whole-body teleoperation (mocap retargeted online to robot joints); (2) VLA design — adapt a [[Pi05|π0.5]]-initialized backbone via weight-surgery action projection, absolute joint targets, proprioception input, multi-step flow matching; (3) heterogeneous co-training — mix in cheap stationary teleop and [[HuMI]] (humanoid analog of UMI) to extend coverage without more whole-body teleop.

# Limitation
If a VLA can't run on a humanoid, nor video-action model. Because video-action model can't solve following question:

humanoid has higher dimension of DoF than dual arm.
pi0.5 tuned VLM perform better than PaliGemma-initialized (pure VLM, no robot data), pi0.5 has manipulation prior: see error, retry.

HuMI half operate-time, but semantic supervision rather than motion supervision (human motion, not robot motion, even after IK)



# Summary
Getting the *design details* right — teleop interface, VLA adaptation, and cheap-data co-training — matters more than scaling humanoid data or model size: a π0.5-backbone trained with **zero** humanoid pretraining data beats GR00T N1.6 and Ψ0 (which both include humanoid data in pretraining) on a long-horizon task at less than half the demonstration time, and action MSE is a poor proxy for real-world task progress.

# Connections
- [[Sonic]] — *builds-on*: SONIC motion-tracking controller is used as the low-level whole-body tracker for joint-based teleop.
- [[Pi05|π0.5]] — *builds-on / is-baseline-for*: the VLA backbone; its non-humanoid robot pretraining transfers surprisingly well to the humanoid's full action space.
- GR00T N1.6 & Ψ0 — *competes-with*: state-of-the-art humanoid VLAs that decouple control and add humanoid data to pretraining; OpenHLM outperforms both ([[Active perception on psi0|Ψ0 idea]]).
