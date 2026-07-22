---
title: "VLK: Learning Humanoid Loco-Manipulation from Synthetic Interactions in Reconstructed Scenes"
authors: Yen-Jen Wang, Jiaman Li, Sirui Chen, Takara E. Truong, Pei Xu, Pieter Abbeel, Rocky Duan, Koushil Sreenath, Angjoo Kanazawa, Carmelo Sferrazza, Guanya Shi, C. Karen Liu
conference: arXiv
year: 2026
tags:
  - humanoid
  - loco-manipulation
  - vision-language-action
  - synthetic-data
  - sim-to-real
link: https://arxiv.org/abs/2606.30645
aliases:
  - VLK
---
# Research Question
Can a humanoid learn perception-based loco-manipulation from synthetic paired vision-language-kinematics (VLK) data generated in reconstructed scenes?

# Motivation
No existing data source provides synchronized egocentric images, language instructions, and robot-compatible whole-body trajectories at scale.

# Method
Reconstruct real scenes with 3D Gaussian Splatting, synthesize 48k G1 interaction trajectories with conditional diffusion using privileged scene information, render egocentric views afterward, and train a π0.5-based policy predicting 1-second kinematic chunks that a contact-aware whole-body tracker executes on the real Unitree G1.

# Limitation
- need to annotate semantic 3D bounding boxes for objects and mark walkable regions
- lower body still using x, y, rotation, not using SONIC like whole-body controller
- only one label means hand contact

# Summary
Decoupling perception (kinematic prediction from synthetic VLK data) from control (a blind whole-body tracker) makes scene-grounded synthetic supervision alone sufficient for sim-to-real humanoid navigation and box transport.

# Connections
- CHOIS (Li et al. 2024) — builds-on: adapts its conditional diffusion human-object interaction synthesis from SMPL to the G1 representation.
- OmniRetarget (Yang et al. 2025) — builds-on: used to retarget OMOMO human-object motions to the G1 for training the interaction synthesis model.
- WholeBodyVLA / Ψ0 (Jiang et al. 2025; Wei et al. 2026) — competes-with: humanoid loco-manipulation VLAs trained on real teleoperation and egocentric video rather than synthetic supervision.
