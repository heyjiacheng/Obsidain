---
title: "OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction"
authors: Lujie Yang, Xiaoyu Huang, Zhen Wu, Angjoo Kanazawa, Pieter Abbeel, Carmelo Sferrazza, C. Karen Liu, Rocky Duan, Guanya Shi
conference: arXiv
year: 2025
tags:
  - humanoid
  - motion-retargeting
  - loco-manipulation
  - reinforcement-learning
  - sim2real
link: https://arxiv.org/abs/2509.26633
aliases:
  - OmniRetarget
---

# OmniRetarget

> [!abstract] One-line
> Retarget human motion to humanoids by **preserving the interaction mesh** (agent–object–terrain spatial/contact relations) via hard-constrained optimization → clean references → RL with only **5 rewards** transfers zero-shot to a Unitree G1.

## Problem
Standard retargeting (keypoint matching with soft/unconstrained optimization) creates artifacts — **foot skating, penetration** — and ignores **human-object / human-environment interactions**. Bad references force tedious reward engineering downstream.

## Key Idea: Interaction Mesh
- Build a volumetric mesh (Delaunay tetrahedralization) over key joints + sampled **object and terrain** points (surfaces sampled densely to keep contact).
- Retarget = minimize **Laplacian deformation energy** between human (source) and robot (target) meshes → preserves *relative* spatial configuration, not absolute keypoints.
- Solve **per-frame** as a constrained nonconvex program with **hard constraints**:
  - collision avoidance (SDF), joint limits, velocity limits, **stance foot stick** (no skating), temporal smoothness.
- Solver: **Sequential SOCP/SQP**, warm-started frame-to-frame; quaternion base handled with Drake autodiff on $\mathbb{S}^3$.

## Data Augmentation (free diversity from 1 demo)
Re-solve optimization with fixed source mesh + perturbed targets:
- **Object**: pose (translate/rotate, exp-decay blend) and shape scaling. Build mesh in **object local frame** so contacts follow the object (Laplacian invariant to object rotation).
- **Terrain**: scale platform height/depth; add ground-contact points.
- Anchor lower body to nominal trajectory ($W$-weighted cost) to avoid trivial rigid-transform solutions → genuinely new upper-body coordination.

## RL with Minimal Formulation
Clean references → no reward hacking needed (follows BeyondMimic):
- **5 rewards**: body tracking, object tracking, action rate, soft joint limit, self-collision.
- **Proprioceptive only** — blind to scene/object, must follow reference precisely.
- Shared simple domain randomization, **no curriculum, no per-task tuning**.

## Results
- Retargets OMOMO (object), in-house MoCap (terrain), LAFAN1 (robot-only); **8h+** trajectories.
- Near-**zero penetration & zero foot skating** vs PHC / GMR / VideoMimic (Table II).
- Higher downstream RL success (e.g. object 82% vs ≤71%, terrain 95% vs ≤79%).
- Zero-shot sim2real on **Unitree G1**: box carrying, 0.9 m platform climb (70% robot height), slope crawl, **wall-flip** (15 rad/s, 3.5 m/s), 30 s parkour sequence (carry chair → climb → leap → roll).
- Augmented dataset trains/evals at 79% vs 82% nominal — coverage up, quality kept.

## Takeaway
> [!tip] Why it matters
> **Fix the data at the source, not the reward.** Hard-constrained interaction-mesh retargeting removes artifacts that previously demanded heavy reward engineering, and turns one demo into many embodiments/terrains/objects.

## Limitations
- Frame-by-frame (not full-trajectory) optimization; minor penetration from constraint linearization (fixed by RL).
- Future: joint trajectory optimization for noisy sources (e.g. video), visuomotor policies.

------
target: interaction mesh + hard constraints → artifact-free references → 5-reward RL → zero-shot G1
