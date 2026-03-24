---
title: Humanoid Teleoperation Dataset
date: 2026-03-13
tags:
  - idea
  - humanoid
  - teleoperation
  - imitation-learning
  - robot-learning
status: seedling
aliases:
  - VR Teleop Dataset
---

# Idea: Humanoid Teleoperation Dataset

## Problem
Humanoid manipulation datasets are small and lab-constrained — most existing datasets (e.g., RoboSet, DROID) use single-arm setups and controlled environments. Humanoid bimanual data in diverse, in-the-wild settings barely exists.

## Idea
Collect large-scale VR teleoperation demonstrations on a humanoid robot across diverse household/lab environments. Operator wears a VR headset + hand controllers; motions retargeted to humanoid in real time. Capture RGB-D from wrist + head cameras, proprioception, and action streams.

## Why Interesting
- Enables imitation learning ([[Diffusion Policy]], [[ACT]]) directly on humanoids without manual reward engineering.
- In-the-wild diversity could unlock zero-shot or few-shot generalization similar to what [[PointWorld]] achieves with sim data.
- Human demonstrations carry implicit dexterity priors — much richer signal than scripted policies.

## Related Papers
- [[Diffusion Policy]]
- [[RT-2]]
- [[ACT]]
- [[PointWorld]]

## Experiments
- [ ] Collect 100 demos across 5 task categories
- [ ] Measure teleoperation latency and operator fatigue
- [ ] Train BC baseline, compare to sim-pretrained checkpoint
- [ ] Ablate dataset size: 10 / 50 / 100 / 500 demos

## Risks
- Teleoperation latency degrades data quality (aim for <50ms round-trip).
- Retargeting from human hand to humanoid hand is lossy — grasps may not transfer cleanly.
- Dataset scale may still be too small compared to sim data.

#humanoid #teleoperation #imitation-learning #robot-learning
