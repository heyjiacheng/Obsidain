---
title: "SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control"
authors: Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Sirui Chen, Fernando Castañeda, Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, Xingye Da, Runyu Ding, Cyrus Hogg, Lina Song, Edy Lim, Eugene Jeong, Tairan He, Haoru Xue, Wenli Xiao, Zi Wang, Simon Yuen, Jan Kautz, Yan Chang, Umar Iqbal, Linxi "Jim" Fan, Yuke Zhu
conference: arXiv
year: 2025
tags:
  - humanoid
  - motion-tracking
  - foundation-model
  - reinforcement-learning
  - whole-body-control
link: https://arxiv.org/abs/2511.07820
aliases:
  - SONIC
---

# SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control


## Problem

Despite billion-parameter foundation models dominating perception and language, scaling laws have not been demonstrated for humanoid control. Existing neural controllers remain small (few million parameters), cover limited behaviors. There is no foundation model for humanoid whole-body control that generalizes broadly across diverse motions.

## Key Idea

Motion tracking from motion-capture data is a **natural and scalable pretraining task** for humanoid control. It provides dense supervision over diverse human motions without manual reward engineering, and learned representations transfer to downstream real-world tasks. Scaling along model capacity, data volume, and compute yields a generalist humanoid controller.

## Method

**Three axes of scaling:**

| Axis | Range |
|------|-------|
| Network size | 1.2M → 42M parameters |
| Dataset volume | 100M+ frames, 700 hours of MoCap |
| Compute | 9,000 GPU hours |

**Two deployment mechanisms:**

1. **Universal Kinematic Planner (UKP)** — real-time planner bridging motion tracking to downstream task execution, enabling natural and interactive control
2. **Unified Token Space** — single policy accepting multiple motion input modalities (VR teleoperation, human videos, VLA models) through a shared token representation

## Why it works

Dense motion-capture supervision covers the full distribution of natural human movement. Unlike sparse reward RL, every frame provides a learning signal. Scaling exposes the policy to more diverse kinematic patterns, improving generalization. The unified token space enables plug-and-play with upstream perception modules without retraining.

## Weakness

- **Sim-to-real gap**: trained in simulation; real-world deployment may face unmodeled dynamics
- **Data dependency**: relies on high-quality MoCap data, which is costly to acquire at scale
- **Pipeline depth**: UKP adds an extra component that could compound errors
- **Dynamic limits**: highly dynamic tasks (jumping, gymnastics) may still be bounded by physics sim fidelity

## Experiments

- Tracking quality across diverse motion categories at varying model/data scales
- Monotonic improvement in tracking accuracy with increased compute and data diversity
- Generalization to ==unseen motions== held out from training
- Downstream task execution via UKP (object manipulation, locomotion)
- Multi-modal control via VR teleoperation, human video retargeting, and VLA integration

## My Ideas

- Can the unified token space extend to language-conditioned motion synthesis (text → motion → control)?
- How does SONIC's representation compare to physics-based motion prior methods like [[CALM]] or [[PHC]]?
- Using SONIC as a low-level controller under a high-level task planner for [[loco-manipulation]]
- UKP architecture may be relevant for bridging learned skills with task-level planning in my own work

## Connections

- [[PHC]] — prior work on physics-based humanoid control from the same group (Zhengyi Luo)
- [[CALM]] — motion prior via adversarial imitation for physics-based characters
- [[OmniH2O]] — humanoid whole-body teleoperation, related downstream application
- [[VLA]] — vision-language-action models used as upstream command interface here
- Scaling laws in LLMs — this paper applies those intuitions to embodied control
