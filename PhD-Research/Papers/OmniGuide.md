---
title: "OmniGuide: Universal Guidance Fields for Enhancing Generalist Robot Policies"
authors: "Yunzhou Song, Long Le, Yong-Hyun Park, Jie Wang, Junyao Shi, Lingjie Liu, Jiatao Gu, Eric Eaton, Dinesh Jayaraman, Kostas Daniilidis"
conference: arXiv
year: 2026
tags:
  - robotics/VLA
  - robotics/manipulation
  - diffusion/guidance
  - inference-time
  - flow-matching
link: https://arxiv.org/abs/2603.10052v1
aliases:
  - OmniGuide
---

# OmniGuide: Universal Guidance Fields for Enhancing Generalist Robot Policies


## Problem

VLA models (e.g., $\pi_{0.5}$, GR00T N1.6) generalize broadly but fail at:
- 3D collision avoidance and cluttered manipulation
- Precise semantic grounding (identifying the right object among candidates)


## Key Idea

> [!tip] Core Insight
> **Differentiable energy functions over 3D Cartesian space**, to shape the generative flow, without retraining or collecting additional data.

Any guidance source (3D geometry, VLM, human pose) defines either an **attractor** (pull end-effector toward a target) or a **repeller** (push away from obstacles). These energy gradients are injected additively into the [[flow matching]] velocity field at each denoising step.

## Method

### Background: Guided Flow Matching

A VLA policy models $p(\mathbf{A}|\mathbf{o})$ via flow matching, learning a velocity field:

$$\frac{d\mathbf{A}^\tau}{d\tau} = \mathbf{v}_\theta(\mathbf{A}^\tau, \mathbf{o}), \quad \tau \in [0,1]$$

To incorporate task constraints $\mathbf{y}$, OmniGuide modifies the velocity using Bayes' rule:

$$\mathbf{v}_\theta(\mathbf{A}^\tau, \mathbf{o} \mid \mathbf{y}) = \mathbf{v}_\theta(\mathbf{A}^\tau, \mathbf{o}) + \lambda \nabla_{\mathbf{A}^\tau} \log p(\mathbf{y}|\mathbf{A}^\tau)$$

The task condition is modeled as an **energy function** $\mathcal{L}_\mathbf{y} = -\log p(\mathbf{y}|\tilde{\mathbf{A}}^\tau)$, where $\tilde{\mathbf{A}}^\tau$ is the Tweedie-estimated clean action.

### Guidance in Cartesian Space

Since constraints live in 3D space, guidance cannot act directly on noisy joint-space actions. At each denoising step $\tau$:

1. Estimate clean action $\tilde{\mathbf{A}}^\tau$ via Tweedie's formula
2. Decode to joint space (if policy uses latent actions, e.g. GR00T)
3. Compute Cartesian trajectory $\mathbf{X} = f(\mathbf{p} \mid \tilde{\mathbf{A}}^\tau, \mathbf{s})$ via differentiable kinematics
4. Evaluate energy $\mathcal{L}_\mathbf{y}(\mathbf{X})$ in 3D
5. Backpropagate to get $\nabla_{\mathbf{A}^\tau} \mathcal{L}_\mathbf{y}$

Final denoising update:

$$\mathbf{A}^{\tau+\delta} = \mathbf{A}^\tau + \delta\Big(\mathbf{v}_\theta(\mathbf{A}^\tau,\mathbf{o}) - \lambda\,\text{clip}(\nabla_{\mathbf{A}^\tau}\mathcal{L}_\mathbf{y}(\mathbf{X}), \alpha)\Big)$$

### Three Guidance Modalities

| Modality | Energy | Source |
|---|---|---|
| **Collision avoidance** | Repulsive — $\mathcal{L}_C = -\log \mathrm{SDF}_O(\mathbf{x})$ | VGGT point cloud → occupancy grid → discrete SDF |
| **Semantic grounding** | Attractive — $\mathcal{L}_S = \|\mathbf{x} - \mathbf{x}^*\|^2 / 2\sigma_S^2$ | VLM pixel localization → depth backprojection |
| **Human imitation** | Attractive — $\mathcal{L}_H = \sum_{(\mathbf{x}_i, \mathbf{h}^*_i)\in\mathcal{M}} \|\mathbf{x}_i - \mathbf{h}^*_i\|^2 / 2\sigma_H^2$ | HaPTIC hand-tracking → DTW monotonic alignment |

**Dual guidance**: additionally, $N$ initial noise samples are evaluated with few denoising steps and the best (lowest energy) is selected as the starting point — combining prior-distribution and denoising guidance.

## Why It Works

- Flow matching (and diffusion) policies have a known velocity–score relation, allowing Bayesian posterior steering via additive gradients.
- Defining energies in Cartesian space is natural for physical constraints and allows use of any 3D foundation model.
- The base VLA prior provides diversity, naturalness, and semantic understanding; guidance shapes toward constraint satisfaction — complementary strengths.
- Guidance fields are composable: multiple energy terms sum without destructive interference because each acts on the same Cartesian trajectory.

## Weakness

> [!warning] Limitations
> - Guidance gradients are deterministic and may have local minima or incomplete contact modeling.
> - Requires differentiable forward kinematics and depth/3D reconstruction at inference time — adds computational overhead.
> - Guidance strength $\lambda$ needs tuning: too high competes with the policy prior and reduces task success.
> - Evaluated mainly on pick-and-place and articulation tasks; generalization to more complex long-horizon tasks not shown.

## Experiments

**Simulation (RoboCasa, GR00T N1.6-3B, Franka Panda):**
- Collision avoidance: safety rate 7% → 93.5%; success rate 24.2% → 92.4%
- Initialization guidance alone: +8% success, −18% collisions
- Denoising guidance alone: +20% success, −34% collisions
- Combined: +26% success, −46% collisions

**Real-world ($\pi_{0.5}$, Franka Research 3, VGGT + ZED cameras, Gemini-2.5-Flash):**
- 9 tasks across 3 guidance types
- Consistently and significantly outperforms base VLA and task-specific specialized baselines
- Guidance types: static/dynamic/reactive collision avoidance, multi-choice semantic grounding, drawer/cabinet/oven articulation via human demo

**Ablation:** Guidance composability confirmed — simultaneous collision + semantic guidance yields cumulative, non-interfering improvements.

## My Ideas

- Could apply OmniGuide to [[legged locomotion]] or [[mobile manipulation]] where collision avoidance is critical.
- The energy framework could naturally incorporate **contact-rich** objectives (e.g., normal force direction) if a differentiable contact model is available.
- Interesting to explore **adaptive** $\lambda_\tau$ schedules (guidance strength varying with denoising timestep) rather than constant $\lambda$.
- Human imitation via DTW is one-shot — could extend to few-shot averaging of multiple demos to get a more robust reference trajectory.
- The initial noise selection trick is essentially a best-of-N sampling with energy scoring — could combine with more expressive energy models (learned critics).

## Connections

- [[Inference-Time Policy Steering (ITPS)]] — closest prior work; limited to human demos + iterative MCMC; OmniGuide is more general and efficient
- [[SafeFlow]] — safety for flow matching at inference via QP; OmniGuide uses gradient-based guidance and extends to multiple objectives
- [[Classifier Guidance]] (Ho & Salimans) — same Bayesian steering idea applied to image diffusion; OmniGuide adapts to robot action space via differentiable kinematics
- [[VGGT]] — 3D reconstruction model used for point-cloud-based collision energy
- [[HaPTIC]] — hand pose estimation for human-to-robot trajectory transfer
- [[π0.5]], [[GR00T N1.6]] — base VLA policies used as backbones
