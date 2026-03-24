---
title: "DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos"
authors: Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, et al. (NVIDIA, UC Berkeley, HKUST)
conference: arXiv
year: 2026
tags:
  - world-model
  - robot-learning
  - video-generation
  - foundation-model
link: https://arxiv.org/abs/2602.06949
aliases:
  - DreamDojo
---

# DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos


## Problem

Existing [[Video World Model|video world models]] are confined to in-distribution setups because robot datasets have limited coverage (expensive teleoperation, hardware variability) and mostly consist of expert demonstrations. This makes them fail to generalize to unseen objects and novel environments, and they are typically unresponsive to counterfactual actions. Additionally, scaling robot data directly is costly and inefficient.

> [!question] Core Challenge
> How do you train a world model that generalizes to diverse, contact-rich robot tasks when labeled robot data is scarce, but unlabeled human video is abundant?

## Key Idea

Pretrain a foundation [[World Model|world model]] on **44k hours of egocentric human videos** (DreamDojo-HV), leveraging the fact that underlying physics is largely consistent between humans and robots despite the embodiment gap. Use **continuous latent actions** as unified proxy labels to recover causality from passive, unlabeled video — without needing any action annotations.

> [!tip] Insight
> Human video at internet scale provides rich physics and interaction knowledge. Latent actions bridge the gap between passive video and action-conditioned world simulation.

## Method

### Three-Phase Training Pipeline

1. **Pretraining from human videos**
   - Dataset: DreamDojo-HV (44k hours egocentric) + In-lab + EgoDex
   - 96× more skills and 2,000× more scenes than the most diverse public robot datasets
   - Continuous [[Latent Action Model]] extracts semantically meaningful actions between frames in a self-supervised manner (no action labels needed)
   - Built on top of **Cosmos-Predict2.5** (latent video diffusion, DiT architecture, flow matching loss)

2. **Post-training on target robots**
   - Reset the action conditioning layer, finetune on small-scale target robot data (e.g., GR-1, G1, AgiBot, YAM)
   - Adapts the world model to the specific robot's continuous action space

3. **Autoregressive Distillation**
   - Follows the **Self Forcing** paradigm to enable real-time autoregressive prediction
   - Improves long-horizon context consistency by modeling a short temporal context efficiently
   - Final model: **10.81 FPS at 640×480** resolution for arbitrary horizon

### Architecture

- Base: Cosmos-Predict2.5 — latent video diffusion (WAN2.2 tokenizer + DiT blocks)
- Conditions: text (cross-attention), timestep (sinusoidal + adaptive LayerNorm), and **action** (new conditioning layer)
- Flow matching objective: $\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}_{x,\epsilon,c,t} \|u(x_t, t, c; \theta) - v_t\|^2$

## Why it Works

- Physics of hand-object interactions is embodiment-agnostic; the model learns transferable contact dynamics at scale
- Latent actions recover the causal structure from passive video without any manual annotation, enabling action-conditioned generation
- Distillation reduces inference cost while improving temporal consistency for long rollouts

## Weakness

- Embodiment gap still exists — post-training on target robot is required; zero-shot transfer may not always hold for highly different morphologies
- Latent actions are proxy labels, not ground-truth robot actions; fidelity of action controllability depends on post-training data quality
- 10.81 FPS is real-time but may still be too slow for fast reactive control loops in some tasks
- Evaluation is on the authors' own benchmarks; independent OOD benchmarks would strengthen claims

## Experiments

- **OOD generalization benchmarks**: multiple challenging out-of-distribution settings for open-world, contact-rich tasks
- **Applications evaluated**:
  - *Live teleoperation*: interactive real-time video prediction driven by human operator actions
  - *Policy evaluation*: large-scale evaluation of robot policies without real-world deployment
  - *Model-based planning*: online planning using the world model as a simulator
- Compared against prior video world models; DreamDojo shows stronger physics understanding and action controllability

## My Ideas

- Could the latent action representation be used to distill an action-labeled dataset from human video — effectively creating pseudo-labeled robot demonstrations?
- How does this interact with [[Diffusion Policy]] or [[ACT]] — can DreamDojo be used as a data augmentation / rollout engine for policy training?
- Would be interesting to test whether the latent actions align with interpretable robot primitives (grasp, push, pour)
- Connection to [[GROOT]] / [[GR-1]] line of work on egocentric pretraining for robot manipulation

## Connections

- [[Cosmos-Predict2]] — base model architecture
- [[Latent Action Model]] — self-supervised action extraction from video
- [[Self Forcing]] — distillation paradigm for autoregressive video models
- [[EgoDex]] — one of the pretraining datasets used
- [[GR00T]] / [[GR-1]] — NVIDIA robot foundation model line
- [[Video World Model]] — broader category this work belongs to
- [[Model-Based RL]] — downstream use case enabled by the world model
