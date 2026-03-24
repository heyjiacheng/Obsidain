---
title: "RDT: Robotics Diffusion Transformer for Bimanual Manipulation"
authors: "Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, Jun Zhu"
conference: arXiv
year: 2024
tags:
  - robotics
  - diffusion-model
  - bimanual-manipulation
  - foundation-model
  - imitation-learning
link: "https://arxiv.org/abs/2410.07864"
aliases:
  - RDT
  - RDT-1B
---

# RDT: Robotics Diffusion Transformer for Bimanual Manipulation

## Problem

Bimanual manipulation is extremely challenging due to:

1. **Multi-modal action distributions** — two arms can accomplish the same task in many ways, making deterministic mappings produce out-of-distribution "average" actions that are physically infeasible
2. **Data scarcity** — dual-arm systems are costly, leaving <10K trajectories for any specific robot, far below what foundation models need

## Key Idea

RDT is a ==1.2B-parameter diffusion-based foundation model== for language-conditioned bimanual manipulation built on three pillars:

- A DiT backbone with targeted modifications for robot data characteristics
- A **Physically Interpretable Unified Action Space** enabling cross-robot pre-training
- Pre-train on large heterogeneous multi-robot data → fine-tune on target bimanual data

## Method

### Architecture

- **Diffusion over action chunks** — models $p(\mathbf{a}_t \mid \ell, \mathbf{o}_t)$ to capture full multi-modal distribution; avoids the "average action" failure mode of deterministic regression
- **Multi-modal encoding**: MLPs + Fourier features for proprioception/actions, SigLIP for images, T5-XXL for language (both vision/language encoders are frozen during training)
- **Stochastic independent masking** across modalities during encoding to prevent shortcut learning on dominant inputs

### Key DiT Modifications for Robot Data

| Modification                              | Problem Solved                                                                                                                            |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **QKNorm + RMSNorm** (no centering)       | Unstable numerical range of robot quantities causes gradient blow-up; LayerNorm centering destroys time-series symmetry                   |
| **MLP Decoder**                           | Replaces linear projection — captures nonlinear robot dynamics essential for dexterous tasks                                              |
| **Alternating Condition Injection (ACI)** | Alternates image/text cross-attention across layers; simultaneous injection lets image tokens dominate and degrades instruction following |

### Physically Interpretable Unified Action Space

- Maps each robot's action dimensions to a shared space **by physical meaning** (joint angles → joint angles, gripper → gripper)
- Remaining positions padded with zeros; physical semantics preserved across embodiments
- Enables pre-training on **46 datasets, 1M+ trajectories, 21TB**

### Training Pipeline

- **Pre-training**: 46 diverse robot datasets (mostly single-arm), 1M training steps on 48×H100 for ~1 month
- **Fine-tuning**: self-collected 6K+ bimanual episodes, 300+ tasks, 100+ objects, 15+ rooms; instructions rewritten by GPT-4-Turbo for text diversity
- **Inference**: DPM-Solver++ reduces diffusion steps 100→5, achieving **6 Hz** chunk frequency on onboard RTX 4090

## Why It Works

- Diffusion captures the full multi-modal action distribution; VAE ([[ACT]]) loses modes, discretization ([[OpenVLA]]) introduces quantization errors
- **Scale matters independently**: large model (1.2B vs. 93M Octo) + large data (1M+ trajectories) + large fine-tuning set all contribute — ablations confirm each factor
- Physical-meaning alignment in unified space promotes **positive transfer** across heterogeneous robots rather than negative transfer from arbitrary concatenation
- ACI prevents vision from drowning out language signals, enabling precise instruction following (e.g., "pour one-third full with the left hand")

## Weakness

- Real-robot evaluation only on ALOHA; generalization to other dual-arm hardware untested
- Pre-training is expensive (1 month on 48×H100) — high barrier to reproduce or extend
- Frozen vision/language encoders limit joint representation learning for manipulation-specific features
- Tiny fine-tuning sets for some skills (1 demo for Fold Shorts) — success may be heavily pre-training-dependent rather than systematic
- No sim-to-real evaluation; unclear what pre-training actually transfers vs. fine-tuning memorization

## Experiments

Evaluated on **7 real-robot tasks** against ACT, OpenVLA, Octo on ALOHA dual-arm:

| Task | Dimension | RDT | Best Baseline |
|---|---|---|---|
| Wash Cup | Unseen object | 50% | ~0% |
| Pour Water | Unseen scene | 62.5% | ~12% (Octo) |
| Pour Water-L/R | Instruction following | 100% correct hand, 75–100% correct amount | 0% |
| Handover | 5-shot learning | 40% | 0% |
| Fold Shorts | 1-shot learning | 68% | 4% (Octo) |
| Robot Dog | Dexterity | 48% walk-straight | 32% (ACT) |

> [!success] Ablation Summary
> All three factors are essential: **diffusion > regression** (12.5% vs. 50% unseen object), **1.2B > 166M** (37.5% vs. 50%), **pre-trained >> scratch** (0% vs. 50%).

## My Ideas

- Could the unified action space be **learned** rather than hand-designed? A learned alignment might generalize better to novel embodiments with unusual kinematic structures
- ACI is a simple heuristic — could **attention routing or modality-specific gating** do better, especially for tasks with ambiguous visual scenes?
- Frozen encoders are a bottleneck; **LoRA on SigLIP/T5** during fine-tuning might improve manipulation-specific vision-language grounding without full fine-tuning cost
- Worth testing whether physical-meaning alignment helps **sim-to-real transfer**, not just cross-robot transfer in real-world pre-training
- How does RDT compare to newer VLA approaches (**π0, OpenPI**) that also use flow matching for continuous action generation?

## Connections

- [[DiffusionPolicy]] — predecessor diffusion policy work; RDT scales and adapts DiT for multimodal robot inputs
- [[ACT]] — key bimanual baseline (VAE-based); fails on multi-modality, no instruction following
- [[Octo]] — diffusion foundation model with 93M params; RDT is its large-scale successor
- [[OpenVLA]] — 7B VLA baseline; discretization of actions hurts precision for dexterous tasks
- [[DiT]] — backbone architecture that RDT modifies for robot data (QKNorm, RMSNorm, MLP decoder, ACI)
- Open X-Embodiment dataset — primary pre-training data source
