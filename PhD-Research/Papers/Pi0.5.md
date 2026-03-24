---
title: "π0.5: a Vision-Language-Action Model with Open-World Generalization"
authors: "Physical Intelligence (Kevin Black, Noah Brown, Chelsea Finn, Sergey Levine, et al.)"
conference: arXiv
year: 2025
tags:
  - robotics/VLA
  - robotics/manipulation
  - robotics/generalization
  - ml/co-training
link: https://arxiv.org/abs/2504.16054
aliases:
  - pi0.5
  - π0.5
---

# π0.5: a Vision-Language-Action Model with Open-World Generalization

## Problem

为什么改进kv  两个transformer  

VLA models achieve strong results in training environments but fail to generalize to entirely new real-world settings. It remains an open question how to build robotic systems that can perform long-horizon, dexterous manipulation tasks (e.g., cleaning a kitchen or bedroom) in homes never seen during training.

## Key Idea

Co-train a single VLA model on **heterogeneous data sources** — other robot embodiments, web data, high-level semantic prediction, and verbal language supervision — to enable broad open-world generalization, even when target-domain data (mobile manipulators in homes) is scarce.

> [!info] Data mixture
> Only **2.4%** of pre-training examples come from mobile manipulators doing household tasks. The rest: other robots (ME, CE) + web data (WD) + high-level subtask labels (HL).

## Method

### Architecture

A unified transformer model that outputs both **discrete text tokens** and **continuous flow-matched actions**:

$$
\pi_{\theta}(\mathbf{a}_{t:t+H}, \hat{\ell} \mid \mathbf{o}_t, \ell) = \pi_{\theta}(\mathbf{a}_{t:t+H} \mid \mathbf{o}_t, \hat{\ell})\, \pi_{\theta}(\hat{\ell} \mid \mathbf{o}_t, \ell)
$$

- $\hat{\ell}$: predicted high-level subtask (e.g., "pick up the plate") or VQA answer
- $\mathbf{a}_{t:t+H}$: low-level action chunk from flow matching
- Same model handles **both** high-level and low-level inference (chain-of-thought style)

**Action representation**: hybrid FAST tokenizer (discrete, efficient pre-training) + flow matching action expert (continuous, fast inference). Combined loss:

$$
\mathcal{L} = H(x_{1:M}, f^\ell_\theta) + \alpha \|\omega - \mathbf{a}_{t:t+H} - f^a_\theta(\mathbf{a}^{\tau,\omega}, \mathbf{o}_t, \ell)\|^2
$$

### Two-Stage Training

| Stage | Steps | Data | Action Head |
|-------|-------|------|-------------|
| Pre-training | 280k | MM + ME + CE + HL + WD | FAST discrete tokens only ($\alpha=0$) |
| Post-training | 80k | MM + ME + HL + WD + VI | Discrete + flow matching action expert ($\alpha=10$) |

### Data Sources

| Code | Description |
|------|-------------|
| **MM** | ~400h mobile manipulator data in ~100 home environments |
| **ME** | Non-mobile robots in diverse home environments |
| **CE** | Cross-embodiment lab data (tabletop, OXE dataset) |
| **HL** | Subtask prediction labels: "clean bedroom" → "pick up pillow" + bounding boxes |
| **WD** | Web data: image captioning, VQA, object localization |
| **VI** | Verbal instruction demos (post-training): expert users "teleoperate" via language commands |

### Inference

At each timestep:
1. **High-level**: predict semantic subtask $\hat{\ell}$ from high-level command + observation
2. **Low-level**: predict action chunk $\mathbf{a}_{t:t+H}$ conditioned on $\hat{\ell}$ via 10-step flow matching denoising at 50 Hz

**Robot**: two mobile manipulator platforms, 4 cameras (forward, backward, 2 wrists), dual 6-DoF arms, holonomic base, torso lift — 18–19 DoF total.

## Why it works

- **Cross-embodiment transfer**: ME and CE data teach manipulation skills that transfer to mobile robots in new scenes
- **Web data**: broadens semantic and object understanding, critical for OOD object generalization and high-level reasoning
- **HL data**: subtask prediction training improves performance even without explicit runtime inference ("implicit HL")
- **Verbal instructions**: small VI dataset (~11% of high-level examples) is critical for high-level subtask inference quality
- **Hybrid discrete/continuous training**: FAST tokens accelerate pre-training, flow matching enables precise real-time control

## Weakness

- Still fails on unfamiliar handles, physically hard-to-open cabinets, arm occlusion (partial observability)
- High-level inference can get distracted (e.g., opening/closing a drawer repeatedly)
- Prompt complexity limited by training data annotations
- Shallow context/memory — no cross-room navigation or object persistence
- Co-training recipe is hand-designed; principled mixture optimization unexplored

## Experiments

- **Real-home eval**: 3 unseen homes, kitchen + bedroom cleaning tasks (2–5 min), high success rates
- **Scaling**: performance improves with more training environments (3 → 104 locations); 104-location model matches a model trained *on* test homes
- **Ablations** (mock homes + language following):
  - Removing ME or CE significantly hurts performance
  - Removing WD hurts OOD object generalization and high-level reasoning
  - Removing VI significantly degrades high-level subtask prediction
- **Baselines**: outperforms [[PiZero|π0]], π0-FAST+Flow (even at 300k steps), and GPT-4 zero-shot as high-level planner

> [!success] Key result
> First end-to-end learned system to perform long-horizon dexterous manipulation (10–15 min tasks) in entirely new homes.

## My Ideas

- HL data via synthetic annotation (VLM-generated subtask labels) could scale labeling cheaply
- Could verbal instruction data be replaced with LLM-generated subtask sequences + low-level rollouts?
- Memory/context extension (e.g., retrieved object locations across rooms) seems like a natural next step
- Mixture optimization (e.g., data scheduler, curriculum) could improve sample efficiency

## Connections

- [[pi0.5]] — predecessor VLA using flow matching for dexterous manipulation
- [[OpenVLA]] — another VLA baseline for comparison
- FAST tokenizer — efficient action tokenization enabling fast VLA pre-training
- Chain-of-thought / test-time compute — analogous to the high-level → low-level inference decomposition
- [[OpenX-Embodiment|OXE dataset]] — cross-embodiment robot data used in CE split
