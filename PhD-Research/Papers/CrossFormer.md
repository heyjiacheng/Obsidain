---
title: "CrossFormer: A Scalable Cross-Embodiment Transformer Policy"
authors:
  - Ria Doshi
  - Homer Walke
  - Oier Mees
  - Sudeep Dasari
  - Sergey Levine
conference: CoRL
year: 2024
tags:
  - imitation-learning
  - cross-embodiment
  - transformer
  - foundation-model
link: https://arxiv.org/abs/2408.11812
aliases:
  - CrossFormer
---

# CrossFormer: A Scalable Cross-Embodiment Transformer Policy

## Problem

Training a single robot policy across diverse embodiments is hard because robots vary widely in sensors, actuators, action spaces, and control frequencies. Prior cross-embodiment methods either restrict to a single robot type (e.g., single arms) or require ==manual alignment== of observation and action spaces across robots.

## Key Idea

Cast cross-embodiment imitation learning as a ==sequence-to-sequence problem==. Tokenize all observations (images, proprioception) into a flat sequence, process them with a shared [[Transformer]] backbone, and decode variable-length actions via **embodiment-specific action heads** using readout tokens. No manual alignment of observation or action spaces is needed.

## Method

**Architecture overview:**

1. **Input tokenization** — Images are encoded with a shared ResNet-26 encoder (weights shared per camera type), proprioception is projected to the token embedding size. Observations across $k$ timesteps are serialized into a flat token sequence.
2. **Task conditioning** — Language instructions are injected via FiLM conditioning; goal images are channel-stacked with current images before encoding. The policy can accept either modality.
3. **Transformer backbone** — A 12-layer decoder-only transformer (130M params total) with block-wise causal attention. Special **readout tokens** $R$ are inserted after observation tokens at each timestep; they attend to prior observations and serve as the representation for action prediction.
4. **Action heads** — Four separate heads project readout token embeddings to actions:
   - Single-arm Cartesian (7-dim, chunk=4, 5–15 Hz)
   - Navigation waypoints (2-dim, chunk=4, 4 Hz)
   - Bimanual joint positions (14-dim, chunk=100, 20 Hz)
   - Quadruped joint positions (12-dim, chunk=1, 20 Hz)

**Training data:** 900K trajectories across 20 embodiments from Open X-Embodiment, DROID, GNM, ALOHA, and Go1 datasets. Target datasets are up-weighted during training.

**Training:** 300K steps, batch size 512, AdamW, L1 action loss, 47h on TPU V5e-256. Hindsight goal relabeling and random language/goal masking for flexible task conditioning.

## Why It Works

- The ==sequential tokenization== naturally handles variable numbers of cameras and proprioceptive inputs without architectural changes per robot.
- Readout tokens at fixed positional locations let the model use positional embeddings to infer which action head to use, avoiding ambiguity between similar-looking embodiments.
- **Action chunking** (varying chunk sizes per embodiment) reduces compounding error at high control frequencies, which is critical for fine bimanual manipulation at 20 Hz.
- Sharing image encoder weights across similar camera types maximizes visual transfer across embodiments.

## Weakness

- No significant ==positive transfer== across embodiments yet — cross-embodiment training matches but does not clearly exceed single-robot training.
- Relies on **hand-picked sampling weights** for the data mix; ideally larger models should fit all data without manual weighting.
- Inference speed becomes a bottleneck as model size scales, limiting applicability to high-frequency control (20+ Hz).
- Evaluation scale is modest (20–48 trials per setting) and the Go1/Tello evaluations have very few trials.

## Experiments

Evaluated on 6 embodiments: WidowX, Franka, ALOHA, LoCoBot, Go1, and Tello quadcopter.

| Setting | Single-Robot | Best Prior | CrossFormer |
| --- | --- | --- | --- |
| WidowX (4 tasks) | 0.42 | 0.34 | 0.40 |
| Franka (2 tasks) | 0.55 | 0.57 | 0.55 |
| ALOHA (2 tasks) | 0.50 | 0.50 | 0.70 |
| LoCoBot (3 skills) | 0.92 | 0.48 | 0.93 |
| Tello (zero-shot) | 0.68 | 0.68 | 0.82 |
| Go1 | 1.0 | N/A | 1.0 |
| **Average** | **0.68** | **0.51** | **0.73** |

Key findings:
- CrossFormer ==matches or exceeds== single-robot baselines across all settings — no negative transfer.
- Outperforms best prior methods (Octo, OpenVLA, ACT, ViNT) on average (73% vs 51%).
- Outperforms the action-aligned cross-embodiment approach from Yang et al. by 3x, showing manual alignment is unnecessary.
- Zero-shot transfer to Tello quadcopter (unseen embodiment) works via the navigation head.

## My Ideas

- Combine with [[Diffusion Policy]] action heads instead of L1 regression — could improve multimodal action distributions, especially for manipulation.
- Explore whether cross-embodiment pretraining helps with few-shot fine-tuning on a new robot (the paper doesn't test this directly).
- The readout token mechanism is similar to [[ACT]]'s approach — compare latent action spaces.

## Connections

- [[ACT]] — Action chunking transformer for bimanual manipulation; CrossFormer uses ACT's chunk size (100) for ALOHA
- [[Diffusion Policy]] — Alternative action decoding; CrossFormer uses L1 regression instead
- [[Transformer]] — Backbone architecture with block-wise causal attention
- Octo — Prior cross-embodiment policy; CrossFormer extends to more diverse embodiments
- Open X-Embodiment (RT-X) — Primary data source
