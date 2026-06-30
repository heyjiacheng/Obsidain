---
title: "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"
authors: Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Remi Cadene, et al. (Hugging Face)
conference: arXiv preprint
year: 2025
tags:
  - VLA
  - robotics
  - imitation-learning
  - flow-matching
  - efficient-models
link: https://arxiv.org/abs/2506.01844
aliases:
  - SmolVLA
---

# SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics

## Problem N → Solution N
1. **VLAs are huge (billions of params), costly to train/deploy** → a compact 450M VLA trainable on one GPU, runnable on CPU.
2. **VLAs depend on costly proprietary/academic datasets** → pretrain only on <30k community-contributed episodes from Hugging Face.
3. **Synchronous chunked inference leaves the robot idle during prediction** → async stack decoupling execution from prediction.

Action expert trained with conditional flow matching:
$$\mathcal{L}^{\tau}(\theta)=\mathbb{E}\left[\left\|\mathbf{v}_{\theta}(\mathbf{A}_{t}^{\tau},\mathbf{o}_{t})-(\epsilon-\mathbf{A}_{t})\right\|^{2}\right],\quad \mathbf{A}_{t}^{\tau}=\tau\mathbf{A}_{t}+(1-\tau)\epsilon$$

## Method
A frozen [[SmolVLM]]-2 VLM (SigLIP + SmolLM2) encodes images (64 tokens/frame, no tiling), language, and a state token; only the **first 16 of L layers** are used ($N=L/2$). Layer-$N$ features condition a flow-matching action expert (hidden size $0.75\times d$) that **interleaves cross-attention** (to VLM keys/values) **and causal self-attention** blocks, emitting a chunk of 50 actions in 10 flow steps.

## Why it works
Intermediate VLM layers already carry the features useful for control, so half the backbone is skippable for free; interleaved self-attention smooths action chunks, and async inference hides server latency behind a partially-drained action queue (threshold $g$).

## Weakness
- Relies on **manual** camera-view normalization and VLM-relabeled noisy community annotations.
- Real-world eval uses synchronous inference → still open-loop lag between chunks.

## Experiments
0.45B SmolVLA hits **LIBERO 87.3%**, **Meta-World 57.3%**, beating Octo/OpenVLA(7B) and matching π₀(3.3B) while being ~40% faster to train and 6× less memory; real-world SO100 **78.3%** avg vs π₀ 61.7%.

## My Ideas
Replace manual camera-view mapping with a VLM that auto-assigns standardized view types, enabling fully unsupervised ingestion of new community datasets.

## Connections
- [[Sonic]]
- Flow-matching VLAs: π₀; layer-skipping for efficient backbones.

------
