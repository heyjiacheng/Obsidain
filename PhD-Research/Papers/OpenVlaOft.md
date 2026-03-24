---
title: "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Performance for Robotics"
authors:
  - Moo Jin Kim
  - Chelsea Finn
  - Percy Liang
conference: ICRA
year: 2025
tags:
  - VLA
  - fine-tuning
  - robot-manipulation
  - action-chunking
  - parallel-decoding
link: https://arxiv.org/abs/2502.19645
aliases:
  - OpenVLA-OFT
---

# Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Performance for Robotics

## Problem

Vision-language-action models (VLAs) require fine-tuning to work on new robot setups, but the best fine-tuning strategy is unclear. The default autoregressive approach inherited from VLM pretraining is too slow for high-frequency control (only 3–5 Hz) and underperforms on dexterous bimanual manipulation tasks. Practitioners face a large, underexplored design space of action decoding schemes, action representations, and learning objectives.

## Key Idea

Replace the standard autoregressive, discrete-token, next-token-prediction fine-tuning recipe with an **Optimized Fine-Tuning (OFT)** recipe that combines three design choices: (1) **parallel decoding with action chunking**, (2) **continuous action representations**, and (3) an **L1 regression objective**. Optionally, **FiLM** (feature-wise linear modulation) is added to inject language embeddings into visual features for better language grounding.

## Method

Starting from OpenVLA (a 7B-parameter VLA built on the Prismatic VLM), OFT modifies fine-tuning in three ways:

- **Parallel decoding + action chunking**: Replace causal attention with bidirectional attention so that all action tokens are predicted in a single forward pass instead of sequentially. Empty action embeddings (differing only in positional encoding) serve as decoder inputs. Extending to action chunks of $K$ timesteps simply means inserting $K \times D$ empty embeddings, yielding $K$-fold throughput gain with minimal latency increase.
- **Continuous action representation**: Replace the 256-bin discretization + softmax head with a 4-layer MLP action head that outputs continuous action values, preserving fine-grained action precision. 为什么连续
- **L1 regression objective**: Train with mean L1 loss between predicted and ground-truth actions. This matches diffusion-based objectives in performance while converging faster and requiring only one forward pass at inference (vs. 50 denoising steps for diffusion).
- **FiLM conditioning (OFT+ variant)**: Project the mean language embedding into scale ($\gamma$) and shift ($\beta$) vectors that modulate visual patch features across all ViT blocks, preventing the policy from ignoring language instructions due to spurious visual correlations.

Additional inputs (wrist camera images, proprioceptive state) are supported by processing them through shared encoders and concatenating embeddings along the sequence dimension.

## Why It Works

- Parallel decoding eliminates the $D$-fold sequential bottleneck, and action chunking captures temporal dependencies while reducing compounding errors.
- Continuous actions avoid information loss from discretization, improving precision.
- L1 regression is sufficient because the high-capacity 7B model can represent the action distribution without needing the expressiveness of diffusion; it also acts as a natural smoother over demonstration noise by learning the median mode.
- FiLM forces the vision backbone to attend to language features at every layer, preventing shortcut learning through spurious visual correlations in multi-camera setups.

## Weakness

- **Multimodal demonstrations**: L1 regression learns the median action and may fail when truly multimodal action distributions exist (multiple valid strategies for the same observation).
- **Pretraining applicability unknown**: OFT is validated only for fine-tuning; whether it benefits large-scale pretraining remains open.
- **Language grounding is fragile**: Without FiLM, language following drops to chance level (33%) on ALOHA tasks, despite working fine in LIBERO—the root cause (pretraining data gap vs. multi-camera spurious correlations) is unclear.
- **Inference speed**: OpenVLA-OFT+ (7.5B params) still lags behind smaller models like ACT (84M) and $\pi_0$ (3.3B, JAX-optimized) in raw throughput, though it is practically sufficient at 77.9 Hz.

## Experiments

**LIBERO simulation benchmark** (4 task suites, 10 tasks each):
- OpenVLA-OFT achieves **97.1%** average success rate (vs. 76.5% for vanilla OpenVLA, 94.2% for $\pi_0$).
- **26×** action generation speedup with $K=8$ action chunks.
- Each design choice contributes incrementally: PD+AC → 90.2%, +continuous → 95.3%, +additional inputs → 97.1%.

**Real-world ALOHA bimanual robot** (4 dexterous tasks at 25 Hz):
- OpenVLA-OFT+ outperforms $\pi_0$, RDT-1B, Diffusion Policy, and ACT by up to **15% absolute** in average success rate.
- Achieves **43×** throughput improvement over base OpenVLA with $K=25$ chunks.
- Succeeds despite base OpenVLA never seeing bimanual data during pretraining, demonstrating that fine-tuning recipe matters more than pretraining data coverage.

## My Ideas

- The finding that L1 regression matches diffusion for fine-tuning is powerful—could this extend to pretraining as well if the dataset is large and diverse enough?
- FiLM's necessity only on real multi-camera setups (not LIBERO) suggests investigating what visual shortcuts the model learns and whether data augmentation could substitute for architectural changes.
- The parallel decoding approach is model-agnostic; applying OFT to newer VLAs (e.g., $\pi_0$, RT-2-X) could yield further gains.

## Connections

- [[ActionChunking]] — ACT introduced action chunking for manipulation; OFT extends it to VLA fine-tuning with parallel decoding.
- [[DiffusionPolicy]] — Diffusion-based action generation is a key baseline; OFT shows L1 regression can match it with simpler training.
- [[OpenVLA]] — Base model that OFT builds upon.
- [[FiLM-Conditioning|FiLM]] — Feature-wise linear modulation used for language grounding in the OFT+ variant.
- [[Pi0]] — Flow-matching VLA that is the strongest baseline in both LIBERO and ALOHA experiments.
