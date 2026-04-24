---
title: "FAST: Efficient Action Tokenization for Vision-Language-Action Models"
authors: "Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, Sergey Levine"
conference: arXiv
year: 2025
tags:
  - robotics
  - vla
  - tokenization
  - manipulation
  - autoregressive
link: https://arxiv.org/abs/2501.09747
aliases:
  - FAST
  - π0-FAST
  - FAST+
---

# FAST: Efficient Action Tokenization for Vision-Language-Action Models

## Problem 1
Autoregressive VLAs discretize continuous actions via **per-dimension, per-timestep binning** (e.g., [[OpenVLA]], RT-2). On high-frequency data (20–50 Hz), this fails completely: adjacent tokens become nearly identical, so the marginal information of token $T_i$ given $T_{1:i-1}$ approaches zero. The model collapses to "copy the previous token" and never fits the underlying signal.

## Solution 1
**Compress actions before tokenization** so each token carries high marginal information. FAST applies the **Discrete Cosine Transform (DCT)** to each action dimension, which concentrates energy in a few low-frequency coefficients.

Objective: learn $\pi(a_{1:H}|o)$ via next-token prediction over a compressed token sequence
$$\mathcal{T}_a: a_{1:H} \rightarrow [T_1, \dots, T_n], \quad n \ll H \cdot |\mathcal{A}|$$
with tokens chosen so $H(T_i \mid T_{1:i-1})$ stays high across control frequencies.

## Problem 2
A raw DCT + quantization matrix is mostly zeros — still too many tokens to decode autoregressively, and lots of redundant structure across dimensions.

## Solution 2
**Lossless compression on top of DCT** via [[BPE]] (byte-pair encoding): flatten the sparse, quantized DCT matrix and learn merges of frequent integer patterns. This squashes zero-runs and yields a fixed-size, VLM-compatible vocabulary. Result: ~30 tokens per 1-second chunk per arm, **independent of control frequency**.

## Problem 3
BPE must be retrained per dataset, adding friction when deploying to a new robot.

## Solution 3
Train **FAST+**, a universal tokenizer, on ~1M 1-second action chunks spanning single-arm / bi-manual / mobile robots and joint / end-effector / camera-frame action spaces (actions zero-padded to 32 dims). Used as a black-box `AutoProcessor` — matches dataset-specific tokenizers in downstream policy performance.

## Method

### Pipeline

1. **Quantile normalization** — map 1st/99th percentile of each action dim to $[-1, 1]$ (robust to outliers, cross-embodiment friendly).
2. **Per-dimension DCT** — convert each $a^i_{1:H}$ to frequency-space coefficients $C^i_j$.
3. **Scale & round** — $\bar{C}^i_j = \text{round}(\gamma \cdot C^i_j)$; hyperparameter $\gamma$ trades lossiness vs. compression (default $\gamma=10$).
4. **Column-first flatten** — interleave dimensions by frequency: *lowest-frequency components of all dims first*. This makes autoregressive prediction output the coarse shape early, which stabilizes rollouts.
5. **BPE compress** — train a BPE tokenizer (vocab size 1024) on flattened integer streams. Overwrite the least-used tokens in the VLM vocabulary with these action tokens.

### Two hyperparameters only
- DCT rounding scale $\gamma$
- BPE vocabulary size

Both are insensitive across datasets — unlike [[VQ-VAE]]/FSQ alternatives that need careful per-dataset tuning.

### Compression results (1-sec chunks, comparable reconstruction error)

| Dataset | Dim | Hz | Naïve tokens | FAST tokens | Compression |
| --- | --- | --- | --- | --- | --- |
| BridgeV2 | 7 | 5 | 35 | 20 | 1.75× |
| DROID | 7 | 15 | 105 | 29 | 3.6× |
| Bussing | 7 | 20 | 140 | 28 | 5.0× |
| Shirt Fold | 14 | 50 | 700 | 53 | **13.2×** |

## Why it works

- **Information-theoretic view**: next-token loss $\propto H(T_i \mid T_{1:i-1})$. DCT decorrelates neighboring samples of a smooth signal, and BPE removes residual structural redundancy — so *every* token carries near-uniform information. The learning signal no longer vanishes at high frequency.
- **Analogy to JPEG**: smooth images concentrate energy in low-frequency DCT coefficients. Robot action chunks are likewise smooth time-series — the same compression principle transfers directly.
- **Column-first flattening** mirrors coarse-to-fine generation: the autoregressive model commits to the overall trajectory shape before filling in high-frequency detail, avoiding compounding drift on long chunks.
- **Simplicity wins over learned VQ**: DCT is analytical, differentiable-free, and has no reconstruction-vs-commitment hyperparameters. Against FSQ (an FSQ-VAE compression baseline), FAST matches or beats it on dexterous tasks despite being parameter-free.

## Weakness

- **Inference latency**: 30–60 autoregressive decoding steps on the full 2B-param backbone → **~750 ms/chunk** vs. ~100 ms for diffusion [[Pi0]] (10 flow-matching steps through a 300M action expert). Precludes dynamic/reactive tasks.
- **Tested only on static manipulators** — mobile, dexterous-hand, humanoid policy performance is not yet validated (only tokenizer compression is).
- **Lossy at aggressive $\gamma$** — very dynamic signals (fine contact events) may lose information if high-frequency DCT coefficients are rounded away.
- **BPE merges are dataset-statistical** — FAST+ works well zero-shot, but pathological action distributions (e.g., extreme impulsive signals) could degrade compression.
- **Not a fundamentally new decoding paradigm** — still bottlenecked by autoregressive sampling; speculative decoding / kernel optimizations left to future work.

## Experiments

- **Case study (cubic-spline toy)**: naïve binning MSE explodes as sampling rate 25 → 800; DCT-tokenized model stays flat.
- **7 eval tasks**: Libero (sim), Table Bussing (20 Hz UR5), T-Shirt Folding (50 Hz ARX bi-manual), Grocery Bagging, Toast-from-Toaster, **Laundry Folding** (50 Hz ARX), **DROID zero-shot** tabletop (15 Hz, 3 campuses, unseen envs).
- **Result**: naïve tokenization makes zero progress on Bussing/T-Shirt; FAST and FSQ both work, FAST > FSQ on dexterous tasks; FAST+ ≈ dataset-specific FAST.
- **Backbone-agnostic**: swapping FAST into [[OpenVLA]] (7B Prismatic) also rescues high-frequency training.
- **Ablation**: removing BPE hurts (but still beats naïve) — DCT carries most of the win; BPE mainly cuts token count.
- **vs. diffusion [[Pi0]]**: comparable final success; 3× fewer steps to converge on Table Bussing; **5× less GPU time overall** when scaled to 10k hours / 903M timesteps.
- **First zero-shot DROID generalist** — language-conditioned, evaluated in fully unseen environments without co-training/fine-tuning.

## My Ideas

Replace DCT with a **learned orthogonal transform conditioned on embodiment metadata** — e.g., a small per-robot linear basis learned jointly with BPE to better concentrate energy for non-smooth signals (contact-rich, humanoid balance). Keep the analytical invertibility of DCT as a prior/regularizer, but allow the basis to adapt where JPEG-style smoothness assumptions break.

## Connections

- [[Pi0]] — the diffusion VLA this paper competes against and is grafted into (π0-FAST).
- [[OpenVLA]] — prior autoregressive VLA with naïve binning; FAST fixes its high-frequency failure mode.
- [[ACT]] — popularized action chunking; FAST is the tokenization layer that makes chunking work for autoregressive models.
- [[RT-X]] / OXE — data sources mixed into FAST+'s 1M-chunk training set.
- [[CrossFormer]] — alternative cross-embodiment policy architecture; complementary to FAST's cross-embodiment tokenizer.
- JPEG / DCT compression — the direct algorithmic ancestor.
- [[BPE]] — the language-side compression inspiration.
