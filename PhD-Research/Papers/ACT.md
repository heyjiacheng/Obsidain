---
title: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
authors: "Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn"
conference: arXiv
year: 2023
tags:
  - robotics
  - imitation-learning
  - manipulation
  - transformers
  - bimanual
link: https://arxiv.org/abs/2304.13705
aliases:
  - ACT
  - ALOHA
---

# Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware



> [!question] Problem
> **Compounding errors** in imitation learning: small prediction mistakes accumulate over time, driving the robot out-of-distribution. Human demonstrations are also non-Markovian (pauses, variable handover positions), making single-step policies brittle.

## Key Idea

Two contributions:

1. **ALOHA** — a sub-$20k bimanual teleoperation system built from off-the-shelf robots, enabling fast and natural data collection at 50 Hz.
2. **ACT (Action Chunking with Transformers)** — an imitation learning algorithm that predicts *chunks* of future actions rather than single steps, dramatically reducing compounding errors and handling non-Markovian behavior.

## Method

### Algorithm: ACT

**1. Action Chunking**
Instead of predicting one action per step, the policy predicts $k$ future actions at once. This:
- Reduces effective task horizon $k$-fold → fewer compounding errors
- Handles non-Markovian behavior by treating a chunk as an atomic unit

**2. Temporal Ensembling**
The policy is queried at every timestep; overlapping chunk predictions are merged via exponential weighting:

$$w_i = \exp(-m \cdot i)$$

where $w_0$ weights the most recent prediction. Produces smooth trajectories.

**3. [[Conditional-VAE|CVAE]] Architecture**
ACT is trained as a [[Conditional-VAE|Conditional VAE]] to model stochasticity in human demonstrations:

The CVAE encoder only serves to train the CVAE decoder (the policy) and is discarded at test time.

Specifically, the CVAE encoder predicts the mean and variance of the style variable z’s distribution, which is parameterized as a diagonal Gaussian, given the current observation and action sequence as inputs.

The CVAE decoder, i.e. the policy, conditions on both z and the current observations (images + joint positions) to predict the action sequence.

- **Encoder:** BERT-style transformer — takes `[CLS]` token (mean and variance of the “style
variable” z) + joint positions + action sequence; outputs $\mu, \sigma$ of style variable $z$ (latent variable for human behavior).
- **Decoder (policy):** ResNet18 encodes 4× RGB images → 300×512 features; fused with joints and $z$ via transformer encoder; transformer decoder generates $k \times 14$ actions (14 = 7 DoF × 2 arms)
- **Loss:** L1 reconstruction + $\beta$-weighted KL to $\mathcal{N}(0, I)$
- At test time: $z$ is fixed to the prior mean (zero). Discard $z$ and encoder, human motion preference will be remember as weight of decoder.

![[Pasted image 20260321182449.png]]

> [!tip] Why CVAE matters
> On scripted data, removing CVAE has negligible effect. On **human** data, removing CVAE drops success from 35% → 2%. The CVAE absorbs the variability in human demonstrations.

## Why It Works

- **Action chunking** is the key design choice: success jumps from ~1% at $k=1$ to ~44% at $k\approx100$, confirming that reducing the effective horizon matters far more than architecture details.
- **Temporal ensembling** smooths out the discontinuities at chunk boundaries without any retraining.
- **High-frequency control (50 Hz)** is essential: reducing to 5 Hz increases task completion time by 62% in user studies.

## Weakness

- **Thread Velcro only 20% success** — low visual contrast (black cable tie, black table) makes localization from RGB alone very hard.
- **Hardware accuracy is marginal** — ViperX arms have 5–8 mm positional accuracy; the hardest tasks (Bimanual Insertion, ~5 mm clearance) still achieve only ~20% final success.
- **Parallel-jaw grippers only** — no in-hand dexterity; tasks requiring pinch grasp or finger-level manipulation (e.g., buttoning a shirt) are out of scope.
- **Perceptual difficulty** — translucent/transparent objects (ziploc, condiment cup) challenge the RGB-only pipeline.
- **No sim-to-real or generalization** — all policies are task-specific; no zero-shot transfer.


**Key ablations:**
- Chunk size $k$: monotonic improvement from $k=1$ → $k\approx100$
- Temporal ensembling: +3–4% on ACT and BC-ConvMLP
- CVAE: critical for human data, negligible on scripted data

## My Ideas

- ACT is essentially a diffusion-free generative policy — how does it compare to [[DiffusionPolicy]]?
- The CVAE style variable $z$ encodes "which mode of demonstration" — could this be used for skill discovery?
- The variable $z$ can set to other value, like some value relate to image? or just gaussian distribution.

## Connections

- [[DiffusionPolicy]] — contemporaneous work on generative imitation learning; both avoid discretization
- [[BehaviorTransformers]] (BeT) — baseline here; ACT outperforms substantially
- [[RT1]] — another baseline; trained on much larger datasets but fails here on fine tasks
- [[VINN]] — non-parametric nearest-neighbor baseline; surprisingly competitive on some tasks
- [[MobileALOHA]] — extends ALOHA to a mobile platform for household tasks


## Entire Architecture
![[Pasted image 20260322202924.png]]![[Pasted image 20260322202944.png]]