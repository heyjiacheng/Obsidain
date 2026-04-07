---
title: "LeWorldModel: A Stable End-to-End JEPA for Latent World Modeling from Pixels"
authors: "Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero"
conference: arXiv
year: 2026
tags:
  - world-models
  - jepa
  - self-supervised
  - planning
  - representation-learning
link: https://arxiv.org/abs/2603.19312
aliases:
  - LeWM
  - LeWorldModel
---

# LeWorldModel

> [!abstract] TL;DR
> **LeWM** is the first [[JEPA]] world model trained stably **end-to-end from raw pixels** with only **two loss terms** — next-embedding prediction + **SIGReg** (enforces isotropic-Gaussian latents). 15M params, one GPU, one hyperparameter. Plans up to ==48× faster== than DINO-WM while staying competitive on 2D/3D control.

## Problem

Existing JEPA world models are fragile and prone to **representation collapse**: the encoder maps all inputs to a constant to trivially minimize the prediction loss. Current remedies are ugly:

- Multi-term losses (PLDM's VICReg uses 7 terms)
- EMA + stop-gradient (I-JEPA, V-JEPA)
- Frozen pretrained encoders (DINO-WM)
- Auxiliary supervision (proprioception, action decoders)

All of these introduce instability, increase hyperparameter tuning cost, or limit encoder expressivity.

> [!question] Goal
> Can we train a JEPA **end-to-end from raw pixels**, stably, with only a single hyperparameter, and small enough to run on one GPU?

## Key Idea

LeWM trains a JEPA with **only two loss terms**:

1. **Next-embedding prediction loss** — MSE in latent space.
2. **SIGReg** — a regularizer forcing latent embeddings to follow an **isotropic Gaussian**, which provably prevents collapse.

No EMA, no stop-gradient, no pretrained encoder, no reward, no reconstruction. Only **one effective hyperparameter** $\lambda$ (SIGReg weight), tunable via logarithmic bisection search.

## Method

### Architecture

- **Encoder** $\text{enc}_\theta$: ViT-tiny (~5M params, patch 14, 12 layers, hidden dim 192). The `[CLS]` token is projected through a 1-layer MLP with **[[BatchNorm]]** — critical because the ViT's final LayerNorm would constrain the latent distribution and break SIGReg.
- **Predictor** $\text{pred}_\phi$: Transformer with 6 layers, 16 heads (~10M params). Actions are injected via **[[AdaLN]]** (zero-initialized for progressive conditioning). Causal masking → autoregressive next-embedding prediction.
- **Total**: ~15M parameters, trainable on a single GPU in a few hours.

$$
\mathbf{z}_t = \text{enc}_\theta(\mathbf{o}_t), \quad \hat{\mathbf{z}}_{t+1} = \text{pred}_\phi(\mathbf{z}_t, \mathbf{a}_t)
$$

### Training Objective

$$
\mathcal{L}_{\text{LeWM}} = \underbrace{\lVert \hat{\mathbf{z}}_{t+1} - \mathbf{z}_{t+1} \rVert_2^2}_{\mathcal{L}_{\text{pred}}} + \lambda \cdot \text{SIGReg}(\mathbf{Z})
$$

> [!info] SIGReg — Sketched Isotropic Gaussian Regularizer
> - Directly testing normality in high-$d$ is intractable.
> - Project embeddings $\mathbf{Z} \in \mathbb{R}^{N \times B \times d}$ onto $M = 1024$ random unit directions $\mathbf{u}^{(m)} \in \mathbb{S}^{d-1}$.
> - Apply the univariate **Epps–Pulley normality test** $T(\cdot)$ on each 1D projection $\mathbf{h}^{(m)} = \mathbf{Z}\mathbf{u}^{(m)}$.
> - Average: $\text{SIGReg}(\mathbf{Z}) = \frac{1}{M} \sum_m T(\mathbf{h}^{(m)})$.
> - By the **Cramér–Wold theorem**, matching all 1D marginals ⇔ matching the full joint distribution.

Defaults: $M = 1024$, $\lambda = 0.1$. All components optimized end-to-end — no stop-gradient, no EMA.

### Latent Planning (Inference)

Given start $\mathbf{o}_1$ and goal $\mathbf{o}_g$:

1. Encode both into latent: $\mathbf{z}_1, \mathbf{z}_g$.
2. Roll out predictor for horizon $H$.
3. Minimize terminal latent cost $\mathcal{C}(\hat{\mathbf{z}}_H) = \lVert \hat{\mathbf{z}}_H - \mathbf{z}_g \rVert_2^2$ via **Cross-Entropy Method (CEM)**.
4. Execute first $K$ actions, then replan (**MPC**).

## Why It Works

> [!success] Core insight
> SIGReg has an **explicit target distribution** (isotropic Gaussian) backed by a proper statistical test — unlike VICReg-style hacks or EMA whose optimization target is not a well-defined objective.

- A collapsed (constant) embedding **cannot look Gaussian** → strong anti-collapse guarantee.
- The two-term loss has **well-behaved, non-competing gradients**, so training curves are smooth and monotonic (vs PLDM's noisy 7-term objective).
- **BatchNorm projection** after the ViT `[CLS]` token is critical — LayerNorm would constrain the embedding distribution and prevent SIGReg from shaping it.

## Weakness

> [!warning] Failure modes
> - **Very low-complexity environments** (Two-Room): forcing a high-$d$ isotropic Gaussian onto data with low intrinsic dimension yields a poorly structured latent space.
> - **Visually-rich 3D tasks** (OGBench-Cube): slightly behind DINO-WM — end-to-end encoder training is harder than leveraging a frozen DINOv2.
> - Requires **offline trajectories that cover the dynamics** well; purely narrow or exploratory data limits the learned world model.
> - Evaluated only on relatively simple control benchmarks — no large-scale real-robot or long-horizon demonstration.

## Experiments

**Environments:** Push-T (2D manipulation), OGBench-Cube (3D manipulation), Two-Room (2D navigation), Reacher.

**Baselines:** PLDM, DINO-WM (JEPA-based WMs); GCBC, GCIVL, GCIQL (goal-conditioned RL).

**Key results:**

| Dimension | Finding |
|---|---|
| Planning success | ==+18%== over PLDM on Push-T; beats DINO-WM (even with proprioception) on Push-T |
| Planning speed | Up to ==48× faster== than DINO-WM; full plan in <1 s |
| Hyperparameter search | $\mathcal{O}(\log n)$ bisection vs PLDM's $\mathcal{O}(n^6)$ grid |
| Ablations | Robust to $M$, embedding dim (saturates), encoder backbone (ViT ≈ ResNet-18) |
| Physical probing | Beats PLDM, competitive with DINOv2 on recovering position/velocity |
| Violation-of-expectation | Prediction error reliably spikes on physically implausible trajectories |

## My Ideas

- SIGReg is **backbone-agnostic** — could drop it into [[Octo]] / [[CrossFormer]] style large pretrained VLAs as a regularizer to reduce reliance on massive datasets?
- The Two-Room failure suggests **adapting the target distribution** to the data's intrinsic dimension (learnable covariance? lower-rank Gaussian?) rather than fixing it to isotropic.
- Could the "latent surprise" signal be used as an **intrinsic reward** for exploration, or as an **OOD detector** for safe robot deployment?
- Combine with [[Diffusion Policy]]-style action generation: LeWM as the dynamics/value model, diffusion for action sampling instead of CEM.

## Connections

- [[ACT]] — action-chunking imitation; complementary (LeWM is a dynamics/planning model, ACT is a reactive policy)
- [[Diffusion Policy]] — alternative generative policy; could use LeWM as world model
- [[Transformer]] — both encoder and predictor are transformer-based
- **JEPA lineage**: I-JEPA, V-JEPA, Brain-JEPA — rely on EMA + stop-grad; LeWM replaces them with SIGReg
- **PLDM** — closest baseline, end-to-end JEPA with VICReg; LeWM simplifies 7 loss terms → 2
- **DINO-WM** — frozen DINOv2 as encoder; LeWM beats it on speed, matches on accuracy
- **TD-MPC / Dreamer** — task-specific, reward-based WMs; LeWM is reward-free and task-agnostic
