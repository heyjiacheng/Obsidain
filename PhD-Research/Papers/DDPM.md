---
title: Denoising Diffusion Probabilistic Models
authors:
  - Jonathan Ho
  - Ajay Jain
  - Pieter Abbeel
conference: NeurIPS
year: 2020
tags:
  - diffusion-models
  - generative-models
  - score-matching
  - image-synthesis
link: https://arxiv.org/abs/2006.11239
aliases:
  - DDPM
---

# Denoising Diffusion Probabilistic Models

## Problem

Prior to this work, diffusion probabilistic models were theoretically well-motivated but had not demonstrated competitive sample quality compared to GANs, VAEs, and autoregressive models. There was no evidence that diffusion models could generate high-quality images.

## Key Idea

Train a diffusion model by learning to reverse a fixed Markov chain that gradually adds Gaussian noise to data. The paper establishes a novel connection between diffusion models and [[Denoising Score Matching]] with [[Langevin Dynamics]], which leads to a simplified training objective: predict the noise $\epsilon$ added at each timestep rather than predicting the mean $\tilde{\mu}$ directly.

The simplified loss is:

$$L_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]$$

## Method

**Forward process (diffusion):** A fixed Markov chain that gradually adds Gaussian noise to data over $T = 1000$ timesteps according to a variance schedule $\beta_1, \dots, \beta_T$:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

A key property is that sampling $\mathbf{x}_t$ at arbitrary $t$ is available in closed form:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

**Reverse process (denoising):** A learned Markov chain parameterized by a neural network that iteratively denoises $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ back to a clean sample $\mathbf{x}_0$:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})$$

**Training (Algorithm 1):**
1. Sample $\mathbf{x}_0 \sim q(\mathbf{x}_0)$, $t \sim \text{Uniform}(\{1, \dots, T\})$, $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. Gradient descent on $\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2$

**Sampling (Algorithm 2):**
1. Start from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. For $t = T, \dots, 1$: compute $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$

**Architecture:** U-Net backbone similar to PixelCNN++ with group normalization, sinusoidal position embeddings for timestep conditioning, and self-attention at 16x16 resolution.

## Why it works

The $\epsilon$-prediction parameterization serves a dual purpose:

1. It simplifies the variational bound into a reweighted denoising score matching objective that down-weights easy (low-noise) denoising tasks, letting the network focus on harder high-noise cases.
2. The sampling procedure with $\epsilon$-prediction resembles [[Langevin Dynamics]] with $\epsilon_\theta$ acting as a learned gradient of the data density, connecting diffusion models to score-based generative modeling.

The simplified objective $L_{\text{simple}}$ discards the per-timestep weighting from the true variational bound, which empirically improves sample quality at the cost of slightly worse log-likelihoods.

## Weakness

- **Slow sampling:** Requires $T = 1000$ sequential denoising steps (300 seconds for a batch of 128 images at 256x256).
- **Log-likelihood:** Not competitive with likelihood-based models (autoregressive, flows). Most of the lossless codelength describes imperceptible image details.
- **Unconditional only:** No class-conditional or text-conditional generation explored.
- **Fixed variance schedule:** The forward process variances $\beta_t$ are fixed constants, not learned.

## Experiments

- **CIFAR-10 (unconditional):** IS = 9.46, FID = **3.17** (state-of-the-art among unconditional models, competitive with conditional GANs like StyleGAN2+ADA at FID 2.67).
- **LSUN 256x256:** Bedroom FID = 4.90, Church FID = 7.89. Quality comparable to ProgressiveGAN.
- **CelebA-HQ 256x256:** High-quality face generation demonstrated.
- **Ablation (Table 2):** $\epsilon$-prediction with $L_{\text{simple}}$ significantly outperforms $\tilde{\mu}$-prediction; learning reverse process variances leads to instability.
- **Progressive generation:** The reverse process naturally produces coarse-to-fine generation — large-scale structure appears first, details emerge last — interpretable as a generalization of autoregressive decoding.
- **Interpolation:** Latent space interpolation at $t = 500$ diffusion steps produces smooth, semantically meaningful transitions between images.

## My Ideas

- The connection to score matching suggests combining with faster sampling methods (e.g., reducing the number of steps via non-Markovian processes — later realized by [[DDIM]]).
- The progressive coarse-to-fine generation property could be exploited for hierarchical editing or inpainting at different semantic levels.
- The framework is flexible enough to incorporate conditioning signals, which opened the door for text-to-image models like [[DALL-E 2]] and [[Stable Diffusion]].

## Connections

- [[Sohl-Dickstein et al. 2015]] — Original diffusion probabilistic models framework using nonequilibrium thermodynamics.
- [[Score Matching with Langevin Dynamics]] (Song & Ermon, 2019) — NCSN/score-based generative models; DDPM's $\epsilon$-prediction is equivalent to denoising score matching.
- [[DDIM]] (Song et al., 2020) — Non-Markovian generalization enabling faster deterministic sampling.
- [[Improved DDPM]] (Nichol & Dhariwal, 2021) — Learns reverse process variances and uses cosine schedule.
- [[Classifier-Free Diffusion Guidance]] — Enables conditional generation without a separate classifier.
