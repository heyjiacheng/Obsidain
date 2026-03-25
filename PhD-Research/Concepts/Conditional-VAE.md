---
title: "Conditional VAE"
tags:
  - concept
  - deep-learning
  - generative-models
  - variational-inference
aliases:
  - CVAE
  - Conditional Variational Autoencoder
---

# Conditional VAE

> [!abstract] One-line summary
> A VAE whose encoder and decoder are both conditioned on an additional observed variable, enabling controlled generation and modeling of multi-modal conditional distributions.

## The Problem It Solves

A standard [[Variational-Autoencoder|VAE]] learns $p(\mathbf{x})$ — it can generate data, but you cannot control *what* it generates. If demonstrations of a task have multiple valid styles (e.g., a human sometimes hands an object from the left, sometimes from the right), a deterministic model averages over modes and produces invalid behavior. A vanilla VAE can capture multimodality, but it cannot condition the generation on an input observation.

CVAE solves this by modeling $p(\mathbf{x} \mid \mathbf{y})$ — a conditional distribution that can produce diverse, multi-modal outputs for a given input $\mathbf{y}$.

## How It Works (Step by Step)

### Standard VAE recap

A VAE learns a generative model $p_\theta(\mathbf{x} \mid \mathbf{z})$ with latent variable $\mathbf{z}$, trained by maximizing the evidence lower bound (ELBO):

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

### CVAE: condition everything on $\mathbf{y}$

CVAE introduces an observed conditioning variable $\mathbf{y}$ (e.g., a class label, an image observation, joint positions). Every distribution is now conditioned on $\mathbf{y}$:

| Component | VAE | CVAE |
|---|---|---|
| Prior | $p(\mathbf{z})$ | $p(\mathbf{z} \mid \mathbf{y})$ |
| Encoder | $q_\phi(\mathbf{z} \mid \mathbf{x})$ | $q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{y})$ |
| Decoder | $p_\theta(\mathbf{x} \mid \mathbf{z})$ | $p_\theta(\mathbf{x} \mid \mathbf{z}, \mathbf{y})$ |

The CVAE ELBO becomes:

$$\log p(\mathbf{x}|\mathbf{y}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x},\mathbf{y})}[\log p_\theta(\mathbf{x}|\mathbf{z},\mathbf{y})] - D_\text{KL}(q_\phi(\mathbf{z}|\mathbf{x},\mathbf{y}) \| p(\mathbf{z}|\mathbf{y}))$$

### Training

```
Training input: observation y, output x (e.g., y = images+joints, x = action sequence)

1. Encoder:  (x, y) → q_φ(z | x, y) = N(μ, σ²I)
             ↓ reparameterization trick
             z = μ + σ ⊙ ε,  ε ~ N(0, I)

2. Decoder:  (z, y) → x̂ = p_θ(x | z, y)

3. Loss:     L = reconstruction(x, x̂) + β · D_KL(q_φ(z|x,y) ‖ p(z|y))
```

The encoder sees *both* the input and the output during training, learning to encode the "style" or "mode" of the output into $\mathbf{z}$. The decoder must reconstruct the output from $\mathbf{z}$ and the input alone.

### Inference (test time)

The encoder is discarded. Sample $\mathbf{z}$ from the prior $p(\mathbf{z} \mid \mathbf{y})$ (often simplified to $\mathcal{N}(0, I)$) and decode:

$$\hat{\mathbf{x}} = p_\theta(\mathbf{x} \mid \mathbf{z}, \mathbf{y}), \quad \mathbf{z} \sim p(\mathbf{z}|\mathbf{y})$$

In practice, setting $\mathbf{z} = \mathbf{0}$ (the prior mean) gives the most likely output mode.

## Analogy

Think of the latent variable $\mathbf{z}$ as a "style knob." During training, the encoder learns to read the demonstration and set the knob to the right position (e.g., "this is a left-side handover"). During inference, you don't know the style, so you set the knob to its default position (zero) — the decoder has learned to produce the most common style for that default.

## In Practice: ACT (Action Chunking with Transformers)

In [[ACT]], the CVAE is used to handle multimodality in human teleoperation demonstrations:

- $\mathbf{y}$ = current observation (4× RGB images + joint positions)
- $\mathbf{x}$ = action chunk ($k \times 14$ joint targets)
- $\mathbf{z}$ = 32-dim "style variable" encoding which human behavior mode this demonstration follows
- **Encoder**: BERT-style transformer with `[CLS]` token → $\mu, \sigma$ of $\mathbf{z}$
- **Decoder (policy)**: transformer that conditions on $\mathbf{z}$ and observations to predict the action chunk
- At test time: $\mathbf{z} = \mathbf{0}$ (prior mean), encoder discarded

> [!tip] When CVAE matters
> On scripted (unimodal) data, removing the CVAE has negligible effect. On human data, removing CVAE drops success from 35% → 2%. The CVAE absorbs the variability that would otherwise cause mode-averaging.

## Trade-offs

- **Pro**: Lightweight way to model multimodality without iterative sampling (unlike diffusion)
- **Pro**: Clean separation — $\mathbf{z}$ captures style, decoder captures structure
- **Con**: KL term can cause posterior collapse if $\beta$ is too high — the encoder learns to ignore $\mathbf{x}$ and just match the prior
- **Con**: Limited expressiveness compared to diffusion models for very complex multi-modal distributions
- **Con**: Choosing the prior $p(\mathbf{z}|\mathbf{y})$ matters — a standard Gaussian may not match the true latent structure

## Related

- [[KL-Divergence]] — the regularization term that keeps the encoder close to the prior
- [[ACT]] — uses CVAE to handle human demonstration variability in bimanual manipulation
- [[Diffusion Policy]] — alternative approach to multimodal action generation using iterative denoising
