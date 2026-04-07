---
title: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
authors: Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song
conference: RSS
year: 2023
tags:
  - diffusion-models
  - imitation-learning
  - visuomotor-policy
  - robot-manipulation
  - behavior-cloning
link: https://arxiv.org/abs/2303.04137
aliases:
  - Diffusion Policy
---

# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion


## Problem

Behavior cloning for robot manipulation is challenging because:
- Action distributions are **multimodal** (multiple valid ways to complete a task)
- High-dimensional action sequences require consistent long-horizon planning
- Existing explicit policies struggle to capture complex distributions; implicit policies (IBC) are expressive but ==training-unstable== due to negative sampling for normalization

## Key Idea

Represent a robot's visuomotor policy as a **conditional denoising diffusion process** over action space. Instead of directly outputting actions, the policy iteratively denoises Gaussian noise into actions conditioned on visual observations, via Stochastic Langevin Dynamics.

> [!tip] Compare to Image Generation
> 1. changing the output x to represent robot actions. 
> 2. making the denoising processes conditioned on input observation $O_t$ .

## Method

### Formulation

At timestep $t$, the policy takes $T_o$ observation steps $\mathbf{O}_t$ and denoises $K$ iterations to predict a $T_p$-step action sequence, executing only $T_a$ steps before replanning (**receding horizon control**):

$$\mathbf{A}^{k-1}_t = \alpha\!\left(\mathbf{A}^k_t - \gamma\,\epsilon_\theta(\mathbf{O}_t, \mathbf{A}^k_t, k) + \mathcal{N}(0, \sigma^2 I)\right)$$

Training loss is a simple MSE on predicted noise:

$$\mathcal{L} = \text{MSE}\!\left(\epsilon^k,\, \epsilon_\theta(\mathbf{O}_t,\, \mathbf{A}^0_t + \epsilon^k,\, k)\right)$$

### Network Architectures for $\epsilon_\theta$

| Backbone | Strength | Weakness |
|---|---|---|
| **1D Temporal CNN** | Robust, easy to tune | Over-smooths high-frequency actions |
| **Time-series Diffusion Transformer** (minGPT-style) | Best on complex/velocity tasks | More hyperparameter-sensitive |

> [!note] Recommendation
> Start with CNN; switch to Transformer if high-rate action changes matter.

### Visual Encoder

- ResNet-18 trained **end-to-end** (no pretraining needed)
- Spatial softmax pooling (preserves spatial info)
- GroupNorm instead of BatchNorm (stable with EMA used in DDPM)
- Observations conditioned via **FiLM** (CNN) or cross-attention (Transformer)

### Fast Inference

DDIM with 100 training / 10 inference iterations → **0.1 s latency** on RTX 3080, enabling real-time closed-loop control.

## Why It Works

1. **Multimodal distributions**: Stochastic initialization + Langevin dynamics sampling naturally explores multiple action basins
2. **High-dimensional outputs**: DDPM scales to long action sequences without sacrificing expressiveness
3. **Stable training**: Score function learning avoids the intractable normalization constant $Z(\mathbf{o}, \theta)$ that destabilizes IBC
4. **Position control synergy**: Multimodality is more pronounced in position space; sequence prediction reduces compounding error — diffusion exploits both

## Weakness

- Slower inference than single-step policies (mitigated by DDIM)
- Transformer backbone is sensitive to hyperparameters
- ViT encoders trained from scratch perform poorly with limited data (needs CLIP pretraining + finetuning)
- Receding horizon design requires careful tuning of $T_o$, $T_p$, $T_a$

## Experiments

Evaluated on **15 tasks across 4 benchmarks** (Robomimic, Push-T, BlockPush, Franka Kitchen) — sim + real, 2–14 DoF actions.

- **+46.9% average improvement** over prior SOTA (LSTM-GMM, IBC, BET)
- **+32%** on BlockPush p2 metric (long-horizon multimodality)
- **+213%** on Kitchen p4 metric
- Real-world: Push-T, 6-DoF liquid pouring, bimanual tasks (egg beater, mat unrolling, shirt folding)
- Robust to latency up to 4 steps; action horizon of 8 optimal for most tasks

## My Ideas

- Can diffusion policy be combined with [[world models]] to reduce demonstration data requirements?
- The connection to LQR / control theory (Sec 4.5) suggests potential for incorporating known dynamics priors
- CLIP-pretrained ViT finetuning reaching 98% with only 50 epochs — worth exploring for low-data regimes
- How does diffusion policy interact with online RL fine-tuning (e.g., DPPO)?

## Connections

- [[Implicit Behavioral Cloning]] (IBC) — predecessor; energy-based; unstable training
- [[Behavior Transformer]] (BET) — k-means tokenized action space; struggles with long-horizon multimodality
- [[DDPM]] — foundational diffusion framework this builds on
- [[ACT]] — another action-chunking approach; uses VAE instead of diffusion
- [[3D Diffusion Policy]] — extends to 3D point cloud observations
