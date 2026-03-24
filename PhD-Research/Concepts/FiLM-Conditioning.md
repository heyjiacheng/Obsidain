---
title: "FiLM Conditioning"
tags:
  - concept
  - deep-learning
  - conditioning
  - vision-language
  - robotics
aliases:
  - FiLM
  - Feature-wise Linear Modulation
---

# FiLM Conditioning

> [!abstract] One-line summary
> Dynamically reprogram any neural network's feature extraction per input by predicting a scale (γ) and shift (β) for each feature channel from a conditioning signal.

## The Problem It Solves

Standard neural networks perform one fixed computation regardless of context. When you want a model to behave differently based on an external input — a language instruction, a task label, a style — naive approaches like concatenating the conditioning signal are often too weak. The conditioning gets mixed into one layer and can be forgotten downstream.

A concrete failure: in Visual Question Answering, a CNN extracts the same visual features for every question. Asked "what color is the cube?" vs. "how many objects are left of the sphere?", the vision backbone returns identical activations — the question never influences what the network looks for in the image.

FiLM's fix: let the conditioning signal **rewrite the feature extraction** at every layer by predicting per-channel scale and shift parameters at runtime.

## How It Works (Step by Step)

```
Conditioning input z  (e.g., language instruction embedding)
        ↓
  [FiLM Generator h_i]  ← simple linear projection, one per FiLM layer
        ↓
  γ_i, β_i  (2C values for a layer with C channels)
        ↓
  FiLM(F_i | γ_i, β_i) = γ_i ⊙ F_i + β_i
        ↓
  Conditioned feature map  →  continues through the main network
```

**Step 1 — Encode the conditioning signal**: Pass `z` (e.g., the output of a sentence encoder) through a small generator network `h_i` to predict `2C` values: C scale factors `γ_i` and C shift factors `β_i`.

**Step 2 — Apply FiLM**: At each FiLM layer, after the main computation (convolution, attention, etc.) but before the nonlinearity, apply the element-wise affine transform:

$$\text{FiLM}(\mathbf{F}_i \mid \boldsymbol{\gamma}_i, \boldsymbol{\beta}_i) = \boldsymbol{\gamma}_i \odot \mathbf{F}_i + \boldsymbol{\beta}_i$$

where $\odot$ is element-wise (channel-wise) multiplication.

**Step 3 — Interpretation**: The γ values can:
- **Suppress** a feature channel: γ → 0
- **Amplify** it: |γ| > 1
- **Invert** it: γ < 0
- **Leave it unchanged**: γ = 1, β = 0

Combined, γ and β let the conditioning signal completely reroute what the network attends to at inference time — with no weight updates.

> [!tip] Why not just concatenate?
> Concatenation mixes the conditioning into one layer; it can vanish as gradients flow through many layers. FiLM injects conditioning at **every layer** with only 2C parameters each — the signal persists throughout the whole network.

## Relation to Other Methods

| Method | Operation | Notes |
|---|---|---|
| Conditional bias | `F + β(z)` | FiLM with γ = 1 |
| Conditional scaling | `γ(z) · F` | FiLM with β = 0 |
| Batch Normalization | fixed (γ, β) after normalizing | FiLM = BN with input-dependent γ, β |
| AdaIN (style transfer) | γ, β from style image statistics | Equivalent to FiLM in style transfer |
| BigGAN class conditioning | class-specific (γ, β) per ResBlock | FiLM with discrete z |
| Diffusion timestep conditioning | timestep embedding → (γ, β) | FiLM applied per denoising step |

## In Practice

**Original paper (VQA on CLEVR):**
- `z` = final hidden state of a GRU encoding the question
- `h_i` = linear layer: `z → (γ_i, β_i)` of size `2C`
- FiLM applied inside each ResNet residual block after BatchNorm, before ReLU
- Reduced CLEVR error by **50%** vs. prior SOTA (Perez et al., AAAI 2018)

**Robotics (RT-1):**
- `z` = pretrained language model embedding of the task instruction
- FiLM conditions EfficientNet-B3 visual backbone
- Allows one backbone to extract task-relevant features for 700+ tasks

**OpenVLA-OFT+ (VLA fine-tuning):**
- `z` = mean language embedding from the VLM tokenizer
- Projected to γ and β that modulate **all ViT patch features** across every ViT block
- Critical for multi-camera setups: without FiLM, language following drops to chance (33%) because the model exploits spurious visual correlations to solve tasks without reading the instruction

| Parameter | Typical value |
|---|---|
| Generator `h_i` | Linear layer (per FiLM layer) |
| Parameters added | 2C per FiLM layer (C = channel count) |
| Placement | After normalization, before nonlinearity |
| Training | End-to-end jointly with main network |

## Why It Works

The generator is learning a **hypernetwork**: given the conditioning signal, it produces the weights for an affine transformation of the main network's internals. Because the transformation is applied channel-wise at every layer, the conditioning permeates the entire representation. Empirically, the learned γ values cluster semantically — questions about color activate color-sensitive channels, questions about shape activate shape-sensitive channels — demonstrating the model routes computation meaningfully rather than randomly.

## Trade-offs

- **Parameter efficient** for the gain: only 2C new parameters per layer
- **No spatial resolution dependence**: the same γ/β apply across all spatial positions in a conv layer
- **Cannot handle multimodal conditioning**: FiLM predicts a single (γ, β) per input — if two very different conditionings should produce similar features, the generator must interpolate smoothly

## Related

- [[OpenVlaOft]] — uses FiLM (OFT+ variant) to inject language into ViT visual features; necessary to prevent shortcut learning in multi-camera ALOHA tasks
- [[OpenVLA]] — base model that OFT+ extends with FiLM conditioning
- [[ActionChunking]] — paired with FiLM in the OFT+ recipe for dexterous manipulation
