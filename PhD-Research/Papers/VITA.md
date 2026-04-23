---
title: "VITA: Vision-To-Action Flow Matching Policy"
authors: Dechen Gao, Boqi Zhao, Andrew Lee, Ian Chuang, Hanchu Zhou, Hang Wang, Zhe Zhao, Junshan Zhang, Iman Soltani
conference: arXiv
year: 2025
tags:
  - flow-matching
  - visuomotor-policy
  - imitation-learning
  - robotics
link: https://arxiv.org/abs/2507.13231
aliases:
  - VITA
  - Vision-To-Action Flow Matching
---

# VITA: Vision-To-Action Flow Matching Policy

## Problem
Conventional flow matching and diffusion policies sample from a Gaussian prior and need **conditioning modules** (cross-attention, AdaLN, FiLM) to inject visual observations at *every* denoising step. This causes large time/memory overhead, hurting real-time robot control (e.g., 50–200 Hz).

## Key Idea
Skip the noise prior and the conditioning entirely: **let visual latents themselves be the source of the flow** and flow directly to latent actions. The flow is *visually grounded at the source*, so no per-step visual conditioning is needed.

## Method
Three components trained end-to-end:

1. **Vision encoder** $\mathcal{E}_v$: maps observation $O \to \mathbf{z}_0 \in \mathbb{R}^{D_{\text{latent}}}$ (flow source).
2. **Action autoencoder** $(\mathcal{E}_a, \mathcal{D}_a)$: lifts low-dim raw action chunks $A$ into a latent $\mathbf{z}_1 \in \mathbb{R}^{D_{\text{latent}}}$ matching the visual dimensionality, then decodes back. Needed because flow matching requires equal-dimensional source/target, and actions are far lower-dim, sparser, and less structured than vision.
3. **Conditioning-free velocity field** $v_\theta(\mathbf{z}_t, t)$: learns the flow from $\mathbf{z}_0$ to $\mathbf{z}_1$, solved with an Euler ODE solver at inference. No $O$ injected after $t=0$.

**Flow Latent Decoding (FLD).** Naively co-training flow + AE collapses the latent action space. The cause is a *training–inference gap*: the decoder is trained on encoder latents $\mathbf{z}_1$ but at test time decodes ODE-generated $\hat{\mathbf{z}}_1$. FLD closes this by backpropagating action reconstruction loss through the ODE solving steps:
$$\mathcal{L}_{\mathrm{FLD}} = \|\mathcal{D}_a(\hat{\mathbf{z}}_1) - A\|.$$
Gradients flow through $\mathcal{D}_a$ and the ODE into both $v_\theta$ and $\mathcal{E}_v$, anchoring latent generation with ground-truth actions.

**Total objective:**
$$\mathcal{L}_{\text{VITA}} = \lambda_{\text{FM}}\mathcal{L}_{\text{FM}} + \lambda_{\text{FLD}}\mathcal{L}_{\text{FLD}} + \lambda_{\text{AE}}\mathcal{L}_{\text{AE}}.$$

A surrogate **Flow Latent Consistency (FLC)** loss $\|\hat{\mathbf{z}}_1 - \mathbf{z}_1\|$ is shown (Theorem 1) to be locally equivalent to FLD under mild Jacobian regularity of $\mathcal{D}_a$.

## Why it works
- Visual latents already carry rich, structured information, so they make a much better starting point than Gaussian noise — the "denoising" trajectory becomes shorter and more informative.
- A learned latent action space provides a high-dim, structured target that flow matching can actually fit (vs. sparse zero-padded raw actions).
- FLD prevents latent collapse by ensuring the *decoder sees ODE-generated samples during training*, eliminating the train/test distribution mismatch.

## Weakness
- Joint end-to-end training of vision encoder + AE + flow + ODE-through-decoder is intricate; FLD requires backprop through Euler steps which adds training cost.
- Latent action AE depends on limited robot data — generalization to large multi-task settings (where frozen pre-trained latents would be desirable) is not addressed; the paper explicitly notes a frozen latent space fails here.
- Tested on imitation learning from 50–200 demos; behavior under distribution shift / longer horizons unclear.
- "Conditioning-free" only at the *flow* — visual encoder still must encode rich enough $\mathbf{z}_0$ to determine the entire trajectory in one shot.

## Experiments
- **Tasks:** 9 simulation + 5 real-world tasks across AV-ALOHA (bimanual, 21-DoF, active vision), single-arm ALOHA, Robomimic, PushT, RLBench CloseBox.
- **Efficiency:** $1.5\times$–$2\times$ faster inference and 18.6%–28.7% lower memory than conventional flow-matching policies of comparable size.
- **Performance:** Matches or surpasses SOTA visuomotor policies in success rate.
- **Architectural simplicity:** With vector visual features, the flow net reduces to an **MLP-only** policy — reportedly the first MLP-only flow matching policy to succeed on ALOHA bimanual manipulation. Scales up to transformers for grid features without needing cross-attention conditioning.
- **Ablations:** Confirm FLD (or FLC) is essential; without it the latent space collapses. Also ablates loss weights $\lambda_{\text{FLD}}, \lambda_{\text{FLC}}, \lambda_{\text{AE}}$.

## My Ideas
- Replace the Euler ODE solver with a learned/distilled few-step solver — possibly collapsing inference to 1 step, since the source is already informative.
- Try VITA with a pretrained vision foundation model (DINOv2, SigLIP) frozen as $\mathcal{E}_v$, training only the AE + $v_\theta$ — would test how much "visual grounding" alone buys you.
- Use VITA's latent action space as a tokenizer for a downstream VLA model: actions become discrete codes in a vision-aligned space.
- Investigate whether FLD's "backprop through ODE" can be replaced by a shorter-horizon FLC + occasional FLD anneal, for cheaper training.
- Extend to stochastic policies: currently $\hat{\mathbf{z}}_1$ is deterministic given $O$; injecting controlled noise into $\mathbf{z}_0$ might recover multimodality.

## Connections
- **Conventional flow matching / diffusion policies:** Diffusion Policy (Chi et al.), $\pi_0$, Helix — VITA's main efficiency baseline.
- **Cross-modal flow matching:** CrossFlow and related works that flow between text and image without conditioning.
- **Latent diffusion (LDM):** VITA borrows the latent-space idea but argues frozen pretrained latents fail for action data, motivating end-to-end training.
- **Action chunking (ACT):** Uses the same $T_{\text{pred}}$ chunking trick for temporal consistency.
- **ALOHA / AV-ALOHA / Robomimic:** Benchmarks.
