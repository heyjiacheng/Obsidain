---
title: Training-Time Action Conditioning as a Drop-In Replacement for Real-Time Chunking
authors: Kevin Black, Allen Z. Ren, Michael Equi, Sergey Levine
conference: arXiv preprint
year: 2025
tags: [VLA, robotics, action-chunking, flow-matching, real-time-control, diffusion-transformer]
link: https://arxiv.org/abs/2512.05964
aliases: [Training-Time RTC, TT-RTC]
---

# Training-Time Action Conditioning as a Drop-In Replacement for Real-Time Chunking

## Problem 1
Real-time chunking (RTC) lets VLAs produce smooth, reactive trajectories by predicting the next action chunk asynchronously while the current one executes, but its **inference-time inpainting** requires a vector–Jacobian product at every denoising step.

## Solution 1
Move the prefix-conditioning out of inference and into **training**. Instead of inpainting at sampling time, simulate the inference delay during training and learn the conditional distribution directly:

$$p_\theta(\mathbf{A}_{t+d:H} \mid \mathbf{o}_t, \mathbf{A}_{t:t+d})$$

The objective becomes a masked flow-matching loss where the prefix $\mathbf{A}_{t:t+d}$ is fed as ground-truth (flow timestep $\tau = 1$) and the loss is computed only on the postfix tokens. Inference cost is identical to vanilla flow sampling — the conditioning is "free" at runtime.

## Method
Three minimal changes to a standard diffusion-transformer action expert (e.g., $\pi_{0.6}$):

1. **Per-token flow timestep.** In adaLN-zero conditioning, allow scale/shift/gate to differ per token so each action position can carry its own $\tau$. No new parameters.
2. **Prefix injection.** For the first $d$ action tokens, pass the ground-truth action and set $\tau = 1$ (no noise). For the remaining $H - d$ tokens, run normal noisy flow matching.
3. **Masked loss.** Compute the flow-matching MSE only on postfix tokens.

Delay $d$ is sampled randomly per batch (uniform $[0, 10)$ in the real-world experiments, exponentially decreasing weights in sim). The resulting policy exposes the same `(prefix, delay) → postfix` interface as inference-time RTC, so it is a **drop-in replacement** with no robot-runtime changes.

> [!note] Implementation
> Algorithm 1 in the paper shows the entire change is roughly 5 added lines of JAX: build a `prefix_mask = arange(H) < delay`, overwrite `time` and `x_t` on prefix positions, mask the loss with `logical_not(prefix_mask)`.

## Why it works
Inference-time RTC enforces the prefix via pseudoinverse guidance, which **linearizes** the model around each denoising step using its Jacobian. As the prefix grows (high $d$), the postfix must satisfy a larger consistency constraint, and the linearization becomes a worse approximation — guidance has to "work harder" and quality drops. Training-time RTC instead learns the true conditional density end-to-end, so accuracy is bounded by training compute rather than by a per-step linear approximation. The cost is paid once, in training, not repeatedly at every denoising step of every chunk.

## Weakness
- **No soft masking.** Inference-time RTC can softly weight all $H - s$ overlapping actions (red + yellow in Fig. 1) with exponentially decaying influence; training-time RTC only conditions on the "hard" prefix of length $d$.
- **Delay distribution must be chosen up front.** The training-time delay distribution has to cover the deployment latency; mismatch hurts.
- **Slightly worse at $d \in \{0, 1\}$** in sim, because supervision on the very first action tokens is diluted by the random-delay sampling.
- Requires extra fine-tuning compute (8 epochs / 8k gradient steps in their setup).

## Experiments
- **Sim (dynamic Kinetix, $H=8$, 2048 trials/point):** training-time RTC matches inference-time RTC at $d \in \{0,1\}$ and **strictly outperforms it at $d \geq 2$**, with the gap widening at $d=4$.
- **Real-world ($\pi_{0.6}$, 50 Hz, H100 inference):** on box building and espresso making, training-time RTC achieves **parity in both success rate and task duration** with inference-time RTC, while reducing end-to-end latency from ~135 ms to ~108 ms. Both clearly beat synchronous inference, which shows visible inter-chunk pauses.

## My Ideas
Add a tiny inference-time **soft-masking residual head** on top of training-time RTC: the backbone is trained with hard prefix conditioning (cheap, robust), and a small auxiliary module — trained jointly — contributes a corrective bias for the $H - s - d$ overlapping postfix actions. This recovers the continuity benefit of soft masking without ever invoking a full vector–Jacobian product in the main flow. Effectively: pay the Jacobian cost only on a lightweight head, not on the full action expert.

## Connections
- [[RealTimeChunking]] — direct predecessor (Black, Galliker, Levine 2025); this paper replaces its inpainting step.
- $\pi_0$ / $\pi_{0.5}$ / $\pi_{0.6}$ / $\pi^*_{0.6}$ — the VLA family the method is fine-tuned on.
- **Flow matching** (Lipman et al. 2022) and **Diffusion Transformer / adaLN-zero** (Peebles & Xie 2023) — the architectural ingredients the per-token-$\tau$ trick relies on.
- **Pseudoinverse-guided diffusion** (Song et al. 2023; Pokle et al. 2023) — the inference-time inpainting technique being replaced.
- **A2C2** (Sendai et al. 2025) — concurrent: lightweight correction head for chunk discontinuities.
- **VLASH** (Tang et al. 2025) — concurrent: conditions on a single future action; this work conditions on a full prefix.
- **SmolVLA** (Shukor et al. 2025) — async execution without solving inter-chunk discontinuity.
- **Hierarchical VLAs** (Gemini Robotics, GR00T-N1) — orthogonal axis for reducing effective control latency.
