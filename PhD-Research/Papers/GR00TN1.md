---
title: "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots"
authors: Scott Reed, Ruijie Zheng, Guanzhi Wang, Johan Bjorck, Joel Jang, Ao Zhang, Linxi "Jim" Fan, Yuke Zhu, et al. (NVIDIA)
conference: arXiv
year: 2025
tags:
  - robotics
  - foundation-model
  - humanoid
  - VLA
  - diffusion-transformer
  - imitation-learning
link: https://arxiv.org/pdf/2503.14734
aliases:
  - GR00T
  - GR00T N1
---

# GR00T N1: An Open Foundation Model for Generalist Humanoid Robots

## Problem

General-purpose humanoid robots lack a scalable foundation model that can reason about novel situations and rapidly learn new tasks. The key bottleneck is data scarcity — no "Internet of humanoid robot data" exists, and the diversity in embodiments, sensors, and control modes creates fragmented "data islands" rather than a coherent training corpus.

## Key Idea

A **Vision-Language-Action (VLA)** model with a **dual-system architecture** inspired by human cognition:
- **System 2** (slow, reasoning): a pre-trained Vision-Language Model ([[Eagle-2 VLM|Eagle-2]]) interprets visual observations and language instructions at 10 Hz.
- **System 1** (fast, reactive): a **Diffusion Transformer (DiT)** generates fluid motor actions at 120 Hz via flow-matching.

Both modules are jointly trained end-to-end. A **data pyramid** strategy unifies heterogeneous sources — human videos (base), synthetic/neural data (middle), and real-robot trajectories (top) — to overcome data scarcity.

## Method

### Dual-System Architecture

GR00T N1 adopts a compositional architecture comprising two tightly coupled modules, motivated by the dual-process theory of human cognition (Kahneman, System 1/System 2).

**System 2 — Vision-Language Module (reasoning, 10 Hz).** The semantic backbone is Eagle-2, a VLM composed of a SigLIP-2 vision encoder and a SmolLM2 language model. Each input image is encoded at $224 \times 224$ resolution and compressed to 64 token embeddings via pixel shuffle. These image tokens, together with the task language instruction, are processed by the LLM in standard chat format. Critically, the authors extract **intermediate representations from layer 12** (rather than the final layer) of the LLM — empirically yielding both faster inference and higher downstream success rate. This suggests that middle-layer features retain richer perceptual-semantic information before the LLM collapses representations toward next-token prediction.

**System 1 — Diffusion Transformer Module (action generation, 120 Hz).** Action generation is formulated as a **conditional flow-matching** problem. Given a ground-truth action chunk $A_t = [a_t, \dots, a_{t+H-1}]$ with $H=16$, a flow-matching timestep $\tau \in [0,1]$, and Gaussian noise $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, the noised action is:

$$A_t^{\tau} = \tau A_t + (1 - \tau)\epsilon$$

A Diffusion Transformer $V_\theta$ is trained to predict the denoising vector field $(\epsilon - A_t)$ by minimizing:

$$\mathcal{L}_{\text{fm}}(\theta) = \mathbb{E}_{\tau}\left[\|V_\theta(\phi_t, A_t^{\tau}, q_t) - (\epsilon - A_t)\|^2\right]$$

where $\phi_t$ are VLM token embeddings and $q_t$ is the proprioceptive state encoding. The timestep distribution follows a truncated Beta prior $p(\tau) = \text{Beta}\left(\frac{s - \tau}{s}; 1.5, 1\right)$ with $s=0.999$, which biases training toward the high-noise regime. At inference, $K=4$ Euler steps suffice for action generation (63.9 ms per 16-step chunk on an L40 GPU).

The DiT consists of alternating **self-attention blocks** (operating over noised action tokens and state embeddings) and **cross-attention blocks** (conditioning on VLM features $\phi_t$). This cross-attention coupling — rather than the mixture-of-experts gating used in prior VLAs like $\pi_0$ — provides a clean factorization: the VLM and DiT can be independently scaled or swapped without architectural entanglement.

**Embodiment-Specific Projectors.** To handle variable state/action dimensions across embodiments, per-embodiment MLP encoders project proprioceptive states and actions into a shared embedding space. A corresponding per-embodiment MLP decoder maps DiT output tokens back to the native action space. This design enables a single set of DiT weights to serve multiple embodiments without dimension conflicts.

### Data Pyramid and Cross-Modal Training

The central insight is structuring heterogeneous data sources as a pyramid where **scale and embodiment-specificity are inversely correlated**:

| Layer | Source | Scale | Action Supervision |
|-------|--------|-------|--------------------|
| Base | Human egocentric videos (Ego4D, Ego-Exo4D, EPIC-KITCHENS, etc.) + web VLM pre-training data | Largest | Latent actions (VQ-VAE) |
| Middle | Neural trajectories (video generation, ~827 hrs) + simulation (DexMimicGen, ~6,500 hrs / 780k trajectories) | Medium | Latent actions / IDM pseudo-actions |
| Top | Real-robot teleoperation (GR-1, Open X-Embodiment, AgiBot-Alpha) | Smallest | Ground-truth actions |

Three techniques enable co-training across these layers:

1. **Latent Action Learning (LAPA).** A VQ-VAE is trained across all embodiments: the encoder maps frame pairs $(x_t, x_{t+H})$ to a continuous latent embedding $z_t$; the decoder reconstructs $x_{t+H}$ from $(z_t, x_t)$. After training, the pre-quantized continuous embedding serves as a pseudo-action label. Because the VQ-VAE is trained on all heterogeneous data jointly, it learns a **shared latent action space** — similar latent codes correspond to semantically equivalent motions (e.g., "move right arm left") across robot and human embodiments. This is treated as a distinct "LAPA embodiment" during policy training.

2. **Neural Trajectory Generation.** Image-to-video generation models are fine-tuned on 3,000 real teleoperation samples and used to synthesize ~10× more counterfactual trajectories with novel language prompts and object configurations. A multimodal LLM generates physically feasible instruction combinations, and another LLM-as-judge filters and re-captions generated videos. The resulting trajectories are labeled with either latent actions or IDM pseudo-actions.

3. **Simulation Trajectory Synthesis.** DexMimicGen decomposes human demonstrations into object-centric subtasks, transforms them to new object poses via relative end-effector alignment, and replays in simulation — retaining only successful rollouts. This scales a few dozen demos to 540k+ trajectories across 54 receptacle-pair combinations.

### Training Protocol

- **Pre-training**: all data layers are sampled jointly; both ground-truth and latent/pseudo-actions are used as flow-matching targets depending on the data source. The VLM language weights are frozen; vision encoder and DiT are trained end-to-end. GR00T-N1-2B (2.2B params, 1.34B in VLM) used ~50,000 H100 GPU hours.
- **Post-training**: single-embodiment fine-tuning on downstream task data. Language weights remain frozen. Feasible on a single A6000 GPU (batch size up to 200 when freezing VLM vision encoder).

## Why It Works

**Decoupled temporal scales match the structure of manipulation.** Semantic understanding (identifying objects, parsing instructions, spatial reasoning) changes slowly relative to motor execution. By running the VLM at 10 Hz and the DiT at 120 Hz, the architecture avoids forcing a single network to simultaneously handle both timescales — the VLM provides a stable latent "goal state" that the DiT conditions on for high-frequency closed-loop control. This is analogous to the separation between task-level planning and servo-level control in classical robotics, but learned end-to-end.

**Flow-matching enables expressive multi-modal action distributions.** Unlike Gaussian mixture or autoregressive action decoders, flow-matching models the full continuous action distribution through iterative denoising. This is critical for manipulation, where multiple valid trajectories often exist for a given goal. The truncated Beta prior on $\tau$ concentrates training on the high-noise regime, improving sample diversity and robustness to distribution shift.

**Latent actions create a common currency across embodiments.** The VQ-VAE latent action space abstracts away embodiment-specific kinematics and maps all data — human videos, simulated robots, real robots — into a shared representation of "what motion occurred." This transforms the heterogeneous data island problem into a standard multi-task learning problem over a unified action space, enabling positive transfer from data-rich sources (human video) to data-scarce targets (humanoid teleoperation).

**Neural trajectory augmentation provides counterfactual coverage.** Real teleoperation data is inherently limited to the configurations actually demonstrated. Video generation models, trained on Internet-scale visual priors, can synthesize plausible trajectories for unseen object-instruction combinations — effectively filling gaps in the task-configuration space. The empirical finding that IDM-labeled actions outperform latent actions in high-data regimes (while latent actions are better in low-data regimes) suggests a bias-variance tradeoff: IDM labels are closer to true actions but require sufficient data to train accurately.

**Cross-attention > MoE for modularity.** The cross-attention interface between VLM and DiT preserves a clean factorization — the VLM can be upgraded or the DiT scaled independently. In contrast, MoE-based coupling (as in $\pi_0$) entangles the two modules through shared routing, making it harder to swap components or interpret the information flow.

## Weakness

- Currently limited to **short-horizon tabletop manipulation** — no long-horizon locomotion or loco-manipulation.
- Neural trajectory generation still struggles with **physical plausibility** — generated videos may violate physics.
- Post-training can **overwrite pre-trained capabilities** (e.g., bimanual handover behavior lost after single-hand-only fine-tuning).
- Simulation-to-reality gap remains a challenge for synthetic data.

## Experiments

### Simulation (100 demos/task)

| Method | RoboCasa | DexMG | GR-1 | Average |
|--------|----------|-------|------|---------|
| BC Transformer | 26.3% | 53.9% | 16.1% | 26.4% |
| [[Diffusion Policy]] | 25.6% | 56.1% | 32.7% | 33.4% |
| **GR00T-N1-2B** | **32.1%** | **66.5%** | **50.0%** | **45.0%** |

### Real-World (GR-1 Humanoid)

| Method | Pick-and-Place | Articulated | Industrial | Coordination | Average |
|--------|---------------|-------------|------------|-------------|---------|
| Diffusion Policy (Full) | 36.0% | 38.6% | 61.0% | 62.5% | 46.4% |
| **GR00T-N1-2B (10% Data)** | 35.0% | 62.0% | 31.0% | 50.0% | 42.6% |
| **GR00T-N1-2B (Full)** | **82.0%** | **70.9%** | **70.0%** | **82.5%** | **76.8%** |

Key takeaway: GR00T-N1-2B with only **10% of the data** nearly matches Diffusion Policy trained on the full dataset, demonstrating strong **data efficiency** from pre-training.

Neural trajectory co-training adds +4–9% in simulation and +5.8% on real robot.

## My Ideas

- The latent action space from VQ-VAE is a powerful idea for unifying cross-embodiment data — could this be extended to learn a **universal action tokenizer** across even more diverse embodiments (quadrupeds, drones)?
- The post-training forgetting problem (losing handover skills) suggests exploring **continual learning** or **elastic weight consolidation** during fine-tuning.
- Neural trajectory generation quality is the bottleneck — integrating **physics-aware video generation** (e.g., with differentiable simulation priors) could improve data quality.
- The 10 Hz / 120 Hz dual-frequency design is interesting for [[hierarchical control]] — could System 2 be made even slower (1 Hz) for higher-level planning while System 1 handles reactive control?

## Connections

- [[Diffusion Policy]] — baseline comparison; GR00T N1 uses similar flow-matching but adds VLM reasoning
- [[LAPA]] — latent action pre-training approach used for action-less video data
- [[DexMimicGen]] — automated simulation data generation pipeline
- [[Eagle-2 VLM]] — vision-language backbone
- [[OpenVLA]] / [[RT-2]] / [[Octo]] — other VLA foundation models for robotics
- [[Action Chunking]] — GR00T N1 uses action chunks of 16 steps, similar to ACT
- [[Flow Matching]] — the generative framework for action prediction
