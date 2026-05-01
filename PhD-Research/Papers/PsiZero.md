---
title: "Ψ₀ (Psi-Zero): An Open Foundation Model Towards Universal Humanoid Loco-Manipulation"
authors: Songlin Wei, Hongyi Jing, Boqian Li, Zhenyu Zhao, Jiageng Mao, Zhenhao Ni, Sicheng He, Jie Liu, Xiawei Liu, Kaidi Kang, Sheng Zang, Weiduo Yuan, Marco Pavone, Di Huang, Yue Wang
conference: arXiv preprint
year: 2026
tags: [humanoid, VLA, loco-manipulation, foundation-model, egocentric-video, flow-matching]
link: https://arxiv.org/abs/2603.12263
aliases: [Psi0, Psi-Zero, PsiZero]
---

# Ψ₀ (Psi-Zero): An Open Foundation Model Towards Universal Humanoid Loco-Manipulation

## Problem 1: Embodied Gap

Existing methods (such as EgoVLA, In-n-On, and H-RDT) leverage human data by co-training a unified strategy on human videos and robot data

There are fundamental differences between humans and humanoid robots in terms of action frequency, and degrees of freedom (DoF); 

therefore, a single monolithic policy that attempts to model two fundamentally different action distributions simultaneously is suboptimal.

## Solution 1: Decoupled Staged Training

Stage 1 (Pre-training VLM): Autoregressively pre-train Qwen3-VL-2B-Instruct on EgoDex (~829 hours of egocentric human video) using a human-robot unified 48-DoF task-space representation (9-DoF wrist pose + 3D positions of 5 fingertips), with the goal of learning task semantics and visual representations

Stage 2 (Post-training Action Expert): Freeze the VLM and train a flow-based action expert from scratch to predict action chunks directly in 36-DoF joint space, using the Humanoid Everyday dataset. (side question: does it need to be 8 dof lower body actions??)

Stage 3 (Fine-tuning): Fine-tune the action expert on 80 teleoperation trajectories for each downstream task

## Problem 2: Excessive Pre-training Computing

Having the VLM make autoregressive predictions for high-dimensional action chunks is computationally expensive and significantly slows down training.

## Solution 2: Tokenizer
Use the FAST tokenizer to discretize continuous actions, and retrain the tokenizer on 500,000 sampled actions from EgoDex (L1 reconstruction loss reduced from 0.01 to 0.005, with each action compressed to ~20 tokens)
Predict only the next-step action $a_t$
at​ rather than the action chunk $a_{t:t+H}$

## Problem 3: Efficiency of VL Feature and Action Feature Fusion
cross-attention receive condition is desined for text-conditioned
## Solution 3: MM DiT
 adopt the MM-DiT architecture 

## Problem 4: Inference Delay

## Solution 4:
Training time RTC

Formally, given a language instruction $\ell$ and observation $\mathbf{o}_t = (\mathbf{I}_t, \mathbf{q}_t)$, predict whole-body action chunk $\mathbf{a}_{t:t+H}$ where $\mathbf{a} \in \mathbb{R}^{36}$ comprises hand, arm, torso, base height, and locomotion velocities.

## Key Idea
**Decouple** what each data source is good for, rather than co-training:

1. **Human egocentric video** → pre-train a VLM to learn *task semantics and visual representations* (next-action token prediction in a unified task space).
2. **Real humanoid robot data** → post-train a *flow-based action expert* in joint space to learn embodiment-specific dynamics.

> Scaling humanoid learning requires scaling the **right data in the right way** — 800h human video + 30h robot data beats baselines trained on >10× more data.

## Method

### Triple-system architecture
- **System-2 (VLM)**: Qwen3-VL-2B-Instruct backbone.
- **System-1 (Action Expert)**: Flow-matching MM-DiT (~500M params) inspired by Stable Diffusion 3.
- **System-0 (Low-level)**: AMO — an off-the-shelf RL controller mapping high-level commands to 15-DoF lower-body joints.

### Three-stage training

**Stage 1 — Pre-training on EgoDex (829h egocentric human video)**
- Unified 48-DoF task-space action: wrist pose (9-DoF) + 5 fingertip positions per hand.
- FAST tokenizer trained on 500k actions; compresses each action sequence to ~20 tokens (L1 loss ≈ 0.005).
- Single-step autoregressive prediction (not chunk) to reduce compute:
$$
p_\theta(\mathbf{a}) = \prod_{t=1}^{N} p_\theta(\mathbf{a}_t \mid \mathbf{a}_{<t}, \ell, \mathbf{o}_t)
$$
- 64× A100 GPUs, 10 days, batch 1024, lr 1e-4.

**Stage 2 — Post-training on Humanoid Everyday (31h robot data)**
- Freeze VLM, train action expert from scratch in **joint space** (36-DoF).
- Flow-matching objective:
$$
\mathcal{L}_{fm} = \mathbb{E}\left[\lVert v_\rho^{flow}(\mathbf{z}_t, \mathbf{a}_t^\tau, \tau) - (\mathbf\epsilon - \mathbf{a}_t) \rVert\right]
$$
where $\mathbf{a}_t^\tau = \tau \mathbf{a}_t + (1-\tau)\mathbf\epsilon$.
- MM-DiT: time $\tau$ separately modulates action and VL features; action and VL tokens perform **joint global attention** within each block.

**Stage 3 — Task-specific fine-tuning**
- 80 teleoperated episodes per task, action expert only, 40k steps, cosine schedule.

### Real-Time Chunking (RTC)
- At training, randomly un-noise the first $d \sim \text{Uniform}(0, d_{\max})$ tokens and mask them from loss, simulating inference delay (~160 ms per forward pass).
- At deployment, asynchronous control loop (30 Hz) and inference loop share buffers; inference triggers when execution crosses $t \geq s_{\min}$, enabling seamless chunk transitions without "stop-and-think" pauses.

### Tailored Teleoperation
Decouples upper-body pose (PICO headset + wrist trackers + multi-target IK), dexterous hand (MANUS gloves), and locomotion (waist/foot trackers → RL policy). One operator, improved lower-body stability, no VR hand-tracking occlusion.

## Why it works
- **Embodiment separation**: the VLM never has to reconcile human vs. humanoid joint spaces — it just learns "what action to take next" at a task level. The action expert then translates task-level intent into embodiment-specific joint commands using clean robot data.
- **Cheap pre-training**: next single-action prediction is far lighter than full-chunk AR, letting the VLM absorb ~900M frames of EgoDex.
- **MM-DiT > naive DiT**: dual modulation + joint attention give stronger VL→action conditioning than text-conditioned image-generation DiTs that were never designed for action prediction.
- **RTC removes jitter**: training-time inpainting teaches the model that the first few tokens are "already committed," so successive chunks are continuous by construction.

## Weakness
- Compute-heavy: 64 A100 × 10 days for pre-training alone.
- Test-time RTC [^6] was unstable on this model; had to fall back to training-time RTC.
- Hardware-bound: Unitree G1 payload limits which manipulation skills are achievable.
- Precision tasks with high-DoF fine motion (e.g., turn faucet with one finger) still degrade (6/10).
- Depends on EgoDex quality — method assumes **high-quality** egocentric data; noisy Internet clips are explicitly argued against.
- Frozen VLM during post-training means visual representations cannot adapt to embodiment-specific visual cues seen only in robot data.

## Experiments
- **Platform**: Unitree G1 (29-DoF) + Dex3-1 hands (7-DoF each) + RealSense D435i.
- **8 long-horizon tasks**, most >2000 steps @ 30 Hz, 3–5 sub-tasks each. 10 rollouts per model.
- **Baselines**: π0.5, GR00T N1.6, InternVLA-M1, H-RDT, EgoVLA, Diffusion Policy, ACT.
- **Result**: Ψ₀ ≥ 40% higher overall success than second-best (GR00T N1.6). Example overall rates: Task 6 (Ψ₀ 9/10 vs. GR00T 0/10), Task 7 (9/10 vs. 5/10), Task 3 (8/10 vs. 4/10).
- **Ablations** (on dual-arm pick-and-carry): pre-train EgoDex +4, post-train HE +2, RTC +1, MM-DiT ties DiT on this task but helps elsewhere.

## My Ideas
1. **Pre-training action representation audit**: The VLM is supervised on a 48-DoF task-space action it never uses downstream — meaningful visual reps emerge anyway. Swap the supervision signal (e.g., contrastive future-frame, or masked video prediction on EgoDex) and check whether the downstream gain survives; this would test whether *any* action-grounded pretext task suffices or if next-action-token is specifically important.
2. **Unfreeze VLM late**: Add a short third stage where the VLM is LoRA-tuned on robot data alongside the action expert, to close the visual domain gap between EgoDex RGB and RealSense D435i without overwriting task priors.

## Connections
- **π₀ / π₀.₅** [[Pi0 (RoboTwin)]]: same flow-matching-action-expert family; Ψ₀ argues against their end-to-end co-training strategy.
- **GR00T N1.6** [[GR00TN1]]: direct competitor; Ψ₀ beats it with far less robot data.
- **AMO** [[AMO]]: reused directly as the lower-body RL controller.
- **EgoVLA, H-RDT, In-n-On**: representative co-training baselines that Ψ₀ critiques.
- **EgoDex**: pre-training video corpus (829h, per-frame upper-body transforms).
- **Humanoid Everyday**: post-training robot corpus (31h, 260 tasks).
- **FAST tokenizer**: discrete-action tokenization for the AR pre-training stage.
- **MM-DiT (Stable Diffusion 3)**: architectural import for the action expert.
- **Real-Time Chunking**: training-time inpainting variant, vs. test-time gradient-guidance RTC.
