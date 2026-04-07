---
title: "OpenVLA: An Open-Source Vision-Language-Action Model"
authors: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn
conference: CoRL
year: 2024
tags:
  - VLA
  - robot-learning
  - manipulation
  - foundation-models
  - imitation-learning
link: https://arxiv.org/abs/2406.09246
aliases:
  - OpenVLA
---

# OpenVLA: An Open-Source Vision-Language-Action Model

## Problem
1. State-of-the-art VLAs are **closed-source** — architectures, training recipes, and data mixtures are hidden.
2. Prior work does not study **efficient fine-tuning** of VLAs to new robots, tasks, and commodity hardware.

## Key Idea
Release a fully open **7B-parameter VLA**, trained on 970k Open X-Embodiment trajectories, that:
- Outperforms the 55B closed RT-2-X by **+16.5%** absolute success rate (with ~7× fewer parameters).
- Supports **LoRA fine-tuning** and **4-bit quantization** so it can be adapted and served on consumer GPUs.
- Comes with model checkpoints, fine-tuning notebooks, and a scalable PyTorch training codebase.

## Method
OpenVLA treats action prediction as a language modeling task on top of a strong VLM backbone.

**Architecture** (built on Prismatic-7B [[VLM]]):
1. **Vision encoder** — concatenates features from [[DINOv2]] (spatial) and [[SigLIP]] (semantic), ~600M params.
2. **Projector** — 2-layer MLP mapping visual tokens into the LLM embedding space.
3. **LLM backbone** — [[Llama 2]] 7B.

**Action tokenization** (following RT-2):
- Each of the 7 action dims is discretized into **256 bins**, where bin widths come from the **1st–99th percentile** of training actions (robust to outliers vs. min-max).
- The 256 **least-used tokens** in the Llama tokenizer are overwritten as action tokens.
- Trained with standard **next-token prediction / cross-entropy**, loss computed only on action tokens.

**Training data**: curated subset of **Open X-Embodiment** (~970k trajectories), restricted to manipulation datasets with a 3rd-person camera and single-arm end-effector control; mixture weights follow Octo's heuristics (down-weights less diverse datasets).

**Key design decisions** (from BridgeData V2 ablations):
- Prismatic backbone > LLaVA > IDEFICS-1 for language grounding with multiple objects.
- 224×224 input resolution (no gain from 384×384, but 3× slower).
- **Fine-tune the vision encoder** (unlike standard VLM practice of freezing it) — crucial for fine-grained spatial control.
- Train for **many epochs** (final run: 27) until action-token accuracy > 95%.
- Constant learning rate **2e-5**, no warmup.

**Infrastructure**: 64× A100 for 14 days (~21.5k A100-hours), batch size 2048. Inference ~6 Hz on one RTX 4090 at 15GB (bf16); supports a remote inference server for low-compute robots.

**Efficient adaptation**:
- **LoRA** fine-tuning to new robot setups.
- **4-bit quantization** for inference on consumer GPUs with no meaningful drop in task success.

## Why it works
- **Fused DINOv2 + SigLIP** features give both semantic and spatial grounding, which pure CLIP/SigLIP backbones lack — important because manipulation needs precise spatial reasoning.
- **End-to-end fine-tuning** of a strong VLM (rather than stitching frozen pretrained modules like Octo) lets Internet-scale priors propagate into the control policy.
- A **larger, cleaner OpenX mixture** (970k vs. 350k for RT-2-X; e.g., filters all-zero Bridge actions) and quantile-based discretization improve the effective action signal.
- Casting actions as vocabulary tokens reuses the entire LLM training stack (FSDP, FlashAttention, AMP), enabling scale with minimal custom code.

## Weakness
- Does not support action chunking or high-frequency control.
- Only supports **single-image, single-arm** setups out of the box — no history, no wrist camera, no bimanual.
- Action outputs are **discretized** (256 bins/dim), limiting precision vs. continuous/diffusion heads.
- Fails to fit high-diversity data like **DROID** at this scale — authors dropped it from the mix for the final training third.
- 7B params still too large for many robots' onboard compute, despite LoRA/quantization improvements.

## Experiments
**Q1 — Out-of-the-box multi-robot control**
- Platforms: BridgeData V2 WidowX (170 rollouts) and Google robot (60 rollouts).
- Axes: visual, motion, physical, semantic generalization + language grounding with distractors.
- Baselines: RT-1-X (35M), Octo (93M), RT-2-X (55B).
- Result: OpenVLA **beats RT-2-X by 16.5%** absolute success on average across 29 tasks; comparable on Google robot, clearly better on Bridge. RT-2-X only wins on *semantic* generalization (benefit of co-fine-tuning with Internet data).

**Q2 — Fine-tuning to new robot setups**
- 7 diverse real-world manipulation tasks (pick-and-place, table cleaning, etc.).
- Compared against fine-tuned Octo and from-scratch [[Diffusion Policy]].
- OpenVLA beats Diffusion Policy by **+20.4%** on multi-object, language-grounded tasks; Diffusion Policy can still be competitive on narrow single-task setups.

**Q3 — Efficient fine-tuning & inference**
- **LoRA** fine-tuning matches full fine-tuning while training on a single A100 (or even consumer GPUs).
- **4-bit quantization** preserves task success rate while cutting VRAM substantially.

## My Ideas
- Replace discrete action head with a **diffusion** or **flow-matching** action expert (cf. [[PhD-Research/Papers/Pi0]]) to recover precision and enable higher control rates.
- Add **action chunking** + temporal ensembling (cf. [[ACT]]) on top of the VLA to improve smoothness and effective frequency.
- Co-fine-tune on a slice of Internet VQA data during VLA training to recover the semantic-generalization gap vs. RT-2-X without going back to 55B params.
- Use an extra **wrist-camera** token stream — current architecture only ingests a single 3rd-person view.
- Distill OpenVLA into a smaller student (e.g., 1–2B) conditioned on the same action tokenizer, for true onboard deployment.

## Connections
- Successor / open counterpart of [[RT-X]] (specifically RT-2-X).
- Built on Prismatic VLM — fuses DINOv2 + SigLIP into a Llama-2 backbone ([[Backbone]], [[Transformer]]).
- Compared against [[Diffusion Policy]] and Octo as generalist baselines.
- Related action-head alternatives: [[ACT]] (action chunking), [[PhD-Research/Papers/Pi0]] (flow-matching action expert), [[CrossFormer]] (cross-embodiment transformer).
- Training dataset: Open X-Embodiment ([[RT-X]]).
