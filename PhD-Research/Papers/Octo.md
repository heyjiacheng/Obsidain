---
title: "Octo: An Open-Source Generalist Robot Policy"
authors: Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Ria Doshi, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, Sergey Levine
conference: RSS
year: 2024
tags:
  - generalist-policy
  - transformer
  - diffusion-policy
  - imitation-learning
  - open-x-embodiment
link: https://arxiv.org/abs/2405.12213
aliases:
  - Octo
---

# Octo: An Open-Source Generalist Robot Policy

## Problem
Existing generalist robot policies (e.g., [[RT-X]], RoboCat) lock users into a fixed set of sensory inputs and action spaces used at pretraining, can't be easily finetuned to new observations/action spaces, and the largest are not publicly available. There is no open, flexible, pretrained policy that can be adapted to arbitrary downstream robot setups.

## Key Idea
Train a single transformer-based policy on the largest cross-embodied robot manipulation dataset (800k trajectories from Open X-Embodiment) using a **modular token-in / token-out design**. Because inputs and outputs are just tokens attended to via a block-wise causal mask, new cameras, proprioception, languages, goal images, or action heads can be **added or removed during finetuning** without re-initializing the pretrained backbone.

## Method

### Architecture
A [[Transformer]]-first policy with three parts:

1. **Tokenizers**
   - Language instruction `ℓ` → T5-base (111M) → language tokens.
   - Image observations / goal images → shallow CNN patch encoder → patch tokens (ViT-style).
2. **Transformer backbone**
   - Input sequence: `[T_task, T_o,1, T_o,2, ...]` with learned positional embeddings.
   - **Block-wise causal attention**: observation tokens at step `t` attend only to task tokens and earlier observation tokens; non-existing modalities are fully masked.
   - Learned **readout tokens** `T_R,t` (analogous to BERT `[CLS]`) passively read from task/obs tokens but are never attended to — they act as compact summaries.
3. **Diffusion action head**
   - A small conditional [[Diffusion Policy]] head applied to readout embeddings predicts a **chunk** of consecutive future actions.
   - Only one transformer forward pass per action prediction; the K-step DDPM denoising runs inside the lightweight head.

> [!note] Why the block-wise mask matters
> Because each modality lives in its own token block with masked attention, adding a new camera or a new action head only requires training new positional embeddings / a new encoder / a new head — **no pretrained weights are re-initialized**.

### Training data
- 25 datasets curated from the [[Open X-Embodiment]] dataset (≈800k episodes — larger than RT-X's 350k subset).
- Removed datasets without images or delta EE control; reweighted "more diverse" datasets up, repetitive ones down.
- Zero-padded missing camera channels; aligned gripper convention (+1 open, 0 closed).

### Training objective
- DDPM denoising loss on action chunks with cosine noise schedule.
- 2 frames of observation history; hindsight goal relabeling.
- Randomly drop language or goal image per example → policy supports **either** conditioning mode at inference.

### Training details
- Two sizes: **Octo-Small** (27M, ViT-S backbone) and **Octo-Base** (93M, ViT-B backbone).
- AdamW, inverse-sqrt LR, weight decay 0.1, grad clip 1.0.
- Octo-Base: 300k steps, batch 2048, TPU v4-128, 14h.
- Finetuning: ~100 demos, 50k steps, cosine decay LR, ~5h on a single A5000 (24GB).

## Why it works
- **Flexibility from tokens + masks**: treating every modality as a token block with a controlled attention pattern decouples the architecture from the interface, making pretrain→finetune domain shifts cheap.
- **Diffusion head**: models multi-modal action distributions (unlike MSE) while keeping continuous precision (unlike discrete action tokens in RT-1/2) — ablations show it beats both.
- **Transformer-first [[Backbone]]**: putting parameters in the transformer rather than a large ResNet encoder scales better on the diverse 800k-episode mixture (ResNet wins on small from-scratch data, but loses at scale).
- **Scale of data**: 25 datasets > 11 (RT-X mix) > single-robot — performance improves monotonically.

## Weakness
- **Wrist cameras underused**: only 27% of pretraining data has wrist cams; finetuning often works better using *only* third-person cameras.
- **Language ≪ goal-image conditioning**: only 56% of data is language-annotated, so goal-image conditioning beats language conditioning by ~25% on BridgeV2.
- **Imitation only**: trains purely on optimal demonstrations; no sub-optimal or online interaction data.
- **Scope**: only single- and dual-arm manipulators — no navigation or mobile manipulation.
- **Degradation on novel scenes / skills** (e.g., flipping, precise insertion) in zero-shot evaluation.

## Experiments

- **9 real robots across 4 institutions**, both zero-shot and finetuning evaluations.
- **Zero-shot vs. [[RT-X]]**: Octo beats RT-1-X by +29% average success across three embodiments; matches RT-2-X (55B VLM) on WidowX and RT-1 Robot tasks.
- **Finetuning (Table I)**: avg success 72% vs. 20% ResNet+Transformer-from-scratch vs. 15% VC-1 pretrained-visual. Same hyperparameters across all 6 domains. Successfully finetunes with *new observations* (force-torque for Berkeley Insertion), *new action spaces* (joint-position for Berkeley Pick-Up), and *new embodiments* (Berkeley Bimanual / Coke).
- **Ablations (Table II, WidowX)**:
  - Data: Octo mix 83% > RT-X mix 60% > single-robot 43%.
  - Policy head: Diffusion 83% > Continuous MSE 35% > Discretized 18%.
  - Arch: ViT+diffusion 83% > ResNet-50 + transformer 70%.
- **Scaling**: Tiny (10M) → Small (27M) → Base (93M) — zero-shot success improves monotonically.

## My Ideas
- **Sub-optimal / online data**: Add RL or weighted imitation objectives so Octo can ingest autonomous rollouts and human corrections, not just expert demos.
- **Wrist-camera curriculum**: Oversample wrist-camera episodes or pretrain a wrist-specific adapter to fix the observed wrist-cam weakness.
- **Unified manipulator + navigator**: The token-block design should straightforwardly absorb navigation datasets — worth testing whether a single Octo can be both a manipulator and a GNM-style navigator.
- **Compare to [[PhD-Research/Papers/Pi0]] / [[OpenVLA]]**: Octo is a non-VLA baseline — ablate whether vision-language pretraining actually helps beyond the diffusion head's gains.
- **Action-chunk length vs. diffusion steps**: The paper fixes both — a scaling study here could tell us where the compute should go.

## Connections
- **Cross-embodiment pretraining**: [[RT-X]], RoboCat, GNM.
- **Diffusion policies**: [[Diffusion Policy]] (Chi et al.), which Octo reuses as its action head.
- **Action chunking**: [[ACT]] — Octo predicts chunks similarly.
- **ViT backbones**: standard vision [[Transformer]] scaling recipes.
- **Follow-ups / contemporaries**: [[OpenVLA]], [[PhD-Research/Papers/Pi0]], [[CrossFormer]] — later generalist policies building on or contrasting with Octo.
- **Dataset**: [[Open X-Embodiment]].
