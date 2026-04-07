---
title: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
authors: Open X-Embodiment Collaboration
conference: ICRA
year: 2024
tags: [robotics, manipulation, foundation-model, cross-embodiment, VLA, imitation-learning]
link: https://arxiv.org/abs/2310.08864
aliases: [Open X-Embodiment, RT-X, OXE]
---

# Open X-Embodiment: Robotic Learning Datasets and RT-X Models

## Problem
The question is whether a **single generalist policy** can be trained across many robots and transferred to new robots, tasks, and environments.

## Key Idea
Transformer policy directly on this mixture — **without any explicit mechanism to bridge the embodiment gap**. Positive transfer emerges naturally once the model is large enough.

- **Open X-Embodiment (OXE) Dataset**: 1M+ trajectories, 22 embodiments, 60 datasets, 21 institutions, 527 skills, 160,266 tasks.
- **RT-X models**: RT-1 and RT-2 retrained on the OXE mixture.

## Method

### Data consolidation
- Pick one canonical camera view per dataset; resize to a shared resolution.
- Normalize action spaces into a **7-DoF end-effector action** (x, y, z, roll, pitch, yaw, gripper) + terminate flag.
- Actions are normalized per-dataset before discretization, then **de-normalized per embodiment** at inference.
- Coordinate frames and absolute/relative/velocity semantics are **not** aligned across datasets — the model must learn to disambiguate via the visual context.

### Architectures
- **RT-1-X** — 35M-param Transformer. EfficientNet (ImageNet-pretrained) + USE language embedding, fused via FiLM → 81 tokens → decoder-only Transformer → tokenized actions. Takes 15-image history.
- **RT-2-X** — VLA built on PaLI-X (ViT + UL2); 5B and 55B variants. Actions are cast as text tokens ("1 128 91 241 ..."). **Co-fine-tuned** on a ~1:1 mix of original web VLM data and robotics data.
- Action space discretized into **256 bins** along each of 8 dimensions.

### Training
- Robotics mixture uses **9 manipulators** (RT-1, QT-Opt, Bridge, TAR Play, Jaco Play, Cable Routing, RoboTurk, NYU VINN, Austin VIOLA, Berkeley UR5, TOTO, Language Table).
- Categorical cross-entropy over discrete action tokens.
- Inference at 3–10 Hz; RT-1 local, RT-2 on cloud.

## Why it works
- **Capacity matters**: with enough parameters, the model can absorb heterogeneous data without needing hand-designed alignment.
- **Web pretraining** gives the VLM backbone strong semantic priors (objects, verbs) that transfer to manipulation.
- Positive transfer happens because skills and objects recur across robots, and a large shared backbone can factor the common structure.

## Weakness
- All embodiments are still **single-arm manipulators with similar sensing**; bi-manual, quadruped, and very different modalities are excluded from the experiments.
- **No evaluation on fully unseen robots** — only in-distribution embodiments.
- **RT-1-X underfits** on large-data domains (Bridge, RT-1) → co-training *hurts* small models.
- No predictive theory of when positive transfer happens vs. when it does not.
- Heavy cloud dependency for the 55B RT-2-X (not edge-deployable).
- Action frame ambiguity (absolute vs. relative vs. velocity) is left to the model to figure out.

## Experiments
**3600 evaluation trials across 6 real robots.**

### In-distribution
- **Small-data domains** (Kitchen, Cable Routing, NYU Door, UR5, Robot Play): RT-1-X beats Original Method on 4/5 — **~50% mean improvement**. Small datasets benefit most from co-training.
- **Large-data domains** (Bridge, RT-1): RT-1-X *worse* than RT-1 (underfits). RT-2-X (55B) matches or beats both.

### Out-of-distribution (RT-2-X)
- **Unseen objects/backgrounds/environments**: RT-2 and RT-2-X roughly tied (VLM backbone already generalizes).
- **Emergent skills** (Google Robot performing Bridge-only tasks): RT-2-X beats RT-2 by **~3×**. Removing Bridge from training collapses this gain → evidence of cross-embodiment transfer.

### Ablations (Table II)
- Image **history** helps significantly.
- **Web pretraining** is critical (from-scratch → 0% emergent skills).
- **55B > 5B** for emergent skill transfer.
- Co-fine-tuning ≈ fine-tuning here (unlike RT-2), likely because robotics mixture is already diverse.

## My Ideas
- Can we add an **explicit embodiment token** (robot ID, DoF, control mode) to disambiguate action frames? The paper deliberately avoids this — worth checking if it hurts or helps given the capacity.
- Apply the OXE recipe to **[[Diffusion Policy]]** or **[[ACT]]**-style action heads rather than discrete-token heads — diffusion may handle multi-modal X-embodiment actions better than 256-bin discretization.
- Use RT-2-X as a pretrained backbone for **sim-to-real** with simulator-only fine-tuning (e.g., RoboTwin).
- Study **negative transfer**: identify dataset pairs where co-training hurts, and learn a per-example gating / mixture weight.

## Connections
- [[ACT]] — alternative action head (chunked continuous actions via CVAE) that might mesh with OXE data.
- [[Diffusion Policy]] — continuous multi-modal action modeling; a natural successor head for cross-embodiment training.
- [[PhD-Research/Papers/Pi0]] — later generalist VLA that builds on the OXE philosophy with flow-matching action experts.
- [[CrossFormer]] — explicit cross-embodiment Transformer design.
- RT-1, RT-2 — direct predecessors providing the architectures.
- [[Backbone]], [[Transformer]] — architectural foundations.
