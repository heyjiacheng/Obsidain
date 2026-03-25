---
title: "MLP + Max-Pooling Point Cloud Encoder"
tags:
  - concept
  - deep-learning
  - point-clouds
  - 3d-representations
  - encoder
  - robotics
aliases:
  - DP3 Encoder
  - Point Cloud MLP Encoder
---

# MLP + Max-Pooling Point Cloud Encoder

> [!abstract] One-line summary
> Turn a cloud of 3D points into a single compact vector — using a small neural network + max-pooling — so a robot policy can act on it.

## The Problem It Solves

A robot's depth camera gives you a **point cloud**: a set of thousands of $(x, y, z)$ dots in 3D space representing the scene. To use this in a neural network you need to compress it into a **fixed-size vector**. Two challenges:

- Points have **no fixed order** — the same object can be described by any shuffling of its points
- The **number of points varies** — you need a consistent output size regardless

This encoder solves both.

## How It Works (Step by Step)

```
500 points (x,y,z each)
        ↓
  [MLP — 3 layers]        ← same small network applied to every point independently
        ↓
  500 feature vectors
        ↓
  [Max-Pooling]            ← for each feature dimension, keep the single largest value
        ↓
  1 vector of 64 numbers   ← ready to feed into the policy
```

**Step 1 — Per-point MLP**: A 3-layer network runs on each point individually. Think of it as asking "what is geometrically interesting about this point?" for every point.

**Step 2 — Global max-pooling**: Collapses all 500 feature vectors into one by taking the max at each position. This keeps the most "activated" signal across the whole cloud — like taking the loudest voice in a crowd.

**Step 3 — Output**: A 64-dimensional vector $\mathbf{z}$ that summarizes the shape of the entire scene.

> [!tip] Why max-pooling (not average or sum)?
> Points have no order, so the aggregation must not care about order. Max-pooling is **symmetric** — it gives the same result regardless of how points are shuffled. Average-pooling also works, but max tends to capture the most salient geometric features.

## Why No Color?

[[3D Diffusion Policy|DP3]] deliberately drops RGB and uses **geometry only**. This means:

- The encoder can't tell what color an object is
- It can only tell where things are in 3D space

**Benefit**: The robot generalizes across different-colored objects, lighting conditions, and textures without retraining.

**Trade-off**: If the task *requires* color (e.g., "pick the red cup, not the blue one"), this encoder won't work.

## Why So Simple?

You might expect a bigger, pretrained model (like PointNet++) to do better. Surprisingly, this 3-layer MLP beats it on robot tasks. The likely reason: pretrained encoders carry **built-in assumptions** from their training data (e.g., ImageNet-like categories) that don't match what matters for manipulation — precise local geometry near contact points.

> [!example] Analogy
> It's like using a fine-tuned food critic to judge if a shelf is well-organized. Their expertise adds noise, not signal.

## In Practice (DP3 Numbers)

| Parameter | Value |
|---|---|
| Input points | 500 (downsampled via FPS) |
| MLP layers | 3 |
| Output size | 64-dim vector |
| Color channels | None |
| Normalization | LayerNorm after each layer |

The 64-dim vector is then concatenated with the robot's joint positions and fed into the diffusion policy.

## Related

- [[3D Diffusion Policy]] — paper that introduced this encoder design
- [[PointNet]] — the foundational idea this simplifies
