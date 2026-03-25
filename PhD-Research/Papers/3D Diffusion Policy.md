---
title: "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations"
authors: Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, Huazhe Xu
conference: RSS
year: 2024
tags:
  - robotics
  - imitation-learning
  - diffusion-policy
  - 3d-representations
  - point-clouds
  - visuomotor
link: https://arxiv.org/abs/2403.03954
aliases:
  - DP3
---

# 3D Diffusion Policy (DP3)

## Problem

Visual imitation learning struggles with data efficiency and generalization:

- **2D image representations** discard spatial depth information critical for manipulation
- Methods typically require **100–200 demonstrations** to learn a task
- Poor generalization across viewpoint shifts, appearance changes, and novel object instances

## Key Idea

DP3 combines **compact 3D point cloud representations** with **diffusion-based action generation**. A simple 3-layer MLP encoder extracts a 64-dimensional feature vector from ~500 downsampled points. Color channels are deliberately excluded to enforce appearance-invariant representations.

> [!tip] Design Philosophy
> "The simple, lightweight DP3 Encoder surprisingly outperforms more complex pre-trained point encoders" like PointNet and PointNet++.

## Method

$$
\text{Depth image} \rightarrow \text{Point cloud} \xrightarrow{\text{crop + FPS}} 500\text{ pts} \xrightarrow{\text{MLP + maxpool}} \mathbf{z} \in \mathbb{R}^{64} \rightarrow \text{DDIM policy}
$$

1. **Perception**: Single-view depth camera → 3D point cloud
2. **Preprocessing**: Bounding box crop to remove background; Farthest Point Sampling to ~500 points
3. **Encoding**: [[MLP-MaxPool-PointCloud-Encoder|3-layer MLP with max-pooling]] → 64-dim feature (no color channels) 
4. **Policy**: DDIM diffusion model conditioned on $\mathbf{z}$ + robot proprioception
   - Training: MSE loss, 100 diffusion timesteps
   - Inference: 10 timesteps (fast)
   - Action horizon: predict 4 steps, execute 3

## Why It Works

- **Point clouds** preserve 3D geometric structure that 2D projections discard — critical for precise manipulation
- **No color** forces the encoder to rely on geometry, yielding robustness to appearance variation (object color, lighting)
- **Simple MLP** is faster and more generalizable than complex pretrained encoders (PointNet++) — avoids overfitting to pretrained feature biases 和pointnet怎么比
- **LayerNorm** stabilizes training; sample prediction (vs. epsilon prediction) accelerates convergence

## Weakness

- Robust only to **minor viewpoint changes** — large camera shifts break performance
- **Single-view depth** creates occlusion blind spots in cluttered scenes
- Dropping color may hurt tasks where **color is semantically meaningful** (e.g., sorting by color)
- Limited evaluation on contact-rich or deformable object tasks

## Experiments

| Setting | DP3 | Baseline (DP) | Relative Gain |
|---|---|---|---|
| 72 simulation tasks | 74.4% | 59.8% | +24.2% |
| Real-world (4 tasks) | ~85% | — | — |

**Generalization** tested across four axes:
- **Spatial**: 4/5 novel object positions
- **Appearance**: varied object colors without retraining
- **Instance**: diverse object shapes and sizes
- **Viewpoint**: minor camera position shifts

**Safety**: Near-zero safety violations vs. frequent human intervention required by baselines.

**Data efficiency**: As few as 10 demos (sim) / 40 demos (real).

## My Ideas

- **Multi-view fusion**: aggregate point clouds from 2–3 cameras to handle viewpoint robustness — likely the biggest current limitation
- **Color-aware variants**: optional color conditioning gated by task type, allowing appearance-invariant or color-sensitive modes
- **Online adaptation**: fine-tune the point encoder with a small number of real-world demos to bridge sim-to-real gaps

## Connections

- [[Diffusion Policy]] — Chi et al.; DP3 replaces 2D encoder with 3D point cloud encoder
- [[PointNet]] / PointNet++ — pretrained 3D encoders; DP3's simple MLP outperforms them here
- [[ACT]] — another imitation learning baseline compared in experiments
- [[3D robot learning]] — broader trend of lifting robot perception into 3D space
