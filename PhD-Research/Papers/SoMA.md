---
title: "SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation"
authors:
  - Mu Huang
  - Hui Wang
  - Kerui Ren
  - Linning Xu
  - Yunsong Zhou
  - Mulin Yu
  - Bo Dai
  - Jiangmiao Pang
conference: ICML
year: 2026
tags:
  - gaussian-splatting
  - robot-manipulation
  - simulation
  - deformable-objects
  - real-to-sim
link: https://arxiv.org/abs/2602.02402
aliases:
  - SoMA
---

# SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation


## Problem

Simulating deformable objects under rich robot manipulation interactions is a fundamental challenge in real-to-sim. The difficulty is that dynamics are jointly driven by:
- **Environmental forces** (gravity, contact, friction)
- **Robot actions** (joint torques, end-effector poses)

Existing simulators fall into two camps, both limited:
- **Physics-based** (e.g., MuJoCo, SOFA): rely on predefined material parameters, brittle to sim-to-real gap
- **Data-driven neural dynamics** (e.g., GausSim, PhysTwin): learn object dynamics but lack explicit **robot-conditioned control**, hurting accuracy and generalizability to unseen actions

## Key Idea

**SoMA** is a 3D Gaussian Splat (GS) neural simulator that couples deformable dynamics, environmental forces, and robot joint actions in a **unified latent neural space**, enabling end-to-end real-to-sim simulation — without any predefined physical models.

> [!tip] One-line summary
> Learn to simulate soft-body deformation *directly on Gaussian splats*, conditioned on robot actions, from multi-view RGB video.

## Method

SoMA has three core components:

### 1. Real-to-Sim (R2S) Scene Initialization
- Reconstructs the scene using **3D Gaussian Splatting** from multi-view RGB observations
- Establishes **robot reference frames** by integrating joint states from the robot's kinematic chain
- Anchors Gaussians to the robot skeleton to define physically consistent interaction regions

### 2. Force-Driven GS Dynamics Modeling
- Models Gaussian splats as **nodes in a hierarchical graph** (coarse-to-fine clustering)
- Two force channels:
  - *Environmental force graph*: models gravity, inertia, internal deformation
  - *Robot force graph*: propagates joint-conditioned forces into the deformable object
- A learned **GNN** propagates forces through the graph to predict Gaussian displacement $\Delta \mu$ at each step

### 3. Multi-Resolution Training Strategy
- **Temporal multi-resolution**: trains with varying temporal strides $k$ to learn both fine-grained and long-horizon dynamics
- **Image multi-resolution**: uses low-res supervision early, gradually increases to full resolution (blended with depth supervision)
- **Blended supervision**: combines occlusion-aware RGB rendering loss + momentum consistency regularization to stabilize long rollouts

$$
\mathcal{L} = \lambda_1 \mathcal{L}_\text{rgb} + \lambda_2 \mathcal{L}_\text{depth} + \lambda_3 \mathcal{L}_\text{momentum}
$$

## Why it works

- **No physics priors needed**: the GNN learns material-agnostic dynamics from data
- **Robot conditioning**: action input at every step lets the model respond correctly to novel manipulations
- **Hierarchical graph**: captures multi-scale deformation — coarse structure for global shape, fine nodes for local detail
- **Multi-res training**: prevents the model from overfitting to short-range correlations; stabilizes long-horizon rollouts

## Weakness

- Requires a **multi-view camera setup** for reconstruction — not applicable to single-view or monocular settings
- Reconstruction quality depends on the initial **3DGS fitting**; thin or highly specular objects may degrade dynamics
- **Topology changes** (tearing, cutting) are not handled — Gaussians are fixed in number
- Generalization is demonstrated within the same object category; cross-category transfer is untested
- Computational cost of GNN rollouts may limit real-time use

## Experiments

**Dataset:** Four real-world deformable objects — rope, doll, cloth, T-shirt — collected on an **ARX-Lift** robotic platform with multi-view cameras.

**Tasks:**
- *Resimulation*: replay recorded trajectories and compare to ground truth
- *Generalization*: evaluate on unseen manipulation trajectories

**Baselines:** PhysTwin, GausSim

**Metrics:**
- RGB quality: PSNR, SSIM, LPIPS
- Geometry: Abs Rel, RMSE (from depth)

**Key results:**
- **+20% improvement** over baselines on both resimulation accuracy and generalization
- Enables stable **long-horizon cloth folding** that baselines fail to complete
- Ablations confirm each component (robot conditioning, hierarchical graph, multi-res training) contributes meaningfully

## My Ideas

- Could the **robot force graph** be replaced by a learned contact model for tool-object interactions (e.g., cutting, pushing with non-rigid tools)?
- Interesting to explore whether Gaussian splat dynamics can support **differentiable planning** — optimize robot trajectories through the simulator
- The hierarchical graph idea is similar to [[PointWorld]] — both work on point-cloud-like representations with graph message passing; worth comparing
- **Sim-to-real policy transfer**: could train manipulation policies inside SoMA and transfer zero-shot? The 20% accuracy gain is promising but is it sufficient for downstream RL?
- Topology change is a hard limit — could consider **dynamic Gaussian spawning/merging** for cutting/tearing scenarios

## Connections

- [[PointWorld]] — similar point-cloud neural dynamics paradigm
- 3D Gaussian Splatting (3DGS) as scene representation
- GausSim, PhysTwin — direct baselines; both learn GS dynamics without robot conditioning
- PAC-NeRF, DNeRF — physics-augmented neural fields (alternative representation)
- Real-to-sim / sim-to-real gap literature in robot learning
