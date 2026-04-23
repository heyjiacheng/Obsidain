---
title: "AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control"
authors:
  - Jialong Li
  - Xuxin Cheng
  - Tianshu Huang
  - Shiqi Yang
  - Ri-Zhao Qiu
  - Xiaolong Wang
conference: arXiv
year: 2025
tags:
  - humanoid
  - whole-body-control
  - reinforcement-learning
  - trajectory-optimization
  - teleoperation
  - imitation-learning
link: https://arxiv.org/abs/2505.03738
aliases:
  - Adaptive Motion Optimization
---

# AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control

> [!info] Project Page
> [amo-humanoid.github.io](https://amo-humanoid.github.io/) — 29-DoF Unitree G1

## Problem
Real humanoid robots struggle to achieve **hyper-dexterous whole-body movements** (e.g., bending to pick objects off the ground) because:
- High-DoF (29) nonlinear dynamics make model-based optimal control intractable in real-time.
- **MoCap-driven RL** suffers kinematic bias — datasets are dominated by bipedal walking, lacking coordinated arm-torso motion.
- **Trajectory optimization (TO)-based RL** relies on a limited motion primitive repertoire and is too slow for reactive teleoperation.
- Existing methods fail on **out-of-distribution (O.O.D.)** commands typical of human teleoperation.

## Key Idea
Bridge TO and RL via a **hybrid motion dataset** plus a learned **adaptation module** that converts sparse torso commands $(\mathbf{rpy}, h)$ into continuous lower-body reference poses, enabling robust generalization to O.O.D. commands without requiring reference motion at deployment.

## Method
The framework is a **hierarchical decoupled** design with four stages (see Figure 2 in paper):

### 1. AMO Dataset Construction (Hybrid Motion Synthesis)
- Sample upper-body arm motions from **AMASS** MoCap dataset.
- Sample **torso orientation** $(\mathbf{rpy})$ and **height** $h$ randomly (eliminates MoCap kinematic bias).
- Run **trajectory optimization** via BoxFDDP in [[Crocoddyl]] to produce dynamically feasible lower-body joint angles. Cost:
$$\mathcal{L} = \mathcal{L}_{\mathbf{x}} + \mathcal{L}_{\mathbf{u}} + \mathcal{L}_{\mathrm{CoM}} + \mathcal{L}_{\mathbf{rpy}} + \mathcal{L}_{h}$$
Enforces CoM stability + wrench cone constraints (double-support stance assumed).

### 2. AMO Module Pre-Training
- A **3-layer MLP** $\phi(\mathbf{q}_{\mathrm{upper}}, \mathbf{rpy}, h) = \mathbf{q}^{\mathrm{ref}}_{\mathrm{lower}}$ trained on the AMO dataset.
- Learns a **continuous mapping** (vs. a discrete look-up), enabling O.O.D. interpolation.
- Frozen during later RL training.

### 3. Lower Policy Training (Teacher-Student)
- Trained in **IsaacGym** with PPO.
- **Teacher** has privileged obs (ground-truth $\mathbf{v}, \mathbf{rpy}, h$, contact).
- **Student** distilled via supervised learning; uses 25-step proprio history instead of privileged obs.
- Observation includes $\mathbf{q}^{\mathrm{ref}}_{\mathrm{lower}}$ from the frozen AMO module — this is the key signal for tracking height/torso.
- Action: 15-dim (2×6 legs + 3 waist).

### 4. Upper Policy (Two Modes)
- **Teleoperation**: **Multi-target weighted IK** (Levenberg–Marquardt via [[Pink]]) matches head + left wrist + right wrist from VR. Solves jointly for upper joints AND intermediate $(\mathbf{rpy}, h)$ command, with higher posture cost on $\mathbf{rpy}, h$ → use legs only when arms alone cannot reach.
- **Autonomous**: [[ACT]] with [[DINOv2]] stereo visual encoder outputs upper joints + intermediate lower commands.

## Why it works
- **Hybrid dataset breaks MoCap bias**: randomly sampling torso commands exposes the policy to full torso configurations, not just walking.
- **TO provides dynamically feasible references**: unlike raw MoCap, the generated poses respect the robot's dynamics (CoM, wrench cones).
- **Continuous MLP enables O.O.D. generalization**: tested yaw up to $\pm2$ rad (trained on $\pm1.57$), height down to $0.4$m (trained $0.5-0.8$m).
- **Hierarchical decoupling**: upper policy runs IK/IL, lower policy handles balance — each subsystem is simpler.
- **Intermediate $(\mathbf{rpy}, h)$ as interface**: a compact, interpretable bridge between upper and lower controllers.

## Weakness
- **Decoupled design limits coordination**: arms don't participate in balance (humans do). In highly dynamic scenes, arm independence could harm stability.
- **No walking in AMO dataset**: references assume double-support — AMO errors may grow during locomotion (still reported low velocity tracking error, but limitation acknowledged).
- **Teleoperation still needs IK solver** at runtime for upper policy (not a learned end-to-end mapping).
- **Sim-to-real gap** handled via privileged → student distillation, but success in contact-rich manipulation still depends on MoCap arm trajectory quality.
- **No dynamic recovery behaviors** (e.g., step-taking to catch falls) — stays in double-support.

## Experiments

### Simulation
- **Torso/height/velocity tracking** (Table II): AMO beats `w/o AMO`, `w/o priv`, `w rand arms` on pitch ($E_p$), roll ($E_r$), and height ($E_h$). `w/o AMO` barely tracks height at all — showing the AMO reference is the key signal.
- **Torso range** (Table III, Fig 4): AMO achieves ranges unattainable by waist-motor-only baselines. E.g., pitch $(-0.45, 1.57)$ — can bend fully flat. Dramatically beats [[ExBody2]], which is constrained by MoCap diversity.
- **O.O.D. generalization** (Fig 5): AMO tracks well beyond training distribution; baseline fails even inside I.D. range.

### Real Robot (Unitree G1 + Dex3-1 hands, ZED Mini stereo)
- **Teleoperation demos**: pick from ground, high-shelf placement, torso pitch/roll/yaw demos.
- **Imitation learning tasks** (Table IV):
  - **Paper Bag Picking** (stereo, chunk 120): 8/10 pick, 8/10 move, 9/10 place.
  - **Trash Bottle Throwing** (stereo, chunk 120): 7/10 pick, 10/10 place.
  - **Basket Picking** (Fig 7): long-horizon loco-manipulation — crouch, grasp two low baskets, walk, place on eye-level shelf.
- Ablations: shorter chunk size hurts long multi-phase tasks; stereo helps grasping (depth).
- Runs onboard **Jetson Orin NX @ 50Hz** (full system).

## My Ideas
- **Balance-aware upper policy**: authors suggest this — feed base state into IK to let arms help balance in dynamic scenarios. Could explore Koopman or diffusion-based balance priors.
- **Extend AMO dataset to single-support**: currently double-support only. Sampling contact schedules during TO could enable stepping-aware references → walking with big torso pitch.
- **Replace IK upper policy with learned upper policy**: end-to-end VLA-style upper controller conditioned on head/wrist targets + $(\mathbf{rpy}, h)$ auto-emerges; might unlock smoother behaviors.
- **Language-conditioned $(\mathbf{rpy}, h)$**: use VLM to set intermediate goals from instructions for autonomous mode, rather than relying on ACT to predict them.
- **Combine with [[PI0]]-style VLA**: AMO's lower policy as an action primitive callable by a high-level VLA.
- **O.O.D. as teleoperation safety feature**: the strong O.O.D. tracking is a huge advantage for human operators — quantify robustness under adversarial noisy VR inputs.

## Connections
- **Motion imitation RL**: [[ExBody2]], [[OmniH2O]], [[HumanPlus]] — AMO removes the kinematic bias these inherit from MoCap.
- **TO + RL hybrid**: [[Opt2Skill]] — AMO makes TO references online-adaptive via an MLP; Opt2Skill relies on reference playback.
- **Whole-body neural controllers**: [[HOVER]] (He et al., 2024) — AMO adds explicit torso/height controllability and O.O.D. robustness.
- **Teleoperation**: [[OpenTeleVision]] (same group, Cheng et al. 2024) — AMO inherits the VR streaming stack.
- **Imitation learning backbone**: [[ACT]] + [[DINOv2]] — standard manipulation stack, here applied to whole-body.
- **TO solver**: [[Crocoddyl]] with BoxFDDP for the MCOP formulation.
- **IK library**: [[Pink]] (Caron et al.) for multi-target weighted IK.
- **Hardware**: 29-DoF Unitree G1 platform — now a common humanoid benchmark alongside H1, Digit, Figure.
