---
title: Humanoid Teleoperation
date: 2026-03-13
tags:
  - project
  - humanoid
  - teleoperation
  - robot-learning
  - imitation-learning
status: active
aliases:
  - Humanoid VR Teleoperation
---

# Humanoid Teleoperation

## Goal
Enable VR teleoperation for humanoid bimanual manipulation and collect robot learning dataset.

## Related Papers
- [[Diffusion Policy]]
- [[ACT]]
- [[RT-2]]
- [[PointWorld]] — 3D point flow as observation; potential observation backbone

## Related Ideas
- [[Humanoid Teleoperation Dataset]]

## Dataset Plan
- 1000 demonstrations across diverse household tasks
- Capture: wrist RGB-D (left + right), head RGB-D, proprioception, action stream
- Environments: lab benchtop, kitchen counter, cluttered desk
- Task categories: pick-place, pouring, tool use, articulated objects, deformable objects

## Experiments
- [[2026-03-teleop-dataset-test]]

## Open Questions
- Latency: what is the acceptable round-trip for high-quality demos? (target <50ms)
- Control interface: hand controllers vs. exoskeleton vs. gloves?
- Retargeting: how to handle finger DoF mismatch between human and robot hand?
- Data quality filtering: auto-reject failed demos or manual curation?

## Next Steps
- [ ] Benchmark teleoperation latency on current hardware setup
- [ ] Implement retargeting pipeline for bimanual humanoid
- [ ] Run pilot collection session (10 demos) and assess quality
- [ ] Define task taxonomy for first dataset release
