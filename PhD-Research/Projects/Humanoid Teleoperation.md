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

## Goal
Enable VR teleoperation for humanoid bimanual manipulation and collect robot learning dataset.

## Quick Start

Real VR teleoperate:

1.  运行deploy.sh 一定要在机器人上！
2. VR APP需要连接在机器人的ip 地址上


Simulation VR teleoperate:

1. open xr app in ubuntu
2. open xr app in PICO (under unknown)
3. launch three terminals (launch vr connect program after see init ok)
4. https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html
5. click re-connect in PICO
6. press A+B+X+Y simultaneously
7. press A+X
8. after robot follow your trajectory
9. then, press 9 on mujoco page
10. press A+B+X+Y for emergent stop (or press 0 in gear_sonic_deploy terminal)

## Next Steps
- [ ]  collect dataset on [lerobot](https://huggingface.co/docs/lerobot/unitree_g1) 
