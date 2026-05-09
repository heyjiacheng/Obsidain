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


## SONIC Training

1. Put two robots hands in front of the robot
2. 运行整个代码都在PC上
3. VR APP需要连接在PC的ip 地址上: 192.168.0.108
4. press A+B+X+Y simultaneously
5. press A+X
6. press A+B+X+Y for emergent stop (or press o in gear_sonic_deploy terminal)
7. press left grip button and A to record (press again to stop, left grip button and B to discard)

## SONIC Teleoperate

Real VR teleoperate:

1. Put two robots hands in front of the robot
2. 运行deploy.sh 一定要在机器人上！
3. VR APP需要连接在机器人的ip 地址上: 192.168.0.130
4. press A+B+X+Y simultaneously
5. press A+X
6. press A+B+X+Y for emergent stop (or press o in gear_sonic_deploy terminal)


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

## Psi-0 Teleoperate

### Env
1. pip install opencv (remove opencv-headless, and install same version one)
2. conda install gst-plugins-ugly
3. adjust 0.6 to 0.8 in scale_vx() under real/teleop/master_whole_body.py
4. 

### quick start
1. launch realsense service on G1
2. open develop mode on G1
3. connect ethernet on PC
4. export DDS
5. open VR app before launch program
6. choose head, controller, hand and send; change state to zedmini; listen to same address as PC service （make sure under same wifi）
7. conda activate psi_deploy
8. under real/teleop, launch python main.py --robot g1 --pico_streamer --pico_ip 192.168.0.128 192.168.0.104
9. write exit and press enter when something danger happened (or ctrl+c)
10. press s and enter to start record an episode, press q to stop

  代码注释（vr_pico.py:115）也明确写了：

  ▎ Only adjust this parameter if g1's pelvis height is too higher or too lower than expected.

  - 如果你站直时机器人骨盆偏低（在蹲）：调大 target_height
  - 如果你站直时机器人腿过伸/抬太高：调小 target_height
  - 经验值：设成你（操作者）的实际身高（米）通常最稳


- [ ]  collect dataset on [lerobot](https://huggingface.co/docs/lerobot/unitree_g1) 


# PPT

## VR Teleoperate

1. IK Solver
2. Learning Based
## IK Solver
