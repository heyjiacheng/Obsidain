---
title: "Latent Action Pretraining from Videos"
authors: Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo
conference: ICLR
year: 2025
tags:
  - VLA
  - latent-action
  - robot-learning
  - pretraining
link: https://arxiv.org/abs/2410.11758
aliases: [LAPA, Latent Action Pretraining]
---
# Research Question
Can a Vision-Language-Action model be pretrained on videos without any ground-truth robot action labels?

# Motivation
Robot action labels require expensive human teleoperation, which blocks scaling VLA pretraining to internet-scale video data.

# Method
A VQ-VAE quantizes the change between video frames into discrete latent actions, a VLM is pretrained to predict them, and a small action-labeled dataset finetunes the mapping to real robot actions.

# Limitation
The change between two frames is not necessary be action, maybe someone moving at background.


# Summary
Latent actions learned purely from pixels form a shared, embodiment-agnostic action space that transfers better than ground-truth actions (beating OpenVLA by +6.2% with ~30x less pretraining compute), even when pretrained only on human videos.

# Connections
- OpenVLA — competes-with (SOTA VLA pretrained on 970k action-labeled Open-X trajectories; LAPA outperforms it without action labels)
- GENIE — builds-on (borrows its latent action model from user inputs, repurposed to label actionless video for robot policies)
- UniPi — competes-with (video-diffusion planning from actionless data; LAPA is more robust on long-horizon and cross-environment transfer)
