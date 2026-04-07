---
title: "ACT (RoboTwin) — Code Walkthrough"
tags:
  - code
  - ACT
  - robotwin
---
# pseudocode

```python
main():
	load_data()
	train()

load_data()
	random_pick(0, last_action):
	return imgs, qpos, action, is_padding

build():
	encoder_1 = Transformer_Encoder(layer, d=256)
	transfomer = Transformer(encoder_1, d=256)
	encoder_2 = Joiner(Resnet18, sinusodial_2D)
	model = CVAE(transfomer, encoder_2, num_quires=k, heads)
	optimizer = Adam([non-encoder_2, lr], [encoder_2, encoder_lr])

forward():
	-- latent action --
	if_train:
		enc_1_input = cat([CLS], qpos, action)
		enc_1_out = enc_1(enc_1_input, sinous_pos)
		mu, logvar = enc_1_out
		z = distr(mu, logvar)

	else:
		-- for deterministic action --	
		z = random or 0
	
	-- vision --
	
	for each camera:
		feat = append.encoder_2(imgs)
	src = cat(feat)
	proprio = proj(src)
	
	src = flatten(src)
	src = cat(stack(proprio + latent), src)
	src = cat(src, sin_pos)

	-- decode --
		
	memory = encoder(src+pos)
	fix_query = zeros
	action = decoder(memory, fix_query)

train():
	-- validation --
	forward()
	eval()
	
	-- train --
	loss = forward()
	zero_gradient()
	loss.backward()
	optimizer()

```






For each style, the model does not explicitly generate a separate Gaussian distribution for z. Instead, after training, different styles may occupy different regions within a shared Gaussian latent space.



# process_data.sh

Converts the original RoboTwin 2.0 data into the format required for ACT training.

---

# train.sh

## 1. Overview

The script implements **Behavior Cloning (BC)** — a supervised learning approach where a policy network learns to map observations (images + joint positions) to actions by imitating expert demonstrations.

Two policy architectures are supported:

| Policy Class | Description                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------- |
| **ACT**      | Transformer-based with a [[Conditional-VAE]]. Predicts a **chunk** of future actions at once. |
| **CNNMLP**   | Simpler CNN backbone + MLP head baseline. Predicts one action per step.                       |

---

## 2. Entry Point and Configuration

### `main(args)` — Lines 35–138

**Step 1 — Parse task configuration**

```python
is_sim = task_name[:4] == "sim-"
```

- If the task name starts with `"sim-"`, loads simulation configs from `constants.SIM_TASK_CONFIGS`.
- Otherwise, loads real-robot configs from `aloha_scripts.constants.TASK_CONFIGS`.

Each task config provides:

| Field | Meaning |
|---|---|
| `dataset_dir` | Path to the HDF5 demonstration dataset |
| `num_episodes` | Number of demonstration episodes available |
| `episode_len` | Maximum timesteps per episode |
| `camera_names` | List of camera views (e.g., `["top", "left_wrist", "right_wrist"]`) |

**Step 2 — Build policy config**

For the ACT policy (lines 66–82):

| Parameter         | Default        | Role                                                       |
| ----------------- | -------------- | ---------------------------------------------------------- |
| `num_queries`     | `chunk_size`   | Number of future action steps predicted at once            |
| `kl_weight`       | user-specified | Weight for the KL divergence loss term in the CVAE         |
| `hidden_dim`      | user-specified | Transformer hidden dimension                               |
| `dim_feedforward` | user-specified | Transformer FFN intermediate dimension                     |
| `enc_layers`      | 4              | Transformer encoder layers                                 |
| `dec_layers`      | 7              | Transformer decoder layers                                 |
| `nheads`          | 8              | Attention heads                                            |
| `backbone`        | `"resnet18"`   | CNN backbone for image feature extraction                  |
| `lr_backbone`     | 1e-5           | Separate (lower) learning rate for the pretrained backbone |

**Step 3 — Dispatch**

- `--eval` set → run `eval_bc()` on `policy_best.ckpt`
- Otherwise → load data, run `train_bc()`, save best checkpoint

---

## 3. Data Loading and Preprocessing

```python
train_dataloader, val_dataloader, stats, _ = load_data(
    dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val
)
```

`load_data()` (from `utils.py`) returns:
- **train/val dataloaders** — PyTorch DataLoaders yielding `(image_data, qpos_data, action_data, is_pad)`
- **stats** — `qpos_mean`, `qpos_std`, `action_mean`, `action_std` for z-score normalization

Each data sample:

| Tensor | Shape | Description |
|---|---|---|
| `image_data` | `(B, num_cameras, C, H, W)` | Multi-view RGB images |
| `qpos_data` | `(B, state_dim)` | Joint positions (14-dim for bimanual: 7 per arm) |
| `action_data` | `(B, chunk_size, state_dim)` | Ground-truth future action sequence |
| `is_pad` | `(B, chunk_size)` | Boolean mask for padded positions at episode boundaries |

Stats are saved to `dataset_stats.pkl` in the checkpoint directory for use during evaluation.

---

## 4. Policy Construction

### `make_policy()` — Lines 141–148

Instantiates `ACTPolicy` or `CNNMLPPolicy`. The policy `nn.Module`:
- Encodes images through a ==ResNet-18 backbone==
- Encodes joint positions through a ==linear projection==
- (ACT only) Uses a Transformer encoder-decoder with a ==CVAE latent variable==
- other parameter in deploy_policy.yml

### `make_optimizer()` — Lines 151–158

Calls `policy.configure_optimizers()`:
- Main optimizer for Transformer parameters at learning rate `lr`
- Separate parameter group for backbone at `lr_backbone` (10×–100× smaller)

---

## 5. Training Loop — `train_bc()`

### Lines 357–430

Standard epoch-based supervised learning:

```
For each epoch:
  1. Validation pass (no gradients)
  2. Training pass (with gradients)
  3. Periodic checkpoint saving
```

### 5.1 Validation Pass (Lines 378–391)

```python
with torch.inference_mode():
    policy.eval()
    for batch_idx, data in enumerate(val_dataloader):
        forward_dict = forward_pass(data, policy)
        epoch_dicts.append(forward_dict)
    epoch_summary = compute_dict_mean(epoch_dicts)
```

- Eval mode (disables dropout, etc.)
- Runs `forward_pass()` on every validation batch → averages losses via `compute_dict_mean()`
- Tracks **best checkpoint** (lowest validation loss):

```python
if epoch_val_loss < min_val_loss:
    min_val_loss = epoch_val_loss
    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
```

### 5.2 Training Pass (Lines 397–412)

```python
policy.train()
optimizer.zero_grad()
for batch_idx, data in enumerate(train_dataloader):
    forward_dict = forward_pass(data, policy)
    loss = forward_dict["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

> [!note] Key observations
> - **No gradient accumulation** — each batch is an independent step (`zero_grad → forward → backward → step`)
> - ACT loss: $L = L_\text{recon} + \lambda_\text{KL} \cdot L_\text{KL}$
>   - $L_\text{recon}$: L1 loss between predicted and ground-truth action chunks
>   - $L_\text{KL}$: KL divergence between encoder posterior and standard normal prior (CVAE regularization)

### 5.3 `forward_pass()` — Lines 346–354

```python
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)
```

During training, the policy receives **both observations and ground-truth actions**:
- **Encoder** sees ground-truth actions → produces latent style variable $z$
- **Decoder** conditions on $z$ + observations → reconstructs the action sequence
- This is the CVAE paradigm: encoder = recognition network, decoder = generation network

### 5.4 Train History Slicing (Line 407)

```python
epoch_summary = compute_dict_mean(
    train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
)
```

`train_history` is a flat list of all per-batch loss dicts across all epochs. Uses `(batch_idx + 1)` as the number of batches per epoch to slice the current epoch's training loss.

---

## 6. Evaluation Loop — `eval_bc()`

### Lines 171–343

Loads a trained checkpoint and runs closed-loop rollouts in the environment.

### 6.1 Policy Loading and Normalization Setup

```python
policy = make_policy(policy_class, policy_config)
policy.load_state_dict(torch.load(ckpt_path))
policy.cuda()
policy.eval()

stats = pickle.load(...)
pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
```

- `pre_process` — normalizes raw qpos before feeding to the policy
- `post_process` — denormalizes predicted actions back to raw joint space

### 6.2 Rollout Loop (Lines 223–318)

```python
num_rollouts = 50
for rollout_id in range(num_rollouts):
```

For each rollout:
1. **Reset** the environment (optionally randomize object poses for sim tasks)
2. **Step through `max_timesteps`** — at each step:
   - Extract observation (images + qpos)
   - Normalize qpos → query policy → denormalize action → execute
3. **Record** rewards, images (for video), and qpos trajectories

### 6.3 Action Querying Logic (Lines 269–287)

```python
if config["policy_class"] == "ACT":
    if t % query_frequency == 0:
        all_actions = policy(qpos, curr_image)  # → (1, chunk_size, action_dim)
    if temporal_agg:
        # ... (see Section 7)
    else:
        raw_action = all_actions[:, t % query_frequency]
```

**Without temporal aggregation** (`temporal_agg=False`):
- Policy queried every `chunk_size` steps
- Between queries, actions indexed from the previously predicted chunk

**With temporal aggregation** (`temporal_agg=True`):
- Policy queried at **every** step (`query_frequency = 1`)
- All predictions stored and aggregated (see §7)

**CNNMLP**: always predicts a single action per step.

### 6.4 Inference Mode

```python
with torch.inference_mode():
```

At inference, the policy receives **only observations** (no ground-truth actions):
- CVAE encoder is **not used** — latent $z$ sampled from prior $\mathcal{N}(0, I)$, or set to zero
- Decoder generates actions conditioned on observations + sampled $z$

---

## 7. Temporal Aggregation

A key technique in ACT for producing ==smoother actions==. Lines 240–281.

### Setup

```python
all_time_actions = torch.zeros(
    [max_timesteps, max_timesteps + num_queries, state_dim]
).cuda()
```

2D buffer where `all_time_actions[t_query, t_exec, :]` stores the action predicted at query time `t_query` for execution time `t_exec`.

### At Each Timestep $t$

```python
# Store the full chunk predicted at time t
all_time_actions[[t], t:t+num_queries] = all_actions

# Gather all predictions for timestep t
actions_for_curr_step = all_time_actions[:, t]
actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
actions_for_curr_step = actions_for_curr_step[actions_populated]
```

At time $t$, multiple past predictions overlap — time $t-k$ predicted an action for time $t$ (the $(k+1)$-th element of its chunk).

### Exponential Weighting

```python
k = 0.01
exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
exp_weights = exp_weights / exp_weights.sum()
raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
```

- More recent predictions get **higher** weight (exponential decay with $k=0.01$)
- Weighted average → **smoother** trajectory, reducing jitter at chunk boundaries
- Small $k$ = gentle decay — many overlapping predictions contribute significantly

> [!example] Visualization
> ```
> Time →       t-2   t-1    t    t+1   t+2
> Query t-2:  [a0]  [a1]  [a2]  [a3]
> Query t-1:        [a0]  [a1]  [a2]  [a3]
> Query t:                [a0]  [a1]  [a2]  [a3]
>                          ↑
>              3 predictions overlap at time t
>              weighted average → final action
> ```

---

## 8. Checkpoint and Logging Strategy

### Saved Checkpoints

| File | When Saved | Purpose |
|---|---|---|
| `dataset_stats.pkl` | Before training | Normalization stats for inference |
| `policy_epoch_{E}_seed_{S}.ckpt` | Every `save_freq` epochs + best | Periodic and best snapshots |
| `policy_last.ckpt` | End of training | Final model state |
| `policy_best.ckpt` | End of training | Best validation loss model |

### Training Curves

`plot_history()` (lines 433–455) generates plots for each metric (loss, KL, etc.):
- X-axis: epoch progress (interpolated for train, per-epoch for validation)
- Saved as `train_val_{key}_seed_{S}.png`

---

## Architecture Diagrams

> [!abstract]- Training Pipeline
> ```
> ┌─────────────────────────────────────────────────────┐
> │  TRAINING                                           │
> │                                                     │
> │  HDF5 Dataset                                       │
> │       │                                             │
> │       ▼                                             │
> │  load_data() → DataLoader                           │
> │       │                                             │
> │       ├── image  (B, C_cam, C, H, W)                │
> │       ├── qpos   (B, 14)                            │
> │       ├── action (B, chunk, 14)                     │
> │       └── is_pad (B, chunk)                         │
> │       │                                             │
> │       ▼                                             │
> │  ┌──────────────────────────────────┐               │
> │  │  ACT Policy                     │               │
> │  │                                 │               │
> │  │  ResNet18 ─┐                    │               │
> │  │            ├─► Transformer Encoder               │
> │  │  qpos MLP ─┘   (CVAE: z ~ q(z|a,o))            │
> │  │                 │                               │
> │  │  z + obs ──► Transformer Decoder                │
> │  │                 │                               │
> │  │            predicted actions                    │
> │  └─────────────────┬───────────────┘               │
> │                    │                                │
> │  L1_loss(pred, gt) + kl_weight * KL                │
> │       │                                             │
> │  backward() → optimizer.step()                      │
> │                                                     │
> │  stats ──► dataset_stats.pkl                        │
> │  best_model ──► policy_best.ckpt                    │
> └─────────────────────────────────────────────────────┘
> ```

> [!abstract]- Evaluation Pipeline
> ```
> ┌─────────────────────────────────────────────────────┐
> │  EVALUATION                                         │
> │                                                     │
> │  policy_best.ckpt + dataset_stats.pkl               │
> │       │                                             │
> │       ▼                                             │
> │  For each rollout (50 total):                       │
> │       │                                             │
> │    env.reset()                                      │
> │       │                                             │
> │    For each timestep:                               │
> │       ├── observe(images, qpos)                     │
> │       ├── pre_process(qpos)    ← z-score normalize  │
> │       ├── policy(qpos, images) ← z ~ N(0,I) prior  │
> │       ├── temporal_agg         ← weighted average   │
> │       ├── post_process(action) ← denormalize        │
> │       └── env.step(action)                          │
> │                                                     │
> │  Output: success_rate, avg_return, videos           │
> └─────────────────────────────────────────────────────┘
> ```

# eval.sh


