---
title: LeWM Training Algorithm
tags:
  - code
  - world-model
  - JEPA
aliases:
  - LeJEPA
---

# LeWM Training Algorithm

This document explains the main training algorithm implemented in this repository (the `LeWM` / LeJEPA world model). It covers the data flow, the model architecture, the loss, and how the training loop is wired up. File and line references point to the exact code corresponding to each step.

---

## 1. High-level idea

The model is a **JEPA-style latent world model**:

1. A vision encoder (ViT) maps every observation frame to a latent embedding.
2. An action embedder maps actions to the same latent space.
3. A causal autoregressive *predictor* (a small Transformer with AdaLN-Zero conditioning on actions) takes a short context of past embeddings and actions, and predicts the embeddings of the next frames.
4. Training is done **entirely in latent space**: predicted embeddings are regressed against the encoder's own embeddings of the future frames (no pixel reconstruction).
5. To prevent representational collapse (the trivial solution of outputting a constant vector), a **SIGReg** (Sketch Isotropic Gaussian Regularizer) pulls the empirical distribution of embeddings toward an isotropic Gaussian via the Epps–Pulley statistic on random 1-D projections.

The total loss is

$$
\mathcal{L} = \text{MSE}(\hat{z}, z) + \lambda \cdot \text{SIGReg}(z)
$$

with $\lambda =$ `cfg.loss.sigreg.weight` (default `0.09`).

> [!tip] LeJEPA Recipe
> **A single forward pass, no EMA / target encoder, no stop-gradient asymmetry, no contrastive negatives.** Collapse is avoided purely by the SIGReg distributional regularizer.

---

## 2. Data pipeline

Defined in `train.py:48-76`.

- `swm.data.HDF5Dataset` loads short trajectory windows from an HDF5 file (e.g. `ogbench/cube_single_expert`). Each sample contains `num_steps` consecutive frames where `num_steps = wm.num_preds + wm.history_size` (configured in `config/train/data/ogb.yaml:3` — by default $1 + 3 = 4$ frames per window).
- Frames are loaded with a `frameskip` (5 in `ogb.yaml`), so each "action" actually corresponds to a block of `frameskip` raw env steps, flattened into `effective_act_dim = frameskip * action_dim` (`train.py:92`).
- Image preprocessing: `get_img_preprocessor` (`utils.py:7-11`) converts frames to ImageNet-normalized images and resizes them to `cfg.img_size` (224).
- For every non-pixel column listed in `keys_to_load`, a per-column z-score normalizer is fit from dataset statistics (`utils.py:14-26`, used at `train.py:62`).
- The dataset is split 90/10 train/val (`train.py:71-73`) and wrapped into shuffled / non-shuffled DataLoaders (`train.py:75-76`).

A batch produced by the loader has shape:

```python
batch["pixels"]      # (B, T, C, H, W)   T = history_size + num_preds
batch["action"]      # (B, T, frameskip*A)
batch["observation"]  # (B, T, obs_dim)    loaded but unused in loss
```

`NaN`s appearing at trajectory boundaries are zeroed out at `train.py:26`.

---

## 3. Model components

Built in `train.py:82-124`.

### 3.1 Encoder — `vit_hf`

`train.py:82-88`

A Hugging-Face ViT (size `tiny`, patch 14, image 224) created via `stable_pretraining.backbone.utils.vit_hf`. It's *not* pretrained — it trains from scratch end-to-end with the predictor. Inside `JEPA.encode` (`jepa.py:29-45`) the time dimension is folded into the batch (`b t ... -> (b t) ...`), the ViT runs with `interpolate_pos_encoding=True`, and only the **CLS token** (`last_hidden_state[:, 0]`) is kept as the per-frame representation.

### 3.2 Projector / predictor projector — `MLP`

`module.py:217-241`

Two 2-layer MLPs (hidden 2048, BatchNorm1d) that map the 192-dim ViT CLS features into the 192-dim "world-model" embedding space:

- **`projector`** — applied right after the encoder, so the embeddings fed to the loss live in `embed_dim = 192` (`train.py:104-109`, `jepa.py:39`).
- **`pred_proj`** — applied to the predictor's outputs so they live in the same space as the targets (`train.py:111-116`, `jepa.py:53-54`).

### 3.3 Action embedder — `Embedder`

`module.py:189-214`

A 1-D conv (kernel size 1, acting per-frame) followed by a 2-layer MLP that maps the per-step action vector $(B, T, \text{frameskip} \times \text{action\_dim}) \to (B, T, \text{embed\_dim})$. Used at `jepa.py:43`.

### 3.4 Predictor — `ARPredictor`

`module.py:244-285`

A causal Transformer made of `ConditionalBlock`s (`module.py:88-111`). Each block is a standard pre-norm Attention + FFN, but the LayerNorms are **affine-less** and modulated by **AdaLN-Zero**: the per-step action embedding $c$ is fed through `SiLU + Linear` to produce $(\text{shift}, \text{scale}, \text{gate})$ for both the attention and the MLP sub-layers. The final linear projection inside `adaLN_modulation` is initialized to zero (`module.py:102-103`), so the block initially behaves like an identity — actions inject information gradually as training progresses.

The attention is **causal** (`is_causal=True`, `module.py:83`), so when predicting the embedding at position $t$ the predictor only attends to context at positions $\leq t$. A learned positional embedding of length `num_frames = history_size = 3` is added at the input (`module.py:262, 282`).

### 3.5 World model wrapper — `JEPA`

`jepa.py:11-55`

Bundles encoder + projector + action encoder + predictor + pred_proj.

- `JEPA.encode(info)` → fills `info["emb"]` and `info["act_emb"]`.
- `JEPA.predict(emb, act_emb)` → runs the predictor and projects its outputs through `pred_proj`.
- `JEPA.rollout(...)` and `JEPA.get_cost(...)` are used only for inference / planning (e.g. by `eval.py`), not during training.

### 3.6 SIGReg

`module.py:10-36`

The collapse-prevention regularizer. Given embeddings `proj` of shape $(T, B, D)$ (note: time as the leading dim — `train.py:41` does `emb.transpose(0, 1)`), it:

1. Samples `num_proj=1024` random unit vectors $A \in \mathbb{R}^{D \times P}$.
2. Computes 1-D projections $z = \text{proj} \cdot A$ of every embedding.
3. Evaluates the **Epps–Pulley** goodness-of-fit statistic between the empirical characteristic function of $z$ and that of $\mathcal{N}(0, 1)$, on a grid of `knots=17` frequencies in $[0, 3]$, weighted by a Gaussian window $\exp(-t^2/2)$ and Simpson-style trapezoid weights.
4. Returns the mean statistic over projections and time-steps.

> [!note] Why SIGReg prevents collapse
> Minimizing this term pushes every 1-D random projection of the embeddings toward $\mathcal{N}(0,1)$, which (for enough random directions) implies the joint distribution is isotropic Gaussian — ruling out collapse, low-rank solutions, and degenerate scaling.

Instantiated alongside the model at `train.py:138`.

---

## 4. The training forward pass — `lejepa_forward`

Defined in `train.py:18-46`. Called once per batch by Lightning via `spt.Module(..., forward=partial(lejepa_forward, cfg=cfg))`.

```python
ctx_len = cfg.wm.history_size   # 3
n_preds = cfg.wm.num_preds      # 1
lambd   = cfg.loss.sigreg.weight # 0.09
```

**Step-by-step:**

1. **Sanitize actions** (`train.py:26`): replace NaNs at sequence boundaries with zeros.
2. **Encode** (`train.py:28`, `jepa.py:29-45`): for the whole window of $T = \text{ctx\_len} + \text{n\_preds} = 4$ frames, run the ViT (with time folded into batch) to get the CLS token, then `projector` → `emb` of shape $(B, T, D)$. Also encode actions → `act_emb` of shape $(B, T, D)$.
3. **Slice context vs. target** (`train.py:33-36`):
   ```python
   ctx_emb = emb[:, :ctx_len]   # frames 0..2
   ctx_act = act_emb[:, :ctx_len]
   tgt_emb = emb[:, n_preds:]   # frames 1..3
   ```
   With `history_size=3, num_preds=1` this gives a perfectly shifted-by-one *next-step* prediction setup over 3 positions. Importantly the targets are the **same encoder's own outputs** — there is no separate target / EMA encoder, and there is no `.detach()` on `tgt_emb`. Gradients flow through both branches; only SIGReg keeps the representation from collapsing.
4. **Predict** (`train.py:37`, `jepa.py:47-55`): run the causal `ARPredictor` on `(ctx_emb, ctx_act)`. Because attention is causal, the prediction at time $t$ only depends on $(\text{emb}_{0..t}, \text{act}_{0..t})$, so it's a clean autoregressive next-state model. Output is then passed through `pred_proj`.
5. **Prediction loss** (`train.py:40`): MSE between predicted and target embeddings:
   `pred_loss = ((pred_emb - tgt_emb)**2).mean()`
6. **SIGReg loss** (`train.py:41`): apply SIGReg to the full embedding tensor `emb` (transposed to $(T, B, D)$). Computed on the *full* sequence — both context and target frames — so every encoded frame is pushed toward the isotropic-Gaussian prior.
7. **Total loss** (`train.py:42`):
   `loss = pred_loss + lambd * sigreg_loss`
8. **Logging** (`train.py:44-45`): every key in `output` whose name contains `"loss"` is logged (`train/pred_loss`, `train/sigreg_loss`, `train/loss`, and the corresponding `val/...` during validation).

`spt.Module` returns the dict, Lightning grabs `output["loss"]` for backprop.

---

## 5. Optimization & training loop

Defined in `train.py:126-178`.

| Parameter | Value | Source |
|---|---|---|
| Optimizer | AdamW | `lewm.yaml:30-33` |
| Learning rate | $5 \times 10^{-5}$ | `lewm.yaml` |
| Weight decay | $10^{-3}$ | `lewm.yaml` |
| Scheduler | LinearWarmupCosineAnnealing (per epoch) | `train.py:126-133` |
| Max epochs | 100 | `lewm.yaml:16-21` |
| Precision | bf16 | `lewm.yaml` |
| Gradient clipping | 1.0 | `lewm.yaml` |

- **DataModule** wraps the train/val loaders (`train.py:135`).
- **`spt.Module`** (`train.py:136-141`) wires the underlying model, the SIGReg module, the custom forward, and the optimizer config into a Lightning module.
- **Checkpointing**:
  - Lightning's own checkpointing is enabled (`enable_checkpointing=True`, `train.py:168`) and resumed from `run_dir / f"{output_model_name}_weights.ckpt"` (`train.py:175`).
  - `ModelObjectCallBack` (`utils.py:28-57`) pickles the full `world_model` Python object to disk at the end of every epoch via `torch.save(model, path)` so that downstream scripts (e.g. `eval.py`) can `torch.load` it directly.
- **Run management**: `spt.Manager` (`train.py:171-178`) ties trainer + module + data + checkpoint path together, and `manager()` starts training.
- **Logging**: optional Weights & Biases logger (`train.py:151-153`, config under `wandb:` in `lewm.yaml`).

---

## 6. What is and isn't trained

| Component | Trained? | Notes |
|---|---|---|
| ViT encoder | Yes | From scratch |
| Projector | Yes | From scratch |
| Action embedder | Yes | From scratch |
| AR predictor | Yes | From scratch, AdaLN gates init to zero |
| Predictor projector | Yes | From scratch |
| SIGReg | No | Buffers only (`t`, `phi`, `weights`); projection matrix $A$ sampled fresh each call |

> [!info] AdaLN-Zero initialization
> The predictor's AdaLN gates start at zero (`module.py:102-103`), which means at step 0 the predictor is essentially the identity on the embedding stream — actions only start to influence predictions as those gates move away from zero.

---

## 7. Compact pseudocode

```python
# ----- config -----
T_ctx  = cfg.wm.history_size    # 3
T_pred = cfg.wm.num_preds       # 1
T      = T_ctx + T_pred          # window length, 4
λ      = cfg.loss.sigreg.weight  # 0.09

# ----- model -----
encoder        = ViT_tiny(patch=14, img=224)           # CLS-token features
projector      = MLP(d_vit -> embed_dim)                # post-encoder
action_encoder = Conv1d + MLP(act_dim -> embed_dim)
predictor      = CausalTransformer(                     # AR next-state model
    blocks=ConditionalBlock(AdaLN_Zero on action),
    pos_emb on T_ctx positions
)
pred_proj      = MLP(hidden -> embed_dim)
sigreg         = SIGReg(knots=17, num_proj=1024)        # Epps-Pulley vs N(0,1)

θ     = params(encoder, projector, action_encoder, predictor, pred_proj)
opt   = AdamW(θ, lr=5e-5, wd=1e-3)
sched = LinearWarmupCosineAnnealing(opt, per_epoch)

# ----- training loop -----
for epoch in range(max_epochs):
    for batch in train_loader:                          # window of T frames
        pixels = batch["pixels"]                        # (B, T, C, H, W)
        action = nan_to_num(batch["action"], 0.0)       # (B, T, frameskip*A)

        # 1. Encode all T frames at once
        z_pix = encoder(flatten_time(pixels)).cls       # (B*T, d_vit)
        emb   = projector(z_pix).view(B, T, D)          # (B, T, D)
        a_emb = action_encoder(action)                   # (B, T, D)

        # 2. Build context / target (no stop-gradient on target)
        ctx_e = emb[:, :T_ctx]                           # frames 0..T_ctx-1
        ctx_a = a_emb[:, :T_ctx]
        tgt_e = emb[:, T_pred:]                          # frames shifted by T_pred

        # 3. Causal AR prediction in latent space
        h      = predictor(ctx_e, ctx_a)                 # (B, T_ctx, hidden)
        pred_e = pred_proj(h)                            # (B, T_ctx, D)

        # 4. Losses
        L_pred   = mean((pred_e - tgt_e) ** 2)           # JEPA latent MSE
        L_sigreg = sigreg(emb.transpose(0, 1))           # isotropy prior
        L        = L_pred + λ * L_sigreg

        # 5. Step
        opt.zero_grad()
        L.backward()
        clip_grad_norm_(θ, 1.0)
        opt.step()

    sched.step()
    save_pickle(world_model, f"{run_dir}/lewm_epoch_{epoch+1}_object.ckpt")
```

> [!abstract] Inference
> Planning is done elsewhere via `JEPA.rollout` / `JEPA.get_cost` (`jepa.py:61-153`): given an initial frame and a batch of candidate action sequences, the predictor is unrolled autoregressively in latent space and each candidate is scored by the MSE between its final predicted embedding and the encoded goal image.
