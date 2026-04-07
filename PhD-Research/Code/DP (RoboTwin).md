
## Data Preprocessing

  

```

FOR each episode HDF5 file:

states ← joint_action/vector // (L, D)

images ← JPEG_DECODE(observation/head_camera/rgb) // (L, H, W, 3)

obs_states.APPEND(states[0 : L-1]) // state at t

actions.APPEND(states[1 : L]) // action = state at t+1

obs_images.APPEND(images[0 : L-1]) // image at t

  

SAVE_ZARR(head_camera=NCHW(obs_images), state=obs_states, action=actions, episode_ends)

```

  

## Training

  

```

// ── Setup ──

replay_buf ← LOAD_ZARR(zarr_path)

normalizer.FIT(replay_buf["action"], replay_buf["state"], mode="min-max → [-1,1]")

  

resnet18 ← ResNet18(GroupNorm instead of BatchNorm, ImageNet normalization)

unet ← ConditionalUnet1D(in=D_action, cond=obs_dim×3, dims=[256,512,1024], FiLM=True)

scheduler ← DDPMScheduler(T=100, β=[0.0001,0.02], cosine schedule, predict=ε)

optimizer ← AdamW(lr=1e-4, betas=[0.95,0.999], wd=1e-6)

lr_sched ← CosineWithWarmup(warmup=500)

ema_model ← DEEP_COPY(model); ema_decay(step) = clamp(1-(1+step)^-0.75, 0, 0.9999)

  

// ── Loop ──

FOR epoch = 0 TO 599:

FOR batch IN DataLoader(batch_size=128, horizon=8, pad_before=2, pad_after=5):

  

// 1. Normalize

nobs ← normalizer(batch.obs) // images→[-1,1], state→[-1,1]

naction ← normalizer(batch.action) // (B, 8, D_action)

  

// 2. Encode observations → global conditioning

obs_flat ← nobs[:, 0:3].RESHAPE(B×3, ...) // first 3 timesteps

obs_feat ← resnet18(CROP(obs_flat)) ⊕ obs_flat["agent_pos"]

global_cond ← obs_feat.RESHAPE(B, obs_dim × 3)

  

// 3. Forward diffusion

ε ← RANDN_LIKE(naction) // sample noise

t ← RANDINT(0, 100, size=B) // sample timestep

x_t ← √ᾱ_t · naction + √(1-ᾱ_t) · ε // noisy trajectory

  

// 4. Predict noise & compute loss

ε_pred ← unet(x_t, t, global_cond) // UNet denoises

loss ← MEAN( ‖ε_pred − ε‖² ) // MSE on predicted vs true noise

  

// 5. Update

loss.BACKWARD()

optimizer.STEP(); lr_sched.STEP()

ema_model ← decay · ema_model + (1-decay) · model

  

// ── Eval (every epoch) ──

val_loss ← MEAN(ComputeLoss(val_batches))

  

// ── Checkpoint (every 300 epochs) ──

IF (epoch+1) % 300 == 0: SAVE(model, ema_model, optimizer, epoch)

```

  

## Key Formulas

  

```

Forward: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε ε ~ N(0,I), t ~ U{0..99}

Loss: L = E[ ‖ε_θ(x_t, t, c) − ε‖² ]

EMA: θ_ema ← decay · θ_ema + (1-decay) · θ decay = clamp(1-(1+step)^-0.75, 0, 0.9999)

FiLM: [s, b] = Linear(cond); out = s ⊙ feat + b

```