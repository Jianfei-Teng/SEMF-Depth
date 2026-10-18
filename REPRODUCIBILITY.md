# Training configuration

The executable training and evaluation commands are in the
[README](README.md#training). The defaults are seed **1234**, input
**480 × 640**, batch size **8**, gradient accumulation **2** and **4** workers.

## Optimizer and schedule

Both stages use AdamW with betas `(0.9, 0.999)` and epsilon `1e-8`.
The following are the actual initial group learning rates.

| Stage | Group | Parameters | Learning rate | Weight decay |
|---|---|---|---:|---:|
| Backbone, epochs 0–49 | 0 | Depth decoder and pose networks | `1e-4` | `0.01` |
| Backbone, epochs 0–49 | 1 | MPViT depth encoder | `5e-5` | `0.01` |
| Refinement, epochs 50–69 | 0 / 1 | SMF and pose decoder; fixed SEE parameters are also listed | `2e-4` | `0.01` / `0` |
| Refinement, epochs 50–69 | 2 / 3 | Depth decoder and pose encoder; depth encoder is frozen | `2e-6` | `0.01` / `0` |

Refinement separates matrix weights from one-dimensional parameters for weight
decay and starts a fresh optimizer and scheduler. Learning rates are used
directly, without batch-size scaling. The first stage uses epoch-wise
`ExponentialLR(gamma=0.9)`. Refinement uses
`CosineAnnealingLR(T_max=65, eta_min=1e-7)` with 20 epoch-wise steps and no warmup.

## Initialization and update counts

Refinement starts from the supplied 50-epoch backbone in `./weights`, containing
`encoder.pth`, `depth.pth`, `pose_encoder.pth` and `pose.pth`. For a new two-stage
run, it starts from `./tmp/SEMF/backbone/models/weights_49`.

There are **4,072 valid training targets** and **676 validation targets** after
requiring both temporal neighbors. Batch 8 gives 509 microbatches and **255
optimizer updates per epoch** with accumulation 2, including the last partial
accumulation window. This gives **12,750 updates** for a new 50-epoch backbone
run and **5,100 updates** for refinement.

The supplied backbone optimizer stores **45,800** steps. The retained Full
`weights_69/adam.pth` stores **5,100** refinement updates, global epoch 69 and
20 cosine-scheduler steps with the group rates listed above.

Every epoch's complete checkpoint is retained, including model weights,
optimizer, scheduler, epoch, update count and Python/NumPy/PyTorch/CUDA random
states. The run also saves the command, arguments, effective data lists,
TensorBoard events and `training_log.jsonl`.

## Gradient paths and loss

```text
F_dec ---------------------------> SMF -> auxiliary inverse depth
F_dec -> detach -> SEE -> detach F_edge -> SMF
                    \-> detach E_dense -> routing mask
RGB -> grayscale -> high-pass -> E_rgb -> routing mask
(backbone inverse depth, auxiliary inverse depth, routing mask) -> fusion -> loss
```

SEE supplies fixed projections and receives no gradient from the depth loss.
The decoder and SMF are updated through the fused prediction. Training uses
photometric reprojection and standard edge-aware smoothness, averaged over
four scales. No multi-head or edge-supervision loss is enabled.

## Code version

The original public implementation is
[`692a823c90bdab9005c715bb349ad80acab46ca4`](https://github.com/Jianfei-Teng/SEMF-Depth/tree/692a823c90bdab9005c715bb349ad80acab46ca4).
Use `git rev-parse HEAD` to identify the current checkout. Each new run records
its Git commit and tracked working-tree status in `run_*.json` and checkpoint
metadata.
