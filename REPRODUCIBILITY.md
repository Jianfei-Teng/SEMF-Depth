# Training and reproducibility

## Commands for this release

Run from the repository root with the dataset at `./ancient_architecture_data`
and the supplied 50-epoch backbone at `./weights`:

```bash
python train.py --stage auto --data_path ./ancient_architecture_data --backbone_weights ./weights --log_dir ./tmp --model_name SEMF --height 480 --width 640 --batch_size 8 --accumulation_steps 2 --num_workers 4 --seed 1234 --learning_rate 2e-6 --head_learning_rate 2e-4
```

`auto` runs refinement at global epochs **50–69** when the four backbone model
files are present. Without them, it trains backbone epochs **0–49** first and
then refines its `weights_49`. A partial checkpoint is an error. To require the
supplied backbone instead of allowing pretraining, use `--stage refine`.

The equivalent explicit commands are:

```bash
python train.py --stage backbone --data_path ./ancient_architecture_data --log_dir ./tmp --model_name SEMF --seed 1234 --batch_size 8 --accumulation_steps 2 --decoder_learning_rate 1e-4 --backbone_learning_rate 5e-5 --backbone_lr_decay 0.9
python train.py --stage refine --data_path ./ancient_architecture_data --backbone_weights ./tmp/SEMF/backbone/models/weights_49 --log_dir ./tmp --model_name SEMF --seed 1234 --batch_size 8 --accumulation_steps 2 --learning_rate 2e-6 --head_learning_rate 2e-4
```

The from-scratch backbone path uses the retained MonoViT two-group convention.
An optional MPViT initialization can be supplied with `--encoder_pretrained`.
The pose encoder defaults to ImageNet initialization; `--weights_init scratch`
disables it. Refinement loads the supplied depth and pose weights without an
ImageNet download. No learning rate is implicitly scaled by batch size.

## Parameter groups and optimization updates

AdamW uses betas `(0.9, 0.999)` and epsilon `1e-8`.

| Phase | Group | Initial learning rate | Weight decay | Trainable modules |
|---|---|---:|---:|---|
| Backbone, 0–49 | 0 | `1e-4` | `0.01` | HRDepth decoder, pose encoder and pose decoder |
| Backbone, 0–49 | 1 | `5e-5` | `0.01` | MPViT depth encoder |
| Refinement, 50–69 | 0 / 1 | `2e-4` | `0.01` / `0` | SMF and pose decoder; SEE is also listed but receives no depth-loss gradient |
| Refinement, 50–69 | 2 / 3 | `2e-6` | `0.01` / `0` | HRDepth decoder and pose encoder; MPViT depth encoder is frozen |

Refinement groups separate matrix weights from one-dimensional parameters. The
new phase starts a **fresh AdamW and scheduler**; it does not reuse the backbone
optimizer. The backbone fallback uses epoch-wise `ExponentialLR(gamma=0.9)`.
Refinement uses `CosineAnnealingLR(T_max=65, eta_min=1e-7)`, stepped once after
each of its 20 epochs, without another warmup. Consequently, its final saved
head/backbone rates are `1.5682807143577903e-4` and `1.5896615093945994e-6`;
the 20-epoch refinement does not reach `eta_min`.

The supplied dataset contains 4074 left training images and 678 left validation
images. Requiring offsets `[0, -1, 1]` excludes each sequence's two endpoints,
leaving **4072 training targets and 676 validation targets**. The original TXT
files also list right-camera entries; only `image_02` is used. Exact lists used
for training are saved alongside the options. Reference left-only lists are
included in `splits/ancient`.

With `drop_last=True`, batch 8 and accumulation 2, there are 509 microbatches
and **255 optimizer updates per epoch** (including the final one-microbatch
accumulation window). This gives **12,750 backbone updates** for a new 50-epoch
run and **5,100 refinement updates** for 20 epochs. In general:

```text
microbatches_per_epoch = floor(number_of_targets / batch_size)
updates_per_epoch = ceil(microbatches_per_epoch / accumulation_steps)
```

Nonfinite loss/gradients stop the run. Actual completed updates, learning rates
and epoch durations are written to `training_log.jsonl`; the startup plan reports
the expected counts before training. Validation and logging do not increment the
optimizer-update count.

## Checkpoint and code identification

The refinement initialization is the **50th completed backbone epoch**, indexed
`weights_49`, containing `encoder.pth`, `depth.pth`, `pose_encoder.pth` and
`pose.pth`. The legacy supplied folder is simply named `weights/`; it contains
those four models plus a bare optimizer state. This directory is explicitly
treated as the author-supplied 50-epoch backbone. A filename alone is not used to
infer the epoch of arbitrary resume checkpoints.

The retained Full `weights_69/adam.pth` records global epoch 69,
the four refinement initial group rates above, cosine `T_max=65`, 20 scheduler
steps, and **5,100 updates** in its active Adam states. The supplied backbone
optimizer records initial group rates `1e-4` and `5e-5` and step count **45,800**.
The **12,750** backbone updates above apply to this release's new 50-epoch
from-scratch run.

The original public implementation is pinned at
[`692a823c90bdab9005c715bb349ad80acab46ca4`](https://github.com/Jianfei-Teng/SEMF-Depth/tree/692a823c90bdab9005c715bb349ad80acab46ca4).
This release makes the training entry point explicit and removes inactive
modules. The commands above use seed **1234** and the retained refinement
configuration.

For every new run, `run_*.json`, `models/opt.json` and the complete checkpoints
record the invocation arguments, Git commit (when run in a Git checkout), tracked
working-tree status, seed, source checkpoint, optimizer groups, scheduler and
update count. Obtain the release identifier using `git rev-parse HEAD`.

Resume with `python train.py --resume` using the same data and options, or pass
an explicit complete checkpoint directory. All enabled models, AdamW, scheduler,
epoch, update count and Python/NumPy/PyTorch/CUDA random states are saved. All
complete per-epoch checkpoints are retained in both phases; no old checkpoint
is automatically deleted. Restarted data-loader worker/prefetch streams can differ, so checkpoint
resume is not a guarantee of bitwise-identical augmentation with multiple workers.

## Gradient paths and loss

```text
RGB -> MPViT encoder -> HRDepth decoder -> backbone inverse depth -----+
                              |                                     |
                              +-> F_dec -> SMF -> auxiliary depth ---+-> routed blend -> loss
                              |          ^                          ^
                              +-> detach -> SEE -> detach F_edge ---+|
                                              +-> detach E_dense ----+|
RGB -> grayscale -> Butterworth high-pass (no grad) -> E_rgb ----------+
```

The input feature is detached before SEE. Both `F_edge` and `E_dense` are
detached before SMF/routing. SEE therefore supplies **fixed feature projections**;
its semantic edge parameters are not learned by the reported depth loss. The
direct `F_dec -> SMF` path trains the decoder and SMF. The loss remains the
photometric reprojection and standard edge-aware smoothness objective, averaged
over four output scales. No multi-head, edge-supervision or separation loss is
enabled. The unused `gate_conv` is absent; legacy gate keys are accepted only
for loading the old checkpoints.
