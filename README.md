# Texture-Structure Decoupled Monocular Depth Estimation for Complex 3D Reconstruction Scenes

PyTorch implementation of SEMF-Depth, with two-stage training and evaluation
of seven depth metrics and Chamfer distance (CD).

## Installation

Use Python 3.10 or newer and PyTorch 2.0 or newer. Install a matching
PyTorch/torchvision build for your GPU following the
[official instructions](https://pytorch.org/get-started/locally/), then run:

```bash
git clone https://github.com/Jianfei-Teng/SEMF-Depth.git
cd SEMF-Depth
pip install -r requirements.txt
```

The model uses `mmcv-lite` and `mmengine`; compiled MMCV CUDA operators
are not required.

## Data and pretrained backbone

Place the dataset and the supplied 50-epoch backbone as follows. All commands
below run from the repository root.

```text
SEMF-Depth/
├── ancient_architecture_data/
│   ├── train/image_02/data/*.jpg
│   ├── val/image_02/data/*.jpg
│   ├── test/<scene>/...
│   └── splits/
│       ├── eigen_zhou/{train,val}_files.txt
│       └── eigen/gt_npy/*.npy
├── splits/
│   ├── ancient/{train,val}_files.txt
│   ├── eigen/{test_files,test_pairs}.txt
│   └── 相机内参.json
└── weights/
    ├── encoder.pth
    ├── depth.pth
    ├── pose_encoder.pth
    └── pose.pth
```

Training uses only the left camera (`image_02`) with frame offsets `[0, -1, 1]`.
It reads the dataset's train/validation lists and keeps targets with both
temporal neighbors. The effective left-only lists are also included in
`splits/ancient`; select them with `--split_dir ./splits/ancient`.
Test image/GT pairs are defined in `splits/eigen/test_pairs.txt`.

## Training

```bash
python train.py \
  --stage auto \
  --data_path ./ancient_architecture_data \
  --backbone_weights ./weights \
  --log_dir ./tmp --model_name SEMF \
  --height 480 --width 640 \
  --batch_size 8 --accumulation_steps 2 --num_workers 4 \
  --seed 1234 --learning_rate 2e-6 --head_learning_rate 2e-4 \
  --backbone_learning_rate 5e-5 --decoder_learning_rate 1e-4
```

With the backbone in `./weights` or `./weights/weights_49`, this runs 20
refinement epochs, numbered 50–69. If it is absent, the same command first
trains the backbone for 50 epochs and then refines `weights_49`.
Use `--stage refine` to require the supplied backbone, or `--stage backbone`
to run only the first stage. Optional encoder initialization for the first
stage is supplied with `--encoder_pretrained path/to/mpvit.pth`.

Outputs are saved under `./tmp/SEMF/backbone/` and `./tmp/SEMF/refinement/`.
Every epoch's complete checkpoint is retained. The final model is
`./tmp/SEMF/refinement/models/weights_69`.

To resume the same run, append `--resume` to its training command. This restores
the models, optimizer, scheduler, epoch and random states. An explicit checkpoint
directory can also follow `--resume`.

Both stages use AdamW with betas `(0.9, 0.999)` and epsilon `1e-8`.
Initial parameter-group learning rates are:

| Phase | Epochs | Parameter groups | Initial learning rates |
|---|---|---|---|
| Backbone | 0–49 | Depth decoder and pose networks; MPViT encoder | `1e-4`; `5e-5` |
| Refinement | 50–69 | SMF and pose decoder; depth decoder and pose encoder | `2e-4`; `2e-6` |

The backbone uses `ExponentialLR(gamma=0.9)`. Refinement starts a fresh
optimizer and `CosineAnnealingLR(T_max=65, eta_min=1e-7)`, with 20 epoch-wise
steps and no warmup or batch-size LR scaling. Weight decay is `0.01`;
one-dimensional parameters use `0` during refinement. The depth encoder is
frozen during refinement. Fixed SEE parameters remain in the head groups.

The default lists give 4,072 training targets, 509 batches and 255 optimizer
updates per epoch, including the last accumulation window: 12,750 updates for
a new backbone run and 5,100 for refinement. The supplied 50-epoch backbone
records 45,800 optimizer steps; the retained Full `weights_69` records 5,100
refinement updates.

Training uses photometric reprojection and edge-aware smoothness losses.
SEE is fixed; the decoder and SMF are updated through the fused prediction.
See the overall architecture figure in the paper for the loss and gradient paths.

Reference implementation:
[`692a823`](https://github.com/Jianfei-Teng/SEMF-Depth/tree/692a823c90bdab9005c715bb349ad80acab46ca4).
Each run records its checkout commit, command, seed and update counts.

Use `python train.py --print_config` to inspect the plan without training.
View logs with `tensorboard --logdir ./tmp/SEMF`.

## Evaluation

```bash
python evaluate_metrics.py \
  --weights ./tmp/SEMF/refinement/models/weights_69 \
  --data-root ./ancient_architecture_data \
  --output-dir ./results/SEMF
```

This evaluates Abs Rel, Sq Rel, RMSE, RMSE log, the three accuracies at
`1.25`, `1.25^2`, `1.25^3`, and CD. The bundled manifest contains 489 RGB/GT
pairs. Use `--scenes 01 11 12`, for example, to select scenes, or `--dry-run`
to check pairs and intrinsics without inference.

The packaged scene IDs `02/04/06/09` correspond to additional scenes
`3/5/10/9`. The two-root command below uses the original scene IDs.

For the paper's 12 scenes in the separately stored original and additional sets:

```bash
python evaluate_metrics.py \
  --weights ./tmp/SEMF/refinement/models/weights_69 \
  --test-root ./test --extra-test-root ./add_new_test_11 \
  --gt-root ./ancient_architecture_data/splits/eigen/gt_npy \
  --scenes 01 03 05 07 08 10 11 12 加3 加5 加9 加10 \
  --output-dir ./results/selected12
```

This selection contains 489 pairs. Adding `06` to `--scenes` includes the
55 images in the original scene 06, giving 13 scenes and 544 pairs.
RGB files may be JPG, JPEG or PNG; ground-truth depth alone is not a model input.

Evaluation uses the Garg crop, per-image median scaling and a scoring range
of 0.001–80 m. CD uses `splits/相机内参.json`, scaled by the full depth-map
dimensions. It is the sum of the two mean squared nearest-neighbor distances
in m², with at most 20,000 sampled points per cloud; `--cd-max-points 0` uses
all points.

The two-root command produces three TXT files: one per test set and one
combined report, with per-scene results, an equal-scene mean and an image mean.
See `python evaluate_metrics.py --help` for other options.

## License

See [LICENSE](LICENSE). The inherited Monodepth2 code is for non-commercial use.
