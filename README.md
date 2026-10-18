# Texture-Structure Decoupled Monocular Depth Estimation for Complex 3D Reconstruction Scenes

PyTorch implementation of SEMF-Depth.

![Overview of SEMF-Depth](assets/overall_architecture.png)

## Installation

Install Python 3.10+, PyTorch 2.0+ and a matching torchvision build, then run:

```bash
git clone https://github.com/Jianfei-Teng/SEMF-Depth.git
cd SEMF-Depth
pip install -r requirements.txt
```

The model uses `mmcv-lite`; compiled MMCV CUDA operators are not required.

## Data and checkpoints

Place the dataset in `./ancient_architecture_data` and the supplied 50-epoch
backbone in `./weights`:

```text
ancient_architecture_data/
├── train/image_02/data/
├── val/image_02/data/
├── test/
└── splits/eigen_zhou/{train,val}_files.txt
weights/
├── encoder.pth
├── depth.pth
├── pose_encoder.pth
└── pose.pth
```

Training uses left-view triplets with offsets `[0, -1, 1]`. Fixed train/validation
lists are also provided in `splits/ancient`; use `--split_dir ./splits/ancient`
to select them. Test pairs and camera intrinsics are in `splits/eigen/test_pairs.txt`
and `splits/相机内参.json`.

## Training

```bash
python train.py \
  --stage auto --data_path ./ancient_architecture_data \
  --backbone_weights ./weights --log_dir ./tmp --model_name SEMF \
  --height 480 --width 640 \
  --batch_size 8 --accumulation_steps 2 --num_workers 4 --seed 1234 \
  --backbone_learning_rate 5e-5 --decoder_learning_rate 1e-4 \
  --learning_rate 2e-6 --head_learning_rate 2e-4
```

With a backbone in `./weights` or `./weights/weights_49`, this runs refinement
for epochs 50–69. Otherwise, it first trains the backbone for epochs 0–49.
Use `--stage refine` to require an existing backbone, or `--stage backbone`
to run only the first stage. For ImageNet encoder initialization, add
`--encoder_pretrained ./ckpt/mpvit_small.pth`.

Both stages use AdamW with the following initial learning rates:

| Phase | Parameter groups | Initial learning rates |
|---|---|---|
| Backbone (50 epochs) | MPViT encoder; depth decoder and pose networks | `5e-5`; `1e-4` |
| Refinement (20 epochs) | Depth decoder and pose encoder; SMF and pose decoder | `2e-6`; `2e-4` |

The backbone uses `ExponentialLR(gamma=0.9)`. Refinement starts a fresh
optimizer and `CosineAnnealingLR(T_max=65, eta_min=1e-7)`, stepped once per
epoch. The depth encoder and SEE are fixed during refinement; the decoder
and SMF are optimized using photometric reprojection and smoothness losses.

The default lists contain 4,072 valid targets: 255 optimizer updates per epoch
at the batch settings above, or 12,750 for a new backbone and 5,100 for refinement.
The supplied 50-epoch backbone's Adam state records 45,800 updates;
the retained Full checkpoint records 5,100 refinement updates.
The architecture follows reference commit
[`692a823`](https://github.com/Jianfei-Teng/SEMF-Depth/tree/692a823c90bdab9005c715bb349ad80acab46ca4).
Each run saves its command, seed, checkout commit and update counts.

All epoch checkpoints are retained. The final model is saved to
`./tmp/SEMF/refinement/models/weights_69`. Append `--resume` to restore a run,
including optimizer, scheduler and random states. Use `--print_config` to inspect
the configuration, and `tensorboard --logdir ./tmp/SEMF` to view logs.

## Evaluation

```bash
python evaluate_metrics.py \
  --weights ./tmp/SEMF/refinement/models/weights_69 \
  --data-root ./ancient_architecture_data --output-dir ./results/SEMF
```

This computes the seven depth metrics and CD on the 489 RGB–GT pairs in the
bundled manifest. Use `--scenes 01 11 12` to select scenes or `--dry-run` to
check the inputs without inference.

For the paper's 12 scenes stored in separate original and additional test folders:

```bash
python evaluate_metrics.py \
  --weights ./tmp/SEMF/refinement/models/weights_69 \
  --test-root ./test --extra-test-root ./add_new_test_11 \
  --gt-root ./ancient_architecture_data/splits/eigen/gt_npy \
  --scenes 01 03 05 07 08 10 11 12 加3 加5 加9 加10 \
  --output-dir ./results/selected12
```

This also selects 489 pairs. Adding original scene `06` gives 544 pairs.
In the packaged dataset, scene IDs `02/04/06/09` instead correspond to
additional scenes `3/5/10/9`. JPG, JPEG and PNG inputs are supported.

Evaluation uses the Garg crop, per-image median scaling and a 0.001–80 m range.
CD uses full-depth-map intrinsics and the sum of bidirectional mean squared
nearest-neighbor distances (m²), sampling up to 20,000 points per cloud.
The two-folder command writes three TXT reports: one per test set and one
combined report, with scene means and image means. See `--help` for options.

## License

See [LICENSE](LICENSE). The inherited Monodepth2 code is for non-commercial use.
