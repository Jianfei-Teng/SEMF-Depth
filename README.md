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

The model uses the Python CNN helpers in `mmcv-lite` and `mmengine`;
compiled MMCV CUDA operators are not required.

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
  --seed 1234 --learning_rate 2e-6 --head_learning_rate 2e-4
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

Use `python train.py --print_config` to check the training plan without
starting training, and `python train.py --help` for available options.
View logs with `tensorboard --logdir ./tmp/SEMF`.

The two-stage learning rates, update counts, initialization and gradient paths
are summarized in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Evaluation

```bash
python evaluate_metrics.py \
  --weights ./tmp/SEMF/refinement/models/weights_69 \
  --data-root ./ancient_architecture_data \
  --output-dir ./results/SEMF
```

This evaluates Abs Rel, Sq Rel, RMSE, RMSE log, the three accuracies at
`1.25`, `1.25^2`, `1.25^3`, and CD. The default manifest contains 489 pairs
across scenes 01–12. Use `--scenes 01 11 12`, for example, to select scenes,
or `--dry-run` to check image/GT pairs and intrinsics before inference.

For the separately stored original and additional test sets:

```bash
python evaluate_metrics.py \
  --weights ./tmp/SEMF/refinement/models/weights_69 \
  --test-root ./test --extra-test-root ./add_new_test_11 \
  --gt-root ./ancient_architecture_data/splits/eigen/gt_npy \
  --scenes 01 03 05 07 08 10 11 12 加3 加5 加9 加10 \
  --output-dir ./results/selected12
```

Evaluation uses the Garg crop, per-image median scaling and a scoring range
of 0.001–80 m. CD uses `splits/相机内参.json`, scaled by the full depth-map
dimensions. It is the sum of the two mean squared nearest-neighbor distances
in m², with at most 20,000 sampled points per cloud; `--cd-max-points 0` uses
all points.

TXT reports contain per-scene results, an equal-scene mean and an image mean.
The two-root command produces three TXT files: one per test set and one
combined mean. See `python evaluate_metrics.py --help` for other options.

## License

See [LICENSE](LICENSE). The inherited Monodepth2 code is for non-commercial use.
