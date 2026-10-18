# Texture-Structure Decoupled Monocular Depth Estimation for Complex 3D Reconstruction Scenes

PyTorch implementation of SEMF-Depth. This release provides the complete model,
two-phase training, fixed data lists, and one evaluator for seven depth metrics
plus calibrated Chamfer distance (CD).

## Installation

Use Python 3.10 or newer. Install a matching PyTorch/torchvision build for your
GPU using the [official PyTorch instructions](https://pytorch.org/get-started/locally/),
then install the remaining requirements:

```bash
pip install -r requirements.txt
```

Only the Python CNN helpers from MMCV are needed: `mmcv-lite` and `mmengine`
are sufficient; no compiled MMCV CUDA operators are used.

## Files and relative paths

Run commands from the repository root. Place the dataset and optional 50-epoch
backbone in this layout:

```text
SEMF-Depth/
├── train.py
├── evaluate_metrics.py
├── ancient_architecture_data/
│   ├── train/image_02/data/*.jpg
│   ├── val/image_02/data/*.jpg
│   ├── test/<scene>/*
│   └── splits/
│       ├── eigen_zhou/train_files.txt
│       ├── eigen_zhou/val_files.txt
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

Dataset images, ground-truth arrays, checkpoints and generated results are
excluded from the repository. Training reads the supplied dataset's TXT lists
and keeps only left-camera targets with exact temporal neighbors. You can
instead pass `--split_dir ./splits/ancient` for the included effective lists.
Missing frames are reported; they are not replaced with unrelated images.

## Training

```bash
python train.py --data_path ./ancient_architecture_data --backbone_weights ./weights --log_dir ./tmp --model_name SEMF --seed 1234
```

- A complete 50-epoch backbone in `./weights` or `./weights/weights_49` starts
  20 refinement epochs, globally numbered 50–69.
- If the backbone is absent, the same command first trains backbone epochs
  0–49, then refines the resulting `weights_49` for 20 epochs.
- An incomplete checkpoint is an error. `--stage refine` requires an existing
  backbone; `--stage backbone` runs only the initial 50 epochs.
- `--resume` restores the latest complete checkpoint in this run, including
  AdamW and the scheduler. An explicit checkpoint path is also accepted.

The default input is 480 × 640, batch size 8, gradient accumulation 2 and seed
1234. Refinement uses actual head-group LR `2e-4` and backbone-group LR
`2e-6`; the depth encoder is frozen. There is no hidden batch-based LR scaling.
Check the complete configuration and update counts without starting training:

```bash
python train.py --print_config
```

Each phase saves to `./tmp/SEMF/<phase>/`. TensorBoard logs, used TXT lists,
run command, code version, optimizer groups and update counts are recorded.
All complete epoch checkpoints are retained. Refinement
ends at `./tmp/SEMF/refinement/models/weights_69`.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact commands, phase learning
rates, optimization-update counts, source checkpoint, gradient paths, historical
result provenance and code-version identification.

## Evaluation: seven depth metrics + CD

```bash
python evaluate_metrics.py --weights ./tmp/SEMF/refinement/models/weights_69 --data-root ./ancient_architecture_data --output-dir ./results/SEMF
```

This is the only evaluation entry point. It computes Abs Rel, Sq Rel, RMSE,
RMSE log, the three accuracies at `1.25`, `1.25^2`, `1.25^3`, and CD from
the same prediction, valid pixels and median scale alignment.

The default fixed pairing manifest is `splits/eigen/test_pairs.txt`, relative
to the dataset root (a copy inside the dataset's `splits/eigen/` takes priority).
It identifies the 489 image/GT pairs present in this dataset across scenes 01–12.
`--scenes 01 11 12`, for example, restricts the fixed cohort. The legacy
`test_files.txt` is retained separately; the evaluator uses explicit image/GT
pairs so scene-local filenames are unambiguous. The default dataset cohort is
not a replacement for a separately specified paper evaluation cohort.

For the separately stored original/additional test sets, supply their roots
and the fixed scene selection explicitly:

```bash
python evaluate_metrics.py --weights ./weights_69 --test-root ./test --extra-test-root ./add_new_test_11 --gt-root ./ancient_architecture_data/splits/eigen/gt_npy --scenes 01 03 05 07 08 10 11 12 加3 加5 加9 加10 --output-dir ./results/selected12
```

Original numeric image IDs map into the sorted GT list as in the released
evaluator (image 0000000000 pairs with 001.npy). Additional test images use
matching JPG/NPY stems. The supplied fixed manifest makes these pairings explicit.
`--dry-run` checks all pairs and valid depth support without inference.

Evaluation uses the Garg crop as a validity mask, per-image median scaling,
network depth conversion 0.1–80 m, and scoring range 0.001–80 m. CD uses
`splits/相机内参.json`, converted from normalized coordinates using the complete
depth-map width and height. Cropping does not change the camera origin.

CD is the sum of the two mean **squared** nearest-neighbor distances, in m².
By default each cloud is independently sampled to at most 20,000 points with
seeds 1234/1235; use `--cd-max-points 0` for full clouds. TXT reports record
the protocol, per-scene metrics, equal-scene mean and image mean separately.
Existing results are protected unless `--overwrite` is passed. When two test
roots are used, the evaluator exports one TXT per root and one combined mean TXT.

## Model and loss

SEE uses a grayscale Butterworth response and fixed projections of detached
decoder features. `F_edge` and `E_dense` are detached before SMF. The decoder
and fusion head receive gradients through the final routed depth prediction;
SEE receives none from that objective. The loss is photometric reprojection
plus standard edge-aware smoothness. No multi-head loss is enabled.

The unused gate module has been removed. Old checkpoints containing its keys
remain loadable without changing active model weights.

## License

See [LICENSE](LICENSE). The inherited Monodepth2 code is for non-commercial use.
