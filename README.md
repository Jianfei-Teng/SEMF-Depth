<h1 align="center">Texture-Structure Decoupled Monocular Depth<br>
Estimation in Complex 3D Reconstruction Scenes</h1>

<p align="center">Jianfei Teng · Xiaoling Yao · Lihua Hu · Yang Ma · Jifu Zhang</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.9.1"></a>
  <img src="https://img.shields.io/badge/GPU-RTX%205090%20%C3%97%201-76B900?logo=nvidia&amp;logoColor=white" alt="One NVIDIA RTX 5090 GPU">
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#data-and-checkpoints">Data and checkpoints</a> ·
  <a href="#training">Training</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#visualization">Visualization</a>
</p>

## Overview

Official PyTorch implementation of SEMF-Depth.

SEMF-Depth is a self-supervised monocular depth estimation framework. Its structural
edge enhancement (SEE) strategy fuses projected features with decoder features
to produce an auxiliary depth prediction. Its structure-edge-aware masked fusion
(SMF) mechanism derives soft masks from RGB-frequency responses to combine the
backbone and auxiliary predictions.

### Structural Edge Enhancement and Structure-Edge-Aware Masked Fusion

SEE enhances the decoder representation through feature fusion. SMF combines
the resulting auxiliary prediction with the backbone prediction through spatial
weighting.

## Installation

The experiments used Python 3.10+, PyTorch 2.9.1, torchvision 0.24.1 and CUDA 13.0:

```bash
git clone https://github.com/Jianfei-Teng/SEMF-Depth.git
cd SEMF-Depth
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

The model uses `mmcv-lite`; compiled MMCV CUDA operators are not required.

## Data and checkpoints

| Resource | Download | Extract to |
|---|---|---|
| Ancient architecture dataset | [Baidu Netdisk](https://pan.baidu.com/s/1l_t3OPJbO8HLGn69DmINTQ?pwd=6pjw) (code: `6pjw`) | `./ancient_architecture_data` |
| Initialization weights (50-epoch backbone) | [Baidu Netdisk](https://pan.baidu.com/s/1jd117vwJB9Smicr0sgiU3g?pwd=i3ng) (code: `i3ng`) | `./weights` |
| Full SEMF-Depth model trained on ancient architecture | [Baidu Netdisk](https://pan.baidu.com/s/1rG6G46p8w1QcTdMRDSUyVQ?pwd=tw5p) (code: `tw5p`) | `./tmp/SEMF/refinement/models/weights_69` |

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
and `splits/camera_intrinsics.json`.

## Training

Training was performed on a single NVIDIA RTX 5090 GPU (32 GB VRAM).

Training consists of 50 backbone epochs followed by 20 refinement epochs:

```bash
python train.py \
  --stage auto --data_path ./ancient_architecture_data \
  --split_dir ./splits/ancient \
  --backbone_weights ./weights --log_dir ./tmp --model_name SEMF \
  --height 480 --width 640 \
  --batch_size 8 --accumulation_steps 2 --num_workers 4 --seed 1234 \
  --backbone_learning_rate 5e-5 --decoder_learning_rate 1e-4 \
  --backbone_lr_decay 0.9 \
  --learning_rate 2e-6 --head_learning_rate 2e-4 \
  --target_sparsity 0.08 --gamma 8
```

Refinement starts from the supplied 50-epoch MonoViT backbone checkpoint
(`weights_49`), containing `encoder.pth`, `depth.pth`, `pose_encoder.pth` and
`pose.pth`. Place these files in `./weights` or `./weights/weights_49`.
The command then runs refinement for epochs 50–69. If the backbone is absent,
it first trains epochs 0–49. Use `--stage refine` to run only refinement or
`--stage backbone` to run only the backbone phase. For ImageNet encoder
initialization, add `--encoder_pretrained ./ckpt/mpvit_small.pth`.

Both stages use AdamW (`betas=(0.9, 0.999)`, `eps=1e-8`). The optimizer
parameter groups and their initial learning rates are:

| Phase | Group | Parameters | Initial LR | Weight decay |
|---|---|---|---|---|
| Backbone | 0 | Depth decoder and pose networks | `1e-4` | `0.01` |
| Backbone | 1 | MPViT encoder | `5e-5` | `0.01` |
| Refinement | 0 | `edge_head`, `depth_head`, `pose`: weight tensors | `2e-4` | `0.01` |
| Refinement | 1 | Same modules: biases and normalization parameters | `2e-4` | `0` |
| Refinement | 2 | `encoder`, `depth`, `pose_encoder`: weight tensors | `2e-6` | `0.01` |
| Refinement | 3 | Same modules: biases and normalization parameters | `2e-6` | `0` |

During refinement, the depth encoder is frozen and the SEE feature projection
is fixed. The decoder, SEE fusion head (`depth_head`) and pose networks are
optimized through photometric reprojection and edge-aware smoothness losses.
Biases and normalization parameters are identified by `ndim == 1`.

The backbone uses `ExponentialLR(gamma=0.9)`. Refinement starts a fresh AdamW
optimizer and `CosineAnnealingLR(T_max=65, eta_min=1e-7)`, stepped once per epoch.
The random seed is `1234`, with batch size `8`, gradient accumulation `2`
(effective batch size `16`) and `4` data workers. SMF uses `s=0.08` and `gamma=8`.

With the bundled training lists, this command performs 255 optimizer updates
per epoch: 12,750 for a new 50-epoch backbone run and 5,100 for the 20-epoch
refinement. The supplied backbone checkpoint records 45,800 optimizer steps
from its original training; the reported final model records 5,100 refinement
updates.

Each run writes its command, seed, initialization checkpoint, parameter-group
learning rates, update counts and checkout commit to `run_*.json` in its log
directory. Use `git rev-parse HEAD` to identify the checked-out source version.

All epoch checkpoints are retained. The final model is saved to
`./tmp/SEMF/refinement/models/weights_69`. Append `--resume` to restore a run,
including optimizer, scheduler and random states. View logs with
`tensorboard --logdir ./tmp/SEMF`.

## Evaluation

```bash
python evaluate_metrics.py \
  --weights ./tmp/SEMF/refinement/models/weights_69 \
  --data-root ./ancient_architecture_data --output-dir ./results/SEMF
```

This computes the seven depth metrics and CD using the RGB–GT pairs in
`splits/eigen/test_pairs.txt`. JPG, JPEG and PNG inputs are supported.

Evaluation uses the Garg crop, per-image median scaling and a 0.001–80 m range.
CD uses full-depth-map intrinsics and the sum of bidirectional mean squared
nearest-neighbor distances (m²), sampling up to 20,000 points per cloud.

## Visualization

Export depth predictions and the SEE/SMF response maps from an image or directory:

```bash
python visualization.py \
  --weights ./tmp/SEMF/refinement/models/weights_69 \
  --input ./ancient_architecture_data/test \
  --output-dir ./results/visualization
```

Directories are searched recursively. Pass multiple paths after `--input` to
process several folders. Each image produces separate PNGs in `depth/`,
`E_rgb/`, `W_soft/` and `F_edge/`, preserving the input folder structure and
original image dimensions. `F_edge` displays the normalized channel RMS of the
projected features. Add `--device cpu` for CPU inference.

For paired ground-truth visualizations, replace `--input` with
`--test-files ./splits/eigen/test_pairs.txt --data-root ./ancient_architecture_data --save-gt`.
This also writes `GT_depth/`, median-aligns each prediction using the evaluation
crop, and uses the same depth color limits for each prediction/GT pair.

## Configuration and data checks

Inspect the training phase, initialization files, fixed splits and update counts:

```bash
python train.py --stage refine --backbone_weights ./weights --print_config
```

Inspect RGB–GT pairing and camera intrinsics without running inference:

```bash
python evaluate_metrics.py \
  --weights ./tmp/SEMF/refinement/models/weights_69 --dry-run
```

## Acknowledgements

This project builds on [MonoViT](https://github.com/zxcqlf/MonoViT),
[Monodepth2](https://github.com/nianticlabs/monodepth2),
[MPViT](https://github.com/youngwanLEE/MPViT), and
[HR-Depth](https://github.com/shawLyu/HR-Depth). We thank the authors for sharing
their code.

## License

See [LICENSE](LICENSE) for the MIT license on original contributions.
Third-party code retains its original license:

- [MonoViT](LICENSES/MonoViT.txt) and [HR-Depth](LICENSES/HR-Depth.txt): MIT.
- [Monodepth2](LICENSES/Monodepth2.txt): non-commercial use only.
- [MPViT](LICENSES/MPViT.md): its License and Service Agreement, including
  [GPLv3](LICENSES/GPL-3.0.txt) and commercial licensing terms.
