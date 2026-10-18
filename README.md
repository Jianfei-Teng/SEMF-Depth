# SEMF-Depth

Official PyTorch implementation of **Texture-Structure Decoupled Monocular Depth
Estimation in Complex 3D Reconstruction Scenes**.

Jianfei Teng, Xiaoling Yao, Lihua Hu, Yang Ma, Jifu Zhang

SEMF-Depth is a self-supervised monocular depth estimation framework. Its structural
edge enhancement (SEE) strategy combines image-frequency responses with projected
decoder features, and its structure-edge-aware masked fusion (SMF) mechanism
uses these signals to control local corrections to the backbone prediction.

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
and `splits/相机内参.json`.

## Training

Training consists of 50 backbone epochs followed by 20 refinement epochs:

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
epoch. During refinement, the depth encoder is frozen. SEE applies fixed
projections to detached decoder features; its edge features and dense response
are detached before SMF. Photometric reprojection and edge-aware smoothness
losses update the depth decoder, SMF and pose networks.

The training set contains 4,074 left-view images. Excluding the first and last
frames leaves 4,072 targets with both temporal neighbors. At batch size 8 with
accumulation 2, a new run uses 255 optimizer updates per epoch: 12,750 for the
backbone and 5,100 for refinement. The supplied backbone's optimizer state records 45,800
updates; the retained Full checkpoint records 5,100 refinement updates.

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

This computes the seven depth metrics and CD on the 489 RGB–GT pairs in the
bundled manifest, `splits/eigen/test_pairs.txt`. JPG, JPEG and PNG inputs are supported.

Evaluation uses the Garg crop, per-image median scaling and a 0.001–80 m range.
CD uses full-depth-map intrinsics and the sum of bidirectional mean squared
nearest-neighbor distances (m²), sampling up to 20,000 points per cloud.

## Configuration and data checks

Inspect the training phase, initialization files, fixed splits and update counts:

```bash
python train.py --stage refine --backbone_weights ./weights --print_config
```

Check the 489 RGB–GT pairs, valid depth support and camera intrinsics:

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
