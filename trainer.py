# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import os
import json
import random
import warnings
import gc

# Suppress PyTorch Inductor warnings about online softmax
warnings.filterwarnings(
    "ignore",
    message=".*Online softmax is disabled on the fly.*",
    category=UserWarning
)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# AMP for mixed precision training
# AMP removed
# from torch.cuda.amp import autocast, GradScaler

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks


# from IPython import embed  # Removed: not available on production servers


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        # Scientific Reproducibility: Custom Seed 1234
        set_seed(1234)

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # Optimization for fixed input sizes (Linux/GPU acceleration)
        if not self.opt.no_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True  # For Ampere+ (A100/3090)
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # [SOTA] Double throughput for fixed resolutions

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # ======================================================================
        # Initialize Models
        # ======================================================================
        # 1. Encoder (Backbone Selection)
        if self.opt.backbone == "mpvit":
            print("  [Model] Using MPViT-Small Backbone")
            self.models["encoder"] = networks.mpvit_small()
            # MPViT-Small outputs 5 scales: Stem(64) + 4 Stages(128, 216, 288, 288)
            # Total channels: [64, 128, 216, 288, 288]
            self.models["encoder"].num_ch_enc = [64, 128, 216, 288, 288]
        elif self.opt.backbone == "resnet":
            print(f"  [Model] Using ResNet{self.opt.num_layers} Backbone (Matching Weights)")
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained")
            self.models["encoder"].num_ch_enc = [64, 64, 128, 256, 512]

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # 2. HR Scale Decoder
        # Dynamic num_ch_enc based on backbone
        self.models["depth"] = networks.HRDepthDecoder(
            num_ch_enc=self.models["encoder"].num_ch_enc,
            ch_enc=self.models["encoder"].num_ch_enc,
            scales=self.opt.scales
        )
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # 3. Multi-Head Architecture (Edge + Normal + Depth Fusion)
        if not self.opt.disable_multihead:
            # F_dec channels: 16 (from Decoder) -> BUT we exposed the 32-ch feature in hr_decoder.py
            # Actually, let's verify hr_decoder.py change.
            # In hr_decoder.py: outputs[("F_dec", 0)] = x where x was input to last conv
            # MPViT small num_ch_enc is [64, 128, 216, 288, 288]
            # HRDecoder num_ch_dec is [16, 32, 64, 128, 256]
            # The code I wrote in hr_decoder puts F_dec as the feature after X_04_Conv_1
            # X_04 comes from num_ch_enc[0]//2 = 32 channels.
            # Then X_04_Conv_0(32->16), X_04_Conv_1(16->16).
            # So F_dec has 16 channels.
            # Wait, my implementation_plan said 32, but standard HRDecoder reduces it.
            # Let's adjust heads to accept 16 channels if that's what comes out,
            # OR relies on the fact I modified hr_decoder to output what I need.
            # Looking at my edit to hr_decoder.py:
            # x = features["X_04"] (32 ch) -> Conv0(32->16) -> Conv1(16->16) -> outputs[F_dec]
            # so F_dec has 16 channels.
            f_dec_channels = 16

            # Edge Head
            self.models["edge_head"] = networks.FFTEdgeHead(
                in_channels=f_dec_channels,
                mid_channels=32
            )
            self.models["edge_head"].to(self.device)
            self.parameters_to_train += list(self.models["edge_head"].parameters())

            # Depth Head (Fusion)
            # F_edge has mid_channels=32
            self.models["depth_head"] = networks.DepthHeadWithFusion(
                fdec_channels=f_dec_channels,
                fedge_channels=32,
                scales=self.opt.scales,
                alpha_rgb=self.opt.alpha_rgb,
                target_sparsity=self.opt.target_sparsity
            )
            self.models["depth_head"].to(self.device)
            self.parameters_to_train += list(self.models["depth_head"].parameters())

        # 4. Pose Network
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            self.models["predictive_mask"] = networks.DepthDecoder(
                num_ch_enc=self.models["encoder"].num_ch_enc,
                ch_enc=self.models["encoder"].num_ch_enc,
                scales=self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # ======================================================================
        # Optimizer & Scheduler & AMP
        # ======================================================================
        # ======================================================================
        # Unified Differential Optimizer & Scheduler (10x Head Speed)
        # ======================================================================
        # Identify Backbone Encoders & Decoders (High Stability, Low LR 1e-6)
        backbone_params = []
        if "encoder" in self.models:
            backbone_params += list(self.models["encoder"].parameters())
        if "depth" in self.models:
            # [FIX Stage II] The HRDecoder is part of the backbone infrastructure.
            # It should be fine-tuned with the same stability (1e-6) as the encoder.
            backbone_params += list(self.models["depth"].parameters())
        if "pose_encoder" in self.models:
            backbone_params += list(self.models["pose_encoder"].parameters())
        backbone_param_ids = set(id(p) for p in backbone_params)

        # Identify Randomly Initialized Heads (Fast Learning, High LR 1e-4)
        head_params = [p for p in self.parameters_to_train if id(p) not in backbone_param_ids]

        # [Architectural Hardening] Weight Decay Isolation
        # Bias and Normalization parameters must NOT be decayed.
        def get_optimizer_params(param_list, lr, weight_decay):
            decay = []
            no_decay = []
            for p in param_list:
                # [BUG FIX] NEVRE SKIP params based on requires_grad here.
                # If we skip them now, they will NEVER be added to the optimizer,
                # even if we set requires_grad=True later in Stage II!
                # if not p.requires_grad: continue

                # len(p.shape) == 1 handles biases and 1D norm weights (BatchNorm/GroupNorm/LayerNorm)
                if len(p.shape) == 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return [
                {'params': decay, 'lr': lr, 'weight_decay': weight_decay},
                {'params': no_decay, 'lr': lr, 'weight_decay': 0.0}
            ]

        self.params = []
        # Group 0-1: Heads (1e-4) — get_optimizer_params returns [decay, no_decay] = 2 groups
        head_groups = get_optimizer_params(head_params, self.opt.head_learning_rate, 1e-2)
        self.params.extend(head_groups)
        self.num_head_groups = len(head_groups)  # Track for warmup LR logic

        # Group 2-3: Backbones (1e-6) - Decoupled to 1/100 of Head LR for absolute stability
        # Head LR is 1e-4, Backbone base is 1e-5.
        # (1e-5 / 10.0) = 1e-6. This protects the pre-trained ViT feature space.
        self.params.extend(get_optimizer_params(backbone_params, self.opt.learning_rate / 10.0, 1e-2))



        self.model_optimizer = optim.AdamW(self.params)

        # [LR Scheduler]
        # CosineAnnealing: T_max = num_epochs - warmup_epochs
        # Ensure it reaches the exact minimum on the last epoch, with absolutely no rebound.
        cosine_T_max = self.opt.num_epochs - self.opt.warmup_epochs

        self.model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer,
            T_max=cosine_T_max,
            eta_min=1e-7
        )

        if self.opt.load_weights_folder is not None:
            self.load_model()

        # [ ACCELERATION] Linux-exclusive torch.compile for core modules.
        # Skip NormalHead/DCNv4 to avoid triton/compiler conflicts with custom kernels.
        # [FIX] Triton is Linux-only. Guard with platform check to prevent TritonMissing on Windows.
        import sys
        if not self.opt.no_cuda and sys.platform.startswith('linux'):
            print("  Enabling torch.compile for Encoder and Depth Decoder...")
            self.models["encoder"] = torch.compile(self.models["encoder"])
            self.models["depth"] = torch.compile(self.models["depth"])
            # [FIX] Do NOT compile depth_head due to PyTorch Dynamo bug with `torch.quantile` 
            # and symbolic sizes inside `_compute_soft_mask`.
            # if not self.opt.disable_multihead:
            #    self.models["depth_head"] = torch.compile(self.models["depth_head"])
        elif not self.opt.no_cuda:
            print("  [INFO] torch.compile skipped (Triton not available on Windows).")

        # RTX 50 Series Acceleration: torch.compile
        # (Silently disabled during ablations to prevent DCNv4 memory explosion)
        try:
            # self.models["encoder"] = torch.compile(self.models["encoder"])
            # self.models["depth"] = torch.compile(self.models["depth"])
            pass
        except Exception as e:
            pass

        try:
            torch.set_float32_matmul_precision("high")
        except:
            pass

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # ======================================================================
        # Data Loading
        # ======================================================================
        if self.opt.no_dataset_scan:
            print("  [Trainer] Skipping dataset scan and loader initialization as requested.")
            train_filenames = []
            val_filenames = []
            train_dataset = None
            val_dataset = None
        else:
            datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                             "kitti_odom": datasets.KITTIOdomDataset,
                             "custom": datasets.CustomDataset}
            self.dataset = datasets_dict[self.opt.dataset]

            if self.opt.dataset == "custom":
                # Check for existing splits
                fpath = os.path.join(os.path.dirname(__file__), "splits", "custom", "{}_files.txt")
                if os.path.exists(fpath.format("train")):
                    train_filenames = readlines(fpath.format("train"))
                    val_filenames = readlines(fpath.format("val"))
                else:
                    print(f"Scanning data path: {self.opt.data_path}")
                    train_filenames = []
                    val_filenames = []

                    if os.path.exists(self.opt.data_path):
                        # Detect if the provided data_path is already inside a specific split folder
                        is_split_folder = any(
                            os.path.isdir(os.path.join(self.opt.data_path, sd)) for sd in ["image_02", "image_03"])

                        if is_split_folder:
                            # If given e.g. "kitti_data/train", the parent "kitti_data" has train/val splits
                            parent_path = os.path.dirname(self.opt.data_path)
                            splits_to_scan = [
                                (os.path.basename(self.opt.data_path), train_filenames),  # Itself, probably 'train'
                            ]
                            # Try to find a sibling 'val' or 'test' folder to serve as validation
                            if os.path.isdir(os.path.join(parent_path, "val")):
                                splits_to_scan.append(("val", val_filenames))
                                print(
                                    f"  [Trainer] Auto-detected sibling validation set at: {os.path.join(parent_path, 'val')}")
                            elif os.path.isdir(os.path.join(parent_path, "test")):
                                splits_to_scan.append(("test", val_filenames))
                                print(
                                    f"  [Trainer] Auto-detected sibling validation set at: {os.path.join(parent_path, 'test')}")

                            scan_base_path = parent_path
                        else:
                            splits_to_scan = [("train", train_filenames), ("val", val_filenames)]
                            scan_base_path = self.opt.data_path

                        for split_dir, file_list in splits_to_scan:
                            split_path = os.path.join(scan_base_path, split_dir)
                            if not os.path.isdir(split_path):
                                continue

                            # User Request: In TJF mono vit fork, only use left images from kitti_data\train\image_02; val uses the same. Do not use image_03.
                            for side_dir, side_label in [("image_02", "l")]:
                                side_path = os.path.join(split_path, side_dir)

                                # Support custom KITTI hierarchy where images are in 'data' subfolder
                                scan_path = side_path
                                if os.path.isdir(os.path.join(side_path, "data")):
                                    scan_path = os.path.join(side_path, "data")
                                elif os.path.isdir(os.path.join(side_path, "data_jpg")):
                                    scan_path = os.path.join(side_path, "data_jpg")
                                elif os.path.isdir(os.path.join(side_path, "data_png")):
                                    scan_path = os.path.join(side_path, "data_png")

                                if os.path.isdir(scan_path):
                                    images = sorted([f for f in os.listdir(scan_path) if
                                                     f.lower().endswith('.jpg') or f.lower().endswith('.png')])
                                    for img in images:
                                        try:
                                            idx = int(os.path.splitext(img)[0])
                                            # format: folder frame_index side
                                            # Since custom dataset splits might interpret 'folder' differently,
                                            # we use relative path from data_path
                                            folder_rel = os.path.join(split_dir, side_dir) if split_dir else side_dir
                                            file_list.append(f"{folder_rel} {idx} {side_label}")
                                        except:
                                            continue

                    if len(train_filenames) > 0 or len(val_filenames) > 0:
                        print(f"Auto-detected images. Train: {len(train_filenames)}, Val: {len(val_filenames)}")
                    else:
                        print("WARNING: No images found during auto-scan.")

            else:
                fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
                train_filenames = readlines(fpath.format("train"))
                val_filenames = readlines(fpath.format("val"))

            img_ext = '.png' if self.opt.png else '.jpg'
            num_train_samples = len(train_filenames)
            self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

            # If trainer auto-detected split folders (e.g. data_path="xxx/train", scan_base="xxx"),
            # we should pass the corrected scan_base_path so custom datasets can attach image paths securely.
            dataset_root_path = scan_base_path if (
                        self.opt.dataset == "custom" and not self.opt.no_dataset_scan) else self.opt.data_path

            train_dataset = self.dataset(
                dataset_root_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
                # [SOTA FIX] Keep worker processes alive and prefetch data to fully saturate your 5090!
                persistent_workers=True if self.opt.num_workers > 0 else False,
                prefetch_factor=2 if self.opt.num_workers > 0 else None)

            val_dataset = self.dataset(
                dataset_root_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
                # [SOTA FIX] Validation set also needs persistent workers to prevent process teardown/rebuild overhead from frequent evaluation.
                persistent_workers=True if self.opt.num_workers > 0 else False,
                prefetch_factor=2 if self.opt.num_workers > 0 else None)
            self.val_iter = iter(self.val_loader)

            # [QUALITATIVE ANCHOR] Create a fixed batch for consistent qualitative comparison
            # We take the first batch with shuffle=False to ensure "The 4 Pillars" (fixed images)
            # are logged every time, allowing for perfect side-by-side evolutionary analysis.
            fixed_dataset = self.dataset(
                dataset_root_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
            self.fixed_loader = DataLoader(
                fixed_dataset, self.opt.batch_size, shuffle=False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            self.fixed_val_batch = next(iter(self.fixed_loader))
            print("  [Visualization] Fixed-index qualitative anchor batch initialized.")

        self.writers = {}
        for mode in ["train", "val", "val_fixed"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        if train_dataset is not None and val_dataset is not None:
            print("There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode with Two-Stage Asymmetric Logic.
        Stage I: Freeze Backbone (Encoder + Depth Decoder) to align randomly initialized heads.
        Stage II: Global Fine-tuning with Decoupled Learning Rates.
        """
        # Determine if we are in Phase 1 (Backbone Frozen)
        # Absolute logic: ensure Stage I only occurs during initial epochs.
        is_phase1 = (self.epoch < self.opt.phase1_epochs)

        for name, m in self.models.items():
            if is_phase1 and (name == "encoder" or name == "depth"):
                # STAGE I: Physical Freezing
                m.eval()  # Keep BN statistics static
                for param in m.parameters():
                    param.requires_grad = False
            elif name == "encoder":
                # [SOTA SECRET: Permanent Encoder Lock]
                # Pre-trained ViT encoder is already "invincible".
                # Decoupling it completely from gradient backprop saves Stage II from collapse.
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
            else:
                # STAGE II or New Heads: Full training mode
                m.train()
                for param in m.parameters():
                    param.requires_grad = True

        if self.step % self.opt.log_frequency == 0:
            if is_phase1:
                print(f"  [TRAIN REGIME] Epoch {self.epoch}: Backbone (Encoder/Decoder) is LOCKED (Stage I).")
            else:
                # Stage II: Unfrozen Decoder
                print(f"  [TRAIN REGIME] Epoch {self.epoch}: Backbone Decoder is UNFROZEN (Stage II). Encoder stays LOCKED.")

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.step = 0
        self.start_time = time.time()
        self.best_avg_loss = float('inf')

        # Resume / Load Logic:
        start_epoch = 0
        load_folder = self.opt.load_weights_folder

        # [NEW] Explicit Resume Logic
        if self.opt.resume:
            models_dir = os.path.join(self.log_path, "models")
            if os.path.exists(models_dir):
                weight_folders = []
                for item in os.listdir(models_dir):
                    if item.startswith("weights_") and not item.endswith("best"):
                        try:
                            epoch_num = int(item.split("_")[1])
                            weight_folders.append((epoch_num, os.path.join(models_dir, item)))
                        except:
                            continue
                if weight_folders:
                    weight_folders.sort(key=lambda x: x[0], reverse=True)
                    latest_epoch, load_folder = weight_folders[0]
                    print(f"  [RESUME] Auto-detected latest checkpoint: weights_{latest_epoch}")
                    self.opt.load_weights_folder = load_folder

                    # Ensure model is LOADED (including optimizer state)
                    # Note: load_model is also called in __init__, but if load_folder changed here,
                    # we must re-load to catch the correct epoch weights.
                    self.load_model()

                    start_epoch = latest_epoch + 1
                    print(f"  [RESUME] Resuming from Epoch {start_epoch}")
                else:
                    print(f"  [RESUME] No checkpoints found in {models_dir}. Starting from scratch.")

        elif load_folder is not None and os.path.isdir(load_folder):
            folder_name = os.path.basename(load_folder)
            if folder_name.startswith("weights_") and "best" not in folder_name:
                try:
                    loaded_epoch = int(folder_name.split("_")[1])
                    # If loading a specific checkpoint from THIS experiment (num_epochs matches)
                    if loaded_epoch < self.opt.num_epochs:
                        start_epoch = loaded_epoch + 1
                        print(f"  [CONTINUE] Specific checkpoint detected. Continuing from Epoch {start_epoch}")
                    else:
                        print(f"  [RESET] Loaded epoch {loaded_epoch} >= {self.opt.num_epochs}. Starting at Epoch 0.")
                except:
                    pass

        self.epoch = start_epoch
        for self.epoch in range(start_epoch, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

            # Best Model Tracking (safety net based on training loss)
            # Note: Do not overwrite per-epoch saved weights; save an additional copy to weights_best/
            epoch_avg_loss = getattr(self, '_epoch_avg_loss', float('inf'))
            if epoch_avg_loss < self.best_avg_loss:
                self.best_avg_loss = epoch_avg_loss
                best_folder = os.path.join(self.log_path, "models", "weights_best")
                if not os.path.exists(best_folder):
                    os.makedirs(best_folder)
                for model_name, model in self.models.items():
                    save_path = os.path.join(best_folder, "{}.pth".format(model_name))
                    to_save = model.state_dict()
                    if model_name == 'encoder':
                        to_save['height'] = self.opt.height
                        to_save['width'] = self.opt.width
                        to_save['use_stereo'] = self.opt.use_stereo
                    torch.save(to_save, save_path)
                print(f"  [BEST] New best avg loss: {epoch_avg_loss:.5f} (Epoch {self.epoch}). Saved to weights_best/")

            # Clear GPU cache after each epoch to prevent system freezes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # [PHASE TRANSITION HOOK]
        # When moving from Phase 1 (Frozen) to Phase 2 (Joint Fine-Tuning),
        # we can inject a specific backbone if requested, but for Stage 1
        # mdp_stage1 (num_epochs=10, phase1=10), it will just end at epoch 9.
        if self.epoch == self.opt.phase1_epochs and self.opt.num_epochs > self.opt.phase1_epochs:
            print(f"\n" + "!" * 60)
            print(f"!!! [PHASE TRANSITION] Entering Phase 2 Joint Training at Epoch {self.epoch}")
            print("!" * 60 + "\n")
            # Backbone will be unfrozen in set_train() below.

        print(f"Epoch {self.epoch} | Training")
        self.set_train()

        epoch_loss = 0.0

        # Warmup Logic: Linearly Ramping LRs during early epochs
        if self.epoch < self.opt.warmup_epochs:
            # We use a 1-based relative mapping for the factor [1..warmup_epochs+1]
            warmup_factor = (self.epoch + 1) / (self.opt.warmup_epochs + 1)
            # Head groups (0 ~ num_head_groups-1): head LR
            for i in range(self.num_head_groups):
                self.model_optimizer.param_groups[i]['lr'] = self.opt.head_learning_rate * warmup_factor
            # Remaining groups: Backbones / Offsets
            for i in range(self.num_head_groups, len(self.model_optimizer.param_groups)):
                self.model_optimizer.param_groups[i]['lr'] = self.opt.learning_rate * warmup_factor

        elif self.epoch == self.opt.warmup_epochs:
            # Warmup just ended: restore exact target LRs
            for i in range(self.num_head_groups):
                self.model_optimizer.param_groups[i]['lr'] = self.opt.head_learning_rate
            for i in range(self.num_head_groups, len(self.model_optimizer.param_groups)):
                self.model_optimizer.param_groups[i]['lr'] = self.opt.learning_rate
            print(f"  [LR] Warmup ended at epoch {self.epoch}. LRs reset to targets.")

        # Print LRs for visibility
        lr_info = " | ".join([f"G{i}={g['lr']:.2e}" for i, g in enumerate(self.model_optimizer.param_groups)])
        print(f"  [LR] Epoch {self.epoch}: {lr_info}")

        # =====================================================================
        # [SOTA FIX] Gradient Accumulation Micro-Op Core (The Gradient Accumulation Engine)
        # =====================================================================
        self.model_optimizer.zero_grad()
        accumulation_steps = getattr(self.opt, "accumulation_steps", 1)

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            if torch.isnan(losses["loss"]):
                print(f" [CRITICAL] NaN loss detected at Epoch {self.epoch} Step {self.step}. Skipping batch...")
                self.model_optimizer.zero_grad()  # Discard the current dirty gradients
                continue

            # Scale loss for gradient accumulation (strictly physically proportional scaling)
            loss = losses["loss"] / accumulation_steps
            loss.backward()

            # [SOTA FIX] Accumulation Step Logic
            # Only allow clipping and stepping when accumulation reaches the specified steps, or this is the last batch of the epoch.
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # Gradient Clipping (ultimate stability optimization: 1.0 -> 0.8)
                # Must be placed after accumulation is complete! Otherwise each incomplete micro-batch will be erroneously clipped!
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, 0.8)

                self.model_optimizer.step()
                self.model_optimizer.zero_grad()

            # Statistics still use the true (unscaled) physical loss value
            epoch_loss += losses["loss"].item()

            duration = time.time() - before_op_time

            # Log frequency
            if self.step % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            # Release intermediate variables to prevent memory leaks
            del outputs, losses

            self.step += 1
        # =====================================================================

        # Log epoch result
        avg_loss = epoch_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        self._epoch_avg_loss = avg_loss  # Used by best model tracking
        log_file = os.path.join(self.log_path, "training_log.txt")
        with open(log_file, "a") as f:
            f.write(f"Epoch: {self.epoch} | Average Loss: {avg_loss:.5f}\n")
        print(f"End of Epoch {self.epoch}, Average Loss: {avg_loss:.5f}")

        # Step LR Scheduler (step every epoch after warmup ends, until the last epoch reaches the exact minimum)
        if self.opt.warmup_epochs <= self.epoch < self.opt.num_epochs:
            self.model_lr_scheduler.step()

        # Memory Hygiene: Clear cache after each epoch to prevent VRAM fragmentation during long 25-epoch runs
        if not self.opt.no_cuda:
            torch.cuda.empty_cache()

    def process_batch(self, inputs):
        """Pass a batch through the network and generate images and losses
        """
        # [PHASE ISOLATION] Reset transient state to prevent cross-batch mask contamination.
        self.master_dampening_mask = None
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # 1. Pipeline: Encoder -> Features
        features = self.models["encoder"](inputs["color_aug", 0, 0])

        # 2. Pipeline: Decoder -> Disparity (+ F_dec)
        # Note: Original HRDepthDecoder outputs only disparity.
        # We modified it to also output F_dec at ("F_dec", 0).
        outputs = self.models["depth"](features)

        # 3. Multi-Head Pipeline
        if not self.opt.disable_multihead:
            F_dec = outputs[("F_dec", 0)]  # (B, 16, H, W)

            # [FIX v13] ALWAYS detach F_dec for auxiliary heads.
            # Old logic switched detach→attached at Epoch 5, causing auxiliary loss gradients to suddenly flood
            # the backbone encoder → catastrophic interference → depth map instant collapse.
            # Principle (CVPR 2024 'Mind The Edge'): Auxiliary tasks use stop-gradient;
            # the backbone is driven only by photometric reconstruction loss (the primary task).
            F_dec_isolated = F_dec.detach()

            # A. Edge Head (High-Pass Frequency Extractor)
            # Input: F_dec_isolated (Detached in P1 / Attached in P2)
            E_dense, E_high, F_edge = self.models["edge_head"](
                F_dec_isolated, rgb_input=inputs[("color", 0, 0)]
            )
            outputs["E_dense"] = E_dense
            outputs["E_high"] = E_high
            outputs["F_edge"] = F_edge

            # [FIX v13] Always detach — prevent auxiliary head gradients from backpropagating through depth_head into the backbone
            F_edge_in = F_edge.detach()

            # [SOTA FIX v25: Passing E_dense as Semantic Gate]
            # Pass E_dense to depth_head for texture filtering during mask generation.
            fusion_outputs = self.models["depth_head"](
                F_dec,
                F_edge_in,
                E_high,
                E_dense.detach(),  # [Instruction 3: Semantic Isolation]
            )

            outputs["W_soft_mask"] = fusion_outputs.get("W_soft_mask")
            outputs["q_threshold"] = fusion_outputs.get("q_threshold")
            outputs["E_high_raw"] = fusion_outputs.get("E_high_raw")

            # [SOTA FIX v30: THE SOUL (Early Gated Injection)]
            # Must happen BEFORE generate_images_pred so that warping uses fused depth!
            W_edge = outputs["W_soft_mask"]
            if W_edge is not None:
                # 1. Generate Master Mask at Scale 0
                # Master Mask: 1=Smooth(Backbone), 0=Edge(DCN)
                # Note: We use 1.0 as the multiplier here to get the full spatial selection power
                # [ABLATION] Bypass Routing
                if getattr(self.opt, "disable_routing", False):
                    # Force master dampening to 0 to uniformly apply Naive Concat across whole image
                    self.master_dampening_mask = torch.zeros_like(W_edge)
                else:
                    self.master_dampening_mask = torch.clamp(1.0 - W_edge, min=0.0, max=1.0)

                    # [SOTA FIX v34: Sky Defensive Gate - Injection Protection]
                    # Stop noisy DCN head from leaking into sky/darkness. Force Backbone dominance.
                    rgb_mean = inputs[("color", 0, 0)].mean(dim=1, keepdim=True)
                    sky_mask = ((rgb_mean > 0.95) | (rgb_mean < 0.05)).float()
                    self.master_dampening_mask = torch.max(self.master_dampening_mask, sky_mask)

                # 2. Perform Injection for ALL scales
                for s in self.opt.scales:
                    disp_backbone = outputs[("disp", s)]
                    disp_fusion = fusion_outputs.get(("disp", s))
                    if disp_fusion is not None:
                        # Store fusion depth for visualization
                        outputs[("disp_fusion", s)] = disp_fusion

                        if s > 0:
                            mask_s = F.interpolate(
                                self.master_dampening_mask,
                                size=disp_backbone.shape[-2:],
                                mode='bilinear', align_corners=False
                            )
                        else:
                            mask_s = self.master_dampening_mask

                        # Selection: Backbone where smooth, DCN where edge
                        outputs[("disp", s)] = disp_backbone * mask_s + disp_fusion * (1.0 - mask_s)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch, separating qualitative and quantitative metrics.
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            # 1. Quantitative Metrics: Run on a random validation Batch to ensure generalization.
            outputs_metric, losses_metric = self.process_batch(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs_metric, losses_metric)

            # Use 'val' mode for standard metric logging
            self.log("val", inputs, outputs_metric, losses_metric)

            # 2. Qualitative Tracking: Run on the fixed anchor batch for "Evolution of the Frieze".
            # Note: We calculate losses only for the qualitative visualization (e.g. masks),
            # but we pass an empty dict for scalar logging to avoid metric pollution.
            outputs_fixed, _ = self.process_batch(self.fixed_val_batch)

            self.log("val_fixed", self.fixed_val_batch, outputs_fixed, losses={})

            del inputs, outputs_metric, losses_metric, outputs_fixed

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute all losses: Reprojection, Smoothness (Repulsive), Frequency
        """
        losses = {}
        total_loss = 0

        # Defensive initialization: ensure master_dampening_mask exists (prevent AttributeError from out-of-order scales)
        if not hasattr(self, 'master_dampening_mask'):
            self.master_dampening_mask = None

        # Phased Loss Activation
        # Epoch 0-4: Photo only. Epoch 5+: Add L_freq and Repulsive Smooth
        use_multihead_loss = (not self.opt.disable_multihead) and \
                             (self.epoch >= self.opt.loss_start_epoch)

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking and combined.shape[1] > 1:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            # Smoothness Loss with Repulsive Field
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            # Standard Smoothness Loss
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        # Logging handled in set_train for cleaner stdout

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_gt = inputs["depth_gt"]
        gt_h, gt_w = depth_gt.shape[-2:]
        max_d = self.opt.max_depth  # 20m for ancient architecture

        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [gt_h, gt_w], mode="bilinear", align_corners=False), 1e-3, max_d)
        depth_pred = depth_pred.detach()

        mask = depth_gt > 0

        # Garg/Eigen crop (adaptive to GT resolution)
        crop_mask = torch.zeros_like(mask)
        top = int(0.40810811 * gt_h)
        bot = int(0.99189189 * gt_h)
        lft = int(0.03594771 * gt_w)
        rgt = int(0.96405229 * gt_w)
        crop_mask[:, :, top:bot, lft:rgt] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=max_d)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    colormap(outputs[("disp", s)][j]).float(), self.step)

                # Log fusion depth if available
                if outputs.get(("disp_fusion", s)) is not None:
                    writer.add_image(
                        "disp_fusion_{}/{}".format(s, j),
                        colormap(outputs[("disp_fusion", s)][j]).float(), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

            # Log Multi-Head Outputs
            # Industrial mode (no ablation_stage) shows EVERYTHING.
            # Ablation mode shows only what is active.
            is_industrial = self.opt.ablation_stage is None

            if not self.opt.disable_multihead:
                if outputs.get("E_dense") is not None:
                    writer.add_image("edge_dense/{}".format(j), outputs["E_dense"][j], self.step)

                if outputs.get("E_high") is not None:
                    writer.add_image("edge_high/{}".format(j), outputs["E_high"][j], self.step)

                if outputs.get("F_edge") is not None:
                    # F_edge has shape (B, mid_channels, H, W), visualize first channel
                    f_edge_vis = outputs["F_edge"][j, 0:1]  # Take first channel
                    f_edge_vis = (f_edge_vis - f_edge_vis.min()) / (f_edge_vis.max() - f_edge_vis.min() + 1e-8)
                    writer.add_image("feature_edge/{}".format(j), f_edge_vis, self.step)

                # [NEW] Direct Raw Mask Visualization
                if outputs.get("W_soft_mask") is not None:
                    # Directly display the raw W_soft_mask (0=smooth, 1=edge segmentation)

                    writer.add_image("w_soft_mask_raw/{}".format(j), outputs["W_soft_mask"][j].float(), self.step)

               # Normal/Curvature visualizers removed

                if outputs.get("W_soft_mask") is not None:
                    # TensorBoard visual probe: directly use the Master Mask already computed in compute_losses.
                    # No redundant recalculation of the 4-Step Pipeline; ensures visualization and loss computation are from the exact same source.

                    if hasattr(self, 'master_dampening_mask') and self.master_dampening_mask is not None:
                            with torch.no_grad():
                                dampening_vis = self.master_dampening_mask[j].float().clamp(0, 1)
                                # W_clean = severed spring region = (1 - dampening) / lambda
                                # More intuitive: directly display (1 - dampening)
                                clean_edge_vis = (1.0 - dampening_vis).clamp(0, 1)

                            # 1. Cleaned edge skeleton (pure black background + white eave lines = correct)
                            writer.add_image("clean_edge_mask/{}".format(j), clean_edge_vis, self.step)
                            # 2. Dampening mask (white=protected region, black=spring-severed region)
                            writer.add_image("dampening_mask/{}".format(j), dampening_vis, self.step)

                            # 3. Core alignment verification: W_clean overlay on original image
                            #    The white skeleton lines must perfectly overlay the eaves of the building.
                            #    If they don't, there is a spatial shift!
                            color_j = inputs[("color", 0, 0)][j]  # (3, H, W) original RGB
                            # Resize clean_edge to match color if needed
                            if clean_edge_vis.shape[-2:] != color_j.shape[-2:]:
                                clean_edge_resized = F.interpolate(
                                    clean_edge_vis.unsqueeze(0), size=color_j.shape[-2:],
                                    mode='bilinear', align_corners=False).squeeze(0)
                            else:
                                clean_edge_resized = clean_edge_vis
                            # Overlay: original image + green skeleton lines (Green channel boost)
                            # Edge regions (W_clean > 0): overlaid in bright green for instant alignment verification
                            overlay = color_j.clone()
                            edge_alpha = clean_edge_resized.expand_as(overlay)  # (3, H, W)
                            # Green channel: enhance edge lines
                            overlay[1] = (overlay[1] * (1.0 - edge_alpha[0] * 0.7) + edge_alpha[0] * 0.9).clamp(0, 1)
                            # Red/Blue: slightly darken in edge regions to make green stand out
                            overlay[0] = (overlay[0] * (1.0 - edge_alpha[0] * 0.5)).clamp(0, 1)
                            overlay[2] = (overlay[2] * (1.0 - edge_alpha[0] * 0.5)).clamp(0, 1)
                            writer.add_image("edge_on_image/{}".format(j), overlay, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk (and optimizer/scheduler/scaler)
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))

            # [SOTA LOCK] Unwrap torch.compile wrapper before saving to disk.
            # This strips the "._orig_mod" prefix from state_dict keys, ensuring the weights
            # remain compatible with non-Linux/Windows/CPU environments.
            real_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            to_save = real_model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        # Save Optimizer, Scheduler
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save({
            'optimizer': self.model_optimizer.state_dict(),
            'scheduler': self.model_lr_scheduler.state_dict(),
            'epoch': self.epoch,
        }, save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))

            if n not in self.models:
                # If model n is disabled (e.g. normal_head in ablation), skip it
                continue

            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

            if not os.path.exists(path):
                # Skip loading auxiliary heads if they don't exist in checkpoint (e.g. resuming baseline)
                print(f"Skipping {n} (not found in checkpoint)")
                continue

            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, weights_only=False)

            # [SOTA MAPPING] Prefix Alignment for torch.compile
            # If model is already compiled but checkpoint is clean, or vice versa, align them.
            if any(k.startswith("_orig_mod.") for k in model_dict.keys()) and \
                    not any(k.startswith("_orig_mod.") for k in pretrained_dict.keys()):
                pretrained_dict = {"_orig_mod." + k: v for k, v in pretrained_dict.items()}
            elif not any(k.startswith("_orig_mod.") for k in model_dict.keys()) and \
                    any(k.startswith("_orig_mod.") for k in pretrained_dict.keys()):
                pretrained_dict = {k.replace("_orig_mod.", ""): v for k, v in pretrained_dict.items()}

            # [Physics-Based Initialization]
            # Differentiating between "Trained Features" and "New Physics Modules"
            checkpoint_keys = set(pretrained_dict.keys())
            model_keys = set(model_dict.keys())

            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys

            if len(missing_keys) > 0:
                print(f"  [PARTIAL LOAD] {n}: Missing keys (random init): {sorted(list(missing_keys))}")
            if len(unexpected_keys) > 0:
                print(f"  [PARTIAL LOAD] {n}: Unexpected keys (ignored): {sorted(list(unexpected_keys))}")

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            # [ADAM MOMENTUM ISOLATION]
            # If we are loading from the 50-epoch baseline (weights_49), we MUST ignore the old optimizer state.
            # The architecture (Multi-Head/DCN) and loss landscape (3D Repulsive Fields) are too different.
            # Loading old momentum would cause the "Muddy Start" effect, slowing down or poisoning convergence.
            # [DE-MINING] Decoupled from string-matching via formal --reset_optimizer flag.
            checkpoint = {}  # Initialize to empty dict to prevent UnboundLocalError
            if self.opt.reset_optimizer:
                print("  [Cold Start] --reset_optimizer is SET. Skipping Adam state loading for new phase alignment.")
            else:
                print("Loading Adam weights")
                try:
                    # Load to CPU to prevent CUDA OOM spikes (crucial for ablation loops)
                    checkpoint = torch.load(optimizer_load_path, map_location='cpu', weights_only=False)
                    if 'optimizer' in checkpoint:
                        self.model_optimizer.load_state_dict(checkpoint['optimizer'])
                    else:
                        self.model_optimizer.load_state_dict(checkpoint)

                    print("[INFO] Optimizer state loaded successfully.")
                except ValueError as e:
                    print(f"[WARNING] Optimizer load failed (mismatched parameter groups): {e}")
                    print("[INFO] Optimizer initialized from scratch (Normal for Ablation/Different Heads).")
                except Exception as e:
                    print(f"[WARNING] Optimizer load failed with error: {e}")

            if 'scheduler' in checkpoint and hasattr(self, 'model_lr_scheduler'):
                print("Loading Scheduler state")
                try:
                    self.model_lr_scheduler.load_state_dict(checkpoint['scheduler'])
                except Exception as e:
                    print(f"[WARNING] Scheduler load failed: {e}")

            if 'scaler' in checkpoint and hasattr(self, 'scaler'):
                print("Loading AMP Scaler state")
                self.scaler.load_state_dict(checkpoint['scaler'])
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def cleanup(self):
        """Explicitly delete models and clear CUDA memory to prevent OOM in ablation loops
        """
        print(f"  [CLEANUP] Clearing memory for stage {self.opt.model_name}...")

        # 1. Close TensorBoard writers
        if hasattr(self, 'writers'):
            for writer in self.writers.values():
                writer.close()
            del self.writers

        # 2. Delete Dataloaders/Iters
        if hasattr(self, 'val_iter'): del self.val_iter
        if hasattr(self, 'train_loader'): del self.train_loader
        if hasattr(self, 'val_loader'): del self.val_loader

        # 3. Delete Models and Optimizers
        if hasattr(self, 'models'):
            for m in self.models.values():
                m.cpu()  # move to CPU before deletion to ensure GPU release
            del self.models

        if hasattr(self, 'model_optimizer'): del self.model_optimizer
        if hasattr(self, 'model_lr_scheduler'): del self.model_lr_scheduler

        # 4. Final Garbage Collection and CUDA flush
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("  [CLEANUP] Memory cleaned.")

