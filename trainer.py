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
import warnings
import gc
import random
from pathlib import Path

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

from common.utils import *
from common.kitti_utils import *
from common.layers import *

import datasets
import networks
from common.checkpoint_compat import optimizer_state_without_unused_gate
from common.training_plan import optimizer_groups


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
        set_seed(self.opt.seed)

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # Optimization for fixed input sizes (Linux/GPU acceleration)
        if not self.opt.no_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True  # For Ampere+ (A100/3090)
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

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
            self.models["encoder"] = networks.mpvit_small(pretrained=(
                self.opt.encoder_pretrained if self.opt.phase == "backbone" and not self.opt.load_weights_folder else None))
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

        # 3. SEE feature enhancement and SMF masked prediction fusion.
        if not self.opt.disable_multihead:
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
                target_sparsity=self.opt.target_sparsity, gamma=self.opt.gamma
            )
            self.models["depth_head"].to(self.device)
            self.parameters_to_train += list(self.models["depth_head"].parameters())

        # 4. Pose Network
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained" and not self.opt.load_weights_folder,
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

        self.params = optimizer_groups(self.models, self.opt)
        self.num_head_groups = 2 if self.opt.phase == "refinement" else 0
        self.model_optimizer = optim.AdamW(self.params, betas=(0.9, 0.999), eps=1e-8)
        if self.opt.phase == "backbone":
            self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.model_optimizer, gamma=self.opt.backbone_lr_decay)
        else:
            self.model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.model_optimizer, T_max=65, eta_min=1e-7)
        self.optimizer_updates = 0
        self.loaded_training_state = None
        if self.opt.load_weights_folder is not None:
            self.load_model()

        # Linux-exclusive compilation of the encoder and decoder.
        # [FIX] Triton is Linux-only. Guard with platform check to prevent TritonMissing on Windows.
        import sys
        if not self.opt.no_cuda and not self.opt.no_compile and sys.platform.startswith('linux'):
            print("  Enabling torch.compile for Encoder and Depth Decoder...")
            self.models["encoder"] = torch.compile(self.models["encoder"])
            self.models["depth"] = torch.compile(self.models["depth"])
            # Leave quantile-based routing outside torch.compile.
        elif not self.opt.no_cuda:
            print("  [INFO] torch.compile skipped (Triton not available on Windows).")

        try:
            torch.set_float32_matmul_precision("high")
        except:
            pass

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # Fixed left-camera lists are validated before model construction.
        self.dataset = datasets.CustomDataset
        train_filenames = self.opt._train_filenames
        val_filenames = self.opt._val_filenames
        img_ext = '.png' if self.opt.png else '.jpg'
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        loader_options = dict(num_workers=self.opt.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=self.opt.num_workers > 0,
                              prefetch_factor=2 if self.opt.num_workers > 0 else None)
        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, True, **loader_options)
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, True, **loader_options)
        self.val_iter = iter(self.val_loader)
        self.fixed_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False, num_workers=0, drop_last=True)
        self.fixed_val_batch = next(iter(self.fixed_loader))
        self.num_total_steps = len(self.train_loader) * (self.opt.num_epochs - self.opt.start_epoch)

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
        """Train the backbone first; freeze the depth encoder during refinement."""
        for name, model in self.models.items():
            frozen = self.opt.phase == "refinement" and name == "encoder"
            model.eval() if frozen else model.train()
            for param in model.parameters():
                param.requires_grad = not frozen
        # SEE outputs are detached at the SMF input. Its projection parameters
        # receive no gradients from the depth loss.

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the selected phase, preserving optimizer state on explicit resume."""
        self.step = 0
        self.start_time = time.time()
        self.invocation_start_epoch = self.opt.start_epoch
        if self.loaded_training_state is not None:
            rng = self.loaded_training_state["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        self.write_run_record()
        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self.run_epoch()
            self.save_model()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def write_run_record(self):
        record = {"command": self.opt.command, "code": self.opt.code_version,
                  "phase": self.opt.phase, "seed": self.opt.seed,
                  "initialization": self.opt.load_weights_folder,
                  "epoch_range": [self.opt.start_epoch, self.opt.num_epochs - 1],
                  "data": self.opt._split_stats,
                  "batch_size": self.opt.batch_size,
                  "accumulation_steps": self.opt.accumulation_steps,
                  "microbatches_per_epoch": len(self.train_loader),
                  "updates_per_epoch": self.opt.updates_per_epoch,
                  "planned_updates_this_invocation": self.opt.planned_phase_updates,
                  "optimizer_updates_at_start": self.optimizer_updates,
                  "optimizer": type(self.model_optimizer).__name__,
                  "groups_at_start": [{k: v for k, v in g.items() if k != "params"}
                                      for g in self.model_optimizer.param_groups],
                  "scheduler_at_start": self.model_lr_scheduler.state_dict()}
        record_path = Path(self.log_path) / ("run_" + str(time.time_ns()) + ".json")
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def run_epoch(self):
        """One epoch with the released reprojection/smoothness objective."""
        print(f"Epoch {self.epoch} | {self.opt.phase} | Training")
        self.set_train()
        epoch_start = time.time()
        epoch_loss = 0.0
        updates_before = self.optimizer_updates
        lrs_before = [g["lr"] for g in self.model_optimizer.param_groups]
        print("  [LR] " + " | ".join(f"G{i}={lr:.8g}" for i, lr in enumerate(lrs_before)))
        self.model_optimizer.zero_grad()
        accumulation_steps = self.opt.accumulation_steps
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            if not torch.isfinite(losses["loss"]).all():
                raise FloatingPointError(f"Nonfinite loss at epoch {self.epoch}, batch {batch_idx}")
            (losses["loss"] / accumulation_steps).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(self.train_loader):
                norm = torch.nn.utils.clip_grad_norm_(self.parameters_to_train, 0.8)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"Nonfinite gradient at epoch {self.epoch}, batch {batch_idx}")
                self.model_optimizer.step()
                self.optimizer_updates += 1
                self.model_optimizer.zero_grad()
            epoch_loss += losses["loss"].item()
            if self.step % self.opt.log_frequency == 0:
                self.log_time(batch_idx, time.time() - before_op_time, losses["loss"].detach())
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                self.val()
            del outputs, losses
            self.step += 1
        actual_updates = self.optimizer_updates - updates_before
        if actual_updates != self.opt.updates_per_epoch:
            raise RuntimeError(f"Update count mismatch: {actual_updates} vs {self.opt.updates_per_epoch}")
        self._epoch_avg_loss = epoch_loss / len(self.train_loader)
        self.model_lr_scheduler.step()
        record = {"epoch": self.epoch, "phase": self.opt.phase, "loss": self._epoch_avg_loss,
                  "duration_seconds": time.time() - epoch_start,
                  "optimizer_updates": actual_updates, "phase_optimizer_updates": self.optimizer_updates,
                  "lrs_used": lrs_before,
                  "lrs_next": [g["lr"] for g in self.model_optimizer.param_groups]}
        with open(os.path.join(self.log_path, "training_log.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"End of Epoch {self.epoch} | loss={self._epoch_avg_loss:.5f} | "
              f"updates={actual_updates} | phase updates={self.optimizer_updates}")

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

            # Released gradient graph: isolate decoder features before SEE.
            F_dec_isolated = F_dec.detach()

            # SEE projects features; SMF uses the RGB-frequency response.
            _, E_high, F_edge = self.models["edge_head"](
                F_dec_isolated, rgb_input=inputs[("color", 0, 0)]
            )
            outputs["E_high"] = E_high
            outputs["F_edge"] = F_edge

            # Neither SEE output receives gradients from the final depth objective.
            F_edge_in = F_edge.detach()

            fusion_outputs = self.models["depth_head"](
                F_dec,
                F_edge_in,
                E_high,
            )

            outputs["W_soft_mask"] = fusion_outputs.get("W_soft_mask")
            outputs["q_threshold"] = fusion_outputs.get("q_threshold")
            outputs["E_high_raw"] = fusion_outputs.get("E_high_raw")

            # Blend auxiliary and backbone predictions before image reconstruction.
            # Must happen BEFORE generate_images_pred so that warping uses fused depth!
            W_edge = outputs["W_soft_mask"]
            if W_edge is not None:
                self.master_dampening_mask = torch.clamp(1.0 - W_edge, min=0.0, max=1.0)
                # Retain the backbone at very bright or dark image pixels.
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

                        # Blend backbone and auxiliary normalized inverse depth.
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
        """Minimum photometric reprojection and standard edge-aware smoothness."""
        losses = {}
        total_loss = 0

        # Defensive initialization: ensure master_dampening_mask exists (prevent AttributeError from out-of-order scales)
        if not hasattr(self, 'master_dampening_mask'):
            self.master_dampening_mask = None

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

            # Standard Monodepth2 edge-aware smoothness.
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
            is_industrial = True

            if not self.opt.disable_multihead:

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
        models_dir = Path(self.log_path) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        options = {k: v for k, v in vars(self.opt).items() if not k.startswith("_")}
        (models_dir / "opt.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
        for split, lines in (("train", self.opt._train_filenames), ("val", self.opt._val_filenames)):
            (models_dir / (split + "_files_used.txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save_model(self):
        """Publish a checkpoint only after every model and training state is saved."""
        root = Path(self.log_path) / "models"
        final = root / f"weights_{self.epoch}"
        temporary = root / f".weights_{self.epoch}.saving"
        if final.exists() or temporary.exists():
            raise FileExistsError(f"Checkpoint target already exists: {final} or {temporary}")
        temporary.mkdir()
        for name, model in self.models.items():
            real_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            state = real_model.state_dict()
            if name == "encoder":
                state.update(height=self.opt.height, width=self.opt.width, use_stereo=self.opt.use_stereo)
            torch.save(state, temporary / (name + ".pth"))
        state = {"optimizer": self.model_optimizer.state_dict(),
                 "scheduler": self.model_lr_scheduler.state_dict(),
                 "epoch": self.epoch, "phase": self.opt.phase, "seed": self.opt.seed,
                 "optimizer_updates": self.optimizer_updates,
                 "options": {k: v for k, v in vars(self.opt).items() if not k.startswith("_")},
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        torch.save(state, temporary / "adam.pth")
        marker = {"format": 1, "epoch": self.epoch, "phase": self.opt.phase,
                  "optimizer_updates": self.optimizer_updates, "seed": self.opt.seed,
                  "code": self.opt.code_version}
        (temporary / "COMPLETE.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
        os.replace(temporary, final)
        # Keep every epoch checkpoint. Existing checkpoints are never pruned.
        print(f"  Saved complete checkpoint: {final}")

    def load_model(self):
        folder = Path(self.opt.load_weights_folder)
        print(f"Loading {self.opt.phase} initialization: {folder}")
        for name in self.opt.models_to_load:
            if name not in self.models:
                continue
            path = folder / (name + ".pth")
            state = torch.load(path, map_location="cpu", weights_only=False)
            if "state_dict" in state:
                state = state["state_dict"]
            state = {key.removeprefix("module.").removeprefix("_orig_mod."): value
                     for key, value in state.items()}
            if name == "encoder":
                for key in ("height", "width"):
                    if key in state and int(state[key]) != getattr(self.opt, key):
                        raise ValueError(f"Checkpoint {key} differs from training input size")
                state = {key: value for key, value in state.items()
                         if key not in ("height", "width", "use_stereo")}
            self.models[name].load_state_dict(state, strict=True)
            if any(not torch.isfinite(value).all() for value in state.values()
                   if isinstance(value, torch.Tensor) and value.is_floating_point()):
                raise ValueError(f"Nonfinite model weights: {path}")
        if self.opt.reset_optimizer:
            print("  New phase: fresh AdamW and scheduler; backbone optimizer state is not reused.")
            return
        checkpoint = torch.load(folder / "adam.pth", map_location="cpu", weights_only=False)
        if checkpoint.get("phase") != self.opt.phase or checkpoint.get("epoch") != self.opt.start_epoch - 1:
            raise ValueError("Resume epoch/phase disagrees with the checkpoint")
        if checkpoint.get("seed") != self.opt.seed:
            raise ValueError("Resume seed differs from checkpoint")
        saved_options = checkpoint["options"]
        for key in ("batch_size", "accumulation_steps", "num_workers", "height", "width",
                    "target_sparsity", "gamma", "learning_rate", "head_learning_rate", "backbone_learning_rate",
                    "decoder_learning_rate", "backbone_lr_decay", "microbatches_per_epoch"):
            if saved_options.get(key) != getattr(self.opt, key):
                raise ValueError(f"Resume option differs: {key}")
        for split, expected in (("train", self.opt._train_filenames), ("val", self.opt._val_filenames)):
            saved_list = folder.parent / (split + "_files_used.txt")
            if saved_list.read_text(encoding="utf-8").splitlines() != expected:
                raise ValueError(f"Resume {split} list differs from the saved run")
        self.model_optimizer.load_state_dict(optimizer_state_without_unused_gate(
            checkpoint["optimizer"], self.model_optimizer, self.models))
        self.model_lr_scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer_updates = int(checkpoint["optimizer_updates"])
        self.loaded_training_state = checkpoint
        print(f"  Restored AdamW, scheduler and {self.optimizer_updates} phase updates.")

    def cleanup(self):
        """Explicitly close loaders/writers and free models between training phases
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

