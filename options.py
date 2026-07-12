# Copyright Niantic 2019. Patent Pending. All rights reserved.
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


def get_default_eval_data_path():
    # Default to TJF/kitti_data for evaluation (Fixed from splits/kitti_data)
    return os.path.join(file_dir, "kitti_data")


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(file_dir, "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "custom"],
                                 default="custom")
        self.parser.add_argument("--backbone",
                                 type=str,
                                 help="backbone model (resnet, mpvit, or hrnet18).",
                                 choices=["resnet", "mpvit", "hrnet18"],
                                 default="mpvit")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="custom",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "custom"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=480)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth (80m for ancient architecture)",
                                 default=80.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        # OPTIMIZATION options (Hardware-Specific for RTX 5090 & 640x480)
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="Physical batch size (Default 8 for stability)",
                                 default=8)
        self.parser.add_argument("--accumulation_steps",
                                 type=int,
                                 help="Gradient accumulation steps to reach effective batch size (Default 2)",
                                 default=2)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate for backbone (1e-5) to prevent catastrophic forgetting",
                                 default=1e-5)
        self.parser.add_argument("--head_learning_rate",
                                 type=float,
                                 help="learning rate for randomly initialized heads (1e-4). 10x backbone LR.",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs (Set to 20 for SOTA strategy)",
                                 default=50)
        self.parser.add_argument("--phase1_epochs",
                                 type=int,
                                 help="Phase 1: Freeze encoder + decoder, only train heads. (Default 5)",
                                 default=5)

        # MULTI-HEAD ARCHITECTURE options (Physics-Based Constraints)
        self.parser.add_argument("--alpha_rgb",
                                 type=float,
                                 help="RGB Gradient balancing coefficient in soft mask (0.8).",
                                 default=0.8)
        self.parser.add_argument("--fft_radius",
                                 type=int,
                                 help="Low-Pass Filter Radius (r=30 for 640x480). Wider coverage for macro-structures.",
                                 default=30)
        self.parser.add_argument("--warmup_epochs",
                                 type=int,
                                 help="Warmup Epochs (5). Lock high-freq/repulsive/normal modules to prevent toxic gradient transfer.",
                                 default=5)
        self.parser.add_argument("--loss_start_epoch",
                                 type=int,
                                 help="Epoch to engage Frequency/Repulsive losses (5). Syncs with Engagement Phase.",
                                 default=5)
        self.parser.add_argument("--disable_multihead",
                                 help="ABLATION: Disable Multi-Head Architecture (Baseline A)",
                                 action="store_true")
        self.parser.add_argument("--disable_routing",
                                 help="ABLATION: Disable edge-guided routing in fusion",
                                 action="store_true")
        self.parser.add_argument("--reset_optimizer",
                                 help="If set, Adam momentum/buffers are reset to zero (Cold Start).",
                                 action="store_true")

        self.parser.add_argument("--target_sparsity",
                                 type=float,
                                 help="Target sparsity for mask cleaning (0.05 = 5%).",
                                 default=0.08)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=2)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load (Default: epoch 49 weights for backbone initialization)",
                                 default=None)
        self.parser.add_argument("--resume",
                                 help="If set, automatically finds the latest checkpoint in the current model folder and resumes training from there.",
                                 action="store_true")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose", "edge_head", "depth_head"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--ablation_stage",
                                 type=str,
                                 help="Auto-configure opts for Ablation Stage (e.g., A, B, C, F)",
                                 default=None)
        self.parser.add_argument("--ablation_epoch",
                                 type=int,
                                 nargs="+",
                                 help="Specific epoch weight(s) to load (e.g., 0 5 10). If provided, will evaluate all.",
                                 default=None)
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--gt_path",
                                 type=str,
                                 help="path to ground truth .npy files directory (for Linux evaluation)")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--no_dataset_scan",
                                 help="if set skips scanning all image files for training split discovery",
                                 action="store_true")
        self.parser.add_argument("--eval_data_path",
                                 type=str,
                                 help="path to evaluation data root (for CustomDataset)",
                                 default=get_default_eval_data_path())

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
