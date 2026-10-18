# Copyright Niantic 2019. Patent Pending. All rights reserved.
# Licensed under the non-commercial Monodepth2 licence; see LICENSE.
"""Command-line options for the released SEMF training pipeline."""
import argparse


class MonodepthOptions:
    def __init__(self):
        p = self.parser = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--data_path", default="./ancient_architecture_data")
        p.add_argument("--split_dir", default=None,
                       help="Directory with train_files.txt and val_files.txt; default: DATA/splits/eigen_zhou")
        p.add_argument("--log_dir", default="./tmp")
        p.add_argument("--model_name", default="SEMF")
        p.add_argument("--stage", choices=("auto", "backbone", "refine"), default="auto",
                       help="auto: load a 50-epoch backbone, or train it first if absent")
        p.add_argument("--backbone_weights", "--load_weights_folder", dest="backbone_weights",
                       default="./weights",
                       help="50-epoch backbone directory (four model PTHs), or its parent")
        p.add_argument("--resume", nargs="?", const="auto", default=None,
                       help="Resume a complete checkpoint; omit path to find the latest checkpoint in this run")
        p.add_argument("--print_config", action="store_true",
                       help="Verify data/checkpoint selection and print the plan without training")
        p.add_argument("--seed", type=int, default=1234)
        p.add_argument("--height", type=int, default=480)
        p.add_argument("--width", type=int, default=640)
        p.add_argument("--batch_size", type=int, default=8)
        p.add_argument("--accumulation_steps", type=int, default=2)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--learning_rate", type=float, default=2e-6,
                       help="Actual refinement LR for decoder and pose encoder; no implicit scaling")
        p.add_argument("--head_learning_rate", type=float, default=2e-4,
                       help="Actual refinement LR for SMF and pose decoder")
        p.add_argument("--backbone_learning_rate", type=float, default=5e-5,
                       help="Depth-encoder LR when training the initial backbone from scratch")
        p.add_argument("--decoder_learning_rate", type=float, default=1e-4,
                       help="Other-model LR when training the initial backbone from scratch")
        p.add_argument("--backbone_lr_decay", type=float, default=0.9,
                       help="Epoch-wise exponential decay for the from-scratch backbone phase")
        p.add_argument("--target_sparsity", type=float, default=0.08)
        p.add_argument("--min_depth", type=float, default=0.1)
        p.add_argument("--max_depth", type=float, default=80.0)
        p.add_argument("--disparity_smoothness", type=float, default=1e-3)
        p.add_argument("--weights_init", choices=("pretrained", "scratch"), default="pretrained",
                       help="Pose-encoder initialization for from-scratch backbone training")
        p.add_argument("--encoder_pretrained", default=None,
                       help="Optional local MPViT initialization for the from-scratch backbone phase")
        p.add_argument("--png", action="store_true", help="Use PNG training images instead of JPG")
        p.add_argument("--no_cuda", action="store_true")
        p.add_argument("--no_compile", action="store_true",
                       help="Disable Linux torch.compile without changing the model")
        p.add_argument("--log_frequency", type=int, default=250)
        # Fixed released architecture/loss. These attributes are used by Trainer.
        p.set_defaults(backbone="mpvit", num_layers=18, dataset="custom", split="eigen_zhou",
                       scales=[0, 1, 2, 3], frame_ids=[0, -1, 1], use_stereo=False,
                       pose_model_input="pairs", pose_model_type="separate_resnet",
                       predictive_mask=False, disable_automasking=False,
                       avg_reprojection=False, no_ssim=False, v1_multiscale=False,
                       disable_multihead=False, disable_routing=False, save_frequency=1)

    def parse(self, args=None):
        return self.parser.parse_args(args)
