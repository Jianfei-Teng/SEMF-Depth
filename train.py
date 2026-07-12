# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

"""
SEMF-Depth Full Module Training Script
======================================
This script trains the complete SEMF-Depth network model (Ablation Stage C).
The model includes both the Dual-Stream FFT Edge Head (SEE) and the
Edge-Guided Soft-Mask Fusion (SMF) with routing enabled.

Default configurations:
  target_sparsity (s) = 0.08  (Sparsity threshold for edge mask)
  gamma (γ)           = 5.0   (Soft-mask activation steepness, defined in depth_head.py)

Usage:
  python train.py
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import warnings

# Force unbuffered stdout to ensure real-time log output
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

# Configure CUDA allocator to reduce memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=FutureWarning, module="DCNv4.functions.dcnv4_func")

from trainer import Trainer
from options import MonodepthOptions


def main():
    options = MonodepthOptions()
    opts = options.parse()

    # --- Complete Module Configuration (Ablation Stage C) ---
    opts.disable_multihead = False      # Enable FFT Edge Head and Depth Head
    opts.disable_routing = False        # Enable Edge-Guided Fusion Routing
    opts.target_sparsity = 0.08         # Set default target sparsity (s = 0.08)
    opts.batch_size = 8                 # Physical batch size
    opts.accumulation_steps = 2         # Gradient accumulation steps

    # Initialize and run training
    trainer = Trainer(opts)
    trainer.train()


if __name__ == "__main__":
    main()
