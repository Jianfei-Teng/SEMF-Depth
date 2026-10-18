# Copyright Niantic 2019. Patent Pending. All rights reserved.
# Licensed under the non-commercial Monodepth2 licence; see LICENSE.
"""Train SEMF: detect a 50-epoch backbone, then refine for 20 epochs."""
import json
import os
import sys

from options import MonodepthOptions
from training_plan import build_plan, describe_plan


def main(args=None):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    options = MonodepthOptions().parse(args)
    phases = build_plan(options)
    print(json.dumps(describe_plan(phases), indent=2, ensure_ascii=False))
    if options.print_config:
        return
    from trainer import Trainer
    for phase in phases:
        trainer = Trainer(phase)
        try:
            trainer.train()
        finally:
            trainer.cleanup()
            del trainer


if __name__ == "__main__":
    main()
