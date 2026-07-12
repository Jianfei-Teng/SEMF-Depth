import os
import sys

# [SOTA FIX: The Stdout Blackhole Defeater]
# Physically disable Python's default full-buffering mechanism, forcing real-time log flushing.
# Regardless of whether bash is invoked with -u, once this code is loaded, logs will never be swallowed!
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

import gc
import time
import torch
import warnings

# Critical: Configure CUDA allocator to reduce fragmentation before any CUDA ops
# [SOTA FIX] MUST BE PYTORCH_CUDA_ALLOC_CONF, otherwise it does NOTHING.
# Changed from expandable_segments:True (requires newer PyTorch) to max_split_size_mb:256 for compatibility
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

# Suppress noisy FutureWarnings from third-party libraries (timm, DCNv4)
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=FutureWarning, module="DCNv4.functions.dcnv4_func")

from options import MonodepthOptions
from trainer import Trainer
from utils import set_seed
import copy

# Ensure global determinism for all ablation configurations
set_seed(1234)

# Hardcode paths for ablation simplicity (Windows compatible)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights","weights_49")

# [MISSION CRITICAL] Assert path existence at second zero.
assert os.path.exists(
    PRETRAINED_WEIGHTS_PATH), f"CRITICAL: Baseline weights NOT FOUND at {PRETRAINED_WEIGHTS_PATH}. Please symlink or copy them."

# --------------------------------------------------------------------------------
# Configurations for Orthogonal Incremental Ablation Study
# --------------------------------------------------------------------------------
CONFIGS = [
    {
        "id": "A",
        "desc": "Baseline (MPViT + HRDepth only)",
        "opts": {
            "disable_multihead": True,
            "phase1_epochs": 0
        },
        "suffix": "ablation_A_baseline",
        "skip": True
    },
    {
        "id": "B",
        "desc": "SEE (Edge Head without routing)",
        "opts": {
            "disable_routing": True
        },
        "suffix": "ablation_B_see",
        "skip": False
    },
    {
        "id": "C",
        "desc": "SMF + SEE (with routing)",
        "opts": {
            "disable_routing": False
        },
        "suffix": "ablation_C_smf_see",
        "skip": True
    },
]


def run_experiment(config, base_opts):
    print(f"\n" + "=" * 60)
    print(f"Running Configuration {config['id']}: {config['desc']}")
    print("=" * 60)

    # 1. Configuration Sandboxing
    opts = copy.deepcopy(base_opts)

    opts.ablation_stage = config["id"]
    for k, v in config["opts"].items():
        setattr(opts, k, v)
        print(f"  Override: {k} = {v}")

    # --- Physical Logic: Dynamic Batch Size Configuration & Gradient Accumulation ---
    target_bs = 8
    accumulation_steps = 2

    opts.batch_size = target_bs
    setattr(opts, "accumulation_steps", accumulation_steps)

    # [WINDOWS RAM STABILITY] Reduce workers to 0 for high-res training in Stage II
    opts.num_workers = 1
    print(f"  [RAM] num_workers set to {opts.num_workers} for Windows stability.")

    # --- [CRITICAL] Learning Rate Alignment ---
    effective_bs = target_bs * accumulation_steps
    lr_scale = effective_bs / 8.0
    opts.learning_rate = base_opts.learning_rate * lr_scale
    opts.head_learning_rate = base_opts.head_learning_rate * lr_scale

    print(
        f"  [SOTA] Stage {config['id']} | Physical BS: {target_bs} | Accum Steps: {accumulation_steps} | Effective BS: {effective_bs} | LR Scale: {lr_scale:.3f}")

    # [T_MAX LOCK] Train for 50 MORE epochs on top of the baseline (which is at epoch 50).
    opts.num_epochs = 70

    opts.model_name = config["suffix"]

    # 2. Check for Resume/Skip (Strict Pathing Protocol)
    trainer_log_path = os.path.join(opts.log_dir, opts.model_name)

    def find_best_checkpoint(folder):
        models_dir = os.path.join(folder, "models")
        if not os.path.isdir(models_dir):
            return -1, None
        ckpts = [f for f in os.listdir(models_dir) if
                 f.startswith("weights_") and
                 f.split("_")[1].isdigit() and
                 len(os.listdir(os.path.join(models_dir, f))) > 0]
        if not ckpts:
            return -1, None
        try:
            latest = sorted(ckpts, key=lambda x: int(x.split("_")[1]))[-1]
            return int(latest.split("_")[1]), os.path.join(models_dir, latest)
        except Exception as e:
            print(f"  [WARN] Checkpoint parse failed: {e}")
            return -1, None

    best_epoch, best_path = find_best_checkpoint(trainer_log_path)

    if best_epoch >= opts.num_epochs - 1:
        print(f"  [SKIP] Stage {config['id']} already reached {opts.num_epochs} epochs. Skipping.")
        return

    if best_path:
        print(f"  [RESUME] Found checkpoint at epoch {best_epoch}: {best_path}")
        opts.load_weights_folder = best_path
        opts.reset_optimizer = False
    else:
        print(f"  [START] New ablation stage. Loading baseline from: {PRETRAINED_WEIGHTS_PATH}")
        opts.load_weights_folder = PRETRAINED_WEIGHTS_PATH
        opts.reset_optimizer = True

    # 3. Instantiate and Run Trainer
    try:
        trainer = Trainer(opts)
        trainer.train()
    except Exception as e:
        print(f"\n[ERROR] Configuration {config['id']} Failed: {e}")
        import traceback
        traceback.print_exc()
        del e  # Break the reference chain
    finally:
        if 'trainer' in locals():
            try:
                trainer.cleanup()
            except Exception as cleanup_err:
                print(f"  [WARNING] Cleanup failed: {cleanup_err}")
            del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        time.sleep(2)
        print(f"Finished {config['id']}. Memory cleaned.")


def main():
    print("Starting Ablation Sequence (A -> D)")
    base_opts = MonodepthOptions().parse()

    for cfg in CONFIGS:
        if cfg.get("skip", False):
            print(f"Skipping {cfg['id']} ({cfg['desc']})")
            continue
        run_experiment(cfg, base_opts)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()