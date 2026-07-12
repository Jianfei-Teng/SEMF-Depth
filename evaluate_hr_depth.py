from __future__ import absolute_import, division, print_function

import warnings
# Filter warnings immediately after future imports
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*")
warnings.filterwarnings("ignore", category=FutureWarning)
# Specific suppression for DCNv4/torch.amp warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_bwd.*")

import os
import cv2
import numpy as np
import glob
import pandas as pd
import torch
import torch.nn.functional as F
import gc
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

from layers import disp_to_depth
from utils import set_seed
from options import MonodepthOptions
import datasets
from trainer import Trainer

cv2.setNumThreads(0)

# Hardware setup
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

try:
    torch.set_float32_matmul_precision('high')
except:
    pass


class SceneTestDataset(Dataset):
    """Simple Dataset that loads images explicitly from a given list of paths."""

    def __init__(self, image_paths, height, width):
        super(SceneTestDataset, self).__init__()
        self.image_paths = image_paths
        self.height = height
        self.width = width
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((self.height, self.width), interpolation=Image.Resampling.LANCZOS)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                color = img.convert('RGB')
        color = self.resize(color)
        color_t = self.to_tensor(color)
        return {("color", 0, 0): color_t, "image_path": image_path}


STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    # Standard a3 calculation without artificial noise
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_gt_depths_from_folder(gt_folder):
    if not os.path.exists(gt_folder):
        raise FileNotFoundError(f"GT folder not found: {gt_folder}")

    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.npy")))
    if len(gt_files) == 0:
        raise FileNotFoundError(f"No .npy files in GT folder: {gt_folder}")

    gt_depths = []
    for f in gt_files:
        gt = np.load(f)
        gt_depths.append(gt)

    print(f"-> Loaded {len(gt_depths)} GT depths from folder")
    return gt_depths


def load_test_filenames(test_file_path):
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test files list not found: {test_file_path}")

    with open(test_file_path, 'r') as f:
        lines = f.read().splitlines()
    print(f"-> Loaded {len(lines)} test filenames")
    return lines


def evaluate(opt, epoch=None):
    """Evaluates a pretrained model using a specified test set
    """
    # Fix seed for determinism in sampling/sorting
    set_seed(1234)

    # --- Benchmark Evaluation Boundaries (Strict KITTI Standard) ---
    # Used ONLY for masking out invalid Ground Truth pixels
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    print("\n" + "#" * 60)
    print(f" EVALUATING ROUND: {epoch if epoch is not None else 'Default'}")
    print("#" * 60)

    # Force disable dataset scanning and reduce batch size in Trainer
    opt.no_dataset_scan = True
    opt.batch_size = 1

    # --- Ablation Auto-Configuration ---
    if hasattr(opt, 'ablation_stage') and opt.ablation_stage:
        stage = opt.ablation_stage.upper()

        # Base mappings — must match run_ablation.py exactly
        configs = {
            "A": {"model_name": "ablation_A_baseline", "disable_multihead": True, "phase1_epochs": 0},
            "B": {"model_name": "ablation_B_see",
                  "disable_routing": True},
            "C": {"model_name": "ablation_C_smf_see",
                  "disable_routing": False},
        }

        if stage in configs:
            cfg = configs[stage]
            for k, v in cfg.items():
                setattr(opt, k, v)
        elif stage == "WEIGHTS_49":
            opt.model_name = "weights_49"
            opt.disable_multihead = False
            opt.load_weights_folder = os.path.join(os.path.dirname(__file__), "weights", "weights_49")
    # --- Weight Path Resolution (Local Relative) ---
    # FORCE re-resolution if ablation_stage is provided, OR if no folder is set
    should_resolve = (getattr(opt, 'ablation_stage', None) is not None) or (
                getattr(opt, 'load_weights_folder', None) is None)

    if should_resolve:
        root = os.path.dirname(os.path.abspath(__file__))

        # 1. First priority: local weights/ folder (for exported models)
        local_weights = os.path.join(root, "weights", opt.model_name)
        # 2. Second priority: local tmp/ folder (for active training)
        local_tmp = os.path.join(root, "tmp", opt.model_name, "models")

        # Determine base directory
        if os.path.exists(local_weights):
            base_dir = local_weights
        elif os.path.exists(local_tmp):
            base_dir = local_tmp
        else:
            # Fallback for ablation folders
            base_dir = os.path.join(root, "tmp", opt.model_name, opt.model_name, "models")

        if epoch is not None:
            opt.load_weights_folder = os.path.join(base_dir, f"weights_{epoch}")
        else:
            opt.load_weights_folder = base_dir

        # Emergency check: if the path contains wrong root, fix it
        if "PycharmProjects\\TJF  Mono-vit\\" in opt.load_weights_folder:
            opt.load_weights_folder = opt.load_weights_folder.replace("PycharmProjects\\TJF  Mono-vit\\",
                                                                      "PycharmProjects\\TJF  Mono-vit - copy\\")

    # 1. Initialize Trainer (Handles Model Creation & Loading)
    trainer = Trainer(opt)

    # Set to evaluation mode
    trainer.set_eval()

    # 2. Setup Data
    project_root = os.path.dirname(os.path.abspath(__file__))
    split_folder = os.path.join(project_root, "splits", opt.eval_split)
    gt_folder = os.path.join(split_folder, "gt_npy")
    test_files_path = os.path.join(split_folder, "test_files.txt")

    if opt.gt_path:
        gt_folder = opt.gt_path

    # Load GT and Filenames
    gt_depths_list = load_gt_depths_from_folder(gt_folder)

    # [FIX] Use opt.data_path instead of hardcoded kitti_data
    if os.path.exists(os.path.join(project_root, "kitti_data", "test")):
        eval_data_root = os.path.join(project_root, "kitti_data", "test")
    else:
        eval_data_root = opt.data_path
    
    scenes = [f"{i:02d}" for i in range(1, 13)]

    all_errors = []
    all_ratios = []
    per_scene_results = []

    print(f"\n-> Start scene-by-scene evaluation (01 -> 12)...")

    for scene in scenes:
        scene_dir = os.path.join(eval_data_root, scene)
        if not os.path.exists(scene_dir):
            continue

        image_files = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
        if not image_files:
            continue

        dataset = SceneTestDataset(image_files, opt.height, opt.width)
        dataloader = DataLoader(dataset, 4, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

        print(f"-> Processing Scene {scene} ({len(image_files)} images)...")

        pred_disps = []
        valid_indices = []
        valid_paths = []

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].to(trainer.device)
                paths = data["image_path"]

                features = trainer.models["encoder"](input_color)
                outputs = trainer.models["depth"](features)

                disp_final = outputs[("disp", 0)]

                if not getattr(opt, 'disable_multihead', False) and "edge_head" in trainer.models and \
                        trainer.models["edge_head"] is not None:
                    F_dec = outputs[("F_dec", 0)].detach()
                    E_dense, E_high, F_edge = trainer.models["edge_head"](F_dec, rgb_input=input_color)

                    if "depth_head" in trainer.models and trainer.models["depth_head"] is not None:
                        fusion_outputs = trainer.models["depth_head"](
                            outputs[("F_dec", 0)],
                            F_edge.detach(),
                            E_high,
                            E_dense.detach(),
                        )

                        W_edge = fusion_outputs.get("W_soft_mask")
                        disp_fusion = fusion_outputs.get(("disp", 0))

                        if W_edge is not None and disp_fusion is not None:
                            master_mask = torch.clamp(1.0 - W_edge, min=0.0, max=1.0)
                            rgb_mean = input_color.mean(dim=1, keepdim=True)
                            sky_mask = ((rgb_mean > 0.95) | (rgb_mean < 0.05)).float()
                            master_mask = torch.max(master_mask, sky_mask)

                            disp_final = disp_final * master_mask + disp_fusion * (1.0 - master_mask)

                pred_disp, _ = disp_to_depth(disp_final, opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                for k in range(pred_disp.shape[0]):
                    pred_disps.append(pred_disp[k])
                    path = paths[k]
                    # Get integer index from filename e.g. "0000000167.jpg" -> 167
                    idx = int(os.path.splitext(os.path.basename(path))[0])
                    valid_indices.append(idx)
                    valid_paths.append(path)

        scene_errors = []
        for i, global_idx in enumerate(valid_indices):
            if global_idx >= len(gt_depths_list):
                print(f"[Warning] Index {global_idx} out of range for GT list size {len(gt_depths_list)}!")
                continue

            gt_depth = gt_depths_list[global_idx]
            if gt_depth.max() > 100:
                gt_depth = gt_depth / 1000.0

            gt_height, gt_width = gt_depth.shape[:2]
            p_disp = pred_disps[i]
            if p_disp.shape[:2] != (gt_height, gt_width):
                p_disp = cv2.resize(p_disp, (gt_width, gt_height))

            pred_depth = 1 / p_disp

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_depth_masked = pred_depth[mask]
            gt_depth_masked = gt_depth[mask]

            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth_masked) / np.median(pred_depth_masked)
                all_ratios.append(ratio)
                pred_depth_masked *= ratio

            pred_depth_masked[pred_depth_masked < MIN_DEPTH] = MIN_DEPTH
            pred_depth_masked[pred_depth_masked > MAX_DEPTH] = MAX_DEPTH

            errs = compute_errors(gt_depth_masked, pred_depth_masked)
            scene_errors.append(errs)
            all_errors.append((errs, valid_paths[i]))

        if scene_errors:
            m_errs = np.array(scene_errors).mean(0)
            print(" | ".join([f"{v:8.4f}" for v in m_errs.tolist()]))
            per_scene_results.append({"scene": scene, "metrics": m_errs.tolist()})

    if not opt.disable_median_scaling and all_ratios:
        ratios_np = np.array(all_ratios)
        med = np.median(ratios_np)
        print("\n Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios_np / med)))

    if not all_errors:
        print("ERROR: No valid samples found for evaluation!")
        return [0.0] * 7

    # --- Final Metrics Computation ---
    if len(all_errors) > 0:
        error_metrics = [e[0] for e in all_errors]
        mean_errors = np.array(error_metrics).mean(0)
        per_scene_results.append({"scene": "ALL", "metrics": mean_errors.tolist()})

    return per_scene_results


def save_per_scene_results_to_txt(all_results, project_root, stage_name):
    """Saves evaluation results grouped by scene for a specific stage"""
    txt_path = os.path.join(project_root, f"hr_evaluation_per_scene_{stage_name}.txt")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Collect all scenes
    scene_set = set()
    for r in all_results:
        for m_obj in r["metrics"]:
            scene_set.add(m_obj["scene"])

    # Sort numeric sort for scenes "01", "02" and "ALL" at end
    scene_list = sorted([s for s in scene_set if s != "ALL"])
    if "ALL" in scene_set:
        scene_list.append("ALL")

    header = ["Stage", "Epoch", "AbsRel", "SqRel", "RMSE", "RMSElog", "a1", "a2", "a3"]
    header_str = "{:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}".format(*header)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n" + "=" * 95 + "\n")
        f.write(f" [{timestamp}] STAGE {stage_name} HR EVALUATION SCENE-BY-SCENE\n")
        f.write("=" * 95 + "\n")

        for sc in scene_list:
            f.write(f"\n--- SCENE: {sc} ---\n")
            f.write(header_str + "\n")
            f.write("-" * 98 + "\n")

            for r in all_results:
                stage = r.get("stage", "Unknown")
                ep = r["epoch"]

                # Find metrics for this scene
                match = next((x["metrics"] for x in r["metrics"] if x["scene"] == sc), None)
                if match:
                    row_str = "{:>10} | {:>10} | {:8.4f} | {:8.4f} | {:8.4f} | {:8.4f} | {:8.4f} | {:8.4f} | {:8.4f}".format(
                        stage, ep, match[0], match[1], match[2], match[3], match[4], match[5], match[6])
                    f.write(row_str + "\n")

        f.write("\n" + "=" * 95 + "\n")

    print(f"-> Scene metrics for Stage {stage_name} saved to: {txt_path}\n")
    print("\n\n" + "=" * 95)
    print(" FINAL GROUPED ABLATION SUMMARY")
    print("=" * 95)
    for sc in scene_list:
        print(f"\n--- SCENE: {sc} ---")
        print(header_str)
        print("-" * 98)
        for r in all_results:
            stage = r.get("stage", "Unknown")
            ep = r["epoch"]
            match = next((x["metrics"] for x in r["metrics"] if x["scene"] == sc), None)
            if match:
                row_str = "{:>10} | {:>10} | {:8.4f} | {:8.4f} | {:8.4f} | {:8.4f} | {:8.4f} | {:8.4f} | {:8.4f}".format(
                    stage, ep, match[0], match[1], match[2], match[3], match[4], match[5], match[6])
                print(row_str)
    print("=" * 95 + "\n")


if __name__ == "__main__":
    # ===========================================================
    # [User Configuration Area] Modify stage and epochs directly here
    # ===========================================================
    STAGES = ["C"]
    EPOCHS = ["99"]  # e.g.: [5, 10, 20] or [8] or ["best"]
    # ===========================================================

    options = MonodepthOptions()

    stages = STAGES if isinstance(STAGES, list) else [STAGES]
    epochs = EPOCHS if isinstance(EPOCHS, list) else ([EPOCHS] if EPOCHS is not None else [None])

    for stage in stages:
        opts = options.parse()
        opts.ablation_stage = stage
        stage_results = []
        for ep in epochs:
            try:
                res = evaluate(opts, epoch=ep)
                stage_results.append({"epoch": ep if ep is not None else "latest", "metrics": res, "stage": stage})
            except Exception as e:
                print(f"[ERROR] Round {ep} failed: {e}")
                import traceback

                traceback.print_exc()

        # Save to TXT for this specific stage
        if stage_results:
            save_per_scene_results_to_txt(stage_results, os.path.dirname(os.path.abspath(__file__)), stage)