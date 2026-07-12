"""
SEMF-Depth Multi-Model Comparative Visualization Script
====================================
Features:
  1. Iterate test images by scene (supports multiple input paths, e.g. kitti_data/test and gt_depth)
  2. Run inference with 5 sets of model weights, output depth pseudo-color maps + edge maps (if multi-head is available)
  3. Visualize GT depth (loaded from npz or npy files)
  4. Save all results organized by input_path / scene / model subdirectories
Usage:
  Run directly; modify all parameters in the [Manual Configuration] section below.
  python visualize_all.py
"""
from __future__ import absolute_import, division, print_function
import os
import sys
import glob
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms
import cv2
# Ensure the project root directory is in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
import networks
from layers import disp_to_depth
from options import MonodepthOptions
from trainer import Trainer
from utils import colormap
# ==============================================================================
#  [Manual Configuration]
# ==============================================================================
# 1. Model weight list — each entry is (display_name, weight_folder_path)
MODEL_WEIGHTS = [
    ("run1_s002_g5",
     "/home/tyust/PycharmProjects/TJF/tmp/run1_s002_g5/models/weights_69"),
    ("run2_s008_g5",
     "/home/tyust/PycharmProjects/TJF/tmp/run2_s008_g5/models/weights_69"),
    ("run3_s005_g5",
     "/home/tyust/PycharmProjects/TJF/tmp/run3_s005_g5/models/weights_69"),
    ("run4_s005_g2",
     "/home/tyust/PycharmProjects/TJF/tmp/run4_s005_g2/models/weights_69"),
    ("run5_s005_g8",
     "/home/tyust/PycharmProjects/TJF/tmp/run5_s005_g8/models/weights_69"),
]
# 2. Input path list — each entry is (display_label, image_folder_path)
#    Supports multiple input paths; each is processed independently by scene, with results saved to separate subdirectories
DATA_PATHS = [
    ("test",     "/home/tyust/PycharmProjects/TJF_copy/kitti_data/test"),
    ("gt_depth", "/home/tyust/PycharmProjects/TJF/gt_depth"),
]
# 3. Output root directory
OUTPUT_BASE = "./visual_results_comparison"
# 4. GT depth settings (set to None to skip GT visualization; only applies to the first DATA_PATH)
GT_NPZ_PATH = None   # e.g. "/path/to/gt_depths.npz"
GT_NPY_FOLDER = None  # e.g. "/path/to/gt_npy/"
# 5. Image dimensions (must match training settings and be a multiple of 32)
HEIGHT = 480
WIDTH = 640
# 6. Depth range (must match training configuration)
MIN_DEPTH = 0.1
MAX_DEPTH = 80.0
# 7. Whether to disable CUDA
NO_CUDA = False
# ==============================================================================
def apply_colormap_np(depth_array, name="magma_r"):
    """Convert a numpy depth map to a uint8 colorized visualization (H, W, 3)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(name)
    d = depth_array.copy().astype(np.float64)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    colored = cm(d)[:, :, :3]
    return (colored * 255).astype(np.uint8)
# Non-scene directories to skip
_SKIP_DIRS = {"image_02", "image_03", "image_2", "image_3"}
def get_scenes(data_path):
    """Get all scene subfolders, sorted numerically (automatically skips image_02/image_03)"""
    if not os.path.exists(data_path):
        print(f"  ⚠ Data path does not exist: {data_path}")
        return []
    scenes = sorted(
        [d for d in os.listdir(data_path)
         if os.path.isdir(os.path.join(data_path, d))
         and d not in _SKIP_DIRS],
        key=lambda x: int(x) if x.isdigit() else float('inf')
    )
    if not scenes:
        # No subfolders found; treat the current directory as the only scene
        scenes = ["."]
    return scenes
def get_scene_images(data_path, scene):
    """Get all image files (jpg/png) within a scene"""
    scene_dir = os.path.join(data_path, scene) if scene != "." else data_path
    if not os.path.isdir(scene_dir):
        return [], scene_dir
    image_files = sorted([
        f for f in os.listdir(scene_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ])
    return image_files, scene_dir
def build_opts_for_inference(weights_folder, no_cuda=False):
    """Build an opts object for inference (bypassing command-line argument parsing)"""
    options = MonodepthOptions()
    # Parse with an empty argument list to avoid conflicts with external command-line arguments
    opts = options.parser.parse_args([])
    opts.load_weights_folder = weights_folder
    opts.no_cuda = no_cuda
    opts.no_dataset_scan = True
    opts.batch_size = 1
    opts.height = HEIGHT
    opts.width = WIDTH
    opts.min_depth = MIN_DEPTH
    opts.max_depth = MAX_DEPTH
    # Automatically detect whether multi-head weights are available
    edge_head_path = os.path.join(weights_folder, "edge_head.pth")
    depth_head_path = os.path.join(weights_folder, "depth_head.pth")
    if os.path.exists(edge_head_path) and os.path.exists(depth_head_path):
        opts.disable_multihead = False
    else:
        opts.disable_multihead = True
    return opts
def run_model_on_all_paths(model_name, weights_folder, data_paths, output_base, no_cuda):
    """Run inference on all input paths for a single model, avoiding redundant model loading"""
    print(f"\n{'=' * 80}")
    print(f"  LOADING MODEL: {model_name}")
    print(f"  Weights: {weights_folder}")
    print(f"{'=' * 80}")
    if not os.path.isdir(weights_folder):
        print(f"  ⚠ Weights directory does not exist, skipping: {weights_folder}")
        return
    # 1. Build opts and initialize Trainer (loaded only once)
    opts = build_opts_for_inference(weights_folder, no_cuda)
    try:
        trainer = Trainer(opts)
        trainer.load_model()
        trainer.set_eval()
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    has_multihead = not opts.disable_multihead
    # 2. Iterate over each input path
    for path_label, data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"\n  ⚠ Input path does not exist, skipping: [{path_label}] {data_path}")
            continue
        scenes = get_scenes(data_path)
        if not scenes:
            print(f"\n  ⚠ [{path_label}] No scenes found, skipping")
            continue
        print(f"\n  --- [{path_label}] {len(scenes)} scene(s) ---")
        with torch.no_grad():
            for scene in scenes:
                image_names, scene_dir = get_scene_images(data_path, scene)
                if not image_names:
                    continue
                # Create output directory: output_base / path_label / scene_XX / model_name / depth|edge
                scene_label = f"scene_{scene}" if scene != "." else "all"
                depth_out_dir = os.path.join(output_base, path_label, scene_label, model_name, "depth")
                os.makedirs(depth_out_dir, exist_ok=True)
                if has_multihead:
                    edge_out_dir = os.path.join(output_base, path_label, scene_label, model_name, "edge")
                    os.makedirs(edge_out_dir, exist_ok=True)
                print(f"    -> [{path_label}] Scene {scene}: {len(image_names)} images...")
                for img_name in image_names:
                    img_path = os.path.join(scene_dir, img_name)
                    original_img = pil.open(img_path).convert('RGB')
                    w_orig, h_orig = original_img.size
                    # Preprocessing
                    feed_img = original_img.resize((opts.width, opts.height), pil.LANCZOS)
                    feed_img = transforms.ToTensor()(feed_img).unsqueeze(0).to(trainer.device)
                    # === Inference ===
                    features = trainer.models["encoder"](feed_img)
                    outputs = trainer.models["depth"](features)
                    # Multi-head fusion inference
                    if has_multihead:
                        F_dec = outputs[("F_dec", 0)].detach()
                        # Edge Head
                        E_dense, E_high, F_edge = trainer.models["edge_head"](
                            F_dec, rgb_input=feed_img
                        )
                        outputs["E_high"] = E_high
                        # Depth Head Fusion
                        fusion_outputs = trainer.models["depth_head"](
                            F_dec,
                            F_edge.detach(),
                            E_high,
                            E_dense.detach()
                        )
                        # Soft Mask fusion
                        W_edge = fusion_outputs.get("W_soft_mask")
                        if W_edge is not None:
                            master_mask = torch.clamp(1.0 - W_edge, min=0.0, max=1.0)
                            # Sky defense gate
                            rgb_mean = feed_img.mean(dim=1, keepdim=True)
                            sky_mask = ((rgb_mean > 0.95) | (rgb_mean < 0.05)).float()
                            master_mask = torch.max(master_mask, sky_mask)
                            disp_backbone = outputs[("disp", 0)]
                            disp_fusion = fusion_outputs.get(("disp", 0))
                            if disp_fusion is not None:
                                outputs[("disp", 0)] = (
                                    disp_backbone * master_mask +
                                    disp_fusion * (1.0 - master_mask)
                                )
                    # === A. Depth map visualization ===
                    disp = outputs[("disp", 0)]
                    _, depth = disp_to_depth(disp, opts.min_depth, opts.max_depth)
                    depth_vis = colormap(depth[0], name="magma_r")
                    depth_np = (depth_vis.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    depth_np = cv2.resize(depth_np, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(
                        os.path.join(depth_out_dir, img_name),
                        cv2.cvtColor(depth_np, cv2.COLOR_RGB2BGR)
                    )
                    # === B. Edge map visualization ===
                    if has_multihead and "E_high" in outputs:
                        edge_np = (outputs["E_high"][0, 0].cpu().numpy() * 255).astype(np.uint8)
                        edge_np = cv2.resize(edge_np, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
                        cv2.imwrite(os.path.join(edge_out_dir, img_name), edge_np)
                print(f"       ✓ [{path_label}] Scene {scene} done.")
    # 3. Free GPU memory
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
def visualize_gt_from_npz(gt_npz_path, data_path, path_label, scenes, output_base):
    """Visualize GT depth from gt_depths.npz"""
    print(f"\n{'=' * 80}")
    print(f"  PROCESSING: GT Depth (from npz) for [{path_label}]")
    print(f"{'=' * 80}")
    if not os.path.exists(gt_npz_path):
        print(f"  ⚠ GT npz file does not exist: {gt_npz_path}")
        return
    gt_data = np.load(gt_npz_path, allow_pickle=True)
    gt_depths = gt_data["data"]
    print(f"  -> Loaded {len(gt_depths)} GT depth map(s)")
    global_idx = 0
    for scene in scenes:
        image_names, scene_dir = get_scene_images(data_path, scene)
        if not image_names:
            continue
        scene_label = f"scene_{scene}" if scene != "." else "all"
        gt_out_dir = os.path.join(output_base, path_label, scene_label, "GT_depth")
        os.makedirs(gt_out_dir, exist_ok=True)
        print(f"  -> Scene {scene}: {len(image_names)} images...")
        for img_name in image_names:
            if global_idx >= len(gt_depths):
                print(f"  ⚠ Insufficient GT samples; processed {global_idx} so far")
                return
            gt_depth = gt_depths[global_idx]
            # Handle dimensions
            if len(gt_depth.shape) == 3:
                if gt_depth.shape[2] == 3:
                    gt_depth = gt_depth[:, :, 0]
                else:
                    gt_depth = gt_depth.squeeze()
            if gt_depth.max() > 100:
                gt_depth = gt_depth / 1000.0
            # Visualize
            depth_vis = apply_colormap_np(gt_depth, name="magma_r")
            # Get original image dimensions for matching
            img_path = os.path.join(scene_dir, img_name)
            if os.path.exists(img_path):
                original_img = pil.open(img_path)
                w_orig, h_orig = original_img.size
                depth_vis = cv2.resize(depth_vis, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
            basename = os.path.splitext(img_name)[0]
            save_path = os.path.join(gt_out_dir, f"{basename}_gt.png")
            cv2.imwrite(save_path, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
            global_idx += 1
        print(f"     ✓ Scene {scene} GT done.")
def visualize_gt_from_npy(gt_npy_folder, data_path, path_label, scenes, output_base):
    """Visualize GT depth from an npy folder"""
    print(f"\n{'=' * 80}")
    print(f"  PROCESSING: GT Depth (from npy folder) for [{path_label}]")
    print(f"{'=' * 80}")
    if not os.path.exists(gt_npy_folder):
        print(f"  ⚠ GT npy folder does not exist: {gt_npy_folder}")
        return
    gt_files = sorted(glob.glob(os.path.join(gt_npy_folder, "*.npy")))
    if not gt_files:
        print(f"  ⚠ No .npy files found in the GT npy folder")
        return
    gt_depths = [np.load(f) for f in gt_files]
    print(f"  -> Loaded {len(gt_depths)} GT depth map(s)")
    global_idx = 0
    for scene in scenes:
        image_names, scene_dir = get_scene_images(data_path, scene)
        if not image_names:
            continue
        scene_label = f"scene_{scene}" if scene != "." else "all"
        gt_out_dir = os.path.join(output_base, path_label, scene_label, "GT_depth")
        os.makedirs(gt_out_dir, exist_ok=True)
        print(f"  -> Scene {scene}: {len(image_names)} images...")
        for img_name in image_names:
            if global_idx >= len(gt_depths):
                print(f"  ⚠ Insufficient GT samples; processed {global_idx} so far")
                return
            gt_depth = gt_depths[global_idx]
            # Handle dimensions
            if len(gt_depth.shape) == 3:
                if gt_depth.shape[2] == 3:
                    gt_depth = gt_depth[:, :, 0]
                else:
                    gt_depth = gt_depth.squeeze()
            if gt_depth.max() > 100:
                gt_depth = gt_depth / 1000.0
            # Visualize
            depth_vis = apply_colormap_np(gt_depth, name="magma_r")
            # Get original image dimensions for matching
            img_path = os.path.join(scene_dir, img_name)
            if os.path.exists(img_path):
                original_img = pil.open(img_path)
                w_orig, h_orig = original_img.size
                depth_vis = cv2.resize(depth_vis, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
            basename = os.path.splitext(img_name)[0]
            save_path = os.path.join(gt_out_dir, f"{basename}_gt.png")
            cv2.imwrite(save_path, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
            global_idx += 1
        print(f"     ✓ Scene {scene} GT done.")
def main():
    """Main entry point: iterate over all models and all input paths"""
    print("=" * 80)
    print("  SEMF-Depth Multi-Model · Multi-Path Comparative Visualization")
    print(f"  Number of models: {len(MODEL_WEIGHTS)}")
    print(f"  Input paths: {len(DATA_PATHS)}")
    for label, path in DATA_PATHS:
        print(f"    [{label}] {path}")
    print(f"  Output directory: {OUTPUT_BASE}")
    print("=" * 80)
    # 1. Validate input paths and collect scenes
    valid_paths = []
    for label, path in DATA_PATHS:
        if os.path.exists(path):
            scenes = get_scenes(path)
            valid_paths.append((label, path, scenes))
            print(f"\n  [{label}] {len(scenes)} scene(s): {scenes[:10]}{'...' if len(scenes) > 10 else ''}")
        else:
            print(f"\n  ⚠ [{label}] Path does not exist, skipping: {path}")
    if not valid_paths:
        print("  ✗ All input paths do not exist, exiting.")
        return
    # 2. Create output root directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    # 3. Run inference per model (each model is loaded once, then applied to all paths)
    for model_name, weights_folder in MODEL_WEIGHTS:
        try:
            run_model_on_all_paths(
                model_name, weights_folder,
                [(label, path) for label, path, _ in valid_paths],
                OUTPUT_BASE, NO_CUDA
            )
        except Exception as e:
            print(f"\n  ✗ CRITICAL ERROR in {model_name}: {e}")
            import traceback
            traceback.print_exc()
    # 4. GT depth visualization (only for the first valid path)
    if GT_NPZ_PATH is not None and valid_paths:
        label, path, scenes = valid_paths[0]
        try:
            visualize_gt_from_npz(GT_NPZ_PATH, path, label, scenes, OUTPUT_BASE)
        except Exception as e:
            print(f"\n  ✗ GT (npz) visualization failed: {e}")
    if GT_NPY_FOLDER is not None and valid_paths:
        label, path, scenes = valid_paths[0]
        try:
            visualize_gt_from_npy(GT_NPY_FOLDER, path, label, scenes, OUTPUT_BASE)
        except Exception as e:
            print(f"\n  ✗ GT (npy) visualization failed: {e}")
    # 5. Done
    print(f"\n{'=' * 80}")
    print(f"  ✓ All done! Results saved to: {OUTPUT_BASE}")
    print(f"  Output directory structure:")
    print(f"    {OUTPUT_BASE}/")
    for label, _, _ in valid_paths:
        print(f"      {label}/                      ← input path label")
        print(f"        scene_01/")
        print(f"          run1_s002_g5/depth/       ← depth pseudo-color")
        print(f"          run1_s002_g5/edge/        ← edge map (if multi-head available)")
        print(f"          run2_s008_g5/depth/")
        print(f"          ...")
        print(f"          GT_depth/                 ← GT depth (if configured)")
        print(f"        scene_02/")
        print(f"          ...")
    print(f"{'=' * 80}")
if __name__ == "__main__":
    main()