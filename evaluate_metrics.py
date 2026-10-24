"""Evaluate seven depth metrics and calibrated Chamfer distance in one pass.

All eight metrics use the same prediction, valid pixels, median alignment,
and scene selection. Camera intrinsics use full-image coordinates even when
the evaluation mask crops the depth maps. No training objects are constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
SELECTED_SCENES = ("01", "03", "05", "07", "08", "10", "11", "12",
                   "加3", "加5", "加9", "加10")
DEPTH_METRICS = ("AbsRel", "SqRel", "RMSE", "RMSElog", "a1", "a2", "a3")
METRICS = DEPTH_METRICS + ("CD",)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Sample:
    scene: str
    source: str
    image: Path
    ground_truth: Path


def make_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Exact checkpoint folder containing encoder.pth and depth.pth")
    parser.set_defaults(model="full")
    parser.add_argument("--data-root", type=Path, default=Path("./ancient_architecture_data"))
    parser.add_argument("--test-files", type=Path,
                        help="Fixed scene, image, GT manifest (tab-separated, relative to --data-root)")
    parser.add_argument("--test-root", type=Path,
                        help="Original test root containing scene folders 01, 02, ...")
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument("--test-gt-pairing", choices=("index", "stem"), default="index",
                        help="index preserves old zero-based image ID -> sorted NPY list mapping")
    parser.add_argument("--extra-test-root", type=Path,
                        help="Additional test root, or its gt_depth child with paired JPG/NPY files")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Fixed scene selection; default is every scene in the test manifest")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--min-depth", type=float, default=0.1,
                        help="Network inverse-depth conversion minimum, matching released options")
    parser.add_argument("--max-depth", type=float, default=80.0,
                        help="Network inverse-depth conversion maximum, matching released options")
    parser.add_argument("--eval-min-depth", type=float, default=1e-3)
    parser.add_argument("--eval-max-depth", type=float, default=80.0)
    parser.add_argument("--crop", choices=("garg", "none"), default="garg")
    parser.add_argument("--gt-unit", choices=("auto", "m", "mm"), default="auto",
                        help="auto preserves old rule: divide GT by 1000 when its maximum exceeds 100")
    parser.add_argument("--disable-median-scaling", action="store_true")
    parser.add_argument("--target-sparsity", type=float, default=0.08)
    parser.add_argument("--gamma", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--intrinsics", type=Path, default=Path("./splits/camera_intrinsics.json"))
    parser.add_argument("--intrinsics-format", choices=("normalized", "pixel"), default="normalized")
    parser.add_argument("--intrinsics-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--cd-max-points", type=int, default=20000,
                        help="Points per cloud for CD; 0 uses all valid points")
    parser.add_argument("--cd-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        help="Separate result directory; never overwrites previous results by default")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify image/GT pairing and counts without loading or running a model")
    return parser


def scene_key(scene):
    return (1, int(scene[1:])) if scene.startswith("加") else (0, int(scene))


def collect_samples(args):
    if args.test_files is not None:
        samples = []
        for line in args.test_files.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            scene, image, gt = line.split("\t")
            for value in (image, gt):
                if Path(value).is_absolute() or ".." in Path(value).parts:
                    raise ValueError(f"Manifest paths must be relative to the data root: {value}")
            if args.scenes is None or scene in args.scenes:
                samples.append(Sample(scene, "test", args.data_root / image, args.data_root / gt))
        if not samples:
            raise ValueError("The fixed test manifest selects no images")
        if args.scenes is not None and set(args.scenes) != {s.scene for s in samples}:
            raise ValueError("Requested scene missing from fixed test manifest")
        for sample in samples:
            if not sample.image.is_file() or not sample.ground_truth.is_file():
                raise FileNotFoundError(f"Manifest image/GT missing: {sample}")
        if len({s.image.resolve() for s in samples}) != len(samples):
            raise ValueError("Duplicate test image in manifest")
        if len({s.ground_truth.resolve() for s in samples}) != len(samples):
            raise ValueError("Duplicate test ground truth in manifest")
        return samples
    requested = args.scenes
    if requested is None:
        requested = sorted([p.name for p in args.test_root.iterdir()
                            if p.is_dir() and p.name.isdigit()], key=scene_key)
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate scene selection")
    allowed = set(SELECTED_SCENES) | {f"{i:02d}" for i in range(1, 13)}
    allowed |= {f"加{i}" for i in range(1, 12)}
    if set(requested) - allowed:
        raise ValueError(f"Unknown scene identifiers: {set(requested) - allowed}")
    gt_files = sorted(args.gt_root.glob("*.npy"))
    stem_map = {}
    if args.test_gt_pairing == "stem":
        for path in gt_files:
            key = int(path.stem)
            if key in stem_map:
                raise ValueError(f"Ambiguous numeric GT filename: {path}")
            stem_map[key] = path
    extra = args.extra_test_root
    if extra is not None and (extra / "gt_depth").is_dir():
        extra = extra / "gt_depth"
    samples = []
    for scene in sorted(requested, key=scene_key):
        is_extra = scene.startswith("加")
        if is_extra:
            if extra is None:
                raise ValueError(f"Scene {scene} requires --extra-test-root")
            folder = extra / str(int(scene[1:]))
        else:
            folder = args.test_root / scene
        if not folder.is_dir():
            raise FileNotFoundError(f"Selected scene folder missing: {folder}")
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"Selected scene has no images: {folder}")
        for image in images:
            if is_extra or not image.stem.isdigit():
                gt = image.with_suffix(".npy")
            elif scene in {"02", "04", "06", "09"} and image.with_suffix(".npy").is_file():
                gt = image.with_suffix(".npy")
            elif args.test_gt_pairing == "index":
                index = int(image.stem)
                if index < 0 or index >= len(gt_files):
                    raise ValueError(f"Image ID outside sorted GT list: {image}")
                gt = gt_files[index]
            else:
                gt = stem_map.get(int(image.stem))
            if gt is None or not gt.is_file():
                raise FileNotFoundError(f"Ground truth missing for {image}: {gt}")
            samples.append(Sample(scene, "extra" if is_extra else "test", image, gt))
    if len({s.image.resolve() for s in samples}) != len(samples):
        raise ValueError("Duplicate image in selected cohort")
    if len({s.ground_truth.resolve() for s in samples}) != len(samples):
        raise ValueError("Two selected images are paired with the same ground truth")
    return samples


def load_ground_truth(path, unit):
    depth = np.load(path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW ground truth: {path}, got {depth.shape}")
    depth = depth.copy()
    finite = depth[np.isfinite(depth)]
    if not len(finite):
        raise ValueError(f"Ground truth has no finite values: {path}")
    if unit == "mm" or (unit == "auto" and finite.max() > 100):
        depth = depth / 1000.0
    return depth


def align_depth(gt, scaled_inverse_depth, args):
    inverse = np.asarray(scaled_inverse_depth)
    if inverse.shape != gt.shape:
        import cv2
        inverse = cv2.resize(inverse, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    if not np.isfinite(inverse).all() or np.any(inverse <= 0):
        raise ValueError("Prediction contains nonfinite or nonpositive inverse depth")
    # Preserve the old evaluator's float32 prediction arithmetic and in-place
    # median scaling rather than silently promoting the seven metrics to float64.
    predicted = 1.0 / inverse
    valid = np.isfinite(gt) & (gt > args.eval_min_depth) & (gt < args.eval_max_depth)
    if args.crop == "garg":
        h, w = gt.shape
        top, bottom, left, right = np.array([0.40810811*h, 0.99189189*h,
                                            0.03594771*w, 0.96405229*w]).astype(int)
        crop = np.zeros(gt.shape, dtype=bool)
        crop[top:bottom, left:right] = True
        valid &= crop
    if not valid.any():
        raise ValueError("No valid ground-truth pixels after range/crop masking")
    ratio = 1.0
    if not args.disable_median_scaling:
        ratio = float(np.median(gt[valid]) / np.median(predicted[valid]))
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Invalid median scale ratio")
        predicted *= ratio
    predicted = np.clip(predicted, args.eval_min_depth, args.eval_max_depth)
    return predicted, valid, ratio


def compute_seven(gt, predicted):
    # The pre-cleanup compute_errors wrapper delegates to chamfer_metric, which
    # promotes the already-aligned valid depths to float64 for these formulas.
    gt = np.asarray(gt, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    threshold = np.maximum(gt / predicted, predicted / gt)
    errors = (np.mean(np.abs(gt-predicted) / gt), np.mean((gt-predicted)**2 / gt),
              np.sqrt(np.mean((gt-predicted)**2)),
              np.sqrt(np.mean((np.log(gt)-np.log(predicted))**2)),
              np.mean(threshold < 1.25), np.mean(threshold < 1.25**2),
              np.mean(threshold < 1.25**3))
    if not np.isfinite(errors).all() or not errors[4] <= errors[5] <= errors[6]:
        raise ValueError("Invalid seven-metric result")
    return np.asarray(errors)


def load_intrinsics(path, mode, reference_size=None):
    matrix = np.asarray(json.loads(Path(path).read_text(encoding="utf-8-sig")), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Intrinsics JSON must contain a finite 3x3 matrix")
    if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError("Invalid camera intrinsic matrix")
    if mode == "pixel" and (reference_size is None or min(reference_size) <= 0):
        raise ValueError("Pixel intrinsics require positive --intrinsics-size WIDTH HEIGHT")
    return matrix


def pixel_intrinsics(matrix, shape, mode="normalized", reference_size=None):
    result = np.array(matrix, dtype=np.float64, copy=True)
    h, w = shape
    if mode == "normalized":
        result[0] *= w
        result[1] *= h
    else:
        ref_w, ref_h = reference_size
        result[0] *= w / ref_w
        result[1] *= h / ref_h
    return result


def point_cloud(depth, valid, pixel_matrix):
    ys, xs = np.nonzero(valid)
    pixels = np.column_stack((xs, ys, np.ones(len(xs))))
    rays = pixels @ np.linalg.inv(pixel_matrix).T
    return rays * depth[ys, xs, None]


def compute_cd(gt, predicted, valid, matrix, mode="normalized", reference_size=None,
               max_points=20000, seed=1234, workers=4):
    from scipy.spatial import cKDTree
    k = pixel_intrinsics(matrix, gt.shape, mode, reference_size)
    clouds = [point_cloud(depth, valid, k) for depth in (gt, predicted)]
    for index, points in enumerate(clouds):
        if max_points and len(points) > max_points:
            # Preserve recorded CD sampling: independent GT/prediction subsets.
            selected = np.random.RandomState(seed + index).choice(len(points), max_points, replace=False)
            clouds[index] = points[selected]
    first, second = clouds
    if not len(first) or not len(second):
        raise ValueError("Empty CD point cloud")
    d12 = cKDTree(second).query(first, workers=workers)[0]
    d21 = cKDTree(first).query(second, workers=workers)[0]
    result = float(np.mean(d12**2) + np.mean(d21**2))
    if not np.isfinite(result):
        raise ValueError("Nonfinite CD")
    return result


def compute_metrics(gt, predicted, valid, matrix, args):
    seven = compute_seven(gt[valid], predicted[valid])
    cd = compute_cd(gt, predicted, valid, matrix, args.intrinsics_format,
                    args.intrinsics_size, args.cd_max_points, workers=args.cd_workers)
    return np.append(seven, cd)


def blend_prediction(models, rgb, mode):
    import torch
    output = models["depth"](models["encoder"](rgb))
    backbone = output[("disp", 0)]
    decoded = output[("F_dec", 0)]
    _, frequency, edge = models["edge_head"](decoded.detach(), rgb_input=rgb)
    fusion = models["depth_head"](decoded, edge.detach(), frequency)
    auxiliary = fusion[("disp", 0)]
    keep_backbone = (1.0 - fusion["W_soft_mask"]).clamp(0.0, 1.0)
    mean_rgb = rgb.mean(dim=1, keepdim=True)
    keep_backbone = torch.maximum(keep_backbone, ((mean_rgb < 0.05) | (mean_rgb > 0.95)).float())
    return backbone * keep_backbone + auxiliary * (1.0 - keep_backbone)


class Predictor:
    def __init__(self, args):
        import random
        import torch
        import networks
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for CPU inference")
        self.args = args
        self.device = torch.device(args.device)
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        config_path = args.weights / "model_config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            spec = config["job"]["spec"]
            if spec.get("routing") != "rgb" or not spec.get("feature_fusion"):
                raise ValueError("Checkpoint is not the complete no-dense SEMF model")
            for key, value in (("sparsity", args.target_sparsity), ("gamma", args.gamma)):
                if spec[key] != value:
                    raise ValueError(f"Checkpoint {key}={spec[key]} differs from the requested setting")
            if config["height"] != args.height or config["width"] != args.width:
                raise ValueError("Checkpoint input resolution differs from the requested setting")
        state_path = args.weights / "training_state.pth"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            expected = {"O_NO_DENSE": "full"}
            variant = state.get("variant")
            if variant is not None and expected.get(variant) != args.model:
                raise ValueError(f"Checkpoint variant {variant} is not the complete SEMF model")
            spec = state.get("job", {}).get("spec", {})
            if spec and (not spec.get("enabled") or not spec.get("feature_fusion")
                         or spec.get("routing") != "rgb"
                         or not all(spec.get(k) for k in ("dilation", "soft_weight", "brightness_protection"))
                         or spec.get("threshold") != "quantile"):
                raise ValueError("This entry point evaluates the complete no-dense SEMF model")
            for key, value in (("sparsity", args.target_sparsity), ("gamma", args.gamma)):
                if key in spec and spec[key] != value:
                    raise ValueError(f"Checkpoint {key}={spec[key]} differs from the requested setting")
            options = state.get("options", {})
            for name in ("height", "width", "min_depth", "max_depth", "target_sparsity"):
                if name in options and options[name] != getattr(args, name):
                    raise ValueError(f"Checkpoint {name}={options[name]} differs from evaluation option")
        channels = [64, 128, 216, 288, 288]
        self.models = {"encoder": networks.mpvit_small(pretrained=None),
                       "depth": networks.HRDepthDecoder(num_ch_enc=channels, ch_enc=channels,
                                                         scales=range(4))}
        self.models["edge_head"] = networks.FFTEdgeHead(16, mid_channels=32)
        self.models["depth_head"] = networks.DepthHeadWithFusion(
            16, 32, scales=range(4), target_sparsity=args.target_sparsity, gamma=args.gamma)
        for name, model in self.models.items():
            path = args.weights / f"{name}.pth"
            state = torch.load(path, map_location="cpu", weights_only=False)
            if "state_dict" in state:
                state = state["state_dict"]
            if name == "encoder":
                for key in ("height", "width"):
                    if key in state and int(state[key]) != getattr(args, key):
                        raise ValueError(f"Encoder checkpoint {key} differs from input size")
                state = {k: v for k, v in state.items() if k not in {"height", "width", "use_stereo"}}
            state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            # This decoder's eval() returns None, so never chain .eval().to(...).
            model.eval()
            model.to(self.device)
            print(f"Loaded {name}: {path}", flush=True)

    def __call__(self, samples):
        import torch
        from PIL import Image
        arrays = []
        for sample in samples:
            with Image.open(sample.image) as image:
                image = image.convert("RGB").resize((self.args.width, self.args.height),
                                                     Image.Resampling.LANCZOS)
                arrays.append(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy())
        rgb = torch.from_numpy(np.stack(arrays)).to(self.device).float().div_(255.0)
        with torch.inference_mode():
            normalized = blend_prediction(self.models, rgb, self.args.model)
            scaled_inverse = 1.0/self.args.max_depth + (
                1.0/self.args.min_depth - 1.0/self.args.max_depth) * normalized
        return scaled_inverse[:, 0].cpu().numpy()


def summarize(records):
    groups = defaultdict(list)
    for sample, values in records:
        groups[sample.scene].append(values)
    means = {scene: np.mean(values, axis=0) for scene, values in groups.items()}
    return means, np.mean(list(means.values()), axis=0), np.mean([v for _, v in records], axis=0)


def write_report(path, records, labels, protocol):
    means, scene_mean, image_mean = summarize(records)
    counts = Counter(s.scene for s, _ in records)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n\n")
        handle.write("Scene | Images | " + " | ".join(labels) + "\n")
        for scene in sorted(means, key=scene_key):
            handle.write(f"{scene} | {counts[scene]} | " + " | ".join(f"{v:.4f}" for v in means[scene]) + "\n")
        handle.write("\nSCENE_MEAN | " + " | ".join(f"{v:.4f}" for v in scene_mean) + "\n")
        handle.write("IMAGE_MEAN | " + " | ".join(f"{v:.4f}" for v in image_mean) + "\n")
        handle.write("\nSCENE_MEAN averages the unrounded per-scene means with equal scene weights.\n")
    print(f"Saved: {path}", flush=True)


def run_evaluation(args):
    if args.scenes is not None and len(args.scenes) != len(set(args.scenes)):
        raise ValueError("Duplicate scene selection")
    if args.test_root is None:
        if args.test_files is None:
            args.test_files = args.data_root / "splits/eigen/test_pairs.txt"
            if not args.test_files.is_file():
                args.test_files = ROOT / "splits/eigen/test_pairs.txt"
        args.test_root = args.data_root / "test"
    if args.gt_root is None:
        args.gt_root = args.data_root / "splits/eigen/gt_npy"
    if args.height % 32 or args.width % 32 or min(args.height, args.width) <= 0:
        raise ValueError("Input height/width must be positive multiples of 32")
    if args.batch_size <= 0 or not 0 < args.target_sparsity < 1 or args.gamma <= 0:
        raise ValueError("Invalid batch size or routing support")
    if not 0 < args.min_depth < args.max_depth or not 0 < args.eval_min_depth < args.eval_max_depth:
        raise ValueError("Invalid depth ranges")
    if args.cd_max_points < 0 or args.cd_workers < 1:
        raise ValueError("Invalid CD point limit or worker count")
    matrix = load_intrinsics(args.intrinsics, args.intrinsics_format, args.intrinsics_size)
    samples = collect_samples(args)
    required = ["encoder", "depth", "edge_head", "depth_head"]
    for name in required:
        if not (args.weights / f"{name}.pth").is_file():
            raise FileNotFoundError(args.weights / f"{name}.pth")
    print(f"Cohort: {len(samples)} images; {dict(Counter(s.scene for s in samples))}", flush=True)
    if args.dry_run:
        for sample in samples:
            gt = load_ground_truth(sample.ground_truth, args.gt_unit)
            align_depth(gt, np.ones(gt.shape), args)
        print("Image/GT pairing, valid support, and camera intrinsics verified; no model or metric evaluation was run.")
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --dry-run is set")
    sources = sorted({s.source for s in samples})
    paths = [args.output_dir / f"{source}_results.txt" for source in sources]
    paths.append(args.output_dir / f"selected{len(set(s.scene for s in samples))}_mean.txt")
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError("Result file already exists; select a new output directory or pass --overwrite")
    protocol = {"checkpoint": str(args.weights.resolve()), "model": args.model,
                "input_hw": [args.height, args.width], "network_depth_range": [args.min_depth, args.max_depth],
                "evaluation_depth_range": [args.eval_min_depth, args.eval_max_depth],
                "gt_unit": args.gt_unit, "crop": args.crop,
                "median_scaling": not args.disable_median_scaling,
                "test_gt_pairing": args.test_gt_pairing, "gt_root": str(args.gt_root.resolve()),
                "test_manifest": str(args.test_files) if args.test_files is not None else None,
                "test_root": str(args.test_root.resolve()),
                "extra_test_root": str(args.extra_test_root.resolve()) if args.extra_test_root else None,
                "scenes": sorted(set(s.scene for s in samples), key=scene_key),
                "image_count": len(samples), "target_sparsity": args.target_sparsity,
                "gamma": args.gamma, "routing": "rgb",
                "seed": 1234, "metrics": list(METRICS),
                "intrinsics_file": str(args.intrinsics.resolve()), "intrinsics": matrix.tolist(),
                "intrinsics_format": args.intrinsics_format, "intrinsics_size": args.intrinsics_size,
                "intrinsics_coordinates": "full image; crop applied as validity mask only",
                "cd_definition": "mean squared NN(GT,pred) + mean squared NN(pred,GT)",
                "cd_unit": "m^2", "cd_max_points": args.cd_max_points,
                "cd_sampling_seeds": [1234, 1235], "cd_workers": args.cd_workers}
    predictor = Predictor(args)
    records = []
    start = time.monotonic()
    for offset in range(0, len(samples), args.batch_size):
        batch = samples[offset:offset+args.batch_size]
        predictions = predictor(batch)
        for sample, inverse in zip(batch, predictions):
            gt = load_ground_truth(sample.ground_truth, args.gt_unit)
            predicted, valid, _ = align_depth(gt, inverse, args)
            values = compute_metrics(gt, predicted, valid, matrix, args)
            if values.shape != (len(METRICS),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid metrics for {sample.image}: {values}")
            records.append((sample, values))
        completed = len(records)
        elapsed = time.monotonic() - start
        remaining = elapsed / completed * (len(samples) - completed)
        print(f"{completed}/{len(samples)} images; scene={batch[-1].scene}; "
              f"elapsed={elapsed:.0f}s; ETA={remaining:.0f}s", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source, path in zip(sources, paths):
        write_report(path, [(s, v) for s, v in records if s.source == source], METRICS, protocol)
    write_report(paths[-1], records, METRICS, protocol)


def main():
    args = make_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
