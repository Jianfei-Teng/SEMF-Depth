"""Save SEMF depth, E_rgb, W_soft and projected-feature responses as PNGs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image

from evaluate_metrics import (IMAGE_EXTENSIONS, Predictor, align_depth,
                              collect_samples, load_ground_truth)


@dataclass(frozen=True)
class ImageItem:
    image: Path
    destination: Path
    ground_truth: Path | None = None


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Full SEMF checkpoint directory")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, nargs="+",
                        help="One or more images or directories, searched recursively")
    inputs.add_argument("--test-files", type=Path,
                        help="Evaluation manifest for exact RGB/GT pairing")
    parser.add_argument("--data-root", type=Path, default=Path("./ancient_architecture_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-gt", action="store_true",
                        help="Save matched GT and median-align predictions; requires --test-files")
    parser.add_argument("--gt-unit", choices=("auto", "m", "mm"), default="auto")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--target-sparsity", type=float, default=0.08)
    parser.add_argument("--gamma", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check inputs and destinations without loading the model")
    parser.set_defaults(model="full", scenes=None, crop="garg",
                        eval_min_depth=1e-3, eval_max_depth=80.0,
                        disable_median_scaling=False)
    return parser


def collect_images(args):
    if args.test_files is not None:
        return [ImageItem(sample.image, Path(sample.scene) / sample.image.name,
                          sample.ground_truth)
                for sample in collect_samples(args)]
    items = []
    for source in args.input:
        source = source.resolve(strict=True)
        if source.is_file():
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Expected JPG, JPEG or PNG: {source}")
            items.append(ImageItem(source, Path(source.parent.name) / source.name))
            continue
        # Keep output directories outside inputs so reruns cannot ingest generated PNGs.
        if args.output_dir.resolve().is_relative_to(source):
            raise ValueError("--output-dir must be outside every input directory")
        images = sorted(p for p in source.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No JPG, JPEG or PNG images in {source}")
        items.extend(ImageItem(image, Path(source.name) / image.relative_to(source))
                     for image in images)
    return items


def destinations(item, args):
    folder = args.output_dir / item.destination.parent
    filename = item.destination.stem + ".png"
    paths = {name: folder / name / filename
             for name in ("depth", "E_rgb", "W_soft", "F_edge")}
    if args.save_gt:
        paths["GT_depth"] = folder / "GT_depth" / filename
    return paths


def preflight(args, items):
    if min(args.height, args.width) <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("Input height/width must be positive multiples of 32")
    if args.batch_size < 1 or not 0 < args.target_sparsity < 1 or args.gamma <= 0:
        raise ValueError("Invalid batch size or routing support")
    if not 0 < args.min_depth < args.max_depth:
        raise ValueError("Invalid network depth range")
    if args.save_gt and args.test_files is None:
        raise ValueError("--save-gt requires --test-files for exact image/GT pairing")
    for name in ("encoder", "depth", "edge_head", "depth_head"):
        if not (args.weights / f"{name}.pth").is_file():
            raise FileNotFoundError(args.weights / f"{name}.pth")
    if not items:
        raise ValueError("No images selected")
    if len({item.image.resolve() for item in items}) != len(items):
        raise ValueError("An input image was selected more than once")
    outputs = [path.resolve() for item in items for path in destinations(item, args).values()]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Output names collide; process identically named inputs separately")
    for source in args.input or []:
        if source.is_dir() and any(path.is_relative_to(source.resolve()) for path in outputs):
            raise ValueError("Generated PNGs must be outside every input directory")
    inputs = {item.image.resolve() for item in items}
    if inputs.intersection(outputs):
        raise ValueError("An output path would overwrite an input image")
    if not args.overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(f"Output exists: {path}; use a new directory or --overwrite")
    for item in items:
        with Image.open(item.image) as image:
            image.verify()
        if args.save_gt:
            gt = load_ground_truth(item.ground_truth, args.gt_unit)
            align_depth(gt, np.ones(gt.shape, dtype=np.float32), args)


def depth_colors(depth, limits=None, valid=None):
    from matplotlib import colormaps
    depth = np.asarray(depth)
    valid = (np.isfinite(depth) & (depth > 0)) if valid is None else valid
    if limits is None:
        limits = np.percentile(depth[valid], [2, 98])
    low, high = limits
    normalized = np.zeros(depth.shape, dtype=np.float32)
    if high > low:
        normalized[valid] = np.clip((depth[valid] - low) / (high - low), 0, 1)
    rgb = colormaps["magma_r"](normalized, bytes=True)[..., :3].copy()
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def render_images(item, inverse, responses, args):
    if not np.isfinite(inverse).all() or np.any(inverse <= 0):
        raise ValueError(f"Invalid predicted inverse depth: {item.image}")
    for name, response in responses.items():
        if response.shape != inverse.shape or not np.isfinite(response).all():
            raise ValueError(f"Invalid {name} response: {item.image}")
    with Image.open(item.image) as original:
        size = original.size
    if args.save_gt:
        gt = load_ground_truth(item.ground_truth, args.gt_unit)
        depth, valid, _ = align_depth(gt, inverse, args)
        limits = np.percentile(gt[valid], [2, 98])
        images = {"depth": depth_colors(depth, limits),
                  "GT_depth": depth_colors(gt, limits)}
    else:
        inverse = np.asarray(Image.fromarray(inverse).resize(size, Image.Resampling.BILINEAR))
        images = {"depth": depth_colors(1.0 / inverse)}
    for name, response in responses.items():
        images[name] = Image.fromarray(
            np.rint(np.clip(response, 0, 1) * 255).astype(np.uint8))
    return {key: image.resize(size, Image.Resampling.BILINEAR)
            for key, image in images.items()}


def run_visualization(args):
    items = collect_images(args)
    preflight(args, items)
    print(f"Inputs: {len(items)} images; output: {args.output_dir}", flush=True)
    if args.dry_run:
        print("Image inputs, checkpoint filenames and output paths checked; no inference run.")
        return
    predictor = Predictor(args)
    captured = {name: [] for name in ("E_rgb", "W_soft", "F_edge")}

    def capture_see(module, inputs, output):
        # Channel RMS is a visualization of features, not a boundary probability.
        feature = output[2].detach().square().mean(1).sqrt()
        feature = (feature - feature.amin((-2, -1), keepdim=True))
        feature = feature / feature.amax((-2, -1), keepdim=True).clamp_min(1e-6)
        captured["F_edge"].append(feature.cpu().numpy())
        captured["E_rgb"].append(output[1][:, 0].detach().cpu().numpy())

    def capture_smf(module, inputs, output):
        captured["W_soft"].append(output["W_soft_mask"][:, 0].detach().cpu().numpy())

    hooks = [predictor.models["edge_head"].register_forward_hook(capture_see),
             predictor.models["depth_head"].register_forward_hook(capture_smf)]
    start = time.monotonic()
    try:
        for offset in range(0, len(items), args.batch_size):
            batch = items[offset:offset + args.batch_size]
            for values in captured.values():
                values.clear()
            predictions = predictor(batch)
            if any(len(values) != 1 or len(values[0]) != len(batch)
                   for values in captured.values()):
                raise RuntimeError("Expected one SEE/SMF response per input image")
            for index, (item, inverse) in enumerate(zip(batch, predictions)):
                responses = {name: values[0][index] for name, values in captured.items()}
                images = render_images(item, inverse, responses, args)
                paths = destinations(item, args)
                for key, image in images.items():
                    paths[key].parent.mkdir(parents=True, exist_ok=True)
                    image.save(paths[key], format="PNG")
            completed = offset + len(batch)
            elapsed = time.monotonic() - start
            eta = elapsed / completed * (len(items) - completed)
            print(f"{completed}/{len(items)} images; elapsed={elapsed:.0f}s; ETA={eta:.0f}s", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    print(f"Saved individual PNG files to {args.output_dir}", flush=True)


def main():
    run_visualization(make_parser().parse_args())


if __name__ == "__main__":
    main()
