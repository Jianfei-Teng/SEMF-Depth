"""Portable, exact left-camera paths and fixed-list validation (no ML imports)."""
from pathlib import Path


def image_path(data_root, folder, index, side="l", extension=".jpg"):
    folder = Path(folder.replace("\\", "/"))
    if folder.is_absolute() or ".." in folder.parts or side not in ("l", "2"):
        raise ValueError(f"Expected a relative left-camera sequence: {folder}, {side}")
    base = Path(data_root) / folder
    camera = base if folder.name == "image_02" else base / "image_02"
    filename = f"{index:010d}{extension}"
    candidates = [camera / sub / filename for sub in ("data", "data_jpg", "data_png", "")]
    candidates.append(base / filename)
    found = list(dict.fromkeys(p for p in candidates if p.is_file()))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one exact frame for {folder}/{filename}; found {found}")
    return found[0]


def read_targets(data_root, list_path, offsets=(0, -1, 1), extension=".jpg"):
    """Select left targets; exclude sequence endpoints, never substitute frames."""
    lines = Path(list_path).read_text(encoding="utf-8-sig").splitlines()
    left, right = [], 0
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"Expected 'sequence frame side' in {list_path}: {line}")
        folder, index, side = fields
        index = int(index)
        if side in ("r", "3"):
            right += 1
            continue
        path = image_path(data_root, folder, index, side, extension)
        left.append((folder.replace("\\", "/"), index, path))
    if not left:
        raise ValueError(f"No left-camera targets in {list_path}")
    by_folder = {}
    for folder, index, _ in left:
        by_folder.setdefault(folder, set()).add(index)
    bounds = {folder: (min(ids), max(ids)) for folder, ids in by_folder.items()}
    valid, paths, boundary = [], set(), 0
    for folder, index, path in left:
        first, last = bounds[folder]
        if any(index + offset < first or index + offset > last for offset in offsets):
            boundary += 1
            continue
        # Missing interior neighbors are errors, not endpoint exclusions.
        for offset in offsets:
            image_path(data_root, folder, index + offset, "l", extension)
        if path.resolve() in paths:
            raise ValueError(f"Duplicate target in {list_path}: {path}")
        paths.add(path.resolve())
        valid.append(f"{folder} {index} l")
    return valid, paths, {"listed_lines": len(lines), "right_entries_excluded": right,
                          "boundary_targets_excluded": boundary, "targets": len(valid)}


def training_splits(options):
    directory = Path(options.split_dir) if options.split_dir else Path(options.data_path) / "splits/eigen_zhou"
    extension = ".png" if options.png else ".jpg"
    train, train_paths, train_stats = read_targets(options.data_path, directory / "train_files.txt",
                                                  options.frame_ids, extension)
    val, val_paths, val_stats = read_targets(options.data_path, directory / "val_files.txt",
                                            options.frame_ids, extension)
    if train_paths & val_paths:
        raise ValueError("Train and validation target paths overlap")
    return train, val, {"directory": str(directory), "train": train_stats, "val": val_stats,
                        "offsets": list(options.frame_ids), "camera": "image_02"}
