"""Two-phase SEMF dispatch and optimizer recipes, independent of Trainer imports."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

from data_paths import training_splits

BACKBONE_MODELS = ("encoder", "depth", "pose_encoder", "pose")
FULL_MODELS = ("encoder", "depth", "edge_head", "depth_head", "pose_encoder", "pose")
REFERENCE_COMMIT = "692a823c90bdab9005c715bb349ad80acab46ca4"


def code_version():
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                            cwd=root, text=True).strip())
        return {"commit": commit, "tracked_changes": dirty, "reference_commit": REFERENCE_COMMIT}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_changes": None, "reference_commit": REFERENCE_COMMIT}


def locate_backbone(path, generated):
    """Absence allows pretraining; a partial/misnamed checkpoint is an error."""
    for folder in dict.fromkeys((Path(path) / "weights_49", Path(path), Path(generated))):
        present = [name for name in BACKBONE_MODELS if (folder / f"{name}.pth").is_file()]
        if not present:
            continue
        if len(present) != len(BACKBONE_MODELS):
            raise ValueError(f"Incomplete backbone at {folder}: found {present}")
        if folder.name.startswith("weights_") and folder.name != "weights_49":
            raise ValueError(f"Expected weights_49 (50 completed epochs), got {folder}")
        if any((folder / f"{name}.pth").is_file() for name in ("edge_head", "depth_head")):
            raise ValueError(f"Expected the backbone initialization, not a full/refined checkpoint: {folder}")
        marker = folder / "COMPLETE.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8"))["epoch"] != 49:
            raise ValueError(f"Backbone metadata does not identify epoch 49: {folder}")
        return folder
    return None


def resume_checkpoint(options):
    if options.resume != "auto":
        path = Path(options.resume)
    else:
        root = Path(options.log_dir) / options.model_name
        candidates = []
        for phase in ("backbone", "refinement"):
            for folder in (root / phase / "models").glob("weights_*"):
                if folder.name[8:].isdigit():
                    candidates.append((int(folder.name[8:]), folder))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint to resume under {root}; omit --resume for a new run")
        path = max(candidates, key=lambda item: item[0])[1]
    marker = path / "COMPLETE.json"
    if not marker.is_file():
        raise ValueError(f"Checkpoint has no completion marker; refusing a partial resume: {path}")
    state = json.loads(marker.read_text(encoding="utf-8"))
    if state.get("phase") not in ("backbone", "refinement"):
        raise ValueError(f"Unknown checkpoint phase: {path}")
    epoch = state.get("epoch")
    low, high = (0, 49) if state["phase"] == "backbone" else (50, 69)
    if not isinstance(epoch, int) or not low <= epoch <= high:
        raise ValueError(f"Invalid epoch/phase in checkpoint: {path}")
    required = BACKBONE_MODELS if state["phase"] == "backbone" else FULL_MODELS
    for name in (*required, "adam"):
        if not (path / f"{name}.pth").is_file():
            raise FileNotFoundError(path / f"{name}.pth")
    return path, state


def phase_options(options, phase, checkpoint=None, resume_state=None):
    opt = deepcopy(options)
    opt.phase = phase
    opt.model_name = str(Path(options.model_name) / phase)
    opt.disable_multihead = phase == "backbone"
    opt.disable_routing = False
    opt.start_epoch = 0 if phase == "backbone" else 50
    opt.num_epochs = 50 if phase == "backbone" else 70
    opt.phase1_epochs = 0
    opt.warmup_epochs = 0
    opt.load_weights_folder = str(checkpoint) if checkpoint is not None else None
    opt.reset_optimizer = resume_state is None
    opt.models_to_load = list((BACKBONE_MODELS if phase == "backbone" else FULL_MODELS)
                             if resume_state else BACKBONE_MODELS)
    opt._resume_state = resume_state
    if resume_state:
        opt.start_epoch = int(resume_state["epoch"]) + 1
    return opt


def build_plan(options):
    if (options.batch_size <= 0 or options.accumulation_steps <= 0 or options.num_workers < 0
            or options.log_frequency < 1):
        raise ValueError("Invalid batch/accumulation/workers/logging settings")
    if (min(options.height, options.width) <= 0 or options.height % 32 or options.width % 32
            or not 0 < options.target_sparsity < 1 or not 0 < options.min_depth < options.max_depth):
        raise ValueError("Invalid image/depth/routing settings")
    if (min(options.learning_rate, options.head_learning_rate, options.backbone_learning_rate,
            options.decoder_learning_rate) <= 0 or not 0 < options.backbone_lr_decay <= 1):
        raise ValueError("Invalid learning-rate settings")
    train, val, stats = training_splits(options)
    microbatches = len(train) // options.batch_size
    if microbatches < 1 or len(val) // options.batch_size < 1:
        raise ValueError("Training and validation must each contain at least one full batch")
    generated = Path(options.log_dir) / options.model_name / "backbone/models/weights_49"
    if options.resume:
        checkpoint, state = resume_checkpoint(options)
        if options.stage != "auto" and options.stage != ("refine" if state["phase"] == "refinement" else "backbone"):
            raise ValueError("--stage disagrees with the resume checkpoint phase")
        phases = [phase_options(options, state["phase"], checkpoint, state)]
        if state["phase"] == "backbone" and options.stage == "auto":
            phases.append(phase_options(options, "refinement", generated))
    else:
        checkpoint = locate_backbone(options.backbone_weights, generated) if options.stage != "backbone" else None
        if options.stage == "refine" and checkpoint is None:
            raise FileNotFoundError(f"50-epoch backbone not found: {options.backbone_weights}")
        if options.stage == "backbone":
            phases = [phase_options(options, "backbone")]
        elif checkpoint is None:
            phases = [phase_options(options, "backbone"), phase_options(options, "refinement", generated)]
        else:
            phases = [phase_options(options, "refinement", checkpoint)]
    updates = math.ceil(microbatches / options.accumulation_steps)
    for opt in phases:
        output = Path(opt.log_dir) / opt.model_name / "models"
        if list(output.glob(".weights_*.saving")):
            raise FileExistsError(f"An incomplete checkpoint save remains in {output}; preserve it outside this run before resuming")
        if opt._resume_state is None and list(output.glob("weights_*")):
            raise FileExistsError(f"Run already contains checkpoints: {output}; use --resume or a new model name")
        opt._train_filenames, opt._val_filenames = train, val
        opt._split_stats = stats
        opt.updates_per_epoch = updates
        opt.microbatches_per_epoch = microbatches
        opt.planned_phase_updates = updates * max(0, opt.num_epochs - opt.start_epoch)
        opt.code_version = code_version()
        opt.command = [sys.executable, *sys.argv]
    return phases


def optimizer_groups(models, options):
    if options.phase == "backbone":
        encoder = list(models["encoder"].parameters())
        others = [p for name, model in models.items() if name != "encoder" for p in model.parameters()]
        return [{"params": others, "lr": options.decoder_learning_rate, "weight_decay": 0.01,
                 "group_name": "decoder_and_pose"},
                {"params": encoder, "lr": options.backbone_learning_rate, "weight_decay": 0.01,
                 "group_name": "depth_encoder"}]
    backbone_names = {"encoder", "depth", "pose_encoder"}
    backbone = [p for name, model in models.items() if name in backbone_names for p in model.parameters()]
    heads = [p for name, model in models.items() if name not in backbone_names for p in model.parameters()]
    groups = []
    for label, params, lr in (("heads", heads, options.head_learning_rate),
                              ("backbone", backbone, options.learning_rate)):
        for decay in (True, False):
            groups.append({"params": [p for p in params if (p.ndim != 1) == decay],
                           "lr": lr, "weight_decay": 0.01 if decay else 0.0,
                           "group_name": label + ("_decay" if decay else "_no_decay")})
    return groups


def describe_plan(phases):
    rows = []
    for opt in phases:
        rows.append({"phase": opt.phase, "epochs": [opt.start_epoch, opt.num_epochs - 1],
                     "initialization": opt.load_weights_folder, "resume": opt._resume_state is not None,
                     "batch_size": opt.batch_size, "accumulation_steps": opt.accumulation_steps,
                     "seed": opt.seed, "data": opt._split_stats,
                     "microbatches_per_epoch": opt.microbatches_per_epoch,
                     "updates_per_epoch": opt.updates_per_epoch,
                     "planned_updates_this_invocation": opt.planned_phase_updates,
                     "initial_group_lrs": ([opt.decoder_learning_rate, opt.backbone_learning_rate]
                                           if opt.phase == "backbone" else
                                           [opt.head_learning_rate] * 2 + [opt.learning_rate] * 2),
                     "schedule": ({"type": "ExponentialLR", "gamma": opt.backbone_lr_decay}
                                  if opt.phase == "backbone" else
                                  {"type": "CosineAnnealingLR", "T_max": 65, "eta_min": 1e-7,
                                   "steps": 20, "warmup_steps": 0}),
                     "output": str(Path(opt.log_dir) / opt.model_name), "code": opt.code_version})
    return rows
