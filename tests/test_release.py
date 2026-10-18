"""CPU checks for dispatch, checkpoint resume, gradients and metric definitions.

Run: python -m unittest discover -s tests -v
The Trainer methods are exercised without importing GPU/vision dependencies.
"""
import ast
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint_compat import optimizer_state_without_unused_gate
from options import MonodepthOptions
from training_plan import build_plan, optimizer_groups
import evaluate_metrics as metrics

ROOT = Path(__file__).resolve().parents[1]


def trainer_method(name):
    tree = ast.parse((ROOT / "trainer.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(torch=torch, np=np, F=F, nn=nn, os=os, Path=Path, time=time,
                     random=random, shutil=shutil, gc=gc, json=json,
                     optimizer_state_without_unused_gate=optimizer_state_without_unused_gate)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "trainer.py", "exec"), namespace)
    return namespace[name]


def load_head(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "networks" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        data = self.root / "data"
        lists = data / "splits/eigen_zhou"
        lists.mkdir(parents=True)
        for split, count in (("train", 26), ("val", 18)):
            folder = data / split / "image_02/data"
            folder.mkdir(parents=True)
            for index in range(count):
                (folder / f"{index:010d}.jpg").touch()
            (lists / f"{split}_files.txt").write_text(
                "\n".join(f"{split} {index} {side}" for side in ("l", "r")
                          for index in range(count)) + "\n", encoding="utf-8")
        self.options = MonodepthOptions().parse([
            "--data_path", str(data), "--backbone_weights", str(self.root / "weights"),
            "--log_dir", str(self.root / "output"), "--num_workers", "0"])

    def tearDown(self):
        self.temporary.cleanup()

    def test_auto_absent_present_and_partial(self):
        phases = build_plan(self.options)
        self.assertEqual([(p.phase, p.start_epoch, p.num_epochs) for p in phases],
                         [("backbone", 0, 50), ("refinement", 50, 70)])
        self.assertEqual(phases[1].microbatches_per_epoch, 3)
        self.assertEqual(phases[1].updates_per_epoch, 2)
        self.assertEqual(phases[1].planned_phase_updates, 40)
        weights = Path(self.options.backbone_weights)
        weights.mkdir()
        (weights / "encoder.pth").touch()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            build_plan(self.options)
        for name in ("depth", "pose_encoder", "pose"):
            (weights / f"{name}.pth").touch()
        phases = build_plan(self.options)
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0].start_epoch, 50)
        self.assertTrue(phases[0].reset_optimizer)
        self.assertEqual(phases[0].models_to_load, ["encoder", "depth", "pose_encoder", "pose"])

    def test_missing_interior_frame_is_not_replaced(self):
        (Path(self.options.data_path) / "train/image_02/data/0000000010.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            build_plan(self.options)

    def test_phase_groups_trainability_and_cosine(self):
        models = {name: nn.Linear(2, 2) for name in
                  ("encoder", "depth", "edge_head", "depth_head", "pose_encoder", "pose")}
        phase = build_plan(self.options)[1]
        groups = optimizer_groups(models, phase)
        self.assertEqual([g["lr"] for g in groups], [2e-4, 2e-4, 2e-6, 2e-6])
        self.assertEqual([g["weight_decay"] for g in groups], [.01, 0, .01, 0])
        trainer_method("set_train")(SimpleNamespace(models=models, opt=phase))
        self.assertFalse(any(p.requires_grad for p in models["encoder"].parameters()))
        self.assertTrue(all(p.requires_grad for p in models["depth"].parameters()))
        optimizer = torch.optim.AdamW(groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=65, eta_min=1e-7)
        for _ in range(20):
            optimizer.step()
            scheduler.step()
        np.testing.assert_allclose(scheduler.get_last_lr(),
                                   [1.5682807143577903e-4]*2 + [1.5896615093945994e-6]*2,
                                   rtol=1e-14)
        phase = build_plan(self.options)[0]
        trainer_method("set_train")(SimpleNamespace(models=models, opt=phase))
        self.assertTrue(all(p.requires_grad for p in models["encoder"].parameters()))

    def test_actual_epoch_checkpoint_and_resume_methods(self):
        opt = build_plan(self.options)[1]
        opt.load_weights_folder = None
        models = {name: nn.Linear(2, 2) for name in
                  ("encoder", "depth", "edge_head", "depth_head", "pose_encoder", "pose")}
        optimizer = torch.optim.AdamW(optimizer_groups(models, opt))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=65, eta_min=1e-7)
        log = Path(opt.log_dir) / opt.model_name
        obj = SimpleNamespace(opt=opt, models=models, model_optimizer=optimizer,
                              model_lr_scheduler=scheduler, optimizer_updates=0,
                              log_path=str(log), epoch=50, step=0, train_loader=[{} for _ in range(3)],
                              parameters_to_train=[p for model in models.values() for p in model.parameters()])
        obj.set_train = lambda: trainer_method("set_train")(obj)
        obj.process_batch = lambda _: ({}, {"loss": models["depth"](torch.ones(2, 2)).square().mean()})
        obj.log_time = obj.log = obj.val = lambda *a, **k: None
        trainer_method("save_opts")(obj)
        trainer_method("run_epoch")(obj)
        self.assertEqual(obj.optimizer_updates, 2)
        trainer_method("save_model")(obj)
        checkpoint = log / "models/weights_50"
        self.options.resume = str(checkpoint)
        resumed = build_plan(self.options)[0]
        self.assertEqual(resumed.start_epoch, 51)
        restored_models = {name: nn.Linear(2, 2) for name in models}
        restored_optimizer = torch.optim.AdamW(optimizer_groups(restored_models, resumed))
        restored = SimpleNamespace(opt=resumed, models=restored_models,
                                   model_optimizer=restored_optimizer,
                                   model_lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                                       restored_optimizer, T_max=65, eta_min=1e-7))
        trainer_method("load_model")(restored)
        self.assertEqual(restored.optimizer_updates, 2)
        self.assertEqual(restored.model_lr_scheduler.state_dict(), scheduler.state_dict())
        self.assertIn("rng", restored.loaded_training_state)
        for name in models:
            for key, value in models[name].state_dict().items():
                torch.testing.assert_close(value, restored_models[name].state_dict()[key])
        # Every epoch checkpoint remains present after later saves.
        for epoch in (51, 52):
            obj.epoch = epoch
            trainer_method("save_model")(obj)
        self.assertTrue((checkpoint / "adam.pth").exists())
        self.assertTrue((log / "models/weights_51/COMPLETE.json").exists())
        self.assertTrue((log / "models/weights_52/adam.pth").exists())
        self.options.resume = None
        with self.assertRaises(FileExistsError):
            build_plan(self.options)

    def test_released_gradient_paths(self):
        edge = load_head("edge_head.py", "edge_under_test").FFTEdgeHead(16)
        fusion = load_head("depth_head.py", "fusion_under_test").DepthHeadWithFusion(16, 32)
        decoded = torch.randn(1, 16, 32, 32, requires_grad=True)
        rgb = torch.rand(1, 3, 32, 32)
        dense, high, feature = edge(decoded.detach(), rgb)
        output = fusion(decoded, feature.detach(), high, dense.detach())
        output[("disp", 0)].mean().backward()
        self.assertTrue(all(p.grad is None for p in edge.parameters()))
        self.assertGreater(decoded.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None for p in fusion.parameters()))
        self.assertFalse(output["W_soft_mask"].requires_grad)
        self.assertFalse(any("gate_conv" in name for name, _ in edge.named_parameters()))

    def test_metrics_and_full_image_intrinsics(self):
        gt = np.array([[2.0]])
        pred = np.array([[3.0]])
        seven = metrics.compute_seven(gt, pred)
        np.testing.assert_allclose(seven, [.5, .5, 1, np.log(1.5), 0, 1, 1])
        cd = metrics.compute_cd(gt, pred, np.ones((1, 1), dtype=bool), np.eye(3),
                                mode="pixel", reference_size=(1, 1), max_points=0)
        self.assertEqual(cd, 2.0)
        k = metrics.pixel_intrinsics(np.array([[1., 0, .5], [0, 1, .5], [0, 0, 1]]), (480, 640))
        np.testing.assert_allclose(k, [[640, 0, 320], [0, 480, 240], [0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
