"""Focused implementation checks that do not require a GPU or DeepWeeds images."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import eval
from starter import inference
from starter.benchmark import bench
from starter.build_results import collect
from starter.train import Config, parse_overrides


class TestNoTorch(unittest.TestCase):
    def test_temperature_preserves_argmax(self):
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(30, 9))
        probs = inference.apply_temperature(logits, 1.7)
        np.testing.assert_array_equal(probs.argmax(1), logits.argmax(1))
        np.testing.assert_allclose(probs.sum(1), 1)

    def test_aggregation_normalized(self):
        logits = np.arange(36, dtype=float).reshape(2, 2, 9)
        for mode in ("prob", "logit"):
            probs = inference.aggregate_views(logits, mode)
            np.testing.assert_allclose(probs.sum(1), 1)

    def test_cli_types(self):
        result = parse_overrides(["epochs=3", "amp=false", "ema_decay=0.999", "mix=none"])
        self.assertEqual(result, {"epochs": 3, "amp": False, "ema_decay": 0.999, "mix": None})
        self.assertIsInstance(Config(**result), Config)

    def test_benchmark_contract(self):
        result = bench(lambda: None, warmup=10, iters=50)
        self.assertEqual(result["n"], 50)
        self.assertLessEqual(result["p50"], result["p95"])
        self.assertLessEqual(result["p95"], result["p99"])

    def test_results_are_derived_from_saved_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs/B04/seed0"
            run_dir.mkdir(parents=True)
            cfg = Config(exp_id="B04", backbone="vit_small_patch16_224")
            (run_dir / "config.json").write_text(json.dumps(cfg.__dict__))
            pd.DataFrame([{"epoch": 1, "train_loss": 1.0, "val_loss": 0.8,
                           "val_macro_f1": 0.9, "epoch_seconds": 2.0}]).to_csv(
                               run_dir / "history.csv", index=False)
            probs = np.eye(9)[np.arange(9)]
            eval.save_predictions(root / "predictions/B04_seed0_val.csv",
                                  [f"{i}.jpg" for i in range(9)], np.arange(9), probs)
            tables = collect(root)
            self.assertEqual(len(tables["Backbones"]), 1)
            self.assertEqual(tables["Backbones"][0]["macro_f1_val"], 1.0)
            self.assertTrue((root / "curves/B04_seed0_vit_small_patch16_224.png").exists())


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch cần chạy trên Kaggle")
class TestTorchPipeline(unittest.TestCase):
    def test_focal_gamma_zero_matches_ce(self):
        import torch
        import torch.nn.functional as F
        from starter.losses import FocalLoss
        logits = torch.randn(10, 9)
        labels = torch.arange(10) % 9
        self.assertTrue(torch.allclose(FocalLoss(gamma=0)(logits, labels),
                                       F.cross_entropy(logits, labels), atol=1e-6))

    def test_cutmix_lambda_matches_changed_area(self):
        import torch
        from starter.losses import mix_batch
        torch.manual_seed(4)
        np.random.seed(4)
        x = torch.arange(8, dtype=torch.float32)[:, None, None, None].expand(8, 3, 32, 32).clone()
        y = torch.arange(8)
        mixed, (_, other, lam) = mix_batch(x, y, alpha=1, mode="cutmix")
        different = other != y
        self.assertTrue(different.any())
        changed = (mixed[different, 0] != x[different, 0]).float().mean().item()
        self.assertAlmostEqual(changed, 1 - lam, places=6)

    def test_head_bias_has_no_weight_decay(self):
        import torch
        from starter.model import param_groups
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(4, 4)
                self.head = torch.nn.Linear(4, 9)
            def get_classifier(self):
                return self.head
        model = Tiny()
        groups = param_groups(model, 1e-4, 1e-3, 0.05)
        by_id = {id(param): group for group in groups for param in group["params"]}
        self.assertEqual(by_id[id(model.head.bias)]["weight_decay"], 0)
        self.assertEqual(by_id[id(model.backbone.bias)]["weight_decay"], 0)
        self.assertEqual(by_id[id(model.head.weight)]["lr"], 1e-3)


if __name__ == "__main__":
    unittest.main()
