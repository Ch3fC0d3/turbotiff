import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.synthetic_log_generator import SyntheticLogConfig, generate_sample, write_dataset
from curve_model.integration import build_phase1_probability
from curve_model.phase2_decode import decode_phase2_path
from curve_model.phase2_infer import predict_phase2_geometry, validate_phase2_checkpoint
from curve_model.phase2_integration import build_phase2_probability
from curve_model.phase2_score import Phase2ScoreConfig, build_phase2_trace_score, phase2_confidence
from curve_model.metrics import calculate_phase2_metrics


class Phase2SyntheticLabelTests(unittest.TestCase):
    def test_geometry_and_grid_labels_are_finite_and_aligned(self):
        sample = generate_sample(7001, SyntheticLogConfig(width=72, height=96, maximum_distance=10))
        self.assertEqual(sample["distance_field"].shape, (96, 72))
        self.assertEqual(sample["direction_field"].shape, (2, 96, 72))
        self.assertEqual(sample["grid_mask"].shape, (96, 72))
        self.assertEqual(sample["valid_direction_mask"].shape, (96, 72))
        self.assertTrue(np.isfinite(sample["distance_field"]).all())
        self.assertGreaterEqual(float(sample["distance_field"].min()), 0.0)
        self.assertLessEqual(float(sample["distance_field"].max()), 1.0)
        valid = sample["valid_direction_mask"] > 0
        norms = np.linalg.norm(sample["direction_field"], axis=0)
        np.testing.assert_allclose(norms[valid], 1.0, atol=1e-5)
        self.assertTrue(np.any((sample["grid_mask"] > 0) & (sample["stroke_mask"] > 0)))


class Phase2ModelLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except Exception as exc:
            raise unittest.SkipTest(str(exc))
        cls.torch = torch

    def _targets(self, batch=2, height=32, width=24):
        torch = self.torch
        center = torch.zeros(batch, 1, height, width)
        center[:, :, :, width // 2] = 1.0
        valid = torch.zeros_like(center)
        valid[:, :, :, width // 2 - 2:width // 2 + 3] = 1.0
        direction = torch.zeros(batch, 2, height, width)
        direction[:, 1] = valid[:, 0]
        return {
            "stroke_mask": center.clone(), "centerline_mask": center,
            "distance_field": valid.clone(), "direction_field": direction,
            "grid_mask": torch.zeros_like(center), "valid_direction_mask": valid,
            "stroke_label_valid": torch.ones(batch, 1, 1, 1),
            "grid_label_valid": torch.ones(batch, 1, 1, 1),
        }

    def test_all_output_shapes_and_direction_normalization(self):
        from curve_model.phase2_model import CurvePhase2UNet
        model = CurvePhase2UNet(base_channels=4).eval()
        with self.torch.no_grad():
            outputs = model(self.torch.rand(2, 3, 32, 24))
        expected = {
            "stroke_logits": (2, 1, 32, 24), "centerline_logits": (2, 1, 32, 24),
            "distance_field": (2, 1, 32, 24), "direction": (2, 2, 32, 24),
            "grid_logits": (2, 1, 32, 24),
        }
        self.assertEqual({key: tuple(value.shape) for key, value in outputs.items()}, expected)
        self.assertTrue(self.torch.isfinite(outputs["distance_field"]).all())
        self.assertTrue(((outputs["distance_field"] >= 0) & (outputs["distance_field"] <= 1)).all())
        norms = self.torch.linalg.vector_norm(outputs["direction"], dim=1)
        self.assertTrue(self.torch.allclose(norms, self.torch.ones_like(norms), atol=1e-5))

    def test_direction_loss_ignores_invalid_background_and_grid_overlap_is_finite(self):
        from curve_model.phase2_losses import CurvePhase2Loss, masked_direction_loss
        targets = self._targets()
        prediction = targets["direction_field"].clone()
        changed = prediction.clone()
        invalid = targets["valid_direction_mask"].expand_as(changed) == 0
        changed[invalid] = 100.0
        first = masked_direction_loss(prediction, targets["direction_field"], targets["valid_direction_mask"], targets["centerline_mask"])
        second = masked_direction_loss(changed, targets["direction_field"], targets["valid_direction_mask"], targets["centerline_mask"])
        self.assertAlmostEqual(float(first), float(second), places=6)
        targets["grid_mask"] = targets["centerline_mask"].clone()
        outputs = {
            "stroke_logits": self.torch.zeros_like(targets["stroke_mask"]),
            "centerline_logits": self.torch.zeros_like(targets["centerline_mask"]),
            "distance_field": self.torch.zeros_like(targets["distance_field"]),
            "direction": self.torch.zeros_like(targets["direction_field"]),
            "grid_logits": self.torch.zeros_like(targets["grid_mask"]),
        }
        losses = CurvePhase2Loss()(outputs, targets)
        self.assertTrue(self.torch.isfinite(losses["total"]))
        self.assertGreater(float(losses["grid"]), 0.0)

    def test_cape_activates_only_at_configured_epoch(self):
        from curve_model.phase2_losses import CapeConfig, CurvePhase2Loss
        criterion = CurvePhase2Loss(cape=CapeConfig(enabled=True, start_epoch=3))
        self.assertFalse(criterion.cape_active(2))
        self.assertTrue(criterion.cape_active(3))

    def test_phase1_checkpoint_transfer_and_phase2_validation(self):
        from curve_model.model import CurvePhase1UNet
        from curve_model.phase2_model import CurvePhase2UNet, transfer_phase1_weights
        phase1 = CurvePhase1UNet(base_channels=4)
        phase2 = CurvePhase2UNet(base_channels=4)
        checkpoint = {"state_dict": phase1.state_dict(), "model_version": phase1.model_version}
        report = transfer_phase1_weights(phase2, checkpoint)
        self.assertGreater(report["loaded_tensor_count"], 10)
        self.assertTrue(any(key.startswith("distance_head") for key in report["missing_keys"]))
        with self.assertRaises(ValueError):
            validate_phase2_checkpoint(checkpoint)
        valid = {
            "state_dict": phase2.state_dict(), "model_format_version": 2, "phase": 2,
            "outputs": list(phase2.outputs), "model_config": phase2.configuration(),
        }
        validate_phase2_checkpoint(valid)


class Phase2ScoreDecodeTests(unittest.TestCase):
    def test_grid_is_soft_penalty_and_non_skeleton_curve_remains_possible(self):
        height, width = 20, 30
        center = np.zeros((height, width), np.float32)
        center[:, 14] = 0.9
        stroke = center.copy()
        distance = center.copy()
        grid = np.zeros_like(center)
        grid[:, 14] = 1.0
        direction = np.zeros((2, height, width), np.float32)
        direction[1] = 1.0
        score, _, _ = build_phase2_trace_score(
            stroke, center, distance, direction, grid,
            skeleton_probability=np.zeros_like(center),
        )
        self.assertTrue(np.isfinite(score).all())
        self.assertGreater(float(score[:, 14].min()), 0.4)
        no_grid, _, _ = build_phase2_trace_score(stroke, center, distance, direction, np.zeros_like(grid))
        self.assertLess(float(score[:, 14].mean()), float(no_grid[:, 14].mean()))

    def test_direction_agreement_favors_correct_crossing(self):
        height, width = 30, 40
        score = np.zeros((height, width), np.float32)
        direction = np.zeros((2, height, width), np.float32)
        truth = 7 + np.arange(height) // 3
        for row, x in enumerate(truth):
            score[row, x] = 0.90
            vector = np.array([1.0 / 3.0, 1.0], np.float32)
            vector /= np.linalg.norm(vector)
            direction[:, row, x] = vector
            score[row, 25] = 0.91
            direction[:, row, 25] = np.array([1.0, 0.0])
        path = decode_phase2_path(score, direction, max_step=3, smooth_lambda=0.001, direction_weight=0.35)
        self.assertLess(float(np.mean(np.abs(path - truth))), 1.0)

    def test_thick_curve_and_missing_section_remain_centered_and_connected(self):
        height, width = 50, 48
        truth = 20.0 + 5.0 * np.sin(np.arange(height) / 8.0)
        columns = np.arange(width)[None]
        distance = np.clip(1.0 - np.abs(columns - truth[:, None]) / 7.0, 0.0, 1.0).astype(np.float32)
        center = np.clip(1.0 - np.abs(columns - truth[:, None]) / 2.0, 0.0, 1.0).astype(np.float32)
        stroke = (np.abs(columns - truth[:, None]) <= 4).astype(np.float32)
        center[20:25] = 0.0
        stroke[20:25] = 0.0
        direction = np.zeros((2, height, width), np.float32)
        dx = np.gradient(truth)
        norm = np.sqrt(dx * dx + 1.0)
        direction[0] = (dx / norm)[:, None]
        direction[1] = (1.0 / norm)[:, None]
        score, direction, _ = build_phase2_trace_score(stroke, center, distance, direction, np.zeros_like(center))
        path = decode_phase2_path(score, direction, max_step=4, direction_weight=0.2)
        self.assertLess(float(np.mean(np.abs(path - truth))), 1.2)
        self.assertLess(float(np.max(np.abs(np.diff(path[19:27])))), 3.0)
        confidence, summary = phase2_confidence(path, score, center, distance, direction, np.zeros_like(center))
        self.assertEqual(confidence.shape, (height,))
        self.assertTrue(np.isfinite(confidence).all())
        self.assertGreaterEqual(summary["mean_confidence"], 0.0)
        self.assertLessEqual(summary["mean_confidence"], 1.0)

    def test_connectivity_metric_finds_one_gap_and_recovery(self):
        truth = np.linspace(10, 20, 30, dtype=np.float32)
        predicted = truth.copy()
        predicted[10:14] += 15.0
        metrics = calculate_phase2_metrics(predicted, truth)
        self.assertEqual(metrics["connectivity"]["major_path_gaps"], 1)
        self.assertEqual(metrics["connectivity"]["maximum_gap_length"], 4)
        self.assertEqual(metrics["connectivity"]["recovery_accuracy"], 1.0)

    def test_phase2_failure_records_chain_and_classic_remains_identical(self):
        image = np.zeros((12, 10, 3), np.uint8)
        classic = np.arange(120, dtype=np.uint8).reshape(12, 10)
        phase1_result, _ = build_phase1_probability(image, classic, mode="classic")
        np.testing.assert_array_equal(phase1_result, classic)
        result, metadata, auxiliary = build_phase2_probability(
            image, classic, "neural_phase2", None, phase1_model_path=None
        )
        np.testing.assert_array_equal(result, classic)
        self.assertEqual(metadata["actual_mode"], "classic")
        self.assertTrue(metadata["fallback"])
        self.assertEqual(len(metadata["fallback_chain"]), 2)
        self.assertEqual(auxiliary, {})


class Phase2TrainingInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except Exception as exc:
            raise unittest.SkipTest(str(exc))
        cls.torch = torch

    def test_tiny_phase2_training_checkpoint_and_source_size_inference(self):
        from curve_model.phase2_train import train_phase2
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            write_dataset(data, 4, SyntheticLogConfig(width=48, height=64), seed=8100)
            summary = train_phase2(
                data, root / "model", epochs=1, batch_size=2, target_size=(32, 32),
                base_channels=4, device="cpu", max_batches_per_epoch=1,
            )
            checkpoint = Path(summary["best_checkpoint"])
            self.assertTrue(checkpoint.exists())
            payload = self.torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(payload["model_format_version"], 2)
            image = generate_sample(9001, SyntheticLogConfig(width=45, height=71))["image"]
            prediction = predict_phase2_geometry(image, str(checkpoint), device="cpu", tile_height=32, overlap=8)
            self.assertEqual(prediction["centerline_probability"].shape, (71, 45))
            self.assertEqual(prediction["direction_field"].shape, (2, 71, 45))

    def test_phase2_failure_uses_available_phase1_checkpoint_before_classic(self):
        from curve_model.model import CurvePhase1UNet
        with tempfile.TemporaryDirectory() as temp:
            checkpoint_path = Path(temp) / "phase1.pt"
            model = CurvePhase1UNet(base_channels=4)
            self.torch.save({
                "state_dict": model.state_dict(),
                "model_version": model.model_version,
                "model_config": model.configuration(),
            }, checkpoint_path)
            image = np.zeros((16, 16, 3), np.uint8)
            classic = np.zeros((16, 16), np.uint8)
            probability, metadata, auxiliary = build_phase2_probability(
                image,
                classic,
                "neural_phase2",
                phase2_model_path=None,
                phase1_model_path=str(checkpoint_path),
                device="cpu",
                tile_height=16,
                overlap=4,
            )
            self.assertEqual(probability.shape, classic.shape)
            self.assertEqual(metadata["actual_mode"], "neural_phase1")
            self.assertTrue(metadata["fallback"])
            self.assertEqual(auxiliary, {})


if __name__ == "__main__":
    unittest.main()
