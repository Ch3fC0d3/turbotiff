import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.synthetic_log_generator import SyntheticLogConfig, generate_sample, write_dataset
from curve_model.integration import build_phase1_probability
from curve_model.metrics import calculate_trace_metrics


class SyntheticGeneratorTests(unittest.TestCase):
    def test_reproducible_images_labels_and_metadata(self):
        config = SyntheticLogConfig(width=64, height=96)
        first = generate_sample(812, config)
        second = generate_sample(812, config)

        for key in ("image", "stroke_mask", "centerline_mask", "centerline_x_by_row"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["metadata"], second["metadata"])

    def test_output_dimensions_and_centerline_inside_complete_stroke(self):
        config = SyntheticLogConfig(
            width=72,
            height=88,
            curve_shape="vertical",
            curve_color="black",
            enable_missing_sections=False,
            enable_dashed_curves=False,
            enable_geometric_distortion=False,
            enable_degradation=False,
            enable_text_fragments=False,
        )
        sample = generate_sample(19, config)

        self.assertEqual(sample["image"].shape, (88, 72, 3))
        self.assertEqual(sample["stroke_mask"].shape, (88, 72))
        self.assertEqual(sample["centerline_mask"].shape, (88, 72))
        self.assertEqual(sample["centerline_x_by_row"].shape, (88,))
        center_columns = np.rint(sample["centerline_x_by_row"]).astype(np.int32)
        rows = np.arange(88)
        self.assertGreater(np.mean(sample["stroke_mask"][rows, center_columns] > 0), 0.97)


class ModelAndLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except Exception as exc:
            raise unittest.SkipTest(f"PyTorch unavailable: {exc}")
        cls.torch = torch

    def test_model_forward_shapes_and_finite_loss(self):
        from curve_model.losses import CurveDetectionLoss
        from curve_model.model import CurvePhase1UNet

        model = CurvePhase1UNet(base_channels=4)
        image = self.torch.rand(2, 3, 32, 40)
        outputs = model(image)
        self.assertEqual(tuple(outputs["stroke_logits"].shape), (2, 1, 32, 40))
        self.assertEqual(tuple(outputs["centerline_logits"].shape), (2, 1, 32, 40))
        targets = {
            "stroke_mask": (self.torch.rand(2, 1, 32, 40) > 0.85).float(),
            "centerline_mask": (self.torch.rand(2, 1, 32, 40) > 0.95).float(),
        }
        losses = CurveDetectionLoss()(outputs, targets)
        self.assertTrue(self.torch.isfinite(losses["total"]).item())

    def test_tiny_dataset_trains_and_saves_checkpoint(self):
        from curve_model.train import train_phase1

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            output_dir = root / "model"
            config = SyntheticLogConfig(width=32, height=48, enable_degradation=False)
            write_dataset(data_dir, 4, config, seed=33)
            summary = train_phase1(
                data_dir,
                output_dir,
                epochs=1,
                batch_size=2,
                target_size=(32, 32),
                base_channels=4,
                device="cpu",
                max_batches_per_epoch=1,
            )

            self.assertEqual(summary["epochs_completed"], 1)
            self.assertTrue((output_dir / "best.pt").exists())
            self.assertTrue((output_dir / "samples" / "epoch_000.png").exists())

    def test_explicit_dataset_splits_are_respected(self):
        from curve_model.train import _split_records
        records=[{"id":"a","split":"train"},{"id":"b","split":"validation"},{"id":"c","split":"test"}]
        train,validation=_split_records(records,.9,123);self.assertEqual([item["id"] for item in train],["a"]);self.assertEqual([item["id"] for item in validation],["b"])

    def test_inference_restores_original_dimensions(self):
        from curve_model.infer import predict_curve_probability
        from curve_model.model import CurvePhase1UNet

        with tempfile.TemporaryDirectory() as temp_dir:
            model = CurvePhase1UNet(base_channels=4)
            checkpoint = Path(temp_dir) / "model.pt"
            self.torch.save({
                "state_dict": model.state_dict(),
                "model_config": model.configuration(),
                "model_version": model.model_version,
            }, checkpoint)
            image = np.full((45, 31, 3), 220, dtype=np.uint8)
            result = predict_curve_probability(image, str(checkpoint), device="cpu", tile_height=24, overlap=8)

            self.assertEqual(result["stroke_probability"].shape, (45, 31))
            self.assertEqual(result["centerline_probability"].shape, (45, 31))
            self.assertTrue(np.isfinite(result["stroke_probability"]).all())


class InferenceIntegrationTests(unittest.TestCase):
    def test_classic_mode_is_byte_for_byte_unchanged(self):
        image = np.zeros((12, 9, 3), dtype=np.uint8)
        classic = np.arange(108, dtype=np.uint8).reshape(12, 9)

        result, metadata = build_phase1_probability(image, classic, mode="classic")

        np.testing.assert_array_equal(result, classic)
        self.assertFalse(metadata["fallback_occurred"])

    def test_missing_model_falls_back_to_classic(self):
        image = np.zeros((12, 9, 3), dtype=np.uint8)
        classic = np.full((12, 9), 77, dtype=np.uint8)

        result, metadata = build_phase1_probability(
            image,
            classic,
            mode="neural_phase1",
            model_path="missing-phase1-model.pt",
        )

        np.testing.assert_array_equal(result, classic)
        self.assertTrue(metadata["fallback_occurred"])
        self.assertEqual(metadata["tracing_mode"], "classic")

    def test_neural_probability_decodes_vertical_curve_with_small_error(self):
        import web_app

        height, width, target_x = 40, 24, 9
        image = np.zeros((height, width, 3), dtype=np.uint8)
        classic = np.zeros((height, width), dtype=np.uint8)
        centerline = np.zeros((height, width), dtype=np.float32)
        centerline[:, target_x] = 1.0

        def predictor(*args, **kwargs):
            return {
                "stroke_probability": centerline,
                "centerline_probability": centerline,
                "metadata": {"model_version": "test", "inference_duration_ms": 1.0},
            }

        probability, metadata = build_phase1_probability(
            image,
            classic,
            mode="neural_phase1",
            model_path="mock.pt",
            predictor=predictor,
        )
        predicted, _ = web_app.trace_curve_with_dp(
            probability, 0.0, 100.0, max_step=2, smooth_lambda=0.01
        )
        mae = float(np.nanmean(np.abs(predicted - target_x)))

        self.assertLessEqual(mae, 0.5)
        self.assertEqual(metadata["tracing_mode"], "neural_phase1")

    def test_tile_blending_has_no_large_seam_discontinuity(self):
        from curve_model.infer import blend_vertical_tiles

        image = np.zeros((73, 19, 3), dtype=np.uint8)

        def predictor(tile):
            probability = np.full(tile.shape[:2], 0.63, dtype=np.float32)
            return probability, probability

        stroke, centerline = blend_vertical_tiles(image, 28, 10, predictor)

        self.assertLess(float(np.max(np.abs(np.diff(stroke[:, 5])))), 1e-6)
        np.testing.assert_allclose(stroke, centerline)


class EvaluationMetricTests(unittest.TestCase):
    def test_known_path_metrics_are_quantitatively_correct(self):
        truth = np.linspace(4.0, 14.0, 20, dtype=np.float32)
        predicted = truth + 2.0
        predicted[3] = np.nan

        metrics = calculate_trace_metrics(predicted, truth)

        self.assertAlmostEqual(metrics["mean_absolute_error"], 2.0, places=5)
        self.assertEqual(metrics["missing_rows"], 1)
        self.assertAlmostEqual(metrics["accuracy"]["within_2px"], 1.0)
        self.assertAlmostEqual(metrics["accuracy"]["within_1px"], 0.0)


if __name__ == "__main__":
    unittest.main()
