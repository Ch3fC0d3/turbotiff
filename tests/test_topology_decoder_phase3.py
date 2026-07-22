import unittest

import numpy as np

from curve_decoder import CurveEvidence, DecoderConfig, decode_curve_path
from curve_decoder.cylindrical_dp import align_unwrapped_paths, compare_forward_backward
from curve_decoder.editing import add_path_break, move_points, remove_wrap_transition, set_wrap_transition
from curve_decoder.metrics import calculate_topology_metrics
from curve_decoder.rendering import build_visible_segments, has_cross_track_connector
from curve_decoder.scale import ScaleConfig, path_to_values
from training.synthetic_log_generator import SyntheticLogConfig, generate_sample


def evidence_from_unwrapped(unwrapped, width, missing_rows=None, distractor=None):
    unwrapped = np.asarray(unwrapped, dtype=np.float32)
    height = unwrapped.size
    visible = np.mod(unwrapped, float(width))
    columns = np.arange(width, dtype=np.float32)[None, :]
    periodic_distance = np.abs(columns - visible[:, None])
    periodic_distance = np.minimum(periodic_distance, float(width) - periodic_distance)
    center = np.exp(-0.5 * np.square(periodic_distance / 0.7)).astype(np.float32)
    distance = np.clip(1.0 - periodic_distance / 7.0, 0.0, 1.0).astype(np.float32)
    if missing_rows is not None:
        center[np.asarray(missing_rows, dtype=np.int32)] = 0.0
    if distractor is not None:
        center = np.maximum(center, np.asarray(distractor, dtype=np.float32))
    slope = np.gradient(unwrapped)
    magnitude = np.sqrt(slope * slope + 1.0)
    direction = np.zeros((2, height, width), dtype=np.float32)
    direction[0] = (slope / magnitude)[:, None]
    direction[1] = (1.0 / magnitude)[:, None]
    wrap = np.floor_divide(unwrapped, float(width)).astype(np.int32)
    rtl = np.zeros(height, dtype=np.float32)
    ltr = np.zeros(height, dtype=np.float32)
    for row in np.flatnonzero(np.diff(wrap) != 0) + 1:
        if wrap[row] > wrap[row - 1]:
            rtl[row] = 1.0
        else:
            ltr[row] = 1.0
    return CurveEvidence(
        centerline_probability=center,
        distance_field=distance,
        direction_field=direction,
        wrap_probability_right_to_left=rtl,
        wrap_probability_left_to_right=ltr,
    ), visible.astype(np.float32), wrap


def decode_known(unwrapped, width=64, **overrides):
    evidence, visible, wrap = evidence_from_unwrapped(unwrapped, width)
    values = dict(
        topology="cylindrical",
        max_step=4,
        max_slope=4,
        slope_bins=9,
        beam_width=72,
        edge_transition_width=6,
        minimum_rows_between_wraps=4,
        maximum_wrap_count=4,
        rendering_discontinuity=20,
    )
    values.update(overrides)
    return decode_curve_path(evidence, DecoderConfig(**values)), visible, wrap


class TopologyDecoderTests(unittest.TestCase):
    def test_bounded_decoder_has_no_wrap_and_small_error(self):
        truth = 25.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, 80))
        evidence, visible, _ = evidence_from_unwrapped(truth, 64)
        result = decode_curve_path(evidence, DecoderConfig(topology="bounded", max_step=3, max_slope=3, slope_bins=7, beam_width=48))
        self.assertLess(float(np.mean(np.abs(result.x_by_row - visible))), 1.0)
        self.assertFalse(result.wrap_events)
        self.assertTrue(np.all(result.wrap_index_by_row == 0))

    def test_right_to_left_wrap_is_explicit_and_continuous(self):
        truth = np.linspace(45.0, 83.0, 96)
        result, visible, wrap = decode_known(truth)
        self.assertEqual([event["direction"] for event in result.wrap_events], ["right_to_left"])
        self.assertLess(float(np.mean(np.abs(result.unwrapped_x_by_row - truth))), 1.1)
        self.assertGreater(float(np.max(np.abs(np.diff(result.x_by_row)))), 50.0)
        self.assertLess(float(np.max(np.abs(np.diff(result.unwrapped_x_by_row)))), 2.0)
        self.assertTrue(np.array_equal(result.wrap_index_by_row, wrap))
        self.assertFalse(result.metadata["cross_track_connector"])
        self.assertEqual(len(result.metadata["wrap_event_confidence"]), 1)
        self.assertGreaterEqual(result.metadata["observation_score_duration_ms"], 0.0)
        self.assertGreater(result.metadata["retained_state_bytes"], 0)

    def test_left_to_right_wrap_is_explicit(self):
        truth = np.linspace(20.0, -18.0, 96)
        result, _, wrap = decode_known(truth)
        self.assertEqual([event["direction"] for event in result.wrap_events], ["left_to_right"])
        self.assertLess(float(np.mean(np.abs(result.unwrapped_x_by_row - truth))), 1.1)
        self.assertTrue(np.array_equal(result.wrap_index_by_row, wrap))

    def test_multiple_wraps_and_event_metrics(self):
        truth = np.linspace(10.0, 150.0, 150)
        result, visible, wrap = decode_known(truth, maximum_wrap_count=3)
        metrics = calculate_topology_metrics(
            result.x_by_row, result.unwrapped_x_by_row, result.wrap_index_by_row,
            visible, truth, wrap, 64,
        )
        self.assertEqual(len(result.wrap_events), 2)
        self.assertEqual(metrics["wrap_events"]["true_positive"], 2)
        self.assertEqual(metrics["false_wraps"], 0)
        self.assertEqual(metrics["missed_wraps"], 0)
        self.assertLess(metrics["unwrapped_mean_absolute_error"], 1.2)

    def test_turning_away_from_border_does_not_false_wrap(self):
        first = np.linspace(25.0, 61.0, 45)
        truth = np.concatenate((first, np.linspace(61.0, 31.0, 45)))
        result, _, _ = decode_known(truth)
        self.assertEqual(result.wrap_events, [])
        self.assertTrue(np.all(result.wrap_index_by_row == 0))

    def test_missing_ink_is_bridged_by_soft_distance_evidence(self):
        truth = np.linspace(46.0, 82.0, 100)
        missing = np.arange(43, 58)
        evidence, _, _ = evidence_from_unwrapped(truth, 64, missing_rows=missing)
        result = decode_curve_path(evidence, DecoderConfig(
            topology="cylindrical", max_step=4, max_slope=4, slope_bins=9,
            beam_width=72, edge_transition_width=6, minimum_rows_between_wraps=4,
        ))
        self.assertEqual(len(result.wrap_events), 1)
        self.assertLess(float(np.mean(np.abs(result.unwrapped_x_by_row[missing] - truth[missing]))), 1.5)

    def test_opposite_border_distraction_does_not_trigger_wrap(self):
        truth = np.linspace(14.0, 38.0, 90)
        distractor = np.zeros((90, 64), dtype=np.float32)
        distractor[:, 62] = 0.88
        evidence, _, _ = evidence_from_unwrapped(truth, 64, distractor=distractor)
        result = decode_curve_path(evidence, DecoderConfig(
            topology="cylindrical", max_step=3, max_slope=3, slope_bins=7,
            beam_width=64, edge_transition_width=5,
        ))
        self.assertFalse(result.wrap_events)
        self.assertLess(float(np.mean(np.abs(result.unwrapped_x_by_row - truth))), 1.2)

    def test_decoder_is_deterministic_and_exact_matches_easy_beam_case(self):
        truth = np.linspace(5.0, 28.0, 28)
        evidence, _, _ = evidence_from_unwrapped(truth, 32)
        beam_config = DecoderConfig(topology="bounded", max_step=2, max_slope=2, slope_bins=5, beam_width=32)
        exact_config = DecoderConfig(topology="bounded", max_step=2, max_slope=2, slope_bins=5, beam_width=None)
        first = decode_curve_path(evidence, beam_config)
        second = decode_curve_path(evidence, beam_config)
        exact = decode_curve_path(evidence, exact_config)
        np.testing.assert_array_equal(first.x_by_row, second.x_by_row)
        np.testing.assert_allclose(first.x_by_row, exact.x_by_row, atol=0.0)

    def test_forward_backward_alignment_removes_whole_track_offset(self):
        reference = np.array([62, 63, 64, 65, 66], dtype=np.float32)
        candidate = reference - 64
        aligned = align_unwrapped_paths(reference, candidate, 64)
        np.testing.assert_array_equal(aligned, reference)
        comparison = compare_forward_backward(reference, candidate, 64)
        self.assertEqual(comparison["maximum_disagreement"], 0.0)

    def test_invalid_configuration_is_rejected(self):
        evidence, _, _ = evidence_from_unwrapped(np.linspace(2, 8, 10), 16)
        with self.assertRaises(ValueError):
            decode_curve_path(evidence, DecoderConfig(max_step=2, max_slope=3))


class TopologyRenderingScaleEditingTests(unittest.TestCase):
    def test_segments_never_join_across_wrap(self):
        x = np.array([60, 62, 63, 0, 1, 3], dtype=np.float32)
        wrap = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        segments = build_visible_segments(x, wrap)
        self.assertEqual(len(segments), 2)
        self.assertFalse(has_cross_track_connector(segments, 64))

    def test_linear_and_log_scale_use_explicit_wrap_index(self):
        x = np.array([0, 31.5, 63, 0, 63], dtype=np.float32)
        wrap = np.array([0, 0, 0, 1, -1], dtype=np.int32)
        linear = path_to_values(x, wrap, ScaleConfig(64, 0, 100, "linear"))
        np.testing.assert_allclose(linear, [0, 50, 100, 100, 0], atol=1e-5)
        logarithmic = path_to_values(x[:4], wrap[:4], ScaleConfig(64, 1, 100, "log"))
        np.testing.assert_allclose(logarithmic, [1, 10, 100, 100], rtol=1e-5)

    def test_editing_helpers_refresh_unwrapped_path_events_and_breaks(self):
        truth = np.linspace(45.0, 75.0, 60)
        result, _, _ = decode_known(truth)
        edited = move_points(result, {10: 12.5}, 64)
        self.assertEqual(float(edited.x_by_row[10]), 12.5)
        edited = add_path_break(edited, 20, 64)
        self.assertIn(20, edited.break_rows)
        edited = set_wrap_transition(edited, 30, "right_to_left", 64)
        self.assertGreater(int(edited.wrap_index_by_row[-1]), int(edited.wrap_index_by_row[0]))
        edited = remove_wrap_transition(edited, 30, 64)
        self.assertEqual(int(edited.wrap_index_by_row[30]), int(edited.wrap_index_by_row[29]))

    def test_wrapped_synthetic_labels_have_no_drawn_connector(self):
        sample = generate_sample(9001, SyntheticLogConfig(
            width=64, height=96, wrap_mode="right_to_left",
            enable_degradation=False, enable_missing_sections=False,
            enable_dashed_curves=False, enable_text_fragments=False,
        ))
        self.assertEqual(sample["metadata"]["topology"], "cylindrical")
        self.assertEqual(len(sample["wrap_events"]), 1)
        event_row = sample["wrap_events"][0]["row_after"]
        mask = sample["centerline_mask"]
        middle = mask[event_row - 1:event_row + 1, 20:44]
        self.assertEqual(int(middle.max()), 0)


if __name__ == "__main__":
    unittest.main()
