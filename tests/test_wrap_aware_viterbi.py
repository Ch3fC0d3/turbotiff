import numpy as np
import cv2

import fast_tracer
import web_app


def test_wrap_aware_viterbi_continues_across_track_boundary():
    height, width = 180, 64
    unwrapped = np.linspace(54.0, 78.0, height, dtype=np.float32)
    expected = np.mod(unwrapped, width)
    probability = np.full((height, width), 0.001, dtype=np.float32)
    for row, x_value in enumerate(expected):
        x = int(round(float(x_value))) % width
        probability[row, x] = 1.0
        probability[row, (x - 1) % width] = 0.55
        probability[row, (x + 1) % width] = 0.55
    cost = -np.log(np.clip(probability, 1e-6, 1.0)).astype(np.float32)

    xs, _ = fast_tracer.run_viterbi(
        cost,
        probability,
        3,
        0.15,
        0.0,
        True,
    )

    circular_error = np.abs((xs - expected + width / 2.0) % width - width / 2.0)
    assert float(np.nanmedian(circular_error)) < 1.0
    assert np.any(xs[:100] > width - 4)
    assert np.any(xs[100:] < 4)


def test_velocity_guard_preserves_a_real_wrap_transition():
    visible = np.array([58.0, 60.0, 62.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32)

    guarded = web_app.guard_trace_velocity(visible, max_dx=6.0, wrap_width=64)

    circular_error = np.abs((guarded - visible + 32.0) % 64.0 - 32.0)
    assert float(np.max(circular_error)) < 0.01


def test_wrap_aware_centerline_refinement_tracks_ink_across_boundary():
    height, width = 48, 64
    expected = np.mod(np.linspace(58.0, 70.0, height, dtype=np.float32), width)
    mask = np.zeros((height, width), dtype=np.uint8)
    for row, x_value in enumerate(expected):
        center = int(round(float(x_value))) % width
        mask[row, [(center - 1) % width, center, (center + 1) % width]] = 255

    initial = np.mod(expected + 3.0, width).astype(np.float32)
    refined = web_app.refine_to_stroke_centerline(
        mask,
        initial,
        threshold_ratio=0.5,
        window_size=6,
        wrap_width=width,
    )

    circular_error = np.abs((refined - expected + width / 2.0) % width - width / 2.0)
    assert float(np.nanmedian(circular_error)) < 1.0


def test_continuous_line_refinement_remains_accurate_through_wrap(monkeypatch):
    height, width = 61, 64
    expected = np.mod(np.linspace(57.0, 71.0, height, dtype=np.float32), width)
    residual = np.zeros((height, width), dtype=np.float32)
    for row, x_value in enumerate(expected):
        center = int(round(float(x_value))) % width
        residual[row, center] = 1.0
        residual[row, (center - 1) % width] = 0.65
        residual[row, (center + 1) % width] = 0.65

    gray = np.full((height, width), 255, dtype=np.uint8)
    roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    monkeypatch.setattr(
        web_app,
        "build_black_prescan_grid_removed",
        lambda _gray: (_gray, residual, np.zeros_like(residual), None),
    )

    initial = np.mod(expected + 4.0, width).astype(np.float32)
    refined = web_app.refine_black_trace_to_continuous_line(
        roi,
        initial,
        search_radius=8,
        guide_window=9,
        vertical_window=5,
        min_line_score=0.01,
        min_score_gain=0.0,
        distance_weight=0.005,
        wrap_width=width,
    )

    circular_error = np.abs((refined - expected + width / 2.0) % width - width / 2.0)
    assert float(np.nanmedian(circular_error)) < 1.5
