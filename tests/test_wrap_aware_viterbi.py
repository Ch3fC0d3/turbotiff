import numpy as np
import cv2

import fast_tracer
import web_app


def test_black_sonic_curves_preserve_short_trace_excursions():
    assert web_app.should_preserve_black_trace_detail("black", "DTC")
    assert web_app.should_preserve_black_trace_detail("black", curve_name="DT")
    assert web_app.should_preserve_black_trace_detail("black", "RHOB", preserve_wiggles=True)
    assert not web_app.should_preserve_black_trace_detail("black", "RHOB")
    assert not web_app.should_preserve_black_trace_detail("green", "DTC")


def test_sonic_hot_side_points_toward_lower_travel_time_value():
    assert web_app.resolve_curve_hot_side(None, 30.0, 110.0, "DTC") == "left"
    assert web_app.resolve_curve_hot_side(None, 140.0, 40.0, "DT") == "right"
    assert web_app.resolve_curve_hot_side(None, 0.0, 150.0, "GR") == "right"
    assert web_app.resolve_curve_hot_side("right", 30.0, 110.0, "DTC") == "right"


def test_sonic_hot_ink_refinement_prefers_curve_over_vertical_rail():
    height, width = 180, 120
    rows = np.arange(height, dtype=np.float32)
    expected = 53.0 + 8.0 * np.sin(rows / 17.0)
    roi = np.full((height, width, 3), 255, dtype=np.uint8)
    points = np.column_stack((np.rint(expected).astype(np.int32), rows.astype(np.int32)))
    cv2.polylines(roi, [points.reshape(-1, 1, 2)], False, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(roi, (88, 0), (88, height - 1), (0, 0, 0), 2)
    for y in range(2, height, 7):
        cv2.circle(roi, (105, y), 1, (0, 0, 0), -1)
    for y in range(20, height, 30):
        cv2.line(roi, (0, y), (width - 1, y), (80, 80, 80), 1)

    # A real wrapped DTC excursion can run horizontally across a large part of
    # the track for several scan rows. Its hot edge is signal, not grid.
    for y in range(72, 79):
        x0 = int(round(expected[y]))
        cv2.line(roi, (x0, y), (x0 + 28, y), (0, 0, 0), 2)

    initial = expected - 18.0
    refined = web_app.refine_black_sonic_trace_to_hot_ink(
        roi,
        initial.astype(np.float32),
        hot_side="right",
        search_radius=60,
    )

    assert float(np.nanmedian(np.abs(refined - expected))) < 4.0
    assert float(np.nanmedian(np.abs(refined - 88.0))) > 20.0
    assert float(np.nanmedian(refined[73:78] - expected[73:78])) > 20.0
    assert float(np.nanmedian(np.abs(refined - 105.0))) > 30.0

    # When the incoming decoder starts on the far-right annotation column, a
    # left-crest sonic pass must reacquire the curve as one continuous path.
    far_right_initial = np.full(height, 105.0, dtype=np.float32)
    reacquired = web_app.refine_black_sonic_trace_to_hot_ink(
        roi,
        far_right_initial,
        hot_side="left",
    )
    assert float(np.nanmedian(np.abs(reacquired - expected))) < 4.0
    adjacent_support = np.isfinite(reacquired[:-1]) & np.isfinite(reacquired[1:])
    assert float(np.max(np.abs(np.diff(reacquired)[adjacent_support]))) < 12.0


def test_sonic_hot_ink_refines_partial_candidate_without_filling_gap(monkeypatch):
    height, width = 40, 90
    roi = np.full((height, width, 3), 255, dtype=np.uint8)
    initial = np.full(height, 20.0, dtype=np.float32)
    candidate = np.linspace(48.0, 62.0, height, dtype=np.float32)
    candidate[17:20] = np.nan

    def fake_viterbi(*_args, **_kwargs):
        return candidate.copy(), np.ones(height, dtype=np.float32)

    monkeypatch.setattr(web_app.fast_tracer, "run_viterbi", fake_viterbi)
    refined = web_app.refine_black_sonic_trace_to_hot_ink(
        roi,
        initial,
        hot_side="right",
    )

    # Supported rows recover the hot-ink path even though the detector has a
    # local dropout. A stale incoming spine in that dropout becomes a gap and
    # is never interpolated from either neighboring section.
    assert float(np.median(refined[:16])) > 45.0
    assert np.all(np.isnan(refined[17:20]))
    assert float(np.median(refined[21:])) > 50.0


def test_sonic_hot_ink_rejects_unconnected_horizontal_shelf_endpoints():
    height, width = 190, 140
    rows = np.arange(height, dtype=np.float32)
    expected = 66.0 + 9.0 * np.sin(rows / 21.0)
    roi = np.full((height, width, 3), 255, dtype=np.uint8)
    curve_points = np.column_stack((
        np.rint(expected).astype(np.int32),
        rows.astype(np.int32),
    ))
    cv2.polylines(
        roi,
        [curve_points.reshape(-1, 1, 2)],
        False,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    shelf_rows = np.asarray([30, 55, 80, 105, 130, 155], dtype=np.int32)
    for row in shelf_rows:
        # These dark shelves cross the real curve but their far endpoints do
        # not continue vertically into curve ink.
        cv2.line(roi, (12, int(row)), (126, int(row)), (0, 0, 0), 2)

    refined = web_app.refine_black_sonic_trace_to_hot_ink(
        roi,
        (expected - 12.0).astype(np.float32),
        hot_side="right",
        search_radius=75,
    )

    shelf_values = refined[shelf_rows]
    supported = np.isfinite(shelf_values)
    assert np.all(np.abs(shelf_values[supported] - 126.0) > 25.0)
    assert float(np.nanmedian(np.abs(refined - expected))) < 5.0


def test_wrapped_sonic_refinement_switches_crest_side_with_visible_branch():
    height, width = 210, 120
    rows = np.arange(height, dtype=np.float32)
    left_branch = 14.0 + 4.0 * np.sin(rows / 13.0)
    right_branch = 105.0 + 4.0 * np.sin(rows / 13.0)
    expected = left_branch.copy()
    expected[70:145] = right_branch[70:145]

    roi = np.full((height, width, 3), 255, dtype=np.uint8)
    for branch in (left_branch, right_branch):
        points = np.column_stack((np.rint(branch).astype(np.int32), rows.astype(np.int32)))
        cv2.polylines(roi, [points.reshape(-1, 1, 2)], False, (0, 0, 0), 2, cv2.LINE_AA)
    for y in range(15, height, 30):
        cv2.line(roi, (0, y), (width - 1, y), (80, 80, 80), 1)

    refined = web_app.refine_black_sonic_trace_to_hot_ink(
        roi,
        expected.astype(np.float32),
        hot_side="left",
        wrap_enabled=True,
    )

    assert float(np.median(refined[:65])) < width * 0.30
    assert float(np.median(refined[78:138])) > width * 0.70
    assert float(np.median(refined[155:])) < width * 0.30


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
