from pathlib import Path

import cv2
import numpy as np

import web_app


def _slow_component_scoring(mask, guide, use_guide_distance=False):
    """Reference implementation matching the former per-label algorithm."""
    _, binary = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = mask.copy()
    for label in range(1, n_labels):
        _, _, width, height, area = [int(v) for v in stats[label]]
        points = np.argwhere(labels == label)
        ys = points[:, 0]
        xs = points[:, 1]
        mean_dist = float(np.mean(np.abs(xs - guide[ys])))
        aspect = float(width) / float(max(1, height))
        horiz_penalty = 1.0
        if aspect > 1.1:
            horiz_penalty *= max(0.01, 1.0 - (aspect - 1.1) * 1.5)
        width_ratio = float(width) / float(mask.shape[1])
        height_ratio = float(height) / float(mask.shape[0])
        if width_ratio > 0.12 and height_ratio < 0.20:
            horiz_penalty *= max(0.01, 1.0 - (width_ratio - 0.12) * 5.0)
        if aspect > 2.0 or (width_ratio > 0.25 and height_ratio < 0.20):
            horiz_penalty = min(horiz_penalty, 0.005)
        text_penalty = 1.0
        if area < 50 and height < 18 and width < 18 and mean_dist > 10.0:
            text_penalty *= max(0.01, 1.0 - (mean_dist - 10.0) * 0.08)
        dist_penalty = 1.0
        if use_guide_distance:
            if mean_dist > 6.0:
                dist_penalty *= np.exp(-0.04 * (mean_dist - 6.0))
            if mean_dist > 35.0:
                dist_penalty = min(dist_penalty, 0.005)
        looks_like_vertical_rail = height_ratio >= 0.25 and width_ratio <= 0.025
        if looks_like_vertical_rail:
            continuity_boost = 0.10
        elif height > 40:
            continuity_boost = 1.10
        elif height < 8 and mean_dist > 8.0:
            continuity_boost = 0.1
        else:
            continuity_boost = 1.0
        multiplier = np.clip(
            horiz_penalty * text_penalty * dist_penalty * continuity_boost,
            0.005,
            1.3,
        )
        component = labels == label
        out[component] = np.clip(mask[component] * multiplier, 0, 255).astype(np.uint8)
    return out


def test_component_scoring_matches_reference_output():
    mask = np.zeros((96, 140), dtype=np.uint8)
    cv2.line(mask, (62, 0), (78, 95), 230, 2)
    cv2.line(mask, (0, 30), (139, 30), 180, 1)
    cv2.rectangle(mask, (8, 8), (14, 13), 210, -1)
    cv2.rectangle(mask, (115, 70), (122, 77), 190, -1)
    guide = np.linspace(62.0, 78.0, mask.shape[0], dtype=np.float32)

    actual = web_app.score_and_suppress_black_components(
        mask,
        guide,
        use_guide_distance=True,
    )
    expected = _slow_component_scoring(mask, guide, use_guide_distance=True)

    np.testing.assert_array_equal(actual, expected)


def test_component_scoring_does_not_rescan_labels_per_component(monkeypatch):
    mask = np.zeros((256, 320), dtype=np.uint8)
    for y in range(2, 254, 5):
        for x in range(2, 318, 7):
            mask[y, x] = 200
    guide = np.full(mask.shape[0], 160.0, dtype=np.float32)

    def fail_argwhere(*_args, **_kwargs):
        raise AssertionError("per-component full-image scan was used")

    monkeypatch.setattr(web_app.np, "argwhere", fail_argwhere)
    result = web_app.score_and_suppress_black_components(mask, guide)

    assert result.shape == mask.shape
    assert result.dtype == np.uint8


def test_wrong_rail_guide_does_not_erase_high_excursion_curve():
    height, width = 180, 140
    rows = np.arange(height, dtype=np.float32)
    expected = 90.0 + 25.0 * np.sin(rows / 13.0)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Deliberately stronger first-pass lure: a full-height narrow rail.
    cv2.line(mask, (30, 0), (30, height - 1), 255, 2)
    curve_points = np.column_stack(
        (np.rint(expected).astype(np.int32), rows.astype(np.int32))
    )
    cv2.polylines(
        mask,
        [curve_points.reshape(-1, 1, 2)],
        False,
        230,
        2,
        cv2.LINE_AA,
    )
    wrong_guide = np.full(height, 30.0, dtype=np.float32)

    scored = web_app.score_and_suppress_black_components(
        mask,
        wrong_guide,
        curve_type='GR',
        use_guide_distance=False,
    )
    traced, _ = web_app.trace_curve_with_dp(
        scored,
        scale_min=0.0,
        scale_max=150.0,
        curve_type='GR',
        max_step=28,
        smooth_lambda=0.001,
        curv_lambda=0.001,
        hot_side='right',
    )

    assert float(np.median(np.abs(traced - expected))) < 5.0
    assert float(np.median(np.abs(traced - 30.0))) > 35.0
    assert float(np.mean(scored[:, 30])) < float(
        np.mean(scored[np.arange(height), np.rint(expected).astype(np.int32)])
    )


def test_black_high_excursion_preserves_connected_horizontal_tips():
    height, width = 240, 180
    rows = np.arange(height, dtype=np.float32)
    spine = 92.0 + 7.0 * np.sin(rows / 18.0)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Full-width grid and rail evidence is present before the stronger curve.
    for x in range(15, width, 20):
        cv2.line(mask, (x, 0), (x, height - 1), 85, 1)
    for y in range(12, height, 24):
        cv2.line(mask, (0, y), (width - 1, y), 85, 1)

    # The curve itself is one continuous path. At each excursion it leaves the
    # central region, traverses a nearly horizontal segment, and returns below
    # it; there is no fixed left/right reading side.
    excursions = [
        (28, 'right', 43),
        (57, 'left', 38),
        (86, 'right', 51),
        (119, 'left', 44),
        (151, 'right', 47),
        (184, 'left', 41),
        (214, 'right', 45),
    ]
    expected_rows = []
    expected_tips = []
    curve_x = spine.copy()
    for y, side, length in excursions:
        direction = 1.0 if side == 'right' else -1.0
        tip_x = float(spine[y]) + direction * float(length)
        for offset in range(-3, 4):
            weight = 1.0 - abs(float(offset)) / 3.0
            row = y + offset
            curve_x[row] = (
                (1.0 - weight) * float(spine[row])
                + weight * tip_x
            )
        expected_rows.append(y)
        expected_tips.append(tip_x)
    curve_points = np.column_stack((
        np.rint(curve_x).astype(np.int32),
        rows.astype(np.int32),
    ))

    cv2.polylines(
        mask,
        [np.asarray(curve_points, dtype=np.int32).reshape(-1, 1, 2)],
        False,
        255,
        1,
        cv2.LINE_8,
    )

    # Faint and broken curve portions must become local gaps, not rail bridges.
    mask[132:135, :] = np.minimum(mask[132:135, :], 45)
    mask[201, :] = 0

    anchors, _ = web_app.trace_curve_with_dp(
        mask,
        scale_min=0.0,
        scale_max=150.0,
        curve_type='GR',
        max_step=28,
        smooth_lambda=0.001,
        curv_lambda=0.001,
        hot_side='right',
    )
    traced, _, diagnostics = web_app.trace_black_skeleton_graph(
        mask,
        guide=anchors,
    )

    expected_rows = np.asarray(expected_rows, dtype=np.int32)
    expected_tips = np.asarray(expected_tips, dtype=np.float32)
    tip_errors = np.abs(traced[expected_rows] - expected_tips)
    finite_errors = tip_errors[np.isfinite(tip_errors)]

    assert finite_errors.size >= int(np.ceil(0.80 * expected_rows.size))
    assert float(np.median(finite_errors)) < 6.0
    assert float(np.percentile(finite_errors, 90)) < 12.0
    assert float(np.mean(tip_errors <= 10.0)) >= 0.80
    assert diagnostics['trace_strategy'] == 'black_skeleton_graph'
    assert diagnostics['skeleton_component_count'] > 0
    assert diagnostics['path_pixel_count'] >= diagnostics['finite_output_rows']

    longest_vertical = 0
    current_vertical = 0
    for previous, current in zip(traced[:-1], traced[1:]):
        if np.isfinite(previous) and np.isfinite(current) and abs(float(current - previous)) <= 0.5:
            current_vertical += 1
            longest_vertical = max(longest_vertical, current_vertical)
        else:
            current_vertical = 0
    assert longest_vertical < 20


def test_real_span_failure_uses_bidirectional_graph_excursions():
    """Regression for the production scan that stayed on the cyan spine.

    The fixture is the reported preview capture. The old cyan overlay is used
    only as the deliberately imperfect seed and is removed from detector
    evidence before tracing.
    """
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "black_span_spine_failure.png"
    )
    image = cv2.imread(str(fixture), cv2.IMREAD_COLOR)
    assert image is not None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(
        hsv,
        np.asarray([75, 80, 80], dtype=np.uint8),
        np.asarray([105, 255, 255], dtype=np.uint8),
    )
    guide = np.full(image.shape[0], np.nan, dtype=np.float32)
    for row in range(image.shape[0]):
        colored_x = np.where(cyan[row] > 0)[0]
        if colored_x.size:
            guide[row] = float(np.median(colored_x))
    supported_guide = np.where(np.isfinite(guide))[0]
    assert supported_guide.size
    guide[:] = np.interp(
        np.arange(guide.size),
        supported_guide,
        guide[supported_guide],
    )

    clean = image.copy()
    overlay_pixels = cv2.dilate(
        cyan,
        np.ones((5, 5), dtype=np.uint8),
    ) > 0
    clean[overlay_pixels] = 255
    probability = web_app.compute_prob_map(clean, mode="black")
    traced, _, diagnostics = web_app.trace_black_skeleton_graph(
        probability,
        guide=guide,
    )

    inside_track = (
        np.isfinite(traced)
        & (traced > 5)
        & (traced < image.shape[1] - 5)
    )
    displacement = traced - guide
    assert np.count_nonzero(inside_track & (displacement < -20)) >= 50
    assert np.count_nonzero(inside_track & (displacement > 20)) >= 20
    assert float(np.nanmedian(np.abs(displacement[inside_track]))) >= 20.0
    assert diagnostics["trace_strategy"] == "black_skeleton_graph"
    assert diagnostics["graph_branch_count"] > 0
    assert diagnostics["path_pixel_count"] > 0
    assert diagnostics["finite_output_rows"] == int(
        np.count_nonzero(np.isfinite(traced))
    )
