import numpy as np

from web_app import (
    build_canonical_trace,
    canonical_trace_to_values,
    resample_values_by_continuous_sections,
    resample_canonical_trace_to_depth_grid,
    smooth_trace_supported_sections,
)


def test_wrap_transition_has_continuous_unwrapped_values_but_a_render_break_boundary():
    trace = build_canonical_trace(
        np.array([97, 99, 1, 3], dtype=np.float32),
        100,
        wrapped=True,
    )

    assert trace["wrap_cycle"] == [0, 0, 1, 1]
    assert trace["unwrapped_x"] == [97.0, 99.0, 101.0, 103.0]
    # The data is continuous; the canvas splits only because the visible cycle changes.
    assert trace["explicit_breaks"] == []
    visible = [x - cycle * 100 for x, cycle in zip(trace["unwrapped_x"], trace["wrap_cycle"])]
    assert visible == [97.0, 99.0, 1.0, 3.0]


def test_missing_rows_are_explicit_non_interpolable_section_boundaries():
    trace = build_canonical_trace(
        np.array([10, np.nan, np.nan, 14], dtype=np.float32),
        100,
        wrapped=False,
    )

    assert trace["explicit_breaks"] == [1, 2, 3]
    assert trace["unwrapped_x"][1:3] == [None, None]


def test_gap_does_not_guess_a_wrap_cycle_for_the_next_supported_row():
    trace = build_canonical_trace(
        np.array([98, np.nan, 2], dtype=np.float32),
        100,
        wrapped=True,
    )

    # There is no confirmed transition across the missing evidence, so the
    # final point remains in its supplied/display cycle rather than being
    # unwrapped relative to the point before the gap.
    assert trace["unwrapped_x"] == [98.0, None, 2.0]
    assert trace["wrap_cycle"] == [0, 0, 0]
    assert trace["explicit_breaks"] == [1, 2]


def test_legacy_visible_trace_only_infers_a_wrap_at_opposite_track_edges():
    trace = build_canonical_trace(
        np.array([20, 90], dtype=np.float32),
        100,
        wrapped=True,
    )

    assert trace["unwrapped_x"] == [20.0, 90.0]
    assert trace["wrap_cycle"] == [0, 0]


def test_las_calibration_uses_unwrapped_coordinates_across_multiple_cycles():
    trace = build_canonical_trace(
        np.array([97, 99, 1, 3], dtype=np.float32),
        100,
        wrapped=True,
    )
    values = canonical_trace_to_values(
        trace,
        left_value=0,
        right_value=100,
        scale_type="linear",
        track_width=100,
    )

    assert np.all(np.diff(values) > 0)
    assert values[-1] > 100


def test_reverse_wrap_direction_and_manual_break_are_preserved():
    trace = build_canonical_trace(
        np.array([3, 1, 99, 97], dtype=np.float32),
        100,
        wrapped=True,
        unwrapped_x=np.array([3, 1, -1, -3], dtype=np.float32),
        wrap_cycle=np.array([0, 0, -1, -1], dtype=np.int32),
        explicit_breaks=[3],
    )

    assert trace["wrap_cycle"] == [0, 0, -1, -1]
    assert trace["unwrapped_x"] == [3.0, 1.0, -1.0, -3.0]
    assert trace["explicit_breaks"] == [3]


def test_resampling_never_interpolates_across_a_manual_break_or_long_gap():
    sampled = resample_values_by_continuous_sections(
        np.array([0, 1, 2, 3], dtype=np.float32),
        np.array([10, 11, 101, 102], dtype=np.float32),
        np.arange(0, 3.1, 0.5, dtype=np.float32),
        explicit_breaks=[2],
        null_value=-999.25,
    )

    assert sampled.tolist() == [10.0, 10.5, 11.0, -999.25, 101.0, 101.5, 102.0]


def test_smoothing_preserves_missing_rows_and_explicit_breaks():
    smoothed = smooth_trace_supported_sections(
        np.array([40, 42, np.nan, np.nan, 44, 46], dtype=np.float32), window=5
    )
    assert np.isnan(smoothed[2:4]).all()
    broken = smooth_trace_supported_sections(
        np.array([40, 42, 80, 82], dtype=np.float32), window=5, explicit_breaks=[2]
    )
    assert broken.tolist() == [41.0, 41.0, 81.0, 81.0]


def test_wrapped_smoothing_uses_unwrapped_coordinates_not_track_middle():
    smoothed = smooth_trace_supported_sections(
        np.array([97, 99, 1, 3], dtype=np.float32), window=3, wrap_width=100
    )
    assert not np.any((smoothed > 20) & (smoothed < 80))


def test_canonical_resampling_adopts_the_committed_depth_grid_without_bridging():
    source = build_canonical_trace(np.array([20, 22, np.nan, np.nan, 28, 30]), 100, wrapped=False)
    committed = resample_canonical_trace_to_depth_grid(
        source, np.array([0, 1, 2, 3, 4, 5]), np.arange(0, 5.1, 0.5)
    )
    assert len(committed["unwrapped_x"]) == len(committed["wrap_cycle"]) == len(committed["visible_x"]) == 11
    assert committed["unwrapped_x"][3:8] == [None] * 5
    assert 3 in committed["explicit_breaks"]
