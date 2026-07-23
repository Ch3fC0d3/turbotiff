import re
from pathlib import Path

import numpy as np

from web_app import (
    APPROVED_INTERPOLATION_PATHS,
    can_fill_trace_gap,
    split_supported_trace_segments,
)


WEB_APP = Path(__file__).resolve().parents[1] / "web_app.py"


def test_unsupported_legacy_gap_stays_split_before_canonical_construction():
    xs = np.array([10, 12, np.nan, np.nan, 80, 82], dtype=np.float32)
    support = np.array([True, True, False, False, True, True])
    assert split_supported_trace_segments(xs, support) == [(0, 1), (4, 5)]


def test_local_gap_fill_requires_support_and_preserves_provenance_boundary():
    probability = np.array([1.0, 0.9, 1.0], dtype=np.float32)
    assert can_fill_trace_gap(0, 2, probability, max_gap_rows=1, minimum_support=0.5)
    assert not can_fill_trace_gap(0, 2, np.array([1.0, 0.0, 1.0]), max_gap_rows=1, minimum_support=0.5)
    assert not can_fill_trace_gap(0, 2, probability, explicit_breaks=[1], max_gap_rows=1, minimum_support=0.5)
    assert not can_fill_trace_gap(0, 2, probability, wrap_events=[1], max_gap_rows=1, minimum_support=0.5)


def test_wrap_event_always_splits_legacy_visible_candidates():
    xs = np.array([198, 2], dtype=np.float32)
    assert split_supported_trace_segments(xs, wrap_events=[1]) == [(0, 0), (1, 1)]


def test_interpolation_primitives_are_limited_to_the_reviewed_allowlist():
    source = WEB_APP.read_text(encoding="utf-8")
    calls = re.findall(
        r"np\.interp\(|\.interpolate\(|\.ffill\(|\.bfill\(|\.fillna\(",
        source,
    )
    assert len(calls) == 7, calls
    assert set(APPROVED_INTERPOLATION_PATHS) == {
        "_resample_supported_rows",
        "resample_values_by_continuous_sections",
        "ml_predict_curve",
        "refine_black_sonic_trace_to_hot_ink",
        "digitize_wrapped_sonic_continuity",
    }
    assert calls.count("np.interp(") == 3
    assert calls.count(".interpolate(") == 2
    assert calls.count(".ffill(") == 1
    assert calls.count(".bfill(") == 1
