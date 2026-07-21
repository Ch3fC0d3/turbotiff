import numpy as np

import web_app


def test_tall_dp_decodes_bounded_rows_and_restores_full_height(monkeypatch):
    height, width = 12_000, 80
    mask = np.zeros((height, width), dtype=np.uint8)
    expected = 30.0 + 8.0 * np.sin(np.linspace(0.0, 8.0 * np.pi, height))
    for row, x_value in enumerate(expected):
        x = int(round(x_value))
        mask[row, max(0, x - 1):min(width, x + 2)] = 255

    monkeypatch.setenv("TURBOTIFF_MAX_DP_ROWS", "3000")
    xs, confidence = web_app.trace_curve_with_dp(
        mask,
        scale_min=0.0,
        scale_max=1.0,
        curve_type="DTC",
        max_step=6,
    )

    assert xs.shape == (height,)
    assert confidence.shape == (height,)
    assert np.isfinite(xs).all()
    assert float(np.median(np.abs(xs - expected))) < 2.0
