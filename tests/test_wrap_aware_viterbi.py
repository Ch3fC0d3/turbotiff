import numpy as np

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
