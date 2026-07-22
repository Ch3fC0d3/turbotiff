"""Direction-aware Phase 2 path decoding without changing classic Viterbi."""

from __future__ import annotations

import numpy as np

try:
    from numba import jit
except Exception:  # pragma: no cover
    def jit(*args, **kwargs):
        return lambda function: function

from .phase2_score import normalize_direction_field


@jit(nopython=True, cache=True)
def _run_direction_viterbi(cost, direction, max_step, smooth_lambda, direction_weight):
    height, width = cost.shape
    large = 1e12
    dp = np.full((height, width), large, dtype=np.float32)
    previous = np.full((height, width), -1, dtype=np.int32)
    dp[0] = cost[0]
    for row in range(1, height):
        for x in range(width):
            start = max(0, x - max_step)
            end = min(width, x + max_step + 1)
            best_cost = large
            best_previous = -1
            predicted_dx = direction[0, row, x]
            predicted_dy = direction[1, row, x]
            for prior_x in range(start, end):
                dx = x - prior_x
                magnitude = np.sqrt(float(dx * dx + 1))
                agreement = (float(dx) / magnitude) * predicted_dx + (1.0 / magnitude) * predicted_dy
                transition_cost = smooth_lambda * float(dx * dx) - direction_weight * agreement
                candidate = dp[row - 1, prior_x] + cost[row, x] + transition_cost
                if candidate < best_cost:
                    best_cost = candidate
                    best_previous = prior_x
            dp[row, x] = best_cost
            previous[row, x] = best_previous
    path = np.full(height, -1, dtype=np.int32)
    path[-1] = int(np.argmin(dp[-1]))
    for row in range(height - 1, 0, -1):
        path[row - 1] = previous[row, path[row]]
    return path


def decode_phase2_path(
    score: np.ndarray,
    direction_field: np.ndarray,
    max_step: int = 40,
    smooth_lambda: float = 0.005,
    direction_weight: float = 0.15,
) -> np.ndarray:
    probability = np.asarray(score, dtype=np.float32)
    if probability.ndim != 2 or not probability.size:
        raise ValueError("Phase 2 score must be a non-empty [H, W] array")
    probability = np.clip(np.nan_to_num(probability, nan=0.0), 0.0, 1.0)
    direction = normalize_direction_field(direction_field)
    if direction.shape[1:] != probability.shape:
        raise ValueError("Direction field does not match the Phase 2 score")
    max_step = max(1, min(int(max_step), probability.shape[1] - 1 if probability.shape[1] > 1 else 1))
    cost = (1.0 - probability).astype(np.float32)
    path = _run_direction_viterbi(
        cost,
        direction.astype(np.float32),
        max_step,
        max(0.0, float(smooth_lambda)),
        min(2.0, max(0.0, float(direction_weight))),
    )
    return path.astype(np.float32)
