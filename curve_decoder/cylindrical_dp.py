"""Deterministic slope-state beam decoder for bounded and cylindrical tracks."""

from __future__ import annotations

import time

import numpy as np

from .confidence import calculate_confidence
from .config import DecoderConfig
from .evidence import CurveEvidence, calculate_observation_score, normalized_direction, _soft_field
from .path_result import CurvePathResult
from .rendering import build_visible_segments, has_cross_track_connector


def _select_initial(score_row: np.ndarray, config: DecoderConfig) -> np.ndarray:
    width = score_row.size
    limit = width if config.beam_width is None else min(width, int(config.beam_width))
    order = np.lexsort((np.arange(width), -score_row))
    if limit >= width:
        return order.astype(np.int32)
    selected = []
    bucket_counts = {}
    radius = max(1, int(config.diversity_radius))
    for x in order:
        bucket = int(x) // radius
        if bucket_counts.get(bucket, 0) >= int(config.states_per_diversity_bucket):
            continue
        selected.append(int(x))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if len(selected) >= limit:
            break
    return np.asarray(selected, dtype=np.int32)


def _wrap_events(result: CurvePathResult, track_width: int) -> list[dict]:
    events = []
    for row in range(1, result.wrap_index_by_row.size):
        before = int(result.wrap_index_by_row[row - 1])
        after = int(result.wrap_index_by_row[row])
        if after == before:
            continue
        direction = "right_to_left" if after > before else "left_to_right"
        events.append({
            "row_before": row - 1,
            "row_after": row,
            "direction": direction,
            "wrap_index_before": before,
            "wrap_index_after": after,
            "x_before": float(result.x_by_row[row - 1]),
            "x_after": float(result.x_by_row[row]),
            "unwrapped_delta": float(result.unwrapped_x_by_row[row] - result.unwrapped_x_by_row[row - 1]),
            "confidence": float(min(result.confidence_by_row[row - 1], result.confidence_by_row[row])),
        })
    return events


def decode_curve_path(evidence: CurveEvidence, config: DecoderConfig | None = None) -> CurvePathResult:
    config = config or DecoderConfig()
    height, width = evidence.validate()
    config.validate(width, height)
    started = time.perf_counter()
    observation_started = time.perf_counter()
    observation = calculate_observation_score(evidence, config)
    observation_duration_ms = (time.perf_counter() - observation_started) * 1000.0
    direction = normalized_direction(evidence)
    wrap_rtl = _soft_field(evidence.wrap_probability_right_to_left, (height,), 0.0).reshape(-1)
    wrap_ltr = _soft_field(evidence.wrap_probability_left_to_right, (height,), 0.0).reshape(-1)
    slopes = config.slope_values().astype(np.int32)
    if slopes.size == 0:
        raise ValueError("Decoder has no permitted slope states")

    initial_x = _select_initial(observation[0], config)
    initial_cost = -np.log(np.maximum(observation[0, initial_x], 1e-6)).astype(np.float32)
    history = [{
        "x": initial_x,
        "slope": np.zeros(initial_x.size, dtype=np.int16),
        "wrap": np.zeros(initial_x.size, dtype=np.int16),
        "cost": initial_cost,
        "prev": np.full(initial_x.size, -1, dtype=np.int32),
        "last_wrap": np.full(initial_x.size, -32768, dtype=np.int32),
        "last_direction": np.zeros(initial_x.size, dtype=np.int8),
        "event_count": np.zeros(initial_x.size, dtype=np.int16),
        "observation_cost": initial_cost.copy(),
        "transition_cost": np.zeros(initial_x.size, dtype=np.float32),
    }]
    states_evaluated = int(initial_x.size)
    edge_width = min(max(1, int(config.edge_transition_width)), max(1, width // 2))
    beam_limit = int(config.maximum_candidate_states) if config.beam_width is None else int(config.beam_width)

    for row in range(1, height):
        previous = history[-1]
        count = previous["x"].size
        previous_index = np.repeat(np.arange(count, dtype=np.int32), slopes.size)
        delta = np.tile(slopes, count).astype(np.int32)
        previous_x = previous["x"][previous_index].astype(np.int32)
        previous_wrap = previous["wrap"][previous_index].astype(np.int32)
        previous_unwrapped = previous_x + previous_wrap * width
        candidate_unwrapped = previous_unwrapped + delta

        if config.topology == "bounded":
            candidate_wrap = np.zeros_like(candidate_unwrapped)
            candidate_x = candidate_unwrapped.copy()
            valid = (candidate_x >= 0) & (candidate_x < width)
        else:
            candidate_wrap = np.floor_divide(candidate_unwrapped, width)
            candidate_x = candidate_unwrapped - candidate_wrap * width
            wrap_change = candidate_wrap - previous_wrap
            valid = np.abs(wrap_change) <= 1
            positive = wrap_change == 1
            negative = wrap_change == -1
            valid &= (~positive) | ((previous_x >= width - edge_width) & (candidate_x < edge_width))
            valid &= (~negative) | ((previous_x < edge_width) & (candidate_x >= width - edge_width))
            valid &= np.abs(candidate_wrap) <= int(config.maximum_wrap_count)

        wrap_change = candidate_wrap - previous_wrap
        is_wrap = wrap_change != 0
        event_count = previous["event_count"][previous_index] + is_wrap.astype(np.int16)
        valid &= event_count <= int(config.maximum_wrap_count)
        rows_since_wrap = row - previous["last_wrap"][previous_index]
        valid &= (~is_wrap) | (rows_since_wrap >= int(config.minimum_rows_between_wraps))
        valid_indices = np.flatnonzero(valid)
        states_evaluated += int(valid_indices.size)
        if not valid_indices.size:
            raise RuntimeError(f"No finite topology path reached row {row}")
        previous_index = previous_index[valid_indices]
        delta = delta[valid_indices]
        previous_x = previous_x[valid_indices]
        previous_wrap = previous_wrap[valid_indices]
        candidate_x = candidate_x[valid_indices].astype(np.int32)
        candidate_wrap = candidate_wrap[valid_indices].astype(np.int32)
        wrap_change = wrap_change[valid_indices].astype(np.int8)
        is_wrap = is_wrap[valid_indices]
        event_count = event_count[valid_indices]

        observation_cost = -np.log(np.maximum(observation[row, candidate_x], 1e-6))
        transition_cost = float(config.step_weight) * np.abs(delta)
        transition_cost = transition_cost + float(config.curvature_weight) * np.abs(
            delta - previous["slope"][previous_index].astype(np.int32)
        )
        magnitude = np.sqrt(delta.astype(np.float32) ** 2 + 1.0)
        transition_dx = delta.astype(np.float32) / magnitude
        transition_dy = 1.0 / magnitude
        predicted_dx = 0.5 * (direction[0, row - 1, previous_x] + direction[0, row, candidate_x])
        predicted_dy = 0.5 * (direction[1, row - 1, previous_x] + direction[1, row, candidate_x])
        predicted_norm = np.maximum(np.sqrt(predicted_dx ** 2 + predicted_dy ** 2), 1e-6)
        agreement = (transition_dx * predicted_dx + transition_dy * predicted_dy) / predicted_norm
        transition_cost = transition_cost + float(config.direction_weight) * 0.5 * (1.0 - agreement)
        if is_wrap.any():
            evidence_bonus = np.where(wrap_change > 0, wrap_rtl[row], wrap_ltr[row])
            transition_cost = transition_cost + is_wrap * (
                float(config.wrap_penalty) - float(config.wrap_evidence_weight) * evidence_bonus
            )
            reverse = is_wrap & (previous["last_direction"][previous_index] != 0) & (
                previous["last_direction"][previous_index] != wrap_change
            )
            transition_cost = transition_cost + reverse * float(config.reverse_wrap_penalty)
        total_cost = previous["cost"][previous_index] + observation_cost + transition_cost
        last_wrap = np.where(is_wrap, row, previous["last_wrap"][previous_index]).astype(np.int32)
        last_direction = np.where(is_wrap, wrap_change, previous["last_direction"][previous_index]).astype(np.int8)

        order = np.lexsort((previous_index, candidate_wrap, delta, candidate_x, total_cost))
        accepted = []
        seen = set()
        buckets = {}
        radius = max(1, int(config.diversity_radius))
        for index in order:
            key = (
                int(candidate_x[index]), int(delta[index]), int(candidate_wrap[index]),
                int(last_wrap[index]), int(last_direction[index]), int(event_count[index]),
            )
            if key in seen:
                continue
            if config.beam_width is not None:
                bucket = (int(candidate_wrap[index]), int(candidate_x[index]) // radius)
                if buckets.get(bucket, 0) >= int(config.states_per_diversity_bucket):
                    continue
                buckets[bucket] = buckets.get(bucket, 0) + 1
            seen.add(key)
            accepted.append(int(index))
            if len(accepted) >= beam_limit:
                break
        if not accepted:
            raise RuntimeError(f"Beam diversity pruning removed every state at row {row}")
        selected = np.asarray(accepted, dtype=np.int32)
        history.append({
            "x": candidate_x[selected].astype(np.int32),
            "slope": delta[selected].astype(np.int16),
            "wrap": candidate_wrap[selected].astype(np.int16),
            "cost": total_cost[selected].astype(np.float32),
            "prev": previous_index[selected].astype(np.int32),
            "last_wrap": last_wrap[selected],
            "last_direction": last_direction[selected],
            "event_count": event_count[selected].astype(np.int16),
            "observation_cost": observation_cost[selected].astype(np.float32),
            "transition_cost": transition_cost[selected].astype(np.float32),
        })

    selected_indices = np.full(height, -1, dtype=np.int32)
    selected_indices[-1] = int(np.argmin(history[-1]["cost"]))
    for row in range(height - 1, 0, -1):
        selected_indices[row - 1] = history[row]["prev"][selected_indices[row]]
    x = np.array([history[row]["x"][selected_indices[row]] for row in range(height)], dtype=np.float32)
    slope = np.array([history[row]["slope"][selected_indices[row]] for row in range(height)], dtype=np.float32)
    wrap = np.array([history[row]["wrap"][selected_indices[row]] for row in range(height)], dtype=np.int32)
    unwrapped = x + wrap.astype(np.float32) * float(width)
    observation_by_row = np.array([
        observation[row, int(x[row])] for row in range(height)
    ], dtype=np.float32)
    transition_by_row = np.array([
        history[row]["transition_cost"][selected_indices[row]] for row in range(height)
    ], dtype=np.float32)
    alternative_margin = np.zeros(height, dtype=np.float32)
    for row in range(height):
        chosen = selected_indices[row]
        state = history[row]
        different = (state["wrap"] != state["wrap"][chosen]) | (np.abs(state["x"] - state["x"][chosen]) > max(1, int(config.diversity_radius)))
        if np.any(different):
            alternative_margin[row] = max(0.0, float(np.min(state["cost"][different]) - state["cost"][chosen]))
    confidence, summary = calculate_confidence(x, unwrapped, observation, evidence, alternative_margin, config)
    result = CurvePathResult(
        x_by_row=x,
        unwrapped_x_by_row=unwrapped,
        wrap_index_by_row=wrap,
        slope_by_row=slope,
        confidence_by_row=confidence,
        observation_score_by_row=observation_by_row,
        transition_score_by_row=transition_by_row,
    )
    result.wrap_events = _wrap_events(result, width)
    result.visible_segments = build_visible_segments(
        x, wrap, discontinuity_threshold=float(config.rendering_discontinuity)
    )
    retained_state_bytes = int(sum(
        value.nbytes
        for state in history
        for value in state.values()
        if isinstance(value, np.ndarray)
    ))
    result.metadata = {
        "decoder": "topology_dp",
        "topology": config.topology,
        "track_width": int(width),
        "track_height": int(height),
        "beam_width": config.beam_width,
        "slope_values": slopes.tolist(),
        "states_evaluated": int(states_evaluated),
        "peak_beam_states": int(max(state["x"].size for state in history)),
        "retained_state_bytes": retained_state_bytes,
        "observation_score_duration_ms": observation_duration_ms,
        "total_energy": float(history[-1]["cost"][selected_indices[-1]]),
        "decoder_duration_ms": (time.perf_counter() - started) * 1000.0,
        "wrap_count": len(result.wrap_events),
        "wrap_event_confidence": [float(event["confidence"]) for event in result.wrap_events],
        "cross_track_connector": has_cross_track_connector(result.visible_segments, width),
        "confidence": summary,
        "config": config.to_dict(),
    }
    return result


def align_unwrapped_paths(reference: np.ndarray, candidate: np.ndarray, track_width: int) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if not valid.any() or int(track_width) < 1:
        return candidate.copy()
    offset_cycles = int(np.rint(np.median((reference[valid] - candidate[valid]) / float(track_width))))
    return candidate + offset_cycles * float(track_width)


def compare_forward_backward(reference: np.ndarray, candidate: np.ndarray, track_width: int) -> dict:
    aligned = align_unwrapped_paths(reference, candidate, track_width)
    valid = np.isfinite(reference) & np.isfinite(aligned)
    errors = np.abs(np.asarray(reference)[valid] - aligned[valid])
    return {
        "aligned_candidate": aligned,
        "mean_disagreement": float(np.mean(errors)) if errors.size else None,
        "maximum_disagreement": float(np.max(errors)) if errors.size else None,
    }
