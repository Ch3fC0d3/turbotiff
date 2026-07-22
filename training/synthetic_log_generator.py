"""Deterministic synthetic cropped well-log track generator.

The generator keeps an analytical centerline for every output row while
rendering the printable stroke, grid, document noise, and scan degradation as
separate layers. This makes difficult cases reproducible and measurable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticLogConfig:
    width: int = 512
    height: int = 1024
    centerline_radius: int = 1
    curve_shape: Optional[str] = None
    curve_color: Optional[str] = None
    enable_missing_sections: bool = True
    enable_dashed_curves: bool = True
    enable_geometric_distortion: bool = True
    enable_degradation: bool = True
    enable_text_fragments: bool = True
    maximum_distance: float = 16.0
    direction_radius: int = 5
    direction_tube_radius: int = 4
    hard_case_probability: float = 0.65
    wrap_mode: Optional[str] = None


CURVE_COLORS = {
    "black": (22, 22, 22),
    "red": (36, 42, 188),
    "blue": (170, 72, 35),
    "green": (58, 130, 54),
    "faded_gray": (132, 132, 132),
}


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    kernel = np.hanning(window).astype(np.float32)
    kernel /= max(float(kernel.sum()), 1e-8)
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _generate_centerline(rng: np.random.Generator, width: int, height: int, shape: str) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    margin = max(2.0, width * 0.025)
    center = float(rng.uniform(width * 0.28, width * 0.72))
    amplitude = float(rng.uniform(width * 0.08, width * 0.34))

    if shape == "sinusoidal":
        cycles = float(rng.uniform(0.8, 4.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        secondary = float(rng.uniform(0.12, 0.35))
        x = center + amplitude * np.sin(2.0 * np.pi * cycles * y + phase)
        x += amplitude * secondary * np.sin(2.0 * np.pi * (cycles * 3.1) * y + phase * 0.4)
    elif shape == "random_walk":
        steps = rng.normal(0.0, max(0.5, width * 0.012), size=height).astype(np.float32)
        x = center + _smooth(np.cumsum(steps), max(9, height // 32))
        x -= float(np.mean(x) - center)
    elif shape == "sharp_extrema":
        anchor_count = int(rng.integers(8, 18))
        anchor_y = np.linspace(0, height - 1, anchor_count)
        anchor_x = rng.uniform(margin, width - 1 - margin, size=anchor_count)
        x = np.interp(np.arange(height), anchor_y, anchor_x).astype(np.float32)
        x = _smooth(x, max(3, height // 160))
    elif shape == "vertical":
        drift = float(rng.uniform(-width * 0.08, width * 0.08))
        x = center + drift * y + _smooth(rng.normal(0.0, 0.7, height), 17)
    elif shape == "rapid":
        anchor_count = int(rng.integers(20, 36))
        anchor_y = np.linspace(0, height - 1, anchor_count)
        anchor_x = center + rng.normal(0.0, amplitude * 0.75, anchor_count)
        x = np.interp(np.arange(height), anchor_y, anchor_x).astype(np.float32)
        x = _smooth(x, 5)
    else:
        raise ValueError(f"Unsupported curve shape: {shape}")

    # Some curves deliberately touch a track border for a measurable hard case.
    border_touch = bool(rng.random() < 0.25)
    if border_touch:
        target = margin if rng.random() < 0.5 else width - 1 - margin
        row = int(rng.integers(max(1, height // 8), max(2, height * 7 // 8)))
        radius = max(8, height // 18)
        influence = np.exp(-0.5 * ((np.arange(height) - row) / radius) ** 2)
        x += (target - x[row]) * influence

    horizontal_distortion = float(rng.uniform(0.0, max(0.5, width * 0.012)))
    distortion_cycles = float(rng.uniform(0.5, 2.5))
    x += horizontal_distortion * np.sin(2.0 * np.pi * distortion_cycles * y)
    return np.clip(x, margin, width - 1 - margin).astype(np.float32)


def _generate_wrap_centerline(
    rng: np.random.Generator,
    width: int,
    height: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(height, dtype=np.float32)
    t = rows / max(1.0, float(height - 1))
    wobble = 0.04 * width * np.sin(2.0 * np.pi * (1.5 * t + float(rng.uniform(0, 1))))
    if mode == "right_to_left":
        unwrapped = width * (0.65 + 0.72 * t) + wobble
    elif mode == "left_to_right":
        unwrapped = width * (0.35 - 0.72 * t) + wobble
    elif mode == "multiple_positive":
        unwrapped = width * (0.18 + 2.45 * t) + wobble
    elif mode == "mixed":
        anchors_y = np.array([0.0, 0.28, 0.55, 0.78, 1.0])
        anchors_x = width * np.array([0.25, 1.18, -0.18, 0.92, 0.38])
        unwrapped = np.interp(t, anchors_y, anchors_x).astype(np.float32) + wobble
    elif mode == "turn_away":
        unwrapped = width * (0.30 + 0.65 * np.sin(np.pi * t))
    elif mode == "border_follow":
        unwrapped = width * (0.22 + 0.75 * np.minimum(1.0, t * 2.8))
        unwrapped[int(height * 0.35):int(height * 0.65)] = width * 0.97
        unwrapped[int(height * 0.65):] -= width * 0.55 * np.linspace(0, 1, height - int(height * 0.65))
    else:
        raise ValueError(f"Unsupported wrap mode: {mode}")
    wrap_index = np.floor(unwrapped / float(width)).astype(np.int32)
    visible = unwrapped - wrap_index.astype(np.float32) * float(width)
    visible = np.clip(visible, 0.0, width - 1.0).astype(np.float32)
    return visible, unwrapped.astype(np.float32), wrap_index


def _wrap_event_records(visible: np.ndarray, unwrapped: np.ndarray, wrap_index: np.ndarray) -> list[dict]:
    events = []
    for row in np.flatnonzero(np.diff(wrap_index) != 0) + 1:
        before = int(wrap_index[row - 1])
        after = int(wrap_index[row])
        events.append({
            "row_before": int(row - 1),
            "row_after": int(row),
            "direction": "right_to_left" if after > before else "left_to_right",
            "wrap_index_before": before,
            "wrap_index_after": after,
            "x_before": float(visible[row - 1]),
            "x_after": float(visible[row]),
            "unwrapped_delta": float(unwrapped[row] - unwrapped[row - 1]),
        })
    return events


def _draw_grid(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, Dict[str, Any]]:
    height, width = image.shape[:2]
    grid_mask = np.zeros((height, width), dtype=np.uint8)
    style = str(rng.choice(["linear", "logarithmic"]))
    color_name = str(rng.choice(["black", "gray", "colored"]))
    if color_name == "black":
        thin_color = (72, 72, 72)
    elif color_name == "colored":
        thin_color = (160, 132, 92)
    else:
        thin_color = (178, 178, 178)

    if style == "logarithmic":
        base = np.array([0.0, 0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954, 1.0])
        positions = sorted(set(int(round(v * (width - 1))) for v in base))
    else:
        spacing = int(rng.integers(max(12, width // 18), max(18, width // 8)))
        positions = list(range(0, width, spacing))

    major_every = int(rng.integers(3, 6))
    for index, x in enumerate(positions):
        major = index % major_every == 0
        thickness = 2 if major else 1
        cv2.line(image, (x, 0), (x, height - 1), thin_color, thickness, cv2.LINE_AA)
        cv2.line(grid_mask, (x, 0), (x, height - 1), 255, thickness, cv2.LINE_8)

    horizontal_spacing = int(rng.integers(max(14, height // 48), max(24, height // 20)))
    horizontal_major_every = int(rng.integers(4, 8))
    for index, y in enumerate(range(0, height, horizontal_spacing)):
        major = index % horizontal_major_every == 0
        thickness = 2 if major else 1
        cv2.line(image, (0, y), (width - 1, y), thin_color, thickness, cv2.LINE_AA)
        cv2.line(grid_mask, (0, y), (width - 1, y), 255, thickness, cv2.LINE_8)

    border_width = int(rng.integers(1, 4))
    cv2.line(image, (0, 0), (0, height - 1), (38, 38, 38), border_width)
    cv2.line(image, (width - 1, 0), (width - 1, height - 1), (38, 38, 38), border_width)
    cv2.line(grid_mask, (0, 0), (0, height - 1), 255, border_width)
    cv2.line(grid_mask, (width - 1, 0), (width - 1, height - 1), 255, border_width)
    return grid_mask, {
        "style": style,
        "color": color_name,
        "vertical_positions": positions,
        "major_every": major_every,
        "horizontal_spacing": horizontal_spacing,
        "horizontal_major_every": horizontal_major_every,
        "border_width": border_width,
    }


def _missing_rows(rng: np.random.Generator, height: int, enabled: bool) -> tuple[np.ndarray, list[dict]]:
    visible = np.ones(height, dtype=bool)
    sections: list[dict] = []
    if not enabled or rng.random() >= 0.55:
        return visible, sections
    count = int(rng.integers(1, 4))
    for _ in range(count):
        length = int(rng.integers(max(2, height // 100), max(4, height // 24)))
        start = int(rng.integers(0, max(1, height - length)))
        visible[start:start + length] = False
        sections.append({"start_row": start, "end_row": start + length - 1})
    return visible, sections


def _render_curve(
    image: np.ndarray,
    centerline_x: np.ndarray,
    rng: np.random.Generator,
    color_name: str,
    enable_missing: bool,
    enable_dashed: bool,
    grid_mask: np.ndarray,
    wrap_index: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    height, width = image.shape[:2]
    stroke_mask = np.zeros((height, width), dtype=np.uint8)
    visible_rows, missing_sections = _missing_rows(rng, height, enable_missing)
    dashed = bool(enable_dashed and rng.random() < 0.24)
    dash_on = int(rng.integers(3, 12))
    dash_off = int(rng.integers(2, 8))
    base_width = int(rng.integers(1, 5))
    width_wave = int(rng.integers(0, 3))
    color = CURVE_COLORS[color_name]
    alpha = float(rng.uniform(0.55, 1.0) if color_name == "faded_gray" else rng.uniform(0.78, 1.0))
    curve_layer = image.copy()

    for row in range(1, height):
        if not visible_rows[row] or not visible_rows[row - 1]:
            continue
        if dashed and row % (dash_on + dash_off) >= dash_on:
            continue
        if wrap_index is not None and int(wrap_index[row]) != int(wrap_index[row - 1]):
            continue
        line_width = max(1, base_width + int(round(width_wave * np.sin(row / max(4.0, height / 12.0)))))
        p0 = (int(round(centerline_x[row - 1])), row - 1)
        p1 = (int(round(centerline_x[row])), row)
        cv2.line(curve_layer, p0, p1, color, line_width, cv2.LINE_AA)
        cv2.line(stroke_mask, p0, p1, 255, line_width, cv2.LINE_AA)

    cv2.addWeighted(curve_layer, alpha, image, 1.0 - alpha, 0.0, dst=image)
    overprinted_grid_lines = []
    if rng.random() < 0.55:
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.65:
                position = int(rng.integers(0, height))
                cv2.line(image, (0, position), (width - 1, position), (74, 74, 74), int(rng.integers(1, 3)), cv2.LINE_AA)
                cv2.line(grid_mask, (0, position), (width - 1, position), 255, 2, cv2.LINE_8)
                overprinted_grid_lines.append({"orientation": "horizontal", "position": position})
            else:
                position = int(rng.integers(0, width))
                cv2.line(image, (position, 0), (position, height - 1), (74, 74, 74), int(rng.integers(1, 3)), cv2.LINE_AA)
                cv2.line(grid_mask, (position, 0), (position, height - 1), 255, 2, cv2.LINE_8)
                overprinted_grid_lines.append({"orientation": "vertical", "position": position})
    stroke_mask = np.where(stroke_mask > 16, 255, 0).astype(np.uint8)
    return stroke_mask, {
        "color": color_name,
        "bgr": list(color),
        "alpha": alpha,
        "base_width": base_width,
        "width_wave": width_wave,
        "dashed": dashed,
        "dash_on": dash_on,
        "dash_off": dash_off,
        "missing_sections": missing_sections,
        "overprinted_grid_lines": overprinted_grid_lines,
    }


def _add_document_artifacts(image: np.ndarray, rng: np.random.Generator, enabled: bool) -> Dict[str, Any]:
    height, width = image.shape[:2]
    artifact_meta: Dict[str, Any] = {"text": [], "spots": 0, "stamp": False}
    if enabled:
        fragments = ["100", "GR", "API", "FT", "CAL", "10", "0.2", "LOG"]
        text_count = int(rng.integers(0, 5))
        for _ in range(text_count):
            text = str(rng.choice(fragments))
            x = int(rng.integers(0, max(1, width - 32)))
            y = int(rng.integers(12, height))
            scale = float(rng.uniform(0.28, 0.58))
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (50, 50, 50), 1, cv2.LINE_AA)
            artifact_meta["text"].append({"text": text, "x": x, "y": y, "scale": scale})

    spot_count = int(rng.integers(0, 12))
    for _ in range(spot_count):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(1, max(2, min(width, height) // 70)))
        cv2.circle(image, center, radius, (int(rng.integers(20, 110)),) * 3, -1, cv2.LINE_AA)
    artifact_meta["spots"] = spot_count

    if enabled and rng.random() < 0.18:
        center = (int(rng.integers(width // 4, max(width // 4 + 1, width * 3 // 4))), int(rng.integers(height // 4, max(height // 4 + 1, height * 3 // 4))))
        axes = (max(5, width // 12), max(4, height // 40))
        cv2.ellipse(image, center, axes, float(rng.uniform(-20, 20)), 0, 360, (55, 55, 125), 2, cv2.LINE_AA)
        artifact_meta["stamp"] = True
    return artifact_meta


def _transform_centerline(centerline_x: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.column_stack((centerline_x, np.arange(height, dtype=np.float32), np.ones(height, dtype=np.float32)))
    transformed = points @ matrix.T
    order = np.argsort(transformed[:, 1])
    y_sorted = transformed[order, 1]
    x_sorted = transformed[order, 0]
    unique_y, unique_indices = np.unique(y_sorted, return_index=True)
    unique_x = x_sorted[unique_indices]
    if unique_y.size < 2:
        return np.clip(centerline_x, 0, width - 1).astype(np.float32)
    output = np.interp(np.arange(height), unique_y, unique_x, left=unique_x[0], right=unique_x[-1])
    return np.clip(output, 0, width - 1).astype(np.float32)


def _degrade_and_transform(
    image: np.ndarray,
    stroke_mask: np.ndarray,
    grid_mask: np.ndarray,
    centerline_x: np.ndarray,
    rng: np.random.Generator,
    config: SyntheticLogConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    height, width = image.shape[:2]
    metadata: Dict[str, Any] = {}

    vertical_amplitude = float(rng.uniform(-2.5, 2.5)) if config.enable_geometric_distortion else 0.0
    vertical_cycles = float(rng.uniform(0.5, 2.0))
    if abs(vertical_amplitude) > 1e-8:
        rows = np.arange(height, dtype=np.float32)
        source_y = np.clip(
            rows + vertical_amplitude * np.sin(2.0 * np.pi * vertical_cycles * rows / max(1, height - 1)),
            0,
            height - 1,
        ).astype(np.float32)
        map_y = np.repeat(source_y[:, None], width, axis=1)
        map_x = np.repeat(np.arange(width, dtype=np.float32)[None, :], height, axis=0)
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        stroke_mask = cv2.remap(stroke_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        grid_mask = cv2.remap(grid_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        centerline_x = np.interp(source_y, rows, centerline_x).astype(np.float32)
    metadata["vertical_distortion_amplitude"] = vertical_amplitude
    metadata["vertical_distortion_cycles"] = vertical_cycles

    angle = float(rng.uniform(-1.25, 1.25)) if config.enable_geometric_distortion else 0.0
    shear = float(rng.uniform(-0.008, 0.008)) if config.enable_geometric_distortion else 0.0
    matrix = cv2.getRotationMatrix2D(((width - 1) * 0.5, (height - 1) * 0.5), angle, 1.0).astype(np.float32)
    matrix[0, 1] += shear
    if abs(angle) > 1e-8 or abs(shear) > 1e-8:
        border = tuple(int(v) for v in np.median(image.reshape(-1, 3), axis=0))
        image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
        stroke_mask = cv2.warpAffine(stroke_mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        grid_mask = cv2.warpAffine(grid_mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        centerline_x = _transform_centerline(centerline_x, matrix, width, height)
    metadata["rotation_degrees"] = angle
    metadata["shear"] = shear

    if not config.enable_degradation:
        metadata.update({"blur_sigma": 0.0, "gaussian_noise_sigma": 0.0, "speckle_probability": 0.0, "jpeg_quality": 100, "ink_bleed": 0})
        return image, stroke_mask, grid_mask, centerline_x, metadata

    bleed_iterations = int(rng.integers(0, 3))
    if bleed_iterations:
        bleed = cv2.dilate(stroke_mask, np.ones((3, 3), np.uint8), iterations=bleed_iterations)
        image[bleed > 0] = np.minimum(image[bleed > 0], np.array([110, 110, 110], dtype=np.uint8))

    blur_sigma = float(rng.uniform(0.0, 1.2))
    if blur_sigma > 0.15:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)

    noise_sigma = float(rng.uniform(0.0, 10.0))
    noise = rng.normal(0.0, noise_sigma, image.shape).astype(np.float32)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    speckle_probability = float(rng.uniform(0.0, 0.006))
    speckle = rng.random((height, width)) < speckle_probability
    if speckle.any():
        image[speckle] = rng.integers(0, 100, size=(int(speckle.sum()), 1), dtype=np.uint8)

    jpeg_quality = int(rng.integers(55, 101))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if ok:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            image = decoded

    low_resolution_scale = 1.0
    resize_passes = 0
    if rng.random() < float(config.hard_case_probability) * 0.45:
        low_resolution_scale = float(rng.uniform(0.25, 0.65))
        small_w = max(8, int(round(width * low_resolution_scale)))
        small_h = max(8, int(round(height * low_resolution_scale)))
        image = cv2.resize(cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA), (width, height), interpolation=cv2.INTER_LINEAR)
        resize_passes = int(rng.integers(1, 4))
        for _ in range(resize_passes - 1):
            medium_w = max(8, int(round(width * rng.uniform(0.45, 0.85))))
            medium_h = max(8, int(round(height * rng.uniform(0.45, 0.85))))
            image = cv2.resize(cv2.resize(image, (medium_w, medium_h), interpolation=cv2.INTER_AREA), (width, height), interpolation=cv2.INTER_LINEAR)

    metadata.update({
        "blur_sigma": blur_sigma,
        "gaussian_noise_sigma": noise_sigma,
        "speckle_probability": speckle_probability,
        "jpeg_quality": jpeg_quality,
        "ink_bleed": bleed_iterations,
        "low_resolution_scale": low_resolution_scale,
        "resize_passes": resize_passes,
    })
    return image, stroke_mask, grid_mask, centerline_x, metadata


def _centerline_mask(
    centerline_x: np.ndarray,
    width: int,
    radius: int,
    wrap_index: Optional[np.ndarray] = None,
) -> np.ndarray:
    height = centerline_x.size
    mask = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, radius * 2 + 1)
    for row in range(height):
        x = int(np.clip(round(float(centerline_x[row])), 0, width - 1))
        cv2.circle(mask, (x, row), radius, 255, -1, cv2.LINE_8)
        if row > 0 and (wrap_index is None or int(wrap_index[row]) == int(wrap_index[row - 1])):
            previous_x = int(np.clip(round(float(centerline_x[row - 1])), 0, width - 1))
            cv2.line(mask, (previous_x, row - 1), (x, row), 255, thickness, cv2.LINE_8)
    return mask


def _geometry_targets(
    centerline_x: np.ndarray,
    width: int,
    maximum_distance: float,
    direction_radius: int,
    direction_tube_radius: int,
    unwrapped_centerline_x: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive smooth center distance and local tangent labels from the exact trace."""
    height = centerline_x.size
    columns = np.arange(width, dtype=np.float32)[None, :]
    distance = np.abs(columns - centerline_x[:, None])
    maximum_distance = max(1.0, float(maximum_distance))
    distance_field = np.clip(1.0 - distance / maximum_distance, 0.0, 1.0).astype(np.float32)

    radius = max(1, int(direction_radius))
    rows = np.arange(height)
    before = np.maximum(0, rows - radius)
    after = np.minimum(height - 1, rows + radius)
    direction_x = centerline_x if unwrapped_centerline_x is None else np.asarray(unwrapped_centerline_x, dtype=np.float32)
    dx = direction_x[after] - direction_x[before]
    dy = (after - before).astype(np.float32)
    magnitude = np.sqrt(dx * dx + dy * dy)
    magnitude = np.maximum(magnitude, 1e-6)
    row_direction = np.stack((dx / magnitude, dy / magnitude), axis=0).astype(np.float32)

    valid = distance <= max(0, int(direction_tube_radius))
    direction = np.zeros((2, height, width), dtype=np.float32)
    direction[0] = np.where(valid, row_direction[0, :, None], 0.0)
    direction[1] = np.where(valid, row_direction[1, :, None], 0.0)
    return distance_field, direction, (valid.astype(np.uint8) * 255)


def _draw_distractor_curves(image: np.ndarray, rng: np.random.Generator, count: int) -> list[dict]:
    height, width = image.shape[:2]
    distractors = []
    for index in range(max(0, int(count))):
        shape = str(rng.choice(["sinusoidal", "random_walk", "vertical", "rapid"]))
        trace = _generate_centerline(rng, width, height, shape)
        color_name = str(rng.choice(list(CURVE_COLORS)))
        color = CURVE_COLORS[color_name]
        thickness = int(rng.integers(1, 4))
        points = np.column_stack((np.rint(trace).astype(np.int32), np.arange(height, dtype=np.int32)))
        cv2.polylines(image, [points.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
        distractors.append({"index": index, "shape": shape, "color": color_name, "width": thickness})
    return distractors


def generate_sample(
    seed: int,
    config: Optional[SyntheticLogConfig] = None,
) -> Dict[str, Any]:
    """Generate one image and exact labels using only the supplied seed."""
    config = config or SyntheticLogConfig()
    width = int(config.width)
    height = int(config.height)
    if width < 16 or height < 16:
        raise ValueError("Synthetic samples must be at least 16x16 pixels")

    rng = np.random.default_rng(int(seed))
    paper_name = str(rng.choice(["white", "gray", "yellowed"]))
    paper = {
        "white": np.array([244, 246, 246], dtype=np.float32),
        "gray": np.array([222, 224, 224], dtype=np.float32),
        "yellowed": np.array([205, 224, 236], dtype=np.float32),
    }[paper_name]
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:] = paper
    brightness_x = np.linspace(float(rng.uniform(-18, 0)), float(rng.uniform(0, 18)), width, dtype=np.float32)
    brightness_y = np.linspace(float(rng.uniform(-12, 8)), float(rng.uniform(-8, 12)), height, dtype=np.float32)
    image += brightness_x[None, :, None] + brightness_y[:, None, None]
    image = np.clip(image, 0, 255).astype(np.uint8)

    grid_mask, grid_meta = _draw_grid(image, rng)
    shape = config.curve_shape or str(rng.choice(["sinusoidal", "random_walk", "sharp_extrema", "vertical", "rapid"]))
    hard_case = bool(rng.random() < float(config.hard_case_probability))
    color_choices = ["black", "black", "faded_gray", "red", "blue", "green"] if hard_case else list(CURVE_COLORS)
    color_name = config.curve_color or str(rng.choice(color_choices))
    wrap_mode = str(config.wrap_mode).lower().strip() if config.wrap_mode else None
    if wrap_mode:
        centerline_x, unwrapped_centerline_x, wrap_index_by_row = _generate_wrap_centerline(
            rng, width, height, wrap_mode
        )
        topology = "cylindrical"
    else:
        centerline_x = _generate_centerline(rng, width, height, shape)
        unwrapped_centerline_x = centerline_x.copy()
        wrap_index_by_row = np.zeros(height, dtype=np.int32)
        topology = "bounded"
    grid_follow = None
    if not wrap_mode and hard_case and grid_meta["vertical_positions"] and rng.random() < 0.42:
        grid_x = int(rng.choice(grid_meta["vertical_positions"]))
        length = int(rng.integers(max(8, height // 24), max(12, height // 8)))
        start = int(rng.integers(0, max(1, height - length)))
        centerline_x[start:start + length] = float(grid_x)
        centerline_x = _smooth(centerline_x, max(3, height // 256))
        grid_follow = {"x": grid_x, "start_row": start, "end_row": start + length - 1}
    stroke_mask, curve_meta = _render_curve(
        image,
        centerline_x,
        rng,
        color_name,
        config.enable_missing_sections,
        config.enable_dashed_curves,
        grid_mask,
        wrap_index_by_row,
    )
    distractor_count = int(rng.integers(1, 3)) if hard_case and rng.random() < 0.45 else 0
    distractors = _draw_distractor_curves(image, rng, distractor_count)
    artifact_meta = _add_document_artifacts(image, rng, config.enable_text_fragments)
    transform_config = replace(config, enable_geometric_distortion=False) if wrap_mode else config
    image, stroke_mask, grid_mask, centerline_x, degradation_meta = _degrade_and_transform(
        image, stroke_mask, grid_mask, centerline_x, rng, transform_config
    )
    if not wrap_mode:
        unwrapped_centerline_x = centerline_x.copy()
    center_mask = _centerline_mask(
        centerline_x, width, int(config.centerline_radius), wrap_index_by_row
    )
    distance_field, direction_field, valid_direction_mask = _geometry_targets(
        centerline_x,
        width,
        config.maximum_distance,
        config.direction_radius,
        config.direction_tube_radius,
        unwrapped_centerline_x,
    )
    wrap_events = _wrap_event_records(centerline_x, unwrapped_centerline_x, wrap_index_by_row)

    metadata = {
        "schema_version": 2,
        "dataset_version": "synthetic-v2",
        "topology": topology,
        "wrap_mode": wrap_mode,
        "wrap_events": wrap_events,
        "seed": int(seed),
        "width": width,
        "height": height,
        "paper": paper_name,
        "brightness_x_endpoints": [float(brightness_x[0]), float(brightness_x[-1])],
        "brightness_y_endpoints": [float(brightness_y[0]), float(brightness_y[-1])],
        "curve_shape": shape,
        "curve": curve_meta,
        "hard_case": hard_case,
        "grid_follow": grid_follow,
        "distractor_curves": distractors,
        "grid": grid_meta,
        "artifacts": artifact_meta,
        "degradation": degradation_meta,
        "config": asdict(config),
    }
    return {
        "image": image,
        "stroke_mask": stroke_mask,
        "centerline_mask": center_mask,
        "centerline_x_by_row": centerline_x.astype(np.float32),
        "visible_centerline_x_by_row": centerline_x.astype(np.float32),
        "unwrapped_centerline_x_by_row": unwrapped_centerline_x.astype(np.float32),
        "wrap_index_by_row": wrap_index_by_row.astype(np.int32),
        "wrap_events": wrap_events,
        "distance_field": distance_field,
        "direction_field": direction_field,
        "grid_mask": np.where(grid_mask > 0, 255, 0).astype(np.uint8),
        "valid_direction_mask": valid_direction_mask,
        "metadata": metadata,
    }


def write_dataset(output_dir: Path | str, count: int, config: SyntheticLogConfig, seed: int) -> Path:
    output = Path(output_dir)
    directories = {
        "images": output / "images",
        "stroke_masks": output / "stroke_masks",
        "centerline_masks": output / "centerline_masks",
        "centerlines": output / "centerlines",
        "unwrapped_centerlines": output / "unwrapped_centerlines",
        "wrap_indexes": output / "wrap_indexes",
        "metadata": output / "metadata",
        "distance_fields": output / "distance_fields",
        "direction_fields": output / "direction_fields",
        "grid_masks": output / "grid_masks",
        "valid_direction_masks": output / "valid_direction_masks",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for index in range(int(count)):
            sample_seed = int(seed) + index
            sample = generate_sample(sample_seed, config)
            stem = f"sample_{index:06d}"
            paths = {
                "image": directories["images"] / f"{stem}.png",
                "stroke_mask": directories["stroke_masks"] / f"{stem}.png",
                "centerline_mask": directories["centerline_masks"] / f"{stem}.png",
                "centerline_x": directories["centerlines"] / f"{stem}.npy",
                "unwrapped_centerline_x": directories["unwrapped_centerlines"] / f"{stem}.npy",
                "wrap_index": directories["wrap_indexes"] / f"{stem}.npy",
                "metadata": directories["metadata"] / f"{stem}.json",
                "distance_field": directories["distance_fields"] / f"{stem}.npy",
                "direction_field": directories["direction_fields"] / f"{stem}.npy",
                "grid_mask": directories["grid_masks"] / f"{stem}.png",
                "valid_direction_mask": directories["valid_direction_masks"] / f"{stem}.png",
            }
            cv2.imwrite(str(paths["image"]), sample["image"])
            cv2.imwrite(str(paths["stroke_mask"]), sample["stroke_mask"])
            cv2.imwrite(str(paths["centerline_mask"]), sample["centerline_mask"])
            np.save(paths["centerline_x"], sample["centerline_x_by_row"], allow_pickle=False)
            np.save(paths["unwrapped_centerline_x"], sample["unwrapped_centerline_x_by_row"], allow_pickle=False)
            np.save(paths["wrap_index"], sample["wrap_index_by_row"], allow_pickle=False)
            np.save(paths["distance_field"], sample["distance_field"], allow_pickle=False)
            np.save(paths["direction_field"], sample["direction_field"], allow_pickle=False)
            cv2.imwrite(str(paths["grid_mask"]), sample["grid_mask"])
            cv2.imwrite(str(paths["valid_direction_mask"]), sample["valid_direction_mask"])
            paths["metadata"].write_text(json.dumps(sample["metadata"], indent=2), encoding="utf-8")
            record = {
                "id": stem,
                "seed": sample_seed,
                "source": "synthetic",
                "hard_case": bool(sample["metadata"].get("hard_case")),
                "dataset_version": "synthetic-v2",
                "topology": sample["metadata"].get("topology", "bounded"),
                "wrap_mode": sample["metadata"].get("wrap_mode"),
            }
            record.update({key: str(path.relative_to(output)).replace("\\", "/") for key, path in paths.items()})
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic well-log track crops")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--wrap-mode",
        choices=("right_to_left", "left_to_right", "multiple_positive", "mixed", "turn_away", "border_follow"),
        default=None,
    )
    args = parser.parse_args()
    config = SyntheticLogConfig(width=args.width, height=args.height, wrap_mode=args.wrap_mode)
    manifest = write_dataset(args.output_dir, args.count, config, args.seed)
    print(json.dumps({"count": args.count, "manifest": str(manifest), "seed": args.seed}))


if __name__ == "__main__":
    main()
