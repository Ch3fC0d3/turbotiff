"""Configuration for bounded and cylindrical slope-state decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DecoderConfig:
    topology: str = "bounded"
    max_step: int = 12
    max_slope: int = 12
    slope_bins: int = 25
    nonlinear_slope_bins: bool = False
    step_weight: float = 0.015
    curvature_weight: float = 0.08
    direction_weight: float = 0.20
    centerline_weight: float = 0.45
    distance_weight: float = 0.30
    stroke_weight: float = 0.20
    classic_weight: float = 0.20
    grid_weight: float = 0.15
    grid_overlap_relief: float = 0.75
    wrap_penalty: float = 0.35
    wrap_evidence_weight: float = 0.50
    reverse_wrap_penalty: float = 0.50
    edge_transition_width: int = 8
    maximum_wrap_count: int = 4
    minimum_rows_between_wraps: int = 8
    beam_width: int | None = 96
    diversity_radius: int = 4
    states_per_diversity_bucket: int = 3
    low_confidence_threshold: float = 0.35
    rendering_discontinuity: float = 24.0
    allow_legacy_fallback: bool = True
    maximum_candidate_states: int = 200000

    def validate(self, width: int, height: int) -> None:
        if self.topology not in {"bounded", "cylindrical"}:
            raise ValueError(f"Invalid topology: {self.topology}")
        if int(width) < 1 or int(height) < 1:
            raise ValueError("Decoder evidence must have positive dimensions")
        if int(self.max_step) < 0 or int(self.max_slope) < 0:
            raise ValueError("Step and slope limits must be non-negative")
        if int(self.max_slope) > int(self.max_step):
            raise ValueError("max_slope cannot exceed max_step")
        if int(self.slope_bins) < 1 or (self.beam_width is not None and int(self.beam_width) < 1):
            raise ValueError("slope_bins and beam_width must be positive")
        if int(self.maximum_wrap_count) < 0:
            raise ValueError("maximum_wrap_count must be non-negative")
        if self.topology == "cylindrical" and int(self.edge_transition_width) < 1:
            raise ValueError("Cylindrical topology requires a positive edge transition width")
        beam = int(self.beam_width) if self.beam_width is not None else int(width) * min(2 * int(self.maximum_wrap_count) + 1, 3)
        candidates = beam * min(int(self.slope_bins), 2 * int(self.max_slope) + 1)
        if candidates > int(self.maximum_candidate_states):
            raise ValueError("Decoder configuration exceeds the candidate-state safeguard")

    def to_dict(self) -> dict:
        return asdict(self)

    def slope_values(self):
        import numpy as np

        maximum = min(int(self.max_step), int(self.max_slope))
        if maximum == 0:
            return np.array([0], dtype=np.int16)
        count = min(max(1, int(self.slope_bins)), 2 * maximum + 1)
        if self.nonlinear_slope_bins:
            base = np.linspace(-1.0, 1.0, count)
            values = np.rint(np.sign(base) * np.square(np.abs(base)) * maximum)
        else:
            values = np.rint(np.linspace(-maximum, maximum, count))
        values = np.unique(np.concatenate((values.astype(np.int16), np.array([0], dtype=np.int16))))
        return values[np.abs(values) <= int(self.max_step)]
