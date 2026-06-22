"""
Scale detection + pixel-to-value conversion for well log tracks.

A "track" (or "panel") on a paper well log plots a curve across an X-axis that
can be:
    * linear          e.g. GR 0..150
    * centered linear e.g. SP -80..+20 with 0 at midpoint
    * logarithmic     e.g. resistivity 0.2..2000 over 4 log decades
    * wrapped         e.g. resistivity that exceeds the right edge and reappears on the left

This module provides:
    1. `classify_curve_type(name)`          -> default scale_type/unit for a curve name
    2. `classify_scale_from_labels(labels)` -> detect linear / log / centered from OCR-extracted numeric labels
    3. `pixel_to_value(xs, ...)`            -> scale-aware conversion of pixel x-positions to real values
    4. `detect_scale(track_header, labels, name)` -> combined helper returning a DetectedScale dataclass

The heavy OCR is delegated to the existing `detect_text_vision_api` in web_app.py;
this module only takes already-extracted numeric labels.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict, field
from typing import Iterable, List, Optional, Sequence

import numpy as np


# ────────────────────────────────────────────────────────────────────
# Curve type registry
# ────────────────────────────────────────────────────────────────────

# (aliases tuple, canonical mnemonic, default scale_type, default min, default max, unit)
_CURVE_REGISTRY = [
    (("GR", "GAMMA", "GAMMARAY", "GAMMA RAY"),                  "GR",   "linear",   0.0,   150.0, "API"),
    (("SP", "SPONTANEOUS", "SPONTANEOUS POTENTIAL"),            "SP",   "centered", -80.0, 20.0,  "MV"),
    (("ILD", "ILM", "RILD", "RT", "RES", "RESISTIVITY",
      "LLD", "LLS", "MSFL", "LLM", "RD", "RS", "INDUCTION",
      # Array / dual induction resistivity at specific depths of investigation.
      # Common vendor naming: RO/RT/AT/A-prefix with 10/20/30/60/90 (inches).
      "RO10", "RO20", "RO30", "RO60", "RO90",
      "RT10", "RT20", "RT30", "RT60", "RT90",
      "AT10", "AT20", "AT30", "AT60", "AT90",
      "AO10", "AO20", "AO30", "AO60", "AO90",
      "A10",  "A20",  "A30",  "A60",  "A90",
      "R10",  "R20",  "R30",  "R60",  "R90",
      # Generic shallow/medium/deep variants
      "RSHA", "RMED", "RDEP"),                                  "RT",   "log",      0.2,   2000.0, "OHMM"),
    (("RHOB", "DENS", "DENSITY", "RHO"),                        "RHOB", "linear",   1.95,  2.95,  "G/CC"),
    (("NPHI", "PHI", "NEUTRON", "TNPH", "NPOR"),                "NPHI", "linear",   0.45,  -0.15, "V/V"),  # note reversed: hot_side=left
    (("DT",  "DTC", "SONIC", "AC"),                             "DT",   "linear",   140.0, 40.0,  "US/F"), # reversed
    (("CALI", "CAL", "CALIPER"),                                "CALI", "linear",   6.0,   16.0,  "IN"),
    (("PE", "PEF"),                                             "PEF",  "linear",   0.0,   10.0,  "B/E"),
]


def classify_curve_type(name: str) -> Optional[dict]:
    """Return default metadata for a curve given its header text/name.

    Returns None if the name doesn't match any known curve.
    """
    if not name:
        return None
    n = re.sub(r"[^A-Z0-9]", "", name.upper())
    for aliases, mnemonic, scale_type, lo, hi, unit in _CURVE_REGISTRY:
        for a in aliases:
            if re.sub(r"[^A-Z0-9]", "", a) == n:
                return {
                    "mnemonic": mnemonic,
                    "scale_type": scale_type,
                    "default_left": lo,
                    "default_right": hi,
                    "unit": unit,
                }
    # Fuzzy "contains" pass for messy OCR like "GR API" or "RES-DEEP"
    for aliases, mnemonic, scale_type, lo, hi, unit in _CURVE_REGISTRY:
        for a in aliases:
            a_norm = re.sub(r"[^A-Z0-9]", "", a)
            if a_norm and a_norm in n:
                return {
                    "mnemonic": mnemonic,
                    "scale_type": scale_type,
                    "default_left": lo,
                    "default_right": hi,
                    "unit": unit,
                }
    return None


def get_mnemonic_defaults(mnemonic: str) -> Optional[dict]:
    """Lookup defaults by canonical mnemonic (GR, SP, RT, RHOB, NPHI, DT, CALI, PEF)."""
    if not mnemonic:
        return None
    m = str(mnemonic).upper().strip()
    for aliases, canonical, scale_type, lo, hi, unit in _CURVE_REGISTRY:
        if canonical == m:
            return {
                "mnemonic": canonical,
                "scale_type": scale_type,
                "default_left": lo,
                "default_right": hi,
                "unit": unit,
            }
    return None


# ────────────────────────────────────────────────────────────────────
# Label parsing & linear/log classification
# ────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_numeric_labels(ocr_text: Iterable[str]) -> List[float]:
    """Pull numeric values out of a sequence of OCR strings.

    Accepts values like '0.2', '-80', '+40', '2000', '1,000' (commas stripped).
    """
    out: List[float] = []
    for s in ocr_text:
        if not s:
            continue
        cleaned = str(s).replace(",", "")
        for m in _NUM_RE.findall(cleaned):
            try:
                out.append(float(m))
            except ValueError:
                continue
    return out


@dataclass
class ScaleClassification:
    scale_type: str            # "linear" | "centered" | "log"
    left_value: Optional[float]
    right_value: Optional[float]
    confidence: float          # 0..1
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def classify_scale_from_labels(labels: Sequence[float]) -> ScaleClassification:
    """Given numeric labels found on a track axis, decide if the scale is
    linear, centered (around zero), or logarithmic.

    Rules:
        * >=2 positive labels whose ratios are roughly constant (~10x, ~2x, etc.)
          and whose min is >0.01 -> log
        * labels span both negative and positive with 0 near midpoint -> centered
        * labels with roughly constant differences -> linear
    """
    vals = sorted(set(v for v in labels if math.isfinite(v)))
    if len(vals) < 2:
        return ScaleClassification("linear", None, None, 0.0, "too few labels")

    lo, hi = vals[0], vals[-1]

    # ── Log detection ──
    positives = [v for v in vals if v > 0]
    if len(positives) >= 3 and all(v > 0 for v in vals):
        ratios = []
        for a, b in zip(positives, positives[1:]):
            if a > 0 and b > 0:
                ratios.append(b / a)
        if ratios:
            log_ratios = [math.log10(r) for r in ratios if r > 0]
            if log_ratios:
                mean = float(np.mean(log_ratios))
                std = float(np.std(log_ratios))
                # Accept if variation is small relative to the mean step
                if mean > 0.2 and std < 0.25 * max(0.3, mean):
                    return ScaleClassification(
                        "log", lo, hi,
                        confidence=float(np.clip(1.0 - std / max(0.1, mean), 0.3, 0.99)),
                        reason=f"log ratios mean={mean:.2f} std={std:.2f}",
                    )

    # ── Centered detection ──
    if lo < 0 < hi:
        midpoint = (lo + hi) / 2.0
        span = hi - lo
        if span > 0 and abs(midpoint) / span < 0.15:
            return ScaleClassification(
                "centered", lo, hi,
                confidence=0.8,
                reason=f"spans negative and positive; midpoint≈0",
            )

    # ── Default to linear ──
    diffs = [b - a for a, b in zip(vals, vals[1:]) if (b - a) > 0]
    if diffs:
        mean = float(np.mean(diffs))
        std = float(np.std(diffs))
        conf = float(np.clip(1.0 - (std / max(1e-6, mean)), 0.3, 0.95))
    else:
        conf = 0.5
    return ScaleClassification("linear", lo, hi, conf, "constant differences")


# ────────────────────────────────────────────────────────────────────
# Scale-aware pixel → value conversion
# ────────────────────────────────────────────────────────────────────

def unwrap_log_trace(
    xs: np.ndarray,
    width_px: int,
    jump_frac: float = 0.7,
) -> np.ndarray:
    """Assign an integer wrap index to each row of a pixel trace.

    A resistivity curve on a log scale that exceeds the right-edge value (e.g.
    > 20 ohm-m on a 0.2→20 track) is drawn by wrapping: the line exits the
    right edge and re-enters on the left, now representing the next decade up.

    This scanner walks xs top-to-bottom and increments/decrements an integer
    wrap count whenever consecutive valid samples jump by more than
    ``jump_frac * width`` across the track. The returned array has the same
    shape as xs with NaN for rows where xs is NaN.

    Wrap convention (for tracks where right_value > left_value):
      - jump from near-right to near-left → wrap += 1 (next decade UP)
      - jump from near-left to near-right → wrap -= 1 (previous decade DOWN)

    Callers should ALSO receive the raw xs and apply the in-track log
    interpolation themselves; this function only reports the wrap index.
    """
    xs = np.asarray(xs, dtype=np.float64)
    w = max(1, int(width_px) - 1)
    wraps = np.full(xs.shape, np.nan, dtype=np.float64)
    threshold = float(w) * float(jump_frac)

    current = 0
    prev_x = None
    for i, x in enumerate(xs):
        if not np.isfinite(x):
            prev_x = None
            continue
        if prev_x is not None:
            dx = x - prev_x
            if dx < -threshold:
                # Big move leftward between adjacent rows → wrapped past right edge
                current += 1
            elif dx > threshold:
                # Big move rightward → came back down a decade
                current -= 1
        wraps[i] = current
        prev_x = x

    return wraps


def pixel_to_value(
    xs: np.ndarray,
    width_px: int,
    left_value: float,
    right_value: float,
    scale_type: str = "linear",
    wrapped: bool = False,
) -> np.ndarray:
    """Convert pixel x-positions (0..width_px-1) to engineering values.

    Args:
        xs: float array of x-positions (NaN allowed for missing samples)
        width_px: width of the track in pixels
        left_value: value at x=0
        right_value: value at x=width_px-1
        scale_type: "linear" | "log" | "centered"
        wrapped: if True AND scale_type=="log", detect multi-decade wraps in
                 the trace and multiply values by (right/left)^wrap_count so
                 a curve that wrapped once reads at the next decade.

    Returns a float array same shape as xs with NaN preserved.
    """
    xs = np.asarray(xs, dtype=np.float64)
    out = np.full(xs.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(xs)
    if not valid.any():
        return out

    w = max(1, int(width_px) - 1)
    x = xs[valid].copy()
    # Clamp pixel x to the track for the in-track fraction; wrap count (if any)
    # provides the between-decade multiplier.
    x_clamped = np.clip(x, 0.0, float(w))
    frac = x_clamped / w

    st = (scale_type or "linear").lower()

    if st == "log":
        lv = float(left_value)
        rv = float(right_value)
        if lv <= 0 or rv <= 0:
            # Fallback to linear if log mapping impossible
            out[valid] = lv + frac * (rv - lv)
        else:
            log_lo = math.log10(lv)
            log_hi = math.log10(rv)
            base_vals = np.power(10.0, log_lo + frac * (log_hi - log_lo))
            if wrapped:
                # Compute wrap index on the ORIGINAL (ordered) xs so the
                # row-to-row jump detection is faithful, then subset to valid.
                wrap_full = unwrap_log_trace(xs, width_px)
                wrap_valid = wrap_full[valid]
                wrap_valid = np.where(np.isnan(wrap_valid), 0.0, wrap_valid)
                decade_factor = rv / lv  # e.g. 20/0.2 = 100 for a 2-decade track
                base_vals = base_vals * np.power(decade_factor, wrap_valid)
            out[valid] = base_vals

    elif st == "centered":
        # Linear mapping, but treat the midpoint of the track as the
        # midpoint of [left_value, right_value]. This is equivalent to
        # plain linear in the simple case, but we expose the intent so
        # callers can pass wrapped=False and still get 0 at center when
        # left/right are symmetric around 0.
        out[valid] = float(left_value) + frac * (float(right_value) - float(left_value))

    else:
        # Linear (default)
        out[valid] = float(left_value) + frac * (float(right_value) - float(left_value))

    return out.astype(np.float32)


# ────────────────────────────────────────────────────────────────────
# Wrap detection (heuristic)
# ────────────────────────────────────────────────────────────────────

def detect_wrap(xs: np.ndarray, width_px: int, jump_frac: float = 0.7) -> bool:
    """Heuristic: if the trace repeatedly jumps across a large fraction of
    the track width between adjacent rows, it is likely wrapping.

    Returns True if a majority of large jumps alternate sides, which is
    characteristic of resistivity wrapping.
    """
    x = np.asarray(xs, dtype=np.float64)
    valid = ~np.isnan(x)
    if valid.sum() < 10 or width_px <= 1:
        return False
    xv = x[valid]
    jumps = np.diff(xv)
    threshold = float(width_px) * jump_frac
    big = np.abs(jumps) > threshold
    if big.sum() < 3:
        return False
    # Alternating sign → characteristic of wrap
    signs = np.sign(jumps[big])
    flips = np.sum(signs[:-1] * signs[1:] < 0)
    return flips >= max(2, int(0.6 * (signs.size - 1)))


# ────────────────────────────────────────────────────────────────────
# Combined detector
# ────────────────────────────────────────────────────────────────────

@dataclass
class DetectedScale:
    curve_name: Optional[str]
    mnemonic: Optional[str]
    unit: Optional[str]
    scale_type: str             # linear | log | centered
    left_value: Optional[float]
    right_value: Optional[float]
    wrapped: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ────────────────────────────────────────────────────────────────────
# Depth-axis detection
# ────────────────────────────────────────────────────────────────────

@dataclass
class DepthLabel:
    value: float
    y_px: int
    raw_text: str


@dataclass
class DetectedDepthAxis:
    top_depth: Optional[float]
    bottom_depth: Optional[float]
    top_px: Optional[int]
    bottom_px: Optional[int]
    unit: Optional[str]            # "FT" | "M" | None
    linear: bool
    r_squared: float               # goodness of linear fit (0..1)
    labels: List[dict]             # list of {value, y_px, raw_text} with fit errors
    confidence: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "top_depth":    self.top_depth,
            "bottom_depth": self.bottom_depth,
            "top_px":       self.top_px,
            "bottom_px":    self.bottom_px,
            "unit":         self.unit,
            "linear":       self.linear,
            "r_squared":    self.r_squared,
            "labels":       self.labels,
            "confidence":   self.confidence,
            "reasons":      self.reasons,
        }


def detect_depth_axis(
    ocr_numbers: Sequence[dict],
    panel_height_px: int,
    unit_hint: Optional[str] = None,
) -> DetectedDepthAxis:
    """Fit a depth axis given OCR-extracted numeric entries from the depth column.

    Args:
        ocr_numbers: list of dicts, each with keys 'value' (float) and 'y' (int pixel row)
        panel_height_px: total height of the panel in pixels
        unit_hint: optional 'FT' or 'M' hint; else inferred from label spacing

    Returns DetectedDepthAxis with linear-fit quality score and optional unit detection.
    """
    reasons: List[str] = []

    # Keep only plausible depth values (0 < v < 50000) and distinct y rows
    cleaned: List[DepthLabel] = []
    seen_y = set()
    for entry in ocr_numbers or []:
        try:
            v = float(entry.get("value"))
            y = int(entry.get("y"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= 0 or v > 50000:
            continue
        if y in seen_y:
            continue
        seen_y.add(y)
        cleaned.append(DepthLabel(value=v, y_px=y, raw_text=str(entry.get("text", v))))

    if len(cleaned) < 2:
        return DetectedDepthAxis(
            top_depth=None, bottom_depth=None, top_px=None, bottom_px=None,
            unit=unit_hint, linear=False, r_squared=0.0, labels=[],
            confidence=0.0, reasons=["fewer than 2 depth labels found"],
        )

    # Sort by pixel row (top to bottom)
    cleaned.sort(key=lambda e: e.y_px)

    # Robust linear fit: y_px = a * value + b  →  value = (y - b) / a
    ys = np.array([e.y_px for e in cleaned], dtype=np.float64)
    vs = np.array([e.value for e in cleaned], dtype=np.float64)

    # If values decrease as y increases (unusual orientation), the math still works
    # because numpy.polyfit handles negative slope.
    try:
        coeffs = np.polyfit(vs, ys, 1)  # y = slope*value + intercept
        slope, intercept = float(coeffs[0]), float(coeffs[1])
    except Exception:
        return DetectedDepthAxis(
            top_depth=None, bottom_depth=None, top_px=None, bottom_px=None,
            unit=unit_hint, linear=False, r_squared=0.0, labels=[],
            confidence=0.0, reasons=["polyfit failed"],
        )

    # Compute R² of the linear fit
    y_pred = slope * vs + intercept
    ss_res = float(np.sum((ys - y_pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
    r2 = float(max(0.0, min(1.0, r2)))

    linear = r2 >= 0.97
    if not linear:
        reasons.append(f"R²={r2:.3f} below 0.97 — depth scale may be non-linear")
    else:
        reasons.append(f"linear fit R²={r2:.3f}")

    # Extrapolate to top / bottom of panel
    if abs(slope) < 1e-9:
        return DetectedDepthAxis(
            top_depth=None, bottom_depth=None, top_px=None, bottom_px=None,
            unit=unit_hint, linear=False, r_squared=r2, labels=[],
            confidence=0.0, reasons=reasons + ["degenerate slope"],
        )

    # Solve for depth at y=0 (top) and y=panel_height_px-1 (bottom)
    top_depth = (0 - intercept) / slope
    bottom_depth = (panel_height_px - 1 - intercept) / slope

    # Infer unit from label spacing if no hint
    unit = unit_hint
    if not unit:
        # Compute typical spacing between consecutive labels in depth units
        sorted_vs = sorted(set(v for v in vs if v > 0))
        if len(sorted_vs) >= 2:
            spacings = [b - a for a, b in zip(sorted_vs, sorted_vs[1:]) if b > a]
            if spacings:
                med_spacing = float(np.median(spacings))
                # Typical well-log label spacings:
                #   Feet: 10, 25, 50, 100, 250 ft
                #   Meters: 5, 10, 25, 50 m
                # If values are in multiples of 10 and go above ~500, more likely feet
                max_v = max(sorted_vs)
                if max_v > 3000:
                    unit = "FT"
                    reasons.append("unit=FT (max value > 3000)")
                elif med_spacing < 10 and max_v < 500:
                    unit = "M"
                    reasons.append("unit=M (small spacing, small max)")
                else:
                    unit = "FT"  # default for oilfield logs
                    reasons.append(f"unit=FT (default; max={max_v:.0f}, spacing={med_spacing:.1f})")

    # Build label residuals for UI display
    label_dicts = []
    for lbl in cleaned:
        predicted_y = slope * lbl.value + intercept
        residual = float(lbl.y_px - predicted_y)
        label_dicts.append({
            "value":    lbl.value,
            "y_px":     lbl.y_px,
            "raw_text": lbl.raw_text,
            "residual_px": residual,
            "ok": abs(residual) < max(5.0, 0.01 * panel_height_px),
        })

    # Confidence: blend of R² and label count
    n_factor = min(1.0, len(cleaned) / 5.0)
    confidence = float(0.5 * r2 + 0.5 * n_factor)
    if not linear:
        confidence *= 0.6

    return DetectedDepthAxis(
        top_depth=float(top_depth),
        bottom_depth=float(bottom_depth),
        top_px=0,
        bottom_px=int(panel_height_px - 1),
        unit=unit,
        linear=linear,
        r_squared=r2,
        labels=label_dicts,
        confidence=confidence,
        reasons=reasons,
    )


# ────────────────────────────────────────────────────────────────────
# Combined detector
# ────────────────────────────────────────────────────────────────────

def detect_scale(
    header_text: Optional[str],
    axis_labels: Sequence[str],
    xs_trace: Optional[np.ndarray] = None,
    width_px: int = 0,
) -> DetectedScale:
    """Run the full detection pipeline for one track.

    Args:
        header_text: OCR text found in the track header (e.g. "GR API 0-150")
        axis_labels: OCR text strings near the top/bottom axis labels
        xs_trace:    optional already-traced pixel x's (for wrap detection)
        width_px:    panel width, for wrap detection

    Returns a DetectedScale. Callers should treat fields with low confidence
    as suggestions to surface in the manual-override UI.
    """
    reasons: List[str] = []
    curve_hint = classify_curve_type(header_text or "")
    if curve_hint:
        reasons.append(f"header matches curve '{curve_hint['mnemonic']}'")

    nums = parse_numeric_labels(axis_labels)
    classification = classify_scale_from_labels(nums)

    # Prefer OCR-derived scale if confident; otherwise fall back to curve default.
    if classification.confidence >= 0.6 and classification.left_value is not None:
        scale_type = classification.scale_type
        lv = classification.left_value
        rv = classification.right_value
        reasons.append(f"OCR labels → {scale_type}")
    elif curve_hint:
        scale_type = curve_hint["scale_type"]
        lv = curve_hint["default_left"]
        rv = curve_hint["default_right"]
        reasons.append(f"using defaults for {curve_hint['mnemonic']}")
    else:
        scale_type = classification.scale_type
        lv = classification.left_value
        rv = classification.right_value
        reasons.append("no curve hint; using label-derived scale")

    wrapped = False
    if xs_trace is not None and width_px > 1 and scale_type == "log":
        # Wrap is mostly a resistivity-log phenomenon
        wrapped = detect_wrap(xs_trace, width_px)
        if wrapped:
            reasons.append("detected alternating large jumps → wrapped")

    # Confidence is the min of the two components we trust
    conf_parts = [classification.confidence]
    if curve_hint:
        conf_parts.append(0.7)
    confidence = float(min(conf_parts)) if conf_parts else 0.3

    return DetectedScale(
        curve_name=(header_text or "").strip() or None,
        mnemonic=(curve_hint or {}).get("mnemonic"),
        unit=(curve_hint or {}).get("unit"),
        scale_type=scale_type,
        left_value=lv,
        right_value=rv,
        wrapped=wrapped,
        confidence=confidence,
        reasons=reasons,
    )
