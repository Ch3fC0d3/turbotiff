"""Deterministic well-log track layout inference from positioned OCR text."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


_CURVE_DEFINITIONS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "GR",
        "unit": "API",
        "scale_min": 0.0,
        "scale_max": 150.0,
        "aliases": (
            (r"\bGAMMA\s*RAY\b", 8),
            (r"\bG[A-Z0-9]{2,5}A?\s+RAY\b", 7),
            (r"\bAPI\s+UNITS?\b", 7),
            (r"\bGAMMA\b", 6),
            (r"\bGR\b", 5),
            (r"\bG\s*R\b", 4),
        ),
    },
    {
        "name": "RHOB",
        "unit": "G/CC",
        "scale_min": 1.95,
        "scale_max": 2.95,
        "aliases": (
            (r"\bBULK\s*DENSITY\b", 8),
            (r"\bGRAMS?\s+(?:PER|/)\s+CUBIC\b", 8),
            (r"\bG\s*/\s*C{1,2}\b", 7),
            (r"\bRHOB\b", 7),
            (r"\bDENSITY\b", 6),
            (r"\bDENS\b", 5),
        ),
    },
    {
        "name": "NPHI",
        "unit": "V/V",
        "scale_min": 0.45,
        "scale_max": -0.15,
        "aliases": (
            (r"\bNEUTRON\s*POROSITY\b", 8),
            (r"\bNPHI\b", 7),
            (r"\bNEUTRON\b", 6),
            (r"\bNEUT\b", 5),
        ),
    },
    {
        "name": "DT",
        "unit": "US/F",
        "scale_min": 40.0,
        "scale_max": 140.0,
        "aliases": (
            (r"\bINTERVAL\s*TRANSIT\s*TIME\b", 9),
            (r"\bTRANSIT\s*TIME\b", 8),
            (r"\b[A-Z]*CONDS\s+(?:PER|PTR|FTR)\s+FOO[TI]\b", 8),
            (r"\bDTCO\b", 7),
            (r"\bDTC\b", 7),
            (r"\bSONIC\b", 6),
            (r"\bACOUSTIC\b", 6),
            (r"\bDT\b", 5),
        ),
    },
    {
        "name": "CALI",
        "unit": "IN",
        "scale_min": 6.0,
        "scale_max": 16.0,
        "aliases": (
            (r"\bCALIPER\b", 7),
            (r"\bCALI\b", 6),
        ),
    },
    {
        "name": "SP",
        "unit": "MV",
        "scale_min": -100.0,
        "scale_max": 100.0,
        "aliases": (
            (r"\bSPONTANEOUS\s*POTENTIAL\b", 8),
            (r"\bSP\b", 5),
        ),
    },
    {
        "name": "RT",
        "unit": "OHMM",
        "scale_min": 0.2,
        "scale_max": 2000.0,
        "aliases": (
            (r"\bDEEP\s*RESISTIVITY\b", 8),
            (r"\bRESISTIVITY\b", 7),
            (r"\bILD\b", 6),
            (r"\bLLD\b", 6),
            (r"\bRT\b", 5),
        ),
    },
)

_INVENTORY_TERMS = re.compile(
    r"\b(?:CALIBRATION|CODING|RECORD|CART|CARTRIDGE|SERIAL|S\s*/?\s*N|MODEL|TOOL|REEL)\b",
    flags=re.IGNORECASE,
)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalized_text(value: Any) -> str:
    text = str(value or "").upper().translate(str.maketrans({"0": "O", "1": "I", "5": "S"}))
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _best_curve_match(text: str) -> Optional[Tuple[Dict[str, Any], int]]:
    best: Optional[Tuple[Dict[str, Any], int]] = None
    inventory_penalty = 6 if _INVENTORY_TERMS.search(text) else 0
    for definition in _CURVE_DEFINITIONS:
        for pattern, base_score in definition["aliases"]:
            if not re.search(pattern, text, flags=re.IGNORECASE):
                continue
            score = int(base_score) - inventory_penalty
            if best is None or score > best[1]:
                best = (definition, score)
    return best


def infer_tracks_from_ocr(
    items: Iterable[Dict[str, Any]],
    image_width: Any,
    max_tracks: int = 6,
) -> List[Dict[str, Any]]:
    """Infer named track bands from OCR labels without requiring an LLM."""
    width = _finite(image_width)
    if width is None or width < 4:
        return []

    candidates: Dict[str, Dict[str, Any]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = _normalized_text(item.get("text"))
        x = _finite(item.get("x"))
        if not text or x is None or not 0 <= x <= width:
            continue
        match = _best_curve_match(text)
        if match is None:
            continue
        definition, score = match
        if score < 3:
            continue
        current = candidates.get(definition["name"])
        candidate = {
            "definition": definition,
            "score": score,
            "x": x,
            "text": text,
        }
        if current is None or (score, len(text)) > (current["score"], len(current["text"])):
            candidates[definition["name"]] = candidate

    ordered = sorted(candidates.values(), key=lambda item: item["x"])
    if not ordered:
        return []

    # Multiple labels at nearly the same X generally describe one combined track.
    deduped: List[Dict[str, Any]] = []
    minimum_gap = max(24.0, width * 0.025)
    for candidate in ordered:
        if deduped and candidate["x"] - deduped[-1]["x"] < minimum_gap:
            if candidate["score"] > deduped[-1]["score"]:
                deduped[-1] = candidate
            continue
        deduped.append(candidate)
    ordered = deduped[:max(1, int(max_tracks))]

    centers = [float(item["x"]) for item in ordered]
    if len(centers) == 1:
        half_span = max(width * 0.18, 80.0)
        boundaries = [max(0.0, centers[0] - half_span), min(width, centers[0] + half_span)]
    else:
        midpoints = [(centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1)]
        left_edge = max(0.0, centers[0] - (centers[1] - centers[0]) / 2.0)
        right_edge = min(width, centers[-1] + (centers[-1] - centers[-2]) / 2.0)
        boundaries = [left_edge, *midpoints, right_edge]

    tracks: List[Dict[str, Any]] = []
    for index, candidate in enumerate(ordered):
        definition = candidate["definition"]
        left_x = max(0.0, min(width, boundaries[index]))
        right_x = max(0.0, min(width, boundaries[index + 1]))
        if right_x - left_x < 3.0:
            continue
        tracks.append({
            "name": definition["name"],
            "left_x": left_x,
            "right_x": right_x,
            "scale_min": definition["scale_min"],
            "scale_max": definition["scale_max"],
            "unit": definition["unit"],
            "hot_side": "right" if definition["scale_max"] >= definition["scale_min"] else "left",
            "color_hint": None,
            "source": "local_ocr",
        })
    return tracks
