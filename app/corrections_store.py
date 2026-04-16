"""User-correction store for scale detection.

Each time the user overrides an AI-detected scale, we record:
  - the OCR header text + axis labels the AI saw
  - what the AI suggested vs. what the user chose
  - timestamp + optional user id

On subsequent detections we run a fast nearest-neighbor lookup against the
store. If a very similar header+axis pattern has been corrected before, we
surface the user's prior choice as a higher-confidence suggestion. This lets
the system adapt to a particular log vintage / vendor format without needing
model retraining.

Storage: single SQLite DB at data/corrections.db. Schema is intentionally
small and human-readable.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence, Tuple


# ────────────────────────────────────────────────────────────────────
# Storage
# ────────────────────────────────────────────────────────────────────

_DB_PATH = os.path.join("data", "corrections.db")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS scale_corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts      REAL NOT NULL,
    user_id         TEXT,
    header_ocr      TEXT NOT NULL,   -- space-joined, normalized
    axis_ocr        TEXT NOT NULL,   -- space-joined, normalized
    header_tokens   TEXT NOT NULL,   -- JSON array of normalized tokens
    axis_numbers    TEXT NOT NULL,   -- JSON array of floats
    ai_mnemonic     TEXT,
    ai_scale_type   TEXT,
    ai_left         REAL,
    ai_right        REAL,
    ai_wrapped      INTEGER,
    user_mnemonic   TEXT,
    user_scale_type TEXT,
    user_left       REAL,
    user_right      REAL,
    user_wrapped    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_corrections_user ON scale_corrections(user_id);
"""


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


# ────────────────────────────────────────────────────────────────────
# Normalization
# ────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Z0-9]+")
_NUM_RE  = re.compile(r"-?\d+(?:\.\d+)?")


def _norm_tokens(strings: Iterable[str]) -> List[str]:
    """Uppercase, strip punctuation, keep alphanumeric tokens of length >= 1."""
    out = []
    for s in strings or []:
        if not s:
            continue
        up = str(s).upper()
        for tok in _WORD_RE.findall(up):
            if tok:
                out.append(tok)
    return out


def _extract_numbers(strings: Iterable[str]) -> List[float]:
    out = []
    for s in strings or []:
        for tok in _NUM_RE.findall(str(s)):
            try:
                out.append(float(tok))
            except ValueError:
                continue
    return out


# ────────────────────────────────────────────────────────────────────
# Recording corrections
# ────────────────────────────────────────────────────────────────────

@dataclass
class CorrectionEntry:
    header_ocr:   Sequence[str]
    axis_ocr:     Sequence[str]
    ai_choice:    dict    # { mnemonic, scale_type, left, right, wrapped }
    user_choice:  dict    # same shape
    user_id:      Optional[str] = None


def record_correction(entry: CorrectionEntry) -> int:
    """Insert a correction row and return its id."""
    header_tokens = _norm_tokens(entry.header_ocr)
    axis_numbers = _extract_numbers(entry.axis_ocr)
    ai = entry.ai_choice or {}
    user = entry.user_choice or {}

    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO scale_corrections
               (created_ts, user_id, header_ocr, axis_ocr, header_tokens, axis_numbers,
                ai_mnemonic, ai_scale_type, ai_left, ai_right, ai_wrapped,
                user_mnemonic, user_scale_type, user_left, user_right, user_wrapped)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(),
                entry.user_id,
                " ".join(str(s) for s in (entry.header_ocr or [])),
                " ".join(str(s) for s in (entry.axis_ocr or [])),
                json.dumps(header_tokens),
                json.dumps(axis_numbers),
                (ai.get("mnemonic") or None),
                (ai.get("scale_type") or None),
                _to_float(ai.get("left")),
                _to_float(ai.get("right")),
                1 if ai.get("wrapped") else 0,
                (user.get("mnemonic") or None),
                (user.get("scale_type") or None),
                _to_float(user.get("left")),
                _to_float(user.get("right")),
                1 if user.get("wrapped") else 0,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ────────────────────────────────────────────────────────────────────
# Nearest-neighbor lookup
# ────────────────────────────────────────────────────────────────────

def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    if not a and not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _numeric_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """How similar are two sorted numeric label sets? Uses log-ratio closeness."""
    if not a or not b:
        return 0.0
    aa = sorted(v for v in a if v > 0)
    bb = sorted(v for v in b if v > 0)
    if not aa or not bb:
        # fall back to linear: check if min/max overlap within 10%
        a_min, a_max = min(a), max(a)
        b_min, b_max = min(b), max(b)
        span = max(abs(a_max - a_min), abs(b_max - b_min), 1.0)
        return max(0.0, 1.0 - (abs(a_min - b_min) + abs(a_max - b_max)) / (2 * span))
    # Log-ratio similarity: useful for spotting "same decade pattern"
    import math
    a_log = [math.log10(v) for v in aa]
    b_log = [math.log10(v) for v in bb]
    a_span = (a_log[0], a_log[-1])
    b_span = (b_log[0], b_log[-1])
    delta = abs(a_span[0] - b_span[0]) + abs(a_span[1] - b_span[1])
    return max(0.0, 1.0 - delta / 6.0)  # 6 decades = totally different


def lookup_similar(
    header_ocr: Sequence[str],
    axis_ocr:   Sequence[str],
    user_id:    Optional[str] = None,
    top_k:      int = 3,
    min_similarity: float = 0.5,
) -> List[dict]:
    """Return up to top_k prior corrections similar to the current OCR input.

    Each result dict has: { similarity, user_choice, ai_choice, created_ts, id }.
    Prefers same-user corrections; falls back to global.
    """
    target_tokens = _norm_tokens(header_ocr)
    target_numbers = _extract_numbers(axis_ocr)
    if not target_tokens and not target_numbers:
        return []

    with _get_conn() as conn:
        cur = conn.cursor()
        # Prefer user-specific rows, then fall back to all
        rows = []
        if user_id:
            cur.execute(
                "SELECT id, created_ts, header_tokens, axis_numbers, "
                "  ai_mnemonic, ai_scale_type, ai_left, ai_right, ai_wrapped, "
                "  user_mnemonic, user_scale_type, user_left, user_right, user_wrapped "
                "FROM scale_corrections WHERE user_id = ? ORDER BY created_ts DESC LIMIT 500",
                (user_id,),
            )
            rows = cur.fetchall()
        if len(rows) < 20:
            cur.execute(
                "SELECT id, created_ts, header_tokens, axis_numbers, "
                "  ai_mnemonic, ai_scale_type, ai_left, ai_right, ai_wrapped, "
                "  user_mnemonic, user_scale_type, user_left, user_right, user_wrapped "
                "FROM scale_corrections ORDER BY created_ts DESC LIMIT 500"
            )
            rows.extend([r for r in cur.fetchall() if r[0] not in {x[0] for x in rows}])

    scored: List[Tuple[float, tuple]] = []
    for row in rows:
        try:
            tokens = json.loads(row[2] or "[]")
            nums = json.loads(row[3] or "[]")
        except json.JSONDecodeError:
            continue
        sim_tok = _jaccard(target_tokens, tokens) if target_tokens else 0.0
        sim_num = _numeric_similarity(target_numbers, nums) if target_numbers else 0.0
        # Weight header tokens heavier if present
        if target_tokens:
            similarity = 0.65 * sim_tok + 0.35 * sim_num
        else:
            similarity = sim_num
        if similarity >= min_similarity:
            scored.append((similarity, row))

    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for similarity, row in scored[:top_k]:
        out.append({
            "similarity":  round(similarity, 3),
            "id":          row[0],
            "created_ts":  row[1],
            "ai_choice": {
                "mnemonic":   row[4],
                "scale_type": row[5],
                "left":       row[6],
                "right":      row[7],
                "wrapped":    bool(row[8]),
            },
            "user_choice": {
                "mnemonic":   row[9],
                "scale_type": row[10],
                "left":       row[11],
                "right":      row[12],
                "wrapped":    bool(row[13]),
            },
        })
    return out


def best_suggestion(
    header_ocr: Sequence[str],
    axis_ocr:   Sequence[str],
    user_id:    Optional[str] = None,
) -> Optional[dict]:
    """Return a single best suggestion dict (user_choice + confidence), or None.

    Confidence blends similarity with the count of matching prior corrections.
    """
    matches = lookup_similar(header_ocr, axis_ocr, user_id=user_id, top_k=5)
    if not matches:
        return None

    # If the top match is very close AND agrees with at least one other match on mnemonic,
    # treat as high confidence.
    top = matches[0]
    agreeing = [m for m in matches if m["user_choice"].get("mnemonic") == top["user_choice"].get("mnemonic")]
    consensus_factor = min(1.0, len(agreeing) / 3.0)
    confidence = min(0.95, 0.6 + 0.35 * top["similarity"] * consensus_factor)

    return {
        "user_choice": top["user_choice"],
        "similarity":  top["similarity"],
        "agreeing_count": len(agreeing),
        "confidence":  confidence,
        "source_id":   top["id"],
    }


def count_corrections(user_id: Optional[str] = None) -> int:
    with _get_conn() as conn:
        cur = conn.cursor()
        if user_id:
            cur.execute("SELECT COUNT(*) FROM scale_corrections WHERE user_id = ?", (user_id,))
        else:
            cur.execute("SELECT COUNT(*) FROM scale_corrections")
        return int(cur.fetchone()[0])
