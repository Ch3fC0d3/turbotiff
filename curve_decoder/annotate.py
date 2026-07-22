"""CLI and helpers for adding explicit wrap annotations to golden traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def apply_wrap_annotations(trace_path: Path | str, track_width: int, events: list[dict]) -> dict:
    path = Path(trace_path)
    with np.load(path, allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    x = np.asarray(payload.get("correct_x_by_row", payload.get("centerline_x_by_row")), dtype=np.float32)
    if x.ndim != 1 or int(track_width) < 1:
        raise ValueError("Wrap annotation requires a one-dimensional trace and positive track width")
    wrap = np.zeros(x.shape, dtype=np.int32)
    normalized = []
    for event in sorted(events, key=lambda item: int(item["row_after"])):
        row = int(event["row_after"])
        direction = str(event["direction"])
        if direction not in {"right_to_left", "left_to_right"} or not (1 <= row < x.size):
            raise ValueError(f"Invalid wrap event: {event}")
        change = 1 if direction == "right_to_left" else -1
        wrap[row:] += change
        normalized.append({"row_before": row - 1, "row_after": row, "direction": direction})
    payload["correct_x_by_row"] = x
    payload["correct_unwrapped_x_by_row"] = x + wrap.astype(np.float32) * float(track_width)
    payload["correct_wrap_index_by_row"] = wrap
    np.savez_compressed(path, **payload)
    return {"trace": str(path), "track_width": int(track_width), "events": normalized}


def main() -> None:
    parser = argparse.ArgumentParser(description="Add explicit wrap events to a golden trace")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--track-width", required=True, type=int)
    parser.add_argument(
        "--event",
        action="append",
        default=[],
        help="Wrap event as ROW:right_to_left or ROW:left_to_right",
    )
    args = parser.parse_args()
    events = []
    for value in args.event:
        row, direction = value.split(":", 1)
        events.append({"row_after": int(row), "direction": direction})
    print(json.dumps(apply_wrap_annotations(args.trace, args.track_width, events), indent=2))


if __name__ == "__main__":
    main()

