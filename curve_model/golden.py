"""Golden-dataset format and correction-capture conversion utilities."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .dataset import _derive_centerline_geometry


def _decode_capture_image(record: dict, capture_path: Path) -> np.ndarray:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
    image_data = payload.get("image") if isinstance(payload, dict) else None
    if isinstance(image_data, str) and image_data.startswith("data:image") and "," in image_data:
        raw = base64.b64decode(image_data.split(",", 1)[1])
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    candidates = [record.get("image_path"), payload.get("image_path") if isinstance(payload, dict) else None]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = capture_path.parent / path
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    raise ValueError(f"Capture has no readable image: {capture_path}")


def _trace_to_rows(trace_points: list, left: int, top: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.full(height, np.nan, dtype=np.float32)
    for point in trace_points or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            row = int(round(float(point[1]) - top))
            x = float(point[0]) - left
        except (TypeError, ValueError):
            continue
        if 0 <= row < height and np.isfinite(x):
            xs[row] = x
    valid = np.isfinite(xs)
    return xs, valid


def convert_capture(capture_path: Path | str, golden_dir: Path | str, case_id: Optional[str] = None) -> dict:
    capture_path = Path(capture_path)
    record = json.loads(capture_path.read_text(encoding="utf-8"))
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
    curve = payload.get("curve_config") or payload.get("track") or {}
    trace_points = payload.get("trace_points") or record.get("trace_points") or []
    image = _decode_capture_image(record, capture_path)
    image_h, image_w = image.shape[:2]
    left = max(0, int(curve.get("left_px", curve.get("leftX", 0))))
    right = min(image_w, int(curve.get("right_px", curve.get("rightX", image_w))))
    top = max(0, int(curve.get("top_px", 0)))
    bottom = min(image_h, int(curve.get("bottom_px", image_h)))
    if right <= left or bottom <= top:
        raise ValueError(f"Capture has invalid track bounds: {capture_path}")
    crop = image[top:bottom, left:right]
    xs, valid = _trace_to_rows(trace_points, left, top, bottom - top)
    if not valid.any():
        raise ValueError(f"Capture has no usable corrected trace points: {capture_path}")

    case_id = case_id or str(record.get("capture_id") or capture_path.stem)
    output = Path(golden_dir)
    image_dir = output / "images"
    trace_dir = output / "traces"
    metadata_dir = output / "metadata"
    for directory in (image_dir, trace_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{case_id}.png"
    trace_path = trace_dir / f"{case_id}.npz"
    metadata_path = metadata_dir / f"{case_id}.json"
    cv2.imwrite(str(image_path), crop)
    centerline_mask, distance_field, direction_field, valid_direction_mask = _derive_centerline_geometry(
        xs, valid, crop.shape[1]
    )
    wrap_events = payload.get("wrap_events") or record.get("wrap_events") or []
    wrap_index = np.zeros(xs.shape, dtype=np.int32)
    for event in sorted((item for item in wrap_events if isinstance(item, dict)), key=lambda item: int(item.get("row_after", 0))):
        row = int(event.get("row_after", 0))
        direction = event.get("direction")
        if 0 <= row < wrap_index.size:
            wrap_index[row:] += 1 if direction == "right_to_left" else -1
    unwrapped_x = xs + wrap_index.astype(np.float32) * float(crop.shape[1])
    np.savez_compressed(
        trace_path,
        centerline_x_by_row=xs,
        correct_x_by_row=xs,
        correct_unwrapped_x_by_row=unwrapped_x,
        correct_wrap_index_by_row=wrap_index,
        valid_row_mask=valid,
        centerline_mask=centerline_mask,
        distance_field=distance_field,
        direction_field=direction_field,
        valid_direction_mask=valid_direction_mask,
    )
    metadata = {
        "schema_version": 2,
        "case_id": case_id,
        "curve_type": curve.get("type") or payload.get("curve_id"),
        "curve_color": curve.get("mode"),
        "notes": payload.get("notes") or record.get("notes"),
        "source_capture": str(capture_path.resolve()),
        "track_bounds": {"left": left, "right": right, "top": top, "bottom": bottom},
        "label_availability": {
            "centerline": True,
            "distance_field": True,
            "direction_field": True,
            "stroke_mask": False,
            "grid_mask": False,
        },
        "topology": "cylindrical" if bool(curve.get("wrapped")) or wrap_events else "bounded",
        "correct_wrap_events": wrap_events,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest_record = {
        "id": case_id,
        "image": str(image_path.relative_to(output)).replace("\\", "/"),
        "trace": str(trace_path.relative_to(output)).replace("\\", "/"),
        "metadata": str(metadata_path.relative_to(output)).replace("\\", "/"),
    }
    with (output / "manifest.jsonl").open("a", encoding="utf-8", newline="\n") as manifest:
        manifest.write(json.dumps(manifest_record, sort_keys=True) + "\n")
    return manifest_record


def iter_capture_files(corrections_dir: Path | str) -> Iterable[Path]:
    for path in sorted(Path(corrections_dir).rglob("*.json")):
        if path.name == "trace_debug.json":
            continue
        yield path


def convert_synthetic_dataset(
    synthetic_dir: Path | str,
    golden_dir: Path | str,
    limit: Optional[int] = None,
) -> list[dict]:
    """Create a disposable golden set from held-out synthetic samples."""
    synthetic_dir = Path(synthetic_dir)
    output = Path(golden_dir)
    records = [
        json.loads(line)
        for line in (synthetic_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        records = records[:max(0, int(limit))]
    for directory in (output / "images", output / "traces", output / "metadata"):
        directory.mkdir(parents=True, exist_ok=True)

    converted = []
    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for record in records:
            case_id = f"synthetic_{record['id']}"
            source_image = synthetic_dir / record["image"]
            image_path = output / "images" / f"{case_id}.png"
            trace_path = output / "traces" / f"{case_id}.npz"
            metadata_path = output / "metadata" / f"{case_id}.json"
            shutil.copy2(source_image, image_path)
            centerline = np.load(synthetic_dir / record["centerline_x"], allow_pickle=False).astype(np.float32)
            trace_payload = {
                "centerline_x_by_row": centerline,
                "correct_x_by_row": centerline,
                "valid_row_mask": np.isfinite(centerline),
            }
            if record.get("unwrapped_centerline_x"):
                trace_payload["correct_unwrapped_x_by_row"] = np.load(
                    synthetic_dir / record["unwrapped_centerline_x"], allow_pickle=False
                ).astype(np.float32)
            else:
                trace_payload["correct_unwrapped_x_by_row"] = centerline.copy()
            if record.get("wrap_index"):
                trace_payload["correct_wrap_index_by_row"] = np.load(
                    synthetic_dir / record["wrap_index"], allow_pickle=False
                ).astype(np.int32)
            else:
                trace_payload["correct_wrap_index_by_row"] = np.zeros(centerline.shape, dtype=np.int32)
            for key in ("distance_field", "direction_field"):
                if record.get(key):
                    trace_payload[key] = np.load(synthetic_dir / record[key], allow_pickle=False)
            for key in ("centerline_mask", "stroke_mask", "grid_mask", "valid_direction_mask"):
                if record.get(key):
                    mask = cv2.imread(str(synthetic_dir / record[key]), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        trace_payload[key] = mask
            np.savez_compressed(trace_path, **trace_payload)
            source_metadata = json.loads((synthetic_dir / record["metadata"]).read_text(encoding="utf-8"))
            curve = source_metadata.get("curve") or {}
            metadata = {
                "schema_version": 2,
                "case_id": case_id,
                "curve_type": "GR",
                "curve_color": curve.get("color") or "black",
                "notes": "Held-out synthetic smoke-evaluation case",
                "source_seed": record.get("seed"),
                "topology": source_metadata.get("topology", "bounded"),
                "correct_wrap_events": source_metadata.get("wrap_events") or [],
                "label_availability": {key: key in trace_payload for key in (
                    "centerline_mask", "stroke_mask", "grid_mask", "distance_field", "direction_field"
                )},
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            output_record = {
                "id": case_id,
                "image": str(image_path.relative_to(output)).replace("\\", "/"),
                "trace": str(trace_path.relative_to(output)).replace("\\", "/"),
                "metadata": str(metadata_path.relative_to(output)).replace("\\", "/"),
            }
            manifest.write(json.dumps(output_record, sort_keys=True) + "\n")
            converted.append(output_record)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TurboTIFF correction captures to the Phase 1 golden format")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture-json", type=Path)
    source.add_argument("--corrections-dir", type=Path)
    source.add_argument("--synthetic-dir", type=Path)
    parser.add_argument("--golden-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.synthetic_dir:
        converted = convert_synthetic_dataset(args.synthetic_dir, args.golden_dir, args.limit)
        print(json.dumps({"converted": len(converted), "failed": 0}, indent=2))
        return
    paths = [args.capture_json] if args.capture_json else list(iter_capture_files(args.corrections_dir))
    converted = []
    failures = []
    for path in paths:
        try:
            converted.append(convert_capture(path, args.golden_dir))
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
    print(json.dumps({"converted": len(converted), "failed": len(failures), "failures": failures[:20]}, indent=2))


if __name__ == "__main__":
    main()
