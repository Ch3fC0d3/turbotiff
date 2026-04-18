import os
import sys


def _configure_stdio() -> None:
    """Avoid Windows console/log crashes when background jobs emit Unicode."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(
                encoding=getattr(stream, "encoding", None) or "utf-8",
                errors="backslashreplace",
            )


_configure_stdio()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
cv2.setNumThreads(0)

import tempfile
import uuid
import threading
import re
import warnings
from flask import Blueprint, request, render_template, send_file, jsonify, abort
from werkzeug.utils import secure_filename
import pandas as pd
from directional_converter import DirectionalSurvey, CurveSet, MdTvdConverter

directional_bp = Blueprint('directional', __name__)
UPLOAD_FOLDER = tempfile.gettempdir()

jobs = {}

HEADER_HINTS = (
    "measured",
    "md",
    "depth",
    "tvd",
    "true vert",
    "vertical",
    "inclination",
    "azimuth",
    "northing",
    "easting",
    "latitude",
    "longitude",
    "gamma",
    "temp",
    "emw",
)

TRACK_COLORS = [
    "#7CB518",
    "#E76F51",
    "#277DA1",
    "#9C6644",
    "#6D597A",
    "#3A86FF",
]


def _clean_pdf_cell(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _normalize_label(value) -> str:
    return re.sub(r" +", " ", _clean_pdf_cell(value).replace("\n", " ")).strip()


def _header_score(values) -> int:
    text = " ".join(v.lower() for v in values if v)
    return sum(1 for hint in HEADER_HINTS if hint in text)


def _detect_header_rows(df: pd.DataFrame) -> int:
    header_rows = 0
    for row_idx in range(min(3, len(df))):
        values = [_clean_pdf_cell(v) for v in df.iloc[row_idx].tolist()]
        row_text = " ".join(values)
        alpha_count = sum(ch.isalpha() for ch in row_text)
        digit_count = sum(ch.isdigit() for ch in row_text)
        looks_like_header = _header_score(values) > 0 or (header_rows > 0 and alpha_count >= digit_count)
        if not looks_like_header:
            break
        header_rows += 1
    return header_rows


def _dedupe_columns(columns) -> list[str]:
    seen = {}
    out = []
    for idx, raw_name in enumerate(columns):
        base_name = _normalize_label(raw_name) or f"column_{idx}"
        suffix = seen.get(base_name, 0)
        seen[base_name] = suffix + 1
        out.append(base_name if suffix == 0 else f"{base_name}_{suffix + 1}")
    return out


def _explode_multiline_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        split_cells = {}
        max_parts = 1
        for col in df.columns:
            cell = _clean_pdf_cell(row[col])
            parts = [part.strip() for part in cell.split("\n") if part.strip()] if cell else [None]
            split_cells[col] = parts
            max_parts = max(max_parts, len(parts))

        for idx in range(max_parts):
            expanded_row = {}
            for col, parts in split_cells.items():
                if len(parts) == 1:
                    expanded_row[col] = parts[0]
                else:
                    expanded_row[col] = parts[idx] if idx < len(parts) else None
            if any(value not in (None, "") for value in expanded_row.values()):
                rows.append(expanded_row)

    if not rows:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(rows, columns=df.columns)


def _normalize_pdf_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().dropna(axis=1, how="all").dropna(how="all").reset_index(drop=True)
    if work.empty:
        return work

    work.columns = list(range(work.shape[1]))
    header_rows = _detect_header_rows(work)

    if header_rows:
        headers = []
        for col_idx in range(work.shape[1]):
            parts = []
            for row_idx in range(header_rows):
                label = _normalize_label(work.iat[row_idx, col_idx])
                if label:
                    parts.append(label)
            headers.append(" ".join(parts))
        work = work.iloc[header_rows:].reset_index(drop=True)
    else:
        original_headers = [_normalize_label(col) for col in df.columns]
        if _header_score(original_headers):
            headers = original_headers
        elif not work.empty and _header_score([_clean_pdf_cell(v) for v in work.iloc[0].tolist()]) > 0:
            headers = [_normalize_label(v) for v in work.iloc[0].tolist()]
            work = work.iloc[1:].reset_index(drop=True)
        else:
            headers = [f"column_{idx}" for idx in range(work.shape[1])]

    work.columns = _dedupe_columns(headers)
    return _explode_multiline_rows(work).dropna(axis=1, how="all").reset_index(drop=True)


def _rename_depth_columns(target_df: pd.DataFrame, is_curve: bool, source_basis=None) -> pd.DataFrame:
    target_df = target_df.copy()
    rename_map = {}
    for c in target_df.columns:
        cl = str(c).lower().replace('.', '').replace('\n', ' ').strip()
        cl = re.sub(' +', ' ', cl)

        if not is_curve:
            if 'measured' in cl or cl == 'md' or cl.startswith('m d'):
                rename_map[c] = 'md'
            elif 'true vert' in cl or 'vertical depth' in cl or cl == 'tvd' or cl.startswith('t v ') or ('vertical' in cl and 'depth' in cl):
                rename_map[c] = 'tvd'
        else:
            if 'depth' not in rename_map.values():
                if source_basis == 'MD' and ('md' in cl or 'measured' in cl or cl.startswith('m d') or cl == 'depth'):
                    rename_map[c] = 'depth'
                elif source_basis == 'TVD' and ('tvd' in cl or 'true vert' in cl or 'vertical depth' in cl or cl.startswith('t v ') or ('vertical' in cl and 'depth' in cl)):
                    rename_map[c] = 'depth'
                elif not source_basis and 'depth' in cl:
                    rename_map[c] = 'depth'

    target_df = target_df.rename(columns=rename_map)

    if is_curve and 'depth' not in [str(c).lower() for c in target_df.columns] and len(target_df.columns) > 0:
        cols = list(target_df.columns)
        cols[0] = 'depth'
        target_df.columns = cols

    return target_df


def _numeric_row_count(df: pd.DataFrame, required_columns: list[str]) -> int:
    if not set(required_columns).issubset(df.columns):
        return 0

    temp_df = df.copy()
    for col in required_columns:
        temp_df[col] = pd.to_numeric(
            temp_df[col].astype(str).str.replace(',', '', regex=False),
            errors='coerce',
        )

    return int(temp_df.dropna(subset=required_columns).shape[0])


def _load_las_curve_df(filepath: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lasio
        las = lasio.read(filepath)

    las_df = las.df().reset_index()
    if las_df.empty:
        raise ValueError("LAS file does not contain any curve samples.")

    depth_column = las_df.columns[0]
    las_df = las_df.rename(columns={depth_column: "depth"})
    las_df.columns = [str(column).strip() for column in las_df.columns]
    return las_df


def _humanize_curve_name(name: str) -> str:
    tokens = []
    for token in str(name).replace("-", "_").split("_"):
        if not token:
            continue
        if token.lower() in {"md", "tvd", "emw", "api"}:
            tokens.append(token.upper())
        else:
            tokens.append(token.capitalize())
    return " ".join(tokens) or str(name)


def _track_sort_key(column_name: str, column_index: int) -> tuple[int, int]:
    lower_name = column_name.lower()
    if lower_name.startswith("source_"):
        return (90, column_index)

    priorities = (
        ("gamma", 0),
        ("gr", 1),
        ("temp", 2),
        ("emw", 3),
        ("density", 4),
        ("resi", 5),
        ("porosity", 6),
        ("caliper", 7),
    )
    for needle, priority in priorities:
        if needle in lower_name:
            return (priority, column_index)

    return (50, column_index)


def _track_style(column_name: str, track_index: int) -> dict:
    lower_name = column_name.lower()
    style = {
        "color": TRACK_COLORS[track_index % len(TRACK_COLORS)],
        "dash": "solid",
        "fill": None,
    }

    if "gamma" in lower_name or lower_name == "gr":
        style["color"] = "#88B04B"
        style["fill"] = "tozerox"
    elif "temp" in lower_name:
        style["color"] = "#E76F51"
    elif "emw" in lower_name or "mud" in lower_name:
        style["color"] = "#277DA1"
    elif lower_name.startswith("source_"):
        style["color"] = "#8D99AE"
        style["dash"] = "dot"

    return style


def _track_range(series: pd.Series, column_name: str) -> list[float]:
    valid_series = series.dropna()
    if valid_series.empty:
        return [0.0, 1.0]

    if len(valid_series) > 50:
        min_value = float(valid_series.quantile(0.01))
        max_value = float(valid_series.quantile(0.99))
    else:
        min_value = float(valid_series.min())
        max_value = float(valid_series.max())

    if min_value == max_value:
        return [min_value - 1.0, max_value + 1.0]

    if column_name.lower().startswith("source_"):
        padding = max((max_value - min_value) * 0.03, 25.0)
        return [min_value - padding, max_value + padding]

    if "gamma" in column_name.lower() and min_value >= 0:
        max_value = max(max_value, 10.0)
        padding = max(max_value * 0.05, 5.0)
        return [0.0, max_value + padding]

    if min_value == max_value:
        padding = max(abs(min_value) * 0.05, 1.0)
        return [min_value - padding, max_value + padding]

    padding = (max_value - min_value) * 0.05
    return [min_value - padding, max_value + padding]


def _series_to_plot_values(series: pd.Series) -> list[float | None]:
    values = []
    for value in series.tolist():
        if pd.isna(value):
            values.append(None)
        else:
            values.append(round(float(value), 6))
    return values


def _build_log_view_payload(job: dict, task_id: str) -> dict:
    output_df = pd.read_csv(job["filepath"])
    numeric_df = output_df.apply(pd.to_numeric, errors="coerce")

    depth_column = "depth" if "depth" in numeric_df.columns else numeric_df.columns[0]
    depth_series = numeric_df[depth_column]
    if depth_series.notna().sum() < 2:
        raise ValueError("Converted output does not contain enough numeric depth data to plot.")

    candidate_columns = list(output_df.columns)
    sorted_columns = [
        column_name
        for _, column_name in sorted(
            (_track_sort_key(column_name, idx), column_name)
            for idx, column_name in enumerate(candidate_columns)
            if column_name != depth_column
        )
    ]

    # Load native curve data (original un-interpolated LAS/CSV source) for clean alt-axis display
    native_df = None
    native_basis = job.get("native_basis")
    native_path = job.get("native_filepath")
    if native_path and os.path.exists(native_path):
        try:
            native_raw = pd.read_csv(native_path)
            native_df = native_raw.apply(pd.to_numeric, errors="coerce")
        except Exception:
            native_df = None

    tracks = []
    for track_index, column_name in enumerate(sorted_columns):
        series = numeric_df[column_name]
        if series.notna().sum() < 2:
            continue

        style = _track_style(column_name, track_index)

        # Native-axis values for this curve (when available)
        native_values = None
        if native_df is not None and column_name in native_df.columns:
            native_values = _series_to_plot_values(native_df[column_name])

        tracks.append({
            "name": column_name,
            "display_name": _humanize_curve_name(column_name),
            "values": _series_to_plot_values(series),
            "native_values": native_values,
            "range": _track_range(series, column_name),
            "color": style["color"],
            "dash": style["dash"],
            "fill": style["fill"],
            "is_reference": column_name.lower().startswith("source_"),
            "value_min": round(float(series.min()), 3),
            "value_max": round(float(series.max()), 3),
        })

    if not tracks:
        raise ValueError("Converted output does not contain any plottable numeric curves.")

    depth_label = job.get("target_basis", "Depth")
    depth_min = round(float(depth_series.min()), 3)
    depth_max = round(float(depth_series.max()), 3)

    # Build the native-axis payload (this is the clean, non-interpolated source basis data)
    alt_depth_values = None
    alt_depth_label = None
    alt_depth_min = None
    alt_depth_max = None
    if native_df is not None and "depth" in native_df.columns:
        native_depth_series = native_df["depth"]
        if native_depth_series.notna().sum() >= 2:
            alt_depth_values = _series_to_plot_values(native_depth_series)
            alt_depth_label = native_basis or "MD"
            alt_depth_min = round(float(native_depth_series.min()), 3)
            alt_depth_max = round(float(native_depth_series.max()), 3)

    return {
        "task_id": task_id,
        "title": f"{depth_label} Log Viewer",
        "depth_label": depth_label,
        "depth_values": _series_to_plot_values(depth_series),
        "depth_min": depth_min,
        "depth_max": depth_max,
        "alt_depth_label": alt_depth_label,
        "alt_depth_values": alt_depth_values,
        "alt_depth_min": alt_depth_min,
        "alt_depth_max": alt_depth_max,
        "sample_count": int(depth_series.notna().sum()),
        "curve_count": int(sum(1 for track in tracks if not track["is_reference"])),
        "tracks": tracks,
        "download_url": f"/directional/download/{task_id}",
    }

def parse_file_to_df(filepath, filename, pages_input, is_curve=False, source_basis=None):
    lower_name = filename.lower()

    if lower_name.endswith('.pdf'):
        import directional_pdf_extractor as pdf_extractor
        import importlib
        importlib.reload(pdf_extractor) # flush cache
        pages_list = None
        if pages_input:
            try:
                pages_list = [int(p.strip()) - 1 for p in pages_input.split(",")]
            except Exception:
                raise ValueError("Invalid pages format. Use comma separated numbers like 1,2,3")
        
        dfs = pdf_extractor.extract_survey_from_pdf(filepath, pages_list, is_curve=is_curve)
        if not dfs:
            raise ValueError(f"No tables extracted from the PDF: {filename}.")

        candidate_tables = []
        for df in dfs:
            normalized_df = _normalize_pdf_table(df)
            if normalized_df.empty:
                continue

            normalized_df = _rename_depth_columns(normalized_df, is_curve=is_curve, source_basis=source_basis)

            required_columns = ['depth'] if is_curve else ['md', 'tvd']
            if _numeric_row_count(normalized_df, required_columns) < 2:
                continue

            candidate_tables.append(normalized_df)

        if not candidate_tables:
            raise ValueError(f"No usable tables extracted from the PDF: {filename}.")

        target_df = pd.concat(candidate_tables, ignore_index=True, sort=False)
    elif lower_name.endswith('.las'):
        if not is_curve:
            raise ValueError("LAS input is currently only supported for curve files.")
        target_df = _load_las_curve_df(filepath)
    else:
        target_df = pd.read_csv(filepath)
        
    target_df.columns = [str(c) for c in target_df.columns]
    
    target_df = _rename_depth_columns(target_df, is_curve=is_curve, source_basis=source_basis)
        
    return target_df

def convert_bg_task(task_id, survey_path, survey_name, survey_pages, curve_path, curve_name, curve_pages, source_basis, target_basis, step, smooth_window):
    try:
        survey_df = parse_file_to_df(survey_path, survey_name, survey_pages, is_curve=False)
        curve_df = parse_file_to_df(curve_path, curve_name, curve_pages, is_curve=True, source_basis=source_basis)
        
        survey = DirectionalSurvey(survey_df)
        curves = CurveSet(curve_df, source_basis=source_basis)
        converter = MdTvdConverter(survey)
        
        out = converter.convert(
            curves=curves,
            target_basis=target_basis,
            step=step,
            smooth_window=smooth_window,
        )
        
        out_filename = f"converted_{target_basis}.csv"
        out_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{out_filename}")
        out.to_csv(out_path, index=False)

        # ALSO save the native source-basis curve data for clean display (no interp noise)
        native_df = curves.df.copy()
        native_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_native_{source_basis}.csv")
        native_df.to_csv(native_path, index=False)

        primary_tracks = [column for column in out.columns if column != 'depth' and not column.lower().startswith('source_')]
        jobs[task_id] = {
            'status': 'done',
            'filepath': out_path,
            'filename': out_filename,
            'native_filepath': native_path,
            'native_basis': source_basis,
            'source_basis': source_basis,
            'target_basis': target_basis,
            'curve_count': len(primary_tracks),
            'sample_count': int(out['depth'].notna().sum()) if 'depth' in out.columns else int(len(out)),
            'depth_min': float(out['depth'].min()) if 'depth' in out.columns else None,
            'depth_max': float(out['depth'].max()) if 'depth' in out.columns else None,
        }
    except Exception as e:
        jobs[task_id] = {'status': 'error', 'error': str(e)}

@directional_bp.route('/')
def index():
    return render_template('directional_index.html')


@directional_bp.route('/viewer/<task_id>')
def viewer(task_id):
    job = jobs.get(task_id)
    if not job or job.get('status') != 'done':
        return "<h3>Session Expired</h3><p>Your conversion task was not found. This usually happens if the server was restarted or the link expired.</p><p><a href='/directional'>&larr; Return to Directional Tool</a></p>", 404
    return render_template('directional_viewer.html', task_id=task_id)

@directional_bp.route('/start_conversion', methods=['POST'])
def start_conversion():
    survey_file = request.files['survey']
    curve_file = request.files['curve']
    
    task_id = str(uuid.uuid4())
    survey_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{secure_filename(survey_file.filename)}")
    curve_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{secure_filename(curve_file.filename)}")
    
    survey_file.save(survey_path)
    curve_file.save(curve_path)
    
    survey_pages = request.form.get('survey_pages', '').strip()
    curve_pages = request.form.get('curve_pages', '').strip()
    source_basis = request.form['source_basis'].upper()
    target_basis = request.form['target_basis'].upper()
    step = float(request.form.get('step', 1.0))
    smooth = request.form.get('smooth', '')
    smooth_window = int(smooth) if smooth else None
    
    jobs[task_id] = {'status': 'processing'}
    
    t = threading.Thread(target=convert_bg_task, args=(
        task_id, survey_path, survey_file.filename, survey_pages, 
        curve_path, curve_file.filename, curve_pages, 
        source_basis, target_basis, step, smooth_window
    ))
    t.start()
    
    return jsonify({'task_id': task_id})

@directional_bp.route('/status/<task_id>')
def get_status(task_id):
    return jsonify(jobs.get(task_id, {'status': 'unknown'}))


@directional_bp.route('/plot_data/<task_id>')
def plot_data(task_id):
    job = jobs.get(task_id)
    if not job or job.get('status') != 'done':
        return jsonify({'error': 'Plot data is only available for completed conversions.'}), 404

    try:
        return jsonify(_build_log_view_payload(job, task_id))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400

@directional_bp.route('/download/<task_id>')
def download(task_id):
    format_type = request.args.get('format', 'csv').lower()
    job = jobs.get(task_id)
    if not job or job['status'] != 'done':
        return "Not found", 404
        
    csv_path = job['filepath']
    
    if format_type == 'csv':
        return send_file(csv_path, as_attachment=True, download_name=job['filename'])
    elif format_type == 'las':
        import lasio
        import pandas as pd
        las_path = csv_path.replace('.csv', '.las')
        las_filename = job['filename'].replace('.csv', '.las')
        
        if not os.path.exists(las_path):
            df = pd.read_csv(csv_path)
            las = lasio.LASFile()
            las.well.WELL = "Converted Log"
            
            depth_col = 'depth' if 'depth' in df.columns else df.columns[0]
            las.add_curve('DEPTH', df[depth_col].values, unit=job.get('target_basis', 'F'), descr='Depth')
            
            for col in df.columns:
                if col != depth_col:
                    las.add_curve(col, df[col].values, descr=col)
                    
            las.write(las_path, version=2.0)
            
        return send_file(las_path, as_attachment=True, download_name=las_filename)
    else:
        return "Unsupported format", 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
