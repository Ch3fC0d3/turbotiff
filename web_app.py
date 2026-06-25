#!/usr/bin/env python3
"""
web_app.py — Flask web app for TIFF→LAS digitizer with Google Vision API

Setup:
1. pip install flask google-cloud-vision opencv-python numpy pandas
2. Get Google Cloud Vision API key: https://console.cloud.google.com
3. Set environment variable: GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json
4. Run: python web_app.py
5. Open: http://localhost:5000

Free hosting: Deploy to Render.com, Railway.app, or Google Cloud Run
"""
# Vision API is optional; initialize later after env vars are loaded.
VISION_API_AVAILABLE = False
vision_client = None
LOCAL_OCR_AVAILABLE = False
easyocr = None
_easyocr_reader = None

# Load environment variables from .env and .env.local
from dotenv import load_dotenv
load_dotenv()  # Load .env
load_dotenv('.env.local', override=True)  # Load .env.local (overrides .env)

from google.cloud import storage

from flask import (
    Flask, render_template, request, jsonify, make_response, 
    send_file, Response, redirect, url_for, session, flash, send_from_directory
)
import math
import os
import atexit
import random
import re
import shutil
import string
import sqlite3
import threading
from app import auth_billing
from app import mailer
from app import scale_detection
from app import corrections_store
import app.config as config
from werkzeug.security import generate_password_hash, check_password_hash
import stripe
import secrets
from datetime import datetime, timedelta, timezone
import tempfile
import textwrap
import time
import heapq
from collections import defaultdict
import cv2
import numpy as np
import fast_tracer
import pandas as pd
import json
from io import BytesIO, StringIO
import base64
import zipfile
from typing import Any, Dict, List, Tuple, Optional
import tempfile
from datetime import datetime
import uuid
from pathlib import Path
import requests
import openai
from huggingface_hub import InferenceClient

try:
    from flask_talisman import Talisman
    TALISMAN_AVAILABLE = True
except ImportError:
    TALISMAN_AVAILABLE = False
    print("[WARN] flask_talisman not installed. Security headers will not be applied.")

TORCH_AVAILABLE = False
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None

try:
    import easyocr as _easyocr_mod
    easyocr = _easyocr_mod
    LOCAL_OCR_AVAILABLE = True
    print("[OK] EasyOCR available for local OCR fallback.")
except Exception as e:
    easyocr = None
    LOCAL_OCR_AVAILABLE = False
    print(f"[INFO] EasyOCR unavailable; local OCR fallback disabled: {e}")

# Phase 1 & 2: Learning system imports
from user_tracker import tracker
from parameter_learner import ParameterLearner
from ai_tracer import AITracer

# Initialize learning system after all imports
learner = ParameterLearner(tracker)
import hashlib

# Initialize AI tracer
ai_tracer = AITracer("curve_trace_model.pt")

# Try to import Google Vision API (optional)
credentials = None
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    
    # Check for credentials in environment
    if 'GOOGLE_VISION_CREDENTIALS_JSON' in os.environ:
        # Railway/Cloud deployment: JSON in environment variable
        creds_json = json.loads(os.environ['GOOGLE_VISION_CREDENTIALS_JSON'])
        credentials = service_account.Credentials.from_service_account_info(creds_json)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        VISION_API_AVAILABLE = True
        print("[OK] Google Vision API: Loaded from environment variable")
    elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        # Local development: JSON file path in env var
        vision_client = vision.ImageAnnotatorClient()
        VISION_API_AVAILABLE = True
        print("[OK] Google Vision API: Loaded from file")
    else:
        # Auto-detect key file in project directory
        _local_key = Path(__file__).parent / 'GOOGLE_APPLICATION_CREDENTIALS.json'
        if _local_key.exists():
            credentials = service_account.Credentials.from_service_account_file(str(_local_key))
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            VISION_API_AVAILABLE = True
            print(f"[OK] Google Vision API: Auto-loaded from {_local_key.name}")
        else:
            print("[WARN] Google Vision API: No credentials found")
            vision_client = None
            VISION_API_AVAILABLE = False
except ImportError:
    print("[WARN] Google Vision API not available. Install: pip install google-cloud-vision")
    vision_client = None
    VISION_API_AVAILABLE = False
except Exception as e:
    print(f"[WARN] Google Vision API error: {e}")
    vision_client = None
    VISION_API_AVAILABLE = False

# Optional LAS validator
LASIO_AVAILABLE = False

try:
    import lasio
    LASIO_AVAILABLE = True
    print("[OK] lasio imported; LAS validation enabled.")
except Exception as e:
    print(f"[INFO] lasio unavailable; LAS validation will be skipped: {e}")

# Default LAS curve label mapping by type (kept in sync with frontend curveTypeDefaults)
CURVE_TYPE_DEFAULTS = {
    "GR":   {"mnemonic": "GR",   "unit": "API"},
    "RHOB": {"mnemonic": "RHOB", "unit": "G/CC"},
    "NPHI": {"mnemonic": "NPHI", "unit": "V/V"},
    "DT":   {"mnemonic": "DTC",  "unit": "US/F"},
    "DTC":  {"mnemonic": "DTC",  "unit": "US/F"},
    "CALI": {"mnemonic": "CALI", "unit": "IN"},
    "SP":   {"mnemonic": "SP",   "unit": "MV"},
}

MISSING_MARKERS = [-999.25, -999.0, -9999.0, 999.25]

CURVE_KEYWORDS = {
    "GR":   ["GR", "GAMMA RAY"],
    "RES":  ["RES", "RESISTIVITY", "ILD", "LLD", "LWD RES"],
    "RHOB": ["RHOB", "DENSITY", "RHO B", "BULK DENSITY"],
    "NPHI": ["NPHI", "NEUTRON POROSITY", "NEUT", "PHI N"],
    "PEF":  ["PEF", "PHOTOELECTRIC", "PE"],
}

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID") or os.getenv("OPENAI_MODEL")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID") or "models/gemini-2.0-flash"

try:
    CURVE_TRACE_UPSCALE = float(os.environ.get("CURVE_TRACE_UPSCALE", "2.0"))
except Exception:
    CURVE_TRACE_UPSCALE = 2.0
CURVE_TRACE_UPSCALE = max(1.0, min(4.0, CURVE_TRACE_UPSCALE))

APP_VERSION = os.environ.get("APP_VERSION", "wrap-repair-tool-20260618")
APP_BUILD_TIME = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

app = Flask(__name__)

from directional_app import directional_bp

@directional_bp.before_request
def require_directional_access():
    user = _current_user(require_access=True)
    if not user:
        return redirect(url_for('login', next=request.url))
    if not auth_billing.can_access_workspace(user):
        flash('Full-service users cannot access the self-serve tools. Upgrade to a self-serve plan to use this feature.', 'warning')
        return redirect(url_for('dashboard'))

app.register_blueprint(directional_bp, url_prefix='/directional')

# Basic security config
is_prod = os.environ.get("FLASK_ENV") == "production" or os.environ.get("RENDER") == "true" or os.environ.get("RAILWAY_ENVIRONMENT") is not None
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max request size
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = is_prod  # Require HTTPS for cookies in production
app.config['SESSION_COOKIE_HTTPONLY'] = True   # Prevent JS access to session cookie
app.secret_key = config.SECRET_KEY

# Initialize Talisman for security headers if in production and installed
if TALISMAN_AVAILABLE and is_prod:
    csp = {
        'default-src': [
            '\'self\'',
            'https://fonts.googleapis.com',
            'https://fonts.gstatic.com',
            'https://cdn.jsdelivr.net',
            'https://cdnjs.cloudflare.com',
            'https://js.stripe.com',
        ],
        'script-src': [
            '\'self\'',
            '\'unsafe-inline\'',  # Needed for some inline scripts in the app
            '\'unsafe-eval\'',    # Needed for some charting/canvas libraries
            'https://cdn.jsdelivr.net',
            'https://cdnjs.cloudflare.com',
            'https://js.stripe.com',
        ],
        'img-src': [
            '\'self\'',
            'data:',
            'blob:',
            'https://images.unsplash.com',
            '*',  # Allow external images for now given the dynamic nature
        ],
        'style-src': [
            '\'self\'',
            '\'unsafe-inline\'',
            'https://fonts.googleapis.com',
            'https://cdn.jsdelivr.net',
            'https://cdnjs.cloudflare.com',
        ],
        'frame-src': [
            '\'self\'',
            'https://js.stripe.com',
            'https://hooks.stripe.com',
        ],
        'connect-src': [
            '\'self\'',
            'https://api.stripe.com',
        ],
    }
    Talisman(app, 
             content_security_policy=csp, 
             force_https=True,
             strict_transport_security=True,
             session_cookie_secure=True)

REMEMBER_COOKIE_NAME = 'remember_token'
REMEMBER_COOKIE_DAYS = 30


def _remember_serializer():
    from itsdangerous import URLSafeTimedSerializer
    return URLSafeTimedSerializer(config.SECRET_KEY, salt='remember-me')


def _create_remember_token(payload: dict) -> str:
    return _remember_serializer().dumps(payload)


def _decode_remember_token(raw: str) -> Optional[dict]:
    from itsdangerous import BadSignature, SignatureExpired
    try:
        return _remember_serializer().loads(raw, max_age=REMEMBER_COOKIE_DAYS * 86400)
    except (BadSignature, SignatureExpired, Exception):
        return None


@app.before_request
def restore_session_from_token():
    """If no active session, check for a remember-me token cookie and restore the session."""
    if session.get('user_id') or session.get('admin_override'):
        return
    raw_token = request.cookies.get(REMEMBER_COOKIE_NAME)
    if not raw_token:
        return
    payload = _decode_remember_token(raw_token)
    if not payload:
        return
    if payload.get('admin'):
        session['admin_override'] = True
        session.permanent = True
    elif payload.get('user_id'):
        user = auth_billing.get_user_by_id(config.AUTH_DB_PATH, int(payload['user_id']))
        if user and not user.get('is_banned'):
            session['user_id'] = user['id']
            session['is_admin'] = user.get('is_admin', 0)
            session.permanent = True

auth_billing.init_db(config.AUTH_DB_PATH)
stripe.api_key = config.STRIPE_SECRET_KEY

# ----------------------------
# Persistent cache cleanup
# ----------------------------
def _env_bool(name: str, default: bool = True) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    try:
        return max(min_value, int(float(os.environ.get(name, default))))
    except Exception:
        return default


CACHE_CLEANUP_ENABLED = _env_bool("TURBOTIFF_CACHE_CLEANUP_ENABLED", True)
CACHE_IMAGE_TTL_SECONDS = _env_int("TURBOTIFF_CACHE_IMAGE_TTL_HOURS", 1, 0) * 3600
CACHE_TEMP_TTL_SECONDS = _env_int("TURBOTIFF_CACHE_TEMP_TTL_HOURS", 6, 1) * 3600
CACHE_CLEANUP_INTERVAL_SECONDS = _env_int("TURBOTIFF_CACHE_CLEANUP_INTERVAL_SECONDS", 900, 300)
CACHE_IMAGES_MAX_BYTES = _env_int("TURBOTIFF_CACHE_IMAGES_MAX_MB", 64, 16) * 1024 * 1024
CACHE_KEEP_SAVED_LOG_IMAGES = _env_bool("TURBOTIFF_CACHE_KEEP_SAVED_LOG_IMAGES", True)
CACHE_STRIP_DB_IMAGE_BLOBS = _env_bool("TURBOTIFF_STRIP_DB_IMAGE_BLOBS", True)


def _extract_api_image_filename(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    marker = "/api/images/"
    if marker not in text:
        return None
    filename = text.split(marker, 1)[1].split("?", 1)[0].split("#", 1)[0].strip("/")
    if not filename or "/" in filename or "\\" in filename:
        return None
    return filename


def _referenced_image_filenames() -> set:
    referenced = set()
    if not CACHE_KEEP_SAVED_LOG_IMAGES:
        return referenced
    db_path = config.AUTH_DB_PATH
    if not db_path or not os.path.exists(db_path):
        return referenced
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT original_image_path, cropped_image_path FROM user_logs"
            ).fetchall()
        for original_path, cropped_path in rows:
            for value in (original_path, cropped_path):
                filename = _extract_api_image_filename(value)
                if filename:
                    referenced.add(filename)
    except Exception as exc:
        print(f"[cache-cleanup] Could not read referenced image paths: {exc}")
    return referenced


def _is_inline_data_blob(value: Any) -> bool:
    text = str(value or "").lstrip()
    return text.startswith("data:")


def _storage_safe_image_ref(value: Any) -> Optional[str]:
    """Keep only lightweight image references; never persist inline base64 blobs."""
    text = str(value or "").strip()
    if not text or _is_inline_data_blob(text):
        return None
    return text


def _cleanup_db_image_blobs() -> Dict[str, int]:
    stats = {"rows": 0, "fields": 0, "bytes_removed": 0, "vacuumed": 0}
    if not CACHE_STRIP_DB_IMAGE_BLOBS:
        return stats
    db_path = config.AUTH_DB_PATH
    if not db_path or not os.path.exists(db_path):
        return stats
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT id, original_image_path, cropped_image_path FROM user_logs"
            ).fetchall()
            for log_id, original_path, cropped_path in rows:
                updates = {}
                for column, value in (
                    ("original_image_path", original_path),
                    ("cropped_image_path", cropped_path),
                ):
                    if _is_inline_data_blob(value):
                        updates[column] = None
                        stats["fields"] += 1
                        stats["bytes_removed"] += len(str(value).encode("utf-8", errors="ignore"))
                if updates:
                    stats["rows"] += 1
                    assignments = ", ".join(f"{column} = ?" for column in updates)
                    conn.execute(
                        f"UPDATE user_logs SET {assignments}, updated_at = ? WHERE id = ?",
                        (*updates.values(), datetime.now(timezone.utc).isoformat(), log_id),
                    )
            if stats["fields"]:
                conn.commit()
                try:
                    conn.execute("VACUUM")
                    stats["vacuumed"] = 1
                except Exception as exc:
                    print(f"[cache-cleanup] DB VACUUM failed: {exc}")
    except Exception as exc:
        print(f"[cache-cleanup] Could not strip DB image blobs: {exc}")
    return stats


def _file_age_seconds(path: Path, now_ts: Optional[float] = None) -> float:
    now_ts = time.time() if now_ts is None else now_ts
    try:
        return max(0.0, now_ts - path.stat().st_mtime)
    except Exception:
        return 0.0


def _safe_unlink_file(path: Path) -> int:
    try:
        size = path.stat().st_size
        path.unlink()
        return int(size)
    except FileNotFoundError:
        return 0
    except Exception as exc:
        print(f"[cache-cleanup] Could not remove {path}: {exc}")
        return 0


def _cleanup_empty_dirs(root: Path) -> int:
    removed = 0
    if not root.exists():
        return removed
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
            removed += 1
        except Exception:
            pass
    return removed


def _cleanup_unreferenced_images(now_ts: float) -> Dict[str, int]:
    images_dir = Path(config.DATA_ROOT) / "images"
    stats = {"deleted": 0, "bytes": 0, "kept_referenced": 0}
    if not images_dir.exists():
        return stats

    referenced = _referenced_image_filenames()
    unreferenced = []
    total_size = 0

    for path in images_dir.glob("*"):
        if not path.is_file():
            continue
        try:
            size = int(path.stat().st_size)
        except Exception:
            size = 0
        total_size += size
        if path.name in referenced:
            stats["kept_referenced"] += 1
            continue
        unreferenced.append((path, size, path.stat().st_mtime if path.exists() else 0.0))
        if _file_age_seconds(path, now_ts) >= CACHE_IMAGE_TTL_SECONDS:
            removed = _safe_unlink_file(path)
            if removed:
                stats["deleted"] += 1
                stats["bytes"] += removed
                total_size -= removed

    if total_size > CACHE_IMAGES_MAX_BYTES:
        remaining = []
        for path, size, mtime in unreferenced:
            if path.exists():
                remaining.append((mtime, path, size))
        for _mtime, path, _size in sorted(remaining):
            if total_size <= CACHE_IMAGES_MAX_BYTES:
                break
            removed = _safe_unlink_file(path)
            if removed:
                stats["deleted"] += 1
                stats["bytes"] += removed
                total_size -= removed

    return stats


def _cleanup_ttl_dirs(now_ts: float) -> Dict[str, int]:
    stats = {"deleted": 0, "bytes": 0, "dirs": 0}
    data_root = Path(config.DATA_ROOT)
    for dirname in ("cache", "tmp", "temp", "uploads"):
        root = data_root / dirname
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and _file_age_seconds(path, now_ts) >= CACHE_TEMP_TTL_SECONDS:
                removed = _safe_unlink_file(path)
                if removed:
                    stats["deleted"] += 1
                    stats["bytes"] += removed
        stats["dirs"] += _cleanup_empty_dirs(root)
    return stats


def _cleanup_app_temp_dir(now_ts: float, shutdown: bool = False) -> Dict[str, int]:
    stats = {"deleted": 0, "bytes": 0, "dirs": 0}
    temp_root = Path(tempfile.gettempdir())
    prefixes = ("turbotiff", "tiflas", "tmp_turbotiff")
    ttl = 0 if shutdown else CACHE_TEMP_TTL_SECONDS
    try:
        candidates = [p for p in temp_root.iterdir() if p.name.lower().startswith(prefixes)]
    except Exception:
        return stats
    for path in candidates:
        if _file_age_seconds(path, now_ts) < ttl:
            continue
        try:
            if path.is_dir():
                size = sum((f.stat().st_size for f in path.rglob("*") if f.is_file()), 0)
                shutil.rmtree(path, ignore_errors=True)
                if not path.exists():
                    stats["dirs"] += 1
                    stats["bytes"] += int(size)
            elif path.is_file():
                removed = _safe_unlink_file(path)
                if removed:
                    stats["deleted"] += 1
                    stats["bytes"] += removed
        except Exception as exc:
            print(f"[cache-cleanup] Could not clean temp path {path}: {exc}")
    return stats


def run_storage_cleanup(reason: str = "manual", shutdown: bool = False) -> Dict[str, Any]:
    if not CACHE_CLEANUP_ENABLED:
        return {"enabled": False, "reason": reason}
    now_ts = time.time()
    stats = {
        "enabled": True,
        "reason": reason,
        "data_root": str(config.DATA_ROOT),
        "db": _cleanup_db_image_blobs(),
        "images": _cleanup_unreferenced_images(now_ts),
        "ttl_dirs": _cleanup_ttl_dirs(now_ts),
        "temp": _cleanup_app_temp_dir(now_ts, shutdown=shutdown),
    }
    print(f"[cache-cleanup] {json.dumps(stats, ensure_ascii=False)}")
    return stats


def _cache_cleanup_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(CACHE_CLEANUP_INTERVAL_SECONDS):
        try:
            run_storage_cleanup(reason="periodic")
        except Exception as exc:
            print(f"[cache-cleanup] Periodic cleanup failed: {exc}")


_cache_cleanup_stop_event = threading.Event()
if CACHE_CLEANUP_ENABLED:
    try:
        run_storage_cleanup(reason="startup")
        _cache_cleanup_thread = threading.Thread(
            target=_cache_cleanup_loop,
            args=(_cache_cleanup_stop_event,),
            name="turbotiff-cache-cleanup",
            daemon=True,
        )
        _cache_cleanup_thread.start()
    except Exception as exc:
        print(f"[cache-cleanup] Startup cleanup failed: {exc}")


def _shutdown_storage_cleanup() -> None:
    try:
        _cache_cleanup_stop_event.set()
        run_storage_cleanup(reason="shutdown", shutdown=True)
    except Exception as exc:
        print(f"[cache-cleanup] Shutdown cleanup failed: {exc}")


atexit.register(_shutdown_storage_cleanup)

PLAN_TO_PRICE = {
    'monthly': config.STRIPE_PRICE_MONTHLY,
    'annual': config.STRIPE_PRICE_ANNUAL,
    'managed_simple': config.STRIPE_PRICE_MANAGED_SIMPLE,
    'managed_standard': config.STRIPE_PRICE_MANAGED_STANDARD,
    'managed_complex': config.STRIPE_PRICE_MANAGED_COMPLEX,
}
PRICE_TO_PLAN = {v: k for k, v in PLAN_TO_PRICE.items() if v}

# ----------------------------
# Auth Decorator
# ----------------------------
def _is_stripe_configured() -> bool:
    return bool(config.STRIPE_SECRET_KEY and config.STRIPE_PRICE_MONTHLY and config.STRIPE_PRICE_ANNUAL)


def _unix_to_iso(ts: Optional[int]) -> Optional[str]:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _current_user(require_access: bool = True):
    if session.get('admin_override'):
        return {'id': 0, 'email': 'admin@tiflas.com', 'is_admin': 1, 'full_name': 'Admin User'}
        
    user_id = session.get('user_id')
    
    # Check for impersonation
    if session.get('impersonate_user_id') and session.get('is_admin'):
        user_id = session.get('impersonate_user_id')
        
    if not user_id:
        return None
    user = auth_billing.get_user_by_id(config.AUTH_DB_PATH, int(user_id))
    
    # Auto-promote owner emails to admin with unlimited access
    OWNER_EMAILS = {'gabepell@hotmail.com', 'gabriel@pellegrini.us'}
    if user and user.get('email', '').lower() in OWNER_EMAILS:
        needs_update = (not user.get('is_admin') or
                        user.get('subscription_status') != 'active' or
                        user.get('plan_code') != 'lifetime_comped')
        if needs_update:
            with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
                conn.execute(
                    "UPDATE users SET is_admin = 1, subscription_status = 'active', plan_code = 'lifetime_comped' WHERE id = ?",
                    (user['id'],)
                )
            user['is_admin'] = 1
            user['subscription_status'] = 'active'
            user['plan_code'] = 'lifetime_comped'
        session['is_admin'] = 1

    # If standard auth, check banned status
    if user and user.get('is_banned') and not session.get('is_admin'):
        session.clear()
        return None
    if not user:
        session.pop('user_id', None)
        return None
    if require_access and _is_stripe_configured() and not auth_billing.subscription_access_allowed(user):
        return None
    return user


from functools import wraps
def login_required(require_access=True):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = _current_user(require_access=require_access)
            if user is None:
                return redirect(url_for('login', next=request.url))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.errorhandler(500)
def _handle_internal_server_error(exc):
    import traceback
    tb = traceback.format_exc()
    original = getattr(exc, 'original_exception', None)
    err_msg = str(original) if original else str(exc)
    
    print(f"500 Error: {err_msg}")
    
    # Hide details in production
    if is_prod:
        return jsonify({
            'success': False,
            'error': 'An internal server error occurred.'
        }), 500

    print(tb)
    return jsonify({
        'success': False,
        'error': f'Internal server error: {err_msg}',
        'traceback': tb.splitlines()[-5:] if tb else []
    }), 500

from werkzeug.exceptions import HTTPException

@app.errorhandler(Exception)
def _handle_unhandled_exception(exc):
    if isinstance(exc, HTTPException):
        return exc
    
    import traceback
    original = getattr(exc, 'original_exception', None)
    err_msg = str(original) if original else str(exc)
    tb = traceback.format_exc()
    
    print(f"500 error: {err_msg}\n{tb}")
    
    # Hide details in production
    if is_prod:
        return jsonify({
            'success': False,
            'error': f'An internal server error occurred: {err_msg}'
        }), 500

    return jsonify({
        'success': False,
        'error': err_msg,
        'traceback': tb.splitlines()[-25:],
    }), 500

# ----------------------------
# Core Processing Functions
# ----------------------------
def clean_values(values):
    """Replace common missing markers with NaN."""
    arr = np.array(values, dtype=float)
    for m in MISSING_MARKERS:
        arr[arr == m] = np.nan
    return arr


def compute_curve_features(depth, curve_values, curve_name):
    """Compute simple numeric features for a curve to help AI reason about it."""
    values = clean_values(curve_values)
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]

    features = {
        "curve": curve_name,
        "num_samples": int(len(values)),
        "num_valid": int(valid_mask.sum()),
    }

    if len(values) == 0:
        features["pct_missing"] = None
        return features

    features["pct_missing"] = float(100.0 * (1.0 - valid_mask.mean()))

    if len(v) == 0:
        return features

    # Basic stats
    features.update({
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v)),
        "p5": float(np.nanpercentile(v, 5)),
        "p95": float(np.nanpercentile(v, 95)),
    })

    # Gradient stats (change per unit depth)
    try:
        depth_arr = np.asarray(depth, dtype=float)
        depth_valid = depth_arr[valid_mask]
        d_depth = np.diff(depth_valid)
        d_vals = np.diff(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = d_vals / d_depth
        grad = grad[~np.isnan(grad) & ~np.isinf(grad)]
        if grad.size > 0:
            features.update({
                "grad_mean": float(np.mean(grad)),
                "grad_std": float(np.std(grad)),
                "grad_p95": float(np.percentile(grad, 95)),
            })
    except Exception:
        pass

    # Very simple spike detection via z-score
    if v.std() > 0:
        z = (v - v.mean()) / v.std()
        spike_threshold = 4.0
        spikes = np.abs(z) > spike_threshold
        features.update({
            "pct_spikes": float(100.0 * spikes.mean()),
            "num_spikes": int(spikes.sum()),
        })
    else:
        features.update({
            "pct_spikes": 0.0,
            "num_spikes": 0,
        })

    # Longest run of consecutive missing samples
    longest_missing = 0
    current = 0
    for is_valid in valid_mask:
        if not is_valid:
            current += 1
            longest_missing = max(longest_missing, current)
        else:
            current = 0

    features["max_consecutive_missing"] = int(longest_missing)

    return features


def summarize_las_curves_from_str(las_text, depth_mnemonics=("DEPT", "DEPTH")):
    """Read LAS content from a string and compute features for each non-depth curve."""
    if not LASIO_AVAILABLE:
        return None

    try:
        las = lasio.read(StringIO(las_text))
    except Exception as exc:
        print(f"LAS summary: failed to parse LAS content: {exc}")
        return None

    depth_curve = None
    for c in las.curves:
        if c.mnemonic.upper() in [d.upper() for d in depth_mnemonics]:
            depth_curve = c
            break

    if depth_curve is None:
        return None

    depth = depth_curve.data
    all_features = []

    for curve in las.curves:
        if curve is depth_curve:
            continue
        f = compute_curve_features(depth, curve.data, curve.mnemonic)
        f["mnemonic"] = curve.mnemonic
        f["unit"] = getattr(curve, "unit", "") or ""
        f["description"] = getattr(curve, "descr", "") or getattr(curve, "description", "") or ""
        f["rule_type_guess"] = guess_curve_type_from_metadata(
            f["mnemonic"], f["unit"], f["description"]
        )
        all_features.append(f)

    return {
        "well_info": {
            "start_depth": float(depth[0]),
            "end_depth": float(depth[-1]),
            "num_samples": int(len(depth)),
        },
        "curve_features": all_features,
    }


def extract_curve_labels_from_text(full_text: str):
    """Use simple keyword matching over OCR text to find which curves appear on the image."""
    if not full_text:
        return []
    text_upper = full_text.upper()
    found = set()
    for mnemo, keywords in CURVE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_upper:
                found.add(mnemo)
                break
    return sorted(found)


def match_vision_to_las_curves(vision_labels, las_curve_mnemonics):
    """Map OCR-detected labels (GR/RHOB/NPHI/RES, etc.) to LAS mnemonics.

    Uses exact match first, then prefix/contains matching.
    """
    if not vision_labels or not las_curve_mnemonics:
        return {}

    las_upper = {m.upper(): m for m in las_curve_mnemonics}
    mapping = {}

    # 1) exact matches
    for label in vision_labels:
        lu = label.upper()
        if lu in las_upper:
            mapping[label] = las_upper[lu]

    # 2) heuristic prefix/contains for remaining
    for label in vision_labels:
        if label in mapping:
            continue
        lu = label.upper()
        candidates = [
            m for m in las_curve_mnemonics
            if m.upper().startswith(lu) or lu in m.upper()
        ]
        mapping[label] = candidates[0] if candidates else None

    return mapping


def guess_curve_type_from_metadata(mnemonic, unit, description):
    m = (mnemonic or "").upper()
    u = (unit or "").upper()
    d = (description or "").upper()

    text = " ".join([m, u, d])

    if "GR" in m or "GAMMA" in text:
        return "GR"
    if "RHOB" in m or "DENSITY" in text or "RHO B" in text:
        return "RHOB"
    if "NPHI" in m or "NEUT" in text or "POROSITY" in text:
        return "NPHI"
    if "DT" in m or "DTC" in m or "SONIC" in text:
        return "DT"
    if "CALI" in m or "CALIPER" in text:
        return "CALI"
    if "SP" in m or "SPONTANEOUS" in text:
        return "SP"
    if "RES" in m or "RESISTIVITY" in text or "OHMM" in u or "OHM-M" in u or "OHMM" in text:
        return "RES"
    return None


def build_ai_analysis_payload(las_text, detected_text, user_curves=None):
    """Build a structured payload combining OCR text + LAS summary + simple mapping."""
    if not las_text:
        return None

    # 1) OCR text from Vision
    full_text = ""
    if isinstance(detected_text, str):
        full_text = detected_text
    elif isinstance(detected_text, dict):
        raw_entries = detected_text.get("raw") or []
        texts = [
            (entry.get("text") or "")
            for entry in raw_entries
            if isinstance(entry, dict) and entry.get("text")
        ]
        full_text = "\n".join(texts)

    vision_curve_labels = extract_curve_labels_from_text(full_text)

    # 2) LAS numeric features
    las_summary = summarize_las_curves_from_str(las_text)
    las_curve_mnemonics = []
    if las_summary and las_summary.get("curve_features"):
        las_curve_mnemonics = [cf["curve"] for cf in las_summary["curve_features"]]

    # 3) Map Vision labels to LAS mnemonics
    vision_to_las = match_vision_to_las_curves(vision_curve_labels, las_curve_mnemonics)

    # Optional: basic flags from features
    if las_summary and las_summary.get("curve_features"):
        for cf in las_summary["curve_features"]:
            cf_flags = []
            pct_missing = cf.get("pct_missing") or 0.0
            pct_spikes = cf.get("pct_spikes") or 0.0
            if pct_missing > 30.0:
                cf_flags.append("high_missing")
            if pct_spikes > 5.0:
                cf_flags.append("spiky")
            cf["flags"] = cf_flags

    # 4) User-provided curve config (manual overrides from frontend)
    user_curve_metadata = None
    user_curve_type_by_mnemonic = None
    if user_curves:
        user_curve_metadata = []
        user_curve_type_by_mnemonic = {}
        for idx, c in enumerate(user_curves):
            sel_type = c.get("type")
            las_mnemonic = (c.get("las_mnemonic") or c.get("name") or "").upper()
            entry = {
                "index": idx + 1,
                "selected_type": sel_type,
                "las_mnemonic": las_mnemonic,
                "las_unit": c.get("las_unit") or c.get("unit") or "",
                "display_name": c.get("display_name"),
                "display_unit": c.get("display_unit"),
            }
            user_curve_metadata.append(entry)
            if las_mnemonic:
                user_curve_type_by_mnemonic[las_mnemonic] = sel_type

    payload = {
        "ocr_text": full_text,
        "vision_curve_labels": vision_curve_labels,
        "vision_to_las_mapping": vision_to_las,
        "las_summary": las_summary,
    }

    if user_curve_metadata is not None:
        payload["user_curves"] = user_curve_metadata
        payload["user_curve_type_by_mnemonic"] = user_curve_type_by_mnemonic

    return payload


def call_hf_curve_analysis(ai_payload):
    """Optional: call a Hugging Face text model to get a human-readable curve analysis.

    This is best-effort and will be skipped if credentials are not configured.
    """
    if not ai_payload:
        return None

    system_msg = (
        "You are a petrophysics assistant. Given OCR text from a well log "
        "image and numeric summaries of each LAS curve, identify which LAS "
        "curves are likely GR, RHOB, NPHI, DT, RES, etc. Provide a detailed, "
        "structured markdown report that: (1) explains your methodology for "
        "identifying each curve (OCR labels, value ranges, units, typical scales), "
        "(2) maps each LAS curve to its most likely identity with specific reasoning, "
        "(3) comments on value ranges, units, and typical petrophysical expectations, "
        "(4) highlights data quality issues such as missing data or spikes, and "
        "(5) calls out any unusual depth intervals. Always explain WHY you identified "
        "each curve the way you did. Do not invent curves that are not present."
    )

    prompt = (
        system_msg
        + "\n\nHere is the JSON payload to analyze:\n\n"
        + json.dumps(ai_payload, indent=2)
    )

    # Prefer Gemini if configured
    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            # Use REST API directly to avoid SDK version issues
            # Model ID should include 'models/' prefix (e.g., 'models/gemini-2.0-flash')
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith('models/') else f'models/{GEMINI_MODEL_ID}'
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        text = parts[0].get('text', '')
                        if text:
                            return str(text)
            else:
                print(f"Gemini API error (analysis): {resp.status_code} {resp.text}")
        except Exception as exc:
            print(f"Gemini API error (analysis): {exc}")

    # Fallback to OpenAI if configured
    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(ai_payload, indent=2)},
            ]
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                if content:
                    return str(content)
        except Exception as exc:
            print(f"OpenAI API error (analysis): {exc}")

    # Fallback to Hugging Face Inference if available
    if not HF_API_TOKEN or not HF_MODEL_ID:
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        print(f"HF InferenceClient init error (analysis): {exc}")
        return None

    try:
        out = client.text_generation(
            prompt,
            model=HF_MODEL_ID,
            max_new_tokens=512,
            temperature=0.3,
        )
    except Exception as exc:
        print(f"HF text_generation error (analysis): {exc}")
        return None

    return out if isinstance(out, str) else str(out)


def call_hf_curve_chat(ai_payload, question):
    """Optional: chat-style helper to answer user questions about this log.

    Reuses the same HF model but tailors the prompt to the specific question.
    """
    question = (question or "").strip()
    if not ai_payload or not question:
        return None

    system_msg = (
        "You are a petrophysics assistant. Given OCR text from a well log "
        "image and numeric summaries of each LAS curve, answer the user's "
        "question with a detailed but focused markdown explanation. When "
        "relevant, discuss which curves are likely GR, RHOB, NPHI, DT, RES, "
        "etc., comment on whether values and ranges look reasonable, and refer "
        "to specific depth intervals or data-quality issues. Be precise about "
        "what is supported by the provided payload, and do not invent curves "
        "that are not present."
    )

    payload_text = (
        "Here is the JSON payload describing this log (OCR + LAS):\n\n"
        + json.dumps(ai_payload, indent=2)
        + "\n\nUser question:\n"
        + question
    )

    # Prefer Gemini if configured
    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            # Use REST API directly to avoid SDK version issues
            # Model ID should include 'models/' prefix (e.g., 'models/gemini-2.0-flash')
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith('models/') else f'models/{GEMINI_MODEL_ID}'
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": payload_text}]}]
            }
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        text = parts[0].get('text', '')
                        if text:
                            return str(text)
            else:
                print(f"Gemini API error (chat): {resp.status_code} {resp.text}")
        except Exception as exc:
            print(f"Gemini API error (chat): {exc}")

    # Fallback to OpenAI if configured
    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": payload_text},
            ]
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                if content:
                    return str(content)
        except Exception as exc:
            print(f"OpenAI API error (chat): {exc}")

    # Fallback to Hugging Face Inference if available
    if not HF_API_TOKEN or not HF_MODEL_ID:
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        print(f"HF InferenceClient init error (chat): {exc}")
        return None

    try:
        out = client.text_generation(
            payload_text,
            model=HF_MODEL_ID,
            max_new_tokens=512,
            temperature=0.3,
        )
    except Exception as exc:
        print(f"HF text_generation error (chat): {exc}")
        return f"AI request failed: {str(exc)}"

    return out if isinstance(out, str) else str(out)


def _extract_json_object(text: str):
    """Best-effort helper to parse a single JSON object from a text response.

    Many LLMs sometimes wrap JSON in Markdown or add prose. We first try to
    parse the whole string, then fall back to the first {...} block.
    """
    if not text:
        return None
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def validate_and_fix_calibration(calibration):
    """Validate calibration and fix obvious curve type/scale mismatches."""
    if not isinstance(calibration, dict):
        return calibration
    
    tracks = calibration.get('tracks', [])
    if not tracks:
        return calibration
    
    # Define expected scale ranges for each curve type
    expected_scales = {
        'GR': (0, 150),      # API units
        'RHOB': (1.5, 3.5),  # g/cc
        'NPHI': (-0.2, 0.6), # v/v (porosity)
        'DT': (40, 200),     # us/ft (sonic)
        'CALI': (6, 16),     # inches
        'SP': (-150, 50),    # mV
    }
    
    for track in tracks:
        name = (track.get('name') or '').upper()
        scale_min = track.get('scale_min')
        scale_max = track.get('scale_max')
        
        if not name or not isinstance(scale_min, (int, float)) or not isinstance(scale_max, (int, float)):
            continue
        
        # Check if scale matches the curve type
        if name in expected_scales:
            exp_min, exp_max = expected_scales[name]
            scale_range = scale_max - scale_min
            exp_range = exp_max - exp_min
            
            # If scale is way off (more than 3x different), try to find correct type
            if scale_range > exp_range * 3 or scale_range < exp_range / 3:
                # Try to match scale to correct curve type
                for curve_type, (type_min, type_max) in expected_scales.items():
                    type_range = type_max - type_min
                    # Check if scale is within reasonable bounds for this type
                    if (abs(scale_min - type_min) < type_range * 0.5 and 
                        abs(scale_max - type_max) < type_range * 0.5):
                        print(f"AI calibration: Fixing curve type from {name} to {curve_type} based on scale {scale_min}-{scale_max}")
                        track['name'] = curve_type
                        break
    
    return calibration


def call_ai_calibration(calib_payload):
    """Ask an LLM to propose depth_axis and track calibration JSON.

    calib_payload is a small dict with fields like:
        {
          "image": {"width_px": W, "height_px": H},
          "depth_label_candidates": [{"value": ..., "x_px": ..., "y_px": ...}, ...],
          "header_text_boxes": [{"text": "GR", "x_px": ..., "y_px": ...}, ...],
        }

    Returns a Python dict with optional keys:
        {
          "depth_axis": {
            "top_depth": float,
            "bottom_depth": float,
            "top_pixel": float,
            "bottom_pixel": float,
          },
          "tracks": [
            {
              "name": "GR",
              "left_x": float,
              "right_x": float,
              "scale_min": float,
              "scale_max": float,
              "hot_side": "left" | "right",
              "color_hint": "black" | "red" | "blue" | "green" | null,
            },
            ...
          ]
        }
    """
    if not calib_payload:
        return None

    schema_hint = (
        "You are a petrophysical log calibration assistant. Given OCR-derived "
        "depth label candidates and header text boxes for a single raster log "
        "panel, infer a plausible depth axis and track calibration.\n\n"
        "Always respond with JSON ONLY, no prose, using this schema:\n\n"
        "{\n"
        "  \"depth_axis\": {\n"
        "    \"top_depth\": number,\n"
        "    \"bottom_depth\": number,\n"
        "    \"top_pixel\": number,\n"
        "    \"bottom_pixel\": number\n"
        "  },\n"
        "  \"tracks\": [\n"
        "    {\n"
        "      \"name\": string,\n"
        "      \"left_x\": number,\n"
        "      \"right_x\": number,\n"
        "      \"scale_min\": number,\n"
        "      \"scale_max\": number,\n"
        "      \"hot_side\": \"left\" or \"right\",\n"
        "      \"color_hint\": string or null\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Pixels are in the coordinate system of the provided panel image, where\n"
        "(0,0) is the top-left corner and Y increases downward. Depth should be\n"
        "monotonic with pixel Y (depth increases as Y increases).\n\n"
        "DEPTH AXIS RULES:\n"
        "- Look for depth labels on the LEFT side of the panel (depth_label_candidates)\n"
        "- The SMALLEST depth value should be at the TOP (smallest y_px)\n"
        "- The LARGEST depth value should be at the BOTTOM (largest y_px)\n"
        "- Typical well logs span 10-500 feet per panel\n"
        "- Depth values are usually round numbers (100, 150, 200, etc.)\n"
        "- top_pixel should be the y_px of the topmost depth label\n"
        "- bottom_pixel should be the y_px of the bottommost depth label\n\n"
        "TRACK CALIBRATION RULES:\n"
        "- Match curve names from header_text_boxes to standard petrophysical curves\n"
        "- CRITICAL: The curve NAME must match its SCALE RANGE:\n"
        "  * GR (Gamma Ray): scale_min=0, scale_max=150, units=API\n"
        "  * RHOB (Density): scale_min=1.95, scale_max=2.95, units=g/cc\n"
        "  * NPHI (Neutron Porosity): scale_min=-0.15, scale_max=0.45, units=v/v\n"
        "  * DT (Sonic): scale_min=40, scale_max=140, units=us/ft\n"
        "  * CALI (Caliper): scale_min=6, scale_max=16, units=inches\n"
        "- Common header text variations:\n"
        "  * 'GR', 'GAMMA', 'Gamma Ray' → name='GR'\n"
        "  * 'RHOB', 'DENS', 'Density', 'RHO' → name='RHOB'\n"
        "  * 'NPHI', 'NEUT', 'Neutron', 'PHI' → name='NPHI'\n"
        "  * 'DT', 'SONIC', 'AC', 'Sonic' → name='DT'\n"
        "- Each track should have left_x < right_x\n"
        "- Tracks are ordered left-to-right across the panel\n\n"
        "EXAMPLE: If you see header text 'GR' at x=100, and the track spans x=80-120,\n"
        "then: name='GR', left_x=80, right_x=120, scale_min=0, scale_max=150\n\n"
        "Here is the input JSON you should analyze:\n\n"
    )

    payload_text = schema_hint + json.dumps(calib_payload, indent=2)

    # Prefer Gemini if configured
    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith("models/") else f"models/{GEMINI_MODEL_ID}"
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            body = {"contents": [{"parts": [{"text": payload_text}]}]}
            resp = requests.post(url, json=body, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        calib = _extract_json_object(text)
                        if isinstance(calib, dict):
                            return calib
            else:
                print(f"Gemini API error (calibration): {resp.status_code} {resp.text}")
        except Exception as exc:
            print(f"Gemini API error (calibration): {exc}")

    # Fallback to OpenAI if configured
    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            messages = [
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": payload_text},
            ]
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                calib = _extract_json_object(content)
                if isinstance(calib, dict):
                    return calib
        except Exception as exc:
            print(f"OpenAI API error (calibration): {exc}")

    # Fallback to Hugging Face text-generation if available
    if not HF_API_TOKEN or not HF_MODEL_ID:
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        print(f"HF InferenceClient init error (calibration): {exc}")
        return None

    try:
        out = client.text_generation(
            payload_text,
            model=HF_MODEL_ID,
            max_new_tokens=512,
            temperature=0.1,
        )
        calib = _extract_json_object(out if isinstance(out, str) else str(out))
        if isinstance(calib, dict):
            return calib
    except Exception as exc:
        print(f"HF text_generation error (calibration): {exc}")

    return None


def call_ai_auto_layout(layout_payload):
    """Ask an LLM to infer logging track layout from header text items.

    layout_payload is a small dict with fields like:
        {
          "image": {"width_px": W, "height_px": H},
          "items": [
            {"text": "GR", "x": 650, "y": 120},
            {"text": "0", "x": 620, "y": 140},
            ...
          ]
        }

    The model should return:
        {
          "tracks": [
            {
              "name": "GR",
              "left_x": float,
              "right_x": float,
              "scale_min": number or null,
              "scale_max": number or null,
              "unit": string or null,
              "hot_side": "left" or "right" or null
            },
            ...
          ]
        }
    """
    if not layout_payload:
        return None

    schema_hint = (
        "You are analyzing the HEADER of a raster well log. The user has "
        "cropped the top portion of a single log panel. You see short text "
        "items (curve mnemonics and scale labels) with approximate x/y "
        "centers in pixels, and a 'full_text' block containing all recognized text.\n\n"
        "Your job is to:\n"
        "1. Infer the logging TRACKS present across the width of the header.\n"
        "2. Extract generic HEADER METADATA (Well, Company, API, etc.) from the 'full_text'.\n\n"
        "Pixels are in the coordinate system of the provided header image, "
        "where x=0 is the left edge and x increases to the right. The overall "
        "image width in pixels is image.width_px.\n\n"
        "OUTPUT FORMAT (JSON ONLY):\n"
        "{\n"
        "  \"tracks\": [\n"
        "    {\n"
        "      \"name\": string,                    // e.g. \"GR\", \"RHOB\", \"NPHI\"\n"
        "      \"left_x\": number,                  // approximate left boundary of this track in pixels\n"
        "      \"right_x\": number,                 // approximate right boundary of this track in pixels\n"
        "      \"scale_min\": number or null,       // inferred scale min if obvious\n"
        "      \"scale_max\": number or null,       // inferred scale max if obvious\n"
        "      \"unit\": string or null,            // e.g. \"API\", \"G/CC\", \"V/V\", \"US/F\"\n"
        "      \"hot_side\": \"left\" | \"right\" | null  // which side is higher / hot values\n"
        "    }\n"
        "  ],\n"
        "  \"header_metadata\": {\n"
        "    \"well\": string or null,\n"
        "    \"company\": string or null,\n"
        "    \"api\": string or null,\n"
        "    \"date\": string or null,\n"
        "    \"field\": string or null,\n"
        "    \"location\": string or null,\n"
        "    \"county\": string or null,\n"
        "    \"state\": string or null,\n"
        "    \"province\": string or null,\n"
        "    \"service_company\": string or null\n"
        "  }\n"
        "}\n\n"
        "GUIDELINES:\n"
        "- Group header items with similar x positions into the same track.\n"
        "- Track NAME should follow standard petrophysical conventions:\n"
        "  * GR (Gamma Ray)\n"
        "  * RHOB (Density)\n"
        "  * NPHI (Neutron Porosity)\n"
        "  * DT (Sonic)\n"
        "  * CALI (Caliper)\n"
        "  * SP (Spontaneous Potential)\n"
        "- Use typical scale ranges when you see numeric labels near a curve name:\n"
        "  * GR:   ~0–150 API\n"
        "  * RHOB: ~1.95–2.95 g/cc\n"
        "  * NPHI: ~-0.15–0.45 v/v\n"
        "  * DT:   ~40–140 us/ft\n"
        "  * CALI: ~6–16 in\n"
        "- Infer left_x/right_x by placing boundaries midway between adjacent "
        "curve label centers along the x-axis.\n"
        "- Ensure left_x < right_x and tracks are ordered left-to-right.\n"
        "- For header_metadata, look for text items like 'WELL:', 'COMPANY:', 'API:', etc.\n"
        "  and try to associate the value next to them. If not found, use null.\n"
        "- If you are unsure about scale_min/scale_max or unit, use null.\n\n"
        "Here is the input JSON you should analyze:\n\n"
    )

    payload_text = schema_hint + json.dumps(layout_payload, indent=2)

    # Prefer Gemini if configured
    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith("models/") else f"models/{GEMINI_MODEL_ID}"
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            body = {"contents": [{"parts": [{"text": payload_text}]}]}
            resp = requests.post(url, json=body, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        layout = _extract_json_object(text)
                        if isinstance(layout, dict):
                            return layout
            else:
                print(f"Gemini API error (auto_layout): {resp.status_code} {resp.text}")
        except Exception as exc:
            print(f"Gemini API error (auto_layout): {exc}")

    # Fallback to OpenAI if configured
    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            messages = [
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": payload_text},
            ]
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                layout = _extract_json_object(content)
                if isinstance(layout, dict):
                    return layout
        except Exception as exc:
            print(f"OpenAI API error (auto_layout): {exc}")

    # Fallback to Hugging Face text-generation if available
    if not HF_API_TOKEN or not HF_MODEL_ID:
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        print(f"HF InferenceClient init error (auto_layout): {exc}")
        return None

    try:
        out = client.text_generation(
            payload_text,
            model=HF_MODEL_ID,
            max_new_tokens=512,
            temperature=0.1,
        )
        layout = _extract_json_object(out if isinstance(out, str) else str(out))
        if isinstance(layout, dict):
            return layout
    except Exception as exc:
        print(f"HF text_generation error (auto_layout): {exc}")

    return None


def hsv_red_mask(hsv_img):
    lower1, upper1 = np.array([0, 80, 80]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 80, 80]), np.array([180, 255, 255])
    return cv2.bitwise_or(
        cv2.inRange(hsv_img, lower1, upper1),
        cv2.inRange(hsv_img, lower2, upper2),
    )


def hsv_blue_mask(hsv_img):
    lower, upper = np.array([90, 80, 80]), np.array([140, 255, 255])
    return cv2.inRange(hsv_img, lower, upper)


def hsv_green_mask(hsv_img):
    lower, upper = np.array([40, 80, 80]), np.array([90, 255, 255])
    return cv2.inRange(hsv_img, lower, upper)


def black_mask(gray_img):
    return cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        10,
    )

def preprocess_curve_track(roi, mode="black"):
    """Clean up a curve track ROI: isolate curve color, remove gridlines, thin.
    
    Args:
        roi: BGR image crop of the track
        mode: "black", "red", "blue", or "green"
    
    Returns:
        Binary mask where curve pixels are 255, background is 0
    """
    if roi is None or roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Step 1: Color isolation
    if mode == "black":
        # Low brightness = curve
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        _, curve_mask = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY_INV)
    elif mode == "red":
        b, g, r = cv2.split(roi)
        curve_mask = ((r > 120) & (r > g + 20) & (r > b + 20)).astype(np.uint8) * 255
    elif mode == "blue":
        b, g, r = cv2.split(roi)
        curve_mask = ((b > 120) & (b > r + 20) & (b > g + 20)).astype(np.uint8) * 255
    elif mode == "green":
        b, g, r = cv2.split(roi)
        curve_mask = ((g > 120) & (g > r + 20) & (g > b + 20)).astype(np.uint8) * 255
    else:
        # Fallback: use existing black mask logic
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        _, curve_mask = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Step 2: Remove vertical gridlines
    if h > 15:
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(15, h // 3)))
        vert_lines = cv2.morphologyEx(curve_mask, cv2.MORPH_OPEN, vertical_kernel)
        curve_mask = cv2.bitwise_and(curve_mask, cv2.bitwise_not(vert_lines))
    
    # Step 3: Remove horizontal gridlines
    if w > 15:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min(15, w // 3), 1))
        horiz_lines = cv2.morphologyEx(curve_mask, cv2.MORPH_OPEN, horizontal_kernel)
        curve_mask = cv2.bitwise_and(curve_mask, cv2.bitwise_not(horiz_lines))
    
    # Step 4: Slight blur to fill 1-pixel gaps
    blurred = cv2.GaussianBlur(curve_mask, (3, 3), 0)
    _, cleaned = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Remove near-solid vertical "spines" that look like grid/border lines.
    # These are typically track borders or grid lines, not the actual log curve,
    # and they can cause the DP tracer to hug the wrong column.
    if h >= 40 and w >= 4:
        col_fraction = np.mean(cleaned > 0, axis=0)  # fraction of rows that are "on" per column
        edge_margin = max(1, int(0.15 * w))
        edge_mask = np.zeros_like(col_fraction, dtype=bool)
        edge_mask[:edge_margin] = True
        edge_mask[-edge_margin:] = True

        # Strong spines near the left/right edges (likely track borders)
        # Use 0.90 threshold: true borders span ~100% of height; slow curves (RHOB/DTC)
        # can occupy 40-80% of one column and must not be removed.
        edge_spines = (col_fraction > 0.90) & edge_mask

        # Very strong vertical spines anywhere inside the band.
        interior_spines = (col_fraction > 0.90) & ~edge_mask

        spine_cols = edge_spines | interior_spines
        if np.any(spine_cols):
            cleaned[:, spine_cols] = 0

    return cleaned


def detect_dominant_curve_hue(roi_bgr, sample_fraction=0.3):
    """Detect the dominant hue of the curve in a sample region.
    
    Samples the middle portion of the image (where the curve is likely to be)
    and finds the most common saturated hue, excluding near-white/black pixels.
    
    Returns:
        (hue_center, hue_range) or None if no dominant hue found
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    
    h, w = roi_bgr.shape[:2]
    if h < 10 or w < 10:
        return None
    
    # Sample the middle portion of the image (where curve likely is)
    x_start = int(w * (0.5 - sample_fraction / 2))
    x_end = int(w * (0.5 + sample_fraction / 2))
    sample_region = roi_bgr[:, x_start:x_end]
    
    hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
    
    # Filter for saturated, non-white, non-black pixels (likely curve pixels)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    
    # Curve pixels: reasonably saturated and not too dark or bright
    curve_mask = (saturation > 50) & (value > 40) & (value < 240)
    
    if np.sum(curve_mask) < 100:
        return None
    
    # Get hues of curve pixels
    curve_hues = hue[curve_mask]
    
    # Build histogram of hues (0-180 in OpenCV)
    hist, bins = np.histogram(curve_hues, bins=36, range=(0, 180))
    
    # Find the dominant hue bin (excluding very low counts)
    threshold = np.max(hist) * 0.3
    dominant_bins = np.where(hist > threshold)[0]
    
    if len(dominant_bins) == 0:
        return None
    
    # Use the bin with highest count
    peak_bin = np.argmax(hist)
    hue_center = (bins[peak_bin] + bins[peak_bin + 1]) / 2
    
    # Adaptive hue range based on how spread the hues are
    hue_std = np.std(curve_hues)
    hue_range = max(10, min(25, hue_std * 2))
    
    return (float(hue_center), float(hue_range))


def apply_local_contrast_normalization(img_bgr):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    faded sections while preserving color information.
    
    This normalizes brightness in small windows so faded curve sections
    don't get lost during detection.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    
    # Convert to LAB color space (L = lightness, A/B = color)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to the L (lightness) channel only
    # clipLimit controls contrast amplification (lower = less noise amplification)
    # tileGridSize controls the window size for local normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    # Merge back and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # SATURATION BOOST: "Up the contrast" for color
    # Make faint ink look vibrant
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.5) # Boost saturation by 50%
    s = np.clip(s, 0, 255).astype(np.uint8)
    enhanced_bgr = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    return enhanced_bgr


def remove_grid_lines_aggressive(gray_img, aggressive=True):
    """
    Aggressively detect and remove grid lines from black and white log images.
    Returns a mask with grid lines removed.
    
    Args:
        gray_img: Grayscale image
        aggressive: If True, use very aggressive grid detection
    """
    if gray_img is None or gray_img.size == 0:
        return gray_img
    
    h, w = gray_img.shape[:2]
    if h < 20 or w < 20:
        return gray_img
    
    # Create a copy to work with
    result = gray_img.copy()
    
    # Detect vertical lines (most common in grid)
    if aggressive:
        # Very aggressive vertical line detection
        v_kernel_size = max(5, min(30, h // 5))  # Smaller kernel = more aggressive detection of broken lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        v_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, v_kernel)
        
        # Detect horizontal lines
        h_kernel_size = max(5, min(30, w // 5))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        h_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, h_kernel)
        
        # Combine detected lines
        grid_lines = cv2.bitwise_or(v_lines, h_lines)
        
        # Dilate slightly to ensure complete removal
        dilate_kernel = np.ones((3, 3), np.uint8)  # Larger dilation to wipe out the intersection bleed
        grid_lines = cv2.dilate(grid_lines, dilate_kernel, iterations=1)
        
        # Remove grid lines from original
        result = cv2.subtract(result, grid_lines)
    else:
        # Standard grid removal
        v_kernel_size = max(10, min(60, h // 2))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        v_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, v_kernel)
        
        h_kernel_size = max(10, min(60, w // 2))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        h_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, h_kernel)
        
        grid_lines = cv2.bitwise_or(v_lines, h_lines)
        result = cv2.subtract(result, grid_lines)
    
    return result


def detect_if_black_and_white_log(roi_bgr):
    """
    Auto-detect if an image is a black and white log (vs colored).
    Returns True if the image appears to be black and white.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    
    try:
        # Convert to HSV and check saturation
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # Calculate percentage of low-saturation pixels
        low_sat_pixels = np.sum(saturation < 30)
        total_pixels = saturation.size
        low_sat_ratio = low_sat_pixels / max(1, total_pixels)
        
        # If >90% of pixels have low saturation, it's likely black and white
        return low_sat_ratio > 0.90
    except Exception:
        return False


def compute_black_curve_residual(gray_img):
    """Estimate dark curve ink after subtracting long straight grid lines.

    Returns a tuple of float32 images in [0, 1]:
      - residual_score: dark-ink signal with most straight grid energy removed
      - grid_score: confidence that a pixel belongs to a long straight grid line
    """
    if gray_img is None or gray_img.size == 0:
        z = np.zeros((1, 1), dtype=np.float32)
        return z, z

    h, w = gray_img.shape[:2]
    if h < 3 or w < 3:
        z = np.zeros((h, w), dtype=np.float32)
        return z, z

    try:
        dark = cv2.GaussianBlur(255 - gray_img, (3, 3), 0)
    except Exception:
        dark = (255 - gray_img).astype(np.uint8, copy=False)

    # Use grayscale morphology to model the repeated straight grid pattern.
    # The curve itself is jagged, so it usually does not survive these long,
    # axis-aligned openings as strongly as the grid does.
    k_v = max(31, min(111, h // 90 if h >= 90 else h - (1 - h % 2)))
    k_h = max(21, min(71, w // 20 if w >= 20 else w - (1 - w % 2)))
    k_v = max(3, int(k_v))
    k_h = max(3, int(k_h))

    try:
        v_lines = cv2.morphologyEx(
            dark, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_v))
        )
        h_lines = cv2.morphologyEx(
            dark, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1))
        )
        grid = cv2.max(v_lines, h_lines)
        grid = cv2.GaussianBlur(grid, (3, 3), 0)
        residual = cv2.subtract(dark, grid)
    except Exception:
        z = np.zeros((h, w), dtype=np.float32)
        return z, z

    residual_score = residual.astype(np.float32)
    grid_score = grid.astype(np.float32)

    r_max = float(residual_score.max())
    if r_max > 0:
        residual_score /= r_max
    else:
        residual_score.fill(0.0)

    g_max = float(grid_score.max())
    if g_max > 0:
        grid_score /= g_max
    else:
        grid_score.fill(0.0)

    return residual_score, grid_score


def build_black_prescan_grid_removed(gray_img):
    """Build a grid-removed grayscale image and residual scores for black logs."""
    if gray_img is None or gray_img.size == 0:
        return gray_img, None, None, None

    h, w = gray_img.shape[:2]
    if h < 3 or w < 3:
        z = np.zeros((h, w), dtype=np.float32)
        return gray_img, z, z, np.zeros((h, w), dtype=np.uint8)

    try:
        residual_score, grid_score = compute_black_curve_residual(gray_img)
    except Exception:
        residual_score = None
        grid_score = None

    try:
        dark = (255 - gray_img).astype(np.uint8, copy=False)
        dark = cv2.GaussianBlur(dark, (3, 3), 0)

        grid_parts = []
        # Use multiple scales: the fine kernels catch thin minor grid lines;
        # the long kernels catch continuous track rails and major grid bars.
        vertical_sizes = {
            max(9, min(45, h // 10)),
            max(21, min(95, h // 4)),
            max(41, min(181, h // 2)),
        }
        horizontal_sizes = {
            max(9, min(45, w // 10)),
            max(21, min(95, w // 4)),
            max(41, min(181, w // 2)),
        }
        for size in sorted(vertical_sizes):
            if size >= 3:
                kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(size)))
                grid_parts.append(cv2.morphologyEx(dark, cv2.MORPH_OPEN, kern))
        for size in sorted(horizontal_sizes):
            if size >= 3:
                kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(size), 1))
                grid_parts.append(cv2.morphologyEx(dark, cv2.MORPH_OPEN, kern))

        if grid_parts:
            grid_u8 = grid_parts[0]
            for part in grid_parts[1:]:
                grid_u8 = cv2.max(grid_u8, part)
            grid_u8 = cv2.dilate(grid_u8, np.ones((3, 3), np.uint8), iterations=1)
        else:
            grid_u8 = np.zeros_like(dark)

        residual_u8 = cv2.subtract(dark, grid_u8)
        if residual_score is not None and residual_score.shape[:2] == (h, w):
            residual_from_score = np.clip(residual_score * 255.0, 0, 255).astype(np.uint8)
            residual_u8 = cv2.max(residual_u8, residual_from_score)

        residual_u8 = cv2.GaussianBlur(residual_u8, (3, 3), 0)
        grid_removed_gray = (255 - residual_u8).astype(np.uint8)

        residual_peak = int(residual_u8.max())
        residual_floor = max(18, int(round(residual_peak * 0.20)))
        _, residual_mask = cv2.threshold(
            residual_u8,
            residual_floor,
            255,
            cv2.THRESH_BINARY,
        )
        try:
            if float(np.mean(residual_mask > 0)) < 0.003:
                adaptive_mask = cv2.adaptiveThreshold(
                    grid_removed_gray,
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV,
                    21,
                    4,
                )
                adaptive_floor = max(12, int(round(residual_floor * 0.75)))
                adaptive_mask = cv2.bitwise_and(
                    adaptive_mask,
                    ((residual_u8 >= adaptive_floor).astype(np.uint8) * 255),
                )
                residual_mask = cv2.bitwise_or(residual_mask, adaptive_mask)
            _, otsu_mask = cv2.threshold(
                residual_u8,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            otsu_coverage = float(np.mean(otsu_mask > 0))
            if 0.0005 <= otsu_coverage <= 0.20:
                residual_mask = cv2.bitwise_or(residual_mask, otsu_mask)
        except Exception:
            pass

        if grid_score is None or grid_score.shape[:2] != (h, w):
            grid_score = grid_u8.astype(np.float32)
            gmax = float(grid_score.max())
            if gmax > 0:
                grid_score /= gmax
        return grid_removed_gray, residual_score, grid_score, residual_mask
    except Exception:
        return gray_img, residual_score, grid_score, None


def _trace_debug_image_data_url(img, max_side=900, ext='.jpg'):
    """Encode a debug image as a small browser-friendly data URL."""
    if img is None:
        return None
    try:
        arr = np.asarray(img)
        if arr.size == 0:
            return None
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32)
            arr_f = arr_f - float(np.nanmin(arr_f))
            max_val = float(np.nanmax(arr_f))
            if max_val > 0:
                arr_f = arr_f / max_val
            arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            enc_img = arr
        else:
            enc_img = arr[:, :, :3]

        h, w = enc_img.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w, 1)))
        if scale < 1.0:
            enc_img = cv2.resize(
                enc_img,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        params = []
        mime = 'image/jpeg'
        if ext.lower() == '.png':
            mime = 'image/png'
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
        else:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 82]

        ok, buf = cv2.imencode(ext, enc_img, params)
        if not ok:
            return None
        b64 = base64.b64encode(buf).decode('ascii')
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _draw_trace_debug_overlay(roi_bgr, xs, guide_radius=12):
    if roi_bgr is None or xs is None or not hasattr(xs, "size"):
        return None
    try:
        if roi_bgr.ndim == 2:
            overlay = cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)
        else:
            overlay = roi_bgr.copy()
        h, w = overlay.shape[:2]
        band = overlay.copy()
        for y in range(min(h, xs.size)):
            x = xs[y]
            if not np.isfinite(x):
                continue
            ix = int(np.clip(round(float(x)), 0, w - 1))
            cv2.line(
                band,
                (max(0, ix - int(guide_radius)), y),
                (min(w - 1, ix + int(guide_radius)), y),
                (0, 180, 255),
                1,
            )
        overlay = cv2.addWeighted(band, 0.32, overlay, 0.68, 0)

        pts = []
        for y in range(min(h, xs.size)):
            x = xs[y]
            if np.isfinite(x):
                pts.append((int(np.clip(round(float(x)), 0, w - 1)), int(y)))
            elif len(pts) > 1:
                cv2.polylines(overlay, [np.asarray(pts, dtype=np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
                pts = []
            else:
                pts = []
        if len(pts) > 1:
            cv2.polylines(overlay, [np.asarray(pts, dtype=np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
        return overlay
    except Exception:
        return None


def _component_debug_image_and_stats(binary_mask):
    if binary_mask is None:
        return None, []
    try:
        mask = (binary_mask > 0).astype(np.uint8)
        if not np.any(mask):
            return np.zeros((*mask.shape, 3), dtype=np.uint8), []
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        comp_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
        rows = []
        for label in range(1, n_labels):
            x, y, w, h, area = [int(v) for v in stats[label]]
            if area <= 0:
                continue
            color = (
                int((37 * label) % 255),
                int((91 * label) % 255),
                int((157 * label) % 255),
            )
            comp_img[labels == label] = color
            rows.append({
                'label': int(label),
                'area': int(area),
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'aspect': float(w / max(1, h)),
                'vertical_span': int(h),
                'horizontal_span': int(w),
            })
        rows.sort(key=lambda r: r['area'], reverse=True)
        return comp_img, rows[:25]
    except Exception:
        return None, []


def build_black_trace_debug_export(curve_name, roi_bgr, prob_mask, xs, curve_type=None, mode=None):
    """Build visual trace-debug artifacts for black curve failures."""
    if roi_bgr is None or prob_mask is None or xs is None:
        return None
    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    try:
        grid_removed, residual_score, grid_score, residual_mask = build_black_prescan_grid_removed(gray)
        residual_u8 = None
        if residual_score is not None:
            residual_u8 = np.clip(residual_score * 255.0, 0, 255).astype(np.uint8)
        grid_u8 = None
        if grid_score is not None:
            grid_u8 = np.clip(grid_score * 255.0, 0, 255).astype(np.uint8)

        comp_img, component_stats = _component_debug_image_and_stats(residual_mask)
        overlay = _draw_trace_debug_overlay(roi_bgr, xs)
        prob_color = cv2.applyColorMap(np.asarray(prob_mask, dtype=np.uint8), cv2.COLORMAP_TURBO)
        residual_color = cv2.applyColorMap(residual_u8, cv2.COLORMAP_TURBO) if residual_u8 is not None else None
        grid_color = cv2.applyColorMap(grid_u8, cv2.COLORMAP_TURBO) if grid_u8 is not None else None

        finite = np.isfinite(xs)
        dx = np.diff(xs[finite].astype(np.float32)) if np.count_nonzero(finite) > 1 else np.asarray([], dtype=np.float32)
        metrics = {
            'roi_shape': [int(v) for v in roi_bgr.shape[:2]],
            'finite_rows': int(np.count_nonzero(finite)),
            'nan_rows': int(xs.size - np.count_nonzero(finite)),
            'max_abs_dx': float(np.nanmax(np.abs(dx))) if dx.size else 0.0,
            'mean_abs_dx': float(np.nanmean(np.abs(dx))) if dx.size else 0.0,
            'residual_mask_coverage': float(np.mean(residual_mask > 0)) if residual_mask is not None else None,
            'component_count': int(len(component_stats)),
        }

        return {
            'curve': curve_name,
            'curve_type': curve_type,
            'mode': mode,
            'metrics': metrics,
            'components_top': component_stats,
            'images': {
                'roi': _trace_debug_image_data_url(roi_bgr, ext='.jpg'),
                'overlay': _trace_debug_image_data_url(overlay, ext='.jpg'),
                'prob_map': _trace_debug_image_data_url(prob_color, ext='.jpg'),
                'grid_removed': _trace_debug_image_data_url(grid_removed, ext='.jpg'),
                'residual_mask': _trace_debug_image_data_url(residual_mask, ext='.png'),
                'residual_score': _trace_debug_image_data_url(residual_color, ext='.jpg'),
                'grid_score': _trace_debug_image_data_url(grid_color, ext='.jpg'),
                'components': _trace_debug_image_data_url(comp_img, ext='.jpg'),
            },
        }
    except Exception as exc:
        return {
            'curve': curve_name,
            'curve_type': curve_type,
            'mode': mode,
            'error': str(exc),
        }


def enhance_curve_roi(roi_bgr):
    """
    Apply lightweight denoise + horizontal super-resolution to a curve ROI.
    Returns (processed_roi, horizontal_scale_factor).
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr, 1.0
    proc = roi_bgr
    scale = 1.0
    try:
        proc = cv2.bilateralFilter(proc, d=5, sigmaColor=25, sigmaSpace=12)
    except Exception:
        pass
    if CURVE_TRACE_UPSCALE > 1.01 and proc.shape[1] >= 2:
        h, w = proc.shape[:2]
        new_w = max(2, int(round(w * CURVE_TRACE_UPSCALE)))
        try:
            proc = cv2.resize(proc, (new_w, h), interpolation=cv2.INTER_CUBIC)
            scale = new_w / max(1, w)
        except Exception:
            scale = 1.0
    try:
        # Mild sharpening to boost ink contrast after upscaling
        blur = cv2.GaussianBlur(proc, (0, 0), sigmaX=0.8)
        proc = cv2.addWeighted(proc, 1.2, blur, -0.2, 0)
    except Exception:
        pass
    return proc, scale


def suppress_grid_hough(gray, h_thresh_ratio=0.25, v_thresh_ratio=0.25):
    """
    Use Probabilistic Hough Transform to detect and remove long straight grid lines
    while preserving jagged curve data.
    """
    if gray is None:
        return gray
        
    h, w = gray.shape
    # Pre-process for edge detection
    # Use adaptive threshold to get binary edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    # 1. Horizontal Lines
    min_len_h = int(w * h_thresh_ratio)
    lines_h = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=min_len_h, maxLineGap=10)
    
    # 2. Vertical Lines
    min_len_v = int(h * v_thresh_ratio)
    lines_v = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=min_len_v, maxLineGap=10)
    
    # Mask to draw the lines to be removed
    grid_mask = np.zeros_like(gray)
    
    if lines_h is not None:
        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2-y1, x2-x1) * 180 / np.pi)
            # Strict horizontal check (+/- 2 degrees)
            if angle < 2 or angle > 178:
                cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 2)
                
    if lines_v is not None:
        for line in lines_v:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2-y1, x2-x1) * 180 / np.pi)
            # Strict vertical check (+/- 2 degrees from 90)
            if 88 < angle < 92:
                cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 2)
                
    # Dilate mask slightly to clean up edge artifacts
    grid_mask = cv2.dilate(grid_mask, np.ones((3,3), np.uint8), iterations=1)
    
    # Inpaint/Erase grid lines (set to white)
    cleaned = gray.copy()
    cleaned[grid_mask > 0] = 255
    
    return cleaned


def compute_prob_map(roi_bgr, mode="black", ui_filters=None, _dual_polarity_allowed=True):
    """Build a soft probability map for the curve in a track ROI.

    Returns an 8-bit image (0–255) where higher values mean higher likelihood
    of belonging to the curve. This can be fed directly to the existing
    DP tracer, which internally rescales to [0, 1].
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    h, w = roi_bgr.shape[:2]
    if h < 2 or w < 2:
        return np.zeros((h, w), dtype=np.uint8)

    ui_filters = ui_filters or {}
    try:
        use_contrast = bool(ui_filters.get('contrast'))
        use_invert = bool(ui_filters.get('invert'))
    except Exception:
        use_contrast = False
        use_invert = False

    enable_grid_suppression = ui_filters.get('enable_grid_suppression', True)
    enable_curve_masking = ui_filters.get('enable_curve_masking', True)

    if _dual_polarity_allowed and mode == "auto" and use_invert:
        try:
            ui_filters_no_invert = dict(ui_filters)
        except Exception:
            ui_filters_no_invert = {}
        ui_filters_no_invert['invert'] = False
        try:
            inv_roi = cv2.bitwise_not(roi_bgr)
        except Exception:
            inv_roi = np.clip(255 - roi_bgr, 0, 255).astype(np.uint8)
        prob_a = compute_prob_map(
            roi_bgr,
            mode=mode,
            ui_filters=ui_filters_no_invert,
            _dual_polarity_allowed=False,
        )
        prob_b = compute_prob_map(
            inv_roi,
            mode=mode,
            ui_filters=ui_filters_no_invert,
            _dual_polarity_allowed=False,
        )
        try:
            if prob_a is None:
                return prob_b
            if prob_b is None:
                return prob_a
            if prob_a.shape != prob_b.shape:
                return prob_a
            return np.maximum(prob_a, prob_b)
        except Exception:
            return prob_a

    roi_pre = roi_bgr
    if use_invert and mode == "auto":
        try:
            roi_pre = cv2.bitwise_not(roi_pre)
        except Exception:
            roi_pre = np.clip(255 - roi_pre, 0, 255).astype(np.uint8)
    if use_contrast:
        try:
            alpha = 2.0
            roi_f = roi_pre.astype(np.float32)
            roi_f = (roi_f - 128.0) * alpha + 128.0
            roi_pre = np.clip(roi_f, 0, 255).astype(np.uint8)
        except Exception:
            pass

    roi_enhanced = apply_local_contrast_normalization(roi_pre)
    
    # Use enhanced image for HSV and grayscale conversion
    hsv = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2GRAY)
    
    # Also apply CLAHE directly to grayscale for edge detection
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    trace_gray = gray

    # 1) Base color/intensity mask
    # For colored modes, try to detect the actual curve hue for better tracking
    # Use enhanced image for better hue detection in faded areas
    colored_modes = {"green", "red", "blue", "auto", "cyan", "magenta", "yellow", "orange", "purple"}
    detected_hue = None
    black_residual_score = None
    black_grid_score = None
    if mode in colored_modes:
        detected_hue = detect_dominant_curve_hue(roi_enhanced)
    
    if mode == "green":
        # PERMISSIVE green detection again - we need to catch the faint tips
        # Lower saturation threshold to catch lighter greens as seen in the user's image
        lower = np.array([25, 25, 25], dtype=np.uint8)
        upper = np.array([95, 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower, upper)

        # Suppress red/orange pixels
        b, g, r = cv2.split(roi_enhanced)
        r16 = r.astype(np.int16)
        g16 = g.astype(np.int16)
        b16 = b.astype(np.int16)

        # Only suppress clearly red pixels
        clearly_red = (r16 > g16 + 30) & (r16 > b16 + 30)
        color_mask[clearly_red] = 0

        # Weak G-dominance check (allow if G is just slightly higher or equal)
        # Relaxed even further to allow more light green/gray-green
        g_dominant = (g16 >= r16 - 15) & (g16 >= b16 - 15)
        color_mask[~g_dominant] = 0

    elif mode == "auto":
        # Auto-detect the curve hue and track it
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            h_lo = max(0, int(hue_center - hue_range))
            h_hi = min(180, int(hue_center + hue_range))
            lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, lower, upper)
            
            # Refine with median hue from detected pixels
            nonzero = np.nonzero(color_mask)
            if len(nonzero[0]) > 50:
                h_channel = hsv[:, :, 0]
                valid_h = h_channel[nonzero]
                med_h = float(np.median(valid_h))
                band = 12.0
                h_lo = max(0, int(med_h - band))
                h_hi = min(180, int(med_h + band))
                dyn_lower = np.array([h_lo, 45, 45], dtype=np.uint8)
                dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
                color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)
        else:
            # Fallback to detecting any saturated colored pixels
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            color_mask = ((saturation > 50) & (value > 40) & (value < 240)).astype(np.uint8) * 255

    elif mode == "red":
        # Red wraps around hue 0/180, so handle both ends
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            # Only use detected hue if it's in the red-ish range (0-15 or 160-180)
            if hue_center <= 20 or hue_center >= 155:
                # Use detected hue with adaptive range
                if hue_center <= 20:
                    h_lo = max(0, int(hue_center - hue_range))
                    h_hi = min(25, int(hue_center + hue_range))
                    lower1 = np.array([h_lo, 60, 40], dtype=np.uint8)
                    upper1 = np.array([h_hi, 255, 255], dtype=np.uint8)
                    color_mask = cv2.inRange(hsv, lower1, upper1)
                else:
                    h_lo = max(150, int(hue_center - hue_range))
                    h_hi = min(180, int(hue_center + hue_range))
                    lower2 = np.array([h_lo, 60, 40], dtype=np.uint8)
                    upper2 = np.array([h_hi, 255, 255], dtype=np.uint8)
                    color_mask = cv2.inRange(hsv, lower2, upper2)
            else:
                # Detected hue outside red range, use default
                lower1 = np.array([0, 70, 40], dtype=np.uint8)
                upper1 = np.array([15, 255, 255], dtype=np.uint8)
                lower2 = np.array([160, 70, 40], dtype=np.uint8)
                upper2 = np.array([180, 255, 255], dtype=np.uint8)
                m1 = cv2.inRange(hsv, lower1, upper1)
                m2 = cv2.inRange(hsv, lower2, upper2)
                color_mask = cv2.bitwise_or(m1, m2)
        else:
            lower1 = np.array([0, 70, 40], dtype=np.uint8)
            upper1 = np.array([15, 255, 255], dtype=np.uint8)
            lower2 = np.array([160, 70, 40], dtype=np.uint8)
            upper2 = np.array([180, 255, 255], dtype=np.uint8)
            m1 = cv2.inRange(hsv, lower1, upper1)
            m2 = cv2.inRange(hsv, lower2, upper2)
            color_mask = cv2.bitwise_or(m1, m2)
        
        # Refine with median hue from detected pixels
        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 12.0
            # Handle red's wrap-around at 0/180
            if med_h <= 20:
                h_lo = max(0, int(med_h - band))
                h_hi = min(30, int(med_h - band))
                dyn_lower = np.array([h_lo, 60, 40], dtype=np.uint8)
                dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
                color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)
            elif med_h >= 160:
                h_lo = max(150, int(med_h - band))
                h_hi = min(180, int(med_h + band))
                dyn_lower = np.array([h_lo, 60, 40], dtype=np.uint8)
                dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
                color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "blue":
        # Use detected hue if available, otherwise fall back to fixed blue range
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            # Only use detected hue if it's in the blue-ish range (90-140)
            if 85 <= hue_center <= 145:
                h_lo = max(80, int(hue_center - hue_range))
                h_hi = min(150, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([90, 40, 40], dtype=np.uint8)
                upper = np.array([140, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([90, 40, 40], dtype=np.uint8)
            upper = np.array([140, 255, 255], dtype=np.uint8)
        
        color_mask = cv2.inRange(hsv, lower, upper)
        
        # Refine with median hue from detected pixels
        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 12.0
            h_lo = max(80, int(med_h - band))
            h_hi = min(150, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "cyan":
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            if 70 <= hue_center <= 110:
                h_lo = max(60, int(hue_center - hue_range))
                h_hi = min(120, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([70, 40, 40], dtype=np.uint8)
                upper = np.array([110, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([70, 40, 40], dtype=np.uint8)
            upper = np.array([110, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 12.0
            h_lo = max(60, int(med_h - band))
            h_hi = min(120, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "yellow":
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            if 10 <= hue_center <= 55:
                h_lo = max(0, int(hue_center - hue_range))
                h_hi = min(70, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([15, 40, 40], dtype=np.uint8)
                upper = np.array([45, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([15, 40, 40], dtype=np.uint8)
            upper = np.array([45, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 10.0
            h_lo = max(0, int(med_h - band))
            h_hi = min(70, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "orange":
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            if 0 <= hue_center <= 50:
                h_lo = max(0, int(hue_center - hue_range))
                h_hi = min(60, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([5, 40, 40], dtype=np.uint8)
                upper = np.array([35, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([5, 40, 40], dtype=np.uint8)
            upper = np.array([35, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 10.0
            h_lo = max(0, int(med_h - band))
            h_hi = min(60, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "magenta":
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            if hue_center >= 110:
                h_lo = max(100, int(hue_center - hue_range))
                h_hi = min(180, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([140, 40, 40], dtype=np.uint8)
                upper = np.array([180, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([140, 40, 40], dtype=np.uint8)
            upper = np.array([180, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 12.0
            h_lo = max(100, int(med_h - band))
            h_hi = min(180, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "purple":
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            if 105 <= hue_center <= 175:
                h_lo = max(90, int(hue_center - hue_range))
                h_hi = min(180, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([115, 40, 40], dtype=np.uint8)
                upper = np.array([170, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([115, 40, 40], dtype=np.uint8)
            upper = np.array([170, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 12.0
            h_lo = max(90, int(med_h - band))
            h_hi = min(180, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)
    else:
        # "black" or fallback: dark pixels relative to local background
        # Auto-detect if this is a black and white log for aggressive grid removal
        is_bw_log = detect_if_black_and_white_log(roi_bgr)

        residual_mask = None
        gray_processed = gray
        if is_bw_log:
            # Pre-scan grid removal: build the black mask from residual curve
            # pixels after straight grid energy has been modeled away.
            try:
                gray_processed, black_residual_score, black_grid_score, residual_mask = build_black_prescan_grid_removed(gray)
                trace_gray = gray_processed
            except Exception:
                gray_processed = gray
                trace_gray = gray
                black_residual_score = None
                black_grid_score = None
                residual_mask = None
        else:
            try:
                black_residual_score, black_grid_score = compute_black_curve_residual(gray)
            except Exception:
                black_residual_score = None
                black_grid_score = None

        raw_mask = cv2.adaptiveThreshold(
            gray_processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 21, 4
        )
        color_mask = raw_mask

        if black_residual_score is not None and black_residual_score.size == color_mask.size:
            try:
                residual_u8 = np.clip(black_residual_score * 255.0, 0, 255).astype(np.uint8)
                residual_mask_from_score = cv2.adaptiveThreshold(
                    255 - residual_u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV, 21, 6
                )
                residual_mask = residual_mask_from_score if residual_mask is None else cv2.bitwise_or(residual_mask, residual_mask_from_score)
            except Exception:
                pass

        if residual_mask is not None:
            residual_coverage = float(np.mean(residual_mask > 0))
            if is_bw_log and 0.0005 <= residual_coverage <= 0.50:
                # For B/W scans, do not OR the raw grid mask back in. Use the
                # residual as the scan mask, with raw darkness only filling
                # immediately adjacent residual pixels.
                dilated_residual = cv2.dilate(residual_mask, np.ones((3, 3), np.uint8), iterations=1)
                local_raw = cv2.bitwise_and(raw_mask, dilated_residual)
                color_mask = cv2.bitwise_or(residual_mask, local_raw)
            else:
                color_mask = cv2.bitwise_or(color_mask, residual_mask)

        # Suppress colored pixels (grid/track lines are often red/green/blue).
        # In black mode we want low-saturation dark ink.
        try:
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            colored = (sat > 55) & (val > 40)
            color_mask[colored] = 0
        except Exception:
            pass

        # Additional grid removal on the mask itself (less aggressive if already processed)
        if enable_grid_suppression and h >= 20 and w >= 20:
            if is_bw_log:
                # Lighter grid removal since we already did aggressive removal
                k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, min(40, h // 3))))
                # AGGRESSIVE: Remove horizontal lines > 10px to kill grid shelves
                k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
            else:
                # Standard grid removal for non-B&W images
                k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, min(60, h // 2))))
                # AGGRESSIVE: Remove horizontal lines > 15px to kill grid shelves even if not detected as B&W
                k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            
            v_lines = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k_v)
            h_lines = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k_h)
            lines = cv2.bitwise_or(v_lines, h_lines)
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(lines))

    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, 1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, 1)
    color_score = color_mask.astype(np.float32) / 255.0

    # Compute 1-pixel skeleton for black mode using ximgproc thinning.
    # Applied AFTER grid removal so only curve pixels are thinned, not grid lines.
    skel_thin = None
    if mode not in colored_modes:
        try:
            if hasattr(cv2, 'ximgproc'):
                skel_thin = cv2.ximgproc.thinning(
                    color_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
                )
        except Exception:
            skel_thin = None

    # 2) Enhanced edge detection using both Canny and Sobel.
    #    Canny finds strong edges; Sobel emphasizes horizontal gradients
    #    which helps track curves that move left/right.
    edge_gray = trace_gray if mode not in colored_modes else gray
    edges_canny = cv2.Canny(edge_gray, 40, 120)
    
    # Sobel for horizontal gradient (curve moving left/right)
    sobel_x = cv2.Sobel(edge_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    sobel_x = (sobel_x / (sobel_x.max() + 1e-6) * 255).astype(np.uint8)
    
    # Combine Canny and Sobel - Canny for sharp edges, Sobel for gradients
    edges_combined = cv2.addWeighted(edges_canny, 0.6, sobel_x, 0.4, 0)
    edges_blur = cv2.GaussianBlur(edges_combined, (5, 5), 0)
    edge_score = edges_blur.astype(np.float32) / 255.0
    
    # For color modes, gate edges by color mask to suppress non-colored edges
    if enable_curve_masking and mode in colored_modes:
        edge_score *= color_score

    # 3) Suppress vertical "rails" (grid / borders) and track edges
    # REMOVED morphological erasure because it was deleting steep curve segments.
    # We will rely on the column-statistics penalty (rail_penalty) instead.

    if enable_grid_suppression and h >= 4 and w >= 2:
        col_on_frac = (color_score > 0).mean(axis=0)
        
        # For black mode, slow-varying curves (DTC, RHOB) can occupy one column for
        # 40-70% of the image height — use a high threshold to only kill near-solid
        # gridline/border columns (~90%+ occupancy). Colored modes can stay at 0.35
        # since their color_score is already hue-gated and gridlines are unsaturated.
        col_rail_threshold = 0.80 if mode not in colored_modes else 0.35
        rail_cols = col_on_frac > col_rail_threshold
        if np.any(rail_cols):
            color_score[:, rail_cols] *= 0.005  # Almost eliminate vertical rails
            edge_score[:, rail_cols] *= 0.005

        if mode not in colored_modes:
            row_on_frac = (color_score > 0).mean(axis=1)
            rail_rows = row_on_frac > 0.80
            if np.any(rail_rows):
                color_score[rail_rows, :] *= 0.02
                edge_score[rail_rows, :] *= 0.02
        
        # REMOVED edge suppression. Gamma Ray curves often hit the track edges.
        # We should rely on the specific 'preprocess_curve_track' logic for borders,
        # not a blind gradient suppression.

    # 4) Centerline boost via distance transform.
    # For colored modes, compute this from the filled stroke area (color mask),
    # not an edge-gated mask. Edge-only masks pull the DT peak toward one side
    # of a thick stroke.
    if mode in colored_modes:
        bin_for_dt = (color_score > 0.12).astype(np.uint8)
        try:
            # Fill small holes so DT peaks at true stroke center.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bin_for_dt = cv2.morphologyEx(bin_for_dt, cv2.MORPH_CLOSE, k, iterations=1)
        except Exception:
            pass
    else:
        # For black mode, keep the conservative ink definition (helps avoid grid).
        ink_mask = (color_score > 0.25) & (edge_score > 0.10)
        if not np.any(ink_mask):
            ink_mask = (color_score > 0.15) & (edge_score > 0.05)
        bin_for_dt = ink_mask.astype(np.uint8)

    if np.any(bin_for_dt):
        dist = cv2.distanceTransform(bin_for_dt, cv2.DIST_L2, 5)
        center_score = dist.astype(np.float32)
        maxd = float(center_score.max())
        if maxd > 0:
            center_score /= maxd
            # Boost toward the center of thick strokes.
            center_score = np.power(center_score, 0.65)
    else:
        center_score = np.zeros_like(color_score, dtype=np.float32)

    # 5) Enhanced probability map with edge-aware filtering
    # For color modes, use adaptive weighting based on local characteristics
    if mode in colored_modes:
        # Edge-enhanced weighting for GR logs
        # Boost edge detection specifically for jagged features
        edge_enhanced = edge_score.copy()
        
        # Apply directional filtering to enhance horizontal edges (curve movement)
        kernel_horizontal = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
        horizontal_edges = cv2.filter2D(edge_score, -1, kernel_horizontal)
        horizontal_edges = np.maximum(0, horizontal_edges)
        
        # Combine with original edges
        edge_enhanced = np.maximum(edge_score, horizontal_edges * 0.5)
        
        # Adaptive weighting based on local edge density.
        # Keep edges helpful but avoid consistent outer-edge bias on thick ink.
        local_edge_density = cv2.GaussianBlur(edge_enhanced, (5, 5), 0)
        edge_weight = 0.18 + 0.16 * local_edge_density

         # Centerline boost; for thick strokes DT already peaks at the true center.
        center_boost = center_score * (1.0 + 0.2 * edge_enhanced)

        # Final probability map: slightly lower color weight, stronger center.
        prob = 0.08 * color_score + edge_weight * edge_enhanced + (0.92 - edge_weight) * center_boost
        
        # Apply gamma correction to boost faint signals
        prob = np.power(prob, 0.8)  # Gamma < 1 boosts low values
        
    else:
        # Enhanced black mode detection
        edge_enhanced = edge_score.copy()
        
        # Boost weak edges in black mode
        weak_edges = (edge_score > 0.05) & (edge_score < 0.3)
        edge_enhanced[weak_edges] = edge_score[weak_edges] * 1.5
        
        # --- Vertical Derivative Boost ---
        # Calculate Sobel Y to detect horizontal changes (edges of horizontal lines/spikes)
        # Vertical grid rails have dy ~ 0. Wiggly curves (even steep ones) have higher dy components.
        sobel_y = cv2.Sobel(edge_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)
        # Normalize
        max_sy = sobel_y.max()
        if max_sy > 0:
            sobel_y_score = (sobel_y / max_sy).astype(np.float32)
        else:
            sobel_y_score = np.zeros_like(edge_score)

        # --- Harris Corner Boost ---
        # Harris response is high at "corners" (jagged peaks) and low on straight edges (grid lines).
        # This is perfect for highlighting the high-frequency nature of GR curves.
        # blockSize=2 (local), ksize=3 (gradients), k=0.04 (sensitivity)
        harris = cv2.cornerHarris(edge_gray, 2, 3, 0.04)
        # Normalize strictly to 0-1
        harris = np.maximum(0, harris) # Clip negatives (flat regions)
        max_h = harris.max()
        if max_h > 0:
            harris_score = (harris / max_h).astype(np.float32)
            # Dilate slightly to make the corner "dots" connect
            harris_score = cv2.dilate(harris_score, np.ones((3,3), np.uint8))
        else:
            harris_score = np.zeros_like(edge_score)
            
        # --- Diagonal Derivative Boost ---
        # Grid lines are 0 or 90 degrees. Curves have diagonal segments.
        # Calculate diagonal gradients: |dx| + |dy| is a simple approx, but we can be more specific.
        # Actually, just using the magnitude of the gradient vector (already largely covered by Canny)
        # isn't enough. We want specifically 45/135 degree energy.
        # Rotate 45 degrees? Easier: |dx| * |dy| is high only when BOTH are present (diagonal).
        diag_score = (sobel_x.astype(np.float32)/255.0) * (sobel_y.astype(np.float32)/max_sy if max_sy > 0 else 0)
        # Normalize
        if diag_score.max() > 0:
            diag_score /= diag_score.max()
            
        # Combine:
        # - color_score (15%): Base intensity
        # - edge_enhanced (30%): Canny + SobelX (strong edges) - reduced slightly
        # - center_score (20%): Distance transform (center of strokes)
        # - sobel_y_score (15%): Boost for wiggles/spikes (dy)
        # - harris_score (10%): Boost for jagged peaks/corners
        # - diag_score (10%): Boost for diagonal segments (non-grid orientations)
        # Build skeleton score from ximgproc thinning (1-pixel centerline)
        skel_score = None
        if skel_thin is not None and skel_thin.any():
            skel_f = cv2.GaussianBlur(skel_thin.astype(np.float32), (3, 3), 0)
            skel_max = float(skel_f.max())
            if skel_max > 0:
                skel_score = skel_f / skel_max

        if skel_score is not None:
            # Skeleton gets the strongest single vote so thick black strokes
            # stay centered, but we lean more on the residual-dark score than
            # on raw corners to avoid snapping to grid intersections.
            residual_score = black_residual_score if black_residual_score is not None and black_residual_score.size == color_score.size else color_score
            prob = (
                0.18 * color_score +
                0.38 * residual_score +
                0.08 * edge_enhanced +
                0.14 * center_score +
                0.06 * sobel_y_score +
                0.02 * harris_score +
                0.02 * diag_score +
                0.12 * skel_score
            )
        else:
            residual_score = black_residual_score if black_residual_score is not None and black_residual_score.size == color_score.size else color_score
            prob = (
                0.22 * color_score +
                0.42 * residual_score +
                0.10 * edge_enhanced +
                0.14 * center_score +
                0.06 * sobel_y_score +
                0.03 * harris_score +
                0.03 * diag_score
            )

        if black_grid_score is not None and black_grid_score.size == prob.size:
            try:
                # Penalize long straight-line evidence, but keep a floor so the
                # curve can survive where it legitimately crosses a grid line.
                prob *= np.clip(1.0 - 0.92 * black_grid_score, 0.03, 1.0)
                if black_residual_score is not None and black_residual_score.size == prob.size:
                    prob = np.maximum(prob, 0.55 * black_residual_score)
            except Exception:
                pass

    # 6) Reuse the stronger grid-removal heuristics from preprocess_curve_track
    #    as a gating mask. This aggressively down-weights columns/rows that
    #    look like grid or track borders, while preserving the wiggly curve
    #    strokes.
    if mode not in colored_modes:
        try:
            cleaned_binary = preprocess_curve_track(roi_bgr, mode=mode)
            if cleaned_binary is not None and cleaned_binary.size == prob.size:
                cleaned_score = cleaned_binary.astype(np.float32) / 255.0
                if black_residual_score is not None and black_residual_score.size == prob.size:
                    cleaned_score = np.maximum(cleaned_score, (black_residual_score > 0.10).astype(np.float32))
                # Where cleaned_score == 0 (likely grid/border), push probability
                # almost to zero; where == 1, keep prob as-is.
                gate = 0.05 + 0.95 * cleaned_score
                prob *= gate
        except Exception:
            # If preprocessing fails for any reason, fall back to the ungated map.
            pass
    maxp = float(prob.max())
    if maxp > 0:
        prob = prob / maxp
    prob = np.clip(prob, 1e-4, 1.0).astype(np.float32)

    return (prob * 255.0).astype(np.uint8)


def trace_curve_with_dp(
    curve_mask,
    scale_min,
    scale_max,
    curve_type="GR",
    max_step=3,
    smooth_lambda=0.5,
    curv_lambda=0.0,
    hot_side=None,
):
    """Trace a curve using dynamic programming for smooth path finding.
    
    Args:
        curve_mask: Binary mask (0 or 255) where curve pixels are bright
        scale_min: Left scale value
        scale_max: Right scale value
        curve_type: Curve type for plausibility checks (GR, RHOB, NPHI, DT, etc.)
        max_step: Max horizontal movement per row (pixels)
        smooth_lambda: First-derivative smoothness penalty weight (penalizes jumps)
        curv_lambda: Second-derivative curvature penalty weight (penalizes kinks)
    
    Returns:
        xs: Array of x-coordinates (one per row), with np.nan for low-confidence rows
        confidence: Array of confidence scores (0-1) per row
    """
    if curve_mask is None or curve_mask.size == 0:
        return np.array([]), np.array([])
    
    h, w = curve_mask.shape
    if h < 2 or w < 2:
        return np.full(h, np.nan), np.zeros(h)

    if not getattr(fast_tracer, "NUMBA_AVAILABLE", False) and w > 420:
        w_small = int(max(64, min(420, w)))
        if w_small < w:
            mask_small = cv2.resize(curve_mask, (w_small, h), interpolation=cv2.INTER_AREA)
            max_step_small = max(1, int(round(max_step * (w_small - 1) / max(1, w - 1))))
            xs_small, conf_small = trace_curve_with_dp(
                mask_small,
                scale_min=scale_min,
                scale_max=scale_max,
                curve_type=curve_type,
                max_step=max_step_small,
                smooth_lambda=smooth_lambda,
                curv_lambda=curv_lambda,
                hot_side=hot_side,
            )
            if xs_small is None or xs_small.size == 0:
                return xs_small, conf_small
            scale_back = (w - 1) / max(1, (w_small - 1))
            xs = xs_small.astype(np.float32, copy=True)
            finite = np.isfinite(xs)
            xs[finite] = xs[finite] * scale_back
            return xs, conf_small

    # Define plausible value ranges per curve type
    plausible_ranges = {
        'GR': (0, 200),
        'RHOB': (1.5, 3.5),
        'NPHI': (-0.2, 0.6),
        'DT': (40, 200),
        'CALI': (4, 20),
        'SP': (-200, 100),
    }
    
    # Convert mask to probability (0-1)
    prob = curve_mask.astype(np.float32) / 255.0

    def _morphological_skeleton(bin_img):
        """Simple morphological skeletonization (Zhang-Suen style via erode-open)."""
        size = np.size(bin_img)
        skel = np.zeros_like(bin_img, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        prev = None
        iteration = 0
        max_iter = 512  # safety
        while True:
            eroded = cv2.erode(bin_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(bin_img, temp)
            skel = cv2.bitwise_or(skel, temp)
            bin_img = eroded.copy()
            iteration += 1
            if cv2.countNonZero(bin_img) == 0 or iteration >= max_iter:
                break
            if prev is not None and np.array_equal(bin_img, prev):
                break
            prev = bin_img
        return skel

    # Live-wire style node score combining probability and centerline distance
    bin_mask = prob > 0.10
    skeleton_score = np.zeros_like(prob, dtype=np.float32)
    if np.any(bin_mask):
        try:
            if hasattr(cv2, 'ximgproc'):
                skel = cv2.ximgproc.thinning(
                    bin_mask.astype(np.uint8) * 255,
                    thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
                )
            else:
                skel = _morphological_skeleton((bin_mask.astype(np.uint8) * 255))
            if skel is not None and skel.size == prob.size:
                skel_f = skel.astype(np.float32) / 255.0
                # Feather skeleton to nearby pixels so DP can stay on the ridge
                skel_f = cv2.GaussianBlur(skel_f, (3, 3), 0)
                skel_max = float(skel_f.max())
                if skel_max > 0:
                    skeleton_score = skel_f / skel_max
        except Exception:
            skeleton_score = np.zeros_like(prob, dtype=np.float32)
    if np.any(bin_mask):
        dist = cv2.distanceTransform(bin_mask.astype(np.uint8), cv2.DIST_L2, 5)
        dist_norm = dist.astype(np.float32)
        maxd = float(dist_norm.max())
        if maxd > 0:
            dist_norm /= maxd
        center_score = np.power(dist_norm, 0.9)
    else:
        center_score = np.zeros_like(prob, dtype=np.float32)

    eps = 1e-6
    live_score = np.power(prob, 0.7) * (0.15 + 0.85 * center_score)
    # Boost with skeleton ridge to keep micro-bumps
    if skeleton_score is not None:
        live_score = np.maximum(live_score, 0.55 * skeleton_score + 0.05 * bin_mask.astype(np.float32))
    live_score = np.clip(live_score, eps, 1.0)

    # Blend in distance transform so the DP prefers the *center* of thick
    # ink strokes over their edges.  A pixel at the stroke center gets a
    # bonus proportional to its distance from the nearest background pixel.
    # 70% original score + 30% centredness keeps the colour/intensity
    # signal dominant while removing the edge-of-stroke bias.
    if bin_mask.any():
        _dist = cv2.distanceTransform(bin_mask.astype(np.uint8), cv2.DIST_L2, 3)
        _d_max = _dist.max()
        if _d_max > 0:
            _dist_norm = (_dist / _d_max).astype(np.float32)
            live_score = live_score * (0.7 + 0.3 * _dist_norm)
            live_score = np.clip(live_score, eps, 1.0)

    cost = -np.log(live_score)

    curve_type_upper = str(curve_type or "").upper()

    # Soft rail penalty: down-weight columns that stay on for many rows, without banning them
    if h >= 4 and w >= 2:
        col_frac = bin_mask.mean(axis=0)
        # Lower threshold so we catch dashed or interrupted vertical grid lines.
        # It's a soft penalty, so a truly straight curve can still power through it.
        rail_thresh = 0.40 if curve_type_upper == 'GR' else 0.50
        rail_mask = col_frac > rail_thresh
        # Expand to runs of length >=2 using a 2-wide moving window (grid lines are usually 2+ px wide)
        rail_run = np.convolve(rail_mask.astype(np.float32), np.ones(2, dtype=np.float32), mode='same') >= 1.5
        if np.any(rail_run):
            # Increase the rail penalty so the trace actively avoids vertical grids
            rail_weight = 12.0 if curve_type_upper == 'GR' else 8.0
            cost += (rail_weight * rail_run.astype(np.float32))[np.newaxis, :]

    # Use live_score for Viterbi likelihoods
    prob = live_score
    
    if hot_side in ("left", "right") and w >= 2:
        frac = np.linspace(0.0, 1.0, w, dtype=np.float32)
        if hot_side == "left":
            dist = frac
        else:
            dist = 1.0 - frac
        # A strong hot-side prior is useful for GR-style crest picking, but on
        # slower black curves such as RHOB/DT it can pull the path onto a
        # neighboring rail when the signal gets weak. Keep the directional bias
        # for GR and effectively disable it for other curve families.
        side_lambda = 1.0 if curve_type_upper == "GR" else 0.0
        if side_lambda > 0.0:
            side_penalty = side_lambda * dist
            cost += side_penalty[np.newaxis, :]
    
    # Add plausibility penalty only when the display scale is in physical units.
    # If scale_min/scale_max don't overlap the known physical range at all (e.g.
    # display range 0-150 vs RHOB physical 1.5-3.5 g/cc), skip the penalty to
    # avoid penalising every column and forcing the DP toward one edge.
    if curve_type.upper() in plausible_ranges:
        pmin, pmax = plausible_ranges[curve_type.upper()]
        scale_lo = min(scale_min, scale_max)
        scale_hi = max(scale_min, scale_max)
        # Only apply if the display range meaningfully overlaps the physical range
        if scale_lo <= pmax and scale_hi >= pmin:
            for x in range(w):
                value = scale_min + (x / max(1, w - 1)) * (scale_max - scale_min)
                if value < pmin or value > pmax:
                    cost[:, x] += 1.0
    
    # Horizontal grid line suppression using morphological opening.
    # A true grid line has many *consecutive* pixels across the row; the actual
    # curve only spans ~1-5 pixels per row and won't survive a wide horizontal
    # opening. This is more discriminative than a raw row-fraction threshold.
    if h >= 4 and w >= 8:
        # Use a slightly smaller kernel for horizontal lines so we catch broken grid rails too
        horiz_kernel_w = max(3, w // 5)
        horiz_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_w, 1))
        horiz_detected = cv2.morphologyEx(bin_mask.astype(np.uint8), cv2.MORPH_OPEN, horiz_kern)
        horiz_row_frac = horiz_detected.mean(axis=1)
        # Lower threshold so more horizontal rails get penalized
        horiz_mask = horiz_row_frac > 0.20  # >20% of row survived wide-kernel opening → true grid line
        if np.any(horiz_mask):
            uniform_cost = float(-np.log(1e-5))  # Much stronger penalty
            cost[horiz_mask, :] = uniform_cost
            prob[horiz_mask, :] = 1e-5

    # Run optimized DP (Forward Pass)
    xs_fwd, conf_fwd = fast_tracer.run_viterbi(
        cost.astype(np.float32), 
        prob.astype(np.float32), 
        int(max_step), 
        float(smooth_lambda), 
        float(curv_lambda)
    )
    
    # Run optimized DP (Backward Pass)
    # Flip cost and prob arrays upside down
    xs_bwd_flipped, conf_bwd_flipped = fast_tracer.run_viterbi(
        cost[::-1].astype(np.float32), 
        prob[::-1].astype(np.float32), 
        int(max_step), 
        float(smooth_lambda), 
        float(curv_lambda)
    )
    # Flip results back to match original orientation
    xs_bwd = xs_bwd_flipped[::-1]
    conf_bwd = conf_bwd_flipped[::-1]

    # Each directional pass already follows a smooth DP path, but the fast
    # tracer marks low-probability rows as NaN. On black scans that creates
    # tiny one-row dropouts right where the curve crosses grid lines, and the
    # forward/backward merge then "zipper switches" onto the other branch.
    # Interpolating the per-pass dropouts preserves each branch's continuity
    # before we decide which direction to trust per span.
    try:
        if xs_fwd.size:
            xs_fwd = pd.Series(xs_fwd).interpolate(
                method='linear',
                limit_direction='both',
            ).to_numpy(dtype=np.float32)
        if xs_bwd.size:
            xs_bwd = pd.Series(xs_bwd).interpolate(
                method='linear',
                limit_direction='both',
            ).to_numpy(dtype=np.float32)
    except Exception:
        pass
    
    # Merge Forward and Backward results with a continuity-aware branch choice.
    # The forward/backward passes can occasionally lock onto different rails on
    # black scans. Picking the locally brighter branch *per row* causes visual
    # teleportation, so resolve disagreements as contiguous spans instead.
    xs = np.full_like(xs_fwd, np.nan)
    confidence = np.zeros_like(conf_fwd)

    merge_tol = 5.0
    switch_penalty = 0.25
    merge_smooth_lambda = max(0.02, float(smooth_lambda) * 0.05)

    candidates = np.stack(
        (xs_fwd.astype(np.float32), xs_bwd.astype(np.float32)),
        axis=1,
    )
    candidate_conf = np.stack(
        (conf_fwd.astype(np.float32), conf_bwd.astype(np.float32)),
        axis=1,
    )
    candidate_valid = np.isfinite(candidates)

    def _candidate_cost(y_idx, x_val, conf_val):
        x_idx = int(min(w - 1, max(0, int(round(float(x_val))))))
        p_here = float(prob[y_idx, x_idx])
        # Favor high-probability pixels first, with a mild confidence tie-breaker.
        return float(-np.log(max(eps, p_here)) - 0.15 * max(0.0, float(conf_val)))

    y = 0
    while y < h:
        if not np.any(candidate_valid[y]):
            y += 1
            continue

        seg_start = y
        while y < h and np.any(candidate_valid[y]):
            y += 1
        seg_end = y
        seg_len = seg_end - seg_start

        merge_dp = np.full((seg_len, 2), np.inf, dtype=np.float64)
        merge_prev = np.full((seg_len, 2), -1, dtype=np.int8)

        for state in range(2):
            if candidate_valid[seg_start, state]:
                merge_dp[0, state] = _candidate_cost(
                    seg_start,
                    candidates[seg_start, state],
                    candidate_conf[seg_start, state],
                )

        for off in range(1, seg_len):
            yy = seg_start + off
            for state in range(2):
                if not candidate_valid[yy, state]:
                    continue

                x_cur = float(candidates[yy, state])
                node_cost = _candidate_cost(yy, x_cur, candidate_conf[yy, state])
                best_val = np.inf
                best_prev_state = -1

                for prev_state in range(2):
                    if not candidate_valid[yy - 1, prev_state]:
                        continue
                    prev_val = float(merge_dp[off - 1, prev_state])
                    if not np.isfinite(prev_val):
                        continue

                    x_prev = float(candidates[yy - 1, prev_state])
                    dx = x_cur - x_prev
                    transition_cost = merge_smooth_lambda * (dx * dx)
                    if prev_state != state:
                        transition_cost += switch_penalty

                    total_cost = prev_val + node_cost + transition_cost
                    if total_cost < best_val:
                        best_val = total_cost
                        best_prev_state = prev_state

                if best_prev_state >= 0:
                    merge_dp[off, state] = best_val
                    merge_prev[off, state] = best_prev_state
                else:
                    merge_dp[off, state] = node_cost

        states = np.full(seg_len, -1, dtype=np.int8)
        end_state = 0
        if not np.isfinite(merge_dp[-1, end_state]) and np.isfinite(merge_dp[-1, 1]):
            end_state = 1
        elif np.isfinite(merge_dp[-1, 1]) and merge_dp[-1, 1] < merge_dp[-1, 0]:
            end_state = 1
        states[-1] = np.int8(end_state)

        for off in range(seg_len - 1, 0, -1):
            state = int(states[off])
            prev_state = int(merge_prev[off, state]) if state >= 0 else -1
            if prev_state < 0:
                if np.isfinite(merge_dp[off - 1, state]):
                    prev_state = state
                elif np.isfinite(merge_dp[off - 1, 1 - state]):
                    prev_state = 1 - state
            states[off - 1] = np.int8(prev_state)

        if states[0] < 0:
            if np.isfinite(merge_dp[0, 0]):
                states[0] = np.int8(0)
            elif np.isfinite(merge_dp[0, 1]):
                states[0] = np.int8(1)

        for off in range(seg_len):
            yy = seg_start + off
            v1 = candidates[yy, 0]
            v2 = candidates[yy, 1]
            valid1 = bool(candidate_valid[yy, 0])
            valid2 = bool(candidate_valid[yy, 1])

            if valid1 and valid2 and abs(float(v1) - float(v2)) <= merge_tol:
                xs[yy] = float(v1 + v2) * 0.5
                confidence[yy] = float(candidate_conf[yy, 0] + candidate_conf[yy, 1]) * 0.5
                continue

            state = int(states[off])
            if state < 0 or not candidate_valid[yy, state]:
                if valid1 and valid2:
                    c1 = _candidate_cost(yy, v1, candidate_conf[yy, 0])
                    c2 = _candidate_cost(yy, v2, candidate_conf[yy, 1])
                    state = 0 if c1 <= c2 else 1
                elif valid1:
                    state = 0
                elif valid2:
                    state = 1
                else:
                    continue

            xs[yy] = float(candidates[yy, state])
            confidence[yy] = float(candidate_conf[yy, state])
    
    return xs, confidence


def trace_curve_skeleton_path(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Photocopy-style tracer: binarize, skeletonize, take top-to-bottom seam along skeleton.
    """
    if mask is None or mask.size == 0:
        return np.array([]), np.array([])
    h, w = mask.shape
    bin_mask = (mask > 25).astype(np.uint8)
    if cv2.countNonZero(bin_mask) == 0:
        return np.full(h, np.nan, dtype=np.float32), np.zeros(h, dtype=np.float32)

    # Light rail removal
    dark = (mask < 40).astype(np.uint8)
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 24)))
    k_horz = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 24), 1))
    rail_v = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_vert)
    rail_h = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_horz)
    rail_mask = cv2.bitwise_or(rail_v, rail_h)
    bin_mask = cv2.subtract(bin_mask, rail_mask)
    bin_mask = np.clip(bin_mask, 0, 1).astype(np.uint8)
    if cv2.countNonZero(bin_mask) == 0:
        return np.full(h, np.nan, dtype=np.float32), np.zeros(h, dtype=np.float32)

    skel = _skeletonize_binary((bin_mask * 255).astype(np.uint8))
    if skel is None or skel.size != mask.size or cv2.countNonZero(skel) == 0:
        return np.full(h, np.nan, dtype=np.float32), np.zeros(h, dtype=np.float32)

    num_labels, labels = cv2.connectedComponents(skel, connectivity=8)
    if num_labels > 1:
        areas = [(labels == i).sum() for i in range(1, num_labels)]
        keep_label = 1 + int(np.argmax(areas))
        skel = np.where(labels == keep_label, skel, 0).astype(np.uint8)
        if cv2.countNonZero(skel) == 0:
            return np.full(h, np.nan, dtype=np.float32), np.zeros(h, dtype=np.float32)

    prob = skel.astype(np.float32) / 255.0
    cost = 1.0 - prob
    dp = cost.copy()
    prev = np.full_like(labels, -1, dtype=np.int16)
    for y in range(1, h):
        for x in range(w):
            best = dp[y - 1, x]
            px = x
            if x > 0 and dp[y - 1, x - 1] < best:
                best = dp[y - 1, x - 1]; px = x - 1
            if x + 1 < w and dp[y - 1, x + 1] < best:
                best = dp[y - 1, x + 1]; px = x + 1
            dp[y, x] += best
            prev[y, x] = px
    end_x = int(np.argmin(dp[-1]))
    xs_path = np.full(h, np.nan, dtype=np.float32)
    x = end_x
    for y in range(h - 1, -1, -1):
        xs_path[y] = float(x)
        x_prev = prev[y, x]
        if x_prev < 0:
            break
        x = int(x_prev)
    conf = np.where(np.isfinite(xs_path), 1.0, 0.0).astype(np.float32)
    return xs_path, conf


def _skeletonize_binary(bin_img: np.ndarray) -> np.ndarray:
    """Simple morphological skeletonization (Zhang-Suen style via erode-open)."""
    skel = np.zeros_like(bin_img, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    prev = None
    iteration = 0
    max_iter = 1024  # safety
    work = bin_img.copy()
    while True:
        eroded = cv2.erode(work, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(work, temp)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded.copy()
        iteration += 1
        if cv2.countNonZero(work) == 0 or iteration >= max_iter:
            break
        if prev is not None and np.array_equal(work, prev):
            break
        prev = work
    return skel


def _parabola_subpixel_x(row_slice: np.ndarray, start: int):
    """Fit a parabola to the log-intensity of a row slice and return subpixel x."""
    if row_slice is None or row_slice.size < 3:
        return float(start + np.argmax(row_slice))
    y = row_slice.astype(np.float32)
    idx = int(np.argmax(y))
    if idx == 0 or idx == y.size - 1:
        return float(start + idx)
    y1, y2, y3 = y[idx - 1:idx + 2]
    denom = (y1 - 2 * y2 + y3)
    if abs(denom) < 1e-9:
        return float(start + idx)
    offset = 0.5 * (y1 - y3) / denom
    return float(start + idx + offset)


def _postprocess_missed_peaks(mask: np.ndarray, prob: np.ndarray, xs: np.ndarray, search_radius: int = 12, min_prob: float = 0.03):
    """Hybrid post-processor: force missed ink peaks onto the curve."""
    h, w = mask.shape
    if xs.size != h:
        return xs
    for y in range(h):
        if not np.isfinite(xs[y]):
            continue
        xi = int(round(xs[y]))
        start = max(0, xi - search_radius)
        end = min(w, xi + search_radius + 1)
        row = prob[y, start:end]
        if row.size == 0:
            continue
        # Find all peaks above min_prob (robust to plateaus)
        peaks = []
        plateau_start = -1
        for i in range(1, row.size - 1):
            val = row[i]
            if val < min_prob:
                plateau_start = -1
                continue
            
            prev = row[i - 1]
            next_val = row[i + 1]
            
            # Rising edge
            if val > prev:
                if val > next_val:
                    peaks.append(i) # Sharp peak
                    plateau_start = -1
                elif val == next_val:
                    plateau_start = i # Start plateau
                else:
                    plateau_start = -1
            # Flat
            elif val == prev:
                if val > next_val:
                    if plateau_start != -1:
                        peaks.append((plateau_start + i) // 2) # End plateau
                    plateau_start = -1
                elif val < next_val:
                    plateau_start = -1
            # Falling
            else:
                plateau_start = -1

        if not peaks:
            continue
        # If the current position is not already on a peak, move to the nearest peak
        cur_rel = xi - start
        on_peak = any(abs(cur_rel - p) <= 2 for p in peaks)
        if not on_peak:
            # Choose the peak closest to the current x
            best = min(peaks, key=lambda p: abs(p - cur_rel))
            # Weighted centroid around the chosen peak
            peak_val = row[best]
            thresh = max(peak_val * 0.6, min_prob)
            left = best
            right = best
            while left > 0 and row[left - 1] >= thresh:
                left -= 1
            while right + 1 < row.size and row[right + 1] >= thresh:
                right += 1
            seg = row[left:right + 1].astype(np.float32)
            coords = np.arange(start + left, start + right + 1, dtype=np.float32)
            wsum = seg.sum()
            if wsum > 1e-6:
                xs[y] = float((coords * seg).sum() / wsum)
            else:
                xs[y] = float(start + best)


def align_rgb_channels(bgr: np.ndarray) -> np.ndarray:
    """Align RGB channels via phase correlation to reduce color fringing."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    try:
        b, g, r = cv2.split(bgr)

        def _shift_to_ref(src, ref):
            src_f = src.astype(np.float32)
            ref_f = ref.astype(np.float32)
            shift, _ = cv2.phaseCorrelate(ref_f, src_f)
            dx, dy = shift
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        r_aligned = _shift_to_ref(r, g)
        b_aligned = _shift_to_ref(b, g)
        return cv2.merge([b_aligned, g, r_aligned])
    except Exception:
        return bgr


def align_rgb_channels(bgr: np.ndarray) -> np.ndarray:
    """Align RGB channels via phase correlation to reduce color fringing."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    try:
        b, g, r = cv2.split(bgr)

        def _shift_to_ref(src, ref):
            src_f = src.astype(np.float32)
            ref_f = ref.astype(np.float32)
            shift, _ = cv2.phaseCorrelate(ref_f, src_f)
            dx, dy = shift
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        r_aligned = _shift_to_ref(r, g)
        b_aligned = _shift_to_ref(b, g)
        return cv2.merge([b_aligned, g, r_aligned])
    except Exception:
        return bgr
    for y in range(h):
        if not np.isfinite(xs[y]):
            continue
        xi = int(round(xs[y]))
        start = max(0, xi - search_radius)
        end = min(w, xi + search_radius + 1)
        row = prob[y, start:end]
        if row.size == 0:
            continue
        # Find all peaks above min_prob
        peaks = []
        for i in range(1, row.size - 1):
            if row[i] > row[i - 1] and row[i] > row[i + 1] and row[i] >= min_prob:
                peaks.append(i)
        if not peaks:
            continue
        # If the current position is not already on a peak, move to the nearest peak
        cur_rel = xi - start
        on_peak = any(abs(cur_rel - p) <= 2 for p in peaks)
        if not on_peak:
            # Choose the peak closest to the current x
            best = min(peaks, key=lambda p: abs(p - cur_rel))
            # Weighted centroid around the chosen peak
            peak_val = row[best]
            thresh = max(peak_val * 0.6, min_prob)
            left = best
            right = best
            while left > 0 and row[left - 1] >= thresh:
                left -= 1
            while right + 1 < row.size and row[right + 1] >= thresh:
                right += 1
            seg = row[left:right + 1].astype(np.float32)
            coords = np.arange(start + left, start + right + 1, dtype=np.float32)
            wsum = seg.sum()
            if wsum > 1e-6:
                xs[y] = float((coords * seg).sum() / wsum)
            else:
                xs[y] = float(start + best)
        else:
            # Even if on a peak, refine to weighted centroid for subpixel accuracy
            cur_idx = int(cur_rel)
            if 0 < cur_idx < row.size - 1:
                peak_val = row[cur_idx]
                thresh = max(peak_val * 0.6, min_prob)
                left = cur_idx
                right = cur_idx
                while left > 0 and row[left - 1] >= thresh:
                    left -= 1
                while right + 1 < row.size and row[right + 1] >= thresh:
                    right += 1
                seg = row[left:right + 1].astype(np.float32)
                coords = np.arange(start + left, start + right + 1, dtype=np.float32)
                wsum = seg.sum()
                if wsum > 1e-6:
                    xs[y] = float((coords * seg).sum() / wsum)
    return xs


def trace_curve_pixel_perfect(mask: np.ndarray, grayscale: np.ndarray = None, bgr: np.ndarray = None, hot_side=None, preserve_wiggles: bool = False, crest_boost: bool = False):
    """Pixel-perfect tracing optimized for dot-matrix style prints: row-by-row peak following."""
    if mask is None or mask.size == 0:
        return np.array([]), np.array([])
    h, w = mask.shape
    if h < 4 or w < 2:
        return np.full(h, np.nan, dtype=np.float32), np.zeros(h, dtype=np.float32)

    # Build probability map with hue weighting if available
    hue_weight = None
    if bgr is not None and bgr.size == mask.size * 3:
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            ink_mask = (mask > np.percentile(mask, 70)).astype(np.uint8)
            if ink_mask.sum() > 20:
                ink_hues = hsv[..., 0][ink_mask.astype(bool)]
                ang = ink_hues.astype(np.float32) * (2.0 * np.pi / 180.0)
                mean_hue = math.atan2(np.sin(ang).mean(), np.cos(ang).mean())
                if mean_hue < 0:
                    mean_hue += 2 * np.pi
                mean_deg = mean_hue * 180.0 / np.pi
                hue = hsv[..., 0].astype(np.float32)
                dh = np.abs(((hue - mean_deg + 90) % 180) - 90)
                hue_weight = np.exp(-(dh ** 2) / (2 * (12.0 ** 2))).astype(np.float32)
                hue_weight = np.clip(hue_weight, 0.15, 1.0)
        except Exception:
            hue_weight = None

    prob_base = mask.astype(np.float32) / 255.0
    if hue_weight is not None:
        prob_base = prob_base * hue_weight
    # Morphological close to connect dot-matrix dots (slightly larger to bridge gaps)
    # Taller kernel for crest_boost to bridge vertical dot gaps
    k_size = (3, 11) if crest_boost else (4, 6)
    prob_closed = cv2.morphologyEx(prob_base, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size))
    prob = np.maximum(prob_base, prob_closed)
    
    # Only blur if NOT crest_boost to preserve sharp dot edges
    if not crest_boost:
        prob = cv2.GaussianBlur(prob, (3, 3), 0)
    
    prob = np.clip(prob * 1.25, 0.0, 1.0)

    # Per-row global maxima (photocopy-style fallback)
    row_max_xs = np.full(h, np.nan, dtype=np.float32)
    for y in range(h):
        row = prob[y]
        peak = row.max()
        if peak < 0.01:
            continue
        idx = int(np.argmax(row))
        thresh = max(peak * 0.6, 0.01)
        left = idx
        right = idx
        while left > 0 and row[left - 1] >= thresh:
            left -= 1
        while right + 1 < row.size and row[right + 1] >= thresh:
            right += 1
        seg = row[left:right + 1].astype(np.float32)
        coords = np.arange(left, right + 1, dtype=np.float32)
        wsum = seg.sum()
        if wsum > 1e-6:
            row_max_xs[y] = float((coords * seg).sum() / wsum)
        else:
            row_max_xs[y] = float(idx)

    def _seam_path(prob_map: np.ndarray):
        """Min-cost vertical seam on (1-prob)."""
        h_s, w_s = prob_map.shape
        if h_s < 2 or w_s < 1:
            return np.full(h_s, np.nan, dtype=np.float32)
        cost = 1.0 - prob_map
        dp = cost.copy()
        prev = np.full((h_s, w_s), -1, dtype=np.int16)
        for y in range(1, h_s):
            for x in range(w_s):
                best = dp[y - 1, x]
                px = x
                if x > 0 and dp[y - 1, x - 1] < best:
                    best = dp[y - 1, x - 1]; px = x - 1
                if x + 1 < w_s and dp[y - 1, x + 1] < best:
                    best = dp[y - 1, x + 1]; px = x + 1
                dp[y, x] += best
                prev[y, x] = px
        end_x = int(np.argmin(dp[-1]))
        xs_path = np.full(h_s, np.nan, dtype=np.float32)
        x = end_x
        for y in range(h_s - 1, -1, -1):
            xs_path[y] = float(x)
            x_prev = prev[y, x]
            if x_prev < 0:
                break
            x = int(x_prev)
        return xs_path

    # ---- Simple row-by-row peak following (dot-matrix style) ----
    # Optional crest boost: use stronger ridge-enhanced prob to stay on tops
    ridge_prob = prob
    if crest_boost:
        # Taller vertical blur to bridge larger dot gaps
        # Also slightly wider to help horizontal connectivity
        k_size = (5, 11)
        ridge_prob = np.maximum(prob, cv2.blur(prob, (1, 15)))
        sobel_y = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)
        if sobel_y.max() > 1e-6:
            sobel_y = sobel_y / (sobel_y.max() + 1e-6)
            ridge_prob = np.maximum(ridge_prob, ridge_prob * (1.0 + 0.6 * sobel_y))
        ridge_prob = cv2.dilate(ridge_prob, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    xs = np.full(h, np.nan, dtype=np.float32)
    confidence = np.zeros(h, dtype=np.float32)
    base_search_rad = 80  # widened to catch larger jumps

    # Find initial x from the row with strongest total ink
    row_sums = ridge_prob.sum(axis=1)
    start_row = int(np.argmax(row_sums))
    best_x = int(np.argmax(ridge_prob[start_row]))
    xs[start_row] = float(best_x)
    confidence[start_row] = ridge_prob[start_row, best_x]

    def find_peak_in_row(y, prev_x, wide_search=False):
        """Find strongest peak near prev_x in row y with adaptive continuity penalty."""
        xi = int(round(prev_x))
        row_strength = float(ridge_prob[y].max()) if crest_boost else 0.0
        
        # Determine search window
        s_rad = base_search_rad + int(6 * row_strength) if crest_boost else base_search_rad
        if wide_search:
            s_rad = s_rad * 3  # Emergency wide search
            
        start = max(0, xi - s_rad)
        end = min(w, xi + s_rad + 1)
        row = ridge_prob[y, start:end]
        
        # Lower threshold to catch faint dots
        min_peak_thresh = 0.005 if crest_boost else 0.012
        if row.size == 0 or row.max() < min_peak_thresh:
            return np.nan, 0.0
        
        # Find all significant peaks (robust to plateaus)
        peaks = []
        plateau_start = -1
        for i in range(1, row.size - 1):
            val = row[i]
            if val < min_peak_thresh:
                plateau_start = -1
                continue
                
            prev = row[i - 1]
            next_val = row[i + 1]
            
            # Rising edge
            if val > prev:
                if val > next_val:
                    peaks.append(i) # Sharp peak
                    plateau_start = -1
                elif val == next_val:
                    plateau_start = i # Start plateau
                else:
                    plateau_start = -1
            # Flat
            elif val == prev:
                if val > next_val:
                    if plateau_start != -1:
                        peaks.append((plateau_start + i) // 2) # End plateau
                    plateau_start = -1
                elif val < next_val:
                    plateau_start = -1
            # Falling
            else:
                plateau_start = -1

        if crest_boost and peaks:
            # Use scoring with quadratic penalty to prefer closer peaks
            # but allow jumping to strong peaks if they are reasonably close.
            best_score = -1e9
            idx_rel = peaks[0]
            for pk in peaks:
                pk_val = row[pk]
                pk_x = start + pk
                dist = abs(pk_x - prev_x)
                
                # Quadratic penalty: penalize large jumps much more than small ones
                # at dist=0, penalty=0
                # at dist=10, penalty is small
                # at dist=80, penalty is large
                norm_dist = dist / float(s_rad) # 0 to 1
                
                # Base penalty factor
                penalty_weight = 0.15 # Stronger weight to prevent jumping to grid lines
                
                continuity_penalty = penalty_weight * (norm_dist ** 2) * (1.0 - pk_val * 0.5)
                
                score = pk_val - continuity_penalty
                if score > best_score:
                    best_score = score
                    idx_rel = pk
            peak_val = row[idx_rel]
        elif crest_boost and row.max() >= 0.008:
             # No peaks found but signal is strong (likely edge peak), use argmax
             idx_rel = int(np.argmax(row))
             peak_val = row[idx_rel]
        elif not peaks:
            idx_rel = int(np.argmax(row))
            peak_val = row[idx_rel]
        else:
            # Choose peak with best score: peak_val - adaptive continuity_penalty
            best_score = -1e9
            idx_rel = peaks[0]
            for pk in peaks:
                pk_val = row[pk]
                pk_x = start + pk
                # Adaptive penalty: lighter to allow close switch-backs; allow bigger excursions on strong peaks
                dist = abs(pk_x - prev_x)
                if crest_boost:
                     continuity_penalty = 0.005 * (dist / float(s_rad)) * (1.0 - pk_val)
                else:
                     continuity_penalty = 0.020 * (dist / float(s_rad)) * (1.0 - pk_val)
                score = pk_val - continuity_penalty
                if score > best_score:
                    best_score = score
                    idx_rel = pk
            peak_val = row[idx_rel]
        
        # Weighted centroid for subpixel
        # Use cleaner 'prob' map for localization if available to avoid blur bias
        if crest_boost:
            raw_row = prob[y, start:end]
            if raw_row.size == row.size:
                # Re-check value on raw map
                pv_raw = raw_row[idx_rel]
                # If raw map has signal, use it. Otherwise fall back to ridge_row
                if pv_raw > 0.001:
                    row_for_centroid = raw_row
                    peak_val = pv_raw
                    # Tighter threshold for crest_boost to stay on peak
                    thresh = max(peak_val * 0.70, 0.0025)
                else:
                    row_for_centroid = row
                    thresh = max(peak_val * 0.70, 0.0025)
            else:
                row_for_centroid = row
                thresh = max(peak_val * 0.70, 0.0025)
        else:
            row_for_centroid = row
            thresh = max(peak_val * 0.40, 0.003)

        left = idx_rel
        right = idx_rel
        while left > 0 and row_for_centroid[left - 1] >= thresh:
            left -= 1
        while right + 1 < row_for_centroid.size and row_for_centroid[right + 1] >= thresh:
            right += 1
        seg = row_for_centroid[left:right + 1].astype(np.float32)
        coords = np.arange(start + left, start + right + 1, dtype=np.float32)
        wsum = seg.sum()
        if wsum > 1e-6:
            x_out = float((coords * seg).sum() / wsum)
        else:
            x_out = float(start + idx_rel)
        return x_out, peak_val

    # Trace downward from start_row
    prev_x = xs[start_row]
    for y in range(start_row + 1, h):
        x_new, conf = find_peak_in_row(y, prev_x)
        if not np.isfinite(x_new) and crest_boost:
            # Fallback: try wide search
            x_new, conf = find_peak_in_row(y, prev_x, wide_search=True)
            
        xs[y] = x_new
        confidence[y] = conf
        if np.isfinite(x_new):
            prev_x = x_new

    # Trace upward from start_row
    prev_x = xs[start_row]
    for y in range(start_row - 1, -1, -1):
        x_new, conf = find_peak_in_row(y, prev_x)
        if not np.isfinite(x_new) and crest_boost:
            # Fallback: try wide search
            x_new, conf = find_peak_in_row(y, prev_x, wide_search=True)
            
        xs[y] = x_new
        confidence[y] = conf
        if np.isfinite(x_new):
            prev_x = x_new

    # Fill small gaps with linear interpolation (more permissive to bridge gaps)
    s = pd.Series(xs)
    s = s.interpolate(method='linear', limit_direction='both', limit=25)
    xs = s.to_numpy(dtype=np.float32)

    # No inertia smoothing to keep every wiggle

    # Photocopy-style fusion: row tracer vs row maxima vs seam
    # 1) Fuse with row maxima when stronger (no margin to keep detail)
    for y in range(h):
        if not np.isfinite(row_max_xs[y]):
            continue
        x_row = xs[y]
        x_max = row_max_xs[y]
        p_max = ridge_prob[y, int(round(np.clip(x_max, 0, w - 1)))]
        p_row = ridge_prob[y, int(round(np.clip(x_row, 0, w - 1)))] if np.isfinite(x_row) else -1.0
        if p_max > p_row:
            xs[y] = x_max

    # 2) Optional seam fusion; skip when crest_boost is enabled
    if not crest_boost:
        xs_seam = _seam_path(prob)
        if xs_seam.size == h:
            xs_fused = xs.copy()
            for y in range(h):
                x_row = xs[y]
                x_seam = xs_seam[y]
                if not np.isfinite(x_seam):
                    continue
                p_seam = prob[y, int(round(np.clip(x_seam, 0, w - 1)))]
                p_row = prob[y, int(round(np.clip(x_row, 0, w - 1)))] if np.isfinite(x_row) else -1.0
                if p_seam > p_row:  # no margin, prefer stronger seam locally
                    xs_fused[y] = x_seam
            xs = xs_fused

    return xs, confidence

def trace_curve_multiscale(curve_mask, scale_min, scale_max, curve_type="GR", max_step=3, smooth_lambda=0.1, curv_lambda=0.01, hot_side=None):
    """
    Enhanced multi-scale curve tracing with 5 scales and weighted fusion.
    
    This improves detection by:
    - More granular scales for better feature capture
    - Weighted fusion based on confidence and scale
    - Scale-adaptive parameters for GR logs
    - Better handling of jagged Gamma Ray spikes
    
    Scales used: 1.0, 0.75, 0.5, 0.33, 0.25, 0.125 (6 scales)
    """
    if curve_mask is None or curve_mask.size == 0:
        return np.array([]), np.array([])
    
    # For GR curves, we want to allow very sharp peaks, so we lower the smoothing significantly.
    # For other curves (like Res), we keep it higher to avoid noise.
    # Note: We respect the passed smooth_lambda, but ensure it's not too high for GR.
    if curve_type.upper() == "GR" and smooth_lambda > 0.01:
        smooth_lambda = 0.001
    
    h, w = curve_mask.shape
    if h < 4 or w < 4:
        return trace_curve_with_dp(curve_mask, scale_min, scale_max, curve_type, max_step, smooth_lambda, hot_side)
    
    # Adaptive scale selection based on image content
    def adaptive_scale_selection(curve_mask, curve_type):
        """Choose optimal scales based on image characteristics"""
        h, w = curve_mask.shape
        prob = curve_mask.astype(np.float32) / 255.0
        
        if curve_type.upper() == "GR":
            # Analyze jaggedness for GR logs
            gray = (prob * 255).astype(np.uint8)
            
            # Edge detection to measure jaggedness
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Measure intensity variance (indicates sharp transitions)
            intensity_var = np.var(prob)
            
            # Determine scale set based on jaggedness
            if edge_density > 0.15 or intensity_var > 0.1:  # Very jagged
                scales = [1.0, 0.8, 0.6, 0.4, 0.2]  # More fine scales
            elif edge_density > 0.08 or intensity_var > 0.05:  # Moderately jagged
                scales = [1.0, 0.75, 0.5, 0.33, 0.25]  # Balanced
            else:  # Smooth
                scales = [1.0, 0.67, 0.33, 0.17]  # More coarse scales
        else:
            # Default for other curve types
            scales = [1.0, 0.67, 0.33, 0.17]
        
        # Filter scales that are too small
        valid_scales = []
        for scale in scales:
            h_s = max(4, int(h * scale))
            w_s = max(4, int(w * scale))
            if h_s >= 4 and w_s >= 4:
                valid_scales.append(scale)
        
        return valid_scales if len(valid_scales) >= 2 else [1.0, 0.5]
    
    # Use adaptive scale selection
    valid_scales = adaptive_scale_selection(curve_mask, curve_type)
    
    all_xs = []
    all_confs = []
    
    # Enhanced scale-adaptive parameters
    def get_scale_params(scale, curve_type, jaggedness_factor=1.0):
        """Get parameters optimized for each scale and curve type"""
        if curve_type.upper() == "GR":
            # Adjust based on jaggedness
            return {
                "smooth_lambda": max(0.000001, smooth_lambda * scale * jaggedness_factor),
                "max_step": max(1, int(max_step * scale * 1.2)),  # Moderate movement to prevent teleportation
                "rail_threshold": max(0.01, 0.1 * scale * jaggedness_factor),
                "curv_lambda": max(0.000001, 0.001 * scale * jaggedness_factor)
            }
        else:
            return {
                "smooth_lambda": smooth_lambda * scale,
                "max_step": max(1, int(max_step * scale)),
                "rail_threshold": 0.1 * scale,
                "curv_lambda": 0.05 * scale
            }
    
    # Calculate jaggedness factor for parameter tuning
    prob = curve_mask.astype(np.float32) / 255.0
    gray = (prob * 255).astype(np.uint8)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    jaggedness_factor = max(0.5, min(2.0, 1.0 + edge_density * 5))
    
    if len(valid_scales) < 2:
        return trace_curve_with_dp(curve_mask, scale_min, scale_max, curve_type, max_step, smooth_lambda, hot_side)
    
    prob = curve_mask.astype(np.float32) / 255.0
    
    for scale in valid_scales:
        if scale == 1.0:
            mask_scaled = curve_mask
            h_s, w_s = h, w
        else:
            h_s = max(4, int(h * scale))
            w_s = max(4, int(w * scale))
            
            # Use appropriate interpolation
            if scale > 0.5:
                mask_scaled = cv2.resize(curve_mask, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
            else:
                mask_scaled = cv2.resize(curve_mask, (w_s, h_s), interpolation=cv2.INTER_AREA)
        
        # Get scale-adaptive parameters
        params = get_scale_params(scale, curve_type)
        
        # Run DP at this scale
        xs_scaled, conf_scaled = trace_curve_with_dp(
            mask_scaled,
            scale_min=scale_min,
            scale_max=scale_max,
            curve_type=curve_type,
            max_step=params["max_step"],
            smooth_lambda=params["smooth_lambda"],
            curv_lambda=params["curv_lambda"],
            hot_side=hot_side,
        )
        
        # Upsample back to full resolution with sub-pixel precision
        if scale != 1.0 and xs_scaled.size > 0:
            xs_full = np.full(h, np.nan, dtype=np.float32)
            conf_full = np.zeros(h, dtype=np.float32)
            
            for y_full in range(h):
                y_scaled = y_full * scale
                y_int = int(y_scaled)
                y_frac = y_scaled - y_int
                
                if y_int < xs_scaled.size and np.isfinite(xs_scaled[y_int]):
                    # Linear interpolation for sub-pixel accuracy
                    x_scaled = xs_scaled[y_int] / scale
                    conf_scaled_val = conf_scaled[y_int]
                    
                    # Handle fractional positions
                    if y_int + 1 < xs_scaled.size and np.isfinite(xs_scaled[y_int + 1]):
                        x_next = xs_scaled[y_int + 1] / scale
                        conf_next = conf_scaled[y_int + 1]
                        
                        # Interpolate both position and confidence
                        x_full = x_scaled * (1 - y_frac) + x_next * y_frac
                        conf_full[y_full] = conf_scaled_val * (1 - y_frac) + conf_next * y_frac
                    else:
                        x_full = x_scaled
                        conf_full[y_full] = conf_scaled_val
                    
                    xs_full[y_full] = x_full
            
            xs_scaled = xs_full
            conf_scaled = conf_full
        
        all_xs.append(xs_scaled)
        all_confs.append(conf_scaled)
    
    # Nuclear option - ultra-aggressive peak detection for missed GR spikes
    def detect_local_peaks(prob_map, min_prominence=0.005, sensitivity_boost=True):
        """Nuclear option: ultra-aggressive peak detection for GR spike detection"""
        h, w = prob_map.shape
        peaks = []
        
        # Use raw probability map for maximum sensitivity (no blur)
        raw = prob_map
        
        search_window = 2       # Very small neighborhood
        
        # Multi-scale detection with different sensitivities
        for y in range(1, h-1):
            row = raw[y]
            
            # Find ALL local maxima, even tiny ones
            for x in range(1, w-1):
                # Check 3x3 neighborhood for local max
                local_max = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        if row[x] < raw[y+dy, x+dx]:
                            local_max = False
                            break
                    if not local_max:
                        break
                
                # Accept even tiny peaks for GR logs
                if local_max and row[x] > min_prominence:
                    # Minimal prominence requirement
                    window_size = min(8, w//8)
                    left_start = max(0, x-window_size)
                    right_end = min(w, x+window_size+1)
                    
                    left_region = row[left_start:x]
                    right_region = row[x+1:right_end]
                    
                    if len(left_region) > 0 and len(right_region) > 0:
                        # Use 20th percentile for valleys (very relaxed)
                        left_valley = np.percentile(left_region, 20)
                        right_valley = np.percentile(right_region, 20)
                        prominence = row[x] - max(left_valley, right_valley)
                        
                        # Use provided min_prominence
                        if prominence > min_prominence:
                            # Accept even flat peaks
                            peaks.append((y, x, prominence))
        
        # Ultra-aggressive edge peak detection
        if curve_type.upper() == "GR":
            for y in range(h):
                # Very low threshold for edge detection
                if prob_map[y, 0] > 0.1:  # Very low edge threshold
                    peaks.append((y, 0, prob_map[y, 0]))
                if prob_map[y, w-1] > 0.1:  # Very low edge threshold
                    peaks.append((y, w-1, prob_map[y, w-1]))
        
        # Accept all peaks, no filtering for maximum sensitivity
        return peaks  # Return all detected peaks, no filtering
    
    def is_near_peak(y, peaks, window=4):
        """Check if y coordinate is near a detected peak with expanded window"""
        return any(abs(py - y) <= window for py, px, prom in peaks)
    
    # AI-powered peak detection fallback
    def ai_detect_peaks(image_roi, curve_type="GR"):
        """Use Google Vision API to detect curve peaks as fallback"""
        if not VISION_API_AVAILABLE or vision_client is None:
            print("AI detection: Vision API not available")
            return []
        
        print(f"AI detection: Starting for {curve_type} curve...")
        try:
            # Ensure ROI is proper format for Vision API
            if len(image_roi.shape) == 3:
                # Convert BGR to RGB for Vision API
                roi_rgb = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
            else:
                roi_rgb = cv2.cvtColor(image_roi, cv2.COLOR_GRAY2RGB)
            
            # Convert to bytes
            _, buffer = cv2.imencode('.jpg', roi_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_bytes = buffer.tobytes()
            
            print("AI detection: Sending to Vision API...")
            
            # Use Vision API for object detection
            image = vision.Image(content=image_bytes)
            response = vision_client.object_localization(image=image)
            
            peaks = []
            objects = response.localized_object_annotations
            print(f"AI detection: Found {len(objects)} objects")
            
            h, w = image_roi.shape[:2]
            
            for obj in objects:
                print(f"AI detection: Object '{obj.name}' with confidence {obj.score:.2f}")
                if obj.name.lower() in ['line', 'curve', 'graph', 'drawing'] and obj.score > 0.3:
                    # Convert normalized vertices to pixel coordinates
                    vertices = obj.bounding_poly.normalized_vertices
                    if len(vertices) >= 4:
                        # Calculate center points for peaks
                        y_coords = [v.y * h for v in vertices]
                        x_coords = [v.x * w for v in vertices]
                        
                        # Create peaks along detected curve
                        for i in range(len(y_coords)):
                            peaks.append((int(y_coords[i]), int(x_coords[i]), obj.score))
            
            # Also try document text detection for numeric patterns
            try:
                doc_response = vision_client.document_text_detection(image=image)
                texts = doc_response.text_annotations
                if texts:
                    print(f"AI detection: Found text - {texts[0].description[:50]}...")
            except Exception as e:
                print(f"AI text detection failed: {e}")
            
            print(f"AI detection: Returning {len(peaks)} AI-detected peaks")
            return peaks
            
        except Exception as e:
            print(f"AI detection error: {e}")
            return []
        
        try:
            # Convert ROI to bytes for Vision API
            _, buffer = cv2.imencode('.jpg', image_roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_bytes = buffer.tobytes()
            
            # Use Vision API to detect curve features
            image = vision.Image(content=image_bytes)
            response = vision_client.text_detection(image=image)
            
            # Alternative: use document text detection for better curve detection
            response = vision_client.document_text_detection(image=image)
            
            # Extract text and analyze for curve patterns
            texts = response.text_annotations
            if texts:
                full_text = texts[0].description
                # Look for numeric patterns that might indicate curve values
                numbers = re.findall(r'\d+\.?\d*', full_text)
                
            # Use object detection for curve features (if available)
            response = vision_client.object_localization(image=image)
            objects = response.localized_object_annotations
            
            peaks = []
            for obj in objects:
                if obj.name.lower() in ['line', 'curve', 'graph']:
                    # Convert bounding box to peak coordinates
                    vertices = obj.bounding_poly.normalized_vertices
                    if len(vertices) >= 4:
                        # Calculate center as peak
                        center_x = (vertices[0].x + vertices[2].x) * w / 2
                        center_y = (vertices[0].y + vertices[2].y) * h / 2
                        peaks.append((int(center_y), int(center_x), 0.8))  # High confidence
            
            return peaks
        except Exception as e:
            print(f"AI peak detection failed: {e}")
            return []
    
    # Hybrid AI + Traditional peak detection
    peaks_traditional = detect_local_peaks(prob, min_prominence=0.005) if curve_type.upper() == "GR" else []
    peaks_ai = ai_detect_peaks(roi, curve_type) if curve_type.upper() == "GR" and VISION_API_AVAILABLE else []
    
    # Merge traditional and AI peaks
    all_peaks = peaks_traditional + peaks_ai
    
    # Final refinement with AI-enhanced sub-pixel accuracy
    def refine_subpixel_parabola(mask, xs, prob_map):
        """Refine positions using parabolic sub-pixel interpolation
        Given peak pixel x, fits parabola to (x-1), x, (x+1) to find true max.
        
        Formula: offset = 0.5 * (left - right) / (left - 2*center + right)
        """
        if mask is None or xs is None:
            return xs, np.zeros_like(xs)
        
        h, w = prob_map.shape
        xs_refined = xs.copy()
        subpixel_conf = np.ones(h, dtype=np.float32)
        
        for y in range(h):
            if not np.isfinite(xs[y]):
                continue
                
            x = int(round(xs[y]))
            if x < 1 or x >= w-1:
                continue
            
            # Get intensity values at peak and neighbors
            center = prob_map[y, x]
            left = prob_map[y, x-1] if x-1 >= 0 else center
            right = prob_map[y, x+1] if x+1 < w else center
            
            # Check if we have a valid parabola (concave down)
            denom = left - 2*center + right
            if abs(denom) > 1e-6:  # Avoid division by zero
                # Calculate sub-pixel offset
                offset = 0.5 * (left - right) / denom
                
                # Only apply if offset is reasonable (< 1 pixel)
                if abs(offset) < 1.0:
                    xs_refined[y] = x + offset
                else:
                    xs_refined[y] = xs[y]
            else:
                xs_refined[y] = xs[y]
        
        return xs_refined, subpixel_conf
    
    # Curvature-based refinement for missed peaks
    def curvature_based_refinement(xs, prob_map, curve_type="GR"):
        """Refine based on local curvature analysis to catch missed peaks"""
        h, w = prob_map.shape
        xs_refined = xs.copy()
        curv_conf = np.ones(h, dtype=np.float32)
        # Only meaningful for jagged GR logs; for smooth curves (RHOB, DTC, etc.)
        # the ±5px snap with a 5% threshold causes hundreds of random jumps on a
        # noisy prob_map, making the trace more erratic than it started.
        if curve_type.upper() != "GR":
            return xs_refined, curv_conf
        # Calculate curvature using second derivative
        valid_mask = np.isfinite(xs)
        if np.sum(valid_mask) < 5:
            return xs_refined, curv_conf
        
        # Fill NaNs for curvature calculation
        xs_smooth = xs.copy()
        xs_smooth[~valid_mask] = np.interp(np.where(~valid_mask)[0], 
                                          np.where(valid_mask)[0], 
                                          xs[valid_mask])
        
        # Calculate curvature (second derivative)
        curvature = np.gradient(np.gradient(xs_smooth))
        
        # Identify high curvature regions (likely missed peaks)
        high_curvature = np.abs(curvature) > np.percentile(np.abs(curvature[valid_mask]), 80)
        
        # Refine high curvature regions
        for y in range(h):
            if not valid_mask[y] or not high_curvature[y]:
                continue
                
            # Search locally around current position
            x_current = int(round(xs[y]))
            search_radius = 5
            
            # Define search window
            start = max(0, x_current - search_radius)
            end = min(w, x_current + search_radius + 1)
            
            # Find local maximum in probability map
            local_prob = prob_map[y, start:end]
            if local_prob.size > 0:
                local_max_idx = np.argmax(local_prob)
                local_max_val = local_prob[local_max_idx]
                
                # Only refine if local max is significantly better
                current_val = prob_map[y, x_current] if 0 <= x_current < w else 0
                if local_max_val > current_val * 1.05:  # 5% improvement threshold (Medium Greedy)
                    xs_refined[y] = start + local_max_idx
        
        return xs_refined, curv_conf

    def ensure_peak_crests(xs, conf, prob_map, peaks, hot_side=None, y_merge_window=3):
        """Ensure each vertical GR peak cluster has at least one crest sample.

        For each cluster of peaks in Y, we pick a single row and move that
        row's sample to the crest (tip) on the hot side, but only if the
        crest candidate is at least as strong in the probability map and
        farther toward the hot side than the current position. This avoids
        reintroducing zig-zag artifacts while still guaranteeing a tip dot.
        """
        if xs is None or peaks is None:
            return xs, conf

        if len(peaks) == 0:
            return xs, conf

        h, w = prob_map.shape
        xs_out = xs.copy()
        conf_out = conf.copy()

        # Filter to reasonably strong peaks to avoid pure noise
        strong_peaks = []
        for py, px, prom in peaks:
            if not np.isfinite(py) or not np.isfinite(px):
                continue
            py_i = int(py)
            px_i = int(px)
            if py_i < 0 or py_i >= h or px_i < 0 or px_i >= w:
                continue
            # Use either provided prominence or local prob as strength
            strength = float(prom) if np.isfinite(prom) else float(prob_map[py_i, px_i])
            if strength <= 0.005:
                continue
            strong_peaks.append((py_i, px_i, strength))

        if not strong_peaks:
            return xs_out, conf_out

        # Group peaks into vertical clusters by Y
        strong_peaks.sort(key=lambda p: p[0])
        clusters = []
        current = [strong_peaks[0]]
        for py, px, prom in strong_peaks[1:]:
            if abs(py - current[-1][0]) <= y_merge_window:
                current.append((py, px, prom))
            else:
                clusters.append(current)
                current = [(py, px, prom)]
        clusters.append(current)

        for cluster in clusters:
            # Pick a crest candidate within this vertical group.
            # Strategy: Find the strongest peak, but prefer "tips" (hot side)
            # if they are reasonably strong (at least 50% of the max score in the cluster).
            
            # 1. Find max score in cluster
            max_s = -1.0
            for py, px, prom in cluster:
                s = float(prob_map[py, px])
                if s > max_s:
                    max_s = s
            
            # 2. Select best candidate among those with reasonable score
            threshold = max(0.01, max_s * 0.5)
            best_key = None
            best_py, best_px, best_score = None, None, -1.0
            
            for py, px, prom in cluster:
                score = float(prob_map[py, px])
                if score < threshold:
                    continue
                    
                # Bias toward the hot side for the "key" (tie-breaking or primary sort)
                if hot_side == "right":
                    # Prioritize right-most X, then score
                    key = (px, score)
                elif hot_side == "left":
                    # Prioritize left-most X (largest -px), then score
                    key = (-px, score)
                else:
                    # Just score
                    key = (score, 0)

                # Select best candidate based on direction-aware key
                if best_key is None or key > best_key:
                    best_key = key
                    best_score = score
                    best_py, best_px = py, px

            if best_py is None:
                continue

            y = int(best_py)
            x_peak = int(best_px)
            if y < 0 or y >= h or x_peak < 0 or x_peak >= w:
                continue

            x_curr = xs_out[y] if np.isfinite(xs_out[y]) else None
            
            # TELEPORTATION GUARD:
            # Prevent snapping to peaks far from the existing trace.
            # If the current row has no trace, check neighbors to establish context.
            ref_x = x_curr
            if ref_x is None:
                # Search neighbors for a valid reference point
                for offset in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                    ny = y + offset
                    if 0 <= ny < h and np.isfinite(xs_out[ny]):
                        ref_x = xs_out[ny]
                        break
            
            # If we have a reference, enforce distance limit (30px)
            if ref_x is not None:
                if abs(x_peak - ref_x) > 30:
                    continue
            else:
                # If no reference nearby (isolated point), unsafe to snap -> skip
                continue

            x_curr_int = int(round(x_curr)) if x_curr is not None else None

            p_peak = float(prob_map[y, x_peak])
            p_curr = float(prob_map[y, x_curr_int]) if x_curr_int is not None and 0 <= x_curr_int < w else 0.0

            move = False
            if x_curr is None:
                move = True
            else:
                if hot_side == "right":
                    if x_peak > x_curr_int and p_peak >= max(0.1, p_curr * 0.9):
                        move = True
                elif hot_side == "left":
                    if x_peak < x_curr_int and p_peak >= max(0.1, p_curr * 0.9):
                        move = True
                else:
                    if p_peak >= p_curr * 1.05:
                        move = True

            if not move:
                continue

            xs_out[y] = float(x_peak)
            conf_out[y] = max(conf_out[y], p_peak)

        return xs_out, conf_out

    # Apply final refinements
    xs_fused = np.full(h, np.nan, dtype=np.float32)
    confidence = np.zeros(h, dtype=np.float32)
    
    # Peak-aware fusion with curvature refinement
    peaks = detect_local_peaks(prob, min_prominence=0.005)
    
    for y in range(h):
        valid_indices = []
        valid_xs = []
        valid_confs = []
        cand_scales = []  # accumulator for per-candidate scale values (renamed to avoid shadowing outer valid_scales)
        
        for i, (xs_s, conf_s, scale) in enumerate(zip(all_xs, all_confs, valid_scales)):
            if xs_s is not None and y < xs_s.size and np.isfinite(xs_s[y]):
                x_int = int(round(xs_s[y]))
                if 0 <= x_int < w:
                    valid_indices.append(i)
                    valid_xs.append(xs_s[y])
                    valid_confs.append(conf_s[y])
                    cand_scales.append(scale)

        # Confidence-weighted fusion across all scales
        if valid_xs:
            total_conf = sum(valid_confs)
            if total_conf > 0:
                xs_fused[y] = sum(x * c for x, c in zip(valid_xs, valid_confs)) / total_conf
                confidence[y] = max(valid_confs)
            else:
                xs_fused[y] = float(np.mean(valid_xs))
                confidence[y] = 0.0

    # Push sampled points to hot-side crest tips
    if peaks:
        xs_fused, confidence = ensure_peak_crests(
            xs_fused, confidence, prob, peaks, hot_side=hot_side
        )

    return xs_fused, confidence


def trace_curve_greedy_peaks(mask, max_jump=30, min_prob=0.02):
    """Trace curve by greedily following the strongest peak in each row.

    This tracer finds the brightest pixel in each row within a search window,
    with NO smoothness penalty - it follows every peak and valley exactly.

    Args:
        mask: Probability map (0-255)
        max_jump: Maximum horizontal jump between rows
        min_prob: Minimum probability to consider

    Returns:
        Array of x-coordinates, one per row
    """
    if mask is None:
        return None

    h, w = mask.shape[:2]
    prob = mask.astype(np.float32) / 255.0
    xs = np.full(h, np.nan, dtype=np.float32)

    row_max_vals = prob.max(axis=1)
    row_max_xs = prob.argmax(axis=1).astype(np.float32)

    valid_rows = row_max_vals >= min_prob

    if not np.any(valid_rows):
        return xs

    start_row = int(np.argmax(row_max_vals))
    xs[start_row] = row_max_xs[start_row]

    current_x = xs[start_row]
    for y in range(start_row + 1, h):
        row = prob[y]
        x0 = max(0, int(current_x) - max_jump)
        x1 = min(w, int(current_x) + max_jump + 1)
        window = row[x0:x1]
        if window.max() >= min_prob:
            best_local = np.argmax(window)
            current_x = float(x0 + best_local)
            xs[y] = current_x

    current_x = xs[start_row]
    for y in range(start_row - 1, -1, -1):
        row = prob[y]
        x0 = max(0, int(current_x) - max_jump)
        x1 = min(w, int(current_x) + max_jump + 1)
        window = row[x0:x1]
        if window.max() >= min_prob:
            best_local = np.argmax(window)
            current_x = float(x0 + best_local)
            xs[y] = current_x

    return xs


def refine_to_smart_edges(mask, xs, min_prob=0.005):
    """
    Smart refinement based on local geometry (SR Tuned):
    - Vertical runs (width < 10px): Center on ink (CoM).
    - Spikes/Bumps: Snap to EDGE if deviation > 3.0px.
    """
    if mask is None or xs is None:
        return xs
    
    h, w = mask.shape
    prob = mask.astype(np.float32)
    xs_smart = xs.copy()
    
    # 1. Compute local median
    try:
        s = pd.Series(xs)
        local_med = s.rolling(window=15, center=True, min_periods=1).median().to_numpy() # Increased window for SR
    except:
        local_med = xs.copy()

    for y in range(h):
        x = xs_smart[y]
        if not np.isfinite(x):
            continue
            
        ix = int(round(x))
        if ix < 0 or ix >= w:
            continue
            
        # 2. Find connected ink chunk
        # TELEPORTATION GUARD: Reduced from 200 to 30 to prevent snapping to distant noise
        w_start = max(0, ix - 30)
        w_end = min(w, ix + 31)
        row_slice = prob[y, w_start:w_end]
        
        # Simple threshold
        ink_indices = np.where(row_slice > min_prob * 255)[0] if prob.max() > 1.0 else np.where(row_slice > min_prob)[0]
        if ink_indices.size == 0:
            continue
            
        ink_indices_global = ink_indices + w_start
        min_ink = ink_indices_global[0]
        max_ink = ink_indices_global[-1]
        ink_width = max_ink - min_ink
        
        # 3. Geometric Decision
        med_val = local_med[y]
        if not np.isfinite(med_val):
            med_val = x
            
        diff = x - med_val
        
        # LOGIC FOR 2x SUPER-RES:
        # Width < 14px (was 10) -> Center
        # Deviation > 5.0px (was 3.0) -> Snap
        
        if ink_width < 14:
            # Narrow Line -> Center of Mass
            weights = row_slice[ink_indices]
            coords = ink_indices_global
            total_w = weights.sum()
            if total_w > 0:
                xs_smart[y] = (coords * weights).sum() / total_w
            else:
                xs_smart[y] = (min_ink + max_ink) / 2.0
        else:
            # Wide/Feature
            if diff > 5.0:
                xs_smart[y] = float(max_ink) # Snap Right
            elif diff < -5.0:
                xs_smart[y] = float(min_ink) # Snap Left
            elif abs(diff) < 2.5:
                # Stable Zone -> Center of Mass
                weights = row_slice[ink_indices]
                coords = ink_indices_global
                total_w = weights.sum()
                if total_w > 0:
                    xs_smart[y] = (coords * weights).sum() / total_w
                else:
                    xs_smart[y] = (min_ink + max_ink) / 2.0
            else:
                # Transition Zone -> Keep current
                xs_smart[y] = x

    return xs_smart

def refine_trace_with_local_maxima(mask, xs, max_shift=6, dominance_ratio=1.1, min_prob=0.2):
    """Nudge the DP path toward obvious local maxima in the prob mask.

    For each row, look in a small window around the current DP x and, when
    there is a clearly stronger nearby maximum, move the x coordinate toward
    the probability-weighted centroid of that local peak. This keeps the
    path glued to the same physical curve while following its wiggles more
    tightly.
    """
    if mask is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size == 0:
        return xs

    h, w = mask.shape[:2]
    if h < 1 or w < 1:
        return xs

    prob = mask.astype(np.float32) / 255.0
    xs_ref = xs.copy()

    n_rows = min(h, xs_ref.size)
    for y in range(n_rows):
        x = xs_ref[y]
        if not np.isfinite(x):
            continue

        x_c = int(round(float(x)))
        if x_c < 0 or x_c >= w:
            continue

        row = prob[y]
        x0 = max(0, x_c - max_shift)
        x1 = min(w, x_c + max_shift + 1)
        window = row[x0:x1]
        if window.size == 0:
            continue

        max_p = float(window.max())
        if max_p < min_prob:
            continue

        # Compare the best pixel in the window to the current DP location.
        local_peak_idx = int(np.argmax(window))
        x_peak = x0 + local_peak_idx
        p_peak = float(row[x_peak])
        p_dp = float(row[x_c])
        if p_dp <= 0:
            p_dp = 1e-6

        if p_peak >= dominance_ratio * p_dp:
            # Use a weighted centroid within the local window, restricted to
            # the top part of the peak, so the path follows the center of the
            # curve stroke instead of a single edge pixel.
            xs_local = np.arange(x0, x1, dtype=np.float32)
            weights = window.astype(np.float32)
            peak_mask = weights >= max_p * 0.6

            try:
                # If we have no clearly strong pixels, fall back to any
                # non-zero weights.
                if not np.any(peak_mask):
                    peak_mask = weights > 0.0
                idx_strong = np.flatnonzero(peak_mask)
                if idx_strong.size > 0:
                    # Group consecutive strong pixels into contiguous
                    # segments so we can snap to the center of a physical
                    # stroke rather than an arbitrary mix of nearby blobs.
                    start = idx_strong[0]
                    prev = idx_strong[0]
                    segments = []
                    for idx in idx_strong[1:]:
                        if idx == prev + 1:
                            prev = idx
                        else:
                            segments.append((start, prev))
                            start = idx
                            prev = idx
                    segments.append((start, prev))

                    # Prefer the segment that actually contains the local
                    # peak; otherwise choose the closest segment by center.
                    seg_best = None
                    for s, e in segments:
                        if s <= local_peak_idx <= e:
                            seg_best = (s, e)
                            break
                    if seg_best is None and segments:
                        seg_best = min(
                            segments,
                            key=lambda se: abs((se[0] + se[1]) * 0.5 - local_peak_idx),
                        )

                    if seg_best is not None:
                        s, e = seg_best
                        seg_slice = slice(s, e + 1)
                        seg_weights = weights[seg_slice]
                        seg_xs = xs_local[seg_slice]
                        wsum = float(seg_weights.sum())
                        if wsum > 0.0:
                            x_centroid = float((seg_xs * seg_weights).sum() / wsum)
                        else:
                            x_centroid = float(seg_xs.mean())
                        xs_ref[y] = x_centroid
                        continue

            except Exception:
                # If anything about the segment-based logic misbehaves for a
                # particular row, quietly fall back to the simpler
                # peak-centered weighted centroid used previously.
                pass

            # Fallback: original behavior - centroid of the strong part of
            # the window around the dominant peak.
            if not np.any(peak_mask):
                peak_mask = weights > 0.0
            weights_centroid = weights * peak_mask.astype(np.float32)
            wsum = float(weights_centroid.sum())
            if wsum > 0.0:
                x_centroid = float((xs_local * weights_centroid).sum() / wsum)
                xs_ref[y] = x_centroid

    return xs_ref



def refine_peaks_and_valleys(mask, xs, search_radius=25, min_prob=0.1):
    """Specifically refine peaks and valleys where the curve changes direction.
    
    This function detects where the traced curve has local extrema (peaks/valleys)
    and searches more aggressively in those regions to find the true curve position.
    
    Args:
        mask: Probability map (0-255)
        xs: Array of x-coordinates from initial trace
        search_radius: How far to search horizontally at peaks/valleys
        min_prob: Minimum probability to consider a pixel as curve
    
    Returns:
        Refined x-coordinates
    """
    if mask is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size < 5:
        return xs
    
    h, w = mask.shape[:2]
    if h < 5 or w < 5:
        return xs
    
    prob = mask.astype(np.float32) / 255.0
    xs_ref = xs.copy()
    
    # Detect peaks and valleys by looking at the derivative of x positions
    # A peak is where x goes from increasing to decreasing (or vice versa for valley)
    valid_mask = np.isfinite(xs_ref)
    if np.sum(valid_mask) < 5:
        return xs_ref
    
    # Fill gaps for derivative calculation
    xs_filled = xs_ref.copy()
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0:
        xs_filled[:valid_indices[0]] = xs_ref[valid_indices[0]]
        xs_filled[valid_indices[-1]:] = xs_ref[valid_indices[-1]]
        # Linear interpolation for gaps
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            if end_idx - start_idx > 1:
                xs_filled[start_idx:end_idx] = np.linspace(
                    xs_ref[start_idx], xs_ref[end_idx], end_idx - start_idx
                )
    
    # Calculate first derivative (velocity)
    dx = np.diff(xs_filled)
    
    # Find zero crossings in derivative (peaks and valleys) - vectorized
    # Sign change indicates peak or valley
    sign_change = (dx[:-1] * dx[1:]) < 0
    # Large magnitude change also indicates sharp turn; use a slightly lower
    # threshold so more subtle bends get refined
    large_change = np.abs(np.diff(dx)) > 2
    extrema_mask = sign_change | large_change
    extrema_rows = np.where(extrema_mask)[0] + 1  # +1 because diff reduces length
    
    # Refine at each extremum with a wider search
    for y in extrema_rows:
        if y < 0 or y >= h:
            continue
        x = xs_ref[y]
        if not np.isfinite(x):
            continue
        
        x_c = int(round(float(x)))
        if x_c < 0 or x_c >= w:
            continue
        
        row = prob[y]
        x0 = max(0, x_c - search_radius)
        x1 = min(w, x_c + search_radius + 1)
        window = row[x0:x1]
        if window.size == 0:
            continue
        
        max_p = float(window.max())
        if max_p < min_prob:
            continue
        
        # Find the peak position
        local_peak_idx = int(np.argmax(window))
        x_peak = x0 + local_peak_idx
        
        # Use weighted centroid of strong pixels around the peak
        # Use HIGH power (4) to stay very close to the peak
        xs_local = np.arange(x0, x1, dtype=np.float32)
        weights = window.astype(np.float32)
        peak_mask = weights >= max_p * 0.3  # Include pixels at 30% of peak
        
        if np.any(peak_mask):
            weights_masked = (weights * peak_mask.astype(np.float32)) ** 4
            wsum = float(weights_masked.sum())
            if wsum > 0:
                x_centroid = float((xs_local * weights_masked).sum() / wsum)
                
                # EXTREMUM PUSH: If this is a sharp peak/valley, push towards the extremity
                # Check 3 rows above and below to see curvature
                if y > 2 and y < h - 3:
                    prev_x = np.nanmedian(xs_ref[y-3:y])
                    next_x = np.nanmedian(xs_ref[y+1:y+4])
                    if np.isfinite(prev_x) and np.isfinite(next_x):
                        # Detected a peak (point is > neighbors) or valley (point is < neighbors)
                        # Use 0.2 buffer to catch almost all real features while ignoring microscopic noise.
                        # LOGIC UPDATE: Allow "Plateaus" (flat tops) to be detected as peaks.
                        # Strict peak: > prev AND > next
                        # Plateau start: > prev AND >= next
                        # Plateau end: >= prev AND > next
                        buf = 0.2
                        
                        # Right Peak (High value)
                        # Check if strictly higher than at least one side, and at least equal (within tolerance) to the other
                        is_right_peak = (x_centroid > prev_x + buf and x_centroid >= next_x - buf) or \
                                        (x_centroid >= prev_x - buf and x_centroid > next_x + buf)

                        # Left Peak (Low value) - Valley
                        is_left_peak = (x_centroid < prev_x - buf and x_centroid <= next_x + buf) or \
                                       (x_centroid <= prev_x + buf and x_centroid < next_x - buf)
                        
                        # Helper to find connected ink span around the centroid
                        def get_connected_range(weights, center_idx):
                            # Ensure center_idx is within bounds
                            center_idx = max(0, min(len(weights)-1, center_idx))
                            
                            # If we landed on empty space, find nearest ink
                            if weights[center_idx] == 0:
                                valid = np.where(weights > 0)[0]
                                if valid.size == 0:
                                    return center_idx, center_idx
                                # Closest valid index
                                center_idx = valid[np.abs(valid - center_idx).argmin()]
                                
                            # Expand Left
                            l_idx = center_idx
                            while l_idx > 0 and weights[l_idx-1] > 0:
                                l_idx -= 1
                                
                            # Expand Right
                            r_idx = center_idx
                            while r_idx < len(weights)-1 and weights[r_idx+1] > 0:
                                r_idx += 1
                                
                            return l_idx, r_idx

                        center_idx_local = int(round(x_centroid - x0))
                        
                        if is_right_peak:
                            # Find rightmost CONNECTED pixel
                            l_idx, r_idx = get_connected_range(weights, center_idx_local)
                            rightmost_x = x0 + r_idx
                            xs_ref[y] = float(rightmost_x)
                            
                        elif is_left_peak:
                            # Find leftmost CONNECTED pixel
                            l_idx, r_idx = get_connected_range(weights, center_idx_local)
                            leftmost_x = x0 + l_idx
                            xs_ref[y] = float(leftmost_x)
                        else:
                            xs_ref[y] = x_centroid
                else:
                    xs_ref[y] = x_centroid
                
                # Also refine a couple rows above and below (reduced from ±5 to ±2 for speed)
                half_radius = search_radius // 3
                for dy in [-2, -1, 1, 2]:
                    y2 = y + dy
                    if y2 < 0 or y2 >= h:
                        continue
                    x2 = xs_ref[y2]
                    if not np.isfinite(x2):
                        continue
                    
                    x2_c = int(round(float(x2)))
                    row2 = prob[y2]
                    x0_2 = max(0, x2_c - half_radius)
                    x1_2 = min(w, x2_c + half_radius + 1)
                    window2 = row2[x0_2:x1_2]
                    if window2.size == 0:
                        continue
                    
                    # Simplified: just snap to max in window
                    best_idx = np.argmax(window2)
                    if window2[best_idx] >= min_prob:
                        xs_ref[y2] = float(x0_2 + best_idx)
    
    return xs_ref


def ensure_peaks_have_points(
    mask,
    xs,
    min_prob=0.08,
    min_peak_prominence=0.03,
    max_shift=40,
):
    """Ensure every significant peak in the probability map has a traced point.
    
    Optimized version using vectorized operations.
    
    Args:
        mask: Probability map (0-255)
        xs: Array of x-coordinates from trace
        min_prob: Minimum probability to consider a peak
        min_peak_prominence: Minimum prominence (height above neighbors) for a peak
    
    Returns:
        Refined x-coordinates with peaks properly captured
    """
    if mask is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size < 3:
        return xs
    
    h, w = mask.shape[:2]
    if h < 3 or w < 3:
        return xs
    
    prob = mask.astype(np.float32) / 255.0
    xs_ref = xs.copy()
    
    # Vectorized peak detection: find where each pixel is greater than both neighbors
    # This is much faster than row-by-row Python loops
    left_shift = np.roll(prob, 1, axis=1)
    right_shift = np.roll(prob, -1, axis=1)
    is_peak = (prob > left_shift) & (prob > right_shift) & (prob >= min_prob)
    # Zero out edges
    is_peak[:, 0] = False
    is_peak[:, -1] = False
    
    # Process only rows that have peaks
    rows_with_peaks = np.where(np.any(is_peak, axis=1))[0]
    
    for y in rows_with_peaks:
        row = prob[y]
        current_x = xs_ref[y]
        
        # Get candidate peak positions in this row
        peak_xs = np.where(is_peak[y])[0]
        if peak_xs.size == 0:
            continue
        
        peak_probs = row[peak_xs]
        # Estimate local prominence vs immediate neighbors
        left_vals = row[peak_xs - 1]
        right_vals = row[peak_xs + 1]
        prominences = peak_probs - np.maximum(left_vals, right_vals)
        valid_mask = prominences >= min_peak_prominence
        if not np.any(valid_mask):
            continue
        peak_xs = peak_xs[valid_mask]
        peak_probs = peak_probs[valid_mask]
        prominences = prominences[valid_mask]
        
        # Sort peaks by strength descending
        sort_idx = np.argsort(-peak_probs)
        peak_xs = peak_xs[sort_idx]
        peak_probs = peak_probs[sort_idx]
        
        # If current trace is far from the best peak, snap to it
        if np.isfinite(current_x):
            current_x_int = int(round(current_x))
            current_prob = row[current_x_int] if 0 <= current_x_int < w else 0
            
            # Prefer a peak near the current trace when possible
            target_peak_x = None
            for px in peak_xs:
                if abs(px - current_x) <= max_shift:
                    target_peak_x = px
                    break
            
            # If no peak is close enough, DO NOT jump to a far-away peak.
            # Stick to the current trace.
            if target_peak_x is None:
                continue

            target_prob = row[target_peak_x]
            
            # Only move if the nearby target is clearly stronger than current position
            if target_prob > current_prob * 1.05:
                x0 = max(0, target_peak_x - 6)
                x1 = min(w, target_peak_x + 7)
                window = row[x0:x1]
                if window.size == 0:
                    continue
                xs_local = np.arange(x0, x1, dtype=np.float32)
                weights = window ** 3
                wsum = weights.sum()
                if wsum > 0:
                    xs_ref[y] = float((xs_local * weights).sum() / wsum)
        else:
            xs_ref[y] = float(peak_xs[0])
    
    return xs_ref


def _push_crest_hot_side(mask, xs, hot_side, curve_type=None, min_prob=0.01, max_shift=30):
    # Only apply this aggressive crest push for GR-type curves
    if curve_type is not None and str(curve_type).upper() != "GR":
        return xs
    if mask is None or xs is None:
        return xs
    if hot_side not in ("left", "right"):
        return xs
    if not hasattr(xs, "size") or xs.size == 0:
        return xs
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return xs
    prob = mask.astype(np.float32) / 255.0
    xs_out = xs.copy()
    moved = 0
    for y in range(h):
        x = xs_out[y]
        if not np.isfinite(x):
            continue
        x_int = int(round(x))
        if x_int < 0 or x_int >= w:
            continue
        row = prob[y]
        if hot_side == "right":
            start = x_int
            end = min(w, x_int + max_shift + 1)
        else:
            start = max(0, x_int - max_shift)
            end = x_int + 1
        if end <= start:
            continue
        seg = row[start:end]
        if seg.size == 0:
            continue
        # Find all pixels above a very low probability floor and move to the
        # **furthest** such pixel toward the hot side. This guarantees that,
        # wherever the GR ink extends horizontally, at least some rows will
        # put their sample at the outer tip.
        if hot_side == "right":
            cand = np.where(seg >= min_prob)[0]
            if cand.size == 0:
                continue
            peak_idx = int(cand[-1])  # furthest right
        else:
            cand = np.where(seg >= min_prob)[0]
            if cand.size == 0:
                continue
            peak_idx = int(cand[0])  # furthest left
        x_peak = start + peak_idx
        if x_peak != x_int:
            xs_out[y] = float(x_peak)
            moved += 1
    try:
        if moved > 0:
            print(f"[crest] moved {moved} rows toward {hot_side} (curve_type={curve_type})")
        else:
            print(f"[crest] no moves (hot_side={hot_side}, curve_type={curve_type})")
    except Exception:
        pass
    return xs_out


def ensure_gr_peak_crests(xs, prob_map, hot_side=None, min_prob=0.002, y_merge_window=5, max_shift_frac=0.6, max_dx_pixels=None):
    """For GR colored-mode traces, guarantee at least one crest sample per spike.

    This is a conservative helper that, for each vertical cluster of rows, moves
    at most ONE row further toward the hot-side tip, based on the probability
    map. It avoids reshaping the whole trace while ensuring every spike has a
    dot at its outermost visible tip.
    """
    if xs is None or prob_map is None:
        return xs
    if hot_side not in ("left", "right"):
        return xs
    if not hasattr(xs, "size") or xs.size == 0:
        return xs

    h, w = prob_map.shape
    n = xs.size
    m = min(h, n)
    if m == 0:
        return xs

    prob = prob_map[:m].astype(np.float32)
    xs_out = xs.copy()

    # Allow large moves, but not across the entire track; cap at a fraction
    # of the track width.
    # TELEPORTATION GUARD: Cap at 30px to prevent shooting off to distant noise
    if max_dx_pixels is not None:
        max_dx_allowed = max(1, int(max_dx_pixels))
    else:
        max_dx_allowed = min(30, max(1, int(max_shift_frac * w)))

    # 1) Build crest candidates per row
    candidates = []  # (y, crest_x, dx)
    for y in range(m):
        x_curr = xs_out[y]
        if not np.isfinite(x_curr):
            continue
        row = prob[y]
        x_curr_int = int(round(x_curr))
        if x_curr_int < 0 or x_curr_int >= w:
            continue

        on = row >= float(min_prob)
        if not np.any(on):
            continue

        if bool(on[x_curr_int]):
            l_idx = int(x_curr_int)
            r_idx = int(x_curr_int)
        else:
            idxs = np.where(on)[0]
            if idxs.size == 0:
                continue
            nearest = int(idxs[int(np.argmin(np.abs(idxs - x_curr_int)))])
            if abs(nearest - x_curr_int) > max_dx_allowed:
                continue
            l_idx = int(nearest)
            r_idx = int(nearest)

        while l_idx > 0 and bool(on[l_idx - 1]):
            l_idx -= 1
        while r_idx + 1 < w and bool(on[r_idx + 1]):
            r_idx += 1

        crest_x = int(r_idx if hot_side == "right" else l_idx)
        dx = crest_x - x_curr_int if hot_side == "right" else x_curr_int - crest_x
        if dx <= 0 or dx > max_dx_allowed:
            continue
        candidates.append((y, crest_x, dx))

    if not candidates:
        return xs_out

    # 2) Group into vertical clusters by Y (each cluster ≈ one spike)
    candidates.sort(key=lambda c: c[0])
    clusters = []
    current = [candidates[0]]
    for y, crest_x, dx in candidates[1:]:
        if abs(y - current[-1][0]) <= y_merge_window:
            current.append((y, crest_x, dx))
        else:
            clusters.append(current)
            current = [(y, crest_x, dx)]
    clusters.append(current)

    # 3) For each cluster, move exactly one row: the one that can move
    #    furthest toward the hot side.
    moved_clusters = 0
    for cluster in clusters:
        y_best, x_best, dx_best = max(cluster, key=lambda t: t[2])
        dx_keep = max(1, int(round(float(dx_best) * 0.8)))
        keep = [t for t in cluster if int(t[2]) >= dx_keep]
        keep.sort(key=lambda t: abs(int(t[0]) - int(y_best)))
        for (yy, xx, _dd) in keep[:3]:
            xs_out[int(yy)] = float(xx)
        moved_clusters += 1

    try:
        if moved_clusters:
            print(f"[gr-crest] moved {moved_clusters} crest rows (clusters={len(clusters)})")
    except Exception:
        pass

    return xs_out


def refine_to_stroke_centerline(mask, xs, threshold_ratio=0.5, window_size=None):
    """Refine trace to the centerline of the curve stroke width.
    
    Uses a half-maximum window (FWHM) to estimate stroke width and blends
    geometric and weighted centers for stability.
    """
    if mask is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size < 3:
        return xs
    
    h, w = mask.shape[:2]
    if h < 3 or w < 3:
        return xs
    
    prob = mask.astype(np.float32) / 255.0
    xs_ref = xs.copy()
    
    valid_rows = np.where(np.isfinite(xs_ref))[0]
    try:
        if window_size is not None:
            search_radius = max(1, int(window_size))
        else:
            search_radius = 15
    except Exception:
        search_radius = 15
    
    for y in valid_rows:
        x_prev = float(xs_ref[y])
        x_c = int(round(x_prev))
        if x_c < 0 or x_c >= w:
            continue
        
        row = prob[y]
        x0 = max(0, x_c - search_radius)
        x1 = min(w, x_c + search_radius + 1)
        window = row[x0:x1]
        if window.size == 0:
            continue
        
        max_val = window.max()
        if max_val < 0.05:
            continue
        
        peak_idx = int(np.argmax(window))
        thr = float(max_val) * float(threshold_ratio)
        if thr <= 0:
            continue

        above = window >= thr
        if not np.any(above):
            continue

        # Choose the contiguous above-threshold band that overlaps the current
        # x (or is nearest). Using argmax alone can lock onto a bright edge on
        # thick strokes.
        segs = []
        in_seg = False
        seg_start = 0
        for i in range(int(above.size)):
            if bool(above[i]) and not in_seg:
                in_seg = True
                seg_start = i
            elif (not bool(above[i])) and in_seg:
                segs.append((int(seg_start), int(i - 1)))
                in_seg = False
        if in_seg:
            segs.append((int(seg_start), int(above.size) - 1))

        x_rel = float(x_prev) - float(x0)
        chosen = None
        best_dist = None
        for (l, r) in segs:
            if l <= x_rel <= r:
                chosen = (l, r)
                best_dist = 0.0
                break
            if x_rel < l:
                dist = float(l) - x_rel
            else:
                dist = x_rel - float(r)
            if best_dist is None or dist < best_dist:
                chosen = (l, r)
                best_dist = dist
            elif best_dist is not None and abs(dist - best_dist) < 1e-6:
                # Tie-breaker: prefer the segment with higher local intensity
                try:
                    if chosen is not None:
                        l0, r0 = chosen
                        if float(window[l:r + 1].max()) > float(window[l0:r0 + 1].max()):
                            chosen = (l, r)
                except Exception:
                    pass

        if chosen is None:
            continue

        left_idx, right_idx = chosen

        stroke_center = (left_idx + right_idx) / 2.0

        # Optional weighted correction within the band. Keep it mild to prevent
        # a consistent pull toward stronger edges.
        final_center = float(stroke_center)
        try:
            stroke_slice = window[left_idx:right_idx + 1]
            if stroke_slice.size > 0:
                xs_local = np.arange(left_idx, right_idx + 1, dtype=np.float32)
                weights = np.power(stroke_slice.astype(np.float32), 1.5)
                wsum = float(weights.sum())
                if wsum > 1e-8:
                    weighted_center = float((xs_local * weights).sum() / wsum)
                    final_center = 0.75 * float(stroke_center) + 0.25 * weighted_center
        except Exception:
            final_center = float(stroke_center)

        x_new = float(x0 + final_center)

        # Clamp and blend for stability
        max_shift = max(1.5, float(search_radius) * 0.6)
        dx = x_new - x_prev
        if dx > max_shift:
            x_new = x_prev + max_shift
        elif dx < -max_shift:
            x_new = x_prev - max_shift

        alpha = 0.85
        xs_ref[y] = float((1.0 - alpha) * x_prev + alpha * x_new)
    
    return xs_ref


def guard_trace_outliers_rolling_median(xs, window=21, max_deviation=45.0):
    """NaN out trace points whose horizontal position is absurdly far from a
    rolling median, then linearly interpolate across them.

    This targets the "left-shooting spike" failure mode where the DP tracer
    briefly jumps to a grid rail or noise column far from the real curve and
    draws a visible horizontal line to the chart edge. The rolling median is
    robust to a few bad points, so genuine excursions (which move gradually
    over multiple rows) are preserved while isolated single-row jumps get
    discarded.

    Args:
        xs: 1D array of x positions (may contain NaN).
        window: Rolling window length (odd). Larger = smoother reference.
        max_deviation: Maximum allowed |xs - rolling_median| in pixels.

    Returns:
        xs with outliers interpolated away.
    """
    if xs is None or not hasattr(xs, "size") or xs.size < 5:
        return xs
    try:
        s = pd.Series(xs.astype(np.float32))
    except Exception:
        return xs
    win = max(5, int(window) | 1)  # force odd
    ref = s.rolling(win, min_periods=3, center=True).median()
    valid = s.notna() & ref.notna()
    if not valid.any():
        return xs
    deviation = (s - ref).abs()
    outliers = valid & (deviation > float(max_deviation))
    if not outliers.any():
        return xs
    s[outliers] = np.nan
    s = s.interpolate(method="linear", limit_direction="both", limit=50)
    return s.to_numpy(dtype=np.float32)


def guard_trace_velocity(xs, max_dx=6.0):
    """Cap row-to-row horizontal displacement and interpolate across spikes.

    Micro-crests created by snap_black_trace_to_wide_darkest are typically
    1-3 rows tall but jump 10-20 px horizontally. Real geological excursions
    move gradually (e.g. 10 px over 5 rows = 2 px/row). Capping |dx/dy|
    removes the micro-crests without clipping legitimate peaks.
    """
    if xs is None or not hasattr(xs, "size") or xs.size < 5:
        return xs
    try:
        s = pd.Series(xs.astype(np.float32))
    except Exception:
        return xs
    dx = s.diff().abs()
    spikes = dx > float(max_dx)
    if not spikes.any():
        return xs
    s[spikes] = np.nan
    s = s.interpolate(method="linear", limit_direction="both", limit=10)
    return s.to_numpy(dtype=np.float32)


def snap_black_trace_to_wide_darkest(roi_bgr, xs, search_radius=55, min_darkness_gain=0.12, neighbor_consistency=25.0):
    """For each row, search a wide window around the current trace point for
    a darker pixel and snap to it if it is clearly darker AND the shift stays
    consistent with the local neighbors.

    Vectorized implementation: uses a (h, 2r+1) gather so we avoid a Python
    loop over every scan line, which previously caused Railway worker
    timeouts (503) on tall TIFFs.
    """
    if roi_bgr is None or xs is None or not hasattr(xs, "size") or xs.size < 3:
        return xs
    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return xs
    h, w = gray.shape[:2]
    if h < 3 or w < 3 or xs.size != h:
        return xs

    xs_out = xs.copy().astype(np.float32)
    dark = (255.0 - gray.astype(np.float32)) / 255.0
    r = max(6, int(search_radius))

    xi = np.round(xs_out).astype(np.int32)
    finite_mask = np.isfinite(xs_out)
    xi = np.where(finite_mask, xi, 0)
    xi = np.clip(xi, 0, w - 1)

    # Build a (h, 2r+1) column index matrix; out-of-bounds clamped to edge.
    offsets = np.arange(-r, r + 1, dtype=np.int32)
    cols = np.clip(xi[:, None] + offsets[None, :], 0, w - 1)  # (h, 2r+1)
    rows = np.arange(h, dtype=np.int32)[:, None]
    windows = dark[rows, cols]  # (h, 2r+1) darkness values

    # Mask columns that were clamped (to avoid picking the edge as a "peak")
    raw_cols = xi[:, None] + offsets[None, :]
    in_bounds = (raw_cols >= 0) & (raw_cols < w)
    windows = np.where(in_bounds, windows, -1.0)

    peak_idx = np.argmax(windows, axis=1)  # (h,)
    peak_col = xi + offsets[peak_idx]
    peak_val = windows[rows[:, 0], peak_idx]
    center_val = dark[rows[:, 0], xi]
    gain = peak_val - center_val

    # Neighbor consistency via rolling median (window ~11 rows)
    try:
        s_med = pd.Series(xs_out).rolling(11, min_periods=3, center=True).median().to_numpy()
    except Exception:
        s_med = xs_out.copy()

    accept = (
        finite_mask
        & (gain >= float(min_darkness_gain))
        & (np.abs(peak_col.astype(np.float32) - s_med) <= float(neighbor_consistency))
    )
    xs_out[accept] = peak_col[accept].astype(np.float32)
    return xs_out


def refine_black_trace_to_continuous_line(
    roi_bgr,
    xs,
    search_radius=18,
    guide_window=31,
    vertical_window=13,
    min_line_score=0.07,
    min_score_gain=0.04,
    trend_pull_pixels=3.0,
    distance_weight=0.025,
):
    """Second pass for black traces: prefer continuous line support over row crests.

    Horizontal grid bars and filled text can be the darkest thing in a single
    row. This pass scores nearby candidates by residual curve ink that persists
    across neighboring rows, so the first trace acts as a guide but one-row
    dark shelves do not pull the output sideways.
    """
    if roi_bgr is None or xs is None or not hasattr(xs, "size") or xs.size < 3:
        return xs

    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return xs

    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return xs

    xs_ref = xs.copy().astype(np.float32)
    n = min(h, xs_ref.size)
    valid = np.isfinite(xs_ref[:n])
    if not np.any(valid):
        return xs_ref

    try:
        filled = pd.Series(xs_ref[:n]).interpolate(method="linear", limit_direction="both").ffill().bfill()
        if filled.isna().any():
            return xs_ref
        guide_win = max(9, int(guide_window) | 1)
        guide_win = min(guide_win, n if n % 2 else n - 1)
        if guide_win >= 5:
            guide_series = filled.rolling(
                guide_win,
                min_periods=max(3, min(9, guide_win // 3)),
                center=True,
            ).median().bfill().ffill()
        else:
            guide_series = filled
        raw_guide = filled.to_numpy(dtype=np.float32)
        guide = guide_series.to_numpy(dtype=np.float32)
    except Exception:
        return xs_ref

    try:
        support_gray, residual_score, grid_score, _ = build_black_prescan_grid_removed(gray)
        if residual_score is None or residual_score.shape[:2] != (h, w):
            return xs_ref
        if grid_score is None or grid_score.shape[:2] != (h, w):
            grid_score = np.zeros((h, w), dtype=np.float32)
        if support_gray is None or support_gray.shape[:2] != (h, w):
            support_gray = gray

        k_y = max(5, int(vertical_window) | 1)
        k_y = min(k_y, h if h % 2 else h - 1)
        if k_y < 3:
            return xs_ref

        residual = np.clip(residual_score.astype(np.float32), 0.0, 1.0)
        grid = np.clip(grid_score.astype(np.float32), 0.0, 1.0)
        dark = (255.0 - support_gray.astype(np.float32)) / 255.0

        vertical_support = cv2.blur(residual, (3, k_y))
        local_residual = cv2.GaussianBlur(residual, (3, 3), 0)
        grid_penalty = cv2.blur(grid, (3, k_y))
        line_score = (
            0.62 * vertical_support
            + 0.28 * local_residual
            + 0.10 * dark
            - 0.42 * grid_penalty
        )
        line_score = np.clip(line_score, 0.0, 1.0).astype(np.float32)
    except Exception:
        return xs_ref

    r = max(4, int(search_radius))
    offsets = np.arange(-r, r + 1, dtype=np.int32)
    rows = np.arange(n, dtype=np.int32)[:, None]
    xi = np.clip(np.round(guide).astype(np.int32), 0, w - 1)
    raw_cols = xi[:, None] + offsets[None, :]
    in_bounds = (raw_cols >= 0) & (raw_cols < w)
    cols = np.clip(raw_cols, 0, w - 1)

    candidate_scores = line_score[rows, cols]
    distance_penalty = np.abs(offsets.astype(np.float32))[None, :] * float(distance_weight)
    candidate_scores = np.where(in_bounds, candidate_scores - distance_penalty, -1.0)

    best_idx = np.argmax(candidate_scores, axis=1)
    best_cols = cols[np.arange(n), best_idx]
    best_scores = candidate_scores[np.arange(n), best_idx]
    raw_xi = np.clip(np.round(raw_guide).astype(np.int32), 0, w - 1)
    current_scores = line_score[np.arange(n), raw_xi]
    trend_delta = np.abs(raw_guide - guide)

    accept = (
        valid
        & (best_scores >= float(min_line_score))
        & (
            (best_scores >= (current_scores + float(min_score_gain)))
            | (current_scores < float(min_line_score))
            | (trend_delta >= float(trend_pull_pixels))
        )
    )

    xs_out = xs_ref.copy()
    for y in np.where(accept)[0]:
        best_col = int(best_cols[y])
        x0 = max(0, best_col - 2)
        x1 = min(w, best_col + 3)
        weights = np.clip(line_score[y, x0:x1], 0.0, 1.0) ** 2
        if float(weights.sum()) > 1e-8:
            xs_local = np.arange(x0, x1, dtype=np.float32)
            xs_out[y] = float((xs_local * weights).sum() / weights.sum())
        else:
            xs_out[y] = float(best_col)

    return xs_out


def refine_black_trace_to_dark_run_center(
    roi_bgr,
    xs,
    threshold_block=21,
    threshold_c=4,
    search_radius=28,
    max_shift=12.0,
    blend=0.95,
    hot_side=None,
    curve_type=None,
):
    """Recenter a black trace onto the visible dark stroke body.

    Unlike the probability-map centerline pass, this works from the ROI's
    grayscale ink directly, which better matches thick filled black curves.
    When the stroke gets wide, bias the target toward the chart-reading side
    so black curves hit visible crest tips instead of sitting on the inner half
    of the printed stroke.
    """
    if roi_bgr is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size < 3:
        return xs

    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return xs

    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return xs

    try:
        if detect_if_black_and_white_log(roi_bgr):
            gray = suppress_grid_hough(gray)
            gray = remove_grid_lines_aggressive(gray, aggressive=False)
    except Exception:
        pass

    try:
        dark_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, int(threshold_block), int(threshold_c)
        )
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), 1
        )
    except Exception:
        return xs

    dark_score = (255.0 - gray.astype(np.float32)) / 255.0
    residual_support = None
    grid_support = None
    try:
        residual_score, grid_score = compute_black_curve_residual(gray)
        if residual_score is not None and residual_score.shape[:2] == (h, w):
            residual_support = cv2.GaussianBlur(
                np.clip(residual_score, 0.0, 1.0).astype(np.float32),
                (5, 5),
                0,
            )
        if grid_score is not None and grid_score.shape[:2] == (h, w):
            grid_support = cv2.GaussianBlur(
                np.clip(grid_score, 0.0, 1.0).astype(np.float32),
                (5, 5),
                0,
            )
    except Exception:
        residual_support = None
        grid_support = None

    xs_ref = xs.copy().astype(np.float32)
    radius = max(4, int(search_radius))
    base_max_shift = max(1.0, float(max_shift))
    blend = float(np.clip(blend, 0.0, 1.0))
    curve_type_upper = str(curve_type or "").upper()
    # Keep black traces centered on the visible stroke body. The earlier
    # hot-side bias helped crest picking in some cases, but on real scans it
    # is the main reason the trace rides the stroke edge and nearby grid rails.
    use_hot_bias = False
    prev_target = None

    for y in range(min(h, xs_ref.size)):
        x_prev = xs_ref[y]
        if not np.isfinite(x_prev):
            prev_target = None
            continue

        x_center = int(round(float(x_prev)))
        if x_center < 0 or x_center >= w:
            prev_target = None
            continue

        x0 = max(0, x_center - radius)
        x1 = min(w, x_center + radius + 1)
        row_mask = dark_mask[y, x0:x1] > 0
        if row_mask.size == 0 or not np.any(row_mask):
            prev_target = None
            continue

        segs = []
        in_seg = False
        seg_start = 0
        for i in range(int(row_mask.size)):
            if bool(row_mask[i]) and not in_seg:
                in_seg = True
                seg_start = i
            elif (not bool(row_mask[i])) and in_seg:
                segs.append((int(seg_start), int(i - 1)))
                in_seg = False
        if in_seg:
            segs.append((int(seg_start), int(row_mask.size) - 1))
        if not segs:
            continue

        x_rel = float(x_prev) - float(x0)
        chosen = None
        best = None
        chosen_info = None
        for l, r in segs:
            if l <= x_rel <= r:
                dist = 0.0
            elif x_rel < l:
                dist = float(l) - x_rel
            else:
                dist = x_rel - float(r)

            width = float(r - l + 1)
            seg_dark = dark_score[y, x0 + l:x0 + r + 1]
            darkness = float(seg_dark.mean()) if seg_dark.size > 0 else 0.0
            res_peak = 0.0
            grid_mean = 0.0
            if residual_support is not None:
                seg_res = residual_support[y, x0 + l:x0 + r + 1]
                res_peak = float(seg_res.max()) if seg_res.size > 0 else 0.0
            if grid_support is not None:
                seg_grid = grid_support[y, x0 + l:x0 + r + 1]
                grid_mean = float(seg_grid.mean()) if seg_grid.size > 0 else 0.0
            score = (dist, -width, -darkness)
            if best is None or score < best:
                best = score
                chosen = (l, r)
                chosen_info = (res_peak, grid_mean)

        if chosen is None:
            prev_target = None
            continue

        l, r = chosen
        if (
            curve_type_upper != "GR"
            and prev_target is not None
            and chosen_info is not None
        ):
            chosen_res_peak, chosen_grid_mean = chosen_info
            # If the nearest dark run looks overwhelmingly grid-like, do not
            # reacquire on this row at all. For RHOB/DT-type curves, holding
            # the incoming trace for one row is more stable than jumping onto
            # a likely rail and then trying to recover a few rows later.
            if chosen_grid_mean > 0.90 and chosen_res_peak < 0.08:
                continue

        seg_dark = dark_score[y, x0 + l:x0 + r + 1]
        xs_local = np.arange(x0 + l, x0 + r + 1, dtype=np.float32)
        weights = np.power(np.clip(seg_dark, 0.0, 1.0), 1.8)
        if float(weights.sum()) > 1e-8:
            x_center = float((xs_local * weights).sum() / weights.sum())
        else:
            x_center = float(x0 + (l + r) * 0.5)

        x_target = x_center
        local_max_shift = base_max_shift
        local_blend = blend

        if use_hot_bias:
            run_width = float(r - l + 1)
            hot_edge = float(x0 + r) if hot_side == "right" else float(x0 + l)
            if curve_type_upper == "GR":
                # GR should preserve visible right-side crest tips, but still
                # stay anchored to the same dark run.
                width_t = float(np.clip((run_width - 3.0) / 14.0, 0.0, 1.0))
                bias = 0.58 + 0.38 * width_t
                local_max_shift = max(local_max_shift, 22.0)
                local_blend = max(local_blend, 0.98)
            else:
                # RHOB/DT-like strokes are slower and thicker; bias more gently
                # but still move well past the geometric center on wide rows.
                width_t = float(np.clip((run_width - 3.0) / 18.0, 0.0, 1.0))
                bias = 0.48 + 0.42 * width_t
                local_max_shift = max(local_max_shift, 22.0)
                local_blend = max(local_blend, 0.94)
            x_target = float(x_center + (hot_edge - x_center) * bias)

        dx = x_target - float(x_prev)
        dx = max(-local_max_shift, min(local_max_shift, dx))
        xs_ref[y] = float((1.0 - local_blend) * float(x_prev) + local_blend * (float(x_prev) + dx))
        prev_target = float(x_target)

    return xs_ref


def recenter_black_trace_post_dp(roi_bgr, xs):
    """
    A purely mathematical post-processing step to center an edge-hugging black trace.
    It looks at the dark ink immediately around the existing trace and shifts the 
    point to the midpoint of that contiguous ink blob.
    """
    if roi_bgr is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size < 3:
        return xs

    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        # Simple threshold for ink
        _, ink_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    except Exception:
        return xs

    h, w = ink_mask.shape
    xs_centered = xs.copy()
    
    for y in range(min(h, xs.size)):
        x_current = xs[y]
        if not np.isfinite(x_current):
            continue
            
        ix = int(round(float(x_current)))
        if ix < 0 or ix >= w:
            continue
            
        # Only center if the current point is actually on ink
        if ink_mask[y, ix] == 0:
            # Search nearby for ink (up to 15px since thick traces can be quite far from the center)
            found_ink = False
            for offset in range(1, 16):
                if ix - offset >= 0 and ink_mask[y, ix - offset] > 0:
                    ix = ix - offset
                    found_ink = True
                    break
                if ix + offset < w and ink_mask[y, ix + offset] > 0:
                    ix = ix + offset
                    found_ink = True
                    break
            if not found_ink:
                continue
                
        # We are on ink. Find the left and right boundaries of this continuous ink blob.
        left_bound = ix
        while left_bound > 0 and ink_mask[y, left_bound - 1] > 0:
            left_bound -= 1
            
        right_bound = ix
        while right_bound < w - 1 and ink_mask[y, right_bound + 1] > 0:
            right_bound += 1
            
        blob_width = right_bound - left_bound + 1
        
        # If the blob is reasonably thick but not obviously a massive grid intersection, center it
        if 3 <= blob_width <= 40:
            # Shift towards the center, but don't move more than 15 pixels to avoid wild jumps
            target_center = float(left_bound + right_bound) / 2.0
            max_shift = 15.0
            dx = target_center - x_current
            dx = max(-max_shift, min(max_shift, dx))
            xs_centered[y] = x_current + dx
        elif blob_width > 40:
            # It's a grid intersection. Don't center on the whole track.
            # Mark as NaN so we can interpolate through it
            xs_centered[y] = np.nan
            
    # Apply interpolation to fill the grid intersection gaps
    try:
        import pandas as pd
        s = pd.Series(xs_centered)
        s = s.interpolate(method='linear', limit_direction='both', limit=20)
        xs_centered = s.to_numpy()
        
        # Apply a rolling median to remove jagged 1-pixel snaps, then a light mean
        s2 = pd.Series(xs_centered)
        s2 = s2.rolling(window=5, center=True, min_periods=1).median()
        s2 = s2.rolling(window=3, center=True, min_periods=1).mean()
        xs_centered = s2.to_numpy()
    except Exception:
        pass
        
    return xs_centered


def suppress_black_grid_lock_runs(roi_bgr, xs, curve_type=None):
    """Remove suspicious black-mode lock-ons to grid-like columns.

    If the trace sits on the same x-column for several rows while the residual
    model says that location is mostly grid and not curve ink, blank that short
    span and let interpolation reconnect the surrounding curve.

    Also catch the common "rail jitter" case where the trace wiggles a couple of
    pixels around one vertical grid column for several rows. That pattern looks
    continuous to a simple dx<=1 detector, but it is still visually locked to a
    grid rail instead of following the printed curve body.
    """
    if roi_bgr is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size < 3:
        return xs

    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return xs

    try:
        if detect_if_black_and_white_log(roi_bgr):
            gray = suppress_grid_hough(gray)
            gray = remove_grid_lines_aggressive(gray, aggressive=False)
    except Exception:
        pass

    try:
        residual_score, grid_score = compute_black_curve_residual(gray)
    except Exception:
        return xs

    if residual_score is None or grid_score is None:
        return xs
    if residual_score.shape[:2] != gray.shape[:2] or grid_score.shape[:2] != gray.shape[:2]:
        return xs

    xs_ref = xs.copy().astype(np.float32)
    h, w = gray.shape[:2]
    valid = np.isfinite(xs_ref)
    if not np.any(valid):
        return xs_ref

    curve_type_upper = str(curve_type or "").upper()
    residual_thr = 0.10 if curve_type_upper == "GR" else 0.08
    min_len = 8 if curve_type_upper == "GR" else 6
    bad = np.zeros(xs_ref.size, dtype=bool)
    rvals = np.full(xs_ref.size, np.nan, dtype=np.float32)
    gvals = np.full(xs_ref.size, np.nan, dtype=np.float32)

    for y in range(min(h, xs_ref.size)):
        if not valid[y]:
            continue
        ix = int(np.clip(round(float(xs_ref[y])), 0, w - 1))
        rvals[y] = float(residual_score[y, ix])
        gvals[y] = float(grid_score[y, ix])

    for y in range(1, min(h, xs_ref.size) - 1):
        if not valid[y - 1] or not valid[y] or not valid[y + 1]:
            continue
        rv = float(rvals[y])
        gv = float(gvals[y])
        dx_prev = abs(float(xs_ref[y] - xs_ref[y - 1]))
        dx_next = abs(float(xs_ref[y + 1] - xs_ref[y]))
        if rv < residual_thr and gv > 0.90 and dx_prev <= 1.0 and dx_next <= 1.0:
            bad[y] = True

    # A rail lock often jitters by 1-3 px while staying pinned to the same
    # vertical grid column. Mark those short near-vertical windows too.
    band_radius = 3 if curve_type_upper == "GR" else 2
    max_band_span = 3.5 if curve_type_upper == "GR" else 3.0
    mean_residual_thr = 0.10 if curve_type_upper == "GR" else 0.08
    mean_grid_thr = 0.90 if curve_type_upper == "GR" else 0.88
    for y in range(band_radius, min(h, xs_ref.size) - band_radius):
        sl = slice(y - band_radius, y + band_radius + 1)
        if not np.all(valid[sl]):
            continue
        window_x = xs_ref[sl]
        if float(np.nanmax(window_x) - np.nanmin(window_x)) > max_band_span:
            continue
        if (
            float(np.nanmean(rvals[sl])) < mean_residual_thr
            and float(np.nanmean(gvals[sl])) > mean_grid_thr
        ):
            bad[sl] = True

    spans = []
    in_span = False
    seg_start = 0
    for i, flag in enumerate(bad):
        if bool(flag) and not in_span:
            in_span = True
            seg_start = i
        elif (not bool(flag)) and in_span:
            spans.append((int(seg_start), int(i - 1)))
            in_span = False
    if in_span:
        spans.append((int(seg_start), int(xs_ref.size - 1)))

    spans = [(s, e) for (s, e) in spans if (e - s + 1) >= min_len]
    if not spans:
        return xs_ref

    for s, e in spans:
        xs_ref[s:e + 1] = np.nan

    try:
        xs_ref = pd.Series(xs_ref).interpolate(
            method='linear',
            limit_direction='both',
            limit=max(25, int(xs_ref.size * 0.03)),
            limit_area=None,
        ).to_numpy(dtype=np.float32)
    except Exception:
        pass

    return xs_ref


def refine_black_trace_to_hot_side_crests(roi_bgr, xs, hot_side=None, threshold_block=21, threshold_c=4, search_radius=22, min_run_width=6.0, max_shift=14.0, blend=0.9, curve_type=None):
    """Snap black traces toward visible crest tips on the chart-reading side.

    This is conservative: it only moves rows where the local black stroke is
    wide enough to indicate a crest/shoulder and the current trace is still
    sitting on the wrong side of that run.
    """
    if roi_bgr is None or xs is None:
        return xs
    if hot_side not in ("left", "right"):
        return xs
    if not hasattr(xs, "size") or xs.size < 3:
        return xs

    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return xs

    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return xs

    curve_type_upper = str(curve_type or "").upper()
    is_gr_curve = curve_type_upper == "GR"

    try:
        if detect_if_black_and_white_log(roi_bgr):
            gray = suppress_grid_hough(gray)
            gray = remove_grid_lines_aggressive(gray, aggressive=False)
    except Exception:
        pass

    residual_support = None
    residual_binary = None
    edge_mask = None
    try:
        residual_score, grid_score = compute_black_curve_residual(gray)
        if residual_score is not None and residual_score.shape[:2] == (h, w):
            residual_support = cv2.GaussianBlur(
                residual_score.astype(np.float32), (5, 5), 0
            )
            residual_support = np.clip(residual_support, 0.0, 1.0)
            residual_binary = (residual_support >= (0.18 if is_gr_curve else 0.12)).astype(np.uint8) * 255
            residual_binary = cv2.morphologyEx(
                residual_binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5 if is_gr_curve else 9, 1)),
                1,
            )
            residual_binary = cv2.morphologyEx(
                residual_binary,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)),
                1,
            )
    except Exception:
        residual_support = None
        residual_binary = None
        edge_mask = None

    try:
        dark_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, int(threshold_block), int(threshold_c)
        )
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), 1
        )
    except Exception:
        return xs

    try:
        if grid_score is not None and grid_score.shape[:2] == (h, w):
            edge_mask = ((dark_mask > 0) & (grid_score < 0.85)).astype(np.uint8) * 255
            edge_mask = cv2.morphologyEx(
                edge_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),
                1,
            )
            edge_mask = cv2.morphologyEx(
                edge_mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)),
                1,
            )
    except Exception:
        edge_mask = None

    xs_ref = xs.copy().astype(np.float32)
    radius = max(4, int(search_radius))
    max_shift = max(1.0, float(max_shift))
    min_run_width = max(1.0, float(min_run_width))
    blend = float(np.clip(blend, 0.0, 1.0))
    if not is_gr_curve:
        max_shift = max(max_shift, 22.0)
        blend = max(blend, 0.98)
    back_radius = max(6 if is_gr_curve else 10, int(round(radius * 0.35)))
    hot_radius = max(radius + (8 if is_gr_curve else 28), 28 if is_gr_curve else 55)
    residual_min_strength = 0.18 if is_gr_curve else 0.14
    residual_peak_frac = 0.72 if is_gr_curve else 0.65
    min_extension = 5.0 if is_gr_curve else 4.0

    def _collect_segments(mask_1d):
        segs = []
        in_seg = False
        seg_start = 0
        for i in range(int(mask_1d.size)):
            if bool(mask_1d[i]) and not in_seg:
                in_seg = True
                seg_start = i
            elif (not bool(mask_1d[i])) and in_seg:
                segs.append((int(seg_start), int(i - 1)))
                in_seg = False
        if in_seg:
            segs.append((int(seg_start), int(mask_1d.size) - 1))
        return segs

    def _merge_segments(segs, max_gap=0):
        if not segs:
            return []
        merged = [list(segs[0])]
        for l, r in segs[1:]:
            if int(l) - int(merged[-1][1]) - 1 <= int(max_gap):
                merged[-1][1] = int(r)
            else:
                merged.append([int(l), int(r)])
        return [(int(l), int(r)) for (l, r) in merged]

    prev_hot_x = None

    for y in range(min(h, xs_ref.size)):
        x_prev = xs_ref[y]
        if not np.isfinite(x_prev):
            prev_hot_x = None
            continue

        x_center = int(round(float(x_prev)))
        if x_center < 0 or x_center >= w:
            prev_hot_x = None
            continue

        moved = False

        if not is_gr_curve and edge_mask is not None:
            anchor_x = float(x_prev)
            if prev_hot_x is not None and np.isfinite(prev_hot_x):
                anchor_x = float(prev_hot_x)
            anchor_center = int(round(anchor_x))
            if anchor_center < 0 or anchor_center >= w:
                anchor_center = x_center
            if hot_side == "right":
                x0 = max(0, anchor_center - max(back_radius, 14))
                x1 = min(w, anchor_center + max(hot_radius, 64) + 1)
            else:
                x0 = max(0, anchor_center - max(hot_radius, 64))
                x1 = min(w, anchor_center + max(back_radius, 14) + 1)
            row_binary = edge_mask[y, x0:x1] > 0
            if row_binary.size >= 4 and np.any(row_binary):
                segs = []
                for l, r in _collect_segments(row_binary):
                    if float(r - l + 1) >= max(3.0, min_run_width - 1.0):
                        segs.append((l, r))
                segs = _merge_segments(segs, max_gap=5)
                if segs:
                    x_rel = anchor_x - float(x0)
                    chosen = None
                    min_width = max(8.0, min_run_width + 2.0)
                    min_peak = 0.22
                    max_gap = 20.0
                    if hot_side == "right":
                        cand_iter = reversed(segs)
                    else:
                        cand_iter = iter(segs)

                    for l, r in cand_iter:
                        width = float(r - l + 1)
                        if width < min_width:
                            continue
                        peak = 0.0
                        if residual_support is not None:
                            seg = residual_support[y, x0 + l:x0 + r + 1]
                            peak = float(seg.max()) if seg.size > 0 else 0.0
                        if peak < min_peak:
                            continue
                        if hot_side == "right":
                            gap = max(0.0, float(l) - x_rel)
                        else:
                            gap = max(0.0, x_rel - float(r))
                        if gap <= max_gap:
                            chosen = (l, r)
                            break

                    if chosen is None:
                        best = None
                        for l, r in segs:
                            if hot_side == "right":
                                hot_edge = float(x0 + r)
                                dist = 0.0 if l <= x_rel <= r else (float(l) - x_rel if x_rel < l else x_rel - float(r))
                            else:
                                hot_edge = float(x0 + l)
                                dist = 0.0 if l <= x_rel <= r else (x_rel - float(r) if x_rel > r else float(l) - x_rel)
                            width = float(r - l + 1)
                            score = (max(0.0, dist), -width, abs(hot_edge - float(x_prev)))
                            if best is None or score < best:
                                best = score
                                chosen = (l, r)
                    if chosen is not None:
                        l, r = chosen
                        x_target = float(x0 + r) if hot_side == "right" else float(x0 + l)
                        dx = x_target - float(x_prev)
                        dx = max(-18.0, min(18.0, dx))
                        if abs(dx) >= 0.5:
                            xs_ref[y] = float((1.0 - blend) * float(x_prev) + blend * (float(x_prev) + dx))
                            moved = True

        if moved:
            prev_hot_x = float(xs_ref[y])
            continue

        if not is_gr_curve:
            prev_hot_x = float(xs_ref[y])
            continue

        x0 = max(0, x_center - radius)
        x1 = min(w, x_center + radius + 1)
        row_mask = dark_mask[y, x0:x1] > 0
        if row_mask.size == 0 or not np.any(row_mask):
            continue

        segs = _collect_segments(row_mask)
        if not segs:
            continue

        x_rel = float(x_prev) - float(x0)
        chosen = None
        best = None
        for l, r in segs:
            if l <= x_rel <= r:
                dist = 0.0
            elif x_rel < l:
                dist = float(l) - x_rel
            else:
                dist = x_rel - float(r)
            width = float(r - l + 1)
            score = (dist, -width)
            if best is None or score < best:
                best = score
                chosen = (l, r)

        if chosen is None:
            continue

        l, r = chosen
        run_width = float(r - l)
        if run_width < min_run_width:
            continue

        frac = (float(x_prev) - float(x0 + l)) / max(1.0, run_width)
        if hot_side == "right":
            keep_frac = 0.58 if is_gr_curve else 0.66
            if frac >= keep_frac:
                continue
            x_target = float(x0 + r)
        else:
            keep_frac = 0.42 if is_gr_curve else 0.34
            if frac <= keep_frac:
                continue
            x_target = float(x0 + l)

        dx = x_target - float(x_prev)
        dx = max(-max_shift, min(max_shift, dx))
        xs_ref[y] = float((1.0 - blend) * float(x_prev) + blend * (float(x_prev) + dx))
        prev_hot_x = float(xs_ref[y])

    return xs_ref


def remove_outliers_and_smooth(xs, window=5, outlier_threshold=3.0):
    """Remove isolated spikes and smooth the curve.
    
    Args:
        xs: Array of x-coordinates with possible NaNs
        window: Window size for median smoothing
        outlier_threshold: Number of std deviations for outlier detection
    
    Returns:
        Smoothed array with outliers removed
    """
    if xs is None or xs.size < 3:
        return xs
    
    # Convert to pandas for easier handling
    s = pd.Series(xs)
    
    # Remove outliers: if a point differs from both neighbors by > threshold * std
    valid = ~s.isna()
    if valid.sum() > 3:
        # Compute rolling std/mean for outlier detection. Pandas requires
        # min_periods <= window, so ensure the rolling window is at least 2
        # even when the smoothing window is 1 (our "almost no smoothing" case).
        win_outlier = max(2, int(window))
        rolling_std = s[valid].rolling(win_outlier, min_periods=2, center=True).std()
        rolling_mean = s[valid].rolling(win_outlier, min_periods=2, center=True).mean()
        
        for i in range(1, len(s) - 1):
            if valid.iloc[i]:
                neighbors = [s.iloc[i - 1], s.iloc[i + 1]]
                neighbors_valid = [x for x in neighbors if not pd.isna(x)]
                if len(neighbors_valid) >= 2:
                    mean_neighbor = np.mean(neighbors_valid)
                    std_val = rolling_std.iloc[i] if i < len(rolling_std) and not pd.isna(rolling_std.iloc[i]) else 1.0
                    if abs(s.iloc[i] - mean_neighbor) > outlier_threshold * max(std_val, 1.0):
                        s.iloc[i] = np.nan
    
    # Smooth with median filter
    if window % 2 == 0:
        window += 1
    if window > 1:
        s = s.rolling(window, min_periods=1, center=True).median()
    
    # Interpolate remaining gaps
    s = s.interpolate(limit_direction="both", limit=50)
    
    return s.to_numpy(dtype=np.float32)


def pick_curve_x_per_row(mask, min_run=2):
    h, w = mask.shape
    xs = np.full(h, np.nan, dtype=np.float32)
    for y in range(h):
        idx = np.flatnonzero(mask[y, :] > 0)
        if idx.size >= min_run:
            xs[y] = float(np.median(idx))
    return xs

def trace_curve_direct_centerline(mask, threshold=10):
    """
    Find the exact centerline by finding the peak intensity in each row.
    Uses a weighted centroid around the peak for sub-pixel accuracy.
    
    Args:
        mask: Grayscale probability map (0-255)
        threshold: Minimum intensity to consider as ink
    
    Returns:
        Array of x-coordinates, one per row
    """
    h, w = mask.shape[:2]
    xs = np.full(h, np.nan, dtype=np.float32)
    
    # For each row, find the peak intensity
    for y in range(h):
        row = mask[y, :].astype(np.float32)
        
        # Find pixels above threshold
        valid_mask = row >= threshold
        
        if np.any(valid_mask):
            # Find the maximum intensity in this row
            max_intensity = row[valid_mask].max()
            
            if max_intensity > threshold:
                # Find all pixels near the peak (within 5% of max for good balance)
                peak_threshold = max_intensity * 0.95
                peak_mask = row >= peak_threshold
                peak_indices = np.where(peak_mask)[0]
                
                if len(peak_indices) > 0:
                    # Use intensity-weighted centroid with cubic weighting
                    # This gives maximum emphasis to the absolute peak
                    weights = row[peak_indices]
                    # Apply cubic emphasis on the peak for ultra-precise centering
                    gaussian_weights = weights ** 3  # Cubic emphasizes peak even more
                    xs[y] = float(np.sum(peak_indices * gaussian_weights) / np.sum(gaussian_weights))
                else:
                    # Fallback: weighted centroid of all valid pixels
                    valid_indices = np.where(valid_mask)[0]
                    weights = row[valid_indices]
                    xs[y] = float(np.sum(valid_indices * weights) / np.sum(weights))
    
    return xs

def smooth_nanmedian(series, window):
    s = pd.Series(series)
    if window % 2 == 0:
        window += 1
    if window > 1:
        s = s.rolling(window, min_periods=1, center=True).median()
    return s.interpolate(limit_direction="both", limit=50).to_numpy(dtype=np.float32)

def compute_depth_vector(nrows, top_depth, bottom_depth):
    ys = np.arange(nrows, dtype=np.float32)
    return top_depth + (ys / max(1, nrows-1)) * (bottom_depth - top_depth)

def write_las_simple(depth, curve_data, depth_unit="FT", header_metadata=None):
    """Generate LAS 1.2-style file compatible with QuickSyn"""
    null_val = -999.25
    unit_token = "F" if depth_unit.upper().startswith("F") else depth_unit.upper()
    eol = "\r\n"

    lines = []

    # Version section (LAS 1.2 style)
    lines.append("~VERSION INFORMATION" + eol)
    lines.append(" VERS.                 1.20:   CWLS LOG ASCII STANDARD -VERSION 1.20" + eol)
    lines.append(" WRAP.                   NO:   ONE LINE PER DEPTH STEP" + eol)

    # Well information section
    lines.append("~WELL INFORMATION BLOCK" + eol)
    lines.append("#MNEM.UNIT       DATA TYPE    INFORMATION" + eol)
    lines.append("#---------     -----------    ---------------------------" + eol)
    lines.append(f" STRT.{unit_token}               {depth[0]:.4f}:" + eol)
    lines.append(f" STOP.{unit_token}               {depth[-1]:.4f}:" + eol)
    step = float(depth[1] - depth[0]) if depth.size > 1 else 0.0
    lines.append(f" STEP.{unit_token}               {step:.4f}:" + eol)
    lines.append(f" NULL.               {null_val:.4f}:" + eol)

    md = header_metadata if isinstance(header_metadata, dict) else {}
    comp = (md.get('comp') or md.get('company') or '').strip() if isinstance(md.get('comp') or md.get('company') or '', str) else ''
    well = (md.get('well') or '').strip() if isinstance(md.get('well') or '', str) else ''
    fld = (md.get('fld') or md.get('field') or '').strip() if isinstance(md.get('fld') or md.get('field') or '', str) else ''
    loc = (md.get('loc') or md.get('location') or '').strip() if isinstance(md.get('loc') or md.get('location') or '', str) else ''
    county = (md.get('county') or '').strip() if isinstance(md.get('county') or '', str) else ''
    state = (md.get('state') or '').strip() if isinstance(md.get('state') or '', str) else ''
    prov = (md.get('prov') or md.get('province') or '').strip() if isinstance(md.get('prov') or md.get('province') or '', str) else ''
    srvc = (md.get('srvc') or md.get('service') or md.get('service_company') or '').strip() if isinstance(md.get('srvc') or md.get('service') or md.get('service_company') or '', str) else ''
    date = (md.get('date') or '').strip() if isinstance(md.get('date') or '', str) else ''
    api = (md.get('api') or '').strip() if isinstance(md.get('api') or '', str) else ''
    uwi = (md.get('uwi') or '').strip() if isinstance(md.get('uwi') or '', str) else ''

    if comp:
        lines.append(f" COMP.       {comp}:" + eol)
    lines.append(f" WELL.       {(well or 'DIGITIZED_LOG')}:" + eol)
    if fld:
        lines.append(f" FLD .       {fld}:" + eol)
    if loc:
        lines.append(f" LOC .       {loc}:" + eol)
    if county:
        lines.append(f" CNTY.       {county}:" + eol)
    if state:
        lines.append(f" STAT.       {state}:" + eol)
    if prov:
        lines.append(f" PROV.       {prov}:" + eol)
    if srvc:
        lines.append(f" SRVC.       {srvc}:" + eol)
    if date:
        lines.append(f" DATE.       {date}:" + eol)
    if api:
        lines.append(f" API .       {api}:" + eol)
    if uwi:
        lines.append(f" UWI .       {uwi}:" + eol)

    # Minimal parameter information section (to match legacy LAS 1.2 style)
    lines.append("~PARAMETER INFORMATION BLOCK" + eol)
    lines.append("#MNEM.UNIT       VALUE        DESCRIPTION" + eol)
    lines.append("#---------     -----------    ---------------------------" + eol)
    lines.append(" EKB .               0.0000:  ELEVATION OF KELLY BUSHING" + eol)

    # Curve information section
    lines.append("~CURVE INFORMATION BLOCK" + eol)
    lines.append("#MNEM.UNIT                 API CODE     CURVE DESCRIPTION" + eol)
    lines.append("#---------               -----------    ---------------------------" + eol)
    lines.append(f" DEPT.{unit_token}                 00 000 000 000:  DEPTH" + eol)
    for name, meta in curve_data.items():
        unit = meta.get("unit", "")
        lines.append(f" {name.upper()}.{unit} :  {name.upper()}" + eol)

    # ASCII data section (~A header with column labels)
    names = list(curve_data.keys())
    arrays = [curve_data[n]["values"] for n in names]

    header_cols = ["DEPTH"] + [n.upper() for n in names]
    header = " ".join(f"{c:>10}" for c in header_cols)
    lines.append("~A " + header + eol)

    for i in range(depth.size):
        row_vals = [f"{depth[i]:10.4f}"] + [f"{arrays[j][i]:10.4f}" for j in range(len(arrays))]
        lines.append(" ".join(row_vals) + eol)

    return "".join(lines)


def build_las_filename_from_metadata(header_metadata, default_name="digitized_log.las"):
    if not isinstance(header_metadata, dict):
        return default_name
    well = header_metadata.get('well')
    comp = header_metadata.get('comp') or header_metadata.get('company')

    def _clean(s):
        if not isinstance(s, str):
            return ''
        s = s.strip()
        if not s:
            return ''
        import re
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
        s = s.strip("_-")
        return s[:80]

    well_s = _clean(well)
    comp_s = _clean(comp)
    if well_s and comp_s:
        return f"{comp_s}__{well_s}.las"
    if well_s:
        return f"{well_s}.las"
    return default_name

# ----------------------------
# Google Vision API Functions
# ----------------------------
def downsample_for_ocr(image_bytes, max_height=2000):
    """Downsample large images before OCR to reduce memory usage"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return image_bytes
    
    h, w, _ = img.shape
    
    # Only downsample if height exceeds max_height
    if h <= max_height:
        return image_bytes
    
    # Calculate new dimensions maintaining aspect ratio
    scale = max_height / h
    new_w = int(w * scale)
    new_h = max_height
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Re-encode to bytes
    _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


def _get_easyocr_reader():
    global _easyocr_reader
    if not LOCAL_OCR_AVAILABLE or easyocr is None:
        return None
    if _easyocr_reader is None:
        use_gpu = bool(TORCH_AVAILABLE and torch is not None and torch.cuda.is_available())
        try:
            import contextlib
            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                _easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
            print(f"[OK] EasyOCR reader initialized ({'GPU' if use_gpu else 'CPU'}).")
        except Exception as exc:
            print(f"[WARN] EasyOCR initialization failed: {exc}")
            _easyocr_reader = False
    return _easyocr_reader if _easyocr_reader is not False else None


def _detect_text_easyocr(image_bytes):
    reader = _get_easyocr_reader()
    if reader is None:
        return {'raw': [], 'numbers': [], 'suggestions': {}}

    try:
        image_bytes = downsample_for_ocr(image_bytes, max_height=2200)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return {'raw': [], 'numbers': [], 'suggestions': {}}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
        ocr_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if max(ocr_img.shape[:2]) < 1800:
            ocr_img = cv2.resize(ocr_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        results = reader.readtext(
            ocr_img,
            detail=1,
            paragraph=False,
            text_threshold=0.5,
            low_text=0.25,
            link_threshold=0.3,
        )

        raw_text = []
        numeric_entries = []
        line_items = []
        scale_x = img.shape[1] / float(ocr_img.shape[1]) if ocr_img.shape[1] else 1.0
        scale_y = img.shape[0] / float(ocr_img.shape[0]) if ocr_img.shape[0] else 1.0

        for entry in results or []:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            bbox = entry[0]
            text = str(entry[1] or '').strip()
            if not text:
                continue

            verts = []
            xs = []
            ys = []
            for pt in bbox:
                try:
                    x = int(round(float(pt[0]) * scale_x))
                    y = int(round(float(pt[1]) * scale_y))
                except Exception:
                    continue
                verts.append({'x': x, 'y': y})
                xs.append(x)
                ys.append(y)
            if not verts:
                continue

            raw_text.append({
                'text': text,
                'vertices': verts,
            })

            if xs and ys:
                x_center = float(sum(xs)) / len(xs)
                y_center = float(sum(ys)) / len(ys)
                line_items.append((y_center, x_center, text))

            numbers = re.findall(r'-?\d*\.?\d+', text)
            if numbers and verts:
                x0 = verts[0]['x']
                y0 = verts[0]['y']
                for num in numbers:
                    try:
                        numeric_entries.append({
                            'value': float(num),
                            'text': text,
                            'x': x0,
                            'y': y0,
                        })
                    except ValueError:
                        continue

        full_text = ""
        if line_items:
            line_items.sort(key=lambda t: (t[0], t[1]))
            lines = []
            current_line = []
            current_y = None
            y_tol = max(10.0, img.shape[0] * 0.01)
            for y, x, text in line_items:
                if current_y is None or abs(y - current_y) <= y_tol:
                    if current_y is None:
                        current_y = y
                    current_line.append((x, text))
                else:
                    current_line.sort(key=lambda t: t[0])
                    lines.append(' '.join(t[1] for t in current_line if t[1]).strip())
                    current_line = [(x, text)]
                    current_y = y
            if current_line:
                current_line.sort(key=lambda t: t[0])
                lines.append(' '.join(t[1] for t in current_line if t[1]).strip())
            full_text = '\n'.join(line for line in lines if line)

        suggestions = build_ocr_suggestions(numeric_entries)
        suggestions = attach_curve_label_hints(suggestions, raw_text)

        return {
            'raw': raw_text,
            'numbers': numeric_entries,
            'suggestions': suggestions,
            'full_text': full_text
        }
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return {'raw': [], 'numbers': [], 'suggestions': {}}

def detect_text_vision_api(image_bytes):
    """Use Google Vision API or local OCR fallback to detect text in image"""
    if not VISION_API_AVAILABLE or vision_client is None:
        return _detect_text_easyocr(image_bytes)

    try:
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)

        full_text = ""
        if response.text_annotations:
            full_text = response.text_annotations[0].description

        raw_text = []
        numeric_entries = []
        for text in response.text_annotations[1:]:  # Skip first (full text)
            bounding = text.bounding_poly.vertices
            entry = {
                'text': text.description,
                'vertices': [{'x': int(v.x), 'y': int(v.y)} for v in bounding]
            }
            raw_text.append(entry)

            # Extract numeric tokens
            import re
            numbers = re.findall(r'-?\d*\.?\d+', text.description)
            for num in numbers:
                try:
                    value = float(num)
                    x = int(bounding[0].x)
                    y = int(bounding[0].y)
                    numeric_entries.append({
                        'value': value,
                        'text': text.description,
                        'x': x,
                        'y': y
                    })
                except ValueError:
                    continue

        suggestions = build_ocr_suggestions(numeric_entries)
        suggestions = attach_curve_label_hints(suggestions, raw_text)

        return {
            'raw': raw_text,
            'numbers': numeric_entries,
            'suggestions': suggestions,
            'full_text': full_text
        }
    except Exception as e:
        print(f"Vision API error: {e}")
        return _detect_text_easyocr(image_bytes)


@app.route('/reanalyze_panel', methods=['POST'])
def reanalyze_panel():
    """Re-run OCR/AI suggestions on a cropped panel region of the current image.

    Expects JSON with:
      - image: data URL string (same as /digitize)
      - region: { left_px, right_px, top_px, bottom_px } in image pixel coords
    """
    data = request.json or {}
    image_data = data.get('image')
    region = data.get('region') or {}

    if not image_data or ',' not in image_data:
        return jsonify({'success': False, 'error': 'Missing image data'}), 400

    try:
        img_payload = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(img_payload)
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid image data'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'error': 'Could not decode image'}), 400

    H, W, _ = img.shape
    try:
        left = max(0, int(region.get('left_px', 0)))
        right = min(W, int(region.get('right_px', W)))
        top = max(0, int(region.get('top_px', 0)))
        bottom = min(H, int(region.get('bottom_px', H)))
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid region'}), 400

    if right <= left or bottom <= top:
        return jsonify({'success': False, 'error': 'Empty region'}), 400

    # Crop to region
    crop = img[top:bottom, left:right]
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode crop'}), 500

    crop_bytes = buf.tobytes()

    # Run Vision OCR on cropped panel
    detected_text = detect_text_vision_api(crop_bytes)
    ocr_suggestions = detected_text.get('suggestions', {}) or {}

    # Attach color hints so curve suggestions stay consistent with panel
    try:
        ocr_suggestions = attach_color_hints_to_ocr_curves(crop, ocr_suggestions)
        detected_text['suggestions'] = ocr_suggestions
    except Exception:
        # If anything goes wrong here, still return basic OCR suggestions
        pass

    return jsonify({
        'success': True,
        'ocr_suggestions': ocr_suggestions,
        'detected_text': detected_text,
    })


@app.route('/crop_to_panel', methods=['POST'])
def crop_to_panel():
    """Crop the uploaded image to a working panel/depth window.

    Expects JSON with:
      - image: data URL string (same as /digitize)
      - region: { left_px, right_px, top_px, bottom_px } in image pixel coords

    Returns a new data URL plus the cropped width/height so the frontend can
    treat it as a shorter working image.
    """
    data = request.json or {}
    image_data = data.get('image')
    region = data.get('region') or {}

    if not image_data or ',' not in image_data:
        return jsonify({'success': False, 'error': 'Missing image data'}), 400

    try:
        img_payload = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(img_payload)
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid image data'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'error': 'Could not decode image'}), 400

    H, W, _ = img.shape
    try:
        left = max(0, int(region.get('left_px', 0)))
        right = min(W, int(region.get('right_px', W)))
        top = max(0, int(region.get('top_px', 0)))
        bottom = min(H, int(region.get('bottom_px', H)))
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid region'}), 400

    if right <= left or bottom <= top:
        return jsonify({'success': False, 'error': 'Empty region'}), 400

    crop = img[top:bottom, left:right]
    ch, cw, _ = crop.shape

    image_filename = f"{uuid.uuid4().hex}.jpg"
    image_path = Path(config.DATA_ROOT) / 'images' / image_filename
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode crop'}), 500

    b64 = base64.b64encode(buf).decode('ascii')
    data_url = f"data:image/jpeg;base64,{b64}"

    return jsonify({
        'success': True,
        'image': data_url,
        'image_path': f'/api/images/{image_filename}',
        'width': int(cw),
        'height': int(ch),
    })


@app.route('/propose_calibration', methods=['POST'])
def propose_calibration():
    """Use Vision + LLM to propose depth_axis and track calibration for a selected panel.

    Expects JSON with:
      - image: data URL string
      - region: { left_px, right_px, top_px, bottom_px } in image pixel coords
    """
    data = request.json or {}
    image_data = data.get('image')
    region = data.get('region') or {}

    if not image_data or ',' not in image_data:
        return jsonify({'success': False, 'error': 'Missing image data'}), 400

    try:
        img_payload = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(img_payload)
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid image data'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'error': 'Could not decode image'}), 400

    H, W, _ = img.shape
    try:
        left = max(0, int(region.get('left_px', 0)))
        right = min(W, int(region.get('right_px', W)))
        top = max(0, int(region.get('top_px', 0)))
        bottom = min(H, int(region.get('bottom_px', H)))
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid region'}), 400

    if right <= left or bottom <= top:
        return jsonify({'success': False, 'error': 'Empty region'}), 400

    # Crop to region
    crop = img[top:bottom, left:right]
    crop_h, crop_w, _ = crop.shape
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode crop'}), 500

    crop_bytes = buf.tobytes()

    # Run Vision OCR on cropped panel
    detected_text = detect_text_vision_api(crop_bytes)
    raw_text = detected_text.get('raw', [])
    numeric_entries = detected_text.get('numbers', [])

    if not numeric_entries:
        return jsonify({
            'success': False,
            'error': 'No numeric OCR entries found in selected panel; cannot propose calibration.'
        }), 400

    # Build calibration payload for the LLM
    # 1) Depth label candidates: include left-side AND center-column numeric entries
    # Many logs print depth in a center column, not just on the left edge.
    xs_all = [float(e['x']) for e in numeric_entries if 'x' in e]
    if xs_all:
        min_x = min(xs_all)
        max_x = max(xs_all)
        span_x = max(max_x - min_x, 1.0)
        # Expand threshold to 60% to catch center-column depth labels
        depth_x_threshold = min_x + 0.60 * span_x
    else:
        depth_x_threshold = None

    depth_label_candidates = []
    for e in numeric_entries:
        val = float(e['value'])
        # Filter: depth values are typically in range 0-50000 ft, not tiny curve scales
        if val < 0 or val > 50000:
            continue
        if depth_x_threshold is not None and float(e['x']) <= depth_x_threshold:
            depth_label_candidates.append({
                'value': val,
                'x_px': float(e['x']),
                'y_px': float(e['y']),
            })

    # 2) Header text boxes: text in top ~30% of crop
    y_vals_all = [float(e['y']) for e in numeric_entries if 'y' in e]
    if y_vals_all:
        y_min = min(y_vals_all)
        y_max = max(y_vals_all)
        header_threshold = y_min + 0.3 * (y_max - y_min)
    else:
        header_threshold = crop_h * 0.3

    header_text_boxes = []
    for entry in raw_text:
        text = (entry.get('text') or '').strip()
        if not text:
            continue
        verts = entry.get('vertices') or []
        ys = [v.get('y') for v in verts if isinstance(v, dict) and 'y' in v]
        xs = [v.get('x') for v in verts if isinstance(v, dict) and 'x' in v]
        if not ys or not xs:
            continue
        y_center = float(sum(ys)) / len(ys)
        x_center = float(sum(xs)) / len(xs)
        if y_center <= header_threshold:
            header_text_boxes.append({
                'text': text,
                'x_px': x_center,
                'y_px': y_center,
            })

    calib_payload = {
        'image': {
            'width_px': crop_w,
            'height_px': crop_h,
        },
        'depth_label_candidates': depth_label_candidates,
        'header_text_boxes': header_text_boxes,
    }

    # Call LLM to propose calibration
    calibration = call_ai_calibration(calib_payload)
    if not calibration:
        return jsonify({
            'success': False,
            'error': 'AI calibration failed or returned no result. Check server logs.'
        }), 500

    # Validate and fix obvious mismatches
    calibration = validate_and_fix_calibration(calibration)

    return jsonify({
        'success': True,
        'calibration': calibration,
    })


@app.route('/propose_curves', methods=['POST'])
def propose_curves():
    """Use Vision + LLM to propose curve tracks for a selected panel.

    Expects JSON with:
      - image: data URL string
      - region: { left_px, right_px, top_px, bottom_px } in image pixel coords
    """
    data = request.json or {}
    image_data = data.get('image')
    region = data.get('region') or {}

    if not image_data or ',' not in image_data:
        return jsonify({'success': False, 'error': 'Missing image data'}), 400

    try:
        img_payload = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(img_payload)
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid image data'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'error': 'Could not decode image'}), 400

    H, W, _ = img.shape
    try:
        left = max(0, int(region.get('left_px', 0)))
        right = min(W, int(region.get('right_px', W)))
        top = max(0, int(region.get('top_px', 0)))
        bottom = min(H, int(region.get('bottom_px', H)))
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid region'}), 400

    if right <= left or bottom <= top:
        return jsonify({'success': False, 'error': 'Empty region'}), 400

    panel = img[top:bottom, left:right]
    panel_h, panel_w, _ = panel.shape
    if panel_h < 2 or panel_w < 2:
        return jsonify({'success': False, 'error': 'Panel too small for curve suggestion'}), 400

    # Detect tracks within the panel using edge-based detector
    local_tracks = auto_detect_tracks(panel) or []
    tracks_out = []
    for idx, (lx, rx) in enumerate(local_tracks):
        try:
            lx_f = float(lx)
            rx_f = float(rx)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lx_f) or not np.isfinite(rx_f) or rx_f <= lx_f:
            continue
        tracks_out.append({
            'index': idx,
            'left_px': lx_f,
            'right_px': rx_f,
        })

    # Fallback: if we only found 0 or 1 track on a reasonably wide panel,
    # synthesize several equal-width tracks so curves can cover the full width.
    if len(tracks_out) <= 1 and panel_w >= 400:
        synth_tracks = []
        n_segments = 4
        seg_w = float(panel_w) / float(n_segments)
        for i in range(n_segments):
            lx_f = i * seg_w
            rx_f = (i + 1) * seg_w
            synth_tracks.append({
                'index': i,
                'left_px': lx_f,
                'right_px': rx_f,
            })
        if synth_tracks:
            print(f"[WARN] auto_detect_tracks found {len(tracks_out)} track(s); using {len(synth_tracks)} synthetic tracks instead.")
            tracks_out = synth_tracks

    if not tracks_out:
        return jsonify({'success': False, 'error': 'No tracks detected in panel for curve suggestion.'}), 400

    # Run Vision OCR on the same panel to get numeric + label hints
    ok, buf = cv2.imencode('.jpg', panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode panel for OCR'}), 500

    crop_bytes = buf.tobytes()
    detected_text = detect_text_vision_api(crop_bytes)
    ocr_suggestions = detected_text.get('suggestions', {}) or {}

    # Attach color hints based on panel image content
    try:
        ocr_suggestions = attach_color_hints_to_ocr_curves(panel, ocr_suggestions)
        detected_text['suggestions'] = ocr_suggestions
    except Exception:
        pass

    curve_payload = build_curve_suggestion_payload(panel, tracks_out, ocr_suggestions, detected_text)
    ai_result = call_ai_curve_suggestions(curve_payload)
    if not ai_result or not isinstance(ai_result, dict):
        return jsonify({'success': False, 'error': 'AI curve suggestion failed or returned no result.'}), 500

    ai_curves = ai_result.get('curves') or []
    print(f"[AI] Returned {len(ai_curves)} curve suggestions for {len(tracks_out)} detected tracks")

    curves_cfg = []
    rejected_reasons = []
    for idx, c in enumerate(ai_curves):
        try:
            track_index = int(c.get('track_index'))
        except (TypeError, ValueError):
            rejected_reasons.append(f"Curve {idx}: invalid track_index type")
            continue
        if track_index < 0 or track_index >= len(tracks_out):
            rejected_reasons.append(f"Curve {idx}: track_index {track_index} out of range (0-{len(tracks_out)-1})")
            continue

        track = tracks_out[track_index]
        abs_left = float(left) + float(track.get('left_px', 0.0))
        abs_right = float(left) + float(track.get('right_px', 0.0))
        if not np.isfinite(abs_left) or not np.isfinite(abs_right) or abs_right <= abs_left:
            rejected_reasons.append(f"Curve {idx}: invalid pixel range")
            continue

        mnemonic = (c.get('mnemonic') or track.get('name') or '').strip()
        if not mnemonic:
            mnemonic = f"CURVE{len(curves_cfg) + 1}"

        mode = (c.get('mode') or 'black').strip().lower()
        if mode not in ('black', 'red', 'blue', 'green'):
            mode = 'black'

        preferred = bool(c.get('preferred', False))

        curves_cfg.append({
            'mnemonic': mnemonic,
            'track_index': track_index,
            'preferred': preferred,
            'mode': mode,
            'left_px': abs_left,
            'right_px': abs_right,
        })

    if not curves_cfg:
        print(f"[ERROR] All {len(ai_curves)} AI curve suggestions rejected:")
        for reason in rejected_reasons:
            print(f"   - {reason}")

        # Fallback: if we have detected tracks, synthesize simple curves so the UI can proceed
        if tracks_out:
            print("[WARN] Falling back to heuristic curves from detected tracks.")
            for idx, t in enumerate(tracks_out[:6]):
                try:
                    abs_left = float(left) + float(t.get('left_px', 0.0))
                    abs_right = float(left) + float(t.get('right_px', 0.0))
                except Exception:
                    continue
                if not np.isfinite(abs_left) or not np.isfinite(abs_right) or abs_right <= abs_left:
                    continue

                mnemonic = (t.get('name') or f'CURVE{len(curves_cfg) + 1}').strip() or f'CURVE{len(curves_cfg) + 1}'

                curves_cfg.append({
                    'mnemonic': mnemonic,
                    'track_index': idx,
                    'preferred': idx == 0,
                    'mode': 'black',
                    'left_px': abs_left,
                    'right_px': abs_right,
                })

        if not curves_cfg:
            return jsonify({'success': False, 'error': 'AI returned no usable curve suggestions.'}), 400

    print(f"[OK] Accepted {len(curves_cfg)} curves, rejected {len(rejected_reasons)}")

    return jsonify({
        'success': True,
        'curves': curves_cfg,
        'raw_ai': ai_result,
        'payload': curve_payload,
    })


@app.route('/api/auto_layout', methods=['POST'])
def auto_layout_tracks():
    data = request.json or {}
    image_data = data.get('image')
    region = data.get('region') or {}
    treat_region_as_header = bool(data.get('treat_region_as_header'))

    if not image_data or ',' not in image_data:
        return jsonify({'success': False, 'error': 'Missing image data'}), 400

    try:
        img_payload = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(img_payload)
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid image data'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'error': 'Could not decode image'}), 400

    H, W, _ = img.shape
    try:
        left = max(0, int(region.get('left_px', 0)))
        right = min(W, int(region.get('right_px', W)))
        top = max(0, int(region.get('top_px', 0)))
        bottom = min(H, int(region.get('bottom_px', H)))
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid region'}), 400

    if right <= left or bottom <= top:
        return jsonify({'success': False, 'error': 'Empty region'}), 400

    panel = img[top:bottom, left:right]
    panel_h, panel_w, _ = panel.shape
    if panel_h < 2 or panel_w < 2:
        return jsonify({'success': False, 'error': 'Panel too small for layout detection'}), 400

    # For normal panel-based layout, only the top band is treated as the
    # header. For explicit header/key capture, the entire region is the
    # header strip, so skip the extra crop.
    if treat_region_as_header:
        header = panel
        header_h = panel_h
    else:
        header_h = max(10, int(panel_h * 0.15))
        header = panel[0:header_h, :]

    ok, buf = cv2.imencode('.jpg', header, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode header crop'}), 500

    header_bytes = buf.tobytes()

    detected_text = detect_text_vision_api(header_bytes)
    raw_text = detected_text.get('raw', []) or []
    full_text_blob = detected_text.get('full_text', '')

    def _extract_header_metadata(raw_entries, full_text=''):
        if not isinstance(raw_entries, list) or not raw_entries:
            raw_entries = []
        try:
            items_local = []
            entry_items = []
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue
                text = (entry.get('text') or '').strip()
                if not text:
                    continue
                verts = entry.get('vertices') or []
                ys_local = [v.get('y') for v in verts if isinstance(v, dict) and 'y' in v]
                xs_local = [v.get('x') for v in verts if isinstance(v, dict) and 'x' in v]
                if not ys_local or not xs_local:
                    continue
                left_local = float(min(xs_local))
                right_local = float(max(xs_local))
                top_local = float(min(ys_local))
                bottom_local = float(max(ys_local))
                y = float(sum(ys_local)) / len(ys_local)
                x = float(sum(xs_local)) / len(xs_local)
                items_local.append((y, x, text))
                entry_items.append({
                    'text': text,
                    'left': left_local,
                    'right': right_local,
                    'top': top_local,
                    'bottom': bottom_local,
                    'x': x,
                    'y': y,
                    'width': max(1.0, right_local - left_local),
                    'height': max(1.0, bottom_local - top_local),
                })
            lines = []
            if items_local:
                items_local.sort(key=lambda t: (t[0], t[1]))

                y_tol = 8.0
                current_y = None
                current_tokens = []
                for y, x, text in items_local:
                    if current_y is None or abs(y - current_y) <= y_tol:
                        if current_y is None:
                            current_y = y
                        current_tokens.append((x, text))
                    else:
                        current_tokens.sort(key=lambda t: t[0])
                        lines.append(' '.join(t[1] for t in current_tokens if t[1]).strip())
                        current_y = y
                        current_tokens = [(x, text)]
                if current_tokens:
                    current_tokens.sort(key=lambda t: t[0])
                    lines.append(' '.join(t[1] for t in current_tokens if t[1]).strip())

            if isinstance(full_text, str) and full_text.strip():
                for raw_line in full_text.splitlines():
                    line = raw_line.strip()
                    if line and line not in lines:
                        lines.append(line)

            if not lines:
                return None

            import re
            next_label_re = re.compile(
                r"\b(?:COMPANY|WELL|FIELD|LOCATION|COUNTY|STATE|PROV(?:INCE)?|SERVICE\s+COMPANY|DATE|API|UWI)\b",
                flags=re.IGNORECASE,
            )
            next_label_with_delim_re = re.compile(
                r"\b(?:COMPANY|WELL|FIELD|LOCATION|COUNTY|STATE|PROV(?:INCE)?|SERVICE\s+COMPANY|DATE|API|UWI)\b(?=\s*[:=])",
                flags=re.IGNORECASE,
            )

            def pick_after(label_re, s):
                m = re.search(label_re, s, flags=re.IGNORECASE)
                if not m:
                    return None
                tail = s[m.end():].strip(" :-\t")
                if not tail:
                    return None
                next_match = next_label_with_delim_re.search(tail)
                if next_match and next_match.start() > 0:
                    tail = tail[:next_match.start()].strip(" :-\t")
                return tail.strip() if tail else None

            def clean_value_text(text):
                value = str(text or '').strip()
                value = re.sub(r"^[\s:._,\-=/\\|]+", "", value)
                value = re.sub(r"[\s:._,\-=/\\|]+$", "", value)
                value = re.sub(r"\s{2,}", " ", value)
                return value.strip()

            def looks_like_another_label(text, current_pattern):
                raw = str(text or '').strip()
                if not raw:
                    return False
                if re.search(current_pattern, raw, flags=re.IGNORECASE):
                    return False
                return bool(next_label_re.search(raw))

            def is_label_anchor(text, label_pattern):
                raw = str(text or '').strip()
                if not raw:
                    return False
                return bool(re.match(rf"^\W*(?:{label_pattern})(?:\W|$)", raw, flags=re.IGNORECASE))

            label_specs = (
                ('comp', r"\bCOMPANY\b"),
                ('well', r"\bWELL\b"),
                ('fld', r"\bFIELD\b"),
                ('loc', r"\bLOCATION\b"),
                ('county', r"\bCOUNTY\b"),
                ('state', r"\bSTATE\b"),
                ('prov', r"\bPROV(?:INCE)?\b"),
                ('srvc', r"\bSERVICE\s+COMPANY\b"),
                ('date', r"\bDATE\b"),
                ('api', r"\bAPI\b"),
                ('uwi', r"\bUWI\b"),
            )

            median_height = float(np.median([item['height'] for item in entry_items])) if entry_items else 14.0

            def spatial_pick_value(label_pattern):
                if not entry_items:
                    return None

                best_value = None
                best_score = None
                for label_entry in entry_items:
                    label_text = label_entry['text']
                    if not is_label_anchor(label_text, label_pattern):
                        continue

                    inline_value = clean_value_text(pick_after(label_pattern, label_text))
                    if inline_value:
                        score = (3, len(inline_value), -int(label_entry['left']))
                        if best_score is None or score > best_score:
                            best_score = score
                            best_value = inline_value

                    row_tol = max(14.0, median_height * 0.9, label_entry['height'] * 0.8)
                    max_dx = max(180.0, panel_w * 0.45)
                    max_gap = max(26.0, median_height * 2.5)

                    candidates = [
                        other for other in entry_items
                        if other is not label_entry
                        and other['left'] >= label_entry['right'] - 4.0
                        and abs(other['y'] - label_entry['y']) <= row_tol
                        and (other['left'] - label_entry['right']) <= max_dx
                    ]
                    candidates.sort(key=lambda item: item['left'])

                    parts = []
                    prev_right = label_entry['right']
                    first_dx = None
                    for cand in candidates:
                        gap = cand['left'] - prev_right
                        if gap > max_gap and parts:
                            break
                        if first_dx is None:
                            first_dx = max(0.0, cand['left'] - label_entry['right'])
                        if looks_like_another_label(cand['text'], label_pattern):
                            if parts:
                                break
                            continue
                        parts.append(cand['text'])
                        prev_right = cand['right']

                    spatial_value = clean_value_text(' '.join(parts))
                    if spatial_value:
                        score = (2, len(spatial_value), -int(first_dx or 0), -int(label_entry['left']))
                        if best_score is None or score > best_score:
                            best_score = score
                            best_value = spatial_value

                return best_value or None

            md = {}
            for key, pat in label_specs:
                if key in md:
                    continue
                val = spatial_pick_value(pat)
                if val:
                    md[key] = val

            for s in lines:
                if not s:
                    continue
                for key, pat in label_specs:
                    if key in md:
                        continue
                    val = clean_value_text(pick_after(pat, s))
                    if val:
                        md[key] = val

                if 'api' not in md:
                    m = re.search(r"\b(\d{2}[- ]?\d{3}[- ]?\d{5})\b", s)
                    if m:
                        md['api'] = m.group(1).replace(' ', '-')
                if 'date' not in md:
                    m = re.search(r"\b(\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", s)
                    if m:
                        md['date'] = m.group(1)

            return md if md else None
        except Exception:
            return None

    header_metadata = _extract_header_metadata(raw_text, full_text_blob) if treat_region_as_header else None

    items = []
    for entry in raw_text:
        text = (entry.get('text') or '').strip()
        if not text:
            continue
        verts = entry.get('vertices') or []
        xs = [v.get('x') for v in verts if isinstance(v, dict) and 'x' in v]
        ys = [v.get('y') for v in verts if isinstance(v, dict) and 'y' in v]
        if not xs or not ys:
            continue
        x_center = float(sum(xs)) / len(xs)
        y_center = float(sum(ys)) / len(ys)
        items.append({
            'text': text,
            'x': x_center,
            'y': y_center,
        })

    # If no header text found, fall back to edge-based track detection
    if not items and not full_text_blob:
        print("[WARN] No header text found; falling back to edge-based track detection")
        try:
            local_tracks = auto_detect_tracks(panel)
            tracks_out = []
            for idx, (lx, rx) in enumerate(local_tracks or []):
                try:
                    lx_f = float(lx)
                    rx_f = float(rx)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(lx_f) or not np.isfinite(rx_f) or rx_f <= lx_f:
                    continue
                tracks_out.append({
                    'name': f'Track{idx+1}',
                    'left_px': float(left) + lx_f,
                    'right_px': float(left) + rx_f,
                    'scale_min': None,
                    'scale_max': None,
                    'unit': None,
                    'hot_side': None,
                })
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()[-1500:]
            if is_prod:
                tb = None
            return jsonify({
                'success': False,
                'error': f'Edge fallback failed: {str(exc)}',
                'traceback': tb
            }), 500
        
        if not tracks_out:
            return jsonify({'success': False, 'error': 'No tracks detected (neither header text nor edge detection found tracks).'}), 400
        
        return jsonify({
            'success': True,
            'tracks': tracks_out,
            'raw_layout': {'tracks': [], 'fallback': 'edge_detection'},
            'header_metadata': header_metadata,
        })

    layout_payload = {
        'image': {
            'width_px': panel_w,
            'height_px': header_h,
        },
        'items': items,
        'full_text': full_text_blob,
    }

    layout = call_ai_auto_layout(layout_payload)
    if not layout:
        has_provider = bool((GEMINI_API_KEY and GEMINI_MODEL_ID) or (OPENAI_API_KEY and OPENAI_MODEL_ID) or (HF_API_TOKEN and HF_MODEL_ID))

        # Fall back to edge-based track detection on the panel, even when no AI
        # provider is configured. Header capture can still be useful with OCR-
        # extracted metadata plus geometric track boxes.
        if not has_provider:
            print("[WARN] AI layout detection is not configured; falling back to edge-based track detection")
        else:
            print("[WARN] AI layout inference returned no result; falling back to edge-based track detection")
        try:
            local_tracks = auto_detect_tracks(panel)
            tracks_out = []
            for idx, (lx, rx) in enumerate(local_tracks or []):
                try:
                    lx_f = float(lx)
                    rx_f = float(rx)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(lx_f) or not np.isfinite(rx_f) or rx_f <= lx_f:
                    continue
                tracks_out.append({
                    'name': f'Track{idx+1}',
                    'left_px': float(left) + lx_f,
                    'right_px': float(left) + rx_f,
                    'scale_min': None,
                    'scale_max': None,
                    'unit': None,
                    'hot_side': None,
                    'color_hint': None,
                })
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()[-1500:]
            if is_prod:
                tb = None
            return jsonify({
                'success': False,
                'error': f'AI layout returned no result, and edge fallback failed: {str(exc)}',
                'traceback': tb
            }), 500

        if tracks_out:
            return jsonify({
                'success': True,
                'tracks': tracks_out,
                'raw_layout': {
                    'tracks': [],
                    'fallback': 'edge_detection_no_ai_provider' if not has_provider else 'edge_detection_after_ai_failure',
                    'ocr_items': len(items),
                },
                'header_metadata': header_metadata,
            })

        if treat_region_as_header and header_metadata:
            return jsonify({
                'success': True,
                'tracks': [],
                'raw_layout': {
                    'tracks': [],
                    'fallback': 'metadata_only_after_ai_failure',
                    'ocr_items': len(items),
                },
                'header_metadata': header_metadata,
            })

        return jsonify({
            'success': False,
            'error': f"AI layout detection failed and edge fallback found no tracks. OCR items={len(items)}. Try selecting a larger/clearer header region."
        }), 500

    raw_tracks = layout.get('tracks') or []
    
    # Merge AI-extracted metadata (often better than regex)
    ai_meta = layout.get('header_metadata')
    if ai_meta and isinstance(ai_meta, dict):
        if header_metadata is None:
            header_metadata = {}
        for k, v in ai_meta.items():
            if v and isinstance(v, str) and v.strip():
                val = v.strip()
                # Map AI keys to internal keys where they differ
                if k == 'company': header_metadata['comp'] = val
                elif k == 'field': header_metadata['fld'] = val
                elif k == 'location': header_metadata['loc'] = val
                elif k == 'province': header_metadata['prov'] = val
                elif k == 'service_company': header_metadata['srvc'] = val
                else:
                    # well, api, date, county, state, etc. match or are new
                    header_metadata[k] = val

    tracks_out = []
    for t in raw_tracks:
        try:
            lx = float(t.get('left_x'))
            rx = float(t.get('right_x'))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lx) or not np.isfinite(rx):
            continue
        lx = max(0.0, min(float(panel_w), lx))
        rx = max(0.0, min(float(panel_w), rx))
        if rx <= lx:
            continue

        track_out = {
            'name': t.get('name'),
            'left_px': float(left) + lx,
            'right_px': float(left) + rx,
            'scale_min': t.get('scale_min'),
            'scale_max': t.get('scale_max'),
            'unit': t.get('unit'),
            'hot_side': t.get('hot_side'),
            'color_hint': t.get('color_hint'),
        }
        tracks_out.append(track_out)

    if not tracks_out:
        if treat_region_as_header and header_metadata:
            return jsonify({
                'success': True,
                'tracks': [],
                'raw_layout': layout,
                'header_metadata': header_metadata,
            })
        return jsonify({'success': False, 'error': 'AI layout returned no usable tracks.'}), 400

    return jsonify({
        'success': True,
        'tracks': tracks_out,
        'raw_layout': layout,
        'header_metadata': header_metadata,
    })


def build_curve_suggestion_payload(panel_image, tracks_out, ocr_suggestions, detected_text):
    h, w = panel_image.shape[:2]

    tracks = []
    for idx, t in enumerate(tracks_out or []):
        try:
            left_px = float(t.get('left_px'))
            right_px = float(t.get('right_px'))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(left_px) or not np.isfinite(right_px) or right_px <= left_px:
            continue
        track = {
            'index': idx,
            'left_px': left_px,
            'right_px': right_px,
        }
        name = t.get('name')
        if isinstance(name, str) and name:
            track['name'] = name
        unit = t.get('unit')
        if isinstance(unit, str) and unit:
            track['unit'] = unit
        smin = t.get('scale_min')
        smax = t.get('scale_max')
        if isinstance(smin, (int, float)) and np.isfinite(smin):
            track['scale_min'] = float(smin)
        if isinstance(smax, (int, float)) and np.isfinite(smax):
            track['scale_max'] = float(smax)
        tracks.append(track)

    curves_hint = (ocr_suggestions or {}).get('curves') or []
    header_labels = []
    for c in curves_hint:
        try:
            lx = float(c.get('left_px'))
            rx = float(c.get('right_px'))
        except (TypeError, ValueError):
            lx = None
            rx = None
        x_center = None
        if lx is not None and rx is not None and np.isfinite(lx) and np.isfinite(rx):
            x_center = 0.5 * (lx + rx)
        label_text = c.get('label_text') or c.get('label_mnemonic') or c.get('type')
        if not label_text:
            continue
        label = {'text': str(label_text)}
        if x_center is not None:
            label['x_px'] = x_center
        label_type = c.get('label_type') or c.get('type')
        if label_type:
            label['curve_type'] = str(label_type)
        header_labels.append(label)

    full_text = ''
    if isinstance(detected_text, dict):
        raw_entries = detected_text.get('raw') or []
        texts = []
        for entry in raw_entries:
            if isinstance(entry, dict):
                t = entry.get('text')
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        full_text = '\n'.join(texts)
    elif isinstance(detected_text, str):
        full_text = detected_text

    return {
        'image': {
            'width_px': int(w),
            'height_px': int(h),
        },
        'tracks': tracks,
        'header_labels': header_labels,
        'raw_text': full_text,
    }


def call_ai_curve_suggestions(curve_payload):
    if not curve_payload:
        return None

    schema_hint = (
        "You are helping digitize paper well logs into LAS. You receive a list of "
        "tracks with x positions, rough names, and header labels. "
        "Your job is to decide which tracks correspond to which curves and which "
        "2-3 curves should be digitized by default. Always respond with JSON ONLY "
        "using this schema:\n\n"
        "{\n"
        "  \"curves\": [\n"
        "    {\n"
        "      \"mnemonic\": string,\n"
        "      \"track_index\": integer,\n"
        "      \"preferred\": boolean,\n"
        "      \"mode\": string\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Guidelines:\n"
        "- Match header_labels to nearest track by x position when possible.\n"
        "- Prefer GR, DT, RHOB, NPHI, RES when choosing preferred curves.\n"
        "- Never invent track indices; only use those present in tracks[].\n"
    )

    payload_text = schema_hint + json.dumps(curve_payload, indent=2)

    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith("models/") else f"models/{GEMINI_MODEL_ID}"
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            body = {"contents": [{"parts": [{"text": payload_text}]}]}
            resp = requests.post(url, json=body, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        curves = _extract_json_object(text)
                        if isinstance(curves, dict):
                            return curves
        except Exception as exc:
            print(f"Gemini API error (curve_suggestions): {exc}")

    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            messages = [
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": payload_text},
            ]
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                curves = _extract_json_object(content)
                if isinstance(curves, dict):
                    return curves
        except Exception as exc:
            print(f"OpenAI API error (curve_suggestions): {exc}")

    if not HF_API_TOKEN or not HF_MODEL_ID:
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        print(f"HF InferenceClient init error (curve_suggestions): {exc}")
        return None

    try:
        out = client.text_generation(
            payload_text,
            model=HF_MODEL_ID,
            max_new_tokens=512,
            temperature=0.1,
        )
        curves = _extract_json_object(out if isinstance(out, str) else str(out))
        if isinstance(curves, dict):
            return curves
    except Exception as exc:
        print(f"HF text_generation error (curve_suggestions): {exc}")

    return None


def build_ocr_suggestions(numeric_entries):
    """Derive depth and curve hints from numeric OCR entries."""
    if not numeric_entries:
        return {}

    # Sort by y (top to bottom)
    sorted_entries = sorted(numeric_entries, key=lambda n: n['y'])

    depth_candidates = []
    curve_candidates = []

    # Estimate horizontal extent of all numeric text so we can treat roughly
    # the left ~35% as potential depth-scale labels and the rest as curve
    # values. This mirrors the reference approach that only uses left-side
    # numbers for depth.
    xs_all = [float(e['x']) for e in sorted_entries if 'x' in e]
    if xs_all:
        min_x = min(xs_all)
        max_x = max(xs_all)
        span_x = max(max_x - min_x, 1.0)
        depth_x_threshold = min_x + 0.35 * span_x
    else:
        depth_x_threshold = None

    for entry in sorted_entries:
        value = entry['value']
        y = entry['y']
        x = entry['x']

        use_as_depth = False
        if depth_x_threshold is not None:
            use_as_depth = float(x) <= depth_x_threshold

        if use_as_depth:
            depth_candidates.append({'value': value, 'y': y})
        else:
            curve_candidates.append({'value': value, 'x': x, 'y': y})

    depth_hint = None
    if len(depth_candidates) >= 2:
        # Filter out obviously non-depth values (e.g., huge magnitudes that would
        # produce spans like 700000 to -200000). If filtering removes everything,
        # fall back to the original set.
        filtered = [d for d in depth_candidates if abs(d['value']) <= 100000]
        use_entries = filtered if len(filtered) >= 2 else depth_candidates

        # Use detected depth labels to fit a simple linear scale
        ys = np.array([d['y'] for d in use_entries], dtype=np.float32)
        vals = np.array([d['value'] for d in use_entries], dtype=np.float32)

        try:
            # depth  a * pixel_y + b
            a, b = np.polyfit(ys, vals, 1)
            y_top = float(ys.min())
            y_bottom = float(ys.max())
            top_depth_fit = float(a * y_top + b)
            bottom_depth_fit = float(a * y_bottom + b)

            # Ensure we have a sensible span and that depth is reasonably
            # monotonic with pixel position.
            span_val = abs(bottom_depth_fit - top_depth_fit)
            max_span = 100000.0  # reject clearly insane ranges
            min_span = 5.0       # avoid noise from tiny spans
            ok_span = (span_val >= min_span and span_val <= max_span)

            corr = np.corrcoef(ys, vals)[0, 1] if ys.size >= 2 else 1.0
            ok_corr = np.isfinite(corr) and abs(corr) >= 0.9

            ok_magnitude = all(abs(v) <= 1e6 for v in (top_depth_fit, bottom_depth_fit))

            if y_bottom > y_top and top_depth_fit != bottom_depth_fit and ok_span and ok_corr and ok_magnitude:
                depth_hint = {
                    'top_depth': top_depth_fit,
                    'bottom_depth': bottom_depth_fit,
                    'top_px': y_top,
                    'bottom_px': y_bottom,
                    'fit_labels': [
                        {'depth': float(v), 'y_px': float(y)}
                        for (v, y) in zip(vals.tolist(), ys.tolist())
                    ]
                }
        except Exception:
            # Fallback to using just the first/last labels if fitting fails,
            # but still reject clearly unreasonable spans.
            top = depth_candidates[0]
            bottom = depth_candidates[-1]
            span_val = abs(bottom['value'] - top['value'])
            if (
                bottom['y'] > top['y']
                and bottom['value'] != top['value']
                and span_val >= 5.0
                and span_val <= 100000.0
                and all(abs(v) <= 1e6 for v in (top['value'], bottom['value']))
            ):
                depth_hint = {
                    'top_depth': top['value'],
                    'bottom_depth': bottom['value'],
                    'top_px': top['y'],
                    'bottom_px': bottom['y']
                }

    # Suggest curve bounds by clustering x positions
    curve_hint = None
    if curve_candidates:
        sorted_curves = sorted(curve_candidates, key=lambda c: c['x'])
        clusters = min(3, len(sorted_curves))
        if clusters:
            chunk_size = int(np.ceil(len(sorted_curves) / clusters))
            curve_hint = []
            for idx in range(clusters):
                start = idx * chunk_size
                end = min(len(sorted_curves), (idx + 1) * chunk_size)
                chunk = sorted_curves[start:end]
                if not chunk:
                    continue
                xs = [p['x'] for p in chunk]
                curve_hint.append({
                    'left_px': min(xs),
                    'right_px': max(xs),
                    'sample_value': float(np.mean([p['value'] for p in chunk]))
                })

    # Try to refine depths using header/table information. Many logs print a
    # small table with "Top" / "Bottom" / "Total depth" values in ft near the
    # top of the page. We look only in that header band and only at numbers that
    # look like depths in feet (e.g. 4449.90 ft, 10026.53 ft), ignoring other
    # units like us/ft or lbf.
    header_top_val = None
    header_bottom_val = None
    if sorted_entries:
        y_vals_all = [e['y'] for e in sorted_entries]
        if y_vals_all:
            y_min = min(y_vals_all)
            y_max = max(y_vals_all)
            # Top 25% OR bottom 25% of text as the "header" band
            # (some logs, e.g. ATR, print header tables at the bottom)
            top_band_cut = y_min + 0.25 * (y_max - y_min)
            bottom_band_start = y_min + 0.75 * (y_max - y_min)

            header_depth_vals_strict = []  # require explicit "ft" (not us/ft)
            header_depth_vals_loose = []   # any plausible depth magnitude
            for e in sorted_entries:
                if top_band_cut < e['y'] < bottom_band_start:
                    continue
                text_l = str(e.get('text', '')).lower()
                val = e.get('value')
                if not np.isfinite(val):
                    continue
                # Depths in feet are typically hundreds to tens of thousands of
                # units, not tiny (0.0) and not enormous.
                if abs(val) < 100 or abs(val) > 50000:
                    continue

                header_depth_vals_loose.append(float(val))

                # Strict: same token mentions "ft" but not sonic units.
                if 'ft' in text_l and 'us/ft' not in text_l and 'ms/ft' not in text_l:
                    header_depth_vals_strict.append(float(val))

            use_vals = header_depth_vals_strict
            if not use_vals and len(header_depth_vals_loose) >= 2:
                # Fallback: no explicit "ft" nearby, but we do see multiple
                # plausible depth values in the header band. Use the min/max
                # only if they span a reasonable interval so we do not mistake
                # small curve scales (e.g. 1.95–2.95) for depths.
                header_depth_vals_loose.sort()
                span = header_depth_vals_loose[-1] - header_depth_vals_loose[0]
                if span >= 200.0:  # require at least a few hundred feet span
                    use_vals = header_depth_vals_loose

            if use_vals:
                use_vals.sort()
                if len(use_vals) >= 2:
                    header_top_val = use_vals[0]
                    header_bottom_val = use_vals[-1]
                else:
                    # Only a single header depth (e.g. "Total Depth @ 10015 ft")
                    header_top_val = header_bottom_val = use_vals[0]

    if depth_hint and header_top_val is not None and header_bottom_val is not None:
        # Override the fitted depths so they match the header values, while
        # preserving the original orientation (increasing or decreasing depth).
        d_top = float(depth_hint['top_depth'])
        d_bottom = float(depth_hint['bottom_depth'])
        if d_top <= d_bottom:
            depth_hint['top_depth'] = header_top_val
            depth_hint['bottom_depth'] = header_bottom_val
        else:
            depth_hint['top_depth'] = header_bottom_val
            depth_hint['bottom_depth'] = header_top_val

    # If we could not infer a depth scale from the physical log but we do have
    # plausible header depths, still provide a depth_hint so the UI can
    # auto-fill the top/bottom depth values. Pixel positions (top_px,
    # bottom_px) will remain unchanged in that case.
    if depth_hint is None and header_top_val is not None and header_bottom_val is not None:
        depth_hint = {
            'top_depth': header_top_val,
            'bottom_depth': header_bottom_val,
        }

    suggestions = {}
    if depth_hint:
        suggestions['depth'] = depth_hint
        labels = depth_hint.get('fit_labels') or [
            {'depth': d['value'], 'y_px': d['y']} for d in depth_candidates
        ]
        suggestions['depth_labels'] = labels
    if curve_hint:
        suggestions['curves'] = curve_hint

    return suggestions


def attach_curve_label_hints(suggestions, raw_text):
    """Attach curve label/type hints to OCR suggestions using nearby text.

    This does NOT auto-apply anything. It only adds optional fields like
    label_type/label_mnemonic/label_text to the curve hint objects so the
    frontend can show them as suggestions.
    """
    if not suggestions or not raw_text:
        return suggestions

    curves = suggestions.get('curves') or []
    if not curves:
        return suggestions

    # Estimate a "header band" (top portion of the image) from text bounding boxes
    y_centers_all = []
    for entry in raw_text:
        verts = entry.get('vertices') or []
        ys_all = [v.get('y') for v in verts if isinstance(v, dict) and 'y' in v]
        if ys_all:
            y_centers_all.append(float(sum(ys_all)) / len(ys_all))

    if not y_centers_all:
        return suggestions

    min_y = min(y_centers_all)
    max_y = max(y_centers_all)
    header_threshold = min_y + 0.3 * (max_y - min_y)   # top ~30%
    footer_threshold = min_y + 0.7 * (max_y - min_y)   # bottom ~30% starts here

    # Build candidate labels from raw text restricted to top OR bottom header band
    candidates = []
    for entry in raw_text:
        text = (entry.get('text') or '').strip()
        if not text:
            continue
        label_upper = text.upper()

        label_type = None
        if 'GAMMA' in label_upper or label_upper.startswith('GR'):
            label_type = 'GR'
        elif 'RHOB' in label_upper or 'RHO' in label_upper or 'DENS' in label_upper:
            label_type = 'RHOB'
        elif 'NPHI' in label_upper or 'NEUTRON' in label_upper:
            label_type = 'NPHI'
        elif 'DTC' in label_upper or 'DT ' in label_upper or label_upper.startswith('DT') or 'SONIC' in label_upper:
            label_type = 'DT'
        elif 'CALI' in label_upper or 'CALIPER' in label_upper:
            label_type = 'CALI'
        elif label_upper == 'SP' or 'SPONTANEOUS' in label_upper:
            label_type = 'SP'

        if not label_type:
            continue

        verts = entry.get('vertices') or []
        xs = [v.get('x') for v in verts if isinstance(v, dict) and 'x' in v]
        ys = [v.get('y') for v in verts if isinstance(v, dict) and 'y' in v]
        if not xs or not ys:
            continue

        y_center = float(sum(ys)) / len(ys)
        if header_threshold < y_center < footer_threshold:
            # Skip labels that are in the middle of the image (log body), not header/footer bands
            continue

        x_center = float(sum(xs)) / len(xs)

        candidates.append({
            'type': label_type,
            'text': text,
            'x': x_center,
            'y': y_center,
        })

    if not candidates:
        return suggestions

    # Associate candidate labels with each curve by horizontal proximity
    for curve in curves:
        left_px = curve.get('left_px')
        right_px = curve.get('right_px')
        if left_px is None or right_px is None:
            continue

        track_center = 0.5 * (left_px + right_px)
        best = None
        best_dist = None
        margin = (right_px - left_px) * 0.5 + 30  # allow some slack

        for cand in candidates:
            dx = cand['x'] - track_center
            if abs(dx) > margin:
                continue
            if best is None or abs(dx) < best_dist:
                best = cand
                best_dist = abs(dx)

        if best is not None:
            label_type = best['type']
            defaults = CURVE_TYPE_DEFAULTS.get(label_type, {})
            curve['label_type'] = label_type
            curve['label_mnemonic'] = defaults.get('mnemonic', label_type)
            curve['label_unit'] = defaults.get('unit')
            curve['label_text'] = best['text']
            curve['label_x'] = best.get('x')
            curve['label_y'] = best.get('y')

    return suggestions


def attach_color_hints_to_ocr_curves(image_array, suggestions):
    """Attach simple color-based hints to OCR curve suggestions.

    For each suggested curve track, look at the underlying image region and
    estimate whether it appears predominantly red or dark. This is used only
    to provide hints / default mode suggestions; the user remains in control.
    """
    if not isinstance(suggestions, dict):
        return suggestions

    curves = suggestions.get('curves') or []
    if not curves:
        return suggestions

    h, w = image_array.shape[:2]

    for curve in curves:
        left_px = curve.get('left_px')
        right_px = curve.get('right_px')
        if left_px is None or right_px is None:
            continue

        try:
            left = int(left_px)
            right = int(right_px)
        except Exception:
            continue

        left = max(0, min(w - 1, left))
        right = max(0, min(w, right))
        if right <= left:
            continue

        roi = image_array[:, left:right]
        if roi.size == 0:
            continue

        mean_color = roi.reshape(-1, 3).mean(axis=0)  # B, G, R
        b, g, r = [float(c) for c in mean_color]

        dominant = "mixed"
        recommended_mode = "black"

        if b > r * 1.2 and b > g * 1.2 and b > 60:
            dominant = "blue"
            recommended_mode = "blue"
        elif g > r * 1.2 and g > b * 1.2 and g > 60:
            dominant = "green"
            recommended_mode = "green"
        elif r > g * 1.2 and r > b * 1.2 and r > 60:
            dominant = "red"
            recommended_mode = "red"
        elif max(b, g, r) < 80:
            dominant = "dark"
            recommended_mode = "black"
        elif max(b, g, r) < 150:
            dominant = "gray"
            recommended_mode = "black"

        if dominant == "red":
            hint_text = "Track appears predominantly red; consider using Red mode for detection."
        elif dominant == "green":
            hint_text = "Track appears predominantly green; consider using Green mode for detection."
        elif dominant == "blue":
            hint_text = "Track appears predominantly blue; consider using Blue mode for detection."
        elif dominant in ("dark", "gray"):
            hint_text = "Track appears mostly dark; Black mode is likely appropriate."
        else:
            hint_text = "Track color is mixed; choose Red/Black/Blue/Green mode based on how the curve is drawn."

        curve['color_dominant'] = dominant
        curve['color_recommended_mode'] = recommended_mode
        curve['color_hint_text'] = hint_text

    return suggestions


def compute_curve_outlier_warnings(curves_cfg, las_curve_data, null_val):
    """Simple range-based sanity checks for GR/RHOB/DT curves.

    This does not block LAS generation; it only returns human-readable
    warning strings that the frontend can display alongside status.
    """
    warnings = []
    if not curves_cfg or not las_curve_data:
        return warnings


def compute_depth_warnings(depth_cfg, image_height):
    """Basic sanity checks for depth configuration.

    This checks for monotonicity and a reasonable depth-per-pixel scale.
    Returns a list of human-readable warning strings.
    """
    if not depth_cfg:
        return []

    warnings = []

    try:
        top_px = float(depth_cfg.get('top_px'))
        bottom_px = float(depth_cfg.get('bottom_px'))
        top_depth = float(depth_cfg.get('top_depth'))
        bottom_depth = float(depth_cfg.get('bottom_depth'))
    except Exception:
        return warnings

    if not np.isfinite(top_px) or not np.isfinite(bottom_px) or not np.isfinite(top_depth) or not np.isfinite(bottom_depth):
        return warnings

    if bottom_px <= top_px:
        warnings.append(f"Bottom pixel ({bottom_px:.0f}) is not below top pixel ({top_px:.0f}); check depth window.")

    depth_span = bottom_depth - top_depth
    pix_span = bottom_px - top_px

    if depth_span == 0:
        warnings.append("Top and bottom depths are identical; depth range is zero.")
    elif pix_span > 0:
        depth_per_pixel = depth_span / pix_span
        # Heuristic similar to the reference compute_depth_scale usage:
        # flag unusual but not impossible scales outside ~0.1–10 depth
        # units per pixel so the user can double-check anchors.
        if abs(depth_per_pixel) < 0.1 or abs(depth_per_pixel) > 10.0:
            warnings.append(
                f"Unusual depth scale (~{depth_per_pixel:.2f} depth units per pixel). Check anchors."
            )

    if image_height and (top_px < 0 or bottom_px > image_height):
        warnings.append(f"Depth pixels ({top_px:.0f}–{bottom_px:.0f}) are outside image bounds (0–{image_height - 1}).")

    return warnings

    for c in curves_cfg:
        curve_type = (c.get('type') or '').upper()
        mnemonic = (c.get('las_mnemonic') or c.get('name') or '').upper()
        if not mnemonic or mnemonic not in las_curve_data:
            continue

        meta = las_curve_data.get(mnemonic) or {}
        vals = np.asarray(meta.get("values"), dtype=np.float32)
        if vals.size == 0:
            continue

        valid_mask = vals != null_val
        if not np.any(valid_mask):
            continue

        vals_valid = vals[valid_mask]
        vmin = float(np.nanmin(vals_valid))
        vmax = float(np.nanmax(vals_valid))

        median = float(np.nanmedian(vals_valid))
        std = float(np.nanstd(vals_valid))
        null_pct = 100.0 * (1.0 - float(np.count_nonzero(valid_mask)) / float(vals.size))

        # Decide expected range based on curve type / mnemonic
        low, high = None, None
        if curve_type == 'GR' or mnemonic == 'GR':
            low, high = 0.0, 300.0  # API units
        elif curve_type == 'RHOB' or mnemonic == 'RHOB':
            low, high = 1.7, 3.0    # g/cc
        elif curve_type in ('DT', 'DTC') or mnemonic in ('DT', 'DTC'):
            low, high = 40.0, 200.0 # us/ft

        if low is None or high is None:
            continue

        issues = []
        if vmin < low:
            issues.append(f"min {vmin:.2f} < {low}")
        if vmax > high:
            issues.append(f"max {vmax:.2f} > {high}")

        span = high - low
        dyn_range = vmax - vmin
        if span > 0 and dyn_range < 0.05 * span:
            issues.append(f"curve is very flat (range {dyn_range:.2f})")

        if null_pct > 40.0:
            issues.append(f"{null_pct:.0f}% of samples are null")

        if issues:
            label = c.get('display_name') or mnemonic or curve_type or 'curve'
            summary = (
                f"{label}: {', '.join(issues)} "
                f"(min={vmin:.2f}, max={vmax:.2f}, median={median:.2f}, std={std:.2f}, null≈{null_pct:.0f}%). "
                f"Expected roughly {low}–{high}."
            )
            warnings.append(summary)

    return warnings


def auto_detect_tracks(image_array):
    """Auto-detect track boundaries."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    vertical_sum = np.sum(edges, axis=0)

    # Simple peak detection
    threshold = np.max(vertical_sum) * 0.3
    peaks = []
    for i in range(1, len(vertical_sum)-1):
        if vertical_sum[i] > threshold and vertical_sum[i] > vertical_sum[i-1] and vertical_sum[i] > vertical_sum[i+1]:
            peaks.append(i)

    # Filter peaks that are too close to each other (e.g., edges of the same thick line)
    w = image_array.shape[1]
    min_dist = w * 0.05  # Assume a track is at least 5% of image width

    filtered_peaks = []
    for p in peaks:
        if not filtered_peaks or (p - filtered_peaks[-1]) >= min_dist:
            filtered_peaks.append(p)

    # Group into tracks (a track is between two consecutive lines)
    if len(filtered_peaks) >= 2:
        tracks = [(int(filtered_peaks[i]), int(filtered_peaks[i + 1])) for i in range(len(filtered_peaks) - 1)]
    else:
        # Fallback: divide into 3 equal sections
        section_width = w // 3
        tracks = [(int(i * section_width), int((i + 1) * section_width)) for i in range(3)]

    return tracks


def select_primary_track_region(tracks, image_width):
    """Cluster tracks into horizontal panels and select the best one.

    This is used when the TIFF contains multiple side-by-side copies of the
    same log. We group tracks by gaps between their horizontal centers and
    then pick the widest, densest panel as the default region to use.
    """
    if not tracks:
        return None

    centers = []
    widths = []
    for left, right in tracks:
        try:
            l = int(left)
            r = int(right)
        except Exception:
            continue
        width = max(1, r - l)
        widths.append(width)
        centers.append(0.5 * (l + r))

    if not widths or not centers:
        return None

    # Sort tracks by horizontal center and compute gaps between neighbors.
    sorted_indices = sorted(range(len(tracks)), key=lambda i: centers[i])
    gaps = []
    for idx in range(len(sorted_indices) - 1):
        c0 = centers[sorted_indices[idx]]
        c1 = centers[sorted_indices[idx + 1]]
        gaps.append(c1 - c0)

    if gaps:
        median_gap = float(np.median(gaps))
        gap_threshold = max(median_gap * 2.5, 40.0)
    else:
        gap_threshold = max(int(image_width * 0.25), 40)

    panels = []
    current = []
    last_center = None
    for idx in sorted_indices:
        center = centers[idx]
        if last_center is not None and (center - last_center) > gap_threshold and current:
            panels.append(current)
            current = []
        current.append(idx)
        last_center = center
    if current:
        panels.append(current)

    if not panels:
        return None

    best_panel = None
    best_score = None
    best_left = None
    best_right = None
    for panel in panels:
        left_vals = []
        right_vals = []
        for i in panel:
            try:
                l = int(tracks[i][0])
                r = int(tracks[i][1])
            except Exception:
                continue
            left_vals.append(l)
            right_vals.append(r)
        if not left_vals or not right_vals:
            continue
        left = min(left_vals)
        right = max(right_vals)
        total_width = max(1, right - left)
        score = total_width * len(panel)
        if best_panel is None or score > best_score:
            best_panel = panel
            best_score = score
            best_left = left
            best_right = right

    if best_panel is None:
        return None

    return {
        "left_px": int(best_left),
        "right_px": int(best_right),
        "track_indices": best_panel,
    }

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/')
def index():
    if _current_user(require_access=False):
        return redirect(url_for('dashboard'))
    return render_template('index.html',
                          version=APP_VERSION,
                          build_time=APP_BUILD_TIME,
                          vision_available=VISION_API_AVAILABLE)


@app.route('/pricing')
def pricing():
    user = _current_user(require_access=False)
    self_service = {
        'eyebrow': 'Pricing & Membership',
        'title': 'Self-Serve Workspace',
        'description': 'Access the AI digitization suite to convert your own well logs with high accuracy and speed. Includes a 7-day free trial.',
        'features': [
            'Upload image and map depth limits',
            'AI detects tracks and snaps to curves',
            'Review and correction tools',
            'LAS export',
            'Saved projects',
            'Account dashboard',
        ],
        'cta': 'Start Free Trial' if not user else 'Submit a Job',
        'href': '/signup' if not user else '/submit-job',
    }
    managed_processing = {
        'eyebrow': 'Done-for-you option',
        'title': 'Full-Service Conversion',
        'description': 'Send us your logs and we’ll process them for you with review, correction, and final QA built into the workflow.',
        'price_lines': [
            '$0.49 per 100 curve-feet',
            '$29.99 minimum per log',
        ],
        'notes': [
            'Poor scans, overlapping traces, wraps, and heavy cleanup may require additional review.',
            'Pricing scales with extracted curve volume',
            'Best for teams that want finished output',
            'Human review stays in the loop',
            'Quoted separately for highly difficult logs',
        ],
        'cta': 'Request a Quote',
        'href': '/managed-conversion',
    }
    return render_template(
        'pricing.html',
        user=user,
        self_service=self_service,
        managed_processing=managed_processing,
        current_plan_label=auth_billing.plan_label(user.get('plan_code')) if user else None,
    )


@app.route('/api/managed-jobs/upload-url', methods=['POST'])
def generate_upload_url():
    """Generate a presigned URL to securely upload a file directly to Google Cloud Storage."""
    data = request.json
    filename = data.get('filename')
    content_type = data.get('contentType')
    
    if not content_type:
        content_type = 'application/octet-stream'
        
    if not filename:
        return jsonify({'success': False, 'error': 'Missing filename'}), 400

    # Ensure Vision API/Cloud credentials exist to use GCS
    if not VISION_API_AVAILABLE:
        return jsonify({'success': False, 'error': 'Cloud storage is not configured.'}), 500

    try:
        # Create a storage client using the exact same credentials loaded for Vision OCR
        global credentials
        if credentials:
            storage_client = storage.Client(credentials=credentials)
        else:
            storage_client = storage.Client()
        
        bucket_name = config.GCS_UPLOADS_BUCKET
        bucket = storage_client.bucket(bucket_name)
        
        # Generate a unique path for the file: uploads/{uuid}/{filename}
        file_uuid = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")
        blob_path = f"uploads/{file_uuid}/{safe_filename}"
        
        blob = bucket.blob(blob_path)
        
        # Generate a presigned URL valid for 30 minutes for a PUT request
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=30),
            method="PUT",
            content_type=content_type,
        )
        
        return jsonify({
            'success': True, 
            'uploadUrl': url, 
            'fileKey': blob_path
        })
    except Exception as e:
        print(f"Failed to generate presigned URL: {e}")
        return jsonify({'success': False, 'error': 'Failed to generate upload URL.'}), 500


MANAGED_JOB_FILES_MARKER = "\n\nFiles: "
MANAGED_JOB_ADMIN_FILES_MARKER = "\n\nAdmin Files: "


def _extract_managed_job_json_section(notes: str, marker: str) -> list:
    if not notes or marker not in notes:
        return []

    start = notes.find(marker) + len(marker)
    end = len(notes)
    for next_marker in (MANAGED_JOB_FILES_MARKER, MANAGED_JOB_ADMIN_FILES_MARKER):
        if next_marker == marker:
            continue
        pos = notes.find(next_marker, start)
        if pos != -1:
            end = min(end, pos)

    try:
        parsed = json.loads(notes[start:end].strip())
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _managed_job_note_body(notes: str) -> str:
    if not notes:
        return ""
    cut_points = [
        pos for pos in (
            notes.find(MANAGED_JOB_FILES_MARKER),
            notes.find(MANAGED_JOB_ADMIN_FILES_MARKER),
        )
        if pos != -1
    ]
    if not cut_points:
        return notes
    return notes[:min(cut_points)].rstrip()


def _build_managed_job_notes(notes: str, source_files: list, admin_files: list | None = None) -> str:
    body = _managed_job_note_body(notes)
    updated = f"{body}{MANAGED_JOB_FILES_MARKER}{json.dumps(source_files or [])}"
    if admin_files:
        updated += f"{MANAGED_JOB_ADMIN_FILES_MARKER}{json.dumps(admin_files)}"
    return updated


def _managed_job_file_key(file_ref) -> str:
    if isinstance(file_ref, str):
        return file_ref
    if isinstance(file_ref, dict):
        return file_ref.get('gcs_key') or file_ref.get('key') or ''
    return ''


def _managed_job_file_name(file_ref, file_key: str) -> str:
    if isinstance(file_ref, dict) and file_ref.get('name'):
        return file_ref['name']
    return file_key.split("/")[-1] if file_key else 'file'


@app.route('/api/managed-jobs/checkout', methods=['POST'])
def create_managed_job_checkout():
    """Create a Stripe Checkout Session in setup mode for a managed job."""
    if not _is_stripe_configured():
        return jsonify({'success': False, 'error': 'Stripe is not configured'}), 500

    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'Invalid payload'}), 400

    email = data.get('email', '').strip()
    if not email:
        return jsonify({'success': False, 'error': 'Email is required'}), 400

    # 1. Create or find Stripe Customer
    try:
        customers = stripe.Customer.list(email=email, limit=1).data
        if customers:
            customer_id = customers[0].id
        else:
            new_customer = stripe.Customer.create(
                email=email,
                name=data.get('contactName', ''),
                metadata={'company_name': data.get('companyName', '')}
            )
            customer_id = new_customer.id
    except stripe.error.StripeError as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    # 2. Create managed job record in DB
    job_id = str(uuid.uuid4())
    user = _current_user(require_access=False)
    user_id = user['id'] if user else None

    # Generate a quick estimate using the user's data to save it in DB
    try:
        with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
            conn.execute("""
                INSERT INTO managed_jobs (
                    id, user_id, stripe_customer_id, company_name, contact_name, email,
                    project_name, well_name, estimated_depth_feet, estimated_curve_count,
                    estimated_complexity, estimated_turnaround, estimated_units,
                    estimated_amount, notes, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)
            """, (
                job_id, user_id, customer_id,
                data.get('companyName'), data.get('contactName'), email,
                data.get('projectName'), data.get('wellName'),
                float(data.get('depthFeet') or 0),
                int(data.get('curveCount') or 0),
                data.get('complexity'), data.get('turnaround'),
                float(data.get('estimatedUnits') or 0),
                float(data.get('estimatedTotal') or 0),
                _build_managed_job_notes(data.get('notes', ''), data.get('files', [])),
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat()
            ))
    except Exception as e:
        return jsonify({'success': False, 'error': f"DB Error: {e}"}), 500

    # 3. Create Stripe Checkout Session in setup mode
    try:
        session = stripe.checkout.Session.create(
            mode='setup',
            customer=customer_id,
            payment_method_types=['card'],
            success_url=f"{config.APP_BASE_URL}/submit-job/success?job_id={job_id}&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{config.APP_BASE_URL}/submit-job",
            metadata={
                'job_id': job_id,
                'job_type': 'managed_conversion'
            }
        )
        
        # Update DB with session ID
        with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
            conn.execute("UPDATE managed_jobs SET stripe_checkout_session_id = ? WHERE id = ?", (session.id, job_id))

        return jsonify({'success': True, 'checkoutUrl': session.url})
    except stripe.error.StripeError as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/submit-job/success', methods=['GET'])
def managed_job_success():
    """Handle successful return from Stripe setup checkout."""
    job_id = request.args.get('job_id')
    session_id = request.args.get('session_id')
    
    if not job_id or not session_id:
        flash('Missing job or session ID', 'error')
        return redirect(url_for('submit_job'))

    try:
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        if checkout_session.setup_intent:
            setup_intent = stripe.SetupIntent.retrieve(checkout_session.setup_intent)
            payment_method_id = setup_intent.payment_method

            with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
                conn.execute("""
                    UPDATE managed_jobs
                    SET stripe_payment_method_id = ?,
                        status = 'payment_method_saved',
                        updated_at = ?
                    WHERE id = ? AND stripe_checkout_session_id = ?
                """, (
                    payment_method_id,
                    datetime.now(timezone.utc).isoformat(),
                    job_id,
                    session_id
                ))
                job = conn.execute("SELECT * FROM managed_jobs WHERE id = ?", (job_id,)).fetchone()
            
            if job and config.MAIL_FROM:
                file_refs = _extract_managed_job_json_section(job['notes'], MANAGED_JOB_FILES_MARKER)
                
                file_urls = []
                if file_refs and VISION_API_AVAILABLE:
                    try:
                        global credentials
                        if 'credentials' in globals() and credentials:
                            storage_client = storage.Client(credentials=credentials)
                        else:
                            storage_client = storage.Client()
                        bucket = storage_client.bucket(config.GCS_UPLOADS_BUCKET)
                        for file_ref in file_refs:
                            fk = _managed_job_file_key(file_ref)
                            if not fk:
                                continue
                            blob = bucket.blob(fk)
                            url = blob.generate_signed_url(version="v4", expiration=timedelta(days=7), method="GET")
                            file_urls.append({'key': fk, 'url': url})
                    except Exception as e:
                        print(f"Error generating signed urls: {e}")
                
                try:
                    mailer.send_managed_job_admin(config.MAIL_FROM, dict(job), file_urls)
                except Exception as e:
                    print(f"Failed to send admin notification: {e}")
    except Exception as e:
        print(f"Error completing managed job checkout: {e}")
        flash('Error verifying your payment method.', 'error')

    return render_template('submit_job_success.html', job_id=job_id)


@app.route('/submit-job')
def submit_job():
    """React-based managed job submission flow."""
    user = _current_user(require_access=False)
    return render_template('submit_job.html', user=user, stripe_ready=_is_stripe_configured())


@app.route('/managed-conversion')
def managed_conversion():
    user = _current_user(require_access=False)
    managed_tiers = [
        {
            'name': 'Simple',
            'price': 'From $29.99',
            'note': 'Usually 3 curves or fewer',
            'features': [
                'Clean scan',
                'Standard turnaround',
                'Basic cleanup',
            ],
        },
        {
            'name': 'Standard',
            'price': '$39.99-$79.99',
            'note': 'Most routine logs',
            'features': [
                'Multiple curves',
                'Moderate cleanup',
                'Final LAS delivery',
            ],
        },
        {
            'name': 'Difficult',
            'price': 'Custom quote',
            'note': 'Messy, faint, wrapped, or unusual logs',
            'features': [
                'Complex cleanup',
                'Manual review',
                'Quoted by complexity',
            ],
        },
    ]
    return render_template(
        'managed_conversion.html',
        user=user,
        managed_tiers=managed_tiers,
    )


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login against persisted user accounts."""
    error = None
    next_url = request.args.get('next')
    if request.method == 'POST':
        next_url = request.form.get('next') or next_url
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password')
        remember = 'remember-me' in request.form

        # Admin backdoor for testing without Stripe
        if email == 'admin@tiflas.com' and password == 'password':
            session.clear()  # prevent session fixation
            session['admin_override'] = True
            session.permanent = remember
            resp = redirect(next_url or url_for('dashboard'))
            if remember:
                resp.set_cookie(
                    REMEMBER_COOKIE_NAME, _create_remember_token({'admin': True}),
                    max_age=REMEMBER_COOKIE_DAYS * 24 * 3600,
                    httponly=True, samesite='Lax',
                )
            return resp

        user = auth_billing.get_user_by_email(config.AUTH_DB_PATH, email)
        if not user or not check_password_hash(user['password_hash'], password or ''):
            error = 'Invalid email or password'
        else:
            session.clear()  # prevent session fixation
            session['user_id'] = user['id']
            session['is_admin'] = user.get('is_admin', 0)
            session.permanent = remember

            if auth_billing.subscription_access_allowed(user):
                dest = next_url or url_for('dashboard')
            else:
                flash('Start your trial or choose a plan to access the app.', 'info')
                dest = url_for('account')

            resp = redirect(dest)
            if remember:
                resp.set_cookie(
                    REMEMBER_COOKIE_NAME, _create_remember_token({'user_id': user['id']}),
                    max_age=REMEMBER_COOKIE_DAYS * 24 * 3600,
                    httponly=True, samesite='Lax',
                )
            return resp
            
    return render_template('login.html', error=error, next_url=next_url)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Send a password reset email."""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = auth_billing.get_user_by_email(config.AUTH_DB_PATH, email)
        if user:
            token = secrets.token_urlsafe(32)
            expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            auth_billing.update_user_fields(config.AUTH_DB_PATH, user['id'],
                                            reset_token=token, reset_token_expires=expires)
            reset_url = f"{config.APP_BASE_URL}/reset-password?token={token}"
            mailer.send_password_reset(email, reset_url)
        flash('If that email is registered, a reset link has been sent.', 'success')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    """Validate reset token and set new password."""
    token = request.args.get('token') or request.form.get('token', '')
    if request.method == 'POST':
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        if not password or len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return render_template('reset_password.html', token=token, valid_token=True)
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html', token=token, valid_token=True)
        with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
            row = conn.execute(
                "SELECT id, reset_token_expires FROM users WHERE reset_token = ?", (token,)
            ).fetchone()
        if not row:
            flash('Invalid or expired reset link.', 'error')
            return render_template('reset_password.html', token=token, valid_token=False)
        expires = row['reset_token_expires']
        if not expires or datetime.fromisoformat(expires) < datetime.now(timezone.utc):
            flash('This reset link has expired. Please request a new one.', 'error')
            return render_template('reset_password.html', token=token, valid_token=False)
        auth_billing.update_user_fields(config.AUTH_DB_PATH, row['id'],
                                        password_hash=generate_password_hash(password),
                                        reset_token=None, reset_token_expires=None)
        flash('Password updated. You can now sign in.', 'success')
        return redirect(url_for('login'))

    with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, reset_token_expires FROM users WHERE reset_token = ?", (token,)
        ).fetchone()
    valid = bool(row and row['reset_token_expires'] and
                 datetime.fromisoformat(row['reset_token_expires']) >= datetime.now(timezone.utc))
    return render_template('reset_password.html', token=token, valid_token=valid)


@app.route('/logout')
def logout():
    """Handle user logout"""
    raw_token = request.cookies.get(REMEMBER_COOKIE_NAME)
    if raw_token:
        auth_billing.delete_remember_token(config.AUTH_DB_PATH, raw_token)
    session.clear()
    resp = redirect(url_for('index'))
    resp.delete_cookie(REMEMBER_COOKIE_NAME)
    return resp

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Create a real user account (password-hashed, persisted in SQLite)."""
    error = None
    managed_mode = request.args.get('managed') == 'true'
    
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        company_name = request.form.get('company_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not full_name or not company_name or not email or len(password) < 8:
            error = 'Please complete all fields. Password must be at least 8 characters.'
        elif auth_billing.get_user_by_email(config.AUTH_DB_PATH, email):
            error = 'An account with that email already exists.'
        else:
            user_id = auth_billing.create_user(
                config.AUTH_DB_PATH,
                email=email,
                password_hash=generate_password_hash(password),
                full_name=full_name,
                company_name=company_name,
            )
            session['user_id'] = user_id
            
            if managed_mode:
                # Managed jobs users skip the trial and subscription flow.
                # Give them a 'managed_only' plan to differentiate them.
                auth_billing.update_user_fields(
                    config.AUTH_DB_PATH,
                    user_id,
                    subscription_status='managed_only',
                    plan_code='managed_only'
                )
                flash('Account created! Welcome to the managed jobs dashboard.', 'success')
                return redirect(url_for('dashboard'))
            
            # Regular self-serve path: start trial immediately
            trial_end_iso = (datetime.now(timezone.utc) + timedelta(days=auth_billing.TRIAL_DAYS)).isoformat()
            auth_billing.update_user_fields(
                config.AUTH_DB_PATH,
                user_id,
                subscription_status='trialing',
                trial_started_at=datetime.now(timezone.utc).isoformat(),
                trial_ends_at=trial_end_iso,
            )
            
            # Send welcome email to new user
            mailer.send_welcome(email, full_name)
            
            # Notify admin of new signup
            if config.MAIL_FROM:
                mailer.send_new_signup_admin(config.MAIL_FROM, email, full_name, company_name)
            
            if _is_stripe_configured():
                return redirect(url_for('create_checkout_session', plan='monthly', mode='trial'))
            flash('Account created! Your 7-day free trial is now active.', 'success')
            return redirect(url_for('dashboard'))

    return render_template('signup.html', error=error, managed_mode=managed_mode)


@app.route('/auth-debug')
def auth_debug():
    """Diagnostic route — shows cookie and session state for remember-me debugging."""
    raw_token = request.cookies.get(REMEMBER_COOKIE_NAME)
    decoded = _decode_remember_token(raw_token) if raw_token else None
    info = {
        'session_keys': list(session.keys()),
        'session_permanent': session.permanent,
        'remember_cookie_present': bool(raw_token),
        'remember_cookie_preview': (raw_token[:20] + '...') if raw_token else None,
        'token_decode_ok': decoded is not None,
        'token_payload': decoded,
        'all_cookie_names': list(request.cookies.keys()),
        'secret_key_prefix': config.SECRET_KEY[:8] + '...' if config.SECRET_KEY else 'NOT SET',
    }
    return jsonify(info)



@app.route('/account')
def account():
    user = _current_user(require_access=False)
    if not user:
        return redirect(url_for('login'))

    trial_countdown = auth_billing.compute_trial_countdown(user)
    trial_eligibility = auth_billing.trial_eligibility(config.AUTH_DB_PATH, user)
    invoices = []
    payment_method = None
    subscription_cancel_at_period_end = False

    if _is_stripe_configured() and user.get('stripe_customer_id'):
        try:
            invoices_resp = stripe.Invoice.list(customer=user['stripe_customer_id'], limit=12)
            invoices = auth_billing.serialize_invoices(list(invoices_resp.data))

            customer = stripe.Customer.retrieve(
                user['stripe_customer_id'],
                expand=['invoice_settings.default_payment_method'],
            )
            default_pm = customer.get('invoice_settings', {}).get('default_payment_method')
            if default_pm:
                payment_method = {
                    'brand': default_pm.get('card', {}).get('brand', '').upper(),
                    'last4': default_pm.get('card', {}).get('last4', ''),
                    'exp_month': default_pm.get('card', {}).get('exp_month', ''),
                    'exp_year': default_pm.get('card', {}).get('exp_year', ''),
                }

            if user.get('stripe_subscription_id'):
                sub = stripe.Subscription.retrieve(user['stripe_subscription_id'])
                subscription_cancel_at_period_end = bool(sub.get('cancel_at_period_end'))
        except Exception as exc:
            flash(f'Billing data temporarily unavailable: {exc}', 'warning')

    return render_template(
        'account.html',
        user=user,
        trial_countdown=trial_countdown,
        trial_eligibility=trial_eligibility,
        current_plan_label=auth_billing.plan_label(user.get('plan_code')),
        can_manage_billing=bool(user.get('stripe_customer_id')),
        billing_ready=_is_stripe_configured(),
        invoices=invoices,
        payment_method=payment_method,
        cancel_at_period_end=subscription_cancel_at_period_end,
    )


@app.route('/account/update', methods=['POST'])
def update_account():
    user = _current_user(require_access=False)
    if not user:
        return redirect(url_for('login'))
        
    full_name = request.form.get('full_name', '').strip()
    company_name = request.form.get('company_name', '').strip()
    
    if not full_name or not company_name:
        flash('Full Name and Company Name are required.', 'error')
        return redirect(url_for('account'))
        
    try:
        # We can re-use update_user_fields from auth_billing
        auth_billing.update_user_fields(
            config.AUTH_DB_PATH, 
            user['id'],
            full_name=full_name,
            company_name=company_name,
            company_name_normalized=" ".join(company_name.lower().split())
        )
        flash('Profile updated successfully.', 'success')
    except Exception as e:
        flash('Failed to update profile.', 'error')
        
    return redirect(url_for('account'))


@app.route('/admin')
@login_required()
def admin():
    """Admin panel."""
    user = _current_user(require_access=True)
    if not user.get('is_admin') and not session.get('is_admin'):
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))
        
    users = auth_billing.get_all_users_for_admin(config.AUTH_DB_PATH)
    logs = auth_billing.get_all_logs_for_admin(config.AUTH_DB_PATH)
    managed_jobs = auth_billing.get_all_managed_jobs_for_admin(config.AUTH_DB_PATH)
    stats = auth_billing.get_admin_stats(config.AUTH_DB_PATH)
    settings = auth_billing.get_admin_settings(config.AUTH_DB_PATH)
    
    # Determine which user we are impersonating, if any
    impersonating_id = session.get('impersonate_user_id')
    
    return render_template('admin.html', user=user, users=users, logs=logs, managed_jobs=managed_jobs, stats=stats, settings=settings, impersonating_id=impersonating_id)


@app.route('/api/admin/managed-jobs/<job_id>/files', methods=['GET'])
@login_required()
def admin_managed_job_files(job_id):
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
        job = conn.execute("SELECT * FROM managed_jobs WHERE id = ?", (job_id,)).fetchone()
        
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
        
    try:
        file_refs = _extract_managed_job_json_section(job['notes'], MANAGED_JOB_FILES_MARKER)
        
        urls = []
        if file_refs and VISION_API_AVAILABLE:
            global credentials
            if 'credentials' in globals() and credentials:
                storage_client = storage.Client(credentials=credentials)
            else:
                storage_client = storage.Client()
            bucket = storage_client.bucket(config.GCS_UPLOADS_BUCKET)
            for file_ref in file_refs:
                fk = _managed_job_file_key(file_ref)
                if not fk:
                    continue
                blob = bucket.blob(fk)
                url = blob.generate_signed_url(version="v4", expiration=timedelta(hours=1), method="GET")
                filename = _managed_job_file_name(file_ref, fk)
                urls.append({'filename': filename, 'url': url})
                
        return jsonify({'success': True, 'urls': urls})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/managed-jobs/<job_id>/delivery-upload-url', methods=['POST'])
@login_required()
def admin_managed_job_delivery_upload_url(job_id):
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403

    data = request.json or {}
    filename = (data.get('filename') or '').strip()
    content_type = data.get('contentType') or 'application/octet-stream'

    if not filename:
        return jsonify({'success': False, 'error': 'Missing filename'}), 400
    if not VISION_API_AVAILABLE:
        return jsonify({'success': False, 'error': 'Cloud storage is not configured.'}), 500

    with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
        job = conn.execute("SELECT id FROM managed_jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404

    try:
        global credentials
        if 'credentials' in globals() and credentials:
            storage_client = storage.Client(credentials=credentials)
        else:
            storage_client = storage.Client()

        bucket = storage_client.bucket(config.GCS_UPLOADS_BUCKET)
        safe_filename = re.sub(r'[^A-Za-z0-9._-]+', '_', filename).strip('._') or 'file'
        blob_path = f"deliveries/{job_id}/{uuid.uuid4()}/{safe_filename}"
        blob = bucket.blob(blob_path)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=30),
            method="PUT",
            content_type=content_type,
        )
        return jsonify({'success': True, 'uploadUrl': url, 'fileKey': blob_path})
    except Exception as e:
        print(f"Failed to generate admin delivery upload URL: {e}")
        return jsonify({'success': False, 'error': 'Failed to generate upload URL.'}), 500


@app.route('/api/admin/managed-jobs/<job_id>/delivery-files', methods=['GET', 'POST'])
@login_required()
def admin_managed_job_delivery_files(job_id):
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403

    with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
        job = conn.execute("SELECT * FROM managed_jobs WHERE id = ?", (job_id,)).fetchone()

    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404

    if request.method == 'POST':
        data = request.json or {}
        new_files = data.get('files') or []
        if not isinstance(new_files, list) or not new_files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400

        normalized_new_files = []
        for file_ref in new_files:
            if not isinstance(file_ref, dict):
                continue
            file_key = _managed_job_file_key(file_ref)
            if not file_key:
                continue
            normalized_new_files.append({
                'name': file_ref.get('name') or file_key.split('/')[-1],
                'size': int(file_ref.get('size') or 0),
                'type': file_ref.get('type') or 'application/octet-stream',
                'gcs_key': file_key,
                'uploaded_at': datetime.now(timezone.utc).isoformat(),
            })

        if not normalized_new_files:
            return jsonify({'success': False, 'error': 'No valid files provided'}), 400

        source_files = _extract_managed_job_json_section(job['notes'], MANAGED_JOB_FILES_MARKER)
        admin_files = _extract_managed_job_json_section(job['notes'], MANAGED_JOB_ADMIN_FILES_MARKER)
        updated_notes = _build_managed_job_notes(
            job['notes'],
            source_files,
            admin_files + normalized_new_files,
        )

        with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
            conn.execute(
                "UPDATE managed_jobs SET notes = ?, updated_at = ? WHERE id = ?",
                (updated_notes, datetime.now(timezone.utc).isoformat(), job_id),
            )

        return jsonify({'success': True, 'uploaded': len(normalized_new_files)})

    try:
        file_refs = _extract_managed_job_json_section(job['notes'], MANAGED_JOB_ADMIN_FILES_MARKER)
        urls = []
        if file_refs and VISION_API_AVAILABLE:
            global credentials
            if 'credentials' in globals() and credentials:
                storage_client = storage.Client(credentials=credentials)
            else:
                storage_client = storage.Client()
            bucket = storage_client.bucket(config.GCS_UPLOADS_BUCKET)
            for file_ref in file_refs:
                fk = _managed_job_file_key(file_ref)
                if not fk:
                    continue
                blob = bucket.blob(fk)
                url = blob.generate_signed_url(version="v4", expiration=timedelta(hours=1), method="GET")
                urls.append({'filename': _managed_job_file_name(file_ref, fk), 'url': url})
        return jsonify({'success': True, 'urls': urls})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/managed-jobs/charge', methods=['POST'])
@login_required()
def charge_managed_job():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    data = request.json or {}
    job_id = data.get('job_id')
    amount_dollars = data.get('amount')
    
    if not job_id or amount_dollars is None:
        return jsonify({'success': False, 'error': 'Missing job_id or amount'}), 400
        
    try:
        amount_cents = int(float(amount_dollars) * 100)
        if amount_cents <= 0:
            return jsonify({'success': False, 'error': 'Invalid amount'}), 400
            
        # Fetch the job
        with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
            job = conn.execute("SELECT * FROM managed_jobs WHERE id = ?", (job_id,)).fetchone()
            
        if not job:
            return jsonify({'success': False, 'error': 'Job not found'}), 404
            
        job = dict(job)
        if job.get('status') == 'paid':
            return jsonify({'success': False, 'error': 'Job is already paid'}), 400
            
        payment_method_id = job.get('stripe_payment_method_id')
        customer_id = job.get('stripe_customer_id')
        
        if not payment_method_id or not customer_id:
            return jsonify({'success': False, 'error': 'No saved payment method found for this job'}), 400
            
        # Charge the card via Stripe PaymentIntent off-session
        intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency='usd',
            customer=customer_id,
            payment_method=payment_method_id,
            off_session=True,
            confirm=True,
            description=f"TifLAS Managed Conversion: {job.get('well_name', 'Well')} - Job ID {job_id[:8]}",
            metadata={'job_id': job_id}
        )
        
        if intent.status == 'succeeded':
            # Update DB
            with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
                conn.execute("""
                    UPDATE managed_jobs 
                    SET status = 'paid', actual_amount = ?, updated_at = ?
                    WHERE id = ?
                """, (float(amount_dollars), datetime.now(timezone.utc).isoformat(), job_id))
            return jsonify({'success': True, 'charge_id': intent.id})
        else:
            return jsonify({'success': False, 'error': f"Charge failed with status: {intent.status}"}), 400
            
    except stripe.error.CardError as e:
        err = e.error
        return jsonify({'success': False, 'error': f"Card error: {err.message}"}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/admin/action', methods=['POST'])
@login_required()
def admin_action():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
    data = request.json or {}
    action = data.get('action')
    
    if action == 'impersonate':
        target_id = data.get('user_id')
        if target_id:
            session['impersonate_user_id'] = int(target_id)
            return jsonify({'success': True, 'message': 'Impersonation started'})
        else:
            return jsonify({'success': False, 'error': 'Missing user_id'})
            
    elif action == 'stop_impersonate':
        session.pop('impersonate_user_id', None)
        return jsonify({'success': True, 'message': 'Impersonation stopped'})
        
    elif action == 'update_setting':
        key = data.get('key')
        val = data.get('value', '')
        if key:
            auth_billing.update_admin_setting(config.AUTH_DB_PATH, key, val)
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Missing key'})
        
    elif action in ['ban', 'unban', 'extend_trial', 'grant_trial', 'make_lifetime', 'delete']:
        target_id = data.get('user_id')
        if target_id:
            try:
                auth_billing.admin_update_user_action(config.AUTH_DB_PATH, int(target_id), action)
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

    elif action == 'mass_unlock':
        try:
            updated = auth_billing.mass_unlock_users(config.AUTH_DB_PATH)
            return jsonify({'success': True, 'updated': updated})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    return jsonify({'success': False, 'error': 'Invalid action'})


@app.route('/api/admin/users', methods=['GET'])
@login_required()
def admin_list_users():
    """List all user accounts with trial status."""
    user = _current_user(require_access=True)
    if not user.get('is_admin') and not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403
    
    users = auth_billing.get_all_users_for_admin(config.AUTH_DB_PATH)
    user_list = []
    for u in users:
        user_list.append({
            'id': u.get('id'),
            'email': u.get('email'),
            'full_name': u.get('full_name'),
            'company_name': u.get('company_name'),
            'subscription_status': u.get('subscription_status'),
            'plan_code': u.get('plan_code'),
            'trial_used': u.get('trial_used'),
            'is_admin': u.get('is_admin'),
            'is_banned': u.get('is_banned'),
            'created_at': u.get('created_at')
        })
    
    return jsonify({
        'total_users': len(user_list),
        'users': user_list
    })


@app.route('/api/admin/test-email', methods=['POST'])
@login_required()
def admin_test_email():
    """Send a test email to verify SMTP configuration."""
    user = _current_user(require_access=True)
    if not user.get('is_admin') and not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403
    
    to = (request.json or {}).get('to') or user.get('email')
    ok = mailer.send_email(
        to,
        subject='TifLAS — SMTP test email',
        html_body='<p>Your TifLAS email configuration is working correctly! 🎉</p>',
        text_body='Your TifLAS email configuration is working correctly!'
    )
    return jsonify({
        'success': ok,
        'to': to,
        'configured': bool(config.MAIL_USERNAME and config.MAIL_PASSWORD),
        'mail_from': config.MAIL_FROM,
        'mail_server': config.MAIL_SERVER,
        'mail_port': config.MAIL_PORT,
    })


@app.route('/api/admin/diagnostics', methods=['GET'])
@login_required()
def admin_diagnostics():
    """Diagnostic endpoint to check database and volume status."""
    user = _current_user(require_access=True)
    if not user.get('is_admin') and not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403
    
    import os
    from pathlib import Path
    
    diagnostics = {
        'database': {},
        'volume': {},
        'logs': {},
        'images': {}
    }
    
    # Check database path and existence
    db_path = config.AUTH_DB_PATH
    diagnostics['database']['path'] = db_path
    diagnostics['database']['exists'] = os.path.exists(db_path)
    if os.path.exists(db_path):
        diagnostics['database']['size_bytes'] = os.path.getsize(db_path)
        diagnostics['database']['readable'] = os.access(db_path, os.R_OK)
        diagnostics['database']['writable'] = os.access(db_path, os.W_OK)
    
    # Check volume mount
    volume_mount = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '')
    diagnostics['volume']['mount_path'] = volume_mount
    diagnostics['volume']['mount_exists'] = os.path.exists(volume_mount) if volume_mount else False
    
    # Check data directory
    data_dir = Path.cwd() / 'data'
    diagnostics['volume']['data_dir'] = str(data_dir)
    diagnostics['volume']['data_dir_exists'] = data_dir.exists()
    
    # Check images directory
    images_dir = data_dir / 'images'
    diagnostics['images']['dir'] = str(images_dir)
    diagnostics['images']['exists'] = images_dir.exists()
    if images_dir.exists():
        try:
            image_files = list(images_dir.glob('*'))
            diagnostics['images']['count'] = len(image_files)
            diagnostics['images']['files'] = [f.name for f in image_files[:10]]  # First 10
        except Exception as e:
            diagnostics['images']['error'] = str(e)
    
    # Check logs in database
    try:
        all_logs = auth_billing.get_all_logs_for_admin(config.AUTH_DB_PATH)
        diagnostics['logs']['total_count'] = len(all_logs)
        diagnostics['logs']['by_user'] = {}
        for log in all_logs:
            user_id = log.get('user_id')
            if user_id not in diagnostics['logs']['by_user']:
                diagnostics['logs']['by_user'][user_id] = []
            diagnostics['logs']['by_user'][user_id].append({
                'id': log.get('id'),
                'name': log.get('name'),
                'created_at': log.get('created_at')
            })
    except Exception as e:
        diagnostics['logs']['error'] = str(e)
    
    return jsonify(diagnostics)


@app.route('/billing/create-checkout-session', methods=['GET', 'POST'])
def create_checkout_session():
    user = _current_user(require_access=False)
    if not user:
        return redirect(url_for('login'))

    if not _is_stripe_configured():
        flash('Stripe is not configured yet. Add Stripe environment variables in Railway.', 'error')
        return redirect(url_for('account'))

    plan = (request.values.get('plan') or '').strip().lower()
    mode = (request.values.get('mode') or 'upgrade').strip().lower()
    if plan not in ('monthly', 'annual', 'managed_simple', 'managed_standard', 'managed_complex'):
        flash('Invalid plan selected.', 'error')
        return redirect(url_for('account'))

    price_id = PLAN_TO_PRICE.get(plan)
    if not price_id:
        flash('Missing Stripe price configuration for that plan.', 'error')
        return redirect(url_for('account'))

    if mode == 'trial':
        eligibility = auth_billing.trial_eligibility(config.AUTH_DB_PATH, user)
        if not eligibility.get('eligible'):
            flash(eligibility.get('reason') or 'Trial is not available for this account.', 'error')
            return redirect(url_for('account'))

    customer_id = user.get('stripe_customer_id')
    try:
        if not customer_id:
            customer = stripe.Customer.create(
                email=user.get('email') or '',
                name=user.get('full_name') or '',
                metadata={
                    'user_id': str(user['id']),
                    'company_name': user.get('company_name') or '',
                },
            )
            customer_id = customer['id']
            auth_billing.update_user_fields(config.AUTH_DB_PATH, user['id'], stripe_customer_id=customer_id)

        if mode == 'managed':
            # One-time payment for managed service
            checkout = stripe.checkout.Session.create(
                mode='payment',
                customer=customer_id,
                line_items=[{'price': price_id, 'quantity': 1}],
                metadata={
                    'user_id': str(user['id']),
                    'plan_code': plan,
                    'mode': mode,
                },
                success_url=f"{config.APP_BASE_URL}/managed-conversion",
                cancel_url=f"{config.APP_BASE_URL}/managed-conversion",
            )
        else:
            # Recurring subscription (monthly, annual, trial)
            subscription_data = {
                'metadata': {
                    'user_id': str(user['id']),
                    'plan_code': plan,
                    'mode': mode,
                }
            }
            if mode == 'trial':
                subscription_data['trial_period_days'] = auth_billing.TRIAL_DAYS
    
            checkout = stripe.checkout.Session.create(
                mode='subscription',
                customer=customer_id,
                line_items=[{'price': price_id, 'quantity': 1}],
                payment_method_collection='always',
                allow_promotion_codes=True,
                metadata={
                    'user_id': str(user['id']),
                    'plan_code': plan,
                    'mode': mode,
                },
                subscription_data=subscription_data,
                success_url=f"{config.APP_BASE_URL}/account?checkout=success",
                cancel_url=f"{config.APP_BASE_URL}/account?checkout=cancel",
            )
        return redirect(checkout.url, code=303)
    except stripe.error.StripeError as exc:
        import traceback; traceback.print_exc()
        flash(f'Stripe error: {getattr(exc, "user_message", None) or str(exc)}', 'error')
        return redirect(url_for('account'))
    except Exception as exc:
        import traceback; traceback.print_exc()
        flash(f'Could not start checkout: {exc}', 'error')
        return redirect(url_for('account'))



@app.route('/billing/portal', methods=['POST'])
def billing_portal():
    user = _current_user(require_access=False)
    if not user:
        return redirect(url_for('login'))
    if not _is_stripe_configured() or not user.get('stripe_customer_id'):
        flash('Billing portal is unavailable until Stripe is configured and a customer exists.', 'error')
        return redirect(url_for('account'))

    portal = stripe.billing_portal.Session.create(
        customer=user['stripe_customer_id'],
        return_url=f"{config.APP_BASE_URL}/account",
    )
    return redirect(portal.url, code=303)



@app.route('/billing/cancel-plan', methods=['POST'])
def cancel_plan():
    user = _current_user(require_access=False)
    if not user:
        return redirect(url_for('login'))
    subscription_id = user.get('stripe_subscription_id')
    if not subscription_id:
        flash('No active subscription to cancel.', 'error')
        return redirect(url_for('account'))

    stripe.Subscription.modify(subscription_id, cancel_at_period_end=True)
    flash('Your plan will cancel at the end of the current billing period.', 'info')
    return redirect(url_for('account'))


def _update_user_from_subscription(user_id: int, subscription_obj) -> None:
    items = subscription_obj.get('items', {}).get('data', [])
    price_id = items[0].get('price', {}).get('id') if items else None
    plan_code = PRICE_TO_PLAN.get(price_id, 'none')
    status = (subscription_obj.get('status') or 'none').lower()
    trial_end = _unix_to_iso(subscription_obj.get('trial_end'))

    auth_billing.update_user_fields(
        config.AUTH_DB_PATH,
        user_id,
        stripe_customer_id=subscription_obj.get('customer'),
        stripe_subscription_id=subscription_obj.get('id'),
        subscription_status=status,
        plan_code=plan_code,
        trial_ends_at=trial_end,
    )
    if status == 'trialing':
        auth_billing.mark_trial_started(config.AUTH_DB_PATH, user_id, trial_end)



@app.route('/billing/webhook', methods=['POST'])
def stripe_webhook():
    if not config.STRIPE_WEBHOOK_SECRET:
        return jsonify({'error': 'webhook secret not configured'}), 400

    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature', '')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, config.STRIPE_WEBHOOK_SECRET)
    except Exception:
        return jsonify({'error': 'invalid webhook signature'}), 400

    event_type = event.get('type')
    obj = event.get('data', {}).get('object', {})

    try:
        if event_type == 'checkout.session.completed' and obj.get('mode') == 'subscription':
            user_id = obj.get('metadata', {}).get('user_id')
            subscription_id = obj.get('subscription')
            if user_id and subscription_id:
                subscription_obj = stripe.Subscription.retrieve(subscription_id)
                _update_user_from_subscription(int(user_id), subscription_obj)

        elif event_type in ('customer.subscription.updated', 'customer.subscription.created'):
            subscription_id = obj.get('id')
            user = auth_billing.get_user_by_subscription_id(config.AUTH_DB_PATH, subscription_id)
            if not user and obj.get('customer'):
                user = auth_billing.get_user_by_customer_id(config.AUTH_DB_PATH, obj.get('customer'))
            if user:
                _update_user_from_subscription(user['id'], obj)

        elif event_type == 'customer.subscription.deleted':
            subscription_id = obj.get('id')
            user = auth_billing.get_user_by_subscription_id(config.AUTH_DB_PATH, subscription_id)
            if user:
                auth_billing.update_user_fields(
                    config.AUTH_DB_PATH,
                    user['id'],
                    subscription_status='canceled',
                    plan_code='none',
                )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    return jsonify({'received': True})


@app.route('/api/logs', methods=['POST'])
@login_required()
def save_log():
    """Save a digitized log to the user's account."""
    user = _current_user(require_access=True)
    if not user:
        return jsonify({'success': False, 'error': 'Not authorized'}), 401
        
    # Enforce trial limits
    if user.get('subscription_status') == 'trialing' and not user.get('is_admin'):
        user_logs = auth_billing.get_user_logs(config.AUTH_DB_PATH, user['id'])
        if len(user_logs) >= 3:
            return jsonify({'success': False, 'error': 'Trial limit reached. You can only save up to 3 logs on the free trial. Please upgrade your account to save more logs.'}), 403

    data = request.json
    
    try:
        import uuid
        passed_log_id = data.get('log_id') or data.get('id')
        name = data.get('name', 'Untitled Log')
        curve_count = data.get('curve_count', 0)
        depth_start = float(data.get('depth_start', 0))
        depth_end = float(data.get('depth_end', 0))
        depth_unit = data.get('depth_unit', 'FT')
        las_content = data.get('las_content', '')
        original_image_path = _storage_safe_image_ref(data.get('original_image_path'))
        cropped_image_path = _storage_safe_image_ref(data.get('cropped_image_path'))

        with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
            # If admin override user_id is 0, we must ensure it exists in the DB to avoid FK constraint errors.
            if user['id'] == 0:
                try:
                    conn.execute("INSERT OR IGNORE INTO users (id, email, password_hash, full_name, company_name, contact_name, domain, phone, verification_token, is_verified, subscription_status, plan_code, stripe_customer_id, stripe_subscription_id, is_admin, created_at, updated_at, is_banned, log_count) VALUES (0, 'admin@tiflas.com', '', 'Admin User', '', '', '', '', '', 1, 'active', 'lifetime_comped', '', '', 1, ?, ?, 0, 0)", (datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()))
                except Exception as e:
                    print("Admin user init error:", e)

            existing_log = None
            if passed_log_id:
                existing_log = conn.execute("SELECT id FROM user_logs WHERE id = ? AND user_id = ?", (str(passed_log_id), user['id'])).fetchone()

            if existing_log:
                log_id = str(passed_log_id)
                conn.execute("""
                    UPDATE user_logs 
                    SET name = ?, curve_count = ?, depth_start = ?, depth_end = ?, depth_unit = ?, las_content = ?, original_image_path = COALESCE(?, original_image_path), cropped_image_path = COALESCE(?, cropped_image_path), updated_at = ?
                    WHERE id = ? AND user_id = ?
                """, (
                    name, curve_count, depth_start, depth_end, depth_unit, las_content, original_image_path, cropped_image_path,
                    datetime.now(timezone.utc).isoformat(),
                    log_id, user['id']
                ))
            else:
                log_id = str(passed_log_id) if passed_log_id else str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO user_logs (id, user_id, name, curve_count, depth_start, depth_end, depth_unit, las_content, original_image_path, cropped_image_path, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_id, user['id'], name, curve_count, depth_start, depth_end, depth_unit, las_content, original_image_path, cropped_image_path,
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat()
                ))
            conn.commit()
            
        print(f"[SAVE LOG] Successfully saved log {log_id} for user {user['id']}: {name}")

        try:
            mailer.send_log_saved(
                to=user['email'],
                full_name=user.get('full_name') or user['email'],
                log_name=name,
                curve_count=curve_count,
                depth_start=depth_start,
                depth_end=depth_end,
                depth_unit=depth_unit,
                log_id=log_id,
            )
        except Exception as mail_err:
            print(f"[SAVE LOG] Email notification failed (non-fatal): {mail_err}")

        return jsonify({'success': True, 'log_id': log_id, 'confirmed': True, 'log_name': name})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/logs/<log_id>/download', methods=['GET'])
@login_required()
def download_log(log_id):
    """Download a saved log as a .las file."""
    user = _current_user(require_access=True)
    log_data = auth_billing.get_user_log(config.AUTH_DB_PATH, log_id, user['id'])
    
    if not log_data:
        return "Log not found", 404
        
    filename = f"{log_data['name'].replace(' ', '_')}.las"
    
    return Response(
        log_data['las_content'],
        mimetype='text/plain',
        headers={'Content-Disposition': f'attachment;filename={filename}'}
    )


@app.route('/api/logs/<log_id>/image', methods=['GET'])
@login_required()
def get_log_image(log_id):
    """Serve the cropped image (if available) or original image for a saved log, regardless of how the
    image path was stored: inline data URL, /api/images/<file>
    server path, an absolute http(s) URL, or a GCS object key.
    """
    user = _current_user(require_access=True)
    if not user:
        return "Not authorized", 401

    log_data = auth_billing.get_user_log(config.AUTH_DB_PATH, log_id, user['id'])
    if not log_data:
        return "Log not found", 404

    # Prefer cropped_image_path if present, else original_image_path
    path = (log_data.get('cropped_image_path') or log_data.get('original_image_path') or '').strip()
    if not path:
        return "No image stored for this log", 404

    # 1. Inline data URL → decode and serve directly
    if path.startswith('data:'):
        try:
            header, b64 = path.split(',', 1)
            mime = header[5:].split(';', 1)[0] or 'image/jpeg'
            raw = base64.b64decode(b64)
            return Response(raw, mimetype=mime)
        except Exception as exc:
            print(f"[get_log_image] Failed to decode data URL for {log_id}: {exc}")
            return "Image decode failed", 500

    # 2. Absolute http(s) URL → redirect
    if path.startswith('http://') or path.startswith('https://'):
        return redirect(path)

    # 3. Server-hosted /api/images/<filename> path → read from disk
    if path.startswith('/api/images/'):
        filename = path.rsplit('/', 1)[-1]
        from pathlib import Path
        images_dir = Path(config.DATA_ROOT) / 'images'
        file_path = images_dir / filename
        if not file_path.exists():
            print(f"[get_log_image] Missing file on disk: {file_path}")
            return "Image file missing", 404
        return send_from_directory(str(images_dir), filename)

    # 4. Otherwise treat as a GCS object key → stream from the bucket
    try:
        bucket_name = getattr(config, 'GCS_UPLOADS_BUCKET', None)
        if not bucket_name:
            return "Cloud storage not configured", 500
        global credentials
        storage_client = (
            storage.Client(credentials=credentials) if credentials else storage.Client()
        )
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(path)
        if not blob.exists():
            print(f"[get_log_image] GCS blob missing: {bucket_name}/{path}")
            return "Image not found in cloud storage", 404
        blob.reload()
        mime = blob.content_type or 'application/octet-stream'
        raw = blob.download_as_bytes()
        return Response(raw, mimetype=mime)
    except Exception as exc:
        print(f"[get_log_image] GCS fetch failed for {log_id}: {exc}")
        return "Image fetch failed", 500


@app.route('/api/logs/<log_id>/rename', methods=['POST'])
@login_required()
def rename_log(log_id):
    """Rename a saved log."""
    user = _current_user(require_access=True)
    if not user:
        return jsonify({'success': False, 'error': 'Not authorized'}), 401

    data = request.json
    new_name = (data.get('name') or '').strip()
    if not new_name:
        return jsonify({'success': False, 'error': 'Name cannot be empty'}), 400

    log_data = auth_billing.get_user_log(config.AUTH_DB_PATH, log_id, user['id'])
    if not log_data:
        return jsonify({'success': False, 'error': 'Log not found'}), 404

    try:
        with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
            conn.execute(
                "UPDATE user_logs SET name = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (new_name, datetime.now(timezone.utc).isoformat(), log_id, user['id'])
            )
            conn.commit()
        return jsonify({'success': True, 'name': new_name})
    except Exception as exc:
        print(f"[RENAME LOG] Failed for {log_id}: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/logs/<log_id>', methods=['DELETE'])
@login_required()
def delete_log(log_id):
    """Delete a saved log. Only the owner can delete their own log."""
    user = _current_user(require_access=True)
    if not user:
        return jsonify({'success': False, 'error': 'Not authorized'}), 401

    # Verify the log exists AND belongs to this user before deleting.
    log_data = auth_billing.get_user_log(config.AUTH_DB_PATH, log_id, user['id'])
    if not log_data:
        return jsonify({'success': False, 'error': 'Log not found'}), 404

    try:
        with auth_billing.get_db(config.AUTH_DB_PATH) as conn:
            cur = conn.execute(
                "DELETE FROM user_logs WHERE id = ? AND user_id = ?",
                (log_id, user['id'])
            )
            conn.commit()
            deleted = cur.rowcount

        if deleted < 1:
            return jsonify({'success': False, 'error': 'Log not found'}), 404

        print(f"[DELETE LOG] User {user['id']} deleted log {log_id} ({log_data.get('name')})")
        return jsonify({'success': True, 'log_id': log_id})
    except Exception as exc:
        print(f"[DELETE LOG] Failed for {log_id}: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500



@app.route('/workspace')
@login_required()
def workspace():
    user = _current_user(require_access=True)
    if not user:
        return redirect(url_for('login'))

    if not auth_billing.can_access_workspace(user):
        flash('Full-service users cannot access the self-serve workspace. Upgrade to a self-serve plan to use this feature.', 'warning')
        return redirect(url_for('dashboard'))

    log_id = request.args.get('log_id')
    log_data = None
    if log_id:
        log_data = auth_billing.get_user_log(config.AUTH_DB_PATH, log_id, user['id'])

    response = make_response(render_template('workspace.html',
                           user=user,
                           app_version=APP_VERSION,
                           build_time=APP_BUILD_TIME,
                           vision_available=VISION_API_AVAILABLE,
                           impersonating=bool(session.get('impersonate_user_id')),
                           log_data=log_data))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/dashboard')
@login_required()
def dashboard():
    """User dashboard listing saved logs."""
    user = _current_user(require_access=True)
    if not user:
        return redirect(url_for('login'))
        
    # Get global banner setting
    settings = auth_billing.get_admin_settings(config.AUTH_DB_PATH)
    global_banner = settings.get('global_banner')
        
    logs = auth_billing.get_user_logs(config.AUTH_DB_PATH, user['id'])
    print(f"[DASHBOARD] User {user['id']} ({user.get('email')}) has {len(logs)} logs")
    if logs:
        print(f"[DASHBOARD] Log names: {[log['name'] for log in logs]}")
    return render_template('dashboard.html', 
                          user=user,
                          logs=logs,
                          global_banner=global_banner,
                          impersonating=bool(session.get('impersonate_user_id')))



@app.route('/las_viewer')
@login_required()
def las_viewer():
    log_id = request.args.get('log_id')
    log_data = None
    if log_id:
        user = _current_user()
        if user:
            log_data = auth_billing.get_user_log(config.AUTH_DB_PATH, log_id, user['id'])
    
    return render_template('las_viewer.html', app_version=APP_VERSION, log_data=log_data)


@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon to prevent 404 errors."""
    return '', 204

@app.route('/api/images/<filename>')
@login_required()
def get_image(filename):
    """Serve saved well log images to authenticated users."""
    user = _current_user(require_access=True)
    if not user:
        return "Not authorized", 401
    
    # We could theoretically verify the image belongs to the user,
    # but the UUID filename acts as a sufficient secure capability URL
    # for users inside their own session.
    from pathlib import Path
    images_dir = Path(config.DATA_ROOT) / 'images'
    return send_from_directory(str(images_dir), filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return image info"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Read image or PDF
    file_bytes = file.read()
    
    if file.filename.lower().endswith('.pdf') or file_bytes.startswith(b'%PDF'):
        try:
            import fitz
            doc = fitz.open("pdf", file_bytes)
            if len(doc) == 0:
                return jsonify({'error': 'PDF is empty'}), 400

            # Render pages at a higher DPI (e.g. 200 DPI instead of 72)
            zoom = 200 / 72.0
            mat = fitz.Matrix(zoom, zoom)

            page_images = []
            max_w = 0

            for page in doc:
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
                # Convert fitz pixmap to numpy array for OpenCV
                # Since colorspace=fitz.csRGB, pix.n will be 3 (RGB)
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                if img_bgr.shape[1] > max_w:
                    max_w = img_bgr.shape[1]
                    
                page_images.append(img_bgr)

            if not page_images:
                return jsonify({'error': 'PDF contains no renderable pages'}), 400

            # Pad images to match the maximum width to avoid vconcat errors
            padded_images = []
            for page_img in page_images:
                h_img, w_img = page_img.shape[:2]
                if w_img < max_w:
                    # Pad on the right with white pixels
                    padded = cv2.copyMakeBorder(page_img, 0, 0, 0, max_w - w_img, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    padded_images.append(padded)
                else:
                    padded_images.append(page_img)

            # Stitch all pages vertically into one long log image
            img = cv2.vconcat(padded_images)
            doc.close()
        except ImportError:
            return jsonify({'error': 'PDF support requires PyMuPDF (fitz) package'}), 500
        except Exception as e:
            return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 400
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Could not read image or PDF'}), 400
    
    h, w, _ = img.shape
    
    # Save the image to the persistent data volume so we can reference its path
    import uuid
    import os
    from pathlib import Path
    images_dir = Path(config.DATA_ROOT) / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    image_filename = f"{uuid.uuid4().hex}.jpg"
    image_path = images_dir / image_filename
    
    # Save as JPEG with 85% quality to save space
    cv2.imwrite(str(image_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    # We still return the base64 version for immediate frontend display,
    # but we also return the permanent path so the frontend can save it.
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Auto-detect tracks
    tracks = auto_detect_tracks(img)

    # If multiple panels are present, pick the "best" region of tracks as a hint,
    # but still return all tracks so the user can manually choose.
    primary_region = select_primary_track_region(tracks, w)

    # Lightweight header-only OCR: try to infer global top/bottom depth values
    # (e.g. from a Pass Summary table) without running full-panel OCR yet.
    detected_text = {'raw': [], 'numbers': [], 'suggestions': {}}
    ocr_suggestions = {}
    if (VISION_API_AVAILABLE and vision_client is not None) or LOCAL_OCR_AVAILABLE:
        try:
            header_h = max(100, int(h * 0.25))
            footer_h = max(100, int(h * 0.50))
            top_crop = img[0:header_h, :]
            bottom_crop = img[max(0, h - footer_h):h, :]
            # Stack top and bottom sections so one OCR call covers logs whose
            # header info appears at the bottom (e.g. ATR / footer-style headers).
            header_crop = np.vstack([top_crop, bottom_crop])
            ok_header, header_buf = cv2.imencode('.jpg', header_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok_header:
                header_bytes = header_buf.tobytes()
                detected_text = detect_text_vision_api(header_bytes)
                ocr_suggestions = detected_text.get('suggestions', {}) or {}
        except Exception as exc:
            print(f"Header OCR error on upload: {exc}")
            detected_text = {'raw': [], 'numbers': [], 'suggestions': {}}
            ocr_suggestions = {}

    return jsonify({
        'success': True,
        'image': f'data:image/jpeg;base64,{img_base64}',
        'image_path': f'/api/images/{image_filename}',
        'width': w,
        'height': h,
        'tracks': tracks,
        'all_tracks': tracks,
        'primary_region': {
            'left_px': primary_region.get('left_px'),
            'right_px': primary_region.get('right_px'),
            'track_indices': primary_region.get('track_indices'),
        } if primary_region else None,
        'detected_text': detected_text,
        'ocr_suggestions': ocr_suggestions or detected_text.get('suggestions', {}),
        'vision_api_available': bool(VISION_API_AVAILABLE or LOCAL_OCR_AVAILABLE)
    })

def pipeline_cc_cleanup(mask, min_size=20):
    """Connected-component noise cleanup. Removes small isolated blobs of pixels."""
    if mask is None or mask.size == 0:
        return mask
    # Binarize mask before CC
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Fast vectorized label filtering to prevent Gunicorn timeouts
    keep = stats[:, cv2.CC_STAT_AREA] >= min_size
    keep[0] = False  # Ignore background
    valid_mask = keep[labels]
    
    clean_mask = np.zeros_like(mask)
    clean_mask[valid_mask] = mask[valid_mask]
    return clean_mask

def pipeline_skeletonize(mask):
    """Skeletonization / Centerline Thinning."""
    if mask is None or mask.size == 0:
        return mask
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    if hasattr(cv2, 'ximgproc'):
        thinned = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        # Restore original probabilities where skeleton is present
        result = np.zeros_like(mask)
        result[thinned > 0] = mask[thinned > 0]
        return result
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded = cv2.erode(binary, kernel, iterations=1)
        result = np.zeros_like(mask)
        result[eroded > 0] = mask[eroded > 0]
        return result

@app.route('/digitize', methods=['POST'])
@login_required()
def digitize():
    """Process digitization request"""
    user = _current_user(require_access=True)
    if not user:
        return jsonify({'success': False, 'error': 'Not authorized'}), 401
        
    # Enforce trial limits on processing as well to prevent abuse
    if user.get('subscription_status') == 'trialing' and not user.get('is_admin'):
        user_logs = auth_billing.get_user_logs(config.AUTH_DB_PATH, user['id'])
        if len(user_logs) >= 3:
            return jsonify({
                'success': False, 
                'error': 'Trial limit reached. You have already processed and saved your 3 free logs. Please upgrade your account to continue digitizing.'
            }), 403

    data = request.json

    # Decode image. Prefer a server-side image path when the frontend has one;
    # reposting full base64 logs can exceed proxy/serverless request limits.
    image_path_ref = (data.get('image_path') or '').strip()
    img = None
    if image_path_ref.startswith('/api/images/'):
        image_filename = image_path_ref.rsplit('/', 1)[-1].split('?', 1)[0]
        disk_path = Path(config.DATA_ROOT) / 'images' / image_filename
        if disk_path.exists():
            img = cv2.imread(str(disk_path), cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': f'Image file not found for digitize: {image_filename}'}), 400

    if img is None:
        image_data = data.get('image')
        if not image_data or ',' not in image_data:
            return jsonify({'error': 'Missing image data for digitize.'}), 400
        img_data = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Could not decode image for digitize.'}), 400

    # Extract config
    cfg = data['config']
    preview_filters = data.get('preview_filters') or {}
    detected_text = data.get('detected_text') or {}
    response_mode = (data.get('response_mode') or 'lean').strip().lower()
    include_heavy_response = response_mode in {'full', 'debug'}
    trace_debug_requested = bool(data.get('trace_debug_export'))
    trace_debug_allowed = (
        request.args.get('trace_debug') == '1'
        or os.environ.get('TURBOTIFF_ALLOW_TRACE_DEBUG_EXPORT') == '1'
    )
    trace_debug_export = trace_debug_requested and trace_debug_allowed
    depth_cfg = cfg['depth']
    curves = (cfg['curves'] or [])[:6]
    gopt = cfg.get('global_options', {})

    header_metadata = data.get('header_metadata') if isinstance(data, dict) else None

    null_val = float(gopt.get('null', -999.25))
    downsample = int(gopt.get('downsample', 1))
    blur = int(gopt.get('blur', 3))
    min_run = int(gopt.get('min_run', 2))
    smooth_window = int(gopt.get('smooth_window', 5))
    snap_threshold = float(gopt.get('snap_threshold', 1.20)) # Default to 1.20 (20% brighter) as requested/observed

    H, W, _ = img.shape
    top = max(0, int(depth_cfg['top_px']))
    bot = min(H, int(depth_cfg['bottom_px']))
    top_depth = float(depth_cfg['top_depth'])
    bottom_depth = float(depth_cfg['bottom_depth'])
    depth_unit = depth_cfg.get('unit', 'FT')

    nrows = bot - top
    base_depth = compute_depth_vector(nrows, top_depth, bottom_depth)

    # Depth sanity checks
    depth_warnings = compute_depth_warnings({
        'top_px': top,
        'bottom_px': bot,
        'top_depth': top_depth,
        'bottom_depth': bottom_depth,
    }, H)
    
    curve_data = {}
    curve_traces = {}
    curve_trace_debug = {}
    curve_warnings = []

    for c in curves:
        # LAS-facing name/unit come from las_mnemonic/las_unit (or name/unit as fallback)
        name = c.get('las_mnemonic') or c.get('name')
        unit = c.get('las_unit') or c.get('unit', '')
        left_px = int(c['left_px'])
        right_px = int(c['right_px'])
        left_value = float(c['left_value'])
        right_value = float(c['right_value'])
        mode = c.get('mode', 'black')
        hot_side = c.get('hot_side')
        pixel_perfect = bool(c.get('pixel_perfect'))
        trace_mode = c.get('trace_mode')
        align_channels = bool(c.get('align_channels'))
        preserve_wiggles = bool(c.get('preserve_wiggles'))
        crest_boost = bool(c.get('crest_boost'))
        if not hot_side and np.isfinite(left_value) and np.isfinite(right_value):
            hot_side = 'right' if right_value >= left_value else 'left'

        # Defensive ROI bounds check: avoid empty slices that crash OpenCV ops.
        # (This can happen if the UI sends left/right reversed, or values are out of range.)
        img_w = int(img.shape[1])
        img_h = int(img.shape[0])
        left_px = max(0, min(img_w - 1, left_px))
        right_px = max(0, min(img_w, right_px))
        if right_px <= left_px:
            curve_warnings.append({
                'curve': name,
                'error': 'Invalid curve bounds (right_px must be > left_px).',
                'left_px': left_px,
                'right_px': right_px,
                'image_width': img_w,
            })
            continue

        top_clamped = max(0, min(img_h - 1, int(top)))
        bot_clamped = max(0, min(img_h, int(bot)))
        if bot_clamped <= top_clamped:
            curve_warnings.append({
                'curve': name,
                'error': 'Invalid depth bounds (bottom_px must be > top_px).',
                'top_px': top_clamped,
                'bottom_px': bot_clamped,
                'image_height': img_h,
            })
            continue

        roi = img[top_clamped:bot_clamped, left_px:right_px]
        if roi is None or roi.size == 0:
            curve_warnings.append({
                'curve': name,
                'error': 'Empty ROI for curve (check left/right and top/bottom).',
                'top_px': top_clamped,
                'bottom_px': bot_clamped,
                'left_px': left_px,
                'right_px': right_px,
            })
            continue

        if align_channels:
            roi = align_rgb_channels(roi)
        if blur > 0:
            bb = blur + 1 if blur % 2 == 0 else blur
            roi = cv2.GaussianBlur(roi, (bb, bb), 0)

        # Define colored modes set (including auto which detects hue automatically)
        colored_modes = {"green", "red", "blue", "auto", "cyan", "magenta", "yellow", "orange", "purple"}

        # Read pipeline toggles
        enable_grid_suppression = bool(c.get('enable_grid_suppression', True))
        enable_curve_masking = bool(c.get('enable_curve_masking', True))
        enable_cc_cleanup = bool(c.get('enable_cc_cleanup', True))
        enable_skeletonization = bool(c.get('enable_skeletonization', True))
        enable_viterbi = bool(c.get('enable_viterbi', True))

        # Update preview_filters for this curve
        curve_ui_filters = dict(preview_filters)
        curve_ui_filters['enable_grid_suppression'] = enable_grid_suppression
        curve_ui_filters['enable_curve_masking'] = enable_curve_masking

        # NEW: Build a soft probability mask for the curve using color/edges
        # plus vertical-rail suppression. This returns an 8-bit image where
        # higher values mean higher likelihood of curve pixels.
        # Use compute_prob_map for all modes - it has sophisticated edge detection
        # and centerline boost that works well
        mask = compute_prob_map(roi, mode=mode, ui_filters=curve_ui_filters)

        # Pipeline: Connected-Component Cleanup
        if enable_cc_cleanup:
            mask = pipeline_cc_cleanup(mask, min_size=20)

        # Pipeline: Skeletonization / Thinning
        if enable_skeletonization:
            mask = pipeline_skeletonize(mask)

        if mode not in {"green", "red", "blue", "auto", "cyan", "magenta", "yellow", "orange", "purple"}:
            _pm = mask.astype(np.float32) / 255.0
            _pct_nonzero = float(np.mean(_pm > 0.01) * 100)
            _pm_max = float(_pm.max())
            _pm_mean = float(_pm.mean())
            # Debug info removed

        # NEW: Use DP-based smooth path tracing with plausibility checks
        curve_type = c.get('type', 'GR')  # Get curve type for plausibility

        # If viterbi is disabled, use a simple argmax over the mask
        if not enable_viterbi:
            h_mask, w_mask = mask.shape
            xs = np.full(h_mask, np.nan, dtype=np.float32)
            confidence = np.zeros(h_mask, dtype=np.float32)
            for y in range(h_mask):
                row = mask[y]
                max_val = np.max(row)
                if max_val > 0:
                    xs[y] = np.argmax(row)
                    confidence[y] = float(max_val) / 255.0
            
            curve_smooth_window = 1
            refine_kwargs = {}
            outlier_threshold = 100.0
        else:
            # For explicit color modes, allow more left-right wiggle (lower
            # smoothness penalty) and rely mostly on the DP + local maxima
            # refinement rather than heavy 1D smoothing so the traced path can
            # hug the colored curve as tightly as possible.
            curve_smooth_window = smooth_window
            refine_kwargs = {}
            outlier_threshold = 3.0
            if mode in colored_modes:
                # NO smoothing: window = 1 means no median filter applied
                curve_smooth_window = 1
                # MAXIMUM local window and absolute minimum threshold to snap to any ink
                refine_kwargs = {"dominance_ratio": snap_threshold, "max_shift": 25, "min_prob": 0.005}
                # Disable outlier removal - keep every point for maximum accuracy
                outlier_threshold = 100.0  # Effectively disabled
            else:
                # Use user threshold for non-colored modes too (default was 1.1)
                refine_kwargs = {"dominance_ratio": snap_threshold}
                # Non-GR black curves are supposed to follow visible crest tips.
                # Any median smoothing here flattens short visible crest tips, so
                # leave the trace unsmoothed and let the black-specific edge snap
                # shape it instead.
                if curve_type.upper() != "GR":
                    curve_smooth_window = 1
                    outlier_threshold = 12.0
            # Effectively zero smoothness penalty for colored modes to prefer jagged ink over smooth artifacts.
            # Black mode needs looser DP than the old max_step=3 / smooth_lambda=0.5 settings;
            # otherwise the path lags real excursions and the later black refiners are forced to
            # drag a too-stiff base trace sideways after the fact.
            if mode in colored_modes:
                dp_smooth_lambda = 0.001
                dp_curv_lambda = 0.001
                max_step_dp = 200  # Allow unlimited movement to follow gamma ray spikes
            else:
                curve_type_upper = curve_type.upper()
                if curve_type_upper == "GR":
                    dp_smooth_lambda = 0.001
                    dp_curv_lambda = 0.001
                    max_step_dp = 150
                else:
                    dp_smooth_lambda = 0.005
                    dp_curv_lambda = 0.001
                    max_step_dp = 150

        # Optional pixel-perfect skeleton tracer (preserve every bump)
        if not enable_viterbi:
            # Simple argmax tracer already ran above, do nothing here
            pass
        elif ai_tracer.is_available() and trace_mode == "ai_tracer":
            # Use the AI model for tracing
            try:
                # The AI model predicts coordinates relative to the ROI's left edge
                # and already handles scaling to the ROI width.
                xs = ai_tracer.trace(roi)
                confidence = np.ones_like(xs) * 0.95 # Mock high confidence for AI
            except Exception as e:
                print(f"⚠️ AI Tracer failed for {name}: {e}")
                # Fallback to empty if AI fails
                xs = np.full(roi.shape[0], np.nan)
                confidence = np.zeros(roi.shape[0])
        elif pixel_perfect and mode in colored_modes:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if trace_mode == "skeleton_path":
                xs, confidence = trace_curve_skeleton_path(mask)
            else:
                xs, confidence = trace_curve_pixel_perfect(
                    mask,
                    grayscale=gray_roi,
                    bgr=roi,
                    hot_side=hot_side,
                    preserve_wiggles=preserve_wiggles,
                    crest_boost=crest_boost,
                )
            width_px = mask.shape[1]
            # Fill gaps gently to avoid dropping rows
            if xs.size:
                s = pd.Series(xs)
                s = s.interpolate(method='linear', limit_direction='both', limit=max(10, int(xs.size * 0.02)))
                xs = s.to_numpy(dtype=np.float32)
            # Hybrid post-processing: force missed ink peaks onto the curve
            prob = mask.astype(np.float32) / 255.0
            if crest_boost:
                xs = _postprocess_missed_peaks(mask, prob, xs, search_radius=40, min_prob=0.004)
            else:
                xs = _postprocess_missed_peaks(mask, prob, xs, search_radius=30, min_prob=0.008)
        # For colored modes, use the "Fusion" strategy from successful memories:
        # Run both DP and Direct Centerline tracers, then merge per-row based on probability.
        # AND DISABLE EXTRA REFINEMENTS which cause the zig-zag snapping.
        elif mode in colored_modes:
            # SUPER-RESOLUTION: Upscale mask by 2x to allow sub-pixel precision
            # Use LINEAR interpolation to create smooth gradients between pixels
            mask_orig = mask
            h_orig, w_orig = mask.shape
            mask = cv2.resize(mask, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LINEAR)
            
            # Adjust parameters for 2x scale
            max_step_dp_sr = max_step_dp * 2
            
            # 1. Run DP Tracer (provides continuity)
            xs_dp, conf_dp = trace_curve_with_dp(
                mask,
                scale_min=left_value,
                scale_max=right_value,
                curve_type=curve_type,
                max_step=max_step_dp_sr,
                smooth_lambda=dp_smooth_lambda,
                curv_lambda=dp_curv_lambda,
                hot_side=hot_side,
            )

            # 2. Local Peak Search Fusion
            # Instead of a global direct tracer (which gets distracted by far-away curves),
            # search locally around the DP path for the true tip of the spike.
            h_mask, w_mask = mask.shape
            prob_map = mask.astype(np.float32) / 255.0
            xs = np.full(h_mask, np.nan, dtype=np.float32)

            search_window = 100

            # Vectorized fusion: build a padded prob strip of width 2*search_window+1
            # centred on each DP position, find the best candidate per row with NumPy.
            ix_dp_arr = np.round(xs_dp).astype(np.int32)
            valid_mask = np.isfinite(xs_dp) & (ix_dp_arr >= 0) & (ix_dp_arr < w_mask)
            col_idx = np.arange(w_mask, dtype=np.float32)  # (w_mask,)

            for y in range(h_mask):
                x_dp = xs_dp[y]
                if not valid_mask[y]:
                    xs[y] = x_dp
                    continue
                ix_dp = ix_dp_arr[y]
                p_dp = prob_map[y, ix_dp]
                start = max(0, ix_dp - search_window)
                end = min(w_mask, ix_dp + search_window + 1)
                local_prob = prob_map[y, start:end]
                if local_prob.size == 0:
                    xs[y] = x_dp
                    continue
                max_p = local_prob.max()
                if max_p <= 0:
                    xs[y] = x_dp
                    continue
                # Score all near-peak candidates in one shot
                local_col = col_idx[start:end]
                scores = local_prob - 0.15 * np.abs(local_col - x_dp)
                best_cand = int(np.argmax(scores))
                p_local = float(local_prob[best_cand])
                # Ridge-centroid snap (vectorized plateau expansion)
                x_local = float(start + best_cand)
                peak_thr = max_p * 0.99
                plateau = local_prob >= peak_thr
                if plateau.any():
                    seg = local_prob * plateau
                    s = float(seg.sum())
                    if s > 1e-8:
                        x_local = float((local_col * seg).sum() / s)
                if p_local > p_dp * 0.06:
                    xs[y] = x_local
                else:
                    xs[y] = x_dp

            # 3. Pure Center-of-Mass refinement (vectorized, no Python row loop)
            h_sr, w_sr = mask.shape
            prob_sr = mask.astype(np.float32) / 255.0
            win = 12
            ix_arr = np.round(xs).astype(np.int32)
            finite_rows = np.isfinite(xs)
            ix_arr_c = np.clip(ix_arr, 0, w_sr - 1)
            for y in range(h_sr):
                if not finite_rows[y]:
                    continue
                ix = ix_arr_c[y]
                start = max(0, ix - win)
                end = min(w_sr, ix + win + 1)
                row_slice = prob_sr[y, start:end]
                s = row_slice.sum()
                if s > 1e-6:
                    xs[y] = float((col_idx[start:end] * row_slice).sum() / s)

            # 4. Refine Peaks (MOVED to after downsampling)
            # We don't run it here to avoid the downsampling smoothing out the sharp tips.
            
            # 8. Minimal cleanup only - NO aggressive snapping to far-away peaks
            s = pd.Series(xs)
            xs = s.interpolate(method='linear', limit_direction='both', limit=max(25, int(xs.size * 0.02))).to_numpy(dtype=np.float32)
            
            # DOWNSAMPLE: Map back to original resolution
            # Take every 2nd point and divide coordinate by 2
            # Use averaging to reduce noise: (y*2 + y*2+1) / 2
            xs_down = np.full(h_orig, np.nan, dtype=np.float32)
            for y_orig in range(h_orig):
                y_sr = y_orig * 2
                val1 = xs[y_sr]
                val2 = xs[y_sr + 1] if y_sr + 1 < h_mask else val1
                
                if np.isfinite(val1) and np.isfinite(val2):
                    xs_down[y_orig] = (val1 + val2) / 4.0 # Divide by 2 (avg) then divide by 2 (scale) -> /4
                elif np.isfinite(val1):
                    xs_down[y_orig] = val1 / 2.0
                elif np.isfinite(val2):
                    xs_down[y_orig] = val2 / 2.0
            
            xs = xs_down
            # Restore original mask for downstream
            mask = mask_orig

            # 8b. Clean up artifacts (single-pixel horizontal glitches)
            # The high-sensitivity plateau logic can sometimes trigger on noise.
            # A median filter of size 3 removes single-pixel spikes but keeps real features (width >= 2).
            try:
                 from scipy.signal import medfilt
                 xs_valid_mask = np.isfinite(xs)
                 if np.sum(xs_valid_mask) > 3:
                     # Fill NaNs temporarily
                     xs_filled = xs.copy()
                     xs_filled[~xs_valid_mask] = np.nanmedian(xs)
                     # Apply median filter
                     xs_smooth = medfilt(xs_filled, kernel_size=3)
                     # Restore valid pixels
                     xs[xs_valid_mask] = xs_smooth[xs_valid_mask]
            except ImportError:
                 pass

            # 9. FINAL TIP REFINEMENT (Post-Downsample)
            # Run the peak pusher on the original resolution to catch the absolute edges
            # that might have been smoothed by downsampling.
            # Use small buffer (0.2) to be very sticky to tips.
            # Reduced search radius to 30px to prevent teleporting to distant noise.
            xs = refine_peaks_and_valleys(mask, xs, search_radius=30, min_prob=0.005)

            # Gentle centerline refinement to re-center on ink after outer-edge bias and fusion
            try:
                xs = refine_to_stroke_centerline(mask, xs, threshold_ratio=0.5, window_size=8)
            except Exception:
                pass

            # Global centering correction: subtract median residual vs. ink center-of-mass
            try:
                h_mask, w_mask = mask.shape
                xs_valid = xs[~np.isnan(xs)]
                if xs_valid.size > 0:
                    # Compute per-row center-of-mass of probability
                    probs = mask.astype(np.float32) / 255.0
                    weight_sums = probs.sum(axis=1)
                    com = np.full(h_mask, np.nan, dtype=np.float32)
                    nonzero_rows = weight_sums > 1e-6
                    com[nonzero_rows] = (probs[nonzero_rows] * np.arange(w_mask, dtype=np.float32)).sum(axis=1) / weight_sums[nonzero_rows]
                    deltas = []
                    for y in range(h_mask):
                        if np.isnan(xs[y]) or np.isnan(com[y]):
                            continue
                        deltas.append(xs[y] - com[y])
                    if deltas:
                        median_delta = float(np.median(deltas))
                        xs = xs - median_delta
                    
                    # Do not apply a fixed pixel shift; it prevents true centerline alignment.
            except Exception:
                pass
            
        else:
            # For black/other modes, use DP with smoothness constraints
            xs, confidence = trace_curve_with_dp(
                mask,
                scale_min=left_value,
                scale_max=right_value,
                curve_type=curve_type,
                max_step=max_step_dp,
                smooth_lambda=dp_smooth_lambda,
                curv_lambda=dp_curv_lambda,
                hot_side=hot_side,
            )

            # Snap the DP path toward obvious local maxima in the prob mask
            xs = refine_trace_with_local_maxima(mask, xs, **refine_kwargs)

            # Skip the aggressive peak/valley pusher for black traces. On
            # dense black logs it can snap sideways onto neighboring rails or
            # filled blocks, which is what creates the horizontal "shelf"
            # artifacts. DP + local maxima already finds the right branch;
            # re-center on the stroke body instead of pushing to extrema.
            try:
                # Give black traces a wider recentering window so thick strokes
                # can settle onto the body of the ink instead of staying pinned
                # near whichever edge the DP pass first touched.
                xs = refine_to_stroke_centerline(mask, xs, threshold_ratio=0.55, window_size=14)
            except Exception:
                pass

            try:
                # Probability maps still bias toward one stroke edge on dense
                # black logs. Recenter once more against the raw dark stroke
                # body in the grayscale ROI so the trace sits in the visual
                # middle of the black ink.
                xs = refine_black_trace_to_dark_run_center(
                    roi,
                    xs,
                    hot_side=hot_side,
                    curve_type=curve_type,
                )
            except Exception:
                pass

            # Guard against spurious "left-shooting" / "right-shooting" jumps
            # where DP briefly locked onto a grid rail or noise column and
            # drew a horizontal line to the chart edge. Rolling-median
            # deviation > ~45 px is almost never a real excursion on a log
            # track because legitimate peaks are curved, not instantaneous.
            try:
                xs = guard_trace_outliers_rolling_median(xs, window=21, max_deviation=20.0)
            except Exception:
                pass

            # Second pass: follow nearby ink that behaves like a continuous
            # line over several rows instead of the darkest single-row crest.
            # This avoids horizontal grid bars and filled blocks pulling the
            # trace into shelf artifacts.
            try:
                xs = refine_black_trace_to_continuous_line(
                    roi,
                    xs,
                    search_radius=20,
                    guide_window=31,
                    vertical_window=13,
                )
            except Exception:
                pass

            # Second outlier pass: the line-following pass is conservative,
            # but keep the tighter guard as protection against noisy scans.
            try:
                xs = guard_trace_outliers_rolling_median(xs, window=15, max_deviation=15.0)
            except Exception:
                pass

            # Velocity guard: micro-crests jump 10-20 px in 1-2 rows.
            # Real geology moves gradually. Cap |dx/dy| to ~6 px/row.
            try:
                xs = guard_trace_velocity(xs, max_dx=6.0)
            except Exception:
                pass

            # Median filter: remove 1-3 row horizontal glitches that survive
            # the outlier guards. The colored pipeline already does this.
            try:
                from scipy.signal import medfilt
                xs_filled = xs.copy()
                nan_mask = ~np.isfinite(xs_filled)
                if nan_mask.any() and np.isfinite(xs_filled).any():
                    xs_filled[nan_mask] = np.nanmedian(xs_filled)
                xs_smooth = medfilt(xs_filled, kernel_size=3)
                valid_mask = np.isfinite(xs)
                xs[valid_mask] = xs_smooth[valid_mask]
            except Exception:
                pass

        # Do not run the old non-GR black smoother here. After the dark-run
        # recenter/hot-side bias pass, even light smoothing pulls RHOB/DT-type
        # traces back toward the inner half of the stroke and weakens the
        # actual printed excursions we are trying to follow.

        width_px = mask.shape[1]

        # UNIVERSAL GAP FILLING:
        # Aggressive grid removal can leave small gaps where the curve crossed a grid line.
        # We linearly interpolate these gaps to ensure continuity.
        if xs.size > 0:
            s = pd.Series(xs)
            h_mask, w_mask = mask.shape
            if mode in colored_modes:
                max_gap = max(25, int(h_mask * 0.02))
            else:
                max_gap = max(25, int(h_mask * 0.02))  # Strict for dashed black curves
            s = s.interpolate(method='linear', limit_direction='both', limit=max_gap, limit_area=None)
            # Handle edge cases
            if s.isna().any():
                s = s.ffill(limit=max_gap).bfill(limit=max_gap)
            xs = s.to_numpy(dtype=np.float32)

        if mode not in colored_modes:
            try:
                xs = suppress_black_grid_lock_runs(roi, xs, curve_type=curve_type)
            except Exception:
                pass

            try:
                # Finish black mode the same way the successful color path
                # does: re-center after the grid-lock cleanup, not before it.
                # This keeps the line on the middle of the visible black ink
                # instead of on the stroke edge or a nearby rail.
                xs = refine_to_stroke_centerline(mask, xs, threshold_ratio=0.45, window_size=16)
            except Exception:
                pass

            try:
                xs = recenter_black_trace_post_dp(roi, xs)
            except Exception:
                pass

        # Likewise, skip the old final non-GR black crest snap. The dark-run
        # recenter helper now already biases wide rows toward the reading-side
        # edge, and a second crest-only shove consistently degraded RHOB on the
        # saved black capture set.

        # For colored modes, apply specific enhancements (peaks, centerline refinement)
        if mode in colored_modes:
            if curve_type.upper() == "GR":
                prob_map = mask.astype(np.float32) / 255.0
                xs = ensure_gr_peak_crests(xs, prob_map, hot_side=hot_side, min_prob=0.01)

            # Final centerline snap for ALL colored modes
            try:
                xs = refine_to_stroke_centerline(mask, xs, threshold_ratio=0.5, window_size=10)
            except Exception:
                pass

            try:
                prob = mask.astype(np.float32) / 255.0
                h_mask, w_mask = prob.shape
                radius = 4
                xs2 = xs.copy()
                for y in range(h_mask):
                    x0 = xs2[y]
                    if not np.isfinite(x0):
                        continue
                    ix = int(round(x0))
                    x_min = max(0, ix - radius)
                    x_max = min(w_mask, ix + radius + 1)
                    row = prob[y, x_min:x_max]
                    if row.size == 0:
                        continue
                    s = float(row.sum())
                    if s <= 1e-8:
                        continue
                    coords = np.arange(x_min, x_max, dtype=np.float32)
                    xs2[y] = float((coords * row).sum() / s)
                xs = xs2
            except Exception:
                pass

            if curve_type.upper() == "GR":
                prob_map = mask.astype(np.float32) / 255.0
                xs = ensure_gr_peak_crests(xs, prob_map, hot_side=hot_side)

            # To avoid staircases when DP tracks a jagged pixelated diagonal,
            # apply a light moving average. This turns blocky steps into smooth diagonals.
            try:
                if xs.size > 0:
                    s = pd.Series(xs)
                    xs = s.rolling(window=5, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)
            except Exception:
                pass

            # Optional final local peak snap; kept disabled because it
            # quantizes to integer columns and can reintroduce zig-zags.
            do_final_peak_snap = False
            if do_final_peak_snap:
                # FINAL STEP: refine each point to local probability maximum
                # Tight window to avoid sideways wander
                h_mask, w_mask = mask.shape
                xs_refined_final = np.copy(xs)
                
                local_search_radius = 2  # Very tight window to avoid sideways wander
                
                for y in range(h_mask):
                    if not np.isnan(xs[y]):
                        x_current = int(round(xs[y]))
                        
                        # Define tight search window around current position
                        x_min = max(0, x_current - local_search_radius)
                        x_max = min(w_mask, x_current + local_search_radius + 1)
                        
                        # Find local maximum within this small window
                        row_segment = mask[y, x_min:x_max].astype(np.float32)
                        
                        if len(row_segment) > 0 and row_segment.max() > 0:
                            # Find peak position within window
                            local_peak_idx = np.argmax(row_segment)
                            # Convert back to full image coordinates
                            xs_refined_final[y] = x_min + local_peak_idx
                
                xs = xs_refined_final
        else:
            # For non-colored modes, keep the original vertical-rail rejection logic
            xs_valid = xs[~np.isnan(xs)]
            if xs_valid.size > 0:
                dyn_range = float(np.nanmax(xs_valid) - np.nanmin(xs_valid))
                min_dyn = max(4.0, 0.02 * float(width_px))
                if dyn_range < min_dyn:
                    xs_fallback = pick_curve_x_per_row(mask, min_run=min_run)
                    xs_fallback = smooth_nanmedian(xs_fallback, window=curve_smooth_window)
                    xs = xs_fallback
                    xs_valid = xs[~np.isnan(xs)]

            if xs_valid.size > 0:
                std_x = float(np.nanstd(xs_valid))
                std_threshold = max(1.0, 0.005 * float(width_px))
                # Only reject near-perfectly-vertical traces (rail lock-on).
                # Use a very tight threshold: 0.5% of track width or 1.0px minimum.
                # Slow curves like DTC/RHOB can legitimately have low std.
                if std_x < std_threshold:
                    xs[:] = np.nan

        # Scale-aware pixel → value conversion.
        # curve config may carry scale_type ('linear' | 'log' | 'centered') and wrapped.
        # If missing, fall back to the curve-type default (e.g. resistivity → log).
        scale_type = (c.get('scale_type') or '').lower().strip()
        wrapped_flag = bool(c.get('wrapped'))
        if not scale_type:
            _hint = scale_detection.classify_curve_type(c.get('name') or c.get('type') or '')
            scale_type = (_hint or {}).get('scale_type', 'linear')

        # Auto-detect wrap for log scales when the user didn't check the box.
        # detect_wrap() fires when the trace has repeated large alternating
        # jumps (characteristic of multi-decade resistivity wrapping).
        wrap_auto_detected = False
        if scale_type == 'log' and not wrapped_flag:
            try:
                if scale_detection.detect_wrap(xs, width_px):
                    wrapped_flag = True
                    wrap_auto_detected = True
            except Exception:
                pass

        vals = scale_detection.pixel_to_value(
            xs=xs,
            width_px=width_px,
            left_value=left_value,
            right_value=right_value,
            scale_type=scale_type,
            wrapped=wrapped_flag,
        )

        if wrap_auto_detected:
            curve_warnings.append({
                'curve': name,
                'info': f"Auto-enabled wrap unwrapping for {name} (multi-decade log trace detected).",
                'auto_wrap': True,
            })

        vals_out = np.where(np.isnan(vals), null_val, vals).astype(np.float32)
        curve_data[name] = {'unit': unit, 'values': vals_out}

        # Build a continuous display trace for the UI overlay. The exported LAS
        # values can still contain nulls, but the visible editing line should
        # remain continuous rather than showing gaps.
        trace_points = []
        if xs.size > 0:
            try:
                xs_display = pd.Series(xs.astype(np.float32)).interpolate(
                    method='linear',
                    limit_direction='both',
                    limit_area=None,
                ).to_numpy(dtype=np.float32)
            except Exception:
                xs_display = xs

            valid_rows = np.where(~np.isnan(xs_display))[0]
            if valid_rows.size > 0:
                for row_idx in valid_rows:
                    x_val = xs_display[row_idx]
                    x_img = float(left_px) + float(x_val)
                    y_img = float(top + row_idx)
                    trace_points.append([x_img, y_img])

        curve_traces[name] = trace_points

        if trace_debug_export and mode not in colored_modes:
            try:
                dbg = build_black_trace_debug_export(
                    name,
                    roi,
                    mask,
                    xs,
                    curve_type=curve_type,
                    mode=mode,
                )
                if dbg:
                    curve_trace_debug[name] = dbg
            except Exception:
                pass
    
    # Resample to fixed 0.5 ft step when using feet
    las_depth = base_depth
    las_curve_data = curve_data
    ai_payload = None
    ai_summary = None
    digitized_depth = None
    digitized_curves = None
    if depth_unit.upper() == "FT" and base_depth.size > 1:
        start = float(base_depth[0])
        stop = float(base_depth[-1])
        step_mag = 0.5

        if stop >= start:
            las_depth = np.arange(start, stop + step_mag * 0.5, step_mag, dtype=np.float32)
        else:
            las_depth = np.arange(start, stop - step_mag * 0.5, -step_mag, dtype=np.float32)

        las_curve_data = {}
        for name, meta in curve_data.items():
            vals = meta["values"].astype(np.float32)
            valid_mask = vals != null_val

            if not np.any(valid_mask):
                new_vals = np.full(las_depth.shape, null_val, dtype=np.float32)
            else:
                depth_valid = base_depth[valid_mask]
                vals_valid = vals[valid_mask]
                order = np.argsort(depth_valid)
                depth_sorted = depth_valid[order]
                vals_sorted = vals_valid[order]
                interp_vals = np.interp(las_depth, depth_sorted, vals_sorted, left=null_val, right=null_val)
                new_vals = interp_vals.astype(np.float32)

            las_curve_data[name] = {"unit": meta.get("unit", ""), "values": new_vals}

    if not curve_data:
        return jsonify({
            'error': 'No valid curves to digitize. Please check curve bounds and depth settings.',
            'curve_warnings': curve_warnings,
            'depth_warnings': depth_warnings,
        }), 400

    # Run simple curve sanity checks (outlier warnings) on the final LAS depth grid
    outlier_warnings = compute_curve_outlier_warnings(curves, las_curve_data, null_val)

    # Prepare digitized vectors for frontend cursor readout (always, even without lasio)
    try:
        digitized_depth = las_depth.tolist()
        digitized_curves = {
            name: {
                "unit": meta.get("unit", ""),
                "values": (meta.get("values").tolist() if meta.get("values") is not None else None),
            }
            for name, meta in las_curve_data.items()
        }
    except Exception:
        digitized_depth = None
        digitized_curves = None

    las_content = None
    ai_payload = None
    ai_summary = None

    # Lean responses avoid server-side LAS generation/validation. The browser
    # rebuilds the LAS from digitized_depth/digitized_curves for download.
    validation = {
        'passed': True,
        'message': 'LAS generation deferred to browser download (lean mode).'
    }
    if include_heavy_response:
        las_content = write_las_simple(las_depth, las_curve_data, depth_unit, header_metadata=header_metadata)
        validation = {
            'passed': True,
            'message': 'LAS validation skipped (lasio not installed).'
        }
        if LASIO_AVAILABLE:
            try:
                lasio.read(StringIO(las_content))
                validation = {
                    'passed': True,
                    'message': 'LAS parsed successfully with lasio.'
                }
            except Exception as exc:
                validation = {
                    'passed': False,
                    'message': f'LAS validation failed: {exc}'
                }

        # Build AI analysis payload (OCR + LAS stats + user curve config)
        ai_payload = build_ai_analysis_payload(las_content, detected_text, curves)
        ai_summary = call_hf_curve_analysis(ai_payload) if ai_payload else None

    return jsonify({
        'success': True,
        'las_content': las_content if include_heavy_response else None,
        'filename': build_las_filename_from_metadata(header_metadata, default_name='digitized_log.las'),
        'validation': validation,
        'outlier_warnings': outlier_warnings,
        'depth_warnings': depth_warnings,
        'curve_warnings': curve_warnings,
        'curve_traces': curve_traces,
        'curve_trace_debug': curve_trace_debug if trace_debug_export else {},
        'ai_payload': ai_payload if include_heavy_response else None,
        'ai_summary': ai_summary if include_heavy_response else None,
        'digitized_depth': digitized_depth,
        'digitized_curves': digitized_curves,
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'vision_api': VISION_API_AVAILABLE
    })


@app.route('/debug-env')
def debug_env():
    """Debug endpoint to check environment variable configuration."""
    return jsonify({
        'HF_API_TOKEN': 'set' if HF_API_TOKEN else 'missing',
        'HF_MODEL_ID': HF_MODEL_ID or 'missing',
        'OPENAI_API_KEY': 'set' if OPENAI_API_KEY else 'missing',
        'OPENAI_MODEL_ID': OPENAI_MODEL_ID or 'missing',
        'GEMINI_API_KEY': 'set' if GEMINI_API_KEY else 'missing',
        'GEMINI_MODEL_ID': GEMINI_MODEL_ID or 'missing',
        'VISION_API_AVAILABLE': VISION_API_AVAILABLE,
        'GOOGLE_VISION_CREDENTIALS_JSON': 'set' if os.getenv('GOOGLE_VISION_CREDENTIALS_JSON') else 'missing',
        'GOOGLE_APPLICATION_CREDENTIALS': 'set' if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') else 'missing'
    })


@app.route('/test-ai')
def test_ai():
    """Test endpoint to verify Hugging Face API is working."""
    # Prefer Gemini if configured
    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            # Use REST API directly to avoid SDK version issues
            # Model ID should include 'models/' prefix (e.g., 'models/gemini-2.0-flash')
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith('models/') else f'models/{GEMINI_MODEL_ID}'
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": "What is 2+2?"}]}]
            }
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        text = parts[0].get('text', '')
                        return jsonify({
                            'success': True,
                            'status_code': 200,
                            'provider': 'gemini',
                            'model': model_name,
                            'response': text,
                        })
            return jsonify({
                'success': False,
                'provider': 'gemini',
                'model': model_name,
                'error': f"{resp.status_code} {resp.text}",
            })
        except Exception as exc:
            return jsonify({
                'success': False,
                'provider': 'gemini',
                'model': GEMINI_MODEL_ID,
                'error': str(exc),
            })

    # Fallback to OpenAI if configured
    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=[
                    {"role": "user", "content": "What is 2+2?"},
                ],
                max_tokens=50,
                temperature=0.3,
            )
            choices = resp.get("choices") or []
            content = ""
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
            return jsonify({
                'success': True,
                'status_code': 200,
                'provider': 'openai',
                'model': OPENAI_MODEL_ID,
                'response': content
            })
        except Exception as exc:
            return jsonify({
                'success': False,
                'provider': 'openai',
                'model': OPENAI_MODEL_ID,
                'error': str(exc),
            })

    if not HF_API_TOKEN or not HF_MODEL_ID:
        return jsonify({
            'success': False,
            'error': 'No AI provider configured (missing Gemini/OpenAI/HF credentials).',
            'HF_API_TOKEN': 'set' if HF_API_TOKEN else 'missing',
            'HF_MODEL_ID': HF_MODEL_ID or 'missing'
        })

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        return jsonify({
            'success': False,
            'error': f'InferenceClient init error: {str(exc)}',
            'model': HF_MODEL_ID
        })

    try:
        out = client.text_generation(
            "What is 2+2?",
            model=HF_MODEL_ID,
            max_new_tokens=50,
            temperature=0.3,
        )
        return jsonify({
            'success': True,
            'status_code': 200,
            'provider': 'hf-inference',
            'model': HF_MODEL_ID,
            'response': out if isinstance(out, str) else str(out)
        })
    except Exception as exc:
        return jsonify({
            'success': False,
            'provider': 'hf-inference',
            'error': str(exc),
            'model': HF_MODEL_ID
        })


@app.route('/list-gemini-models')
def list_gemini_models():
    """List available Gemini models from the API."""
    if not GEMINI_API_KEY:
        return jsonify({'error': 'GEMINI_API_KEY not set'}), 400
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models?key={GEMINI_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get('models', [])
            # Filter for models that support generateContent
            generate_models = [
                {
                    'name': m.get('name', ''),
                    'displayName': m.get('displayName', ''),
                    'supportedMethods': m.get('supportedGenerationMethods', [])
                }
                for m in models
                if 'generateContent' in m.get('supportedGenerationMethods', [])
            ]
            return jsonify({
                'success': True,
                'models': generate_models,
                'total': len(generate_models)
            })
        else:
            return jsonify({
                'success': False,
                'error': f"{resp.status_code} {resp.text}"
            })
    except Exception as exc:
        return jsonify({
            'success': False,
            'error': str(exc)
        })


@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    """Chat-style endpoint: answer a question about the current log using ai_payload.

    Expects JSON with:
      - ai_payload: the object returned from /digitize
      - question: user's natural language question
    """
    data = request.json or {}
    ai_payload = data.get('ai_payload')
    question = (data.get('question') or '').strip()

    if not ai_payload or not question:
        return jsonify({'success': False, 'error': 'Missing ai_payload or question.'}), 400

    answer = call_hf_curve_chat(ai_payload, question)
    if answer is None:
        return jsonify({'success': False, 'error': 'AI chat is not configured. Please set GEMINI_API_KEY, OPENAI_API_KEY, or HF_API_TOKEN in your environment.'}), 500
    
    # If answer contains error message from AI API, still return success but show the error
    return jsonify({'success': True, 'answer': answer})


@app.route('/refine_edit', methods=['POST'])
def refine_edit():
    """
    Refine a curve edit using multi-scale tracing on a local segment.
    
    Takes a small vertical window around the edited point and runs
    the improved line detection to find the best x-position.
    
    Request JSON:
        image: base64 encoded image
        track: {leftX, rightX, leftValue, rightValue}
        editY: pixel Y coordinate of the edit
        editX: current pixel X coordinate (user's drag position)
        windowSize: vertical window size in pixels (default 50)
        curveType: curve type (GR, RHOB, etc.)
        mode: detection mode (green, black, etc.)
    
    Returns:
        refinedX: the best x-position from multi-scale detection
        confidence: detection confidence
    """
    try:
        data = request.json
        
        # Decode image
        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'})
        
        h_img, w_img = img.shape[:2]
        
        # Get parameters
        track = data.get('track', {})
        # Note: if isCrop is True, leftX/rightX are used mainly for scale_min/max mapping,
        # but the image provided is ALREADY the cropped track.
        left_value = float(track.get('leftValue', 0))
        right_value = float(track.get('rightValue', 100))
        
        edit_y = int(data.get('editY', h_img // 2))
        edit_x = float(data.get('editX', w_img // 2))
        window_size = int(data.get('windowSize', 50))
        curve_type = data.get('curveType', 'GR').upper()
        mode = data.get('mode', 'green').lower()
        
        is_crop = data.get('isCrop', False)
        crop_origin_x = int(data.get('cropOriginX', 0))
        edit_relative_y = int(data.get('editRelativeY', -1))

        ui_filters = data.get('uiFilters') or data.get('ui_filters') or {}
        try:
            ui_filters = {
                'invert': bool(ui_filters.get('invert')),
                'contrast': bool(ui_filters.get('contrast')),
            }
        except Exception:
            ui_filters = {}

        request_max_snap_dist = data.get('maxSnapDist')
        try:
            snap_threshold = float(data.get('snapThreshold', 1.05))
        except Exception:
            snap_threshold = 1.05

        if is_crop:
        # Image is already cropped to the track/ROI
            track_crop = img
            # If editRelativeY is provided, use it; otherwise assume center
            if edit_relative_y >= 0:
                edit_row_in_window = edit_relative_y
            else:
                edit_row_in_window = h_img // 2
            
            # For multiscale tracer, the "track" is the whole image provided
            # effectively left_x=0, right_x=w_img in local coords
            # BUT we need to be careful if the user provided leftX/rightX in the request
            # they might be absolute. For the tracer, we just need scale mapping.
            pass
        else:
            # Standard mode: image is the full page, we crop it
            left_x = int(track.get('leftX', 0))
            right_x = int(track.get('rightX', w_img))
            
            left_x = max(0, min(left_x, w_img - 1))
            right_x = max(left_x + 1, min(right_x, w_img))
            
            # Extract vertical window around edit point
            y_start = max(0, edit_y - window_size // 2)
            y_end = min(h_img, edit_y + window_size // 2)
            
            track_crop = img[y_start:y_end, left_x:right_x]
            
            edit_row_in_window = edit_y - y_start
            
            # We want to add this offset back to the result
            crop_origin_x = left_x
            
        track_proc = track_crop
        x_scale_factor = 1.0
        if track_crop.size == 0:
            return jsonify({'success': False, 'error': 'Empty track region'})
        try:
            track_proc, x_scale_factor = enhance_curve_roi(track_crop)
        except Exception:
            track_proc = track_crop
            x_scale_factor = 1.0
        if x_scale_factor <= 0:
            x_scale_factor = 1.0
        
        # Build probability map for this segment
        colored_modes = {'green', 'red', 'blue', 'auto', 'cyan', 'magenta', 'yellow', 'orange', 'purple'}
        
        try:
            # ALWAYS use compute_prob_map so we get grid removal, UI filters, and enhanced detection
            # for both colored and black modes.
            mask = compute_prob_map(track_proc, mode, ui_filters=ui_filters)
        except Exception as _prob_err:
            print(f'refine_edit: compute_prob_map failed ({_prob_err}), returning edit position')
            return jsonify({'success': True, 'refinedX': float(edit_x), 'confidence': 0.0,
                            'originalX': float(edit_x), 'refinedPath': []})

        weight_map = mask.astype(np.float32)
        # Calculate distance transform for weighting if needed (compute_prob_map already does this internally,
        # but we do it here for the refined_centroid helper)
        try:
            bin_mask = (mask > 0).astype(np.uint8)
            dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 3)
            if dist is not None and dist.size:
                maxv = float(np.max(dist))
                if maxv > 1e-6:
                    weight_map = (dist / maxv) * 255.0
        except Exception:
            weight_map = mask.astype(np.float32)
        
        # Run multi-scale tracing on this segment
        # Use parameters consistent with the main digitization loop
        smooth_l = 0.001 if (mode in colored_modes or curve_type == 'GR') else 0.02
        max_s = 200 if mode in colored_modes else (30 if curve_type == 'GR' else 50)
        curv_l = 0.001 if (mode in colored_modes or curve_type == 'GR') else 0.005
        
        # For non-colored modes (black), we need higher max_step to track large excursions
        # and slightly lower smoothing so it doesn't just average through peaks
        max_step_eff = max_s if curve_type == 'GR' else 35
        smooth_eff = smooth_l if curve_type == 'GR' else 0.005
        curv_eff = curv_l if curve_type == 'GR' else 0.001
        
        try:
            xs_refined, confidence = trace_curve_multiscale(
                mask,
                scale_min=left_value,
                scale_max=right_value,
                curve_type=curve_type,
                max_step=max_step_eff,
                smooth_lambda=smooth_eff,
                curv_lambda=curv_eff,
                hot_side=None,
            )
        except Exception as _trace_err:
            print(f'refine_edit: trace_curve_multiscale failed ({_trace_err}), returning edit position')
            return jsonify({'success': True, 'refinedX': float(edit_x), 'confidence': 0.0,
                            'originalX': float(edit_x), 'refinedPath': []})
        
        # UNIVERSAL GAP FILLING (Same as main loop):
        # Fill gaps from grid removal so the refined path is continuous
        if xs_refined is not None and xs_refined.size > 0:
            try:
                s = pd.Series(xs_refined)
                # Use at least 25px gap fill to bridge grid cuts
                max_gap = 25
                s = s.interpolate(method='linear', limit_direction='both', limit=max_gap, limit_area=None)
                if s.isna().any():
                    s = s.ffill(limit=max_gap).bfill(limit=max_gap)
                xs_refined = s.to_numpy(dtype=np.float32)
            except Exception:
                pass

        # Helper for centroid refinement
        def get_refined_centroid(x_viterbi, row_idx):
            try:
                search_r = 10
                center_int = int(round(x_viterbi))
                r_start = max(0, center_int - search_r)
                r_end = min(mask.shape[1], center_int + search_r + 1)
                
                if r_end > r_start:
                    row_vals = weight_map[row_idx, r_start:r_end].astype(float)
                    row_vals = cv2.GaussianBlur(row_vals.reshape(1, -1), (3, 1), 0).flatten()
                    
                    total_mass = np.sum(row_vals)
                    if total_mass > 1e-3:
                        indices = np.arange(len(row_vals))
                        com_local = np.sum(indices * row_vals) / total_mass
                        refined_pos = r_start + com_local
                        
                        # Only accept if close to Viterbi
                        if abs(refined_pos - x_viterbi) < search_r:
                            return refined_pos
            except Exception:
                pass
            return x_viterbi

        if not (pixel_perfect and mode not in colored_modes):
            try:
                # Mathmatical centering pass for thick black ink inside refine segment
                xs_refined = recenter_black_trace_post_dp(track_proc, xs_refined)
            except Exception:
                pass

        if 0 <= edit_row_in_window < len(xs_refined) and np.isfinite(xs_refined[edit_row_in_window]):
            # Refine the specific click point
            refined_x_local = get_refined_centroid(xs_refined[edit_row_in_window], edit_row_in_window)

            try:
                w_local = mask.shape[1]
                edit_x_local = float(edit_x) - float(crop_origin_x)
                edit_x_local_proc = edit_x_local * x_scale_factor
                if w_local > 2 and np.isfinite(edit_x_local_proc):
                    try:
                        max_snap = float(request_max_snap_dist) if request_max_snap_dist is not None else 15.0
                        max_snap = float(np.clip(max_snap, 4.0, 60.0))
                    except Exception:
                        max_snap = 15.0
                    max_snap_proc = max_snap * x_scale_factor

                    search_r = int(max(4, min(80, round(max_snap_proc + 6))))
                    center_int = int(round(edit_x_local_proc))
                    r_start = max(0, center_int - search_r)
                    r_end = min(w_local, center_int + search_r + 1)
                    if r_end > r_start:
                        row_vals = weight_map[edit_row_in_window, r_start:r_end].astype(np.float32)
                        if row_vals.size > 0:
                            local_best = int(np.argmax(row_vals))
                            local_best_x = float(r_start + local_best)

                            dp_x = float(refined_x_local)
                            local_score = float(row_vals[local_best])
                            dp_score = float(weight_map[edit_row_in_window, int(np.clip(round(dp_x), 0, w_local - 1))])

                            if abs(local_best_x - edit_x_local_proc) <= max_snap_proc and (
                                abs(dp_x - edit_x_local_proc) > max_snap_proc or local_score > dp_score * 1.10
                            ):
                                refined_x_local = get_refined_centroid(local_best_x, edit_row_in_window)
            except Exception:
                pass

            refined_x_orig = refined_x_local / x_scale_factor
            refined_x = crop_origin_x + refined_x_orig  # Convert back to full image coordinates
            conf = float(confidence[edit_row_in_window]) if edit_row_in_window < len(confidence) else 0.5
        else:
            # Fallback: return the user's edit position
            refined_x = edit_x
            conf = 0.0
        
        # Prepare the full path segment relative to the crop, with centroid refinement applied to ALL points
        refined_path_segment = []
        if xs_refined is not None and len(xs_refined) > 0:
            for i, val in enumerate(xs_refined):
                if np.isfinite(val):
                    # Refine every point for sub-pixel accuracy and smoother crests
                    final_val = get_refined_centroid(val, i)
                    refined_path_segment.append({
                        'offsetY': i,
                        'x': float(crop_origin_x + (final_val / x_scale_factor)),
                        'confidence': float(confidence[i]) if confidence is not None and i < len(confidence) else 0.0
                    })

        return jsonify({
            'success': True,
            'refinedX': float(refined_x),
            'confidence': conf,
            'originalX': float(edit_x),
            'refinedPath': refined_path_segment
        })
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        if is_prod:
            tb = None
            print(f"Auto capture error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': tb
        })


@app.route('/api/log_correction', methods=['POST'])
def log_correction():
    try:
        data = request.json or {}
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

        now = datetime.utcnow()
        date_str = now.strftime('%Y-%m-%d')
        ts = now.strftime('%Y%m%dT%H%M%S.%fZ')
        event_id = data.get('event_id') or str(uuid.uuid4())

        base_dir = Path(__file__).resolve().parent
        out_dir = base_dir / 'corrections' / date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        image_path = None
        image_data = data.get('image')
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            try:
                header, b64 = image_data.split(',', 1)
                ext = 'jpg'
                if 'image/png' in header:
                    ext = 'png'
                raw = base64.b64decode(b64)
                image_path = out_dir / f'{ts}_{event_id}.{ext}'
                image_path.write_bytes(raw)
            except Exception:
                image_path = None

        record = {
            'event_id': event_id,
            'ts_utc': now.isoformat() + 'Z',
            'type': data.get('type'),
            'curve_id': data.get('curve_id'),
            'curve_index': data.get('curve_index'),
            'mode': data.get('mode'),
            'track': data.get('track'),
            'depth_index': data.get('depth_index'),
            'depth': data.get('depth'),
            'before': data.get('before'),
            'after': data.get('after'),
            'refine': data.get('refine'),
            'fallback': data.get('fallback'),
            'notes': data.get('notes'),
            'image_path': str(image_path) if image_path else None,
        }

        jsonl_path = out_dir / 'corrections.jsonl'
        with jsonl_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        return jsonify({'success': True, 'event_id': event_id})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


def _black_capture_output_dir(now_utc: datetime) -> Path:
    date_str = now_utc.strftime('%Y-%m-%d')
    out_dir = Path(__file__).resolve().parent / 'corrections' / date_str / 'black_segments'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _safe_capture_component(value, default='capture') -> str:
    text = '' if value is None else str(value).strip()
    if not text:
        return default
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "", text)
    text = text.strip("_-")
    return (text[:80] or default)


def _write_capture_image(image_data, out_dir: Path, stem: str) -> Optional[Path]:
    if not isinstance(image_data, str) or not image_data.startswith('data:image'):
        return None

    try:
        header, b64 = image_data.split(',', 1)
        ext = 'jpg'
        if 'image/png' in header:
            ext = 'png'
        elif 'image/webp' in header:
            ext = 'webp'

        raw = base64.b64decode(b64)
        image_path = out_dir / f'{stem}.{ext}'
        image_path.write_bytes(raw)
        return image_path
    except Exception:
        return None


def _write_trace_debug_artifacts(trace_debug, out_dir: Path, stem: str) -> Dict[str, Any]:
    """Persist trace-debug image layers next to a bad black segment capture."""
    if not isinstance(trace_debug, dict):
        return {}

    debug_dir = out_dir / f'{stem}_trace_debug'
    image_paths: Dict[str, str] = {}
    meta_paths: Dict[str, str] = {}

    images = trace_debug.get('images')
    if isinstance(images, dict):
        for name, image_data in images.items():
            safe_name = _safe_capture_component(name, default='debug_image')
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
                image_path = _write_capture_image(image_data, debug_dir, safe_name)
                if image_path:
                    image_paths[str(name)] = str(image_path)
            except Exception:
                continue

    meta = {}
    for key in ('curve', 'curve_type', 'mode', 'metrics', 'components_top', 'error'):
        if key in trace_debug:
            meta[key] = trace_debug.get(key)
    if meta:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            meta_path = debug_dir / 'trace_debug.json'
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
            meta_paths['trace_debug'] = str(meta_path)
        except Exception:
            pass

    if not image_paths and not meta_paths:
        return {}
    return {
        'debug_dir': str(debug_dir),
        'images': image_paths,
        'meta': meta_paths,
    }


@app.route('/api/save_bad_black_segment', methods=['POST'])
def save_bad_black_segment():
    try:
        data = request.json or {}
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

        now = datetime.utcnow()
        ts = now.strftime('%Y%m%dT%H%M%S.%fZ')
        capture_id = data.get('capture_id') or str(uuid.uuid4())
        out_dir = _black_capture_output_dir(now)
        curve_slug = _safe_capture_component(data.get('curve_id'), default='BLACK')
        stem = f'{ts}_{curve_slug}_{capture_id}'

        trace_points = data.get('trace_points')
        if not isinstance(trace_points, list):
            trace_points = []

        image_path = _write_capture_image(data.get('image'), out_dir, stem)
        trace_debug_artifacts = _write_trace_debug_artifacts(data.get('trace_debug'), out_dir, stem)
        payload_path = out_dir / f'{stem}.json'
        summary_path = out_dir / 'captures.jsonl'
        user = _current_user(require_access=False)

        payload_record = {
            'capture_id': capture_id,
            'ts_utc': now.isoformat() + 'Z',
            'curve_id': data.get('curve_id'),
            'trace_key': data.get('trace_key'),
            'status': data.get('status') or 'needs_review',
            'capture_source': data.get('capture_source'),
            'capture_event': data.get('capture_event'),
            'capture_session_id': data.get('capture_session_id'),
            'auto_capture': bool(data.get('auto_capture')),
            'download_format': data.get('download_format'),
            'notes': data.get('notes'),
            'trace_rows': len(trace_points),
            'image_path': str(image_path) if image_path else None,
            'trace_debug_artifacts': trace_debug_artifacts or None,
            'user_id': user['id'] if user else None,
            'payload': data,
        }

        payload_path.write_text(
            json.dumps(payload_record, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        summary_record = {
            'capture_id': capture_id,
            'ts_utc': payload_record['ts_utc'],
            'curve_id': payload_record['curve_id'],
            'trace_key': payload_record['trace_key'],
            'status': payload_record['status'],
            'capture_source': payload_record['capture_source'],
            'capture_event': payload_record['capture_event'],
            'capture_session_id': payload_record['capture_session_id'],
            'auto_capture': payload_record['auto_capture'],
            'download_format': payload_record['download_format'],
            'notes': payload_record['notes'],
            'trace_rows': payload_record['trace_rows'],
            'image_path': payload_record['image_path'],
            'trace_debug_artifacts': payload_record['trace_debug_artifacts'],
            'payload_path': str(payload_path),
            'user_id': payload_record['user_id'],
        }
        with summary_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(summary_record, ensure_ascii=False) + '\n')

        return jsonify({
            'success': True,
            'capture_id': capture_id,
            'trace_rows': len(trace_points),
            'payload_path': str(payload_path),
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/learn_from_user', methods=['POST'])
def learn_from_user():
    """Record user curve adjustments for learning (Phase 1)"""
    data = request.json or {}
    
    required_fields = ['curve_type', 'original_params', 'user_params']
    for field in required_fields:
        if field not in data:
            return jsonify({
                'success': False, 
                'error': f'Missing required field: {field}'
            }), 400
    
    try:
        curve_type = data['curve_type']
        original_params = data['original_params']
        user_params = data['user_params']
        quality_score = data.get('quality_score', 1.0)
        image_context = data.get('image_context')
        
        # Validate curve type
        valid_types = ['GR', 'RHOB', 'NPHI', 'DT', 'CALI', 'SP', 'OTHER']
        if curve_type not in valid_types:
            curve_type = 'OTHER'
        
        # Record the adjustment
        tracker.record_adjustment(
            curve_type=curve_type,
            original_params=original_params,
            user_params=user_params,
            quality_score=quality_score,
            image_context=image_context
        )
        
        # Return stats for feedback
        stats = tracker.get_stats(curve_type)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'message': f'Adjustment recorded for {curve_type}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/user_preferences', methods=['GET'])
def get_user_preferences():
    """Get user preference statistics"""
    curve_type = request.args.get('curve_type')
    
    if curve_type:
        adjustments = tracker.get_adjustments(curve_type)
        stats = tracker.get_stats(curve_type)
        return jsonify({
            'curve_type': curve_type,
            'adjustments': adjustments,
            'stats': stats
        })
    else:
        all_adjustments = tracker.get_all_adjustments()
        all_stats = {ct: tracker.get_stats(ct) for ct in all_adjustments.keys()}
        return jsonify({
            'all_adjustments': all_adjustments,
            'all_stats': all_stats
        })


@app.route('/api/clear_preferences', methods=['POST'])
def clear_preferences():
    """Clear user preferences (for testing/reset)"""
    curve_type = request.json.get('curve_type')
    
    if curve_type:
        tracker.adjustments[curve_type] = []
    else:
        tracker.adjustments.clear()
    
    tracker.save_preferences()
    
    return jsonify({
        'success': True,
        'message': f'Preferences cleared for {curve_type or "all curves"}'
    })


@app.route('/api/batch_digitize', methods=['POST'])
def batch_digitize():
    """Process multiple TIFF images for ML training dataset generation.

    Expects JSON with:
      - jobs: list of { image, config, preview_filters, detected_text, header_metadata }
      - export_format: 'json' (default) or 'las'
      - include_images: bool (include cropped panel images in output)

    Returns:
      - results: list of digitization results with metadata
      - summary: { total, success, failed }
    """
    data = request.json or {}
    jobs = data.get('jobs', [])
    export_format = data.get('export_format', 'json')
    include_images = data.get('include_images', True)

    if not jobs:
        return jsonify({'success': False, 'error': 'No jobs provided'}), 400

    results = []
    success_count = 0
    failed_count = 0

    for idx, job in enumerate(jobs):
        try:
            image_data = job.get('image')
            image_path = job.get('image_path')
            config = job.get('config')
            preview_filters = job.get('preview_filters', {})
            detected_text = job.get('detected_text', {})
            header_metadata = job.get('header_metadata')

            if not image_data and not image_path:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': 'Missing image or image_path'
                })
                failed_count += 1
                continue

            if not config:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': 'Missing config'
                })
                failed_count += 1
                continue

            # Load image (path or base64)
            img = None
            if image_path:
                if os.path.exists(image_path):
                    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                else:
                    results.append({
                        'index': idx,
                        'success': False,
                        'error': f'Image path not found: {image_path}'
                    })
                    failed_count += 1
                    continue
            elif image_data:
                try:
                    # Decode image
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    img_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception as e:
                    results.append({
                        'index': idx,
                        'success': False,
                        'error': f'Failed to decode image: {e}'
                    })
                    failed_count += 1
                    continue

            if img is None:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': 'Failed to decode image'
                })
                failed_count += 1
                continue

            # Extract config
            depth_cfg = config['depth']
            curves = (config['curves'] or [])[:6]
            gopt = config.get('global_options', {})

            null_val = float(gopt.get('null', -999.25))
            downsample = int(gopt.get('downsample', 1))
            blur = int(gopt.get('blur', 3))
            min_run = int(gopt.get('min_run', 2))
            smooth_window = int(gopt.get('smooth_window', 5))

            H, W, _ = img.shape
            top = max(0, int(depth_cfg['top_px']))
            bot = min(H, int(depth_cfg['bottom_px']))
            top_depth = float(depth_cfg['top_depth'])
            bottom_depth = float(depth_cfg['bottom_depth'])
            depth_unit = depth_cfg.get('unit', 'FT')

            nrows = bot - top
            base_depth = compute_depth_vector(nrows, top_depth, bottom_depth)

            curve_data = {}
            curve_traces = {}

            for c in curves:
                name = c.get('las_mnemonic') or c.get('name')
                unit = c.get('las_unit') or c.get('unit', '')
                left_px = int(c['left_px'])
                right_px = int(c['right_px'])
                left_value = float(c['left_value'])
                right_value = float(c['right_value'])
                mode = c.get('mode', 'black')
                hot_side = c.get('hot_side')
                pixel_perfect = bool(c.get('pixel_perfect'))
                trace_mode = c.get('trace_mode')
                align_channels = bool(c.get('align_channels'))
                preserve_wiggles = bool(c.get('preserve_wiggles'))
                crest_boost = bool(c.get('crest_boost'))

                if not hot_side and np.isfinite(left_value) and np.isfinite(right_value):
                    hot_side = 'right' if right_value >= left_value else 'left'

                left_px = max(0, min(W - 1, left_px))
                right_px = max(0, min(W, right_px))

                if right_px <= left_px:
                    continue

                top_clamped = max(0, min(H - 1, int(top)))
                bot_clamped = max(0, min(H, int(bot)))

                if bot_clamped <= top_clamped:
                    continue

                roi = img[top_clamped:bot_clamped, left_px:right_px]
                if roi is None or roi.size == 0:
                    continue

                if align_channels:
                    roi = align_rgb_channels(roi)
                if blur > 0:
                    bb = blur + 1 if blur % 2 == 0 else blur
                    roi = cv2.GaussianBlur(roi, (bb, bb), 0)

                mask = compute_prob_map(roi, mode=mode, ui_filters=preview_filters)
                curve_type = c.get('type', 'GR')

                colored_modes = {"green", "red", "blue", "auto", "cyan", "magenta", "yellow", "orange", "purple"}

                if pixel_perfect and mode in colored_modes:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    if trace_mode == "skeleton_path":
                        xs, confidence = trace_curve_skeleton_path(mask)
                    else:
                        xs, confidence = trace_curve_pixel_perfect(
                            mask, grayscale=gray_roi, bgr=roi, hot_side=hot_side,
                            preserve_wiggles=preserve_wiggles, crest_boost=crest_boost,
                        )
                    width_px = mask.shape[1]
                    if xs.size:
                        s = pd.Series(xs)
                        s = s.interpolate(method='linear', limit_direction='both', limit=max(10, int(xs.size * 0.02)))
                        xs = s.to_numpy(dtype=np.float32)
                    prob = mask.astype(np.float32) / 255.0
                    if crest_boost:
                        xs = _postprocess_missed_peaks(mask, prob, xs, search_radius=40, min_prob=0.004)
                    else:
                        xs = _postprocess_missed_peaks(mask, prob, xs, search_radius=30, min_prob=0.008)
                elif mode in colored_modes:
                    mask_orig = mask
                    h_orig, w_orig = mask.shape
                    mask = cv2.resize(mask, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LINEAR)
                    max_step_dp = 200 * 2
                    dp_smooth_lambda = 0.001
                    dp_curv_lambda = 0.001

                    xs_dp, conf_dp = trace_curve_with_dp(
                        mask, scale_min=left_value, scale_max=right_value,
                        curve_type=curve_type, max_step=max_step_dp,
                        smooth_lambda=dp_smooth_lambda, curv_lambda=dp_curv_lambda,
                        hot_side=hot_side,
                    )

                    h_mask, w_mask = mask.shape
                    prob_map = mask.astype(np.float32) / 255.0
                    xs = np.full(h_mask, np.nan, dtype=np.float32)

                    for row in range(h_mask):
                        dp_x = xs_dp[row] if row < len(xs_dp) else None
                        if np.isnan(dp_x):
                            continue

                        search_radius = 30
                        x_start = max(0, int(dp_x) - search_radius)
                        x_end = min(w_mask, int(dp_x) + search_radius + 1)
                        row_probs = prob_map[row, x_start:x_end]

                        if row_probs.size == 0:
                            continue

                        local_max_idx = np.argmax(row_probs)
                        xs[row] = x_start + local_max_idx

                    xs = xs / 2.0

                    mask = mask_orig
                else:
                    xs, confidence = trace_curve_with_dp(
                        mask, scale_min=left_value, scale_max=right_value,
                        curve_type=curve_type, max_step=3, smooth_lambda=0.5, curv_lambda=0.05,
                        hot_side=hot_side,
                    )

                if xs.size != nrows:
                    if xs.size > nrows:
                        xs = xs[:nrows]
                    else:
                        xs = np.pad(xs, (0, nrows - xs.size), mode='edge')

                xs = pd.Series(xs).interpolate(method='linear', limit_direction='both', limit=10).to_numpy()

                scale_range = right_value - left_value
                if scale_range == 0:
                    scale_range = 1.0

                values = left_value + (xs / (right_px - left_px)) * scale_range
                values = np.where(np.isnan(values), null_val, values)

                # Clean NaN/inf from xs and values before converting to list
                xs_clean = np.where(np.isnan(xs) | np.isinf(xs), null_val, xs)
                values_clean = np.where(np.isnan(values) | np.isinf(values), null_val, values)

                if downsample > 1:
                    values_clean = values_clean[::downsample]
                    base_depth = base_depth[::downsample]

                curve_data[name] = values_clean.tolist()
                curve_traces[name] = xs_clean.tolist()

            # Clean NaN/inf from depth values before converting to list
            base_depth_clean = np.where(np.isnan(base_depth) | np.isinf(base_depth), null_val, base_depth)

            result = {
                'index': idx,
                'success': True,
                'depth': {
                    'top_px': top,
                    'bottom_px': bot,
                    'top_depth': top_depth,
                    'bottom_depth': bottom_depth,
                    'unit': depth_unit,
                    'values': base_depth_clean.tolist(),
                },
                'curves': curve_data,
                'curve_traces': curve_traces,
                'metadata': {
                    'image_width': W,
                    'image_height': H,
                    'curve_count': len(curve_data),
                    'null_value': null_val,
                }
            }

            if include_images:
                ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    result['image'] = base64.b64encode(buf.tobytes()).decode('utf-8')

            if header_metadata:
                result['header_metadata'] = header_metadata

            results.append(result)
            success_count += 1

        except Exception as e:
            results.append({
                'index': idx,
                'success': False,
                'error': str(e)
            })
            failed_count += 1
            continue

    return jsonify({
        'success': True,
        'results': results,
        'summary': {
            'total': len(jobs),
            'success': success_count,
            'failed': failed_count
        }
    })


@app.route('/api/export_training_data', methods=['POST'])
def export_training_data():
    """Export digitized data as ML-ready training dataset.

    Expects JSON with:
      - data: list of digitization results (from batch_digitize or digitize)
      - format: 'json' (default) or 'csv'
      - include_metadata: bool (include image metadata)

    Returns:
      - JSON or CSV formatted training data with:
        - image_id
        - depth_values
        - curve_data (pixel traces and value mappings)
        - curve_metadata (type, scale, parameters)
    """
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    data = request.json or {}
    dataset = data.get('data', [])
    export_format = data.get('format', 'json')
    include_metadata = data.get('include_metadata', True)

    if not dataset:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    training_data = []

    for idx, item in enumerate(dataset):
        if not item.get('success'):
            continue

        depth_info = item.get('depth', {})
        curves = item.get('curves', {})
        curve_traces = item.get('curve_traces', {})
        metadata = item.get('metadata', {})
        header_metadata = item.get('header_metadata', {})

        # Clean depth values of NaN/inf
        depth_values_raw = depth_info.get('values', [])
        depth_values_clean = [None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for v in depth_values_raw]

        training_item = {
            'image_id': f"img_{idx:04d}",
            'depth': {
                'top_px': depth_info.get('top_px'),
                'bottom_px': depth_info.get('bottom_px'),
                'top_depth': depth_info.get('top_depth'),
                'bottom_depth': depth_info.get('bottom_depth'),
                'unit': depth_info.get('unit', 'FT'),
                'values': depth_values_clean,
            },
            'curves': []
        }

        for curve_name, values in curves.items():
            trace = curve_traces.get(curve_name, [])
            # Filter out NaN and inf values for JSON serialization
            trace_clean = [None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for v in trace]
            values_clean = [None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for v in values]

            training_item['curves'].append({
                'name': curve_name,
                'pixel_trace': trace_clean,
                'depth_values': values_clean,
                'sample_count': len(values),
            })

        if include_metadata:
            # Clean metadata values
            training_item['metadata'] = {
                'image_width': metadata.get('image_width'),
                'image_height': metadata.get('image_height'),
                'curve_count': metadata.get('curve_count'),
                'null_value': metadata.get('null_value'),
            }
            if header_metadata:
                training_item['header_metadata'] = header_metadata

        training_data.append(training_item)

    if export_format == 'csv':
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        writer.writerow(['image_id', 'depth_top_px', 'depth_bottom_px', 'depth_top_depth',
                        'depth_bottom_depth', 'depth_unit', 'curve_name', 'pixel_trace',
                        'depth_values'])

        for item in training_data:
            depth = item['depth']
            for curve in item['curves']:
                # Filter out NaN and inf values for JSON serialization
                pixel_trace = curve['pixel_trace']
                depth_values = curve['depth_values']

                # Replace NaN with null for JSON compatibility
                pixel_trace_clean = [None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for v in pixel_trace]
                depth_values_clean = [None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for v in depth_values]

                writer.writerow([
                    item['image_id'],
                    depth['top_px'],
                    depth['bottom_px'],
                    depth['top_depth'],
                    depth['bottom_depth'],
                    depth['unit'],
                    curve['name'],
                    json.dumps(pixel_trace_clean),
                    json.dumps(depth_values_clean),
                ])

        csv_output = output.getvalue()
        return jsonify({
            'success': True,
            'format': 'csv',
            'data': csv_output,
            'count': len(training_data)
        })

    # Use custom encoder to handle numpy types and NaN/inf
    response_data = {
        'success': True,
        'format': 'json',
        'data': training_data,
        'count': len(training_data)
    }

    json_str = json.dumps(response_data, cls=NpEncoder)
    return Response(json_str, mimetype='application/json')


_ML_CURVE_TRACE_MODEL_CACHE = {
     'model_path': None,
     'model': None,
     'meta': None,
 }


if TORCH_AVAILABLE:
     class _CurveTraceNet(nn.Module):
         def __init__(self, in_ch: int = 1, base: int = 16):
             super().__init__()
             self.enc = nn.Sequential(
                 nn.Conv2d(in_ch, base, 3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(base, base, 3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2),
                 nn.Conv2d(base, base * 2, 3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(base * 2, base * 2, 3, padding=1),
                 nn.ReLU(inplace=True),
             )
             self.dec = nn.Sequential(
                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                 nn.Conv2d(base * 2, base, 3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(base, base, 3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(base, 1, 1),
             )

         def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
             feat = self.enc(x)
             logits = self.dec(feat).squeeze(1)
             prob = torch.softmax(logits, dim=-1)
             xs = torch.linspace(0.0, 1.0, logits.shape[-1], device=logits.device)
             pred = (prob * xs).sum(dim=-1)
             return pred


def _ml_decode_image_data_url(image_data: str) -> np.ndarray:
     img_data = image_data.split(',', 1)[1]
     img_bytes = base64.b64decode(img_data)
     nparr = np.frombuffer(img_bytes, np.uint8)
     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
     if img is None:
         raise ValueError('Failed to decode image')
     return img


def _ml_load_curve_trace_model(model_path: str, device: str = 'cpu') -> Tuple['torch.nn.Module', Dict]:
     cache = _ML_CURVE_TRACE_MODEL_CACHE
     if cache.get('model') is not None and cache.get('model_path') == model_path:
         return cache['model'], (cache.get('meta') or {})

     payload = torch.load(model_path, map_location=device)
     state_dict = payload.get('state_dict')
     if not state_dict:
         raise ValueError('Model file missing state_dict')

     model = _CurveTraceNet()
     model.load_state_dict(state_dict)
     model.eval()
     model.to(device)

     meta = {
         'input_h': int(payload.get('input_h', 256)),
         'input_w': int(payload.get('input_w', 128)),
         'curve': payload.get('curve'),
     }

     cache['model_path'] = model_path
     cache['model'] = model
     cache['meta'] = meta

     return model, meta


def _ml_resolve_curve_trace_model_path(requested_path: Optional[str]) -> str:
     if requested_path:
         return requested_path

     env_path = os.environ.get('CURVE_TRACE_MODEL_PATH')
     if env_path:
         return env_path

     candidates = [
         Path(__file__).with_name('curve_trace_model.pt'),
         Path.cwd() / 'curve_trace_model.pt',
     ]

     try:
         desktop_dir = Path(__file__).resolve().parent.parent
         candidates.append(desktop_dir / 'TestTiflas' / 'curve_trace_model.pt')
     except Exception:
         pass

     for p in candidates:
         try:
             if p.exists():
                 return str(p)
         except Exception:
             continue

     return str(candidates[0])


@app.route('/api/download_las_zip', methods=['POST'])
def download_las_zip():
    """Generate a ZIP file containing individual LAS files for each curve."""
    data = request.json or {}
    
    depths = data.get('depths')
    curves = data.get('curves')
    header_metadata = data.get('header_metadata') or {}
    depth_unit = data.get('depth_unit', 'FT')
    
    if not depths or not curves:
        return jsonify({'error': 'Missing depth or curve data'}), 400
        
    try:
        depth_arr = np.array(depths, dtype=np.float32)
        
        # --- MERGE WRAPPED CURVES ---
        processed_curves = {}
        wrap_curves = {}
        
        # Separate normal curves from wrapped curves (case-insensitive suffix)
        for curve_name, curve_info in curves.items():
            c_lower = curve_name.lower()
            if c_lower.endswith('_wrap') or c_lower.endswith('_wrapped'):
                wrap_curves[curve_name] = curve_info
            else:
                # Make a deep copy to avoid mutating original dict if reused
                processed_curves[curve_name] = {
                    'unit': curve_info.get('unit', ''),
                    'values': list(curve_info.get('values', []))
                }
                
        # Apply wrapped curve data to main curves
        for wrap_name, wrap_info in wrap_curves.items():
            c_lower = wrap_name.lower()
            if c_lower.endswith('_wrapped'):
                main_name_lower = c_lower[:-8]
            else:
                main_name_lower = c_lower[:-5]
                
            # Find actual main curve name (case-insensitive)
            actual_main_name = None
            for name in processed_curves.keys():
                if name.lower() == main_name_lower:
                    actual_main_name = name
                    break
                    
            if actual_main_name:
                main_vals = np.array(processed_curves[actual_main_name]['values'], dtype=np.float32)
                wrap_vals = np.array(wrap_info.get('values', []), dtype=np.float32)
                
                # Check for length mismatch (shouldn't happen but be safe)
                min_len = min(len(main_vals), len(wrap_vals))
                
                # A value is valid if it's not nan and not the standard LAS null value
                valid_wrap_mask = ~np.isnan(wrap_vals[:min_len]) & (wrap_vals[:min_len] != -999.25)
                
                main_vals[:min_len][valid_wrap_mask] = wrap_vals[:min_len][valid_wrap_mask]
                processed_curves[actual_main_name]['values'] = main_vals.tolist()
                
                # Inherit units if main curve doesn't have one and wrapped does
                if not processed_curves[actual_main_name]['unit'] and wrap_info.get('unit'):
                    processed_curves[actual_main_name]['unit'] = wrap_info['unit']
                    
        # Update curves dictionary for the loop below
        curves = processed_curves
        # ----------------------------
        
        # Create in-memory ZIP
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            base_name = build_las_filename_from_metadata(header_metadata, default_name='digitized_log').replace('.las', '')
            
            for curve_name, curve_info in curves.items():
                # Prepare single-curve data dict
                single_curve_data = {
                    curve_name: curve_info
                }
                
                # Generate LAS content
                las_content = write_las_simple(depth_arr, single_curve_data, depth_unit, header_metadata)
                
                # Add to ZIP
                # Filename: WellName_CurveName.las
                # Sanitize curve name for filename
                safe_curve_name = "".join([c for c in curve_name if c.isalnum() or c in ('_', '-')])
                filename = f"{base_name}_{safe_curve_name}.las"
                zf.writestr(filename, las_content)
                
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{base_name}_curves.zip"
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml_predict_curve_trace', methods=['POST'])
def ml_predict_curve_trace():
     if not TORCH_AVAILABLE:
         return jsonify({'success': False, 'error': 'torch is not available in this environment'}), 400

     data = request.json or {}
     image_data = data.get('image')
     roi = data.get('roi') or {}

     if not image_data:
         return jsonify({'success': False, 'error': 'Missing image'}), 400

     model_path = _ml_resolve_curve_trace_model_path(data.get('model_path'))

     device = data.get('device') or 'cpu'

     try:
         img = _ml_decode_image_data_url(image_data)
         H, W = img.shape[:2]

         top_px = int(roi.get('top_px', 0))
         bottom_px = int(roi.get('bottom_px', H))
         left_px = int(roi.get('left_px', 0))
         right_px = int(roi.get('right_px', W))

         top_px = max(0, min(H - 1, top_px))
         bottom_px = max(0, min(H, bottom_px))
         left_px = max(0, min(W - 1, left_px))
         right_px = max(0, min(W, right_px))

         if bottom_px <= top_px or right_px <= left_px:
             return jsonify({'success': False, 'error': 'Invalid ROI'}), 400

         roi_img = img[top_px:bottom_px, left_px:right_px]
         if roi_img is None or roi_img.size == 0:
             return jsonify({'success': False, 'error': 'Empty ROI'}), 400

         roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

         if not Path(model_path).exists():
             return jsonify({
                 'success': False,
                 'error': f'Model file not found: {model_path}',
             }), 400

         model, meta = _ml_load_curve_trace_model(model_path=model_path, device=device)
         in_h = int(meta.get('input_h', 256))
         in_w = int(meta.get('input_w', 128))

         roi_resized = cv2.resize(roi_gray, (in_w, in_h), interpolation=cv2.INTER_AREA)
         x = (roi_resized.astype(np.float32) / 255.0)[None, None, :, :]
         x_t = torch.from_numpy(x).to(device)

         with torch.no_grad():
             pred_norm = model(x_t)[0].detach().cpu().numpy().astype(np.float32)

         roi_w = int(right_px - left_px)
         roi_h = int(bottom_px - top_px)
         if roi_w <= 1 or roi_h <= 1:
             return jsonify({'success': False, 'error': 'ROI too small'}), 400

         pred_px = pred_norm * float(roi_w - 1)

         src_y = np.linspace(0.0, float(in_h - 1), num=in_h, dtype=np.float32)
         dst_y = np.linspace(0.0, float(in_h - 1), num=roi_h, dtype=np.float32)
         pred_px_full = np.interp(dst_y, src_y, pred_px).astype(np.float32)

         return jsonify({
             'success': True,
             'model_path': model_path,
             'model_meta': meta,
             'roi': {
                 'top_px': top_px,
                 'bottom_px': bottom_px,
                 'left_px': left_px,
                 'right_px': right_px,
             },
             'pixel_trace': pred_px_full.tolist(),
         })
     except Exception as e:
         return jsonify({'success': False, 'error': str(e)}), 500


def _decode_image_data_url(image_data):
    """Decode a data-URL image. Returns (img_bgr, error_message)."""
    if not image_data or ',' not in image_data:
        return None, 'Missing image data'
    try:
        img_payload = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(img_payload)
    except Exception:
        return None, 'Invalid image data'
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 'Could not decode image'
    return img, None


def _ocr_strings(crop):
    """Run OCR on a small crop and return a list of plain strings."""
    try:
        ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return []
        result = detect_text_vision_api(buf.tobytes()) or {}
        raw = result.get('raw') or []
        out = []
        for item in raw:
            if isinstance(item, dict):
                t = item.get('text') or item.get('description')
                if t:
                    out.append(str(t))
            elif isinstance(item, str):
                out.append(item)
        return out
    except Exception as exc:
        print(f'[detect-scale] OCR failed: {exc}')
        return []


def _document_ocr_numbers(crop_bgr):
    """Run document-level OCR (better for dense text + rotation) and return numeric entries.

    Prefers Google Vision's DOCUMENT_TEXT_DETECTION when available; falls back to the
    generic text detector (which is what our EasyOCR fallback implements anyway).

    Returns a list of dicts { value: float, text: str, x: int, y: int }.
    """
    try:
        ok, buf = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return []
        img_bytes = buf.tobytes()
    except Exception:
        return []

    numeric_entries = []

    if VISION_API_AVAILABLE and vision_client is not None:
        try:
            image = vision.Image(content=img_bytes)
            response = vision_client.document_text_detection(image=image)
            # Iterate word-level blocks, which preserves position accurately
            if response.full_text_annotation and response.full_text_annotation.pages:
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        for para in block.paragraphs:
                            for word in para.words:
                                text = ''.join(sym.text for sym in word.symbols)
                                vtx = word.bounding_box.vertices
                                if not vtx:
                                    continue
                                x = int(vtx[0].x)
                                y = int(vtx[0].y)
                                # middle-y of the word is more representative
                                ys = [v.y for v in vtx]
                                y_mid = int(sum(ys) / len(ys))
                                for tok in re.findall(r'-?\d+(?:\.\d+)?', text):
                                    try:
                                        numeric_entries.append({
                                            'value': float(tok),
                                            'text': text,
                                            'x': x,
                                            'y': y_mid,
                                        })
                                    except ValueError:
                                        continue
                return numeric_entries
        except Exception as exc:
            print(f'[document-ocr] Vision API error, falling back: {exc}')

    # Fallback: reuse the generic detector and synthesize numeric_entries from 'numbers'
    try:
        result = detect_text_vision_api(img_bytes) or {}
        for n in result.get('numbers') or []:
            try:
                numeric_entries.append({
                    'value': float(n.get('value')),
                    'text':  str(n.get('text', n.get('value'))),
                    'x':     int(n.get('x', 0)),
                    'y':     int(n.get('y', 0)),
                })
            except (TypeError, ValueError):
                continue
    except Exception as exc:
        print(f'[document-ocr] fallback failed: {exc}')

    return numeric_entries


def _llm_identify_curve(header_strings, axis_strings, existing_guess=None):
    """When local pattern matching fails, ask the LLM to pick a canonical mnemonic.

    Returns a dict { mnemonic, scale_type, left, right, unit, confidence } or None.
    Uses the existing call_ai_curve_suggestions plumbing to avoid adding a new API key.
    """
    if existing_guess and (existing_guess.get('confidence') or 0) >= 0.6:
        return None  # local detection is already confident enough

    payload = {
        "task": "identify_well_log_curve",
        "header_ocr":  header_strings or [],
        "axis_ocr":    axis_strings or [],
        "allowed_mnemonics": ["GR", "SP", "RT", "ILD", "LLD", "LLS", "MSFL",
                              "RHOB", "NPHI", "DT", "CALI", "PEF", "OTHER"],
        "instruction": (
            "Given OCR strings from a paper well-log track header and axis labels, "
            "return the most likely canonical curve mnemonic (one of allowed_mnemonics). "
            "Also return the scale_type as one of 'linear', 'log', 'centered'. "
            "For resistivity (RT/ILD/LLD/LLS/MSFL) always return scale_type='log'. "
            "For SP always return 'centered'. For everything else return 'linear'. "
            "If axis numeric labels are present, return left and right as the numeric "
            "scale endpoints you read; otherwise omit them. "
            "Respond with JSON ONLY using this schema: "
            '{"mnemonic": string, "scale_type": string, "left": number|null, '
            '"right": number|null, "unit": string|null, "confidence": number}'
        ),
    }
    try:
        result = call_ai_curve_suggestions(payload)
    except Exception as exc:
        print(f'[detect-scale] LLM call failed: {exc}')
        return None
    if not isinstance(result, dict):
        return None

    # Accept both the schema above and a wrapped {curves:[{mnemonic:...}]} shape
    candidate = result
    if 'curves' in result and isinstance(result['curves'], list) and result['curves']:
        candidate = result['curves'][0]

    mnemonic = str(candidate.get('mnemonic') or '').upper().strip()
    if not mnemonic or mnemonic == 'OTHER':
        return None
    defaults = scale_detection.get_mnemonic_defaults(mnemonic) or {}
    scale_type = (candidate.get('scale_type') or defaults.get('scale_type') or 'linear').lower()
    left = candidate.get('left', defaults.get('default_left'))
    right = candidate.get('right', defaults.get('default_right'))
    try:
        left = float(left) if left is not None else None
        right = float(right) if right is not None else None
    except (TypeError, ValueError):
        left = defaults.get('default_left')
        right = defaults.get('default_right')
    unit = candidate.get('unit') or defaults.get('unit')
    confidence = float(candidate.get('confidence') or 0.7)
    return {
        'mnemonic': mnemonic,
        'scale_type': scale_type,
        'left_value': left,
        'right_value': right,
        'unit': unit,
        'confidence': max(0.0, min(1.0, confidence)),
    }


def _detect_scale_for_region(img, region, curve_name='', xs_trace=None, use_llm=True):
    """Shared detector used by both single-curve and batch endpoints.

    Returns dict: { ok, error?, scale?, ocr_header?, ocr_axis_labels?, llm_used? }
    """
    H, W, _ = img.shape
    try:
        left = max(0, int(region.get('left_px', 0)))
        right = min(W, int(region.get('right_px', W)))
        top = max(0, int(region.get('top_px', 0)))
        bottom = min(H, int(region.get('bottom_px', H)))
    except Exception:
        return {'ok': False, 'error': 'Invalid region'}
    if right <= left or bottom <= top:
        return {'ok': False, 'error': 'Empty region'}

    panel = img[top:bottom, left:right]
    panel_h, panel_w, _ = panel.shape
    if panel_h < 2 or panel_w < 2:
        return {'ok': False, 'error': 'Panel too small'}

    header_band = panel[: max(1, panel_h // 5), :]
    axis_top    = panel[: max(1, panel_h // 12), :]
    axis_bot    = panel[-max(1, panel_h // 12):, :]

    header_strings = _ocr_strings(header_band)
    axis_strings = _ocr_strings(axis_top) + _ocr_strings(axis_bot)

    header_text = curve_name or ' '.join(header_strings)
    xs_np = None
    if isinstance(xs_trace, list) and xs_trace:
        try:
            xs_np = np.asarray(xs_trace, dtype=np.float32)
        except Exception:
            xs_np = None

    detected = scale_detection.detect_scale(
        header_text=header_text,
        axis_labels=axis_strings,
        xs_trace=xs_np,
        width_px=panel_w,
    )
    scale_dict = detected.to_dict()
    llm_used = False
    memory_used = False

    # Prior-correction memory: check if we've seen a similar OCR pattern before.
    # Stored user choices beat generic rules when available.
    try:
        uid = session.get('user_id') if 'session' in globals() else None
    except Exception:
        uid = None
    try:
        prior = corrections_store.best_suggestion(
            header_ocr=header_strings,
            axis_ocr=axis_strings,
            user_id=uid,
        )
    except Exception as exc:
        print(f'[detect-scale] corrections lookup failed: {exc}')
        prior = None

    if prior and prior.get('confidence', 0.0) > scale_dict.get('confidence', 0.0):
        uc = prior['user_choice']
        memory_used = True
        if uc.get('mnemonic'):
            scale_dict['mnemonic'] = uc['mnemonic']
        if uc.get('scale_type'):
            scale_dict['scale_type'] = uc['scale_type']
        if uc.get('left') is not None:
            scale_dict['left_value'] = uc['left']
        if uc.get('right') is not None:
            scale_dict['right_value'] = uc['right']
        if 'wrapped' in uc:
            scale_dict['wrapped'] = bool(uc['wrapped'])
        scale_dict['confidence'] = max(scale_dict.get('confidence', 0.0), prior['confidence'])
        scale_dict.setdefault('reasons', []).append(
            f"Applied prior correction (similarity={prior['similarity']:.2f}, {prior['agreeing_count']} matches)"
        )

    # LLM fallback when local detection AND memory lookup are still unsure
    if use_llm and scale_dict.get('confidence', 0.0) < 0.6:
        llm = _llm_identify_curve(header_strings, axis_strings, existing_guess=scale_dict)
        if llm:
            llm_used = True
            scale_dict['mnemonic'] = llm['mnemonic']
            scale_dict['scale_type'] = llm['scale_type']
            if llm.get('left_value') is not None:
                scale_dict['left_value'] = llm['left_value']
            if llm.get('right_value') is not None:
                scale_dict['right_value'] = llm['right_value']
            if llm.get('unit'):
                scale_dict['unit'] = llm['unit']
            # Boost confidence, but cap at 0.85 so UI still shows review for LLM-only picks
            scale_dict['confidence'] = max(scale_dict.get('confidence', 0.0), min(0.85, llm['confidence']))
            scale_dict.setdefault('reasons', []).append('LLM fallback used')

    return {
        'ok': True,
        'scale': scale_dict,
        'ocr_header': header_strings,
        'ocr_axis_labels': axis_strings,
        'llm_used': llm_used,
        'memory_used': memory_used,
    }


@app.route('/api/detect-scale', methods=['POST'])
def api_detect_scale():
    """Detect scale for a single curve region. See /api/detect-all-scales for batch."""
    data = request.json or {}
    img, err = _decode_image_data_url(data.get('image'))
    if err:
        return jsonify({'success': False, 'error': err}), 400

    res = _detect_scale_for_region(
        img,
        region=data.get('region') or {},
        curve_name=(data.get('curve_name') or '').strip(),
        xs_trace=data.get('xs_trace'),
        use_llm=bool(data.get('use_llm', True)),
    )
    if not res.get('ok'):
        return jsonify({'success': False, 'error': res.get('error')}), 400
    return jsonify({
        'success': True,
        'scale': res['scale'],
        'ocr_header': res['ocr_header'],
        'ocr_axis_labels': res['ocr_axis_labels'],
        'llm_used': res['llm_used'],
        'memory_used': res['memory_used'],
    })


def _cluster_numbers_into_columns(numbers, bandwidth_px: int = 60):
    """Group OCR numeric entries by x-position into vertical "columns".

    Args:
        numbers: list of {value, x, y, text} dicts
        bandwidth_px: merge-distance in x; entries within this are one column

    Returns:
        list of columns, each a dict { x_center, entries: [...] }, sorted by x_center.
    """
    if not numbers:
        return []
    # Sort by x so we can do a simple linear-merge pass
    xs_sorted = sorted(numbers, key=lambda n: int(n.get('x', 0)))
    columns = []
    for n in xs_sorted:
        x = int(n.get('x', 0))
        if columns and x - columns[-1]['x_center'] <= bandwidth_px:
            col = columns[-1]
            col['entries'].append(n)
            # update center as running mean for stability
            xs = [int(e.get('x', 0)) for e in col['entries']]
            col['x_center'] = sum(xs) // len(xs)
        else:
            columns.append({'x_center': x, 'entries': [n]})
    return columns


def _pick_best_depth_column(columns, panel_height_px, unit_hint):
    """Run detect_depth_axis on each column and pick the best fit.

    Scoring favors: many labels, high R², depth values in plausible range.
    Returns (best_axis, best_column_xcenter) or (None, None) if nothing plausible.
    """
    best = None
    best_score = -1.0
    best_x = None
    for col in columns:
        entries = col['entries']
        if len(entries) < 2:
            continue
        # Skip columns where all values are tiny (likely scale labels like 0.2/20)
        vals = [float(e.get('value', 0)) for e in entries]
        if max(vals) < 50:
            continue
        axis = scale_detection.detect_depth_axis(
            ocr_numbers=entries,
            panel_height_px=panel_height_px,
            unit_hint=unit_hint,
        )
        n_labels = len(axis.labels)
        if n_labels < 2 or axis.top_depth is None:
            continue
        # Score: R² weighted by number of labels (log-scaled) and depth plausibility
        r2 = max(0.0, axis.r_squared)
        depth_span = abs(axis.bottom_depth - axis.top_depth) if axis.bottom_depth else 0
        # Penalize tiny spans (could be scale labels) or absurdly large
        span_factor = 1.0 if 50 < depth_span < 20000 else 0.3
        score = r2 * math.log(n_labels + 1) * span_factor
        if score > best_score:
            best_score = score
            best = axis
            best_x = col['x_center']
    return best, best_x


@app.route('/api/detect-depth-axis', methods=['POST'])
def api_detect_depth_axis():
    """Detect the depth axis by auto-finding the depth column anywhere in the image.

    Request JSON:
        image:       data URL
        region:      optional { left_px, right_px, top_px, bottom_px } to constrain search
        unit_hint:   optional 'FT' or 'M'

    Response:
        success:   bool
        axis:      DetectedDepthAxis dict with extra x_center + image-space coords
    """
    data = request.json or {}
    img, err = _decode_image_data_url(data.get('image'))
    if err:
        return jsonify({'success': False, 'error': err}), 400

    region = data.get('region') or {}
    unit_hint = (data.get('unit_hint') or '').upper().strip() or None
    if unit_hint and unit_hint not in ('FT', 'M'):
        unit_hint = None

    H, W, _ = img.shape
    try:
        left = max(0, int(region.get('left_px', 0))) if region.get('left_px') is not None else 0
        right = min(W, int(region.get('right_px', W))) if region.get('right_px') is not None else W
        top = max(0, int(region.get('top_px', 0))) if region.get('top_px') is not None else 0
        bottom = min(H, int(region.get('bottom_px', H))) if region.get('bottom_px') is not None else H
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid region'}), 400
    if right <= left or bottom <= top:
        return jsonify({'success': False, 'error': 'Empty region'}), 400

    crop = img[top:bottom, left:right]
    crop_h, crop_w, _ = crop.shape
    if crop_h < 10 or crop_w < 5:
        return jsonify({'success': False, 'error': 'Region too small'}), 400

    # OCR the full (optionally constrained) panel, then auto-find the depth column
    numbers = _document_ocr_numbers(crop)
    if not numbers:
        return jsonify({
            'success': True,
            'axis': {'top_depth': None, 'bottom_depth': None, 'labels': [],
                     'confidence': 0.0, 'reasons': ['no numbers detected in region']},
        })

    columns = _cluster_numbers_into_columns(numbers, bandwidth_px=max(40, crop_w // 30))
    axis, col_x_center = _pick_best_depth_column(columns, panel_height_px=crop_h, unit_hint=unit_hint)

    if axis is None:
        # Return diagnostic info to help the user adjust
        col_summary = [
            {'x_center': c['x_center'], 'count': len(c['entries']),
             'sample_values': sorted({float(e['value']) for e in c['entries']})[:6]}
            for c in columns
        ]
        return jsonify({
            'success': True,
            'axis': {
                'top_depth': None, 'bottom_depth': None, 'labels': [],
                'confidence': 0.0,
                'reasons': [f'scanned {len(numbers)} numbers in {len(columns)} columns; none yielded a linear depth fit'],
                'candidate_columns': col_summary,
            },
        })

    axis_dict = axis.to_dict()
    axis_dict['x_center'] = int(left + (col_x_center or 0))
    # Map y coords back to full-image space for the frontend
    if axis_dict.get('top_px') is not None:
        axis_dict['top_px_image'] = int(top + axis_dict['top_px'])
    if axis_dict.get('bottom_px') is not None:
        axis_dict['bottom_px_image'] = int(top + axis_dict['bottom_px'])
    for lbl in axis_dict.get('labels', []):
        lbl['y_px_image'] = int(top + lbl['y_px'])

    return jsonify({'success': True, 'axis': axis_dict})


@app.route('/api/detect-all-scales', methods=['POST'])
def api_detect_all_scales():
    """Batch detect scales for many curves in one image with a single OCR pass per region.

    Request JSON:
        image:   data URL
        curves:  [ { id, left_px, right_px, top_px, bottom_px, name? }, ... ]
        use_llm: bool (default True) — enable LLM fallback for low-confidence tracks

    Response:
        { success: bool, results: [ { id, scale, ocr_header, ocr_axis_labels, llm_used, error? } ] }
    """
    data = request.json or {}
    img, err = _decode_image_data_url(data.get('image'))
    if err:
        return jsonify({'success': False, 'error': err}), 400

    curves = data.get('curves') or []
    if not isinstance(curves, list) or not curves:
        return jsonify({'success': False, 'error': 'Missing curves[]'}), 400

    use_llm = bool(data.get('use_llm', True))
    results = []
    for idx, c in enumerate(curves):
        region = {
            'left_px':   c.get('left_px', 0),
            'right_px':  c.get('right_px', 0),
            'top_px':    c.get('top_px', 0),
            'bottom_px': c.get('bottom_px', img.shape[0]),
        }
        res = _detect_scale_for_region(
            img,
            region=region,
            curve_name=(c.get('name') or '').strip(),
            xs_trace=c.get('xs_trace'),
            use_llm=use_llm,
        )
        entry = {'id': c.get('id', idx)}
        if not res.get('ok'):
            entry['error'] = res.get('error')
        else:
            entry['scale'] = res['scale']
            entry['ocr_header'] = res['ocr_header']
            entry['ocr_axis_labels'] = res['ocr_axis_labels']
            entry['llm_used'] = res['llm_used']
            entry['memory_used'] = res.get('memory_used', False)
        results.append(entry)

    return jsonify({'success': True, 'results': results})


@app.route('/api/record-correction', methods=['POST'])
def api_record_correction():
    """Record a user correction of a previously-detected scale.

    Request JSON:
        header_ocr:  list[str] — the OCR strings the detector saw in the header
        axis_ocr:    list[str] — axis band OCR strings
        ai_choice:   {mnemonic, scale_type, left, right, wrapped}
        user_choice: same shape — what the user chose/edited to
    """
    data = request.json or {}
    header_ocr = data.get('header_ocr') or []
    axis_ocr = data.get('axis_ocr') or []
    ai_choice = data.get('ai_choice') or {}
    user_choice = data.get('user_choice') or {}

    if not isinstance(header_ocr, list) or not isinstance(axis_ocr, list):
        return jsonify({'success': False, 'error': 'header_ocr/axis_ocr must be arrays'}), 400
    if not isinstance(user_choice, dict) or not user_choice:
        return jsonify({'success': False, 'error': 'user_choice required'}), 400

    try:
        uid = session.get('user_id')
    except Exception:
        uid = None

    try:
        correction_id = corrections_store.record_correction(
            corrections_store.CorrectionEntry(
                header_ocr=header_ocr,
                axis_ocr=axis_ocr,
                ai_choice=ai_choice,
                user_choice=user_choice,
                user_id=uid,
            )
        )
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500

    return jsonify({
        'success': True,
        'id': correction_id,
        'total_corrections': corrections_store.count_corrections(),
    })


@app.route('/api/corrections-stats', methods=['GET'])
def api_corrections_stats():
    """Return count of stored corrections (useful for a 'trained on N examples' UI badge)."""
    try:
        uid = session.get('user_id')
    except Exception:
        uid = None
    try:
        return jsonify({
            'success': True,
            'user_total': corrections_store.count_corrections(user_id=uid) if uid else 0,
            'global_total': corrections_store.count_corrections(),
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/robots.txt')
def static_from_root_robots():
    return send_from_directory('static', 'robots.txt')

@app.route('/sitemap.xml')
def static_from_root_sitemap():
    return send_from_directory('static', 'sitemap.xml')

@app.route('/BingSiteAuth.xml')
def static_from_root_bing():
    return send_from_directory('templates', 'BingSiteAuth.xml')

if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("Starting TIFF-to-LAS Web App")
    if VISION_API_AVAILABLE:
        print("Google Vision API: Available")
    elif LOCAL_OCR_AVAILABLE:
        print("Google Vision API: Not configured (using EasyOCR fallback)")
    else:
        print("Google Vision API: Not configured")
    print("Open: http://localhost:5000")
    
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
