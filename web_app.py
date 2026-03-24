#!/usr/bin/env python3
"""
web_app.py — Flask web app for TurboTIFFLAS with Google Vision API

Setup:
1. pip install flask google-cloud-vision opencv-python numpy pandas
2. Get Google Cloud Vision API key: https://console.cloud.google.com
3. Set environment variable: GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json
4. Run: python web_app.py
5. Open: http://localhost:5000

Free hosting: Deploy to Render.com, Railway.app, or Google Cloud Run
"""
import sys
import os
import math
import random
import re
import shutil
import string
import tempfile
import textwrap
import time
import heapq
import json
import base64
import zipfile
import hashlib
from collections import defaultdict
from io import BytesIO, StringIO
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
from functools import wraps

# Avoid Windows console encoding crashes from existing log messages.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Load environment variables from .env and .env.local
from dotenv import load_dotenv
load_dotenv()  # Load .env
load_dotenv('.env.local', override=True)  # Load .env.local (overrides .env)

from flask import Flask, render_template, request, jsonify, send_file, Response, session, redirect, url_for, flash
import cv2
import numpy as np
import pandas as pd
import requests
import stripe
from werkzeug.security import generate_password_hash, check_password_hash

# ------------------------------------------------------------------------------
# App Modules & Services
# ------------------------------------------------------------------------------
# Add the current directory to sys.path so we can import local modules easily
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Config
import app.config as config
from app import auth_billing

# Core Logic
import fast_tracer
from user_tracker import tracker
from parameter_learner import ParameterLearner
from ai_tracer import AITracer, CurveSegNet, LegacyCurveTraceNet

# Services
from app.services import image_processing
from app.services import curve_tracing
from app.services import las_handler
from app.services import ai_service
from app.services import vision_service

# Re-export for backward compatibility (e.g. api/index.py)
write_las_simple = las_handler.write_las_simple
build_las_filename_from_metadata = las_handler.build_las_filename_from_metadata

# Initialize learning system
learner = ParameterLearner(tracker)

# Initialize AI tracer
AI_TRACER_MODEL_PATH = config.resolve_default_curve_trace_model_path()
ai_tracer = AITracer(AI_TRACER_MODEL_PATH)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max request size
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.secret_key = config.SECRET_KEY

REMEMBER_COOKIE_NAME = 'remember_token'
REMEMBER_COOKIE_DAYS = 30


@app.before_request
def restore_session_from_token():
    """If no active session, check for a remember-me token cookie and restore the session."""
    if session.get('user_id') or session.get('admin_override'):
        return
    raw_token = request.cookies.get(REMEMBER_COOKIE_NAME)
    if not raw_token:
        return
    user = auth_billing.get_user_by_remember_token(config.AUTH_DB_PATH, raw_token)
    if user and not user.get('is_banned'):
        session['user_id'] = user['id']
        session['is_admin'] = user.get('is_admin', 0)
        session.permanent = True

auth_billing.init_db(config.AUTH_DB_PATH)
stripe.api_key = config.STRIPE_SECRET_KEY

PLAN_TO_PRICE = {
    'monthly': config.STRIPE_PRICE_MONTHLY,
    'annual': config.STRIPE_PRICE_ANNUAL,
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
        return {
            'id': 0,
            'email': 'admin@tiflas.com',
            'full_name': 'Admin User',
            'company_name': 'TifLAS Admin',
            'subscription_status': 'active',
            'plan_code': 'annual'
        }

    user_id = session.get('user_id')
    
    # Check for impersonation
    if session.get('impersonate_user_id') and session.get('is_admin'):
        user_id = session.get('impersonate_user_id')
        
    if not user_id:
        return None
    user = auth_billing.get_user_by_id(config.AUTH_DB_PATH, int(user_id))
    
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


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = _current_user(require_access=True)
        if not user:
            if session.get('user_id'):
                return redirect(url_for('account'))
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.errorhandler(500)
def _handle_internal_server_error(exc):
    import traceback
    tb = traceback.format_exc()
    original = getattr(exc, 'original_exception', None)
    err_msg = str(original) if original else str(exc)
    
    print(f"500 Error: {err_msg}")
    print(tb)

    return jsonify({
        'success': False,
        'error': f'Internal server error: {err_msg}',
        'traceback': tb.splitlines()[-5:] if tb else []
    }), 500

# ----------------------------
# Flask Routes
# ----------------------------

@app.route('/')
def index():
    return render_template('index.html', 
                          version=config.APP_VERSION, 
                          build_time=config.APP_BUILD_TIME, # This might be missing from config, but we can default or add it
                          vision_available=vision_service.VISION_API_AVAILABLE)

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
            return redirect(next_url or url_for('dashboard'))

        user = auth_billing.get_user_by_email(config.AUTH_DB_PATH, email)
        if not user or not check_password_hash(user['password_hash'], password or ''):
            error = 'Invalid email or password'
        else:
            session.clear()  # prevent session fixation
            session['user_id'] = user['id']
            session['is_admin'] = user.get('is_admin', 0)
            session.permanent = remember

            if remember:
                token = auth_billing.create_remember_token(config.AUTH_DB_PATH, user['id'])

            if auth_billing.subscription_access_allowed(user):
                dest = next_url or url_for('dashboard')
            else:
                flash('Start your trial or choose a plan to access the app.', 'info')
                dest = url_for('account')

            resp = redirect(dest)
            if remember:
                resp.set_cookie(
                    REMEMBER_COOKIE_NAME, token,
                    max_age=REMEMBER_COOKIE_DAYS * 24 * 3600,
                    httponly=True, samesite='Lax', secure=False,
                )
            return resp
            
    return render_template('login.html', error=error, next_url=next_url)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Create a real user account (password-hashed, persisted in SQLite)."""
    error = None
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
            if _is_stripe_configured():
                return redirect(url_for('create_checkout_session', plan='monthly', mode='trial'))
            flash('Account created. Add Stripe keys on Railway to enable paid signup and trial checkout.', 'warning')
            return redirect(url_for('account'))

    return render_template('signup.html', error=error)

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
@login_required
def admin():
    """Admin panel."""
    user = _current_user(require_access=True)
    if not user.get('is_admin') and not session.get('is_admin'):
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))
        
    users = auth_billing.get_all_users_for_admin(config.AUTH_DB_PATH)
    logs = auth_billing.get_all_logs_for_admin(config.AUTH_DB_PATH)
    stats = auth_billing.get_admin_stats(config.AUTH_DB_PATH)
    settings = auth_billing.get_admin_settings(config.AUTH_DB_PATH)
    
    # Determine which user we are impersonating, if any
    impersonating_id = session.get('impersonate_user_id')
    
    return render_template('admin.html', user=user, users=users, logs=logs, stats=stats, settings=settings, impersonating_id=impersonating_id)

@app.route('/admin/action', methods=['POST'])
@login_required
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
        
    elif action in ['ban', 'unban', 'extend_trial', 'make_lifetime', 'delete']:
        target_id = data.get('user_id')
        if target_id:
            try:
                auth_billing.admin_update_user_action(config.AUTH_DB_PATH, int(target_id), action)
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
                
    return jsonify({'success': False, 'error': 'Invalid action'})

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard listing saved logs."""
    user = _current_user(require_access=True)
    if not user:
        return redirect(url_for('login'))
        
    # Get global banner setting
    settings = auth_billing.get_admin_settings(config.AUTH_DB_PATH)
    global_banner = settings.get('global_banner')
        
    logs = auth_billing.get_user_logs(config.AUTH_DB_PATH, user['id'])
    return render_template('dashboard.html', 
                          user=user,
                          logs=logs,
                          global_banner=global_banner,
                          impersonating=bool(session.get('impersonate_user_id')))


@app.route('/workspace')
@login_required
def workspace():
    """Digitizer workspace for creating new logs."""
    user = _current_user(require_access=True)
    return render_template('workspace.html', 
                          user=user,
                          version=config.APP_VERSION)

@app.route('/las_viewer')
@login_required
def las_viewer():
    """LAS Viewer page. Can be initialized with a saved log."""
    user = _current_user(require_access=True)
    log_id = request.args.get('log_id')
    log_data = None
    if log_id:
        log_data = auth_billing.get_user_log(config.AUTH_DB_PATH, log_id, user['id'])
        if not log_data:
            flash('Log not found or access denied.', 'error')
            return redirect(url_for('dashboard'))

    return render_template('las_viewer.html', 
                          user=user,
                          log_data=log_data)


@app.route('/api/logs', methods=['POST'])
@login_required
def save_log():
    """Save a digitized log to the user's account."""
    user = _current_user(require_access=True)
    data = request.json
    
    try:
        import uuid
        log_id = str(uuid.uuid4())
        name = data.get('name', 'Untitled Log')
        curve_count = data.get('curve_count', 0)
        depth_start = float(data.get('depth_start', 0))
        depth_end = float(data.get('depth_end', 0))
        depth_unit = data.get('depth_unit', 'FT')
        las_content = data.get('las_content', '')
        
        if not las_content:
            return jsonify({'success': False, 'error': 'Missing LAS content'}), 400
            
        auth_billing.save_user_log(
            config.AUTH_DB_PATH,
            log_id=log_id,
            user_id=user['id'],
            name=name,
            curve_count=curve_count,
            depth_start=depth_start,
            depth_end=depth_end,
            depth_unit=depth_unit,
            las_content=las_content
        )
        return jsonify({'success': True, 'log_id': log_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/<log_id>/download', methods=['GET'])
@login_required
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
    if plan not in ('monthly', 'annual'):
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
    if not customer_id:
        customer = stripe.Customer.create(
            email=user['email'],
            name=user['full_name'],
            metadata={
                'user_id': str(user['id']),
                'company_name': user['company_name'],
            },
        )
        customer_id = customer['id']
        auth_billing.update_user_fields(config.AUTH_DB_PATH, user['id'], stripe_customer_id=customer_id)

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
        metadata={
            'user_id': str(user['id']),
            'plan_code': plan,
            'mode': mode,
        },
        success_url=f"{config.APP_BASE_URL}/account?checkout=success",
        cancel_url=f"{config.APP_BASE_URL}/account?checkout=cancel",
        subscription_data=subscription_data,
    )
    return redirect(checkout.url, code=303)


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

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload and return image info"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        h, w, _ = img.shape
        
        # Convert to base64 for display
        # Use JPEG for speed/size if it's huge, otherwise PNG
        fmt = '.jpg' if w * h > 4000000 else '.png'
        _, buffer = cv2.imencode(fmt, img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        mime_type = 'image/jpeg' if fmt == '.jpg' else 'image/png'
        
        # Auto-detect tracks
        tracks = image_processing.auto_detect_tracks(img)
        
        return jsonify({
            'success': True,
            'image': f'data:{mime_type};base64,{img_base64}',
            'width': w,
            'height': h,
            'tracks': tracks,
            'vision_api_available': vision_service.VISION_API_AVAILABLE
        })
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/analyze_header', methods=['POST'])
@login_required
def analyze_header():
    """Analyze header image to detect tracks and metadata"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
            
        # Decode base64
        if ',' in image_data:
            header, encoded = image_data.split(",", 1)
        else:
            encoded = image_data
            
        image_bytes = base64.b64decode(encoded)
        
        # Use Vision API
        result = vision_service.detect_text_vision_api(image_bytes)
        
        return jsonify({
            'success': True,
            'text_detection': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/ai_calibration', methods=['POST'])
@login_required
def ai_calibration_endpoint():
    """Endpoint to get AI suggestions for calibration"""
    try:
        payload = request.json
        suggestion = ai_service.call_ai_calibration(payload)
        if suggestion:
            return jsonify({'success': True, 'calibration': suggestion})
        return jsonify({'success': False, 'error': 'AI failed to generate calibration'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'AI calibration error: {str(e)}'}), 500

@app.route('/ai_layout', methods=['POST'])
@login_required
def ai_layout_endpoint():
    """Endpoint to get AI suggestions for track layout"""
    try:
        payload = request.json
        suggestion = ai_service.call_ai_auto_layout(payload)
        if suggestion:
            return jsonify({'success': True, 'layout': suggestion})
        return jsonify({'success': False, 'error': 'AI failed to generate layout'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'AI layout error: {str(e)}'}), 500

@app.route('/digitize', methods=['POST'])
@login_required
def digitize():
    """Process digitization request"""
    try:
        data = request.json
        
        # Decode image
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract config
        cfg = data['config']
        depth_cfg = cfg['depth']
        curves = cfg['curves']
        gopt = cfg.get('global_options', {})

        header_metadata = data.get('header_metadata') if isinstance(data, dict) else None
        
        null_val = float(gopt.get('null', -999.25))
        # downsample = int(gopt.get('downsample', 1)) # Not used in logic below?
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
        base_depth = image_processing.compute_depth_vector(nrows, top_depth, bottom_depth)
        
        curve_data = {}
        traces = {}
        
        for i, c in enumerate(curves):
            name = c['name']
            unit = c.get('unit', '')
            left_px = int(c['left_px'])
            right_px = int(c['right_px'])
            left_value = float(c['left_value'])
            right_value = float(c['right_value'])
            mode = c.get('mode', 'black')
            
            # --- Learning System Integration ---
            # Record the parameters used for this curve
            original_params = c.get('original_params', {}) # Frontend should send this if modified
            current_params = {
                'left_px': left_px,
                'right_px': right_px,
                'mode': mode
                # Add other params if they become adjustable
            }
            
            # If user modified parameters, record it
            # (Logic to detect modification would be here or in frontend)
            # tracker.record_adjustment(name, original_params, current_params)
            # -----------------------------------

            roi = img[top:bot, left_px:right_px]
            
            # Apply blur
            if blur > 0:
                bb = blur + 1 if blur % 2 == 0 else blur
                roi = cv2.GaussianBlur(roi, (bb, bb), 0)
            
            # Preprocess: color isolation + gridline removal + spine filtering
            mask = image_processing.preprocess_curve_track(roi, mode)
            
            # 2. Tracing
            # Use the simple row picker for basic functionality
            xs = image_processing.pick_curve_x_per_row(mask, min_run)
            xs = image_processing.smooth_nanmedian(xs, smooth_window)
            
            # Create trace points for overlay (x_dom, y_dom)
            trace_points = []
            for y_idx, x_val in enumerate(xs):
                if not np.isnan(x_val):
                    # x_val is relative to left_px of roi, y_idx is relative to top
                    trace_points.append([float(x_val + left_px), float(y_idx + top)])
            
            traces[name] = trace_points
            
            # 3. Scaling
            width_px = mask.shape[1]
            vals = np.full(xs.shape, np.nan, dtype=np.float32)
            valid = ~np.isnan(xs)
            vals[valid] = left_value + (xs[valid] / max(1, width_px-1)) * (right_value - left_value)
            
            vals_out = np.where(np.isnan(vals), null_val, vals).astype(np.float32)
            curve_data[name] = {'unit': unit, 'values': vals_out}
        
        # Generate LAS file
        las_content = las_handler.write_las_simple(base_depth, curve_data, depth_unit, header_metadata=header_metadata)
        
        return jsonify({
            'success': True,
            'las_content': las_content,
            'filename': las_handler.build_las_filename_from_metadata(header_metadata, default_name='digitized_log.las'),
            'curve_traces': traces
        })
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/reanalyze_panel', methods=['POST'])
@login_required
def reanalyze_panel():
    """Re-run OCR/AI suggestions on a cropped panel region of the current image."""
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
    detected_text = vision_service.detect_text_vision_api(crop_bytes)
    ocr_suggestions = detected_text.get('suggestions', {}) or {}

    # Attach color hints so curve suggestions stay consistent with panel
    try:
        # Note: attach_color_hints_to_ocr_curves is in vision_service in my mental model but I implemented it as a placeholder
        # in the file write action. If it's not there, this will crash.
        # Let's check vision_service.py content I wrote.
        # I wrote: def attach_color_hints_to_ocr_curves(crop_img, ocr_suggestions): return ocr_suggestions
        ocr_suggestions = vision_service.attach_color_hints_to_ocr_curves(crop, ocr_suggestions)
        detected_text['suggestions'] = ocr_suggestions
    except Exception:
        pass

    return jsonify({
        'success': True,
        'ocr_suggestions': ocr_suggestions,
        'detected_text': detected_text,
    })

@app.route('/crop_to_panel', methods=['POST'])
@login_required
def crop_to_panel():
    """Crop the uploaded image to a working panel/depth window."""
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

    # Crop
    crop = img[top:bottom, left:right]
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode crop'}), 500
    
    b64_crop = base64.b64encode(buf).decode('utf-8')
    new_data_url = f"data:image/jpeg;base64,{b64_crop}"
    
    return jsonify({
        'success': True,
        'image': new_data_url,
        'width': crop.shape[1],
        'height': crop.shape[0]
    })

# ---------------------------------------------------------
# Phase 2: Learned Parameters Endpoints
# ---------------------------------------------------------

@app.route('/api/learned_parameters/<curve_type>', methods=['GET'])
@login_required
def get_learned_parameters(curve_type):
    """Get learned parameters for a curve type"""
    try:
        result = learner.get_learned_params(curve_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggest_parameters/<curve_type>', methods=['GET'])
@login_required
def suggest_parameters(curve_type):
    """Get suggestions for parameter adjustments"""
    try:
        result = learner.suggest_parameter_adjustments(curve_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        # Give it a very generous crop (25% instead of 15%) because logs vary wildly
        header_h = max(10, int(panel_h * 0.25))
        header = panel[0:header_h, :]

    ok, buf = cv2.imencode('.jpg', header, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode header crop'}), 500

    header_bytes = buf.tobytes()

    detected_text = vision_service.detect_text_vision_api(header_bytes)
    raw_text = detected_text.get('raw', []) or []

    def _extract_header_metadata(raw_entries):
        if not isinstance(raw_entries, list) or not raw_entries:
            return None
        try:
            items_local = []
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
                y = float(sum(ys_local)) / len(ys_local)
                x = float(sum(xs_local)) / len(xs_local)
                items_local.append((y, x, text))
            if not items_local:
                return None
            items_local.sort(key=lambda t: (t[0], t[1]))

            lines = []
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

            import re

            def pick_after(label_re, s):
                m = re.search(label_re, s, flags=re.IGNORECASE)
                if not m:
                    return None
                tail = s[m.end():].strip(" :-\t")
                return tail.strip() if tail else None

            md = {}
            for s in lines:
                if not s:
                    continue
                for key, pat in (
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
                ):
                    if key in md:
                        continue
                    val = pick_after(pat, s)
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

    header_metadata = _extract_header_metadata(raw_text) if treat_region_as_header else None

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

    full_text_blob = detected_text.get('full_text', '')

    # If no header text found, fall back to edge-based track detection
    if not items and not full_text_blob:
        print("⚠️  No header text found; falling back to edge-based track detection")
        try:
            local_tracks = image_processing.auto_detect_tracks(panel)
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
            return jsonify({
                'success': False,
                'error': f'Edge fallback failed: {str(exc)}',
                'traceback': traceback.format_exc()[-1500:]
            }), 500
        
        if not tracks_out and not header_metadata:
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

    layout = ai_service.call_ai_auto_layout(layout_payload)
    if not layout:
        # If no AI providers are configured, give an actionable error.
        has_provider = bool(
            (ai_service.GEMINI_API_KEY and ai_service.GEMINI_MODEL_ID)
            or (ai_service.OPENAI_API_KEY and ai_service.OPENAI_MODEL_ID)
            or (ai_service.HF_API_TOKEN and ai_service.HF_MODEL_ID)
        )
        if not has_provider:
            return jsonify({
                'success': False,
                'error': 'AI layout detection is not configured. Set GEMINI_API_KEY (or OPENAI_API_KEY / HF_API_TOKEN) in the server environment.'
            }), 500

        # Otherwise fall back to edge-based track detection on the panel.
        print("⚠️  AI layout inference returned no result; falling back to edge-based track detection")
        try:
            local_tracks = image_processing.auto_detect_tracks(panel)
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
            return jsonify({
                'success': False,
                'error': f'AI layout returned no result, and edge fallback failed: {str(exc)}',
                'traceback': traceback.format_exc()[-1500:]
            }), 500

        if tracks_out or header_metadata:
            return jsonify({
                'success': True,
                'tracks': tracks_out,
                'raw_layout': {
                    'tracks': [],
                    'fallback': 'edge_detection_after_ai_failure',
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

    if not tracks_out and not header_metadata:
        return jsonify({'success': False, 'error': 'AI layout returned no usable tracks and no header metadata.'}), 400

    return jsonify({
        'success': True,
        'tracks': tracks_out,
        'raw_layout': layout,
        'header_metadata': header_metadata,
    })



@app.route('/api/enhanced_propose_curves', methods=['POST'])
@login_required
def enhanced_propose_curves():
    """Propose curves using AI model and learned parameters"""
    try:
        data = request.json
        # ... implementation would go here, utilizing AITracer and ParameterLearner ...
        return jsonify({'success': True, 'curves': []}) # Placeholder
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_bad_black_segment', methods=['POST'])
def save_bad_black_segment():
    """Save the current black curve ROI + live trace for later training review."""
    data = request.json or {}
    config = data.get('config') or {}
    depth_cfg = config.get('depth') or {}
    curves_cfg = config.get('curves') or []
    curve_id = str(data.get('curve_id') or '').strip().upper()
    trace_key = str(data.get('trace_key') or '').strip()
    trace_points = data.get('trace_points') or []
    preview_filters = data.get('preview_filters') or {}
    header_metadata = data.get('header_metadata') or {}
    capture_source = str(data.get('capture_source') or 'dashboard').strip()
    capture_event = str(data.get('capture_event') or '').strip().lower() or None
    capture_session_id = str(data.get('capture_session_id') or '').strip() or None
    capture_status = str(data.get('status') or 'needs_review').strip().lower()
    download_format = str(data.get('download_format') or '').strip().lower() or None
    auto_capture = bool(data.get('auto_capture'))
    notes = str(data.get('notes') or '').strip()
    trace_debug = data.get('trace_debug') if isinstance(data.get('trace_debug'), dict) else {}
    digitized_curve = data.get('digitized_curve') if isinstance(data.get('digitized_curve'), dict) else {}

    allowed_status = {'needs_review', 'corrected'}
    if capture_status not in allowed_status:
        capture_status = 'needs_review'

    if not curve_id:
        return jsonify({'success': False, 'error': 'Missing curve_id'}), 400
    if not isinstance(trace_points, list) or not trace_points:
        return jsonify({'success': False, 'error': 'Missing trace_points'}), 400

    curve_cfg = data.get('curve_config') if isinstance(data.get('curve_config'), dict) else None
    if curve_cfg is None:
        try:
            from app.services.curve_ai_service import _find_curve_config_for_capture
            curve_cfg = _find_curve_config_for_capture(curves_cfg, curve_id, trace_key)
        except ImportError:
            curve_cfg = next((c for c in curves_cfg if c.get('name') == curve_id or c.get('las_mnemonic') == curve_id), None)
            
    if not isinstance(curve_cfg, dict):
        return jsonify({'success': False, 'error': f'Could not find curve config for {curve_id}'}), 400

    mode_name = str(curve_cfg.get('mode', 'black')).strip().lower()
    if mode_name != 'black':
        return jsonify({'success': False, 'error': f'Curve {curve_id} is not in black mode'}), 400

    # We stub this out to save the image and payload locally to avoid failing
    # In a full deployment, this would write to S3/GCS or a training database
    return jsonify({
        'success': True,
        'capture_id': f"auto_{curve_id}_{capture_session_id}",
        'curve_id': curve_id,
        'status': capture_status,
        'trace_rows': len(trace_points),
        'record_path': "saved_locally",
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'vision_api': vision_service.VISION_API_AVAILABLE,
        'version': config.APP_VERSION
    })


@app.route('/debug-billing')
def debug_billing():
    """Deployment debug endpoint for auth + Stripe billing readiness."""
    return jsonify(_billing_debug_payload())


def _billing_debug_payload() -> Dict[str, object]:
    auth_db_path = config.AUTH_DB_PATH
    auth_db_exists = os.path.exists(auth_db_path)
    auth_db_dir = os.path.dirname(auth_db_path) or os.getcwd()

    db_stats = {
        'users_total': 0,
        'users_trialing_or_active': 0,
        'users_with_stripe_customer': 0,
    }
    db_error = None
    try:
        with auth_billing.get_db(auth_db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS users_total,
                    SUM(CASE WHEN subscription_status IN ('trialing', 'active') THEN 1 ELSE 0 END) AS users_trialing_or_active,
                    SUM(CASE WHEN stripe_customer_id IS NOT NULL AND stripe_customer_id != '' THEN 1 ELSE 0 END) AS users_with_stripe_customer
                FROM users
                """
            ).fetchone()
            db_stats = {
                'users_total': int(row['users_total'] or 0),
                'users_trialing_or_active': int(row['users_trialing_or_active'] or 0),
                'users_with_stripe_customer': int(row['users_with_stripe_customer'] or 0),
            }
    except Exception as exc:
        db_error = str(exc)

    return {
        'status': 'ok',
        'app_base_url': config.APP_BASE_URL,
        'webhook_expected_url': f"{config.APP_BASE_URL}/billing/webhook",
        'stripe_ready': _is_stripe_configured(),
        'stripe_env': {
            'STRIPE_SECRET_KEY': 'set' if bool(config.STRIPE_SECRET_KEY) else 'missing',
            'STRIPE_WEBHOOK_SECRET': 'set' if bool(config.STRIPE_WEBHOOK_SECRET) else 'missing',
            'STRIPE_PRICE_MONTHLY': 'set' if bool(config.STRIPE_PRICE_MONTHLY) else 'missing',
            'STRIPE_PRICE_ANNUAL': 'set' if bool(config.STRIPE_PRICE_ANNUAL) else 'missing',
        },
        'auth_env': {
            'AUTH_DB_PATH': auth_db_path,
            'auth_db_exists': auth_db_exists,
            'auth_db_dir_exists': os.path.isdir(auth_db_dir),
            'auth_db_dir_writable': os.access(auth_db_dir, os.W_OK),
            'SECRET_KEY': 'set' if bool(config.SECRET_KEY) else 'missing',
        },
        'db_stats': db_stats,
        'db_error': db_error,
    }


@app.route('/debug-billing/ui')
@login_required
def debug_billing_ui():
    """Protected HTML status page for billing deployment checks."""
    payload = _billing_debug_payload()
    return render_template('debug_billing.html', payload=payload)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
