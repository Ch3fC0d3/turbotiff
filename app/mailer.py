"""
Simple SMTP email helper using Namecheap Private Email (or any SMTP provider).
Configure via environment variables:
  MAIL_SERVER, MAIL_PORT, MAIL_USE_TLS, MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM
"""
import smtplib
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from app import config


def _is_configured() -> bool:
    return bool(config.MAIL_USERNAME and config.MAIL_PASSWORD)


def send_email(to: str, subject: str, html_body: str, text_body: Optional[str] = None) -> bool:
    """Send an email. Returns True on success, False on failure."""
    if not _is_configured():
        print(f"[MAIL] Not configured - skipping email to {to}: {subject}")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config.MAIL_FROM
        msg['To'] = to

        if text_body:
            msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        if config.MAIL_USE_TLS:
            server = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
            server.ehlo()
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(config.MAIL_SERVER, config.MAIL_PORT)

        server.login(config.MAIL_USERNAME, config.MAIL_PASSWORD)
        server.sendmail(config.MAIL_FROM, [to], msg.as_string())
        server.quit()
        print(f"[MAIL] Sent '{subject}' to {to}")
        return True

    except Exception as exc:
        print(f"[MAIL] Failed to send '{subject}' to {to}: {exc}")
        traceback.print_exc()
        return False


# ── Pre-built email templates ──────────────────────────────────────────────

def send_welcome(to: str, full_name: str) -> bool:
    subject = "Welcome to TifLAS – Your trial is active!"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <div style="background:#0f172a;padding:24px;text-align:center;">
        <img src="https://tiflas.com/static/images/logo.svg" alt="TifLAS" style="height:60px;">
      </div>
      <div style="padding:32px;background:#f8fafc;">
        <h2 style="color:#0f172a;">Welcome, {full_name}!</h2>
        <p style="color:#374151;">Your 7-day free trial is now active. Start digitizing well log images right away.</p>
        <a href="https://tiflas.com/workspace"
           style="display:inline-block;margin-top:16px;padding:12px 28px;background:#667eea;color:#fff;text-decoration:none;border-radius:6px;font-weight:bold;">
          Open Workspace
        </a>
        <p style="margin-top:24px;color:#6b7280;font-size:0.9em;">
          Questions? Reply to this email and we'll help you out.
        </p>
      </div>
      <div style="padding:16px;text-align:center;color:#9ca3af;font-size:0.8em;">
        © 2024 TifLAS. All rights reserved.
      </div>
    </div>
    """
    text = f"Welcome, {full_name}! Your TifLAS 7-day free trial is now active. Visit https://tiflas.com/workspace to get started."
    return send_email(to, subject, html, text)


def send_trial_expiring(to: str, full_name: str, days_left: int) -> bool:
    subject = f"Your TifLAS trial expires in {days_left} day{'s' if days_left != 1 else ''}"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <div style="background:#0f172a;padding:24px;text-align:center;">
        <img src="https://tiflas.com/static/images/logo.svg" alt="TifLAS" style="height:60px;">
      </div>
      <div style="padding:32px;background:#f8fafc;">
        <h2 style="color:#0f172a;">Hi {full_name},</h2>
        <p style="color:#374151;">Your free trial expires in <strong>{days_left} day{'s' if days_left != 1 else ''}</strong>. Upgrade now to keep access to all your saved logs.</p>
        <a href="https://tiflas.com/account"
           style="display:inline-block;margin-top:16px;padding:12px 28px;background:#667eea;color:#fff;text-decoration:none;border-radius:6px;font-weight:bold;">
          Upgrade My Account
        </a>
      </div>
      <div style="padding:16px;text-align:center;color:#9ca3af;font-size:0.8em;">
        © 2024 TifLAS. All rights reserved.
      </div>
    </div>
    """
    text = f"Hi {full_name}, your TifLAS trial expires in {days_left} days. Upgrade at https://tiflas.com/account"
    return send_email(to, subject, html, text)


def send_password_reset(to: str, reset_url: str) -> bool:
    subject = "Reset your TifLAS password"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <div style="background:#0f172a;padding:24px;text-align:center;">
        <img src="https://tiflas.com/static/images/logo.svg" alt="TifLAS" style="height:60px;">
      </div>
      <div style="padding:32px;background:#f8fafc;">
        <h2 style="color:#0f172a;">Password Reset Request</h2>
        <p style="color:#374151;">We received a request to reset your password. Click the button below to set a new one. This link expires in 1 hour.</p>
        <a href="{reset_url}"
           style="display:inline-block;margin-top:16px;padding:12px 28px;background:#667eea;color:#fff;text-decoration:none;border-radius:6px;font-weight:bold;">
          Reset Password
        </a>
        <p style="margin-top:24px;color:#6b7280;font-size:0.9em;">
          If you didn't request this, you can safely ignore this email.
        </p>
      </div>
      <div style="padding:16px;text-align:center;color:#9ca3af;font-size:0.8em;">
        © 2024 TifLAS. All rights reserved.
      </div>
    </div>
    """
    text = f"Reset your TifLAS password: {reset_url} (expires in 1 hour)"
    return send_email(to, subject, html, text)


def send_managed_job_admin(admin_to: str, full_name: str, company: str, email: str, job_id: str, well_name: str) -> bool:
    subject = f"New TifLAS Managed Job: {company} - {well_name}"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <div style="background:#0f172a;padding:24px;text-align:center;">
        <img src="https://tiflas.com/static/images/logo.svg" alt="TifLAS" style="height:60px;">
      </div>
      <div style="padding:32px;background:#f8fafc;">
        <h2 style="color:#0f172a;">New Full-Service Job Submitted</h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr><td style="padding:8px;color:#6b7280;">Name</td><td style="padding:8px;font-weight:bold;">{full_name}</td></tr>
          <tr style="background:#f1f5f9;"><td style="padding:8px;color:#6b7280;">Email</td><td style="padding:8px;">{email}</td></tr>
          <tr><td style="padding:8px;color:#6b7280;">Company</td><td style="padding:8px;">{company}</td></tr>
          <tr style="background:#f1f5f9;"><td style="padding:8px;color:#6b7280;">Well Name</td><td style="padding:8px;">{well_name}</td></tr>
          <tr><td style="padding:8px;color:#6b7280;">Job ID</td><td style="padding:8px;">{job_id}</td></tr>
        </table>
        <a href="https://tiflas.com/admin"
           style="display:inline-block;margin-top:16px;padding:12px 28px;background:#0f172a;color:#fff;text-decoration:none;border-radius:6px;font-weight:bold;">
          View in Admin Panel
        </a>
      </div>
    </div>
    """
    text = f"New Full-Service Job Submitted by {full_name} ({company}, {email}) for Well: {well_name}. Job ID: {job_id}."
    return send_email(admin_to, subject, html, text)
def send_log_saved(to: str, full_name: str, log_name: str, curve_count: int, depth_start: float, depth_end: float, depth_unit: str, log_id: str) -> bool:
    subject = f"Your TifLAS log \"{log_name}\" has been saved"
    depth_range = f"{depth_start:,.1f} – {depth_end:,.1f} {depth_unit}"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <div style="background:#0f172a;padding:24px;text-align:center;">
        <img src="https://tiflas.com/static/images/logo.svg" alt="TifLAS" style="height:60px;">
      </div>
      <div style="padding:32px;background:#f8fafc;">
        <h2 style="color:#0f172a;margin-top:0;">Log Saved Successfully ✓</h2>
        <p style="color:#374151;">Hi {full_name}, your digitized log has been saved to your TifLAS account.</p>
        <table style="width:100%;border-collapse:collapse;margin:16px 0;">
          <tr><td style="padding:8px;color:#6b7280;border-bottom:1px solid #e5e7eb;">Log Name</td><td style="padding:8px;font-weight:bold;border-bottom:1px solid #e5e7eb;">{log_name}</td></tr>
          <tr style="background:#f1f5f9;"><td style="padding:8px;color:#6b7280;border-bottom:1px solid #e5e7eb;">Curves</td><td style="padding:8px;border-bottom:1px solid #e5e7eb;">{curve_count}</td></tr>
          <tr><td style="padding:8px;color:#6b7280;">Depth Range</td><td style="padding:8px;">{depth_range}</td></tr>
        </table>
        <a href="https://tiflas.com/dashboard"
           style="display:inline-block;margin-top:8px;padding:12px 28px;background:#0284c7;color:#fff;text-decoration:none;border-radius:6px;font-weight:bold;">
          View My Logs
        </a>
      </div>
      <div style="padding:16px 32px;background:#e2e8f0;font-size:12px;color:#6b7280;">
        You received this email because you saved a log on TifLAS. <a href="https://tiflas.com" style="color:#0284c7;">tiflas.com</a>
      </div>
    </div>
    """
    text = f"Hi {full_name}, your log \"{log_name}\" ({curve_count} curves, {depth_range}) has been saved. View it at https://tiflas.com/dashboard"
    return send_email(to, subject, html, text)


def send_new_signup_admin(admin_to: str, new_user_email: str, full_name: str, company: str) -> bool:
    subject = f"New TifLAS signup: {full_name} ({company})"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <div style="background:#0f172a;padding:24px;text-align:center;">
        <img src="https://tiflas.com/static/images/logo.svg" alt="TifLAS" style="height:60px;">
      </div>
      <div style="padding:32px;background:#f8fafc;">
        <h2 style="color:#0f172a;">New Account Created</h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr><td style="padding:8px;color:#6b7280;">Name</td><td style="padding:8px;font-weight:bold;">{full_name}</td></tr>
          <tr style="background:#f1f5f9;"><td style="padding:8px;color:#6b7280;">Email</td><td style="padding:8px;">{new_user_email}</td></tr>
          <tr><td style="padding:8px;color:#6b7280;">Company</td><td style="padding:8px;">{company}</td></tr>
        </table>
        <a href="https://tiflas.com/admin"
           style="display:inline-block;margin-top:16px;padding:12px 28px;background:#0f172a;color:#fff;text-decoration:none;border-radius:6px;font-weight:bold;">
          View Admin Panel
        </a>
      </div>
    </div>
    """
    return send_email(admin_to, subject, html)
