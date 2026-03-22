import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


TRIAL_DAYS = 30


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _normalize_company(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _email_domain(email: str) -> str:
    parts = (email or "").split("@", 1)
    return parts[1].strip().lower() if len(parts) == 2 else ""


@contextmanager
def get_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    with get_db(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                company_name TEXT NOT NULL,
                company_name_normalized TEXT NOT NULL,
                company_domain TEXT NOT NULL,
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                subscription_status TEXT NOT NULL DEFAULT 'none',
                plan_code TEXT NOT NULL DEFAULT 'none',
                trial_started_at TEXT,
                trial_ends_at TEXT,
                trial_used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_company_domain ON users(company_domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_company_normalized ON users(company_name_normalized)")


def create_user(
    db_path: str,
    *,
    email: str,
    password_hash: str,
    full_name: str,
    company_name: str,
) -> int:
    now = _utc_now_iso()
    normalized_company = _normalize_company(company_name)
    domain = _email_domain(email)
    with get_db(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO users (
                email, password_hash, full_name, company_name,
                company_name_normalized, company_domain, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                email.strip().lower(),
                password_hash,
                full_name.strip(),
                company_name.strip(),
                normalized_company,
                domain,
                now,
                now,
            ),
        )
        return int(cur.lastrowid)


def get_user_by_email(db_path: str, email: str) -> Optional[Dict[str, Any]]:
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    return dict(row) if row else None


def get_user_by_id(db_path: str, user_id: int) -> Optional[Dict[str, Any]]:
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_customer_id(db_path: str, stripe_customer_id: str) -> Optional[Dict[str, Any]]:
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (stripe_customer_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_subscription_id(db_path: str, stripe_subscription_id: str) -> Optional[Dict[str, Any]]:
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE stripe_subscription_id = ?", (stripe_subscription_id,)).fetchone()
    return dict(row) if row else None


def update_user_fields(db_path: str, user_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _utc_now_iso()
    keys = list(fields.keys())
    assignments = ", ".join(f"{k} = ?" for k in keys)
    values = [fields[k] for k in keys]
    values.append(user_id)
    with get_db(db_path) as conn:
        conn.execute(f"UPDATE users SET {assignments} WHERE id = ?", values)


def trial_eligibility(db_path: str, user: Dict[str, Any]) -> Dict[str, Any]:
    # Rule: 1 trial per company/email/domain
    if int(user.get("trial_used") or 0) == 1:
        return {"eligible": False, "reason": "A trial has already been used for this account."}

    email = (user.get("email") or "").strip().lower()
    domain = (user.get("company_domain") or "").strip().lower()
    company = _normalize_company(user.get("company_name") or "")

    with get_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, email, company_domain, company_name
            FROM users
            WHERE trial_used = 1
              AND id != ?
              AND (
                    email = ?
                 OR company_domain = ?
                 OR company_name_normalized = ?
              )
            LIMIT 1
            """,
            (user["id"], email, domain, company),
        ).fetchone()

    if row:
        return {
            "eligible": False,
            "reason": "A trial has already been used for this company or email domain.",
        }
    return {"eligible": True, "reason": "Eligible for trial"}


def mark_trial_started(db_path: str, user_id: int, trial_ends_at: Optional[str]) -> None:
    update_user_fields(
        db_path,
        user_id,
        trial_used=1,
        trial_started_at=_utc_now_iso(),
        trial_ends_at=trial_ends_at,
    )


def compute_trial_countdown(user: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ends_at = _parse_iso(user.get("trial_ends_at"))
    if not ends_at:
        return None

    now = datetime.now(timezone.utc)
    delta = ends_at - now
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return {"days": 0, "hours": 0, "expired": True}

    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    return {"days": days, "hours": hours, "expired": False}


def plan_label(plan_code: str) -> str:
    return {
        "trial": "Free Trial",
        "monthly": "$99 / month",
        "annual": "$999 / year",
        "none": "No active plan",
    }.get(plan_code or "none", "No active plan")


def subscription_access_allowed(user: Dict[str, Any]) -> bool:
    status = (user.get("subscription_status") or "none").lower()
    return status in {"trialing", "active"}


def serialize_invoices(stripe_invoices: List[Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for inv in stripe_invoices:
        amount_paid = (inv.get("amount_paid") or 0) / 100.0
        created_ts = inv.get("created")
        created_at = datetime.fromtimestamp(created_ts, tz=timezone.utc).strftime("%Y-%m-%d") if created_ts else ""
        output.append(
            {
                "id": inv.get("id"),
                "number": inv.get("number") or inv.get("id"),
                "amount_paid": amount_paid,
                "currency": (inv.get("currency") or "usd").upper(),
                "status": inv.get("status") or "",
                "created_at": created_at,
                "hosted_invoice_url": inv.get("hosted_invoice_url"),
                "invoice_pdf": inv.get("invoice_pdf"),
            }
        )
    return output
