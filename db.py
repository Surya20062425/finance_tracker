"""
SQLite data layer. Kept dependency-free (stdlib sqlite3) so the project
runs with zero external DB setup — a deliberate choice for a portfolio
project: reviewers should be able to clone and run in under a minute.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = "finance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    amount REAL NOT NULL,
    category TEXT NOT NULL,
    predicted_category TEXT,
    confidence REAL,
    is_manual_override INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_transaction(date, description, amount, category, predicted_category=None,
                        confidence=None, is_manual_override=0):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO transactions
               (date, description, amount, category, predicted_category, confidence,
                is_manual_override, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (date, description, amount, category, predicted_category, confidence,
             is_manual_override, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def bulk_insert_transactions(rows):
    """rows: list of dicts with keys date, description, amount, category,
    predicted_category, confidence, is_manual_override"""
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO transactions
               (date, description, amount, category, predicted_category, confidence,
                is_manual_override, created_at)
               VALUES (:date, :description, :amount, :category, :predicted_category,
                       :confidence, :is_manual_override, :created_at)""",
            [{**r, "created_at": datetime.utcnow().isoformat()} for r in rows],
        )


def get_all_transactions(limit=500):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM transactions ORDER BY date DESC, id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cur.fetchall()]


def update_transaction_category(txn_id, new_category):
    with get_conn() as conn:
        conn.execute(
            "UPDATE transactions SET category = ?, is_manual_override = 1 WHERE id = ?",
            (new_category, txn_id),
        )


def delete_transaction(txn_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM transactions WHERE id = ?", (txn_id,))


def get_category_summary():
    """Returns spend per category (expenses only, i.e. negative amounts /
    excluding Income) for chart rendering."""
    with get_conn() as conn:
        cur = conn.execute(
            """SELECT category, SUM(ABS(amount)) as total, COUNT(*) as count
               FROM transactions
               WHERE category != 'Income'
               GROUP BY category
               ORDER BY total DESC"""
        )
        return [dict(row) for row in cur.fetchall()]


def get_monthly_summary():
    with get_conn() as conn:
        cur = conn.execute(
            """SELECT substr(date, 1, 7) as month,
                      SUM(CASE WHEN category = 'Income' THEN amount ELSE 0 END) as income,
                      SUM(CASE WHEN category != 'Income' THEN ABS(amount) ELSE 0 END) as expenses
               FROM transactions
               GROUP BY month
               ORDER BY month"""
        )
        return [dict(row) for row in cur.fetchall()]


def get_stats():
    with get_conn() as conn:
        row = conn.execute(
            """SELECT
                 COUNT(*) as total_transactions,
                 SUM(CASE WHEN category = 'Income' THEN amount ELSE 0 END) as total_income,
                 SUM(CASE WHEN category != 'Income' THEN ABS(amount) ELSE 0 END) as total_expenses,
                 SUM(CASE WHEN is_manual_override = 1 THEN 1 ELSE 0 END) as manual_overrides
               FROM transactions"""
        ).fetchone()
        return dict(row) if row else {}
