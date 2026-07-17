"""
Parses uploaded bank-statement CSVs into a normalized list of transactions.

Real bank exports vary wildly in column naming (Date/Transaction Date/Posted Date,
Description/Merchant/Memo, Amount/Debit+Credit split, etc). This module tries to
auto-detect common column layouts rather than forcing the user into one exact format
-- a small detail, but it's the difference between a toy demo and something that
actually survives contact with a real CSV.
"""

import pandas as pd
from datetime import datetime

DATE_CANDIDATES = ["date", "transaction date", "posted date", "trans date"]
DESC_CANDIDATES = ["description", "merchant", "memo", "narration", "details", "particulars"]
AMOUNT_CANDIDATES = ["amount", "transaction amount"]
DEBIT_CANDIDATES = ["debit", "withdrawal", "debit amount"]
CREDIT_CANDIDATES = ["credit", "deposit", "credit amount"]


def _find_column(columns_lower, candidates):
    for cand in candidates:
        if cand in columns_lower:
            return columns_lower[cand]
    return None


def _parse_date(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # last resort: let pandas guess
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return None


def parse_bank_csv(filepath: str) -> list[dict]:
    """Returns list of {date, description, amount} dicts.
    Amount convention: negative = expense, positive = income."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    columns_lower = {c.lower().strip(): c for c in df.columns}

    date_col = _find_column(columns_lower, DATE_CANDIDATES)
    desc_col = _find_column(columns_lower, DESC_CANDIDATES)
    amount_col = _find_column(columns_lower, AMOUNT_CANDIDATES)
    debit_col = _find_column(columns_lower, DEBIT_CANDIDATES)
    credit_col = _find_column(columns_lower, CREDIT_CANDIDATES)

    if not date_col or not desc_col:
        raise ValueError(
            "Could not detect required columns. CSV needs a date column "
            f"(one of {DATE_CANDIDATES}) and a description column (one of {DESC_CANDIDATES})."
        )

    if not amount_col and not (debit_col or credit_col):
        raise ValueError(
            f"Could not detect an amount column. Need one of {AMOUNT_CANDIDATES}, "
            f"or separate {DEBIT_CANDIDATES} / {CREDIT_CANDIDATES} columns."
        )

    results = []
    for _, row in df.iterrows():
        date = _parse_date(row[date_col])
        description = str(row[desc_col]).strip()
        if not date or not description or description.lower() == "nan":
            continue  # skip malformed rows rather than crashing the whole upload

        if amount_col:
            try:
                amount = float(row[amount_col])
            except (ValueError, TypeError):
                continue
        else:
            debit = float(row[debit_col]) if debit_col and pd.notna(row.get(debit_col)) else 0.0
            credit = float(row[credit_col]) if credit_col and pd.notna(row.get(credit_col)) else 0.0
            amount = credit - debit  # positive = income, negative = expense

        results.append({"date": date, "description": description, "amount": amount})

    if not results:
        raise ValueError("No valid transaction rows found in the uploaded file.")

    return results
