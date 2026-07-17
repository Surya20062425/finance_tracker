import os
import joblib
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

import db
import csv_parser
from train_model import CATEGORY_MERCHANTS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

MODEL_PATH = "model.joblib"
_model = None

ALL_CATEGORIES = sorted(CATEGORY_MERCHANTS.keys())


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model not found. Run `python train_model.py` first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_category(description: str):
    model = get_model()
    pred = model.predict([description])[0]
    proba = model.predict_proba([description])[0]
    confidence = float(max(proba))
    return pred, confidence


@app.before_request
def ensure_db():
    if not os.path.exists(db.DB_PATH):
        db.init_db()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    transactions = db.get_all_transactions(limit=20)
    category_summary = db.get_category_summary()
    monthly_summary = db.get_monthly_summary()
    stats = db.get_stats()
    return render_template(
        "dashboard.html",
        transactions=transactions,
        category_summary=category_summary,
        monthly_summary=monthly_summary,
        stats=stats,
        categories=ALL_CATEGORIES,
    )


# ---------------------------------------------------------------------------
# Manual transaction entry (single row, with live ML suggestion)
# ---------------------------------------------------------------------------

@app.route("/add", methods=["GET", "POST"])
def add_transaction():
    if request.method == "POST":
        date = request.form.get("date")
        description = request.form.get("description", "").strip()
        amount_raw = request.form.get("amount", "").strip()
        category = request.form.get("category", "").strip()

        if not date or not description or not amount_raw:
            flash("Date, description, and amount are required.", "error")
            return redirect(url_for("add_transaction"))

        try:
            amount = float(amount_raw)
        except ValueError:
            flash("Amount must be a number.", "error")
            return redirect(url_for("add_transaction"))

        predicted_category, confidence = predict_category(description)
        final_category = category if category else predicted_category
        is_override = 1 if (category and category != predicted_category) else 0

        db.insert_transaction(
            date=date,
            description=description,
            amount=amount,
            category=final_category,
            predicted_category=predicted_category,
            confidence=confidence,
            is_manual_override=is_override,
        )
        flash(f'Transaction added, categorized as "{final_category}".', "success")
        return redirect(url_for("dashboard"))

    return render_template("add_transaction.html", categories=ALL_CATEGORIES)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Live prediction endpoint — called via JS as the user types a description,
    so they see the suggested category before submitting."""
    description = request.json.get("description", "").strip()
    if not description:
        return jsonify({"category": None, "confidence": 0})
    category, confidence = predict_category(description)
    return jsonify({"category": category, "confidence": round(confidence, 3)})


# ---------------------------------------------------------------------------
# CSV upload (bulk import + auto-categorization)
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please choose a CSV file.", "error")
            return redirect(url_for("upload"))

        if not file.filename.lower().endswith(".csv"):
            flash("Only .csv files are supported.", "error")
            return redirect(url_for("upload"))

        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        try:
            raw_rows = csv_parser.parse_bank_csv(filepath)
        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("upload"))
        finally:
            os.remove(filepath)  # don't retain uploaded files longer than needed

        model = get_model()
        descriptions = [r["description"] for r in raw_rows]
        predictions = model.predict(descriptions)
        probabilities = model.predict_proba(descriptions)

        rows_to_insert = []
        for row, pred, proba in zip(raw_rows, predictions, probabilities):
            confidence = float(max(proba))
            category = "Income" if row["amount"] > 0 and pred != "Income" and confidence < 0.5 else pred
            rows_to_insert.append({
                "date": row["date"],
                "description": row["description"],
                "amount": row["amount"],
                "category": category,
                "predicted_category": pred,
                "confidence": confidence,
                "is_manual_override": 0,
            })

        db.bulk_insert_transactions(rows_to_insert)
        flash(f"Imported {len(rows_to_insert)} transactions and auto-categorized them.", "success")
        return redirect(url_for("dashboard"))

    return render_template("upload.html")


# ---------------------------------------------------------------------------
# Correcting a prediction (feedback loop pattern — important for ML products)
# ---------------------------------------------------------------------------

@app.route("/transaction/<int:txn_id>/recategorize", methods=["POST"])
def recategorize(txn_id):
    new_category = request.form.get("category")
    if new_category not in ALL_CATEGORIES:
        flash("Invalid category.", "error")
        return redirect(url_for("dashboard"))
    db.update_transaction_category(txn_id, new_category)
    flash("Category updated.", "success")
    return redirect(url_for("dashboard"))


@app.route("/transaction/<int:txn_id>/delete", methods=["POST"])
def delete_transaction(txn_id):
    db.delete_transaction(txn_id)
    flash("Transaction deleted.", "success")
    return redirect(url_for("dashboard"))


@app.route("/transactions")
def all_transactions():
    transactions = db.get_all_transactions(limit=1000)
    return render_template("transactions.html", transactions=transactions, categories=ALL_CATEGORIES)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("No trained model found — training now...")
        from train_model import train_and_save
        train_and_save(MODEL_PATH)
    db.init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
