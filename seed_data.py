"""
Populates the database with ~4 months of realistic sample transactions.
Run this after training the model so the dashboard looks meaningful on
first launch — an empty dashboard is a weak first impression in a demo/interview.
"""

import random
import joblib
from datetime import date, timedelta

import db
from train_model import CATEGORY_MERCHANTS, make_noisy_description

random.seed(7)

MONTHLY_BILLS = [
    ("Rent & Housing", "RENT PAYMENT", -1450.00, 1),
    ("Utilities", "ELECTRIC COMPANY", -85.00, 3),
    ("Utilities", "COMCAST CABLE", -79.99, 5),
    ("Subscriptions", "NETFLIX.COM", -15.49, 8),
    ("Subscriptions", "SPOTIFY PREMIUM", -10.99, 8),
    ("Subscriptions", "PLANET FITNESS", -24.99, 2),
    ("Income", "PAYROLL DEPOSIT", 4200.00, 1),
    ("Income", "PAYROLL DEPOSIT", 4200.00, 15),
]

VARIABLE_CATEGORIES = [
    "Food & Dining", "Groceries", "Transportation", "Shopping",
    "Entertainment", "Healthcare",
]


def random_amount_for(category):
    ranges = {
        "Food & Dining": (8, 55),
        "Groceries": (25, 140),
        "Transportation": (10, 65),
        "Shopping": (15, 200),
        "Entertainment": (10, 80),
        "Healthcare": (20, 250),
    }
    lo, hi = ranges.get(category, (10, 100))
    return -round(random.uniform(lo, hi), 2)


def generate_transactions(months_back=4):
    model = joblib.load("model.joblib")
    today = date.today()
    start = today - timedelta(days=30 * months_back)

    rows = []
    current = start
    while current <= today:
        # monthly bills fire on their fixed day-of-month
        for category, merchant, amount, day in MONTHLY_BILLS:
            if current.day == day:
                desc = make_noisy_description(merchant)
                rows.append({"date": current.isoformat(), "description": desc,
                             "amount": amount, "category": category})

        # 0-3 random variable transactions per day
        for _ in range(random.randint(0, 3)):
            category = random.choice(VARIABLE_CATEGORIES)
            merchant = random.choice(CATEGORY_MERCHANTS[category])
            desc = make_noisy_description(merchant)
            amount = random_amount_for(category)
            rows.append({"date": current.isoformat(), "description": desc,
                         "amount": amount, "category": category})

        current += timedelta(days=1)

    # run everything through the model, same as a real CSV upload would
    descriptions = [r["description"] for r in rows]
    predictions = model.predict(descriptions)
    probabilities = model.predict_proba(descriptions)

    final_rows = []
    for row, pred, proba in zip(rows, predictions, probabilities):
        confidence = float(max(proba))
        final_rows.append({
            "date": row["date"],
            "description": row["description"],
            "amount": row["amount"],
            "category": pred,  # trust the model's prediction, like a real upload
            "predicted_category": pred,
            "confidence": confidence,
            "is_manual_override": 0,
        })

    return final_rows


if __name__ == "__main__":
    db.init_db()
    rows = generate_transactions()
    db.bulk_insert_transactions(rows)
    print(f"Seeded {len(rows)} sample transactions.")
