# Personal Finance Tracker with ML-Based Expense Categorization

A full-stack Flask web app that automatically categorizes bank transactions
using a trained text-classification model, with a dashboard for visualizing
spending patterns over time.

## What it does

- **Upload a bank statement CSV** → every row is automatically categorized
  (Food & Dining, Groceries, Transportation, etc.) using a trained ML model.
- **Add transactions manually** with a live "as-you-type" category suggestion.
- **Correct any prediction** with one click — the app tracks which
  categorizations were manually overridden vs. trusted from the model.
- **Dashboard** with spend-by-category (doughnut chart) and monthly
  income-vs-expenses (bar chart), plus running totals.
- **Auto-detects common bank CSV formats** — handles both single "Amount"
  columns and split Debit/Credit columns, with flexible date/description
  column naming.

## Why it's built this way (design notes)

- **TF-IDF (character n-grams) + Linear SVM**, not a deep learning model.
  Transaction descriptions are short, noisy strings ("STARBUCKS #4521 POS"),
  not natural language — character-level n-grams handle merchant codes,
  numbers, and abbreviations far better than word embeddings would, and
  the whole pipeline trains in under a second. This is a deliberate
  "right tool for the problem" choice, not a limitation.
- **SQLite**, not Postgres. Zero setup for anyone reviewing the project —
  clone, `pip install`, run. The `db.py` module is small and isolated
  enough that swapping in Postgres later would only mean changing the
  connection layer.
- **Human-in-the-loop correction**, not just prediction. Every ML product
  needs a feedback path for when the model is wrong; the recategorize
  flow (and the `is_manual_override` flag) reflects that.
- **Synthetic training data.** The model is trained on generated
  merchant-style strings (see `train_model.py`) rather than real bank
  data, for obvious privacy reasons. This is disclosed here rather than
  hidden — accuracy on real, messier statements will be lower than the
  ~100% test accuracy on synthetic data, and that gap is worth discussing
  in an interview, not glossing over.

## Setup

```bash
pip install -r requirements.txt

# Train the model (creates model.joblib)
python train_model.py

# Optional: populate ~4 months of sample data so the dashboard isn't empty
python seed_data.py

# Run the app
python app.py
```

Then open **http://127.0.0.1:5000**.

## Project structure

```
finance-tracker/
├── app.py              # Flask routes
├── db.py                # SQLite data layer
├── csv_parser.py        # Bank CSV auto-detection + normalization
├── train_model.py       # ML training pipeline + synthetic dataset generator
├── seed_data.py          # Demo data generator (optional)
├── templates/            # Jinja2 HTML templates
├── static/style.css      # Styling
└── requirements.txt
```

## Possible extensions (good "what would you add next" answers)

- Swap the synthetic dataset for a real (anonymized) one and report the
  accuracy delta — this is the single most credible way to strengthen
  the ML story here.
- Add a `/retrain` endpoint that incorporates manually-corrected rows
  back into training data, closing the feedback loop end-to-end.
- Move from SQLite to Postgres + add user accounts, so the tracker is
  multi-user instead of single-user/local.
- Add budget thresholds per category with alerting when exceeded.
