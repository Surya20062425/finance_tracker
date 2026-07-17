"""
Trains a text-classification model that predicts an expense category
from a transaction description (e.g. "STARBUCKS #4521" -> "Food & Dining").

Pipeline: TF-IDF (character + word n-grams) -> Linear SVM (calibrated for probabilities)
This combo works well on short, noisy merchant strings, which is exactly what
real bank statement descriptions look like.
"""

import random
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

random.seed(42)

# ---------------------------------------------------------------------------
# 1. Synthetic-but-realistic training data.
#    In production you'd replace this with real (anonymized) transaction
#    history. Here we generate merchant-style strings per category, with
#    realistic noise (store numbers, city codes, asterisks) since that's
#    exactly what bank CSV exports look like.
# ---------------------------------------------------------------------------

CATEGORY_MERCHANTS = {
    "Food & Dining": [
        "STARBUCKS", "MCDONALDS", "CHIPOTLE", "DOMINOS PIZZA", "SUBWAY",
        "PANERA BREAD", "TACO BELL", "KFC", "PIZZA HUT", "DUNKIN",
        "OLIVE GARDEN", "CHICK-FIL-A", "WENDYS", "SWIGGY", "ZOMATO",
        "UBER EATS", "DOORDASH", "LOCAL CAFE", "RESTAURANT", "BURGER KING",
    ],
    "Groceries": [
        "WALMART GROCERY", "WHOLE FOODS", "TRADER JOES", "KROGER",
        "SAFEWAY", "COSTCO WHOLESALE", "ALDI", "BIGBASKET", "RELIANCE FRESH",
        "MORE SUPERMARKET", "TARGET GROCERY", "PUBLIX", "SPROUTS FARMERS",
    ],
    "Transportation": [
        "UBER TRIP", "LYFT RIDE", "SHELL OIL", "CHEVRON GAS", "OLA CABS",
        "METRO TRANSIT", "PARKING GARAGE", "EXXONMOBIL", "BP GAS STATION",
        "INDIAN RAILWAYS", "AIRLINE TICKET", "TOLL PLAZA", "CAR SERVICE",
    ],
    "Shopping": [
        "AMAZON.COM", "AMAZON MKTPLACE", "TARGET STORE", "BEST BUY",
        "MYNTRA", "FLIPKART", "H&M", "ZARA", "NIKE STORE", "APPLE STORE",
        "IKEA", "MACYS", "ETSY", "EBAY", "H AND M ONLINE",
    ],
    "Entertainment": [
        "NETFLIX.COM", "SPOTIFY PREMIUM", "AMC THEATERS", "DISNEY PLUS",
        "HULU", "PVR CINEMAS", "STEAM GAMES", "PLAYSTATION STORE",
        "HBO MAX", "YOUTUBE PREMIUM", "BOWLING ALLEY", "CONCERT TICKETS",
    ],
    "Utilities": [
        "ELECTRIC COMPANY", "WATER UTILITY", "COMCAST CABLE", "AT&T WIRELESS",
        "VERIZON WIRELESS", "GAS UTILITY CO", "INTERNET SERVICE", "T-MOBILE",
        "AIRTEL POSTPAID", "JIO FIBER", "SPECTRUM INTERNET", "CON EDISON",
    ],
    "Healthcare": [
        "CVS PHARMACY", "WALGREENS", "MEDICAL CENTER", "DENTAL CLINIC",
        "APOLLO PHARMACY", "URGENT CARE", "HEALTH INSURANCE", "RITE AID",
        "FAMILY CLINIC", "EYE CARE CENTER", "PHYSIOTHERAPY CLINIC",
    ],
    "Rent & Housing": [
        "RENT PAYMENT", "PROPERTY MGMT", "APARTMENT LEASE", "HOA FEES",
        "MORTGAGE PAYMENT", "HOME INSURANCE", "MAINTENANCE FEE",
    ],
    "Income": [
        "PAYROLL DEPOSIT", "SALARY CREDIT", "DIRECT DEPOSIT", "EMPLOYER PAY",
        "FREELANCE PAYMENT", "INTEREST CREDIT", "REFUND CREDIT", "BONUS PAY",
    ],
    "Subscriptions": [
        "ADOBE CREATIVE", "MICROSOFT 365", "GYM MEMBERSHIP", "LINKEDIN PREMIUM",
        "NEWSPAPER SUB", "AUDIBLE MEMBERSHIP", "ICLOUD STORAGE", "NOTION PLAN",
        "PLANET FITNESS", "AMAZON PRIME",
    ],
}

NOISE_SUFFIXES = [
    "", " #{}".format(random.randint(1000, 9999)), " *{}".format(random.randint(100, 999)),
    " REF{}".format(random.randint(10000, 99999)), " {}".format(random.choice(["NY", "CA", "TX", "MUM", "BLR"])),
    " ONLINE", " POS", " WEB",
]


def make_noisy_description(merchant: str) -> str:
    suffix = random.choice(NOISE_SUFFIXES)
    # occasionally randomize casing/spacing to mimic messy bank exports
    desc = f"{merchant}{suffix}"
    if random.random() < 0.15:
        desc = desc.replace(" ", "  ")  # double space noise
    return desc


def build_dataset(samples_per_category: int = 120) -> pd.DataFrame:
    rows = []
    for category, merchants in CATEGORY_MERCHANTS.items():
        for _ in range(samples_per_category):
            merchant = random.choice(merchants)
            rows.append({
                "description": make_noisy_description(merchant),
                "category": category,
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def train_and_save(model_path: str = "model.joblib") -> None:
    df = build_dataset()
    print(f"Dataset size: {len(df)} rows across {df['category'].nunique()} categories")

    X_train, X_test, y_train, y_test = train_test_split(
        df["description"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",   # char n-grams handle merchant noise (numbers, symbols) well
            ngram_range=(2, 4),
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=42), cv=3
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train_and_save()
