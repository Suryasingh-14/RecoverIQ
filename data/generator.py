"""
RecoverIQ synthetic payment-event generator.

Output path (do not rename — ML / eval code depends on it):
    data/sample_data/payments.csv

Exact columns (order preserved):
    payment_id, customer_id, timestamp, amount, payment_method, failure_reason,
    customer_age, previous_successes, previous_failures, customer_value,
    subscription_age, days_since_last_payment

All 15,000 rows are FAILED payment events (the recovery decision surface).
Feature relationships are generated so that recoverability.py is a realistic oracle,
not independent random noise.

Run from repo root:
    python data/generator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Script dir on path so `python data/generator.py` and package imports both work.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recoverability import (  # noqa: E402
    FAILURE_REASONS,
    PAYMENT_METHODS,
    recoverability_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_EVENTS = 15_000
N_CUSTOMERS = 4_000
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "sample_data" / "payments.csv"

COLUMNS = [
    "payment_id",
    "customer_id",
    "timestamp",
    "amount",
    "payment_method",
    "failure_reason",
    "customer_age",
    "previous_successes",
    "previous_failures",
    "customer_value",
    "subscription_age",
    "days_since_last_payment",
]

# Baseline *failure* rates by method (used to mix methods + reason types).
# Higher-value customers are steered toward lower-failure methods (UPI / netbanking).
METHOD_FAILURE_RATE = {
    "card": 0.085,
    "netbanking": 0.060,
    "wallet": 0.050,
    "upi": 0.038,
}

# P(method | customer_value tertile). High-value customers use better rails.
METHOD_MIX_BY_VALUE_TERTILE = {
    0: np.array([0.50, 0.22, 0.12, 0.16]),  # card, upi, netbanking, wallet
    1: np.array([0.30, 0.40, 0.15, 0.15]),
    2: np.array([0.14, 0.46, 0.25, 0.15]),
}

# P(failure_reason | payment_method). Cards fail more on expiry / hard decline;
# UPI / netbanking fail more on transient bank / network issues.
REASON_MIX_BY_METHOD = {
    "card": {
        "temporary_bank_failure": 0.15,
        "insufficient_funds": 0.25,
        "card_expired": 0.25,
        "hard_decline": 0.20,
        "network_error": 0.15,
    },
    "upi": {
        "temporary_bank_failure": 0.35,
        "insufficient_funds": 0.25,
        "card_expired": 0.02,
        "hard_decline": 0.08,
        "network_error": 0.30,
    },
    "netbanking": {
        "temporary_bank_failure": 0.30,
        "insufficient_funds": 0.30,
        "card_expired": 0.03,
        "hard_decline": 0.12,
        "network_error": 0.25,
    },
    "wallet": {
        "temporary_bank_failure": 0.20,
        "insufficient_funds": 0.40,
        "card_expired": 0.05,
        "hard_decline": 0.10,
        "network_error": 0.25,
    },
}


def _sample_categorical(
    rng: np.random.Generator, labels: list[str], probs: np.ndarray, size: int
) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(labels, size=size, p=probs)


def build_customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Customer-level attributes. Value, history, and age are correlated."""
    # Right-skewed LTV (INR). Median roughly a few thousand, long tail of whales.
    customer_value = np.clip(rng.lognormal(mean=8.2, sigma=0.85, size=n), 200, 500_000)

    customer_age = rng.integers(18, 71, size=n)
    # Subscription tenure in months; older accounts tend to be higher value.
    value_rank = (customer_value - customer_value.min()) / (
        customer_value.max() - customer_value.min() + 1e-9
    )
    subscription_age = np.clip(
        rng.poisson(lam=6 + 24 * value_rank, size=n), 1, 84
    ).astype(int)

    # Successful history grows with tenure and value; failures fall with value.
    previous_successes = np.clip(
        rng.poisson(lam=2 + 0.35 * subscription_age + 4 * value_rank, size=n),
        0,
        80,
    ).astype(int)
    previous_failures = np.clip(
        rng.poisson(lam=3.5 - 2.2 * value_rank, size=n),
        0,
        25,
    ).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
            "customer_age": customer_age,
            "customer_value": np.round(customer_value, 2),
            "subscription_age": subscription_age,
            "previous_successes": previous_successes,
            "previous_failures": previous_failures,
        }
    )


def assign_methods(rng: np.random.Generator, customer_value: np.ndarray) -> np.ndarray:
    """Higher customer_value → more UPI/netbanking (better rails, higher recovery)."""
    tertile = pd.qcut(customer_value, 3, labels=False, duplicates="drop").astype(int)
    methods = np.empty(len(customer_value), dtype=object)
    labels = list(PAYMENT_METHODS)
    for t in range(3):
        mask = tertile == t
        k = int(mask.sum())
        if k == 0:
            continue
        mix = METHOD_MIX_BY_VALUE_TERTILE.get(t, METHOD_MIX_BY_VALUE_TERTILE[1])
        methods[mask] = _sample_categorical(rng, labels, mix, k)
    return methods


def assign_failure_reasons(
    rng: np.random.Generator, payment_method: np.ndarray
) -> np.ndarray:
    reasons = np.empty(len(payment_method), dtype=object)
    for method in PAYMENT_METHODS:
        mask = payment_method == method
        k = int(mask.sum())
        if k == 0:
            continue
        mix = REASON_MIX_BY_METHOD[method]
        labels = list(mix.keys())
        probs = np.array(list(mix.values()), dtype=float)
        reasons[mask] = _sample_categorical(rng, labels, probs, k)
    return reasons


def generate_payments(
    n_events: int = N_EVENTS,
    n_customers: int = N_CUSTOMERS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Build a 15k-row failed-payment table with correlated features."""
    rng = np.random.default_rng(seed)
    customers = build_customers(rng, n_customers)

    # How many failed events per customer (sum renormalized to n_events).
    raw_counts = rng.poisson(lam=n_events / n_customers, size=n_customers) + 1
    weights = raw_counts / raw_counts.sum()
    counts = rng.multinomial(n_events, weights)

    customer_index = np.repeat(np.arange(n_customers), counts)
    # Shuffle so timestamps are not blocked by customer_id.
    rng.shuffle(customer_index)
    base = customers.iloc[customer_index].reset_index(drop=True)

    # Amount: right-skewed (mostly small–medium, some large). Slightly higher for whales.
    log_mean = 6.1 + 0.35 * np.log1p(base["customer_value"].to_numpy()) / 10.0
    amount = rng.lognormal(mean=log_mean, sigma=0.95)
    amount = np.clip(amount, 49.0, 75_000.0)
    amount = np.round(amount, 2)

    payment_method = assign_methods(rng, base["customer_value"].to_numpy())
    failure_reason = assign_failure_reasons(rng, payment_method)

    # Event-level jitter on history so two events from the same customer are not clones.
    prev_ok = np.clip(
        base["previous_successes"].to_numpy() + rng.integers(0, 3, size=n_events),
        0,
        90,
    )
    prev_fail = np.clip(
        base["previous_failures"].to_numpy() + rng.integers(0, 2, size=n_events),
        0,
        30,
    )

    # Days since last payment: longer gaps when the customer already fails often.
    days_since = np.clip(
        rng.poisson(lam=8 + 1.6 * prev_fail, size=n_events),
        0,
        120,
    ).astype(int)

    # Timestamps over the last ~120 days ending 2026-08-26 (see user_info date).
    end = np.datetime64("2026-08-26T21:00:00")
    offsets_min = rng.integers(0, 120 * 24 * 60, size=n_events)
    timestamps = end - offsets_min.astype("timedelta64[m]")

    df = pd.DataFrame(
        {
            "payment_id": [f"PAY-{i:07d}" for i in range(1, n_events + 1)],
            "customer_id": base["customer_id"].to_numpy(),
            "timestamp": pd.to_datetime(timestamps),
            "amount": amount,
            "payment_method": payment_method,
            "failure_reason": failure_reason,
            "customer_age": base["customer_age"].to_numpy(),
            "previous_successes": prev_ok,
            "previous_failures": prev_fail,
            "customer_value": base["customer_value"].to_numpy(),
            "subscription_age": base["subscription_age"].to_numpy(),
            "days_since_last_payment": days_since,
        }
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["payment_id"] = [f"PAY-{i:07d}" for i in range(1, n_events + 1)]
    return df[COLUMNS]


def print_summary(df: pd.DataFrame) -> None:
    """Sanity-check stats: counts, reason mix, distributions, recoverability by reason."""
    print("=" * 72)
    print("RecoverIQ payments.csv summary")
    print("=" * 72)
    print(f"row_count: {len(df):,}")
    print(f"unique_customers: {df['customer_id'].nunique():,}")
    print(f"date_range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print()

    print("failure_reason mix (share of failed events - all rows are failures):")
    reason_counts = df["failure_reason"].value_counts().reindex(FAILURE_REASONS)
    reason_pct = (reason_counts / len(df) * 100).round(2)
    print(
        pd.DataFrame({"count": reason_counts, "pct": reason_pct}).to_string()
    )
    print()

    print("payment_method mix:")
    method_counts = df["payment_method"].value_counts().reindex(list(PAYMENT_METHODS))
    print(
        pd.DataFrame(
            {
                "count": method_counts,
                "pct": (method_counts / len(df) * 100).round(2),
                "baseline_failure_rate": [
                    METHOD_FAILURE_RATE[m] for m in PAYMENT_METHODS
                ],
            }
        ).to_string()
    )
    print()

    print("amount distribution (right-skew expected: mean > median):")
    print(df["amount"].describe(percentiles=[0.5, 0.75, 0.9, 0.99]).to_string())
    print(f"skewness: {df['amount'].skew():.3f}")
    print()

    print("customer_value / history / tenure:")
    print(
        df[
            [
                "customer_value",
                "previous_successes",
                "previous_failures",
                "subscription_age",
                "days_since_last_payment",
                "customer_age",
            ]
        ]
        .describe()
        .round(2)
        .to_string()
    )
    print()

    # Oracle diagnostics — these columns are NOT written to CSV.
    scores = df.apply(recoverability_score, axis=1)
    print("latent recoverability_score (oracle, not in CSV) by failure_reason:")
    print(
        df.assign(_score=scores)
        .groupby("failure_reason")["_score"]
        .mean()
        .reindex(FAILURE_REASONS)
        .round(3)
        .to_string()
    )
    print()
    print("mean latent recoverability by previous_failures bucket:")
    buckets = pd.cut(
        df["previous_failures"],
        bins=[-0.1, 0, 2, 5, 10, 99],
        labels=["0", "1-2", "3-5", "6-10", "11+"],
    )
    print(scores.groupby(buckets, observed=False).mean().round(3).to_string())
    print()
    print("mean latent recoverability by customer_value tertile:")
    tertile = pd.qcut(df["customer_value"], 3, labels=["low", "mid", "high"])
    print(scores.groupby(tertile, observed=False).mean().round(3).to_string())
    print()
    print("mean customer_value by payment_method (high-value -> better rails):")
    print(
        df.groupby("payment_method")["customer_value"]
        .mean()
        .reindex(list(PAYMENT_METHODS))
        .round(2)
        .to_string()
    )
    print("=" * 72)


def save_payments(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {len(df):,} rows -> {path}")
    return path


def main() -> None:
    df = generate_payments()
    save_payments(df)
    print_summary(df)


if __name__ == "__main__":
    main()
