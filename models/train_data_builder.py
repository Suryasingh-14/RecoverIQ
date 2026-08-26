"""
Build supervised training data from payments.csv + the simulator.

For every failed payment, label = simulate_outcome(row, intervention="retry")
["payment_recovered"] as 0/1.

This is a *retry* policy dataset. It does not label payment_link / notification /
escalate / stop. Phase 3 EV / decision code must not treat these probabilities
as intervention-agnostic.

Does NOT import data/recoverability.py (oracle). Labels come only from
simulate_outcome() in data/simulator.py.

Output: models/training_data.csv
    original payments.csv columns + payment_recovered

Run from repo root:
    python models/train_data_builder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAYMENTS_PATH = PROJECT_ROOT / "data" / "sample_data" / "payments.csv"
OUTPUT_PATH = PROJECT_ROOT / "models" / "training_data.csv"

# Public simulator only — never import recoverability from model code.
sys.path.insert(0, str(PROJECT_ROOT / "data"))
from simulator import simulate_outcome  # noqa: E402


def label_retry_outcomes(payments: pd.DataFrame) -> pd.Series:
    """Bernoulli retry outcomes from the simulator, aligned to ``payments`` index."""
    labels = []
    for row in payments.itertuples(index=False):
        outcome = simulate_outcome(row._asdict(), "retry")
        labels.append(int(bool(outcome["payment_recovered"])))
    return pd.Series(labels, index=payments.index, name="payment_recovered")


def build_training_data(
    payments_path: Path = PAYMENTS_PATH,
) -> pd.DataFrame:
    if not payments_path.exists():
        raise FileNotFoundError(
            f"{payments_path} not found. Run: python data/generator.py"
        )
    payments = pd.read_csv(payments_path)
    if "payment_recovered" in payments.columns:
        raise ValueError(
            "payments.csv already has payment_recovered; refusing to overwrite labels."
        )
    train = payments.copy()
    train["payment_recovered"] = label_retry_outcomes(payments)
    return train


def save_training_data(
    df: pd.DataFrame, path: Path = OUTPUT_PATH
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    df = build_training_data()
    out = save_training_data(df)
    n = len(df)
    n_pos = int(df["payment_recovered"].sum())
    print(f"wrote {n:,} rows -> {out}")
    print(f"payment_recovered=1: {n_pos:,} ({100.0 * n_pos / n:.2f}%)")
    print(f"payment_recovered=0: {n - n_pos:,} ({100.0 * (n - n_pos) / n:.2f}%)")
    print("label source: simulate_outcome(row, intervention='retry')")
    print("oracle recoverability.py was not imported")


if __name__ == "__main__":
    main()
