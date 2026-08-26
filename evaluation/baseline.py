"""
Non-ML recovery strategies.

    naive_strategy(payment_row)       -> always "retry"
    rule_based_strategy(payment_row)  -> retry / stop / notification from failure_reason

Run from repo root (generates CSV first if missing):
    python evaluation/baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Union

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_data" / "payments.csv"

# Allow `python evaluation/baseline.py` without installing the package.
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from simulator import simulate_outcome  # noqa: E402

RowLike = Union[Mapping[str, Any], Any]
StrategyFn = Callable[[RowLike], str]


def naive_strategy(payment_row: RowLike) -> str:
    """Always retry — ignores failure reason and customer context."""
    return "retry"


def rule_based_strategy(payment_row: RowLike) -> str:
    """
    Simple ops playbook:
    - retry transient bank / network failures
    - stop hard declines and expired cards (retrying is usually wasted)
    - notify otherwise (typically insufficient_funds)
    """
    if isinstance(payment_row, Mapping):
        reason = payment_row["failure_reason"]
    else:
        reason = payment_row["failure_reason"]

    if reason in ("temporary_bank_failure", "network_error"):
        return "retry"
    if reason in ("hard_decline", "card_expired"):
        return "stop"
    return "notification"


def evaluate_strategy(df: pd.DataFrame, strategy_fn: StrategyFn) -> dict:
    """Run a strategy through the simulator on every row."""
    n_recovered = 0
    revenue = 0.0
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        intervention = strategy_fn(row_dict)
        outcome = simulate_outcome(row_dict, intervention)
        if outcome["payment_recovered"]:
            n_recovered += 1
            revenue += outcome["recovered_amount"]
    n = len(df)
    return {
        "n": n,
        "n_recovered": n_recovered,
        "recovery_rate_pct": 100.0 * n_recovered / n if n else 0.0,
        "total_revenue_recovered": revenue,
    }


def main() -> None:
    if not DATA_PATH.exists():
        print(f"{DATA_PATH} not found - generating dataset first...")
        sys.path.insert(0, str(PROJECT_ROOT / "data"))
        from generator import generate_payments, save_payments

        save_payments(generate_payments())

    df = pd.read_csv(DATA_PATH)
    naive = evaluate_strategy(df, naive_strategy)
    rules = evaluate_strategy(df, rule_based_strategy)

    print("=" * 72)
    print("Baseline evaluation (simulator over full payments.csv)")
    print("=" * 72)
    print(f"rows: {len(df):,}")
    print()
    print("naive_strategy (always retry)")
    print(f"  total revenue recovered: {naive['total_revenue_recovered']:,.2f}")
    print(f"  recovery rate:           {naive['recovery_rate_pct']:.2f}%")
    print(f"  recovered events:        {naive['n_recovered']:,} / {naive['n']:,}")
    print()
    print("rule_based_strategy (retry transient / stop hard / notify else)")
    print(f"  total revenue recovered: {rules['total_revenue_recovered']:,.2f}")
    print(f"  recovery rate:           {rules['recovery_rate_pct']:.2f}%")
    print(f"  recovered events:        {rules['n_recovered']:,} / {rules['n']:,}")
    print()
    delta_rev = rules["total_revenue_recovered"] - naive["total_revenue_recovered"]
    delta_rate = rules["recovery_rate_pct"] - naive["recovery_rate_pct"]
    print(f"rule_based - naive revenue: {delta_rev:,.2f}")
    print(f"rule_based - naive rate:    {delta_rate:+.2f} pp")
    print("=" * 72)


if __name__ == "__main__":
    main()
