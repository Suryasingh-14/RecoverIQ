"""Reusable metrics for RecoverIQ evaluation and the Phase 7 dashboard."""
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def calculate_recovery_rate(df_with_outcomes: pd.DataFrame) -> float:
    """Return recovered-event rate as a percentage (0-100)."""
    if len(df_with_outcomes) == 0:
        return 0.0
    if "payment_recovered" not in df_with_outcomes.columns:
        raise KeyError("df_with_outcomes must contain 'payment_recovered'")
    return float(df_with_outcomes["payment_recovered"].astype(bool).mean() * 100.0)


def calculate_total_revenue(df_with_outcomes: pd.DataFrame) -> float:
    """Return total recovered revenue from an outcomes frame."""
    if "recovered_amount" not in df_with_outcomes.columns:
        raise KeyError("df_with_outcomes must contain 'recovered_amount'")
    return float(df_with_outcomes["recovered_amount"].fillna(0).sum())


def summarize_decision_mix(df_with_decisions: pd.DataFrame) -> dict[str, int]:
    """Return intervention counts from a frame containing a ``decision`` column."""
    if "decision" not in df_with_decisions.columns:
        raise KeyError("df_with_decisions must contain 'decision'")
    return {str(k): int(v) for k, v in Counter(df_with_decisions["decision"]).items()}
