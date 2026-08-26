"""
RecoverIQ payment-recovery simulator.

Given one failed payment row and a chosen intervention, sample an outcome from
the same recoverability process that generated data/sample_data/payments.csv.

Public API (stable for ML / agent teammates):
    simulate_outcome(payment_row, intervention) -> dict

Interventions:
    retry | payment_link | notification | escalate | stop

Run baselines over the full dataset from repo root:
    python evaluation/baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recoverability import (  # noqa: E402
    INTERVENTIONS,
    recovery_probability,
    rng_for,
    sample_recovery_time_hours,
    _get,
)

RowLike = Union[Mapping[str, Any], Any]


def simulate_outcome(payment_row: RowLike, intervention: str) -> dict:
    """
    Simulate recovery for a single payment event.

    Parameters
    ----------
    payment_row :
        dict, pandas Series, or namedtuple with the payments.csv columns.
        Required keys: payment_id, amount, failure_reason, payment_method,
        previous_successes, previous_failures, customer_value.
    intervention :
        One of retry, payment_link, notification, escalate, stop.

    Returns
    -------
    dict
        payment_recovered : bool
        recovered_amount : float   (full `amount` if recovered, else 0.0)
        recovery_time_hours : float | None  (None if not recovered or stop)
        intervention : str
        recovery_probability : float  (oracle P used for the draw; for debugging)
    """
    if intervention not in INTERVENTIONS:
        raise ValueError(
            f"Unknown intervention {intervention!r}. Expected one of {INTERVENTIONS}."
        )

    payment_id = _get(payment_row, "payment_id")
    amount = float(_get(payment_row, "amount"))
    p = recovery_probability(payment_row, intervention)

    rng = rng_for(payment_id, intervention)
    recovered = bool(rng.random() < p)

    if not recovered or intervention == "stop":
        return {
            "payment_recovered": False,
            "recovered_amount": 0.0,
            "recovery_time_hours": None,
            "intervention": intervention,
            "recovery_probability": p,
        }

    return {
        "payment_recovered": True,
        "recovered_amount": round(amount, 2),
        "recovery_time_hours": sample_recovery_time_hours(intervention, rng),
        "intervention": intervention,
        "recovery_probability": p,
    }
