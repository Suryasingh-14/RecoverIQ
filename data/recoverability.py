"""
Shared recoverability logic used by the simulator (and diagnostics in the generator).

This is the ground-truth process that produced realistic feature relationships in
payments.csv. Downstream ML should NOT import this module for training labels in
production-style experiments — treat it as an oracle for simulation only.

Column names here MUST match data/sample_data/payments.csv exactly.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional, Union

import numpy as np

FAILURE_REASONS = (
    "temporary_bank_failure",
    "insufficient_funds",
    "card_expired",
    "hard_decline",
    "network_error",
)

PAYMENT_METHODS = ("card", "upi", "netbanking", "wallet")

INTERVENTIONS = ("retry", "payment_link", "notification", "escalate", "stop")

# Base P(recover) given failure reason (before customer/method/history adjustments).
# temporary_bank_failure / network_error are MORE recoverable;
# hard_decline / card_expired are LESS recoverable.
REASON_BASE_RECOVERY = {
    "temporary_bank_failure": 0.62,
    "network_error": 0.58,
    "insufficient_funds": 0.32,
    "card_expired": 0.10,
    "hard_decline": 0.06,
}

# Method quality: UPI/netbanking/wallet recover slightly better than cards.
METHOD_RECOVERY_DELTA = {
    "upi": 0.08,
    "netbanking": 0.05,
    "wallet": 0.03,
    "card": -0.02,
}

# Multipliers applied to the latent recoverability score.
# retry is strong on transient errors, weak on hard/expired.
# notification is strongest on insufficient_funds (nudge / wait for balance).
# payment_link helps when the instrument itself is the problem (expired card).
# escalate helps a bit everywhere, more for high-value customers.
# stop always yields no recovery.
INTERVENTION_MULTIPLIER = {
    "retry": {
        "temporary_bank_failure": 1.25,
        "network_error": 1.30,
        "insufficient_funds": 0.55,
        "card_expired": 0.20,
        "hard_decline": 0.12,
    },
    "payment_link": {
        "temporary_bank_failure": 0.90,
        "network_error": 0.85,
        "insufficient_funds": 1.15,
        "card_expired": 2.20,
        "hard_decline": 0.35,
    },
    "notification": {
        "temporary_bank_failure": 0.80,
        "network_error": 0.75,
        "insufficient_funds": 1.45,
        "card_expired": 0.90,
        "hard_decline": 0.25,
    },
    "escalate": {
        "temporary_bank_failure": 1.05,
        "network_error": 1.00,
        "insufficient_funds": 1.10,
        "card_expired": 1.30,
        "hard_decline": 0.80,
    },
    "stop": {
        "temporary_bank_failure": 0.0,
        "network_error": 0.0,
        "insufficient_funds": 0.0,
        "card_expired": 0.0,
        "hard_decline": 0.0,
    },
}

# Hours until recovery, sampled log-uniform in [low, high] when recovered.
RECOVERY_TIME_HOURS = {
    "retry": (0.25, 8.0),
    "payment_link": (1.0, 48.0),
    "notification": (6.0, 96.0),
    "escalate": (2.0, 72.0),
    "stop": (0.0, 0.0),
}

RowLike = Union[Mapping[str, Any], Any]


def _get(row: RowLike, key: str) -> Any:
    """Read a column from a dict, pandas Series, or namedtuple-like row."""
    if isinstance(row, Mapping):
        return row[key]
    if hasattr(row, "loc") or hasattr(row, "index"):
        return row[key]
    return getattr(row, key)


def recoverability_score(row: RowLike) -> float:
    """
    Latent recoverability in (0.01, 0.95) from payment features only.

    Relationships (must stay aligned with generator.py):
    - failure_reason drives most of the signal
    - more previous_failures lowers recovery
    - more previous_successes raises recovery
    - higher customer_value raises recovery slightly
    - larger amount lowers recovery slightly
    """
    reason = str(_get(row, "failure_reason"))
    method = str(_get(row, "payment_method"))
    prev_fail = float(_get(row, "previous_failures"))
    prev_ok = float(_get(row, "previous_successes"))
    value = float(_get(row, "customer_value"))
    amount = float(_get(row, "amount"))

    p = REASON_BASE_RECOVERY[reason]
    p += METHOD_RECOVERY_DELTA[method]
    p -= 0.035 * min(prev_fail, 12.0)
    p += 0.012 * min(prev_ok, 20.0)
    p += 0.018 * np.tanh(np.log1p(value) / 10.0)
    p -= 0.020 * np.tanh(amount / 10_000.0)
    return float(np.clip(p, 0.01, 0.95))


def recovery_probability(row: RowLike, intervention: str) -> float:
    """P(payment_recovered | features, intervention)."""
    if intervention not in INTERVENTIONS:
        raise ValueError(
            f"Unknown intervention {intervention!r}. "
            f"Expected one of {INTERVENTIONS}."
        )
    reason = str(_get(row, "failure_reason"))
    if reason not in REASON_BASE_RECOVERY:
        raise ValueError(f"Unknown failure_reason {reason!r}.")

    base = recoverability_score(row)
    multiplier = INTERVENTION_MULTIPLIER[intervention][reason]
    if intervention == "escalate":
        value = float(_get(row, "customer_value"))
        multiplier += 0.15 * np.tanh(np.log1p(value) / 10.0)
    if intervention == "stop":
        return 0.0
    return float(np.clip(base * multiplier, 0.0, 0.95))


def rng_for(payment_id: Any, intervention: str) -> np.random.RandomState:
    """Deterministic RNG so the same (payment, intervention) always simulates the same outcome."""
    payload = f"{payment_id}|{intervention}".encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    seed = int(digest[:8], 16)
    return np.random.RandomState(seed)


def sample_recovery_time_hours(
    intervention: str, rng: np.random.RandomState
) -> Optional[float]:
    low, high = RECOVERY_TIME_HOURS[intervention]
    if high <= 0:
        return None
    # Log-uniform: most recoveries relatively fast, some long-tail delays.
    log_low, log_high = np.log(low), np.log(high)
    hours = float(np.exp(rng.uniform(log_low, log_high)))
    return round(hours, 2)
