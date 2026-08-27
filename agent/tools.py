"""
RecoverIQ Phase 4 -- agent tools.

Thin wrappers around existing data lookups and Phase 3 EV functions.
Does not reimplement expected value or decision logic.
Does not import data/recoverability.py.
Does not call simulate_outcome except via calculate_expected_value().
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "engine"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from expected_value import VALID_INTERVENTIONS, calculate_expected_value  # noqa: E402

PAYMENTS_CSV = PROJECT_ROOT / "data" / "sample_data" / "payments.csv"

_ACTION_NOT_EXECUTED = (
    "This agent is decision-only. Actions are not executed. "
    "Produce a final decision JSON instead of calling action tools."
)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _row_to_dict(row: pd.Series) -> dict:
    return {str(k): _json_safe(v) for k, v in row.items()}


@lru_cache(maxsize=1)
def _payments_frame() -> pd.DataFrame:
    if not PAYMENTS_CSV.exists():
        raise FileNotFoundError(f"payments dataset not found: {PAYMENTS_CSV}")
    df = pd.read_csv(PAYMENTS_CSV)
    return df


def _get_payment_row(payment_id: str) -> dict:
    df = _payments_frame()
    matches = df[df["payment_id"] == payment_id]
    if matches.empty:
        raise KeyError(f"Unknown payment_id: {payment_id}")
    return _row_to_dict(matches.iloc[0])


def get_payment(payment_id: str) -> dict:
    """Return the payment event row as a JSON-serializable dict."""
    try:
        row = _get_payment_row(payment_id)
    except KeyError as exc:
        return {"error": str(exc), "payment_id": payment_id}
    return {"payment_id": payment_id, "payment": row}


def get_customer_history(customer_id: str) -> dict:
    """Return prior failed-payment events for this customer."""
    df = _payments_frame()
    matches = df[df["customer_id"] == customer_id]
    events = [_row_to_dict(row) for _, row in matches.iterrows()]
    return {
        "customer_id": customer_id,
        "event_count": len(events),
        "events": events,
    }


def get_failure_details(payment_id: str) -> dict:
    """Return failure-focused fields for a payment."""
    try:
        row = _get_payment_row(payment_id)
    except KeyError as exc:
        return {"error": str(exc), "payment_id": payment_id}
    keys = [
        "payment_id",
        "customer_id",
        "timestamp",
        "amount",
        "payment_method",
        "failure_reason",
        "previous_successes",
        "previous_failures",
        "customer_value",
        "subscription_age",
        "days_since_last_payment",
    ]
    return {"payment_id": payment_id, "failure": {k: row.get(k) for k in keys}}


def calculate_recovery_probability(payment_id: str, intervention: str) -> dict:
    """
    Recovery probability for one intervention, via calculate_expected_value().

    retry uses the Phase 2 ML model; payment_link/notification/escalate use
    the simulator probability field; stop is 0.0. This wrapper does not
    re-derive those sources.
    """
    try:
        row = _get_payment_row(payment_id)
        result = calculate_expected_value(row, intervention)
    except (KeyError, ValueError) as exc:
        return {"error": str(exc), "payment_id": payment_id, "intervention": intervention}
    return {
        "payment_id": payment_id,
        "intervention": result["intervention"],
        "probability": float(result["probability_used"]),
    }


def calculate_expected_value_tool(payment_id: str, intervention: str) -> dict:
    """Call engine.expected_value.calculate_expected_value for one intervention."""
    try:
        row = _get_payment_row(payment_id)
        result = calculate_expected_value(row, intervention)
    except (KeyError, ValueError) as exc:
        return {"error": str(exc), "payment_id": payment_id, "intervention": intervention}
    return {
        "payment_id": payment_id,
        "intervention": result["intervention"],
        "probability_used": float(result["probability_used"]),
        "expected_value": float(result["expected_value"]),
        "cost": float(result["cost"]),
    }


def retry_payment(payment_id: str) -> dict:
    return {
        "executed": False,
        "action": "retry",
        "payment_id": payment_id,
        "message": _ACTION_NOT_EXECUTED,
    }


def generate_payment_link(payment_id: str) -> dict:
    return {
        "executed": False,
        "action": "payment_link",
        "payment_id": payment_id,
        "message": _ACTION_NOT_EXECUTED,
    }


def send_notification(payment_id: str) -> dict:
    return {
        "executed": False,
        "action": "notification",
        "payment_id": payment_id,
        "message": _ACTION_NOT_EXECUTED,
    }


def escalate_to_human(payment_id: str) -> dict:
    return {
        "executed": False,
        "action": "escalate",
        "payment_id": payment_id,
        "message": _ACTION_NOT_EXECUTED,
    }


def stop_recovery(payment_id: str) -> dict:
    return {
        "executed": False,
        "action": "stop",
        "payment_id": payment_id,
        "message": _ACTION_NOT_EXECUTED,
    }


TOOL_FUNCTIONS = {
    "get_payment": get_payment,
    "get_customer_history": get_customer_history,
    "get_failure_details": get_failure_details,
    "calculate_recovery_probability": calculate_recovery_probability,
    "calculate_expected_value_tool": calculate_expected_value_tool,
    "retry_payment": retry_payment,
    "generate_payment_link": generate_payment_link,
    "send_notification": send_notification,
    "escalate_to_human": escalate_to_human,
    "stop_recovery": stop_recovery,
}

VALID_INTERVENTION_LIST = list(VALID_INTERVENTIONS)
