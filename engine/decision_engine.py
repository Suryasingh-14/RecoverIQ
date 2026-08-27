"""
RecoverIQ Phase 3 -- Decision engine.

Public API:
    decide_best_action(payment_row, guardrail_config=None) -> dict

Evaluates calculate_expected_value() for all five interventions, applies a
small set of hardcoded business guardrails, and picks the highest-EV action
(unless a guardrail forces a different one). All five per-intervention EV
dicts are returned for audit/transparency.

Phase 5 will replace the two hardcoded rules below with a real guardrail
engine -- keep them isolated in `_apply_guardrails` so that swap is a
drop-in replacement of this one function.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expected_value import VALID_INTERVENTIONS, calculate_expected_value  # noqa: E402

RowLike = Union[Mapping[str, Any], Any]

DEFAULT_GUARDRAIL_CONFIG = {
    "high_amount_threshold": 50000,
    "high_amount_action": "escalate",
    "hard_decline_repeat_failures": 3,
    "hard_decline_repeat_action": "stop",
}


def _get(row: RowLike, key: str) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(row, key)


def _apply_guardrails(payment_row: RowLike, config: dict) -> Optional[tuple[str, str]]:
    """
    Hardcoded business rules, evaluated before EV comparison.

    Returns (forced_intervention, reason) if a rule fires, else None.
    """
    amount = float(_get(payment_row, "amount"))
    if amount > config["high_amount_threshold"]:
        return (
            config["high_amount_action"],
            f"guardrail: amount {amount:,.2f} > {config['high_amount_threshold']:,} "
            "-- forced to escalate for human review",
        )

    failure_reason = _get(payment_row, "failure_reason")
    previous_failures = float(_get(payment_row, "previous_failures"))
    if (
        failure_reason == "hard_decline"
        and previous_failures >= config["hard_decline_repeat_failures"]
    ):
        return (
            config["hard_decline_repeat_action"],
            f"guardrail: hard_decline with previous_failures={previous_failures:.0f} "
            f">= {config['hard_decline_repeat_failures']} -- forced to stop",
        )

    return None


def decide_best_action(payment_row: RowLike, guardrail_config: Optional[dict] = None) -> dict:
    """
    Choose the best intervention for a single failed payment.

    Returns
    -------
    dict: {payment_id, decision, expected_value, reason, all_evaluated}
        all_evaluated is the list of all five calculate_expected_value() dicts,
        in VALID_INTERVENTIONS order, for audit purposes.
    """
    config = {**DEFAULT_GUARDRAIL_CONFIG, **(guardrail_config or {})}

    all_evaluated = [
        calculate_expected_value(payment_row, intervention)
        for intervention in VALID_INTERVENTIONS
    ]
    ev_by_intervention = {ev["intervention"]: ev for ev in all_evaluated}

    forced = _apply_guardrails(payment_row, config)

    if forced is not None:
        decision, reason = forced
    else:
        best = max(all_evaluated, key=lambda ev: ev["expected_value"])
        decision = best["intervention"]
        reason = (
            f"highest expected value ({best['expected_value']:.2f}) "
            f"among all evaluated interventions"
        )

    return {
        "payment_id": _get(payment_row, "payment_id"),
        "decision": decision,
        "expected_value": ev_by_intervention[decision]["expected_value"],
        "reason": reason,
        "all_evaluated": all_evaluated,
    }