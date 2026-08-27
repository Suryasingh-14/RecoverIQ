"""
RecoverIQ Phase 3 -- Decision engine.

Public API:
    decide_best_action(payment_row, guardrail_config=None,
                        retry_attempts_so_far=0, incentive_pct=0.0) -> dict

Evaluates calculate_expected_value() for all five interventions, then asks
the standalone guardrail/policy engine (`guardrails.policy_engine.
check_guardrails`) whether the highest-EV action is actually allowed to
run. All five per-intervention EV dicts are returned for audit/transparency
regardless of which action was ultimately chosen.

Phase 5 note: this module used to contain its own hardcoded
`_apply_guardrails` rules. Those rules have moved to
`guardrails/policy_engine.py` (plus two new ones), and this file now just
calls into that engine -- it has no guardrail logic of its own anymore.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from engine.expected_value import VALID_INTERVENTIONS, calculate_expected_value
from guardrails.policy_engine import check_guardrails

RowLike = Union[Mapping[str, Any], Any]


def _get(row: RowLike, key: str) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(row, key)


def decide_best_action(
    payment_row: RowLike,
    guardrail_config: Optional[dict] = None,
    retry_attempts_so_far: int = 0,
    incentive_pct: float = 0.0,
) -> dict:
    """
    Choose the best intervention for a single failed payment.

    The EV argmax proposes an action; `guardrails.policy_engine.
    check_guardrails` has final say on whether that proposal is allowed to
    execute, and can force a different, safer action.

    Parameters
    ----------
    payment_row : RowLike
    guardrail_config : dict, optional
        Overrides passed straight through to `check_guardrails`.
    retry_attempts_so_far : int, default 0
        Passed through to `check_guardrails` (Rule 3 -- max retries).
    incentive_pct : float, default 0.0
        Passed through to `check_guardrails` (Rule 4 -- max incentive) and
        also used in the EV calculation itself.

    Returns
    -------
    dict: {payment_id, decision, expected_value, reason, all_evaluated}
        all_evaluated is the list of all five calculate_expected_value()
        dicts, in VALID_INTERVENTIONS order, for audit purposes.
    """
    all_evaluated = [
        calculate_expected_value(payment_row, intervention, incentive_pct=incentive_pct)
        for intervention in VALID_INTERVENTIONS
    ]
    ev_by_intervention = {ev["intervention"]: ev for ev in all_evaluated}

    best = max(all_evaluated, key=lambda ev: ev["expected_value"])
    proposed_decision = best["intervention"]

    guardrail_result = check_guardrails(
        payment_row,
        proposed_decision,
        guardrail_config=guardrail_config,
        retry_attempts_so_far=retry_attempts_so_far,
        incentive_pct=incentive_pct,
    )

    decision = guardrail_result["final_decision"]

    if guardrail_result["allowed"]:
        reason = (
            f"highest expected value ({best['expected_value']:.2f}) "
            f"among all evaluated interventions"
        )
    else:
        reason = (
            f"proposed '{proposed_decision}' (EV {best['expected_value']:.2f}) "
            f"overridden by guardrails: {'; '.join(guardrail_result['violated_rules'])}"
        )

    return {
        "payment_id": _get(payment_row, "payment_id"),
        "decision": decision,
        "expected_value": ev_by_intervention[decision]["expected_value"],
        "reason": reason,
        "all_evaluated": all_evaluated,
    }