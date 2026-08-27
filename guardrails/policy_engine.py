"""
RecoverIQ Phase 5 -- Guardrail / policy engine.

Public API:
    check_guardrails(payment_row, proposed_decision, guardrail_config=None,
                      retry_attempts_so_far=0, incentive_pct=0.0) -> dict

This is the trust/safety layer for RecoverIQ. Upstream code (the EV
argmax in `engine.decision_engine`, or the Phase 4 AI agent) PROPOSES an
intervention; this module has final say on whether that intervention is
ALLOWED to execute. Nothing outside this file should decide that a
guardrail did or didn't fire -- callers just pass in a proposal and act on
`final_decision`.

Rules
-----
1. High amount            -- amount > high_amount_threshold           -> only "escalate" allowed
2. Repeated hard decline   -- hard_decline & previous_failures >= N    -> only "stop" allowed
3. Max retries             -- proposed "retry" & retry_attempts_so_far
                               >= max_retry_attempts                  -> "retry" blocked
4. Max incentive           -- incentive_pct > max_incentive_pct       -> incentive must be capped
5. Unauthorized action     -- proposed_decision not one of the 5      -> rejected outright
                               valid interventions

Priority order (most conservative wins) when multiple rules fire
------------------------------------------------------------------
Rule 2 (stop) > Rule 1 (escalate) > Rule 5 (invalid action) > Rule 3 (retry
cap) > Rule 4 (incentive cap).

Rationale: "stop" means no action at all, which is always the safest
outcome, so a confirmed repeat hard-decline always wins even over a
high-amount escalate requirement (per spec: if both "must escalate" and
"must stop" fire, stop wins). An unauthorized/garbage proposal is resolved
before we consider whether *that specific* proposal was "retry" run too
many times, since rules 3 only makes sense once we know the proposal is a
real, valid intervention. Rule 4 (incentive cap) is evaluated last and,
unlike the others, does not change *which* intervention runs -- it only
flags that the incentive attached to the run must be capped -- so it never
overrides an intervention-level rule above it, but still marks the
proposal as not-allowed-as-is.

Note Rule 3 and Rule 5 can never both fire for the same call: Rule 3 only
applies when `proposed_decision == "retry"`, which by definition is a
valid intervention, so Rule 5 (invalid action) cannot also be true.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from engine.expected_value import VALID_INTERVENTIONS
from guardrails.config import DEFAULT_GUARDRAIL_CONFIG

RowLike = Union[Mapping[str, Any], Any]


def _get(row: RowLike, key: str) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(row, key)


def check_guardrails(
    payment_row: RowLike,
    proposed_decision: str,
    guardrail_config: Optional[dict] = None,
    retry_attempts_so_far: int = 0,
    incentive_pct: float = 0.0,
) -> dict:
    """
    Validate a proposed intervention against RecoverIQ's business guardrails.

    Parameters
    ----------
    payment_row : RowLike
        A single failed-payment row (dict / Series / namedtuple) with at
        least `amount`, `failure_reason`, `previous_failures`.
    proposed_decision : str
        The intervention some upstream logic (EV argmax or the AI agent)
        wants to execute.
    guardrail_config : dict, optional
        Overrides merged on top of `DEFAULT_GUARDRAIL_CONFIG`.
    retry_attempts_so_far : int, default 0
        How many times "retry" has already been attempted for this exact
        payment_id. There's no persistent action log yet (per HANDOFF), so
        the caller is responsible for tracking/passing this in.
    incentive_pct : float, default 0.0
        The incentive percentage attached to the proposed action.

    Returns
    -------
    dict: {
        "proposed_decision": str,
        "allowed": bool,               # True only if nothing fired
        "final_decision": str,         # proposed_decision if allowed,
                                        # else the forced/safe alternative
        "violated_rules": [str, ...],  # empty if allowed
        "guardrail_config_used": dict,
    }
    """
    config = {**DEFAULT_GUARDRAIL_CONFIG, **(guardrail_config or {})}

    violated_rules: list[str] = []
    # Each fired rule appends (priority, forced_intervention_or_None) --
    # forced_intervention is None for rules (like Rule 4) that flag a
    # violation without forcing a different intervention.
    forced_candidates: list[tuple[int, str]] = []

    is_valid_action = proposed_decision in VALID_INTERVENTIONS

    # --- Rule 1: high amount -> escalate -------------------------------
    amount = float(_get(payment_row, "amount"))
    if amount > config["high_amount_threshold"]:
        violated_rules.append(
            f"high_amount: amount {amount:,.2f} > "
            f"{config['high_amount_threshold']:,} -- only "
            f"'{config['high_amount_action']}' is allowed"
        )
        forced_candidates.append((2, config["high_amount_action"]))

    # --- Rule 2: repeated hard decline -> stop --------------------------
    failure_reason = _get(payment_row, "failure_reason")
    previous_failures = float(_get(payment_row, "previous_failures"))
    if (
        failure_reason == "hard_decline"
        and previous_failures >= config["hard_decline_repeat_failures"]
    ):
        violated_rules.append(
            f"hard_decline_repeat: hard_decline with "
            f"previous_failures={previous_failures:.0f} >= "
            f"{config['hard_decline_repeat_failures']} -- only "
            f"'{config['hard_decline_repeat_action']}' is allowed"
        )
        forced_candidates.append((1, config["hard_decline_repeat_action"]))

    # --- Rule 5: unauthorized / unknown action --------------------------
    if not is_valid_action:
        violated_rules.append(
            f"unauthorized_action: '{proposed_decision}' is not one of the "
            f"valid interventions {VALID_INTERVENTIONS} -- rejected"
        )
        forced_candidates.append((3, config["invalid_action_fallback"]))

    # --- Rule 3: max retries for this payment ---------------------------
    if (
        is_valid_action
        and proposed_decision == "retry"
        and retry_attempts_so_far >= config["max_retry_attempts"]
    ):
        violated_rules.append(
            f"max_retries: retry_attempts_so_far={retry_attempts_so_far} >= "
            f"{config['max_retry_attempts']} -- 'retry' is blocked for this "
            f"payment"
        )
        forced_candidates.append((4, config["max_retry_action"]))

    # --- Rule 4: max incentive ------------------------------------------
    # This rule never overrides *which* intervention runs -- it only says
    # the incentive attached to it is out of policy and must be capped.
    if incentive_pct > config["max_incentive_pct"]:
        violated_rules.append(
            f"max_incentive: incentive_pct={incentive_pct:.4f} > "
            f"{config['max_incentive_pct']:.4f} -- must be capped to "
            f"{config['max_incentive_pct']:.4f}"
        )

    if forced_candidates:
        # Lowest priority number wins (most conservative / strictest).
        _, final_decision = min(forced_candidates, key=lambda pair: pair[0])
    else:
        final_decision = proposed_decision

    allowed = len(violated_rules) == 0

    return {
        "proposed_decision": proposed_decision,
        "allowed": allowed,
        "final_decision": final_decision,
        "violated_rules": violated_rules,
        "guardrail_config_used": config,
    }