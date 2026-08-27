"""
RecoverIQ Phase 5 -- Guardrail configuration.

All guardrail thresholds live here so they aren't magic numbers buried in
`policy_engine.py`. `check_guardrails()` merges this with any per-call
overrides, so a demo can tweak a single number here (or pass an override
dict) without touching rule logic.
"""

from __future__ import annotations

DEFAULT_GUARDRAIL_CONFIG: dict = {
    # Rule 1: High amount -> forced to escalate for human review.
    "high_amount_threshold": 50000,
    "high_amount_action": "escalate",
    # Rule 2: Repeated hard decline -> forced to stop.
    "hard_decline_repeat_failures": 3,
    "hard_decline_repeat_action": "stop",
    # Rule 3: Maximum retries for this specific payment.
    "max_retry_attempts": 2,
    "max_retry_action": "escalate",
    # Rule 4: Maximum incentive percentage allowed on any action.
    "max_incentive_pct": 0.05,
    # Rule 5: Fallback used when the proposed_decision isn't a valid
    # intervention at all.
    "invalid_action_fallback": "escalate",
}


def get_default_config() -> dict:
    """Return a fresh copy of the default guardrail config (safe to mutate)."""
    return dict(DEFAULT_GUARDRAIL_CONFIG)