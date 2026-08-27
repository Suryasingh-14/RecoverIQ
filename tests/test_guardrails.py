"""
Tests for guardrails/policy_engine.py

Run with:
    pytest tests/test_guardrails.py -v
"""

from guardrails.policy_engine import check_guardrails


def make_row(**overrides):
    row = {
        "payment_id": "PAY-TEST",
        "amount": 1000.0,
        "failure_reason": "temporary_bank_failure",
        "previous_failures": 0,
    }
    row.update(overrides)
    return row


def test_normal_payment_passes_through_unchanged():
    row = make_row(amount=1000.0, failure_reason="temporary_bank_failure", previous_failures=0)
    result = check_guardrails(row, "retry")
    assert result["allowed"] is True
    assert result["final_decision"] == "retry"
    assert result["violated_rules"] == []


def test_high_amount_forces_escalate():
    row = make_row(amount=75000.0, failure_reason="temporary_bank_failure", previous_failures=0)
    result = check_guardrails(row, "retry")
    assert result["allowed"] is False
    assert result["final_decision"] == "escalate"
    assert any("high_amount" in r for r in result["violated_rules"])


def test_hard_decline_repeat_forces_stop():
    row = make_row(amount=1000.0, failure_reason="hard_decline", previous_failures=3)
    result = check_guardrails(row, "retry")
    assert result["allowed"] is False
    assert result["final_decision"] == "stop"
    assert any("hard_decline_repeat" in r for r in result["violated_rules"])


def test_stop_wins_over_escalate_when_both_fire():
    # Both high amount (-> escalate) AND repeated hard decline (-> stop) fire.
    # Per spec, "stop" wins because it's the more conservative outcome.
    row = make_row(amount=75000.0, failure_reason="hard_decline", previous_failures=4)
    result = check_guardrails(row, "retry")
    assert result["allowed"] is False
    assert result["final_decision"] == "stop"
    assert len(result["violated_rules"]) == 2


def test_max_retries_blocks_retry():
    row = make_row(amount=1000.0, failure_reason="temporary_bank_failure", previous_failures=0)
    result = check_guardrails(row, "retry", retry_attempts_so_far=2)
    assert result["allowed"] is False
    assert result["final_decision"] != "retry"
    assert any("max_retries" in r for r in result["violated_rules"])


def test_max_retries_does_not_affect_other_proposed_actions():
    row = make_row(amount=1000.0, failure_reason="insufficient_funds", previous_failures=0)
    result = check_guardrails(row, "notification", retry_attempts_so_far=5)
    assert result["allowed"] is True
    assert result["final_decision"] == "notification"


def test_incentive_over_cap_is_flagged():
    row = make_row(amount=1000.0, failure_reason="temporary_bank_failure", previous_failures=0)
    result = check_guardrails(row, "payment_link", incentive_pct=0.10)
    assert result["allowed"] is False
    assert any("max_incentive" in r for r in result["violated_rules"])
    # incentive rule doesn't change which intervention runs
    assert result["final_decision"] == "payment_link"


def test_incentive_at_or_below_cap_is_fine():
    row = make_row(amount=1000.0, failure_reason="temporary_bank_failure", previous_failures=0)
    result = check_guardrails(row, "payment_link", incentive_pct=0.05)
    assert result["allowed"] is True
    assert result["final_decision"] == "payment_link"


def test_invalid_proposed_decision_is_rejected():
    row = make_row(amount=1000.0, failure_reason="temporary_bank_failure", previous_failures=0)
    result = check_guardrails(row, "refund_everything")
    assert result["allowed"] is False
    assert result["final_decision"] in [
        "retry", "payment_link", "notification", "escalate", "stop",
    ]
    assert any("unauthorized_action" in r for r in result["violated_rules"])


def test_amount_exactly_at_threshold_does_not_trigger():
    row = make_row(amount=50000.0, failure_reason="temporary_bank_failure", previous_failures=0)
    result = check_guardrails(row, "retry")
    assert result["allowed"] is True
    assert result["final_decision"] == "retry"


def test_previous_failures_exactly_at_threshold_triggers():
    row = make_row(amount=1000.0, failure_reason="hard_decline", previous_failures=3)
    result = check_guardrails(row, "retry")
    assert result["allowed"] is False
    assert result["final_decision"] == "stop"


def test_guardrail_config_used_reflects_overrides():
    row = make_row(amount=1000.0, failure_reason="temporary_bank_failure", previous_failures=0)
    result = check_guardrails(row, "retry", guardrail_config={"high_amount_threshold": 500})
    assert result["guardrail_config_used"]["high_amount_threshold"] == 500
    # 1000 > overridden threshold of 500 -> should now trigger
    assert result["allowed"] is False
    assert result["final_decision"] == "escalate"