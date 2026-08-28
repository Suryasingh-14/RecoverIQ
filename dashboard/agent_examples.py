"""
Static AI Agent decision examples for the dashboard.
These are real outputs captured from run_agent_decision() during Phase 4
testing (Gemini tool-calling). Shown statically here since live LLM calls
are rate-limited and too slow for an interactive dashboard.
"""

AGENT_EXAMPLES = [
    {
        "payment_id": "PAY-0000001",
        "decision": "retry",
        "expected_value": 258.52,
        "reason": "Failure reason is temporary_bank_failure. Based on EV tool "
                  "calculation, retry yields the highest expected value ($258.52) "
                  "with a model-predicted recovery probability of 71.3%.",
        "phase3_match": True,
    },
    {
        "payment_id": "PAY-0000003",
        "decision": "notification",
        "expected_value": 129.44,
        "reason": "Notification yields the highest expected value ($129.44) among "
                  "all candidate actions for this $433.62 payment failing due to "
                  "insufficient_funds. Based on the EV calculations, notification "
                  "has a simulator-sourced recovery probability of 30.31% with a "
                  "$2.00 cost, outperforming payment_link ($101.24 EV), escalate "
                  "($59.46 EV), retry ($33.54 EV via ML model p=8.89%), and stop "
                  "($0 EV). Immediate notification allows the customer to top up "
                  "their account promptly.",
        "phase3_match": True,
    },
]