"""
Compare the Phase 4 agent against Phase 3 decide_best_action on mixed cases.

Run from repo root:
    python agent/test_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "engine"))

from agent.agent import run_agent_decision  # noqa: E402
from agent.tools import get_payment  # noqa: E402
from decision_engine import decide_best_action  # noqa: E402

# Mix of easy/hard cases from data/sample_data/payments.csv
SAMPLE_PAYMENT_IDS = [
    "PAY-0000001",  # temporary_bank_failure — often retry
    "PAY-0000003",  # insufficient_funds — often notification
    "PAY-0000005",  # hard_decline + previous_failures>=3 — Phase 3 guardrail stop
    "PAY-0000013",  # card_expired — often payment_link
    "PAY-0000016",  # larger-amount transient failure
]


def _phase3_decision(payment_id: str) -> dict:
    payload = get_payment(payment_id)
    if "error" in payload:
        raise KeyError(payload["error"])
    return decide_best_action(payload["payment"])


def main() -> None:
    print("RecoverIQ Phase 4 agent vs Phase 3 decide_best_action\n")
    for payment_id in SAMPLE_PAYMENT_IDS:
        print("=" * 72)
        print(f"payment_id: {payment_id}")
        phase3 = _phase3_decision(payment_id)
        print("Phase 3 decide_best_action():")
        print(f"  decision       = {phase3['decision']}")
        print(f"  expected_value = {phase3['expected_value']:.4f}")
        print(f"  reason         = {phase3['reason']}")

        try:
            agent = run_agent_decision(payment_id)
        except Exception as exc:  # noqa: BLE001 — keep remaining cases running
            print("Phase 4 run_agent_decision(): FAILED")
            print(f"  error          = {type(exc).__name__}: {exc}")
            print()
            continue

        print("Phase 4 run_agent_decision():")
        print(f"  decision       = {agent['decision']}")
        print(f"  delay_hours    = {agent['delay_hours']}")
        print(f"  expected_value = {agent['expected_value']}")
        print(f"  reason         = {agent['reason']}")
        match = "MATCH" if agent["decision"] == phase3["decision"] else "DIFFER"
        print(f"  comparison     = {match}")
        print()
        print(json.dumps(agent, indent=2))
        print()


if __name__ == "__main__":
    main()
