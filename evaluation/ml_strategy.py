"""
RecoverIQ Phase 3 -- evaluate the ML/EV decision engine ("ML strategy") over
the full payments.csv dataset and compare against the naive and rule-based
baselines from evaluation/baseline.py.

For each row: decide_best_action() picks an intervention using EV logic
(retry probabilities from the ML model, other-intervention probabilities
from the simulator's own oracle). We then call simulate_outcome(row, decision)
ONCE to realize the actual outcome. This is the one place in the codebase
where "peeking" at the simulated boolean outcome for the chosen action is
appropriate -- it's evaluation, not the decision itself, and it happens
strictly after the decision has already been made.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "engine"))

from simulator import simulate_outcome  # noqa: E402
from decision_engine import decide_best_action  # noqa: E402

PAYMENTS_PATH = PROJECT_ROOT / "data" / "sample_data" / "payments.csv"

# Baselines from HANDOFF.md (seed=42 run of evaluation/baseline.py).
BASELINE_RESULTS = {
    "naive_strategy": {"revenue_recovered": 6_030_389.05, "recovery_rate": 0.4243},
    "rule_based_strategy": {"revenue_recovered": 7_070_455.81, "recovery_rate": 0.5007},
}


def evaluate_ml_strategy(df: pd.DataFrame) -> dict:
    total_recovered = 0.0
    n_recovered = 0
    decision_counts: dict[str, int] = {}

    for row in df.to_dict(orient="records"):
        decision = decide_best_action(row)["decision"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

        outcome = simulate_outcome(row, decision)
        if outcome["payment_recovered"]:
            total_recovered += outcome["recovered_amount"]
            n_recovered += 1

    return {
        "revenue_recovered": total_recovered,
        "recovery_rate": n_recovered / len(df),
        "decision_counts": decision_counts,
    }


def main() -> None:
    df = pd.read_csv(PAYMENTS_PATH)
    ml_result = evaluate_ml_strategy(df)

    print("ml_strategy (decide_best_action)")
    print(f"  revenue recovered: {ml_result['revenue_recovered']:,.2f}")
    print(f"  recovery rate:     {ml_result['recovery_rate'] * 100:.2f}%")
    print(f"  decision mix:      {ml_result['decision_counts']}")
    print()
    print("comparison")
    for name, res in BASELINE_RESULTS.items():
        print(f"  {name}: {res['revenue_recovered']:,.2f} ({res['recovery_rate'] * 100:.2f}%)")
    print(
        f"  ml_strategy:         {ml_result['revenue_recovered']:,.2f} "
        f"({ml_result['recovery_rate'] * 100:.2f}%)"
    )

    beats_naive = ml_result["revenue_recovered"] > BASELINE_RESULTS["naive_strategy"]["revenue_recovered"]
    beats_rule_based = (
        ml_result["revenue_recovered"] > BASELINE_RESULTS["rule_based_strategy"]["revenue_recovered"]
    )
    print()
    print(f"  beats naive_strategy:      {beats_naive}")
    print(f"  beats rule_based_strategy: {beats_rule_based}")


if __name__ == "__main__":
    main()