"""Phase 6 experimentation/evaluation engine for RecoverIQ."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_data" / "payments.csv"

# Make direct execution (``python evaluation/experiments.py``) work from repo root.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "engine"))

from simulator import simulate_outcome  # noqa: E402
from engine.decision_engine import decide_best_action  # noqa: E402
from evaluation.baseline import naive_strategy, rule_based_strategy  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    calculate_recovery_rate,
    calculate_total_revenue,
    summarize_decision_mix,
)
from engine.expected_value import calculate_expected_value, VALID_INTERVENTIONS  # noqa: E402


def _evaluate_rows(df: pd.DataFrame, strategy: Callable[[dict], str]) -> dict:
    records = []
    for row in df.to_dict(orient="records"):
        decision = strategy(row)
        outcome = simulate_outcome(row, decision)
        records.append({**row, "decision": decision, **outcome})
    out = pd.DataFrame(records)
    return {
        "n": len(out),
        "n_recovered": int(out["payment_recovered"].astype(bool).sum()) if len(out) else 0,
        "recovery_rate_pct": calculate_recovery_rate(out),
        "total_revenue_recovered": calculate_total_revenue(out),
        "decision_mix": summarize_decision_mix(out),
        "outcomes": out,
    }


def _ml_only_strategy(row: dict) -> str:
    """Return the EV/ML proposal before guardrails are applied.

    ``evaluation/ml_strategy.py`` currently calls ``decide_best_action``, which
    includes Phase 5 guardrails. For the requested progression table, ML-only is
    therefore defined as the same EV argmax *before* ``check_guardrails``;
    RecoverIQ is the guarded ``decide_best_action`` result.
    """
    decision_result = decide_best_action(row)
    return max(
        decision_result["all_evaluated"],
        key=lambda item: item["expected_value"],
    )["intervention"]


def _recoveriq_strategy(row: dict) -> str:
    return decide_best_action(row)["decision"]


def _strip_internal(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "outcomes"}


def run_control_experiment(
    df=None,
    control_strategy="rule_based",
    treatment_strategy="ml_strategy",
    split_ratio=0.5,
    random_state=42,
) -> dict:
    """Run a reproducible randomized control-vs-RecoverIQ experiment.

    The control uses the existing rule-based business process. The treatment
    uses ``decide_best_action`` and then realizes exactly one simulator outcome
    for the selected decision. Revenue is scaled to the same event count before
    computing incremental revenue.
    """
    if df is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"{DATA_PATH} not found")
        df = pd.read_csv(DATA_PATH)
    else:
        df = df.copy()

    if not 0 < split_ratio < 1:
        raise ValueError("split_ratio must be strictly between 0 and 1")
    if len(df) < 2:
        raise ValueError("df must contain at least 2 rows")

    controls = {"rule_based": rule_based_strategy, "rule_based_strategy": rule_based_strategy}
    treatments = {"ml_strategy": _recoveriq_strategy, "recoveriq": _recoveriq_strategy}
    if control_strategy not in controls:
        raise ValueError(f"Unknown control_strategy {control_strategy!r}")
    if treatment_strategy not in treatments:
        raise ValueError(f"Unknown treatment_strategy {treatment_strategy!r}")

    shuffled = df.sample(frac=1.0, random_state=random_state)
    control_n = int(len(df) * split_ratio)
    control_df = shuffled.iloc[:control_n].copy()
    treatment_df = shuffled.iloc[control_n:].copy()

    control = _evaluate_rows(control_df, controls[control_strategy])
    treatment = _evaluate_rows(treatment_df, treatments[treatment_strategy])

    # Normalize both groups to the treatment/control common comparison size.
    common_n = min(control["n"], treatment["n"])
    control_scaled = control["total_revenue_recovered"] / control["n"] * common_n
    treatment_scaled = treatment["total_revenue_recovered"] / treatment["n"] * common_n
    incremental_revenue = treatment_scaled - control_scaled
    incremental_rate_pp = treatment["recovery_rate_pct"] - control["recovery_rate_pct"]

    return {
        "control": _strip_internal(control),
        "treatment": _strip_internal(treatment),
        "control_sample_size": control["n"],
        "treatment_sample_size": treatment["n"],
        "common_comparison_size": common_n,
        "control_revenue_scaled": float(control_scaled),
        "treatment_revenue_scaled": float(treatment_scaled),
        "incremental_revenue": float(incremental_revenue),
        "incremental_rate_pp": float(incremental_rate_pp),
        "split_ratio": float(split_ratio),
        "random_state": random_state,
    }


def run_multiple_splits(n_seeds=10):
    """Run the control experiment across multiple random seeds.

    Seeds are the integers ``0`` through ``n_seeds - 1``.  Each split uses
    the same control/treatment logic and split ratio as ``run_control_experiment``.
    The returned summary includes every run plus the mean incremental revenue
    (and mean incremental recovery-rate lift) across all seeds.
    """
    if not isinstance(n_seeds, int) or isinstance(n_seeds, bool) or n_seeds < 1:
        raise ValueError("n_seeds must be a positive integer")

    results = [run_control_experiment(random_state=seed) for seed in range(n_seeds)]
    incremental_revenues = [r["incremental_revenue"] for r in results]
    incremental_rates = [r["incremental_rate_pp"] for r in results]

    return {
        "n_seeds": n_seeds,
        "seeds": list(range(n_seeds)),
        "runs": results,
        "average_incremental_revenue": float(sum(incremental_revenues) / n_seeds),
        "average_incremental_rate_pp": float(sum(incremental_rates) / n_seeds),
    }


def _progression(df: pd.DataFrame) -> dict[str, dict]:
    """Evaluate all four strategies on the same full dataset."""
    strategies = {
        "naive_strategy": naive_strategy,
        "rule_based_strategy": rule_based_strategy,
        "ml_strategy": _ml_only_strategy,
        "RecoverIQ": _recoveriq_strategy,
    }
    return {name: _strip_internal(_evaluate_rows(df, fn)) for name, fn in strategies.items()}


def _print_experiment(result: dict) -> None:
    c, t = result["control"], result["treatment"]
    print("=" * 78)
    print("Phase 6 — Control Experiment")
    print("=" * 78)
    print(f"Control sample:              {result['control_sample_size']:,}")
    print(f"Treatment sample:            {result['treatment_sample_size']:,}")
    print(f"Control recovered:           ₹{c['total_revenue_recovered']:,.2f}")
    print(f"Treatment (RecoverIQ) recovered: ₹{t['total_revenue_recovered']:,.2f}")
    print(f"Incremental revenue (scaled): ₹{result['incremental_revenue']:,.2f}")
    print(f"Incremental rate:             {result['incremental_rate_pp']:+.2f} percentage points")
    print()


def _print_progression(results: dict[str, dict]) -> None:
    print("Four-strategy progression (same full dataset)")
    print("-" * 78)
    print(f"{'Strategy':<22} {'Recovered revenue':>20} {'Recovery rate':>16} {'Recovered events':>17}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{name:<22} ₹{r['total_revenue_recovered']:>18,.2f} {r['recovery_rate_pct']:>14.2f}% {r['n_recovered']:>17,}")
    print("-" * 78)


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    experiment = run_control_experiment(data, random_state=42)
    _print_experiment(experiment)
    progression = _progression(data)
    _print_progression(progression)

    multi = run_multiple_splits(n_seeds=10)
    print()
    print("10-seed control experiment")
    print("-" * 78)
    print(f"Average incremental revenue: ₹{multi['average_incremental_revenue']:,.2f}")
    print(f"Average incremental rate:    {multi['average_incremental_rate_pp']:+.2f} percentage points")
