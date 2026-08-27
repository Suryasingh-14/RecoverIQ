"""
RecoverIQ Phase 3 -- Expected Value calculator.

Public API:
    calculate_expected_value(payment_row, intervention, retry_cost=5,
                              notification_cost=2, payment_link_cost=3,
                              escalate_cost=50, incentive_pct=0.0) -> dict

Probability source per intervention (see HANDOFF.md "Phase 3 caveats"):
    retry          -> models/recovery_model.py: predict_recovery_probability(row)
                      (this is the ONLY model-backed probability; it is trained
                      and calibrated specifically for intervention="retry")
    payment_link,
    notification,
    escalate       -> data/simulator.py: simulate_outcome(row, intervention)
                      ["recovery_probability"] -- the simulator's own oracle
                      probability for that action. simulate_outcome is called
                      exactly ONCE per intervention and only its probability
                      field is used; the boolean payment_recovered outcome is
                      never consulted here, so no simulated randomness leaks
                      into the decision.
    stop           -> 0.0 by definition; simulate_outcome is not called.

The retry model's probability must never be reused unchanged for the other
interventions -- they have materially different effectiveness profiles per
HANDOFF.md, and doing so would silently misrepresent the EV comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from simulator import simulate_outcome  # noqa: E402
from recovery_model import predict_recovery_probability  # noqa: E402

RowLike = Union[Mapping[str, Any], Any]

VALID_INTERVENTIONS = ["retry", "payment_link", "notification", "escalate", "stop"]

_SIMULATOR_PRICED_INTERVENTIONS = {"payment_link", "notification", "escalate"}


def _get(row: RowLike, key: str) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(row, key)


def _intervention_cost(
    intervention: str,
    retry_cost: float,
    notification_cost: float,
    payment_link_cost: float,
    escalate_cost: float,
) -> float:
    return {
        "retry": retry_cost,
        "notification": notification_cost,
        "payment_link": payment_link_cost,
        "escalate": escalate_cost,
        "stop": 0.0,
    }[intervention]


def calculate_expected_value(
    payment_row: RowLike,
    intervention: str,
    retry_cost: float = 5,
    notification_cost: float = 2,
    payment_link_cost: float = 3,
    escalate_cost: float = 50,
    incentive_pct: float = 0.0,
) -> dict:
    """
    Expected value of taking `intervention` on `payment_row`.

    expected_value = (P(recovery) * amount * (1 - incentive_pct)) - intervention_cost

    Returns
    -------
    dict: {intervention, probability_used, expected_value, cost}
    """
    if intervention not in VALID_INTERVENTIONS:
        raise ValueError(
            f"Unknown intervention {intervention!r}. Expected one of {VALID_INTERVENTIONS}."
        )

    amount = float(_get(payment_row, "amount"))
    cost = _intervention_cost(
        intervention, retry_cost, notification_cost, payment_link_cost, escalate_cost
    )

    if intervention == "retry":
        probability_used = predict_recovery_probability(payment_row)
    elif intervention == "stop":
        probability_used = 0.0
    elif intervention in _SIMULATOR_PRICED_INTERVENTIONS:
        # Single call; only the probability field is used for the decision --
        # never the sampled boolean outcome (that would leak simulator
        # randomness into the decision engine).
        outcome = simulate_outcome(payment_row, intervention)
        probability_used = outcome["recovery_probability"]
    else:  # pragma: no cover - guarded by VALID_INTERVENTIONS check above
        raise ValueError(f"Unhandled intervention {intervention!r}")

    expected_value = (probability_used * amount * (1 - incentive_pct)) - cost

    return {
        "intervention": intervention,
        "probability_used": float(probability_used),
        "expected_value": float(expected_value),
        "cost": float(cost),
    }