# RecoverIQ HANDOFF — dataset generator & payment simulator

This folder is **data + simulation only**. There is no ML, agent, API, or dashboard code here.

Use this file as the contract for the next teammate (models / agent / UI).

---

## What exists

| Path | Role |
|------|------|
| `data/generator.py` | Builds 15,000 synthetic **failed** payment events |
| `data/simulator.py` | `simulate_outcome(payment_row, intervention) -> dict` |
| `data/recoverability.py` | Ground-truth recoverability oracle (simulation only) |
| `data/sample_data/payments.csv` | Generated dataset (exact columns below) |
| `evaluation/baseline.py` | `naive_strategy`, `rule_based_strategy`, full-set eval |

Do **not** rename columns, CSV path, intervention strings, or `simulate_outcome`.

---

## Dataset

**File:** `data/sample_data/payments.csv`  
**Rows:** 15,000 (all failed payments — the recovery decision surface)  
**Seed:** `42` in `data/generator.py` (`N_EVENTS`, `N_CUSTOMERS`)

Regenerate:

```bash
pip install pandas numpy
python data/generator.py
```

### Columns (exact names, this order)

| Column | Type | Meaning |
|--------|------|---------|
| `payment_id` | string | `PAY-0000001` … unique per row |
| `customer_id` | string | `CUST-00001` … customers repeat |
| `timestamp` | datetime | Event time (≈ last 120 days ending 2026-08-26) |
| `amount` | float | Transaction amount (INR-like). **Right-skewed** |
| `payment_method` | string | `card` \| `upi` \| `netbanking` \| `wallet` |
| `failure_reason` | string | See enums below |
| `customer_age` | int | Years, 18–70 |
| `previous_successes` | int | Prior successful charges for that customer |
| `previous_failures` | int | Prior failed charges |
| `customer_value` | float | Customer LTV / value (right-skewed) |
| `subscription_age` | int | Tenure in **months** |
| `days_since_last_payment` | int | Gap since last (attempted) payment |

There is **no** `payment_recovered` column in the CSV. Labels come from the simulator.

### Enums

**`failure_reason`**

- `temporary_bank_failure` — more recoverable
- `network_error` — more recoverable
- `insufficient_funds` — medium
- `card_expired` — less recoverable
- `hard_decline` — less recoverable

**`payment_method`** (different baseline failure rates, encoded via mix + recovery delta)

- `card` — highest failure rate, worse recovery
- `netbanking`
- `wallet`
- `upi` — lowest failure rate, better recovery

High `customer_value` customers are assigned better methods (more UPI / netbanking).

### Embedded relationships (not random noise)

These are implemented in `data/generator.py` and scored in `data/recoverability.py`:

1. Transient reasons (`temporary_bank_failure`, `network_error`) recover more often.
2. `hard_decline` and `card_expired` recover less often.
3. Higher `previous_failures` **reduces** recoverability.
4. Higher `previous_successes` **increases** recoverability.
5. Higher `customer_value` **slightly** increases recoverability (and better methods).
6. `amount` is right-skewed; larger amounts are slightly harder to recover.
7. Method mix and reason mix are coupled (cards → more expiry / hard decline).

---

## Simulator API

```python
from simulator import simulate_outcome  # if sys.path includes data/
# or: from data.simulator import simulate_outcome  if run as a package

outcome = simulate_outcome(payment_row, intervention)
```

**`intervention`** (exact strings):

`retry` | `payment_link` | `notification` | `escalate` | `stop`

**Return dict**

| Key | Type | Notes |
|-----|------|--------|
| `payment_recovered` | bool | |
| `recovered_amount` | float | Full `amount` if recovered, else `0.0` |
| `recovery_time_hours` | float \| None | `None` if not recovered or `stop` |
| `intervention` | str | Echo of the input |
| `recovery_probability` | float | Oracle P used for the Bernoulli draw (debug / analysis) |

Outcomes are **deterministic** given `(payment_id, intervention)` (MD5-seeded RNG).

`stop` always returns `payment_recovered=False`, `recovered_amount=0.0`.

### Intervention effects (oracle)

- **retry** — strong on transient bank/network; weak on insufficient funds; very weak on expired/hard decline.
- **payment_link** — best on `card_expired` (new instrument); decent on insufficient funds.
- **notification** — best on `insufficient_funds`.
- **escalate** — modest lift everywhere; extra lift for high `customer_value`.
- **stop** — never recover.

Do **not** train production models by importing `recoverability.py` as a feature. Treat it as the environment. A reasonable ML setup is: for each row, simulate all five interventions (or the policy you care about) and learn `arg max` expected `recovered_amount`.

---

## Baselines

File: `evaluation/baseline.py`

```python
naive_strategy(payment_row)        # always "retry"
rule_based_strategy(payment_row)   # retry if temporary/network;
                                   # stop if hard_decline/card_expired;
                                   # notification otherwise
```

Evaluate:

```bash
python evaluation/baseline.py
```

Prints:

- total revenue recovered by naive strategy
- total revenue recovered by rule_based strategy
- recovery rate % for each

**Seeded run after `python data/generator.py` (seed=42):**

| Strategy | Revenue recovered | Recovery rate |
|----------|-------------------|---------------|
| `naive_strategy` | 6,030,389.05 | 42.43% |
| `rule_based_strategy` | 7,070,455.81 | 50.07% |

Rule-based should beat always-retry. An ML policy that also uses `payment_link` / `escalate` should be able to beat both.

---

## Suggested next work (do not implement here)

- Supervised model / policy over interventions using simulator rollouts
- Agent that calls `simulate_outcome` as the environment
- Dashboard over `payments.csv` + simulated outcomes

Do not add FastAPI, Streamlit, or UI in this data slice.

---

## Repro

```bash
python data/generator.py
python evaluation/baseline.py
```

Python 3.10+ with `pandas` and `numpy`.

## Phase 2: ML Model

### Prediction API

The public prediction function is:

```python
predict_recovery_probability(payment_row: RowLike) -> float
```

It returns `P(payment_recovered | features, intervention=retry)` as a float in `[0, 1]`.

`payment_row` may be a `dict`, `pandas.Series`, or namedtuple containing the `payments.csv` columns. It must contain the required model feature columns; extra keys are ignored. The model internally converts the row to a one-row DataFrame and calls `predict_proba(...)[0, 1]`.

### Saved model

`models/recovery_model.pkl` is a **joblib payload dictionary**, not a raw scikit-learn estimator.

The saved dictionary has this structure:

```python
{
    "pipeline": <sklearn Pipeline>,
    "feature_columns": [...],
    "categorical_features": [...],
    "numeric_features": [...],
    "target": <target column name>,
    "label_intervention": "retry",
    "metadata": {
        "chosen_name": <chosen model name>,
        "test_auc": {
            "logistic_regression": <AUC>,
            "random_forest": <AUC>,
        },
    },
}
```

To load it directly:

```python
import joblib

payload = joblib.load("models/recovery_model.pkl")
pipeline = payload["pipeline"]
```

The project loader `load_recovery_model()` already unwraps this dictionary and returns the stored sklearn `Pipeline`.

### Test-set metrics

Evaluation uses a 20% test split with `random_state=42` and stratification.

**Chosen model — Logistic Regression**

| Metric | Score |
|---|---:|
| AUC-ROC | 0.8932 |
| Accuracy | 0.8313 |
| Precision | 0.7611 |
| Recall | 0.8782 |

**Alternative — Random Forest**

| Metric | Score |
|---|---:|
| AUC-ROC | 0.8910 |

Logistic regression is preferred because its AUC is slightly higher and it is simpler and more interpretable.

### Training / oracle boundary

The Phase 2 training data is specifically a **retry-policy dataset**. Labels are generated with:

```python
simulate_outcome(row, intervention="retry")
```

`data/recoverability.py` (the oracle) was **not imported or used for training**. Training labels come only from the public simulator's `simulate_outcome()` function.

This distinction is important: the resulting probability is specifically for the **retry intervention**, not an intervention-agnostic probability of recovery.

### Calibration

Calibration is good: across predicted-probability deciles, the predicted probabilities closely match the observed/actual recovery rates. The evaluation code explicitly compares mean predicted probability with actual recovery rate for each decile.

### Phase 3 caveats — Expected Value calculator

Phase 3 must not treat `predict_recovery_probability()` as a probability that applies equally to every possible intervention.

The current model estimates recovery probability for **`retry` only**. The simulator supports `retry`, `payment_link`, `notification`, `escalate`, and `stop`, with different intervention effects.

Therefore, an Expected Value calculator should:

- Treat the current ML probability as `P(recovery | features, intervention=retry)`.
- Not reuse the retry probability unchanged for `payment_link`, `notification`, or `escalate`.
- Treat `stop` as no recovery according to the simulator.
- Incorporate the payment `amount` and intervention-specific outcomes when calculating expected recovered value.
- Keep the model probability and the simulator/oracle's intervention-specific probabilities conceptually separate.
- If Phase 3 needs to compare interventions, a future model/policy should provide intervention-specific estimates (or otherwise obtain reliable intervention-specific recovery probabilities) before using `argmax` expected value across actions.

