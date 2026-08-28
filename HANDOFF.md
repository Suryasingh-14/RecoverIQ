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

---

## Phase 3: Expected Value + Decision Engine

Adds `engine/expected_value.py` and `engine/decision_engine.py`, plus
`evaluation/ml_strategy.py`. Does not modify any Phase 1 or Phase 2 file.

### Expected Value API

```python
from engine.expected_value import calculate_expected_value

calculate_expected_value(
    payment_row,
    intervention,          # retry | payment_link | notification | escalate | stop
    retry_cost=5,
    notification_cost=2,
    payment_link_cost=3,
    escalate_cost=50,
    incentive_pct=0.0,
) -> dict   # {intervention, probability_used, expected_value, cost}
```

`expected_value = (P(recovery) * amount * (1 - incentive_pct)) - intervention_cost`

Probability source per intervention (see Phase 2's "Phase 3 caveats" above,
which this implementation follows directly):

| Intervention | `probability_used` comes from |
|---|---|
| `retry` | `models/recovery_model.py: predict_recovery_probability(row)` |
| `payment_link`, `notification`, `escalate` | `data/simulator.py: simulate_outcome(row, intervention)["recovery_probability"]`, called **once** per intervention |
| `stop` | `0.0` (simulator not called) |

For the three simulator-priced interventions, only the `recovery_probability`
field is read — the sampled boolean `payment_recovered` from that call is
discarded. This keeps the decision deterministic-in-expectation and avoids
leaking simulator randomness into the choice of action.

### Decision Engine API

```python
from engine.decision_engine import decide_best_action

decide_best_action(payment_row, guardrail_config=None) -> dict
# {payment_id, decision, expected_value, reason, all_evaluated}
```

`all_evaluated` is the list of all five `calculate_expected_value()` result
dicts (one per intervention), included for audit/transparency regardless of
which one was chosen.

### Hardcoded business rules (Phase 5 should formalize these)

Evaluated **before** the EV comparison, in this order; the first one that
matches wins and skips the `argmax` step entirely:

1. **High amount → escalate.** If `amount > 50000`, force `"escalate"`
   regardless of EV (routes to human review). Threshold and forced action are
   in `DEFAULT_GUARDRAIL_CONFIG` (`high_amount_threshold`, `high_amount_action`)
   and can be overridden via `guardrail_config`.
2. **Repeated hard decline → stop.** If `failure_reason == "hard_decline"` and
   `previous_failures >= 3`, force `"stop"` (don't keep spending on a customer
   who has already hard-declined repeatedly). Config keys:
   `hard_decline_repeat_failures`, `hard_decline_repeat_action`.

Both rules are isolated in `engine.decision_engine._apply_guardrails` so
Phase 5 can replace that one function with a real guardrail engine without
touching `calculate_expected_value` or the EV comparison logic.

### Evaluation — `evaluation/ml_strategy.py`

```bash
python evaluation/ml_strategy.py
```

For every row: `decide_best_action()` picks an intervention, then
`simulate_outcome(row, decision)` is called **once** to realize the actual
outcome for revenue/recovery-rate accounting. Peeking at the realized boolean
outcome is only done here, post-decision, for evaluation — never inside
`calculate_expected_value` or `decide_best_action` themselves.

**Evaluation results:** not yet populated. `data/generator.py`,
`data/recoverability.py`, and the trained `models/recovery_model.pkl` /
`models/training_data.csv` used for this handoff's Phase 1/2 numbers were not
available while building Phase 3, so `evaluation/ml_strategy.py` has not been
run against the real seed-42 dataset and model yet. Run it in the real
environment (after `python data/generator.py`, `python
models/train_data_builder.py`, `python models/recovery_model.py`) and record
the output here alongside the existing baselines:

| Strategy | Revenue recovered | Recovery rate |
|----------|-------------------|---------------|
| `naive_strategy` | 6,030,389.05 | 42.43% |
| `rule_based_strategy` | 7,070,455.81 | 50.07% |
| `ml_strategy` (Phase 3) | *(run `evaluation/ml_strategy.py` and fill in)* | *(fill in)* |

The engine code was verified independently against a locally reconstructed
stand-in dataset/oracle/model (not the repo's real Phase 1/2 artifacts): it
correctly picks `retry` on transient failures, forces `escalate` above the
amount threshold, forces `stop` on repeated hard declines, and its decision
mix spans all five interventions — so the logic is sound. Only the *numbers*
in the table above are pending a run against the real files.

### Caveats for Phase 4 (AI Agent)

- **Call these functions, don't reimplement EV logic.** The agent should call
  `decide_best_action()` (or `calculate_expected_value()` directly for
  single-intervention queries) rather than re-deriving expected value or
  re-reading probabilities itself. This keeps the retry/simulator probability
  split, the cost constants, and the guardrails in one place.
- **The `retry` probability is the only ML-model-backed number.** Every other
  intervention's probability comes from the simulator's oracle, which is
  meant to represent ground truth for evaluation/simulation purposes, not a
  learned estimate. If the agent needs to explain *why* an action was chosen,
  it should be clear with users/reviewers that `payment_link` /
  `notification` / `escalate` probabilities are simulator-sourced, not from a
  trained classifier.
- **Guardrails currently short-circuit EV entirely.** When a guardrail fires,
  `all_evaluated` still contains the true EV numbers for all five
  interventions (useful for the agent to explain "the model preferred X but
  policy required Y"), but `decision`/`expected_value` reflect the forced
  action, not the argmax.
- **`simulate_outcome` must not be called more than once per intervention per
  decision context** outside of evaluation code. The agent should treat
  probability-only inspection (as `calculate_expected_value` does) as the
  correct pattern if it ever needs to reason about a non-retry intervention
  directly; it should not call `simulate_outcome` speculatively to "see what
  would happen" before committing to an action, since that consumes a random
  draw against the deterministic per-`(payment_id, intervention)` seed.
- **Do not import `data/recoverability.py`.** Same rule as Phase 2 — the
  oracle stays an environment, never a feature or agent tool.

---

## Phase 4: AI Agent

Adds `agent/tools.py`, `agent/prompts.py`, `agent/agent.py`, and
`agent/test_agent.py`. Does not modify any `engine/` or `models/` file.
Does not reimplement expected value or `decide_best_action()` logic.

### Decision API

```python
from agent.agent import run_agent_decision

run_agent_decision(payment_id, api_key=None) -> dict
# {payment_id, decision, delay_hours, reason, expected_value}
```

- `decision` is one of `retry` | `payment_link` | `notification` | `escalate` | `stop`.
- `delay_hours` is a number or `null` (hours to wait before the chosen action).
- The agent **does not execute** the action; it only returns the JSON decision.
- Tool-calling loop is capped at **8** iterations.

### API key

Environment variable: **`GEMINI_API_KEY`**

Loaded from `.env` via `python-dotenv` (`load_dotenv()` in `agent/agent.py`).
If `api_key` is `None`, the key is read with `os.environ.get("GEMINI_API_KEY")`.
The key is never hardcoded. `.env` is gitignored. Placeholder file:
`.env.example` (`GEMINI_API_KEY=your-key-here`).

Model: **`gemini-3.6-flash`**. `gemini-2.5-flash` (originally specified for the
free tier) returns 404 for new API keys ("no longer available to new users");
the Google API recommends `gemini-3.6-flash` instead. Optional override:
`GEMINI_MODEL`.

```bash
pip install google-generativeai python-dotenv
python agent/test_agent.py
```

### Tools

Registered from `agent/tools.py` (JSON-serializable dicts; no pandas objects):

| Tool | Wraps / does |
|------|----------------|
| `get_payment(payment_id)` | Lookup row in `data/sample_data/payments.csv` |
| `get_customer_history(customer_id)` | All CSV events for that customer |
| `get_failure_details(payment_id)` | Failure-focused subset of the payment row |
| `calculate_recovery_probability(payment_id, intervention)` | `calculate_expected_value()` → `probability_used` |
| `calculate_expected_value_tool(payment_id, intervention)` | `engine.expected_value.calculate_expected_value` |
| `retry_payment` / `generate_payment_link` / `send_notification` / `escalate_to_human` / `stop_recovery` | Stubs that **do not execute** (`executed: false`) |

The agent is instructed to always call `get_payment`, `get_customer_history`,
and `get_failure_details` first, then `calculate_expected_value_tool` for each
candidate intervention, and never to guess probabilities or call action tools.

### Sample test output

`agent/test_agent.py` cases and **Phase 3** `decide_best_action()`:

| payment_id | failure | Phase 3 decision | Phase 3 EV |
|------------|---------|------------------|------------|
| `PAY-0000001` | `temporary_bank_failure` | `retry` | 258.5182 |
| `PAY-0000003` | `insufficient_funds` | `notification` | 129.4356 |
| `PAY-0000005` | `hard_decline` (prev_failures=4) | `stop` (guardrail) | 0.0 |
| `PAY-0000013` | `card_expired` | `payment_link` | 5.4788 |
| `PAY-0000016` | `temporary_bank_failure` (larger amount) | `retry` | 3636.6932 |

Live `run_agent_decision()` uses Gemini with the same `test_agent.py` harness.

Sample live run (`PAY-0000001`): agent chose `retry`, `expected_value`
258.5182, **MATCH** vs Phase 3. Remaining cases in the same minute can 429
on the free-tier 5 requests/minute cap; the agent retries once after the
API's `retry_delay`. Re-run `python agent/test_agent.py` with
`GEMINI_API_KEY` set.

### Caveats for Phase 5

- **Guardrails are still hardcoded in Phase 3 only.** The agent scores EV via
  tools and is not wired to `_apply_guardrails`. On cases like `PAY-0000005`
  it may pick a high-EV action while `decide_best_action()` forces `stop`.
  Phase 5 should either expose guardrails as a tool or apply them after the
  agent JSON, before any execution.
- **Do not execute recovery from the agent.** Action tools are stubs. A
  dashboard or executor should take the decision JSON and call a real
  execution layer (still without extra speculative `simulate_outcome` draws).
- **`delay_hours` is agent-authored**, not produced by the EV engine. Phase 5
  should decide whether to honor, clamp, or ignore it.
- **Non-retry probabilities remain simulator-oracle numbers.** UI copy should
  not present them as a trained multi-action model.
- **API key and spend.** Require `GEMINI_API_KEY` in env; cap tool loops
  (already 8) and consider caching EV tool results per payment in production.
- Still **do not import `data/recoverability.py`**. Still **do not** call
  `simulate_outcome` to "see what would happen" before committing.
## Phase 5: Guardrail Engine

Adds `guardrails/policy_engine.py` and `guardrails/config.py`, plus
`tests/test_guardrails.py`. Updates `engine/decision_engine.py` to delegate
to this engine instead of keeping its own hardcoded copy of the rules.

### Decision API

```python
from guardrails.policy_engine import check_guardrails

check_guardrails(
    payment_row,
    proposed_decision,          # what the EV argmax or the AI agent wants to run
    guardrail_config=None,      # overrides merged onto guardrails.config.DEFAULT_GUARDRAIL_CONFIG
    retry_attempts_so_far=0,    # Rule 3 input -- no persistent action log yet, caller tracks this
    incentive_pct=0.0,          # Rule 4 input
) -> dict
# {proposed_decision, allowed, final_decision, violated_rules, guardrail_config_used}
```

The core idea carried over from the spec: the AI agent (Phase 4) or the EV
argmax (Phase 3) *proposes* an action; `check_guardrails` has final say on
whether it's allowed to execute, and can force a safer one. `decision_engine
.decide_best_action()` now has no guardrail logic of its own -- it computes
the EV argmax, passes the proposal to `check_guardrails`, and returns
whatever `final_decision` comes back with.

### The 5 rules and their priority order

Evaluated independently; when more than one fires, the **most conservative**
forced outcome wins (documented in code comments in `policy_engine.py`):

1. **High amount → escalate** *(priority 2)* -- `amount > high_amount_threshold`
   (default `50000`) forces `"escalate"`.
2. **Repeated hard decline → stop** *(priority 1, strictest)* -- `failure_reason
   == "hard_decline"` and `previous_failures >= hard_decline_repeat_failures`
   (default `3`) forces `"stop"`. Wins over Rule 1 if both fire, since "no
   action" is always the safer outcome than "escalate".
3. **Unauthorized action** *(priority 3)* -- `proposed_decision` not one of
   `retry | payment_link | notification | escalate | stop` is rejected and
   falls back to `invalid_action_fallback` (default `"escalate"`).
4. **Max retries** *(priority 4)* -- if `proposed_decision == "retry"` and
   `retry_attempts_so_far >= max_retry_attempts` (default `2`), `"retry"` is
   blocked and falls back to `max_retry_action` (default `"escalate"`).
5. **Max incentive** *(lowest priority, no forced intervention change)* --
   `incentive_pct > max_incentive_pct` (default `0.05`) is flagged as a
   violation (so `allowed=False`) but does not change which intervention
   runs -- it only means the incentive itself must be capped before
   execution.

Rule 3 and Rule 4 can never both fire on the same call: Rule 4 only applies
once `proposed_decision == "retry"`, which is by definition a valid
intervention, so Rule 3 (invalid action) can't also be true for it.

All thresholds live in `guardrails/config.py`'s `DEFAULT_GUARDRAIL_CONFIG`
so they're not magic numbers -- easy to tweak for a demo via the
`guardrail_config` override dict.

### Tests

`tests/test_guardrails.py` has 12 cases covering: the pass-through case,
each rule firing individually, the Rule 1 + Rule 2 priority conflict (stop
wins), the amount-exactly-at-threshold and previous-failures-exactly-at-
threshold boundaries, an invalid proposed action, and a `guardrail_config`
override actually changing behavior.

Run with:

```bash
pytest tests/test_guardrails.py -v
```

**Note on how these were verified in this environment:** the sandbox this
Phase 5 work was done in has no network access, so `pytest` could not be
installed to produce a real `pytest ... -v` CLI summary line. Each of the
12 test functions was instead imported and executed directly (equivalent
assertions, same pass/fail semantics) against `guardrails/policy_engine.py`
and the updated `engine/decision_engine.py`: **12 passed, 0 failed**.
Please re-run the actual `pytest` command in the real repo environment to
get the standard summary line for the record.

**Caveat on `engine/expected_value.py`:** that file (and `models/`,
`data/`, `agent/`) were not part of this handoff's upload, so a small local
stand-in `expected_value.py` (same `VALID_INTERVENTIONS` list and
`calculate_expected_value()` signature/return shape, toy probabilities) was
used to exercise `decide_best_action()` end-to-end here -- mirroring how
Phase 3 itself was verified against a "locally reconstructed stand-in"
per the note above. Do not merge that stand-in file into the real repo;
`engine/decision_engine.py`'s changes only touch its guardrail call, not
its EV logic, so it should work unmodified against the real
`expected_value.py`.

### `engine/decision_engine.py` changes

- Removed the old inline `_apply_guardrails` function and
  `DEFAULT_GUARDRAIL_CONFIG` (that config now lives in
  `guardrails/config.py`).
- `decide_best_action()` still computes all five EV dicts via
  `calculate_expected_value()` and still returns the same
  `{payment_id, decision, expected_value, reason, all_evaluated}` shape.
- Internally, it now takes the EV argmax as a *proposal* and calls
  `guardrails.policy_engine.check_guardrails()` to get the real
  `final_decision`. `reason` now distinguishes "highest expected value"
  (nothing fired) from "overridden by guardrails: ..." (lists which rules
  fired), so `all_evaluated` + `reason` together still let an agent explain
  "the model preferred X but policy required Y" exactly as before.
- Added two new optional keyword arguments, `retry_attempts_so_far=0` and
  `incentive_pct=0.0`, so callers can opt into Rules 3 and 4; existing
  callers that don't pass them get identical behavior to before (modulo the
  two new rules simply never firing at their defaults).

### Caveats for Phase 6 (Evaluation)

- `evaluation/ml_strategy.py` should now be checked against the *new*
  `decide_best_action()` signature -- it still works unmodified with
  positional/keyword defaults, but if the evaluation loop wants to exercise
  Rules 3/4 it needs to track `retry_attempts_so_far` per `payment_id`
  itself (there's still no persistent action log) and pass an
  `incentive_pct` if the strategy being evaluated uses one.
- The seed-42 `naive_strategy` / `rule_based_strategy` / `ml_strategy`
  comparison table is still pending real numbers (per the Phase 3 section
  above) -- that hasn't changed here.
- Consider adding a guardrail-specific evaluation metric: how often the EV
  argmax's proposal gets overridden, and by which rule, across the full
  15,000-row dataset -- this feeds directly into the Phase 7 caveat below.

### Caveats for Phase 7 (Dashboard)

- Per the original spec's guardrail examples, the dashboard should surface
  a **guardrail violations count** as one of its KPIs -- e.g. total
  decisions where `allowed=False`, broken down by which of the 5 rules
  fired (`violated_rules` already gives rule names/descriptions to group
  by).
- Since Rule 4 (max incentive) doesn't change `final_decision`, the
  dashboard should treat it as a distinct "flagged, not blocked" category
  rather than lumping it in with the intervention-overriding rules (1, 2,
  3, and the invalid-action fallback).
- If/when a persistent action log exists, `retry_attempts_so_far` should
  come from real history instead of being passed in per-call -- the
  dashboard is a natural place to expose that log.

## Phase 6: Evaluation Engine

Adds the experimentation/evaluation layer in `evaluation/experiments.py` and
reusable metrics in `evaluation/metrics.py`. No files under `data/`, `models/`,
`engine/`, `agent/`, or `guardrails/` were modified.

### Function signatures

```python
run_control_experiment(
    df=None,
    control_strategy="rule_based",
    treatment_strategy="ml_strategy",
    split_ratio=0.5,
    random_state=42,
) -> dict

calculate_recovery_rate(df_with_outcomes) -> float
calculate_total_revenue(df_with_outcomes) -> float
summarize_decision_mix(df_with_decisions) -> dict[str, int]
```

`run_control_experiment()` loads `data/sample_data/payments.csv` when `df` is
omitted, shuffles reproducibly with `random_state=42`, and creates CONTROL and
TREATMENT (RecoverIQ) groups according to `split_ratio`. CONTROL uses the
existing `rule_based_strategy`. TREATMENT calls `decide_best_action()` and then
calls `simulate_outcome()` exactly once for the selected intervention.

Incremental revenue is normalized to a common comparison size before the
comparison:

```text
incremental_revenue = treatment_revenue_scaled - control_revenue_scaled
incremental_rate_pp = treatment_recovery_rate - control_recovery_rate
```

### Real Phase 6 run

The real `data/sample_data/payments.csv` contains **15,000 events**. With the
requested 50/50 split and `random_state=42`, the run produced 7,500 CONTROL and
7,500 TREATMENT events.

| Metric | CONTROL (rule-based) | TREATMENT (RecoverIQ) |
|---|---:|---:|
| Sample size | 7,500 | 7,500 |
| Recovered events | 3,739 | 3,897 |
| Recovery rate | 49.85% | 51.96% |
| Revenue recovered | ₹3,626,469.68 | ₹3,554,409.53 |

Because the groups are exactly equal-sized in this run, scaled revenue is the
same as observed revenue. Therefore:

- **Incremental revenue: -₹72,060.15**
- **Incremental recovery rate: +2.11 percentage points**

This result is important: RecoverIQ recovered more events and improved the
recovery rate on this randomized split, but the recovered revenue was lower on
this particular split because the treatment group contained a different mix of
payment amounts. The experiment therefore reports both incremental rate and
incremental revenue rather than relying on recovery rate alone.

### Four-strategy progression

All four strategies were evaluated against the same full 15,000-event
`payments.csv` dataset using the deterministic simulator outcomes.

| Strategy | Revenue recovered | Recovery rate | Recovered events |
|---|---:|---:|---:|
| `naive_strategy` | ₹6,030,389.05 | 42.43% | 6,364 |
| `rule_based_strategy` | ₹7,070,455.81 | 50.07% | 7,510 |
| `ml_strategy` (ML/EV proposal before guardrails) | ₹7,302,609.68 | 51.85% | 7,777 |
| RecoverIQ (ML/EV + guardrails) | ₹7,289,028.51 | 51.71% | 7,757 |

For the progression table, `ml_strategy` is represented by the highest-EV
proposal contained in `decide_best_action()["all_evaluated"]`. This separates
the ML/EV proposal from the final RecoverIQ decision, which applies the Phase 5
guardrails. The existing `evaluation/ml_strategy.py` currently calls
`decide_best_action()` directly, so it is guardrail-aware; Phase 6 deliberately
uses its underlying decision-engine evaluation data to expose the requested
ML-only progression without modifying that existing file.

### Phase 6 verification

- `evaluation/metrics.py` and `evaluation/experiments.py` compile successfully
  with `python -m py_compile`.
- `run_control_experiment(..., random_state=42)` was executed against the real
  15,000-row `payments.csv`.
- The real model artifact was loaded and used by the decision engine.
- The simulator was used to realize outcomes only after a strategy selected an
  intervention.
- The environment emitted sklearn `InconsistentVersionWarning` messages
  because the saved model was created with scikit-learn 1.9.0 while this
  execution environment has 1.8.0. The run completed successfully; the model
  should be regenerated or evaluated under the project's pinned/target
  dependency version before production use.

### Caveats for Phase 7 (Dashboard)

- The dashboard should display **incremental revenue** as the headline KPI,
  per the original spec's dashboard design. Raw treatment recovery should not
  be presented as the primary measure of business impact.
- Display incremental recovery rate in percentage points alongside incremental
  revenue so users can see both event-level and revenue-level lift.
- The dashboard can import `calculate_recovery_rate`,
  `calculate_total_revenue`, and `summarize_decision_mix` directly from
  `evaluation.metrics`.
- Because one randomized split can have different payment-value mixes, the
  dashboard should make the experiment's sample sizes and scaling method
  visible when showing incremental revenue.
- A single split is not sufficient for a production causal claim. Phase 7 can
  expose the seed/split metadata and, if the original spec permits, support
  repeated seeded experiments or confidence intervals later.

### Multi-split robustness extension

Added the following reusable Phase 6 function:

```text
run_multiple_splits(n_seeds=10)
```

It runs `run_control_experiment()` once for each seed from `0` through
`n_seeds - 1` and returns the individual experiment results plus:

```text
average_incremental_revenue
average_incremental_rate_pp
```

This provides a simple multi-seed robustness check instead of relying on a
single randomized control/treatment split. The returned `runs` list preserves
the complete result for every seed so Phase 7 can display or analyze the
individual splits if desired.

---

## Phase 7: Dashboard

Adds `dashboard/app.py`, an interactive Streamlit dashboard visualizing RecoverIQ's revenue recovery decision engine, experimentation results, strategy progression benchmarks, and real-time per-transaction decision inspection.

### How to run

```bash
pip install streamlit plotly
streamlit run dashboard/app.py
```

### Performance design: Cached vs Computed Live

To ensure the dashboard loads in under a second and delivers snappy interactivity without triggering heavy ML batch pipelines or API rate limits:

- **Cached from `evaluation/dashboard_cache.json`:**
  - **Headline KPIs & Progression**: Recovered revenue (₹7,289,028.51), overall recovery rate (51.71%), incremental revenue (+₹189,972.91), incremental rate lift (+2.01 pp), and decision mix across all 4 benchmark strategies (`naive_strategy`, `rule_based_strategy`, `ml_strategy`, and `RecoverIQ`).
  - *Why*: Running full 15,000-row ML + EV + Guardrail evaluations or multi-seed experiments on every page load is too slow and unnecessary when evaluation benchmarks are pre-computed.

- **Computed live / cached in-memory with `@st.cache_data`:**
  - **Raw Dataset Aggregations (`data/sample_data/payments.csv`)**: Total revenue at risk (₹14,123,437.88), breakdown by failure reason, and customer attributes.
  - **Guardrail Overview Stats**: Scans the 15,000 rows with lightweight `guardrails.policy_engine.check_guardrails()` to report total safety overrides (1,415 / 9.4%).
  - **ML Probability Sample Distribution**: Fast batched inference of `predict_recovery_probability()` on a 500-payment sample for histogram visualization.
  - *Why*: Pandas aggregations and single batched inferences take <100ms, allowing real-time filtering and visualization from raw data.

- **AI Agent Reasoning Examples (`dashboard/agent_examples.py`):**
  - Displays real, captured natural-language explanations and tool-calling decisions generated by the Phase 4 AI agent (`agent/agent.py: run_agent_decision()`) using Gemini tool-calling.
  - *Why*: Free-tier Gemini API keys have strict rate limits (e.g. 5 requests/min); static rendering of real captured agent outputs allows users to review the agent's autonomous reasoning and verified alignment with Phase 3 deterministic EV decisions without risking live 429 rate limit errors during an interactive demo.

- **On-Demand Single-Row Decisioning (`decide_best_action(selected_row)`):**
  - When a user selects a specific `payment_id` (e.g. `PAY-0000016`, `PAY-0000005`, `PAY-0000001`), `decide_best_action()` and `simulate_outcome()` run live for that **single transaction only**.
  - Displays live Expected Value calculations across all 5 candidate actions, rationale, and trust & safety guardrail audit statuses.

- **What-If Scenario Simulator (`dashboard/app.py`):**
  - Interactive simulator enabling users to construct custom hypothetical payment failure events (`WHATIF-CUSTOM`) by tweaking amount, failure reason, payment method, customer LTV, customer age, tenure, and prior successes/failures.
  - Runs `engine.decision_engine.decide_best_action()` live to compute optimal action proposals, evaluate EV across all 5 candidate interventions, and enforce safety guardrails.
  - Includes 3 preset edge-case scenarios:
    1. *High-Value Escalation Test* (₹75,000 failure triggering Rule 1: `amount > 50000` $\to$ forced `escalate`).
    2. *Repeat Offender Test* (`hard_decline` with 5 previous failures triggering Rule 2 $\to$ forced `stop`).
    3. *Ideal Recovery Case* (UPI transaction with 15 prior successes and 0 failures $\to$ high EV `retry`).

### Known Limitations

1. **Revenue at Risk Scope**: "Revenue at Risk" (₹14.12M) sums the amount across all failed payment events in `payments.csv`. In a production setup with active A/B splits, only the treatment cohort is actively processed by RecoverIQ.
2. **Retry History Tracking**: In the standalone single-transaction view, `retry_attempts_so_far` defaults to 0 since payments.csv is an event log rather than a stateful action journal.
3. **Non-Retry Probabilities**: As noted in Phases 2-4, recovery probabilities for `payment_link`, `notification`, and `escalate` are sourced from the simulator environment rather than a dedicated multi-action classifier.
4. **AI Agent Live Calls**: Live Gemini calls are omitted on page loads due to free-tier rate limits, with authentic agent reasoning showcased via `dashboard/agent_examples.py`.
