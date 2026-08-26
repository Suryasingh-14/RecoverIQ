"""
Retry-recovery probability model.

Features: CSV columns only (one-hot method/reason + listed numerics).
Target: payment_recovered from models/training_data.csv (retry simulator labels).

Does NOT import data/recoverability.py.

Public API:
    predict_recovery_probability(payment_row) -> float
    load_recovery_model(path=None) -> sklearn Pipeline

Train from repo root:
    python models/recovery_model.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = PROJECT_ROOT / "models" / "training_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "recovery_model.pkl"

CATEGORICAL_FEATURES = ["payment_method", "failure_reason"]
NUMERIC_FEATURES = [
    "customer_age",
    "previous_successes",
    "previous_failures",
    "customer_value",
    "subscription_age",
    "days_since_last_payment",
    "amount",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "payment_recovered"
RANDOM_STATE = 42
TEST_SIZE = 0.20

RowLike = Union[Mapping[str, Any], pd.Series, Any]

_LOADED_MODEL: Optional[Pipeline] = None


def load_training_frame(path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python models/train_data_builder.py"
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"training_data.csv missing columns: {missing}")
    return df


def payment_row_to_frame(payment_row: RowLike) -> pd.DataFrame:
    """One-row DataFrame with the original payment feature columns."""
    if isinstance(payment_row, pd.DataFrame):
        if len(payment_row) != 1:
            raise ValueError("payment_row DataFrame must have exactly one row")
        frame = payment_row.copy()
    elif isinstance(payment_row, pd.Series):
        frame = payment_row.to_frame().T
    elif isinstance(payment_row, Mapping):
        frame = pd.DataFrame([dict(payment_row)])
    elif hasattr(payment_row, "_asdict"):
        frame = pd.DataFrame([payment_row._asdict()])
    else:
        frame = pd.DataFrame([dict(payment_row)])
    return frame.reset_index(drop=True)


def make_logistic_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def make_forest_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    # Leaf size regularizes against Bernoulli label noise from the simulator.
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def split_train_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def transformed_feature_names(pipeline: Pipeline) -> np.ndarray:
    return pipeline.named_steps["pre"].get_feature_names_out()


def train_models(
    df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """
    Fit logistic regression and random forest on the same 80/20 split.

    Selection criterion for the saved artifact: higher test ROC-AUC.
    """
    if df is None:
        df = load_training_frame()
    X_train, X_test, y_train, y_test = split_train_test(df)

    logistic = make_logistic_pipeline()
    forest = make_forest_pipeline()
    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    lr_proba = logistic.predict_proba(X_test)[:, 1]
    rf_proba = forest.predict_proba(X_test)[:, 1]
    lr_auc = float(roc_auc_score(y_test, lr_proba))
    rf_auc = float(roc_auc_score(y_test, rf_proba))

    if rf_auc >= lr_auc:
        chosen_name = "random_forest"
        chosen_model = forest
        chosen_auc = rf_auc
    else:
        chosen_name = "logistic_regression"
        chosen_model = logistic
        chosen_auc = lr_auc

    return {
        "logistic_regression": logistic,
        "random_forest": forest,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "auc": {
            "logistic_regression": lr_auc,
            "random_forest": rf_auc,
        },
        "chosen_name": chosen_name,
        "chosen_model": chosen_model,
        "chosen_auc": chosen_auc,
    }


def save_recovery_model(
    pipeline: Pipeline,
    path: Path = MODEL_PATH,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Persist the sklearn Pipeline plus light metadata for Phase 3 loaders."""
    payload = {
        "pipeline": pipeline,
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "target": TARGET_COLUMN,
        "label_intervention": "retry",
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


def load_recovery_model(path: Optional[Path] = None) -> Pipeline:
    """Load the saved Pipeline (unwraps the joblib payload dict)."""
    global _LOADED_MODEL
    model_path = Path(path) if path is not None else MODEL_PATH
    if _LOADED_MODEL is not None and path is None:
        return _LOADED_MODEL
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run: python models/recovery_model.py"
        )
    payload = joblib.load(model_path)
    if isinstance(payload, Pipeline):
        pipeline = payload
    else:
        pipeline = payload["pipeline"]
    if path is None:
        _LOADED_MODEL = pipeline
    return pipeline


def predict_recovery_probability(payment_row: RowLike) -> float:
    """
    P(payment_recovered | features, intervention=retry) in [0, 1].

    ``payment_row`` may be a dict, Series, or namedtuple with the payments.csv
    columns (at least FEATURE_COLUMNS). Extra keys are ignored.
    """
    pipeline = load_recovery_model()
    frame = payment_row_to_frame(payment_row)
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"payment_row missing required feature columns: {missing}")
    proba = pipeline.predict_proba(frame[FEATURE_COLUMNS])[0, 1]
    return float(np.clip(proba, 0.0, 1.0))


def main() -> None:
    bundle = train_models()
    meta = {
        "chosen_name": bundle["chosen_name"],
        "test_auc": bundle["auc"],
    }
    out = save_recovery_model(bundle["chosen_model"], metadata=meta)
    print("test ROC-AUC")
    print(f"  logistic_regression: {bundle['auc']['logistic_regression']:.4f}")
    print(f"  random_forest:       {bundle['auc']['random_forest']:.4f}")
    print(f"chosen model: {bundle['chosen_name']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
