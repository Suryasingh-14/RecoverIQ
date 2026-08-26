"""
Test-set evaluation for the retry-recovery models.

Prints accuracy, AUC-ROC, precision, recall for logistic regression and
random forest; a 10-bucket calibration table; and top feature importances.

Does NOT import data/recoverability.py.

Run from repo root (builds training data / model if missing):
    python models/model_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from recovery_model import (  # noqa: E402
    MODEL_PATH,
    TRAINING_DATA_PATH,
    load_training_frame,
    train_models,
    transformed_feature_names,
)
from train_data_builder import (  # noqa: E402
    build_training_data,
    save_training_data,
)


def classification_metrics(
    y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc_roc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"{name}")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  auc_roc:   {metrics['auc_roc']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")


def calibration_table(
    y_true: pd.Series, proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Decile buckets of predicted probability vs observed recovery rate."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    frame["bucket"] = pd.qcut(
        frame["p"], q=n_bins, labels=False, duplicates="drop"
    )
    rows = []
    for b, grp in frame.groupby("bucket", sort=True):
        rows.append(
            {
                "decile": int(b) + 1,
                "n": int(len(grp)),
                "mean_predicted": float(grp["p"].mean()),
                "actual_recovery_rate": float(grp["y"].mean()),
                "gap": float(grp["p"].mean() - grp["y"].mean()),
            }
        )
    return pd.DataFrame(rows)


def logistic_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    names = transformed_feature_names(pipeline)
    coef = pipeline.named_steps["clf"].coef_.ravel()
    out = pd.DataFrame({"feature": names, "coefficient": coef})
    out["abs_coefficient"] = out["coefficient"].abs()
    return out.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def forest_importances(pipeline: Pipeline) -> pd.DataFrame:
    names = transformed_feature_names(pipeline)
    imp = pipeline.named_steps["clf"].feature_importances_
    out = pd.DataFrame({"feature": names, "importance": imp})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def print_calibration(table: pd.DataFrame) -> None:
    print("calibration (test set, predicted-probability deciles)")
    print(
        table.assign(
            mean_predicted=lambda d: d["mean_predicted"].round(4),
            actual_recovery_rate=lambda d: d["actual_recovery_rate"].round(4),
            gap=lambda d: d["gap"].round(4),
        ).to_string(index=False)
    )


def ensure_training_data() -> None:
    if TRAINING_DATA_PATH.exists():
        return
    print(f"{TRAINING_DATA_PATH} missing - building labels via simulator...")
    save_training_data(build_training_data())


def main() -> None:
    ensure_training_data()
    df = load_training_frame()
    bundle = train_models(df)

    lr = bundle["logistic_regression"]
    rf = bundle["random_forest"]
    X_test, y_test = bundle["X_test"], bundle["y_test"]
    lr_proba = lr.predict_proba(X_test)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]
    lr_metrics = classification_metrics(y_test, lr_proba)
    rf_metrics = classification_metrics(y_test, rf_proba)

    print("=" * 72)
    print("Phase 2 model evaluation (test 20%, random_state=42, stratify)")
    print("labels: simulate_outcome(..., intervention='retry')")
    print("=" * 72)
    print_metrics("logistic_regression", lr_metrics)
    print()
    print_metrics("random_forest", rf_metrics)
    print()
    print(f"chosen for {MODEL_PATH.name}: {bundle['chosen_name']}")
    print()

    chosen_proba = rf_proba if bundle["chosen_name"] == "random_forest" else lr_proba
    print_calibration(calibration_table(y_test, chosen_proba))
    print()

    print("top logistic_regression coefficients (abs rank)")
    print(logistic_coefficients(lr).head(15).round(4).to_string(index=False))
    print()
    print("top random_forest feature_importances_")
    print(forest_importances(rf).head(15).round(4).to_string(index=False))
    print("=" * 72)

    # Returned for HANDOFF updates when this module is imported.
    return {
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "chosen_name": bundle["chosen_name"],
    }


if __name__ == "__main__":
    main()
