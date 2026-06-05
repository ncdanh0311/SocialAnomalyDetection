"""Huấn luyện mô hình phát hiện bất thường và phân loại bot."""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SEED = 42


def _positive_class_scores(model, X_test: pd.DataFrame | np.ndarray) -> np.ndarray | None:
    """Return positive-class scores when a model exposes probabilities."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        if probabilities.shape[1] > 1:
            return probabilities[:, 1]
    return None


def _safe_roc_auc(y_true: pd.Series | np.ndarray, y_score: np.ndarray | None) -> float:
    """Compute ROC AUC, returning NaN when only one class is present."""
    if y_score is None or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _binary_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard binary metrics with safe zero-division behavior."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_isolation_forest(
    X_human_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    contamination: float = 0.05,
) -> dict:
    """Huấn luyện Isolation Forest trên tài khoản người thật và đánh giá trên tập test.

    Isolation Forest trả về -1 cho điểm bất thường và 1 cho điểm bình thường.
    Hàm ánh xạ -1 thành bất thường=1 và 1 thành bình thường=0 để đánh giá.
    """
    print("[mô hình] Đang huấn luyện Isolation Forest trên tài khoản người thật...")
    if not 0 < contamination <= 0.5:
        clipped = min(max(float(contamination), 0.001), 0.5)
        warnings.warn(
            f"Isolation Forest yêu cầu contamination thuộc (0, 0.5]. "
            f"Sử dụng {clipped:.4f} thay cho {contamination:.4f}.",
            stacklevel=2,
        )
        contamination = clipped

    model = IsolationForest(contamination=contamination, random_state=SEED)
    model.fit(X_human_train)

    raw_predictions = model.predict(X_test)
    y_pred = np.where(raw_predictions == -1, 1, 0)
    anomaly_scores = -model.decision_function(X_test)
    metrics = _binary_metrics(y_test, y_pred)
    metrics.update(
        {
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["human", "bot"], zero_division=0
            ),
            "roc_auc": _safe_roc_auc(y_test, anomaly_scores),
            "predictions": y_pred,
            "scores": anomaly_scores,
            "model": model,
        }
    )
    print("[mô hình] Hoàn tất Isolation Forest")
    return metrics


def run_random_forest(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> dict:
    """Huấn luyện Random Forest và trả về các chỉ số đánh giá."""
    print("[mô hình] Đang huấn luyện Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = _positive_class_scores(model, X_test)
    feature_names = (
        list(X_train.columns)
        if hasattr(X_train, "columns")
        else [f"feature_{index}" for index in range(model.n_features_in_)]
    )
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )

    metrics = _binary_metrics(y_test, y_pred)
    metrics.update(
        {
            "classification_report": classification_report(
                y_test, y_pred, target_names=["human", "bot"], zero_division=0
            ),
            "classification_report_dict": classification_report(
                y_test,
                y_pred,
                target_names=["human", "bot"],
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "feature_importances": importances,
            "roc_auc": _safe_roc_auc(y_test, y_score),
            "predictions": y_pred,
            "probabilities": y_score,
            "model": model,
        }
    )
    print("[mô hình] Hoàn tất Random Forest")
    return metrics
