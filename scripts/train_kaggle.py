"""End-to-end training entrypoint for Kaggle or local retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features  # noqa: E402
from src.models import run_isolation_forest, run_random_forest  # noqa: E402
from src.preprocess import DATASET_FOLDERS, load_cresci2017  # noqa: E402


SEED = 42
SKEWED_COLUMNS = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "followers_friends_ratio",
]
ISO_FEATURES = [
    "followers_count",
    "friends_count",
    "followers_friends_ratio",
    "statuses_count",
    "favourites_count",
    "account_age_days",
    "tweets_per_day",
]


def _looks_like_raw_data_dir(path: Path) -> bool:
    return any(
        (path / folder / "users.csv").exists()
        or (path / f"{folder}.csv").exists()
        or (path / f"{folder}.csv" / "users.csv").exists()
        for folder in DATASET_FOLDERS
    )


def _auto_data_dir() -> Path:
    local_raw = PROJECT_ROOT / "data" / "raw"
    if _looks_like_raw_data_dir(local_raw):
        return local_raw

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for users_file in kaggle_input.rglob("users.csv"):
            if users_file.parent.name in DATASET_FOLDERS:
                return users_file.parent.parent

    raise FileNotFoundError(
        "Cannot find Cresci-2017 raw data. Pass --data-dir pointing to the folder "
        "that contains genuine_accounts/, social_spambots_1/, ..."
    )


def _default_output_root() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working
    return PROJECT_ROOT


def _log_transform_counts(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_log = X_train.copy()
    X_test_log = X_test.copy()
    for column in SKEWED_COLUMNS:
        if column in X_train_log.columns:
            X_train_log[column] = np.log1p(X_train_log[column].clip(lower=0))
            X_test_log[column] = np.log1p(X_test_log[column].clip(lower=0))
    return X_train_log, X_test_log


def _prepare_matrices(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, SimpleImputer, StandardScaler, SimpleImputer, StandardScaler]:
    X_train_log, X_test_log = _log_transform_counts(X_train, X_test)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_log), columns=X_train.columns, index=X_train.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_log), columns=X_train.columns, index=X_test.index)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_train.columns, index=X_test.index)

    missing_iso = [column for column in ISO_FEATURES if column not in X_train_log.columns]
    if missing_iso:
        raise ValueError(f"Missing Isolation Forest features: {missing_iso}")

    X_train_iso = X_train_log[ISO_FEATURES].copy()
    X_test_iso = X_test_log[ISO_FEATURES].copy()

    imputer_iso = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_imputed_iso = pd.DataFrame(
        imputer_iso.fit_transform(X_train_iso),
        columns=ISO_FEATURES,
        index=X_train.index,
    )
    X_test_imputed_iso = pd.DataFrame(
        imputer_iso.transform(X_test_iso),
        columns=ISO_FEATURES,
        index=X_test.index,
    )

    scaler_iso = StandardScaler()
    X_train_scaled_iso = pd.DataFrame(
        scaler_iso.fit_transform(X_train_imputed_iso),
        columns=ISO_FEATURES,
        index=X_train.index,
    )
    X_test_scaled_iso = pd.DataFrame(
        scaler_iso.transform(X_test_imputed_iso),
        columns=ISO_FEATURES,
        index=X_test.index,
    )

    return (
        X_train_scaled,
        X_test_scaled,
        X_train_scaled_iso,
        X_test_scaled_iso,
        imputer,
        scaler,
        imputer_iso,
        scaler_iso,
    )


def _choose_contamination(X_train_iso: pd.DataFrame, y_train: pd.Series) -> tuple[float, pd.DataFrame]:
    X_iso_fit, X_iso_val, y_iso_fit, y_iso_val = train_test_split(
        X_train_iso,
        y_train,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train,
    )

    rows = []
    for contamination in [0.01, 0.02, 0.03, 0.04, 0.05]:
        result = run_isolation_forest(
            X_iso_fit.loc[y_iso_fit == 0],
            X_iso_val,
            y_iso_val,
            contamination=contamination,
        )
        rows.append({"contamination": contamination, "f1": result["f1"]})

    validation = pd.DataFrame(rows)
    best = float(validation.loc[validation["f1"].idxmax(), "contamination"])
    return best, validation


def train(data_dir: Path, output_root: Path, test_size: float) -> None:
    processed_dir = output_root / "data" / "processed"
    models_dir = output_root / "outputs" / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Project root: {PROJECT_ROOT}")
    print(f"[train] Raw data dir: {data_dir}")
    print(f"[train] Output root: {output_root}")

    raw = load_cresci2017(data_dir)
    X, y = build_features(raw)
    if X.empty or y.empty:
        raise ValueError("Feature matrix or label vector is empty. Check --data-dir.")

    features_df = X.copy()
    features_df["label"] = y.to_numpy()
    features_df.to_csv(processed_dir / "features.csv", index=False)
    print(f"[train] Saved features: {processed_dir / 'features.csv'}")
    print(f"[train] X={X.shape}, bot_rate={y.mean() * 100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )

    (
        X_train_scaled,
        X_test_scaled,
        X_train_scaled_iso,
        X_test_scaled_iso,
        imputer,
        scaler,
        imputer_iso,
        scaler_iso,
    ) = _prepare_matrices(X_train, X_test)

    best_contamination, validation = _choose_contamination(X_train_scaled_iso, y_train)
    validation.to_csv(models_dir / "isolation_contamination_validation.csv", index=False)
    print(f"[train] Best contamination: {best_contamination:.2f}")

    iso_results = run_isolation_forest(
        X_train_scaled_iso.loc[y_train == 0],
        X_test_scaled_iso,
        y_test,
        contamination=best_contamination,
    )
    rf_results = run_random_forest(X_train_scaled, X_test_scaled, y_train, y_test)

    joblib.dump(imputer, models_dir / "imputer.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(imputer_iso, models_dir / "imputer_iso.pkl")
    joblib.dump(scaler_iso, models_dir / "scaler_iso.pkl")
    joblib.dump(iso_results["model"], models_dir / "isolation_forest.pkl")
    joblib.dump(rf_results["model"], models_dir / "random_forest.pkl")

    train_data = X_train.copy()
    train_data["label"] = y_train.to_numpy()
    train_data.to_csv(processed_dir / "train_data.csv", index=False)

    test_data = X_test.copy()
    test_data["label"] = y_test.to_numpy()
    test_data.to_csv(processed_dir / "test_data.csv", index=False)

    rf_results["feature_importances"].to_csv(models_dir / "feature_importance.csv", header=["importance"])

    iso_cm = iso_results["confusion_matrix"]
    rf_cm = rf_results["confusion_matrix"]
    metrics = {
        "isolation_forest": {
            "name": "Isolation Forest",
            "accuracy": round(iso_results["accuracy"], 4),
            "precision": round(iso_results["precision"], 4),
            "recall": round(iso_results["recall"], 4),
            "f1": round(iso_results["f1"], 4),
            "auc": round(iso_results["roc_auc"], 4),
            "contamination": best_contamination,
            "tn": int(iso_cm[0, 0]),
            "fp": int(iso_cm[0, 1]),
            "fn": int(iso_cm[1, 0]),
            "tp": int(iso_cm[1, 1]),
        },
        "random_forest": {
            "name": "Random Forest",
            "accuracy": round(rf_results["accuracy"], 4),
            "precision": round(rf_results["precision"], 4),
            "recall": round(rf_results["recall"], 4),
            "f1": round(rf_results["f1"], 4),
            "auc": round(rf_results["roc_auc"], 4),
            "tn": int(rf_cm[0, 0]),
            "fp": int(rf_cm[0, 1]),
            "fn": int(rf_cm[1, 0]),
            "tp": int(rf_cm[1, 1]),
        },
    }
    with (models_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print("[train] Random Forest metrics:")
    print(json.dumps(metrics["random_forest"], indent=2))
    print("[train] Saved model artifacts:")
    for path in sorted(models_dir.glob("*")):
        print(f"  - {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bot detection models for Kaggle/local use.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder containing Cresci-2017 subset folders. Defaults to local data/raw or auto-detected Kaggle input.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Writable output root. Defaults to /kaggle/working on Kaggle, otherwise project root.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve() if args.data_dir else _auto_data_dir()
    output_root = args.output_root.resolve() if args.output_root else _default_output_root()
    train(data_dir=data_dir, output_root=output_root, test_size=args.test_size)


if __name__ == "__main__":
    main()
