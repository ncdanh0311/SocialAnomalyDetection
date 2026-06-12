from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.features import SPAM_KEYWORDS, URL_MARKERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
SAMPLE_ACCOUNTS_FILE = PROJECT_ROOT / "data" / "sample_accounts.csv"
LATEST_BATCH_RESULTS_FILE = PREDICTIONS_DIR / "latest_batch_results.csv"
TESTED_ACCOUNTS_FILE = PREDICTIONS_DIR / "tested_accounts.csv"

FEATURES_FILE = PROCESSED_DIR / "features.csv"
METRICS_FILE = MODELS_DIR / "metrics.json"
IMPUTER_PATH = MODELS_DIR / "imputer.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
IMPUTER_ISO_PATH = MODELS_DIR / "imputer_iso.pkl"
SCALER_ISO_PATH = MODELS_DIR / "scaler_iso.pkl"
ISOLATION_FOREST_PATH = MODELS_DIR / "isolation_forest.pkl"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.pkl"

REQUIRED_CSV_COLUMNS = {
    "screen_name",
    "followers_count",
    "friends_count",
    "statuses_count",
}
RECOMMENDED_CSV_COLUMNS = {
    "name",
    "description",
    "favourites_count",
    "listed_count",
    "verified",
    "default_profile",
    "default_profile_image",
    "geo_enabled",
    "protected",
}

DEFAULT_FEATURE_COLUMNS = [
    "statuses_count",
    "followers_count",
    "friends_count",
    "favourites_count",
    "listed_count",
    "default_profile",
    "default_profile_image",
    "geo_enabled",
    "profile_use_background_image",
    "utc_offset",
    "is_translator",
    "protected",
    "verified",
    "followers_friends_ratio",
    "friends_followers_ratio",
    "friends_followers_gap",
    "account_age_days",
    "tweets_per_day",
    "statuses_followers_ratio",
    "statuses_friends_ratio",
    "favourites_statuses_ratio",
    "has_description",
    "name_length",
    "screen_name_length",
    "description_length",
    "screen_name_digit_ratio",
    "screen_name_has_digits",
    "screen_name_has_spam_keyword",
    "description_has_spam_keyword",
    "description_has_url",
    "spam_keyword_count",
]

FALLBACK_METRICS = {
    "isolation_forest": {
        "name": "Isolation Forest",
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "auc": None,
    },
    "random_forest": {
        "name": "Random Forest",
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "auc": None,
    },
}

BOT_DEFAULT_THRESHOLD = 0.50
BOT_REVIEW_THRESHOLD = 0.40


app = FastAPI(
    title="Hệ thống phát hiện tài khoản bất thường và bot",
    description=(
        "Isolation Forest phát hiện tài khoản bất thường. "
        "Random Forest phân loại tài khoản người thật hoặc bot."
    ),
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


def load_artifact(path: Path) -> Optional[Any]:
    """Tải artifact đã lưu nếu file tồn tại."""
    if not path.exists():
        return None
    return joblib.load(path)


def load_metrics() -> dict[str, dict[str, Any]]:
    """Tải chỉ số đánh giá được sinh sau bước huấn luyện."""
    if not METRICS_FILE.exists():
        return FALLBACK_METRICS
    with METRICS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_feature_schema() -> tuple[list[str], pd.Series]:
    """Tải thứ tự đặc trưng và trung vị dự phòng từ dữ liệu đã xử lý."""
    if not FEATURES_FILE.exists():
        medians = pd.Series(0.0, index=DEFAULT_FEATURE_COLUMNS)
        return DEFAULT_FEATURE_COLUMNS, medians

    features = pd.read_csv(FEATURES_FILE)
    columns = [column for column in features.columns if column != "label"]
    medians = features[columns].median(numeric_only=True).reindex(columns).fillna(0.0)
    return columns, medians


def load_dataset_summary() -> dict[str, Any]:
    """Tổng hợp quy mô và tỷ lệ nhãn từ dữ liệu đã xử lý."""
    summary: dict[str, Any] = {
        "total": 0,
        "human": 0,
        "bot": 0,
        "feature_count": len(DEFAULT_FEATURE_COLUMNS),
        "human_rate": 0.0,
        "bot_rate": 0.0,
    }
    if not FEATURES_FILE.exists():
        return summary

    features = pd.read_csv(FEATURES_FILE)
    summary["total"] = len(features)
    summary["feature_count"] = len([column for column in features.columns if column != "label"])
    if "label" in features.columns:
        counts = features["label"].value_counts()
        summary["human"] = int(counts.get(0, 0))
        summary["bot"] = int(counts.get(1, 0))
        if summary["total"]:
            summary["human_rate"] = round(summary["human"] / summary["total"] * 100, 2)
            summary["bot_rate"] = round(summary["bot"] / summary["total"] * 100, 2)
    return summary


FEATURE_COLUMNS, FEATURE_MEDIANS = load_feature_schema()
DATASET_SUMMARY = load_dataset_summary()
MODEL_METRICS = load_metrics()
IMPUTER = load_artifact(IMPUTER_PATH)
SCALER = load_artifact(SCALER_PATH)
IMPUTER_ISO = load_artifact(IMPUTER_ISO_PATH)
SCALER_ISO = load_artifact(SCALER_ISO_PATH)
ISOLATION_FOREST = load_artifact(ISOLATION_FOREST_PATH)
RANDOM_FOREST = load_artifact(RANDOM_FOREST_PATH)


def startup_warnings() -> list[str]:
    """Trả về cảnh báo nếu thiếu artifact cần thiết cho demo."""
    warnings: list[str] = []
    if IMPUTER is None:
        warnings.append("Thiếu imputer.pkl. Demo sẽ dùng trung vị từ features.csv.")
    if SCALER is None:
        warnings.append("Thiếu scaler.pkl. Demo sẽ sử dụng đặc trưng chưa chuẩn hóa.")
    if IMPUTER_ISO is None:
        warnings.append("Thiếu imputer_iso.pkl.")
    if SCALER_ISO is None:
        warnings.append("Thiếu scaler_iso.pkl.")
    if ISOLATION_FOREST is None:
        warnings.append("Thiếu isolation_forest.pkl. Không thể phát hiện bất thường.")
    if RANDOM_FOREST is None:
        warnings.append("Thiếu random_forest.pkl. Không thể phân loại bot.")
    if not FEATURES_FILE.exists():
        warnings.append("Thiếu features.csv. Demo đang dùng danh sách đặc trưng dự phòng.")
    if not METRICS_FILE.exists():
        warnings.append("Thiếu metrics.json. Bảng chỉ số chưa được cập nhật.")
    return warnings


def normalize_key(key: str) -> str:
    """Chuẩn hóa tên cột đầu vào."""
    return str(key).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_row(data: dict[str, Any]) -> dict[str, Any]:
    """Chuẩn hóa alias thường gặp trong form và CSV."""
    normalized = {normalize_key(key): value for key, value in data.items()}
    aliases = {
        "follower_count": "followers_count",
        "followers": "followers_count",
        "following_count": "friends_count",
        "friend_count": "friends_count",
        "friends": "friends_count",
        "favorite_count": "favourites_count",
        "favorites_count": "favourites_count",
        "status_count": "statuses_count",
        "tweet_count": "statuses_count",
        "username": "screen_name",
        "bio": "description",
    }
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    return normalized


def validate_csv_records(records: list[dict[str, Any]]) -> list[str]:
    """Kiểm tra schema CSV và trả về cảnh báo cho các cột phụ còn thiếu."""
    if not records:
        raise ValueError("File CSV không có dòng dữ liệu.")

    normalized_columns = set(normalize_row(records[0]).keys())
    missing_required = sorted(REQUIRED_CSV_COLUMNS - normalized_columns)
    if "account_age_days" not in normalized_columns and "created_at" not in normalized_columns:
        missing_required.append("account_age_days hoặc created_at")
    if missing_required:
        raise ValueError(
            "File CSV thiếu cột bắt buộc: " + ", ".join(missing_required) + "."
        )

    missing_recommended = sorted(RECOMMENDED_CSV_COLUMNS - normalized_columns)
    if not missing_recommended:
        return []
    return [
        "Thiếu một số cột tùy chọn: "
        + ", ".join(missing_recommended)
        + ". Hệ thống sẽ điền giá trị trung vị hoặc giá trị mặc định."
    ]


async def read_csv_upload(csv_file: UploadFile) -> tuple[list[dict[str, Any]], list[str]]:
    """Đọc file CSV tải lên, kiểm tra định dạng và trả về bản ghi hợp lệ."""
    if not csv_file.filename or not csv_file.filename.lower().endswith(".csv"):
        raise ValueError("Vui lòng tải lên file .csv.")
    try:
        records = pd.read_csv(BytesIO(await csv_file.read())).to_dict(orient="records")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("File CSV rỗng hoặc không có header.") from exc
    return records, validate_csv_records(records)


def save_batch_results(results: list[dict[str, Any]]) -> None:
    """Lưu kết quả phân tích hàng loạt mới nhất để người dùng tải xuống."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(LATEST_BATCH_RESULTS_FILE, index=False, encoding="utf-8-sig")


def safe_float(value: Any, default: float = np.nan) -> float:
    """Chuyển dữ liệu sang số thực và coi chuỗi rỗng là thiếu dữ liệu."""
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in {"nan", "none", "null"}:
                return default
        return float(value)
    except Exception:
        return default


def bool_to_int(value: Any, default: float = np.nan) -> float:
    """Chuyển giá trị boolean phổ biến thành 0 hoặc 1."""
    if value is None or pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "verified"}:
        return 1.0
    if text in {"0", "false", "no", "n", ""}:
        return 0.0
    return safe_float(value, default)


def get_account_age_days(data: dict[str, Any]) -> float:
    """Lấy tuổi tài khoản từ account_age_days hoặc created_at."""
    explicit_age = safe_float(data.get("account_age_days"))
    if not np.isnan(explicit_age):
        return max(explicit_age, 1.0)

    created_at = pd.to_datetime(data.get("created_at"), errors="coerce", utc=True)
    if pd.isna(created_at):
        return np.nan

    now = pd.Timestamp.now(tz="UTC").normalize()
    return float(max((now - created_at).days, 1))


def digit_ratio(value: Any) -> float:
    """Tính tỷ lệ ký tự số trong screen name."""
    if value is None or pd.isna(value):
        return np.nan
    text = str(value)
    if not text:
        return 0.0
    return sum(char.isdigit() for char in text) / len(text)


def text_value(value: Any) -> str:
    """Normalize missing text values to a lowercase string."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def has_spam_keyword(value: Any) -> float:
    """Return 1.0 when text contains a common spam/bot keyword."""
    text = text_value(value)
    return float(int(any(keyword in text for keyword in SPAM_KEYWORDS)))


def spam_keyword_count(*values: Any) -> float:
    """Count spam keyword occurrences across multiple text fields."""
    text = " ".join(text_value(value) for value in values)
    return float(sum(text.count(keyword) for keyword in SPAM_KEYWORDS))


def has_url(value: Any) -> float:
    """Return 1.0 when text looks like it contains a URL."""
    text = text_value(value)
    return float(int(any(marker in text for marker in URL_MARKERS)))


def build_demo_feature_row(data: dict[str, Any]) -> dict[str, float]:
    """Xây dựng một dòng đặc trưng từ form, API hoặc CSV."""
    data = normalize_row(data)
    features = {column: np.nan for column in FEATURE_COLUMNS}

    for column in FEATURE_COLUMNS:
        if column in data:
            features[column] = safe_float(data[column])

    followers = safe_float(data.get("followers_count"), features.get("followers_count", np.nan))
    friends = safe_float(data.get("friends_count"), features.get("friends_count", np.nan))
    statuses = safe_float(data.get("statuses_count"), features.get("statuses_count", np.nan))
    favourites = safe_float(data.get("favourites_count"), features.get("favourites_count", np.nan))
    age_days = get_account_age_days(data)

    features["followers_count"] = followers
    features["friends_count"] = friends
    features["statuses_count"] = statuses
    features["favourites_count"] = favourites
    features["listed_count"] = safe_float(data.get("listed_count"), features.get("listed_count", np.nan))
    features["verified"] = bool_to_int(data.get("verified"), features.get("verified", np.nan))
    features["default_profile"] = bool_to_int(
        data.get("default_profile"), features.get("default_profile", np.nan)
    )
    features["default_profile_image"] = bool_to_int(
        data.get("default_profile_image"), features.get("default_profile_image", np.nan)
    )
    features["geo_enabled"] = bool_to_int(
        data.get("geo_enabled"), features.get("geo_enabled", np.nan)
    )
    features["protected"] = bool_to_int(data.get("protected"), features.get("protected", np.nan))

    if not np.isnan(followers) and not np.isnan(friends):
        features["followers_friends_ratio"] = followers / (friends + 1)
        features["friends_followers_ratio"] = friends / (followers + 1)
        features["friends_followers_gap"] = friends - followers

    features["account_age_days"] = age_days
    if not np.isnan(statuses) and not np.isnan(age_days) and age_days > 0:
        features["tweets_per_day"] = statuses / age_days
    if not np.isnan(statuses) and not np.isnan(followers):
        features["statuses_followers_ratio"] = statuses / (followers + 1)
    if not np.isnan(statuses) and not np.isnan(friends):
        features["statuses_friends_ratio"] = statuses / (friends + 1)
    if not np.isnan(favourites) and not np.isnan(statuses):
        features["favourites_statuses_ratio"] = favourites / (statuses + 1)

    description = str(data.get("description", "") if data.get("description") is not None else "")
    name = str(data.get("name", "") if data.get("name") is not None else "")
    screen_name = str(data.get("screen_name", "") if data.get("screen_name") is not None else "")

    features["has_description"] = float(int(description.strip() != ""))
    features["name_length"] = float(len(name.strip()))
    features["screen_name_length"] = float(len(screen_name.strip()))
    features["description_length"] = float(len(description.strip()))
    features["screen_name_digit_ratio"] = digit_ratio(screen_name)
    features["screen_name_has_digits"] = float(int(features["screen_name_digit_ratio"] > 0))
    features["screen_name_has_spam_keyword"] = has_spam_keyword(screen_name)
    features["description_has_spam_keyword"] = has_spam_keyword(description)
    features["description_has_url"] = has_url(description)
    features["spam_keyword_count"] = spam_keyword_count(screen_name, name, description)

    has_profile_image = data.get("has_profile_image")
    if has_profile_image is not None:
        features["has_profile_image"] = bool_to_int(has_profile_image)
    elif not np.isnan(features.get("default_profile_image", np.nan)):
        features["has_profile_image"] = 1.0 - features["default_profile_image"]

    return features


def build_feature_matrix(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Xây dựng ma trận đặc trưng đúng thứ tự khi huấn luyện."""
    rows = [build_demo_feature_row(record) for record in records]
    matrix = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in matrix.columns:
            matrix[column] = np.nan
    return matrix[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def prepare_matrices_for_prediction(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Xây dựng và chuẩn bị ma trận đặc trưng cho Random Forest và Isolation Forest."""
    raw_matrix = build_feature_matrix(records)
    
    # 1. Log transform skewed count features
    matrix_log = raw_matrix.copy()
    skewed_cols = ["followers_count", "friends_count", "statuses_count", "favourites_count", "followers_friends_ratio"]
    for col in skewed_cols:
        if col in matrix_log.columns:
            matrix_log[col] = np.log1p(matrix_log[col].clip(lower=0))
            
    # 2. Chuẩn bị ma trận cho Random Forest (tất cả 20 đặc trưng)
    if IMPUTER is not None:
        values_rf = IMPUTER.transform(matrix_log)
        imputed_rf = pd.DataFrame(values_rf, columns=FEATURE_COLUMNS, index=raw_matrix.index)
    else:
        imputed_rf = matrix_log.fillna(FEATURE_MEDIANS)
        
    if SCALER is not None:
        values_rf = SCALER.transform(imputed_rf)
    else:
        values_rf = imputed_rf.to_numpy()
    matrix_rf = pd.DataFrame(values_rf, columns=FEATURE_COLUMNS, index=raw_matrix.index)
    
    # 3. Chuẩn bị ma trận cho Isolation Forest (7 đặc trưng liên tục cốt lõi)
    iso_features = ["followers_count", "friends_count", "followers_friends_ratio", "statuses_count", "favourites_count", "account_age_days", "tweets_per_day"]
    matrix_iso_raw = matrix_log[iso_features].copy()
    
    if IMPUTER_ISO is not None:
        values_iso = IMPUTER_ISO.transform(matrix_iso_raw)
        imputed_iso = pd.DataFrame(values_iso, columns=iso_features, index=raw_matrix.index)
    else:
        iso_medians = FEATURE_MEDIANS.reindex(iso_features).fillna(0.0)
        imputed_iso = matrix_iso_raw.fillna(iso_medians)
        
    if SCALER_ISO is not None:
        values_iso = SCALER_ISO.transform(imputed_iso)
    else:
        values_iso = imputed_iso.to_numpy()
    matrix_iso = pd.DataFrame(values_iso, columns=iso_features, index=raw_matrix.index)
    
    return matrix_rf, matrix_iso


def prepare_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Điền dữ liệu thiếu và chuẩn hóa ma trận đặc trưng (giữ lại để tương thích ngược)."""
    if IMPUTER is not None:
        values = IMPUTER.transform(matrix)
        imputed = pd.DataFrame(values, columns=FEATURE_COLUMNS, index=matrix.index)
    else:
        imputed = matrix.fillna(FEATURE_MEDIANS)
    if SCALER is not None:
        values = SCALER.transform(imputed)
    else:
        values = imputed.to_numpy()
    return pd.DataFrame(values, columns=FEATURE_COLUMNS, index=matrix.index)


def ensure_models() -> None:
    """Kiểm tra các mô hình cần thiết trước khi dự đoán."""
    if ISOLATION_FOREST is None:
        raise HTTPException(status_code=503, detail="Không tìm thấy mô hình Isolation Forest.")
    if RANDOM_FOREST is None:
        raise HTTPException(status_code=503, detail="Không tìm thấy mô hình Random Forest.")


def risk_level(is_anomaly: bool, is_bot: bool) -> tuple[str, str]:
    """Tổng hợp hai đầu ra thành mức rủi ro dễ giải thích."""
    if is_anomaly and is_bot:
        return "Rủi ro cao", "Cả hai mô hình đều cảnh báo tài khoản này."
    if is_anomaly and not is_bot:
        return "Cần kiểm tra", "Hành vi bất thường nhưng bộ phân loại chưa xác định là bot."
    if not is_anomaly and is_bot:
        return "Cần kiểm tra", "Có dấu hiệu bot nhưng hành vi chưa nằm ngoài vùng bình thường."
    return "Rủi ro thấp", "Cả hai mô hình đều đánh giá tài khoản có dấu hiệu bình thường."


def bot_probability_threshold(record: dict[str, Any], is_anomaly: bool) -> float:
    """Use a lower review threshold for suspicious accounts to reduce missed bots."""
    normalized = normalize_row(record)
    text = " ".join(
        text_value(normalized.get(column, ""))
        for column in ("screen_name", "name", "description")
    )
    has_keyword = any(keyword in text for keyword in SPAM_KEYWORDS)
    has_link = any(marker in text for marker in URL_MARKERS)
    if is_anomaly or has_keyword or has_link:
        return BOT_REVIEW_THRESHOLD
    return BOT_DEFAULT_THRESHOLD


def predict_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chạy Isolation Forest và Random Forest cho một hoặc nhiều tài khoản."""
    ensure_models()
    matrix_rf, matrix_iso = prepare_matrices_for_prediction(records)

    anomaly_raw = ISOLATION_FOREST.predict(matrix_iso)
    anomaly_scores = -ISOLATION_FOREST.decision_function(matrix_iso)
    probabilities = RANDOM_FOREST.predict_proba(matrix_rf)

    output = []
    for index, record in enumerate(records):
        is_anomaly = bool(anomaly_raw[index] == -1)
        bot_probability = float(probabilities[index, 1])
        threshold = bot_probability_threshold(record, is_anomaly)
        is_bot = bool(bot_probability >= threshold)
        level, explanation = risk_level(is_anomaly, is_bot)
        normalized = normalize_row(record)
        classifier_confidence = bot_probability if is_bot else 1.0 - bot_probability
        output.append(
            {
                "row": index + 1,
                "screen_name": normalized.get("screen_name", ""),
                "is_anomaly": is_anomaly,
                "anomaly_result": "Bất thường" if is_anomaly else "Bình thường",
                "anomaly_score": round(float(anomaly_scores[index]), 6),
                "bot_prediction": "Bot" if is_bot else "Người thật",
                "bot_probability": round(bot_probability * 100, 2),
                "classifier_confidence": round(classifier_confidence * 100, 2),
                "bot_threshold": round(threshold * 100, 2),
                "risk_level": level,
                "risk_explanation": explanation,
            }
        )
    return output


def summarize_batch(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Tổng hợp kết quả dự đoán CSV."""
    total = len(results)
    anomaly_count = sum(1 for row in results if row["is_anomaly"])
    bot_count = sum(1 for row in results if row["bot_prediction"] == "Bot")
    high_risk_count = sum(1 for row in results if row["risk_level"] == "Rủi ro cao")
    return {
        "total": total,
        "anomaly_count": anomaly_count,
        "bot_count": bot_count,
        "high_risk_count": high_risk_count,
        "anomaly_rate": round((anomaly_count / total) * 100, 2) if total else 0.0,
        "bot_rate": round((bot_count / total) * 100, 2) if total else 0.0,
    }


def record_tested_accounts(results: list[dict[str, Any]]) -> None:
    """Ghi nhận kết quả các tài khoản đã kiểm tra vào file lưu trữ lâu dài (không trùng lặp)."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    new_rows = []
    for r in results:
        new_rows.append({
            "screen_name": r.get("screen_name", ""),
            "anomaly_result": r.get("anomaly_result", ""),
            "bot_prediction": r.get("bot_prediction", ""),
            "risk_level": r.get("risk_level", ""),
        })
    df_new = pd.DataFrame(new_rows)
    
    if TESTED_ACCOUNTS_FILE.exists():
        try:
            df_old = pd.read_csv(TESTED_ACCOUNTS_FILE)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new
        
    if not df_combined.empty:
        # Loại bỏ trùng lặp dựa trên screen_name (không phân biệt hoa thường), giữ lại kết quả kiểm tra mới nhất
        df_combined["_sn_lower"] = df_combined["screen_name"].astype(str).str.lower().str.strip()
        df_combined = df_combined.drop_duplicates(subset=["_sn_lower"], keep="last").drop(columns=["_sn_lower"])
        
    df_combined.to_csv(TESTED_ACCOUNTS_FILE, index=False, encoding="utf-8-sig")


def get_tested_summary() -> dict[str, Any]:
    """Tổng hợp số liệu thống kê từ các tài khoản người dùng đã kiểm tra."""
    summary = {
        "total": 0,
        "human": 0,
        "bot": 0,
        "anomaly": 0,
        "normal": 0,
        "human_rate": 0.0,
        "bot_rate": 0.0,
        "anomaly_rate": 0.0,
        "normal_rate": 0.0,
    }
    if not TESTED_ACCOUNTS_FILE.exists():
        return summary
        
    try:
        df = pd.read_csv(TESTED_ACCOUNTS_FILE)
        total = len(df)
        if total == 0:
            return summary
            
        bot_count = int((df["bot_prediction"] == "Bot").sum())
        human_count = total - bot_count
        anomaly_count = int((df["anomaly_result"] == "Bất thường").sum())
        normal_count = total - anomaly_count
        
        summary.update({
            "total": total,
            "human": human_count,
            "bot": bot_count,
            "anomaly": anomaly_count,
            "normal": normal_count,
            "human_rate": round(human_count / total * 100, 2) if total else 0.0,
            "bot_rate": round(bot_count / total * 100, 2) if total else 0.0,
            "anomaly_rate": round(anomaly_count / total * 100, 2) if total else 0.0,
            "normal_rate": round(normal_count / total * 100, 2) if total else 0.0,
        })
    except Exception:
        pass
    return summary


def demo_context(
    *,
    prediction: Optional[dict[str, Any]] = None,
    batch_results: Optional[list[dict[str, Any]]] = None,
    batch_summary: Optional[dict[str, Any]] = None,
    upload_warnings: Optional[list[str]] = None,
    error: Optional[str] = None,
    active_tab: Optional[str] = None,
    active_view: Optional[str] = None,
    form_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Tạo context chung cho giao diện."""
    return {
        "prediction": prediction,
        "batch_results": batch_results,
        "batch_summary": batch_summary,
        "upload_warnings": upload_warnings or [],
        "can_download_results": LATEST_BATCH_RESULTS_FILE.exists(),
        "dataset_summary": get_tested_summary(),
        "metrics": MODEL_METRICS,
        "warnings": startup_warnings(),
        "error": error,
        "active_tab": active_tab or "quick",
        "active_view": active_view or "analysis-view",
        "form_data": form_data or {},
    }


class AccountInput(BaseModel):
    """Schema API để dự đoán một tài khoản."""

    screen_name: str = "free_airdrop_999"
    name: str = "Khuyến mãi"
    description: str = "Nhận quà miễn phí ngay hôm nay."
    followers_count: float = 12
    friends_count: float = 2500
    statuses_count: float = 8400
    favourites_count: float = 5
    listed_count: float = 0
    verified: int = 0
    default_profile: int = 1
    default_profile_image: int = 0
    geo_enabled: int = 0
    protected: int = 0
    account_age_days: float = 12


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Hiển thị giao diện demo."""
    return templates.TemplateResponse(request=request, name="index.html", context=demo_context())


@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    scan_mode: str = Form("quick"),
    screen_name: str = Form(""),
    name: str = Form(""),
    description: str = Form(""),
    followers_count: Optional[float] = Form(None),
    friends_count: Optional[float] = Form(None),
    statuses_count: Optional[float] = Form(None),
    favourites_count: Optional[float] = Form(None),
    listed_count: Optional[float] = Form(None),
    verified: Optional[int] = Form(None),
    default_profile: Optional[int] = Form(None),
    default_profile_image: Optional[int] = Form(None),
    geo_enabled: Optional[int] = Form(None),
    protected: Optional[int] = Form(None),
    account_age_days: Optional[float] = Form(None),
):
    """Dự đoán một tài khoản từ form."""
    record = {
        "screen_name": screen_name,
        "name": name,
        "description": description,
    }
    
    if scan_mode == "quick":
        if followers_count is not None:
            record["followers_count"] = followers_count
        if friends_count is not None:
            record["friends_count"] = friends_count
    else:
        record.update({
            "followers_count": followers_count if followers_count is not None else 0.0,
            "friends_count": friends_count if friends_count is not None else 0.0,
            "statuses_count": statuses_count if statuses_count is not None else 0.0,
            "favourites_count": favourites_count if favourites_count is not None else 0.0,
            "listed_count": listed_count if listed_count is not None else 0.0,
            "verified": verified if verified is not None else 0,
            "default_profile": default_profile if default_profile is not None else 0,
            "default_profile_image": default_profile_image if default_profile_image is not None else 0,
            "geo_enabled": geo_enabled if geo_enabled is not None else 0,
            "protected": protected if protected is not None else 0,
            "account_age_days": account_age_days if account_age_days is not None else 1.0,
        })
        
    try:
        prediction = predict_records([record])[0]
        record_tested_accounts([prediction])
        error = None
    except Exception as exc:
        prediction = None
        error = str(exc)
        
    form_data = {
        "scan_mode": scan_mode,
        "screen_name": screen_name,
        "name": name,
        "description": description,
        "followers_count": followers_count,
        "friends_count": friends_count,
        "statuses_count": statuses_count,
        "favourites_count": favourites_count,
        "listed_count": listed_count,
        "verified": verified,
        "default_profile": default_profile,
        "default_profile_image": default_profile_image,
        "geo_enabled": geo_enabled,
        "protected": protected,
        "account_age_days": account_age_days,
    }
        
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=demo_context(
            prediction=prediction,
            error=error,
            active_tab=scan_mode,
            form_data=form_data,
        ),
    )


@app.post("/upload-csv", response_class=HTMLResponse)
async def upload_csv(request: Request, csv_file: UploadFile = File(...)):
    """Dự đoán nhiều tài khoản từ CSV."""
    try:
        records, upload_warnings = await read_csv_upload(csv_file)
        results = predict_records(records)
        record_tested_accounts(results)
        save_batch_results(results)
        error = None
    except Exception as exc:
        results = []
        upload_warnings = []
        error = str(exc)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=demo_context(
            batch_results=results[:100],
            batch_summary=summarize_batch(results) if results else None,
            upload_warnings=upload_warnings,
            error=error,
        ),
    )


@app.post("/clear-stats", response_class=HTMLResponse)
async def clear_stats(request: Request):
    """Xóa lịch sử tài khoản đã kiểm tra."""
    if TESTED_ACCOUNTS_FILE.exists():
        try:
            TESTED_ACCOUNTS_FILE.unlink()
        except Exception:
            pass
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=demo_context(active_view="stats-view"),
    )


@app.get("/download-sample-csv")
async def download_sample_csv():
    """Tải file CSV mẫu để phân tích danh sách tài khoản."""
    if not SAMPLE_ACCOUNTS_FILE.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file CSV mẫu.")
    return FileResponse(
        SAMPLE_ACCOUNTS_FILE,
        media_type="text/csv",
        filename="sample_accounts.csv",
    )


@app.get("/download-results")
async def download_results():
    """Tải kết quả phân tích CSV gần nhất."""
    if not LATEST_BATCH_RESULTS_FILE.exists():
        raise HTTPException(status_code=404, detail="Chưa có kết quả phân tích hàng loạt.")
    return FileResponse(
        LATEST_BATCH_RESULTS_FILE,
        media_type="text/csv",
        filename="bot_detection_results.csv",
    )


@app.get("/api/models")
async def api_models():
    """Trả về vai trò và trạng thái của hai mô hình."""
    return {
        "anomaly_detector": {
            "model": "isolation_forest",
            "role": "phát hiện tài khoản bất thường",
            "available": ISOLATION_FOREST is not None,
        },
        "bot_classifier": {
            "model": "random_forest",
            "role": "phân loại tài khoản người thật hoặc bot",
            "available": RANDOM_FOREST is not None,
        },
        "warnings": startup_warnings(),
    }


@app.post("/api/predict")
async def api_predict(account: AccountInput):
    """API dự đoán một tài khoản."""
    data = account.model_dump() if hasattr(account, "model_dump") else account.dict()
    return predict_records([data])[0]


@app.post("/api/upload-csv")
async def api_upload_csv(csv_file: UploadFile = File(...)):
    """API dự đoán nhiều tài khoản từ CSV."""
    try:
        records, upload_warnings = await read_csv_upload(csv_file)
        results = predict_records(records)
        save_batch_results(results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "summary": summarize_batch(results),
        "warnings": upload_warnings,
        "download_url": "/download-results",
        "rows": results,
    }
