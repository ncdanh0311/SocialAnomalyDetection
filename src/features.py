"""Xây dựng đặc trưng hồ sơ tài khoản Twitter từ bộ dữ liệu Cresci-2017."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


SEED = 42
REFERENCE_DATE = pd.Timestamp("2015-05-01", tz="UTC")
SPAM_KEYWORDS = (
    "free",
    "airdrop",
    "bonus",
    "crypto",
    "bitcoin",
    "coin",
    "token",
    "nft",
    "giveaway",
    "click",
    "link",
    "claim",
    "followback",
    "follow",
    "bot",
    "spam",
    "auto",
    "win",
    "winner",
    "money",
    "cash",
    "prize",
    "reward",
    "promo",
    "deal",
    "offer",
    "now",
    "discount",
    "earn",
    "loan",
    "forex",
    "profit",
    "gift",
)
URL_MARKERS = ("http://", "https://", "www.", "bit.ly", "t.co/")

HIGH_CARDINALITY_COLUMNS = {
    "id",
    "id_str",
    "name",
    "screen_name",
    "created_at",
    "url",
    "description",
    "location",
    "time_zone",
    "lang",
    "profile_background_image_url",
    "profile_background_image_url_https",
    "profile_background_tile",
    "profile_banner_url",
    "profile_image_url",
    "profile_image_url_https",
    "profile_link_color",
    "profile_sidebar_border_color",
    "profile_sidebar_fill_color",
    "profile_text_color",
    "status",
    "source",
    "timestamp",
    "crawled_at",
    "updated",
    "test_set_1",
    "test_set_2",
}


def _warn_missing(column_names: list[str], feature_name: str) -> None:
    """Cảnh báo khi không thể tạo đặc trưng do thiếu cột đầu vào."""
    missing = ", ".join(column_names)
    warnings.warn(
        f"Không thể tạo '{feature_name}' vì thiếu cột: {missing}",
        stacklevel=3,
    )


def _to_int_flag(series: pd.Series) -> pd.Series:
    """Chuyển giá trị boolean phổ biến thành số nguyên 0/1."""
    if series.dtype == bool:
        return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f", "nan", "none", ""}
    return normalized.map(
        lambda value: 1 if value in truthy else 0 if value in falsy else np.nan
    )


def _screen_name_digit_ratio(screen_name: object) -> float:
    """Tính tỷ lệ ký tự số trong screen name."""
    if pd.isna(screen_name):
        return np.nan

    value = str(screen_name)
    if not value:
        return 0.0

    return sum(char.isdigit() for char in value) / len(value)


def _text_value(value: object) -> str:
    """Normalize missing text values to a lowercase string."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _has_spam_keyword(value: object) -> int:
    """Return 1 when text contains at least one common spam/bot keyword."""
    text = _text_value(value)
    return int(any(keyword in text for keyword in SPAM_KEYWORDS))


def _spam_keyword_count(value: object) -> int:
    """Count keyword occurrences in a combined text field."""
    text = _text_value(value)
    return int(sum(text.count(keyword) for keyword in SPAM_KEYWORDS))


def _has_url(value: object) -> int:
    """Return 1 when text looks like it contains a URL."""
    text = _text_value(value)
    return int(any(marker in text for marker in URL_MARKERS))


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Xây dựng đặc trưng số từ hồ sơ tài khoản Cresci-2017.

    Hàm tạo đặc trưng tỷ lệ, mức độ hoạt động, tuổi tài khoản, độ đầy đủ hồ sơ
    và screen name; loại bỏ cột định danh, văn bản và metadata; sau đó chỉ giữ
    cột số. Dữ liệu thiếu được xử lý sau khi chia train/test để tránh leakage.

    Parameters
    ----------
    df:
        Raw merged user-profile DataFrame returned by load_cresci2017.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Numeric feature matrix X and label vector y.
    """
    print("[đặc trưng] Bắt đầu xây dựng đặc trưng...")

    if df is None or df.empty:
        warnings.warn("DataFrame đầu vào rỗng. Trả về X và y rỗng.", stacklevel=2)
        return pd.DataFrame(), pd.Series(dtype="int64", name="label")

    work = df.copy()

    if "label" in work.columns:
        y = pd.to_numeric(work["label"], errors="coerce").fillna(0).astype(int)
        y.name = "label"
    else:
        warnings.warn("Thiếu cột label. Trả về vector nhãn rỗng.", stacklevel=2)
        y = pd.Series(dtype="int64", name="label")

    if {"followers_count", "friends_count"}.issubset(work.columns):
        followers = pd.to_numeric(work["followers_count"], errors="coerce")
        friends = pd.to_numeric(work["friends_count"], errors="coerce")
        work["followers_friends_ratio"] = followers / (friends + 1)
        work["friends_followers_ratio"] = friends / (followers + 1)
        work["friends_followers_gap"] = friends - followers
        print("[đặc trưng] Đã tạo followers_friends_ratio")
    else:
        _warn_missing(["followers_count", "friends_count"], "followers_friends_ratio")
        work["followers_friends_ratio"] = np.nan
        work["friends_followers_ratio"] = np.nan
        work["friends_followers_gap"] = np.nan

    if "created_at" in work.columns:
        created_at = pd.to_datetime(
            work["created_at"],
            format="%a %b %d %H:%M:%S %z %Y",
            errors="coerce",
            utc=True,
        )
        fallback_mask = created_at.isna() & work["created_at"].notna()
        if fallback_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                created_at.loc[fallback_mask] = pd.to_datetime(
                    work.loc[fallback_mask, "created_at"],
                    errors="coerce",
                    utc=True,
                )
        if "crawled_at" in work.columns:
            crawled_at = pd.to_datetime(work["crawled_at"], errors="coerce", utc=True)
            reference_date = crawled_at.fillna(REFERENCE_DATE)
        else:
            reference_date = REFERENCE_DATE
        account_age_days = (reference_date - created_at).dt.days
        work["account_age_days"] = account_age_days.clip(lower=1)
        print("[đặc trưng] Đã tạo account_age_days")
    else:
        _warn_missing(["created_at"], "account_age_days")
        work["account_age_days"] = np.nan

    if {"statuses_count", "account_age_days"}.issubset(work.columns):
        statuses = pd.to_numeric(work["statuses_count"], errors="coerce")
        age_days = pd.to_numeric(work["account_age_days"], errors="coerce")
        work["tweets_per_day"] = statuses / age_days.replace(0, np.nan)
        print("[đặc trưng] Đã tạo tweets_per_day")
    else:
        _warn_missing(["statuses_count", "account_age_days"], "tweets_per_day")
        work["tweets_per_day"] = np.nan

    if {"statuses_count", "followers_count", "friends_count"}.issubset(work.columns):
        statuses = pd.to_numeric(work["statuses_count"], errors="coerce")
        followers = pd.to_numeric(work["followers_count"], errors="coerce")
        friends = pd.to_numeric(work["friends_count"], errors="coerce")
        work["statuses_followers_ratio"] = statuses / (followers + 1)
        work["statuses_friends_ratio"] = statuses / (friends + 1)
        print("[features] Created statuses_followers_ratio and statuses_friends_ratio")
    else:
        _warn_missing(
            ["statuses_count", "followers_count", "friends_count"],
            "statuses_followers_ratio/statuses_friends_ratio",
        )
        work["statuses_followers_ratio"] = np.nan
        work["statuses_friends_ratio"] = np.nan

    if {"favourites_count", "statuses_count"}.issubset(work.columns):
        favourites = pd.to_numeric(work["favourites_count"], errors="coerce")
        statuses = pd.to_numeric(work["statuses_count"], errors="coerce")
        work["favourites_statuses_ratio"] = favourites / (statuses + 1)
        print("[features] Created favourites_statuses_ratio")
    else:
        _warn_missing(["favourites_count", "statuses_count"], "favourites_statuses_ratio")
        work["favourites_statuses_ratio"] = np.nan

    if "has_profile_image" in work.columns:
        work["has_profile_image"] = _to_int_flag(work["has_profile_image"])
    elif "default_profile_image" in work.columns:
        work["has_profile_image"] = 1 - _to_int_flag(work["default_profile_image"])
    elif "profile_image_url" in work.columns:
        work["has_profile_image"] = work["profile_image_url"].notna().astype(int)
    elif "profile_image_url_https" in work.columns:
        work["has_profile_image"] = work["profile_image_url_https"].notna().astype(int)
    else:
        _warn_missing(["has_profile_image/default_profile_image/profile_image_url"], "has_profile_image")
        work["has_profile_image"] = np.nan
    print("[đặc trưng] Đã tạo has_profile_image")

    if "description" in work.columns:
        description_text = work["description"].fillna("").astype(str)
        work["has_description"] = (description_text.str.strip().str.len() > 0).astype(int)
        work["description_length"] = description_text.str.strip().str.len()
        work["description_has_spam_keyword"] = description_text.apply(_has_spam_keyword)
        work["description_has_url"] = description_text.apply(_has_url)
    elif "has_description" in work.columns:
        work["has_description"] = _to_int_flag(work["has_description"])
        work["description_length"] = np.nan
        work["description_has_spam_keyword"] = np.nan
        work["description_has_url"] = np.nan
    else:
        _warn_missing(["description"], "has_description")
        work["has_description"] = np.nan
        work["description_length"] = np.nan
        work["description_has_spam_keyword"] = np.nan
        work["description_has_url"] = np.nan
    print("[đặc trưng] Đã tạo has_description")

    if "name" in work.columns:
        work["name_length"] = work["name"].fillna("").astype(str).str.len()
        print("[đặc trưng] Đã tạo name_length")
    else:
        _warn_missing(["name"], "name_length")
        work["name_length"] = np.nan

    if "screen_name" in work.columns:
        screen_name_text = work["screen_name"].fillna("").astype(str)
        work["screen_name_length"] = screen_name_text.str.strip().str.len()
        work["screen_name_digit_ratio"] = work["screen_name"].apply(_screen_name_digit_ratio)
        work["screen_name_has_digits"] = (work["screen_name_digit_ratio"] > 0).astype(int)
        work["screen_name_has_spam_keyword"] = screen_name_text.apply(_has_spam_keyword)
        print("[đặc trưng] Đã tạo screen_name_digit_ratio và screen_name_has_digits")
    else:
        _warn_missing(["screen_name"], "screen_name_digit_ratio")
        work["screen_name_length"] = np.nan
        work["screen_name_digit_ratio"] = np.nan
        work["screen_name_has_digits"] = np.nan
        work["screen_name_has_spam_keyword"] = np.nan

    empty_text = pd.Series("", index=work.index)
    screen_name_text = (
        work["screen_name"].fillna("").astype(str) if "screen_name" in work.columns else empty_text
    )
    name_text = work["name"].fillna("").astype(str) if "name" in work.columns else empty_text
    description_text = (
        work["description"].fillna("").astype(str) if "description" in work.columns else empty_text
    )
    combined_text = screen_name_text + " " + name_text + " " + description_text
    work["spam_keyword_count"] = combined_text.apply(_spam_keyword_count)
    print("[features] Created spam_keyword_count")

    drop_columns = [column for column in HIGH_CARDINALITY_COLUMNS if column in work.columns]
    work = work.drop(columns=drop_columns, errors="ignore")
    print(f"[đặc trưng] Đã loại {len(drop_columns)} cột định danh, văn bản hoặc metadata")

    numeric = work.select_dtypes(include=[np.number, "bool"]).copy()
    if "label" in numeric.columns:
        numeric = numeric.drop(columns=["label"])

    bool_columns = numeric.select_dtypes(include=["bool"]).columns
    if len(bool_columns) > 0:
        numeric[bool_columns] = numeric[bool_columns].astype(int)

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    empty_columns = numeric.columns[numeric.isna().all()].tolist()
    if empty_columns:
        warnings.warn(
            f"Loại cột hoàn toàn rỗng: {', '.join(empty_columns)}",
            stacklevel=2,
        )
    X = numeric.drop(columns=empty_columns)

    print(f"[đặc trưng] Ma trận cuối: {X.shape[0]:,} dòng x {X.shape[1]:,} cột")
    print(f"[đặc trưng] Số giá trị thiếu cần xử lý sau khi chia train/test: {int(X.isna().sum().sum()):,}")
    return X, y
