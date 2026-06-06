"""Tải và gộp dữ liệu hồ sơ tài khoản từ bộ Cresci-2017."""

from pathlib import Path
import warnings

import pandas as pd


SEED = 42

DATASET_FOLDERS = (
    "genuine_accounts",
    "social_spambots_1",
    "social_spambots_2",
    "social_spambots_3",
    "traditional_spambots_1",
    "traditional_spambots_4",
)

LABELS = {
    "genuine_accounts": 0,
    "social_spambots_1": 1,
    "social_spambots_2": 1,
    "social_spambots_3": 1,
    "traditional_spambots_1": 1,
    "traditional_spambots_4": 1,
}


def _candidate_user_files(data_dir: Path, folder_name: str) -> list[Path]:
    """Trả về các vị trí có thể chứa file users.csv."""
    return [
        data_dir / folder_name / "users.csv",
        data_dir / f"{folder_name}.csv" / "users.csv",
        data_dir / f"{folder_name}.csv",
    ]


def load_cresci2017(data_dir: str | Path) -> pd.DataFrame:
    """Tải và gộp users.csv của Cresci-2017 với nhãn bot/người thật.

    Parameters
    ----------
    data_dir:
        Directory containing Cresci-2017 subset folders. Expected subset names
        are genuine_accounts, social_spambots_1, social_spambots_2,
        social_spambots_3, and traditional_spambots_1.

    Returns
    -------
    pandas.DataFrame
        Merged user-profile table with label and source columns. label=0 means
        human/genuine account, and label=1 means bot/spambot account.
    """
    data_path = Path(data_dir)
    frames: list[pd.DataFrame] = []

    print(f"[dữ liệu] Đang tìm Cresci-2017 tại: {data_path.resolve()}")
    if not data_path.exists():
        warnings.warn(f"Không tồn tại thư mục dữ liệu: {data_path}", stacklevel=2)
        return pd.DataFrame(columns=["label", "source"])

    for folder_name in DATASET_FOLDERS:
        users_file = next(
            (path for path in _candidate_user_files(data_path, folder_name) if path.exists()),
            None,
        )

        if users_file is None:
            warnings.warn(
                f"Thiếu users.csv của nhóm '{folder_name}' trong {data_path}",
                stacklevel=2,
            )
            continue

        try:
            subset = pd.read_csv(users_file, low_memory=False)
        except Exception as exc:  # pragma: no cover - defensive file handling
            warnings.warn(f"Không thể đọc {users_file}: {exc}", stacklevel=2)
            continue

        subset["label"] = LABELS[folder_name]
        subset["source"] = folder_name
        frames.append(subset)
        print(
            f"[dữ liệu] Đã tải {folder_name}: {subset.shape[0]:,} dòng, "
            f"{subset.shape[1]:,} cột"
        )

    if not frames:
        warnings.warn("Không tải được file users.csv nào của Cresci-2017.", stacklevel=2)
        return pd.DataFrame(columns=["label", "source"])

    merged = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[dữ liệu] Kích thước sau khi gộp: {merged.shape[0]:,} dòng x {merged.shape[1]:,} cột")
    print("[dữ liệu] Phân phối nhãn:")
    print(merged["label"].value_counts(dropna=False).rename(index={0: "người thật", 1: "bot"}))

    return merged
