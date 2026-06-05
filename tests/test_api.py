"""Smoke tests for the FastAPI bot-detection demo."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = PROJECT_ROOT / "data" / "sample_accounts.csv"

client = TestClient(app)


def test_home_page_loads() -> None:
    """Render the demo home page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Phân tích bot" in response.text


def test_predict_single_account() -> None:
    """Return anomaly and classifier outputs for one account."""
    response = client.post(
        "/api/predict",
        json={
            "screen_name": "free_airdrop_999",
            "followers_count": 12,
            "friends_count": 2500,
            "statuses_count": 8400,
            "account_age_days": 12,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["anomaly_result"] in {"Bất thường", "Bình thường"}
    assert payload["bot_prediction"] in {"Bot", "Người thật"}


def test_upload_csv_rejects_missing_required_columns() -> None:
    """Reject a batch file when core behavioral fields are missing."""
    response = client.post(
        "/api/upload-csv",
        files={"csv_file": ("invalid.csv", b"screen_name,followers_count\nbot_1,12\n", "text/csv")},
    )
    assert response.status_code == 400
    assert "thiếu cột bắt buộc" in response.json()["detail"]


def test_upload_csv_and_download_results() -> None:
    """Analyze the sample batch and expose its downloadable result file."""
    with SAMPLE_FILE.open("rb") as file:
        response = client.post(
            "/api/upload-csv",
            files={"csv_file": ("sample_accounts.csv", file, "text/csv")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total"] == 3
    assert payload["download_url"] == "/download-results"

    download = client.get("/download-results")
    assert download.status_code == 200
    assert "screen_name" in download.text


def test_download_sample_csv() -> None:
    """Expose a CSV template for batch analysis."""
    response = client.get("/download-sample-csv")
    assert response.status_code == 200
    assert "followers_count" in response.text


def test_predict_form_quick_scan() -> None:
    """Submit the HTML form with quick scan mode."""
    response = client.post(
        "/predict-form",
        data={
            "scan_mode": "quick",
            "screen_name": "free_airdrop_999",
            "name": "Khuyến mãi",
            "description": "Nhận quà miễn phí ngay hôm nay.",
            "followers_count": 12,
            "friends_count": 2500,
        },
    )
    assert response.status_code == 200
    assert "Kết quả phân tích" in response.text
    assert "Bất thường" in response.text or "Bình thường" in response.text
    assert "Bot" in response.text or "Người thật" in response.text


def test_predict_form_advanced_scan() -> None:
    """Submit the HTML form with advanced scan mode."""
    response = client.post(
        "/predict-form",
        data={
            "scan_mode": "advanced",
            "screen_name": "genuine_user_123",
            "name": "Genuine User",
            "description": "Just a normal user.",
            "followers_count": 150,
            "friends_count": 100,
            "statuses_count": 1200,
            "favourites_count": 50,
            "listed_count": 2,
            "verified": 1,
            "default_profile": 0,
            "default_profile_image": 0,
            "geo_enabled": 1,
            "protected": 0,
            "account_age_days": 365,
        },
    )
    assert response.status_code == 200
    assert "Kết quả phân tích" in response.text

