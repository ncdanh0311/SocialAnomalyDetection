# Hệ thống phát hiện tài khoản bất thường và phân loại bot

Đề tài xây dựng hệ thống phân tích tài khoản Twitter bằng bộ dữ liệu Cresci-2017. Hệ thống sử dụng **Isolation Forest** để phát hiện tài khoản có hồ sơ và mức độ hoạt động bất thường, đồng thời sử dụng **Random Forest** để phân loại tài khoản người thật hoặc bot. Hai kết quả được hiển thị song song để hỗ trợ đánh giá rủi ro.

## Bộ dữ liệu

Cresci-2017 là bộ dữ liệu nghiên cứu về tài khoản Twitter người thật và bot. Tải dữ liệu tại:

https://botometer.osome.iu.edu/bot-repository/datasets.html

Đặt các thư mục sau trong `data/raw/`:

- `genuine_accounts/`
- `social_spambots_1/`
- `social_spambots_2/`
- `social_spambots_3/`
- `traditional_spambots_1/`

Project sử dụng `users.csv` để xây dựng đặc trưng hồ sơ và hoạt động tổng hợp. Các file `tweets.csv` không thuộc phạm vi triển khai hiện tại nên không cần lưu trong project.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy notebook

Chạy lần lượt:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_modeling.ipynb`
4. `notebooks/04_evaluation.ipynb`

Mỗi notebook đều có cell cấu hình đường dẫn dành cho máy local và Google Colab. Khi dùng Colab, đặt project tại:

```text
/content/drive/MyDrive/bot-detection-project
```

## Giao diện web demo

```bash
uvicorn src.main:app --reload
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8000
```

## Phân tích CSV

Giao diện cho phép tải file mẫu và tải kết quả sau khi phân tích hàng loạt. File CSV đầu vào cần có các cột bắt buộc:

```text
screen_name, followers_count, friends_count, statuses_count
```

Ngoài ra cần có một trong hai cột `account_age_days` hoặc `created_at`. File mẫu đầy đủ nằm tại `data/sample_accounts.csv`.

## Chạy test API

```bash
pytest -q
```

## Vai trò mô hình

| Mô hình | Vai trò | Cách huấn luyện |
|---|---|---|
| Isolation Forest | Phát hiện tài khoản bất thường | Chỉ fit trên tài khoản người thật thuộc tập train |
| Random Forest | Phân loại người thật hoặc bot | Fit trên toàn bộ tập train có nhãn |

## Kết quả

Kết quả mới nhất được lưu tại `outputs/models/metrics.json` và hiển thị trong `04_evaluation.ipynb`. Các biểu đồ dễ trình bày được lưu tại `outputs/figures/`:

| Mô hình | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 0.9163 | 0.8944 | 0.9922 | 0.9407 | 0.9603 |
| Random Forest | 0.9862 | 0.9943 | 0.9851 | 0.9897 | 0.9966 |

Isolation Forest sử dụng `contamination=0.25` (được chọn tự động trên tập validation) kết hợp log-transform trên các cột đếm, mang lại hiệu năng tối ưu vượt trội. Random Forest là bộ phân loại bot chính.

- `class_dist.png`: Phân bố lớp dữ liệu (Bot vs Người thật) trong tập dữ liệu.
- `followers_friends_scatter.png`: Biểu đồ phân tán so sánh lượng followers và friends (thang đo log), trực quan hóa sự khác biệt rõ rệt về hành vi kết nối của bot (thường theo dõi rất nhiều nhưng có ít người theo dõi lại) so với người thật.
- `anomaly_score_dist.png`: Phân phối điểm bất thường (Anomaly Score) của Isolation Forest đối với bot và người thật. Biểu đồ này giải thích trực quan lý do lựa chọn ngưỡng `contamination = 0.25` (điểm phân cắt tối ưu trên tập validation) và chỉ ra mức độ phân tách rõ rệt giữa hai phân phối sau khi áp dụng log-transform.
- `model_metrics_comparison.png`: So sánh trực quan các chỉ số hiệu năng (Accuracy, Precision, Recall, F1, AUC) của các mô hình.
- `confusion_matrices.png`: Ma trận nhầm lẫn của Isolation Forest và Random Forest trên tập test.
- `roc_curve.png`: Đường cong ROC biểu diễn khả năng phân tách của các mô hình.
- `feature_importance.png`: Các đặc trưng quan trọng đóng góp nhiều nhất vào quyết định của Random Forest.
