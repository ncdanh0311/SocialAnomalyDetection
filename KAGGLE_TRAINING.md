# Kaggle training

Use this when you want to retrain the models with the new bot/spam features.

## 1. Upload inputs

In Kaggle, add:

- this project folder
- the Cresci-2017 dataset folder

The `--data-dir` path must point to the folder that contains folders such as:

```text
genuine_accounts/
social_spambots_1/
social_spambots_2/
social_spambots_3/
traditional_spambots_1/
traditional_spambots_4/
```

## 2. Run training

Example Kaggle notebook cells:

```python
%cd /kaggle/working/AnomalyDetection-main
!pip install -q -r requirements.txt
```

```python
!python scripts/train_kaggle.py --data-dir /kaggle/input/cresci-2017
```

If your dataset is nested, point to the exact raw folder, for example:

```python
!python scripts/train_kaggle.py --data-dir /kaggle/input/cresci-2017/data/raw
```

The script auto-saves outputs to `/kaggle/working`:

```text
/kaggle/working/data/processed/features.csv
/kaggle/working/data/processed/train_data.csv
/kaggle/working/data/processed/test_data.csv
/kaggle/working/outputs/models/imputer.pkl
/kaggle/working/outputs/models/scaler.pkl
/kaggle/working/outputs/models/imputer_iso.pkl
/kaggle/working/outputs/models/scaler_iso.pkl
/kaggle/working/outputs/models/isolation_forest.pkl
/kaggle/working/outputs/models/random_forest.pkl
/kaggle/working/outputs/models/metrics.json
/kaggle/working/outputs/models/feature_importance.csv
```

## 3. Copy artifacts back to the demo

Download the Kaggle output and replace these local folders:

```text
data/processed/
outputs/models/
```

Then run the demo:

```powershell
cd "C:\Users\KHOAHEU\Downloads\AnomalyDetection-main (1)3\AnomalyDetection-main"
python -m uvicorn src.main:app --reload
```

## What changed

The new training pipeline adds these features:

```text
friends_followers_ratio
friends_followers_gap
statuses_followers_ratio
statuses_friends_ratio
favourites_statuses_ratio
screen_name_length
description_length
screen_name_has_spam_keyword
description_has_spam_keyword
description_has_url
spam_keyword_count
```

Random Forest now uses 300 trees, `max_features="sqrt"`, and `class_weight="balanced_subsample"` for more stable bot classification.
