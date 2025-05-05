import os
import json
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
import logging as log

DATASET_DIR = "dataset"
MODEL_PATH = "/app/model/model.pkl"
METRICS_PATH = "/app/reports/model_metrics.json"

def test_model():
    """
    Загружает модель и тестовые данные, считает метрики и сохраняет
    """
    test_data_path = os.path.join(DATASET_DIR, "iris_test.csv")
    df = pd.read_csv(test_data_path)

    X = df.drop(columns=["target"])
    y_true = df["target"]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        'accuracy': accuracy,
        'report': report
    }

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    log.info(f'Метрики сохранены в {METRICS_PATH} — accuracy: {accuracy:.4f}')