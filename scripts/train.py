import os
import pandas as pd
import joblib 
from sklearn.linear_model import LogisticRegression
import logging as log

DATASET_DIR = 'dataset'
MODEL_PATH = "/app/model/model.pkl"

def train_model():
    """
    Обучает модель логистической регрессии и сохраняет модель в файл
    """
    train_data_path = os.path.join(DATASET_DIR, "iris_train.csv")
    df = pd.read_csv(train_data_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    log.info(f"Модель сохранена в путь {MODEL_PATH}")