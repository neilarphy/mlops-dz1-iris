import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import logging as log

DATASET_DIR = 'dataset'

def load_data():
    """
    Загружаем датасет Ирисов и сохраяенм в csv
    """
    os.makedirs(DATASET_DIR, exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(os.path.join(DATASET_DIR, "iris.csv"), index=False)

    log.info(f"iris.csv сохранен ({len(df)} строк)")

def prepare_data():
    """
    Считываем датасет iris.csv и сохраняем выборки
    """
    df = pd.read_csv(os.path.join(DATASET_DIR, "iris.csv"))

    X = df.drop(columns=["target"])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_data = X_train.copy()
    train_data["target"] = y_train

    test_data = X_test.copy()
    test_data["target"] = y_test

    train_data.to_csv(os.path.join(DATASET_DIR, "iris_train.csv"), index=False)
    test_data.to_csv(os.path.join(DATASET_DIR, "iris_test.csv"), index=False)

    log.info(f" iris_train.csv ({len(train_data)} строк), iris_test.csv ({len(test_data)} строк)")
