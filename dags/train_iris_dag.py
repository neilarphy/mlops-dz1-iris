import sys
import os
sys.path.insert(0, os.path.abspath("/app/scripts"))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from data import load_data, prepare_data
from train import train_model
from test import test_model

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='iris_classification_pipeline',
    default_args=default_args,
    description='Train & test Logistic regression on Iris dataset',
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "iris"],
) as dag:
    
    t1 = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )

    t2 = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data
    )

    t3 = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    t4 = PythonOperator(
        task_id='test_model',
        python_callable=test_model
    )


    t1 >> t2 >> t3 >> t4