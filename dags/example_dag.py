from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging as log

def say_hello():
    log.info("Hello from Airflow!")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),    
}

with DAG(
    dag_id='example_hello_dag',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['example'],
) as dag:
    hello_task = PythonOperator(
        task_id='say_hello_task',
        python_callable=say_hello
    )