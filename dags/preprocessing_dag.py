# dags/preprocessing_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess

def run_preprocessing_script():
    """Run the preprocessing script."""
    subprocess.run(["python", "src/preprocess_data.py"], check=True)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'data_preprocessing_dag',
    default_args=default_args,
    description='A simple data preprocessing DAG',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 6, 10),
    catchup=False,
) as dag:
    run_preprocessing = PythonOperator(
        task_id='preprocess_data',
        python_callable=run_preprocessing_script,
    )

    run_preprocessing
