from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import logging

def log_pipeline_status():
    """Function to log the status of the pipeline."""
    logging.info("Pipeline is running smoothly.")
    # You can add more detailed monitoring and logging functionality here

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'monitoring_dag',
    default_args=default_args,
    description='DAG for monitoring and logging',
    schedule_interval='@daily',  # Runs daily
    start_date=datetime(2024, 6, 10),
    catchup=False,
) as dag:
    start = EmptyOperator(task_id='start')

    monitor_pipeline = PythonOperator(
        task_id='log_pipeline_status',
        python_callable=log_pipeline_status,
    )

    end = EmptyOperator(task_id='end')

    start >> monitor_pipeline >> end
