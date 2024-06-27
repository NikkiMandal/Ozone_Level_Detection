from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from scripts.submit_vertex_ai_jobs import create_vertex_ai_training_job
from scripts.evaluate_model import evaluate_model
from scripts.report_generation import generate_report

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_pipeline_dag',
    default_args=default_args,
    description='DAG for end-to-end model pipeline',
    schedule_interval='@daily',
    catchup=False,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=create_vertex_ai_training_job,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    dag=dag,
)

train_model_task >> evaluate_model_task >> generate_report_task
