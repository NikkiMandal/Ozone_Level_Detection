from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Add the directory containing `data_modelling.py` to the Python path
sys.path.append('src/modeling.py')

# Import functions from data_modelling.py
from data_modelling import load_data, split_data, train_model_eighthr, train_model_onehr, evaluate_model_eighthr, evaluate_model_onehr

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Initialize the DAG
dag = DAG(
    'ozone_level_detection',
    default_args=default_args,
    description='A simple DAG to train and evaluate logistic regression models for ozone level detection',
    schedule_interval='@once',
)

# Define the tasks
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    dag=dag,
)

train_model_eighthr_task = PythonOperator(
    task_id='train_model_eighthr',
    python_callable=train_model_eighthr,
    dag=dag,
)

train_model_onehr_task = PythonOperator(
    task_id='train_model_onehr',
    python_callable=train_model_onehr,
    dag=dag,
)

evaluate_model_eighthr_task = PythonOperator(
    task_id='evaluate_model_eighthr',
    python_callable=evaluate_model_eighthr,
    dag=dag,
)

evaluate_model_onehr_task = PythonOperator(
    task_id='evaluate_model_onehr',
    python_callable=evaluate_model_onehr,
    dag=dag,
)

# Define task dependencies
load_data_task >> split_data_task
split_data_task >> train_model_eighthr_task >> evaluate_model_eighthr_task
split_data_task >> train_model_onehr_task >> evaluate_model_onehr_task