from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from google.cloud import storage
import pandas as pd
import io
import os
from sklearn.model_selection import StratifiedKFold

# Set the environment variable for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account-key.json"

def load_csv_from_gcs(bucket_name, object_name):
    """Load a CSV file from Google Cloud Storage into a Pandas DataFrame."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    data = blob.download_as_string()
    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
    return df

def upload_to_gcs(bucket_name, destination_blob_name, df):
    """Upload a pandas DataFrame to GCS."""
    try:
        if df is not None:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            data_bytes = df.to_csv(index=False).encode()
            blob.upload_from_string(data_bytes, content_type='text/csv')
            logging.info(f"Uploaded data to {bucket_name}/{destination_blob_name}")
        else:
            raise ValueError("No data provided to upload.")
    except Exception as e:
        logging.error(f"Error uploading to GCS: {e}")
        raise

def perform_cross_validation(df, **kwargs):
    """Perform stratified cross-validation."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(skf.split(df, df['Ozone_Level'])):
        X_train, X_val = df.iloc[train_index], df.iloc[val_index]
        y_train, y_val = df['Ozone_Level'].iloc[train_index], df['Ozone_Level'].iloc[val_index]
        # Perform further processing or model training/validation here

# Define DAG parameters
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 30),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    'ozone_level_detection_pipeline',
    default_args=default_args,
    description='Pipeline for ozone level detection using Airflow',
    schedule_interval=None,  # Or define your schedule interval
)

# Define tasks

# Task 1: Load data from GCS
def load_data_task(bucket_name, object_name, **kwargs):
    df = load_csv_from_gcs(bucket_name, object_name)
    return df

load_eighthr_task = PythonOperator(
    task_id='load_eighthr_data',
    python_callable=load_data_task,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'object_name': 'data/cleaned/eighthr_train_data.csv'},
    provide_context=True,
    dag=dag,
)

load_onehr_task = PythonOperator(
    task_id='load_onehr_data',
    python_callable=load_data_task,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'object_name': 'data/cleaned/onehr_train_data.csv'},
    provide_context=True,
    dag=dag,
)

# Task 2: Stratified cross-validation
cross_validation_eighthr_task = PythonOperator(
    task_id='cross_validation_eighthr',
    python_callable=perform_cross_validation,
    op_kwargs={'df': "{{ task_instance.xcom_pull(task_ids='load_eighthr_data') }}"},
    provide_context=True,
    dag=dag,
)

cross_validation_onehr_task = PythonOperator(
    task_id='cross_validation_onehr',
    python_callable=perform_cross_validation,
    op_kwargs={'df': "{{ task_instance.xcom_pull(task_ids='load_onehr_data') }}"},
    provide_context=True,
    dag=dag,
)

# Task 3: Upload processed data to GCS
def upload_data_task(bucket_name, destination_blob_name, **kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=kwargs['task_id'])
    upload_to_gcs(bucket_name, destination_blob_name, df)

upload_eighthr_train_task = PythonOperator(
    task_id='upload_eighthr_train',
    python_callable=upload_data_task,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': 'data/train/eighthr_afterCV_train.csv'},
    provide_context=True,
    dag=dag,
)

upload_onehr_train_task = PythonOperator(
    task_id='upload_onehr_train',
    python_callable=upload_data_task,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': 'data/train/onehr_afterCV_train.csv'},
    provide_context=True,
    dag=dag,
)

upload_eighthr_val_task = PythonOperator(
    task_id='upload_eighthr_val',
    python_callable=upload_data_task,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': 'data/validation/eighthr_afterCV_val.csv'},
    provide_context=True,
    dag=dag,
)

upload_onehr_val_task = PythonOperator(
    task_id='upload_onehr_val',
    python_callable=upload_data_task,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': 'data/validation/onehr_afterCV_val.csv'},
    provide_context=True,
    dag=dag,
)

# Define task dependencies
load_eighthr_task >> cross_validation_eighthr_task
load_onehr_task >> cross_validation_onehr_task

cross_validation_eighthr_task >> upload_eighthr_train_task
cross_validation_onehr_task >> upload_onehr_train_task

cross_validation_eighthr_task >> upload_eighthr_val_task
cross_validation_onehr_task >> upload_onehr_val_task
