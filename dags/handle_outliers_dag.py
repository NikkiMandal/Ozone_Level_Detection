from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
import pandas as pd
import numpy as np
import io
import logging

def download_from_gcs(bucket_name, source_blob_name, **kwargs):
    """Download file from GCS and return as a pandas DataFrame."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        data = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(data))
        logging.info(f"Successfully downloaded data from {bucket_name}/{source_blob_name}")
        return df
    except Exception as e:
        logging.error(f"Error downloading from GCS: {e}")
        raise

def upload_to_gcs(bucket_name, destination_blob_name, **kwargs):
    """Upload a pandas DataFrame to GCS."""
    try:
        ti = kwargs['ti']
        df = ti.xcom_pull(task_ids=kwargs['task_ids'][0])
        if df is not None:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            data_bytes = df.to_csv(index=False).encode()
            blob.upload_from_string(data_bytes, content_type='text/csv')
            logging.info(f"Uploaded data to {bucket_name}/{destination_blob_name}")
        else:
            raise ValueError("No data returned from XCom.")
    except Exception as e:
        logging.error(f"Error uploading to GCS: {e}")
        raise

def handle_outliers(**kwargs):
    """Remove outliers from the data."""
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=kwargs['task_ids'][0])
    if df is not None:
        numerical_cols = df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df
    else:
        raise ValueError("No data returned from XCom.")

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
    'handle_outliers_dag',
    default_args=default_args,
    description='DAG for handling outliers in ozone level data',
    schedule_interval='@daily',
    catchup=False,
)

download_cleaned_eighthr = PythonOperator(
    task_id='download_cleaned_eighthr',
    python_callable=download_from_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': "data/cleaned/eighthr_data_no_missing_values.csv"
    },
    provide_context=True,
    dag=dag,
)

download_cleaned_onehr = PythonOperator(
    task_id='download_cleaned_onehr',
    python_callable=download_from_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': "data/cleaned/onehr_data_no_missing_values.csv"
    },
    provide_context=True,
    dag=dag,
)

remove_outliers_eighthr = PythonOperator(
    task_id='remove_outliers_eighthr',
    python_callable=handle_outliers,
    op_kwargs={'task_ids': ['download_cleaned_eighthr']},
    provide_context=True,
    dag=dag,
)

remove_outliers_onehr = PythonOperator(
    task_id='remove_outliers_onehr',
    python_callable=handle_outliers,
    op_kwargs={'task_ids': ['download_cleaned_onehr']},
    provide_context=True,
    dag=dag,
)

upload_no_outliers_eighthr = PythonOperator(
    task_id='upload_no_outliers_eighthr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': "data/cleaned/eighthr_data_no_outliers.csv",
        'task_ids': ['remove_outliers_eighthr']
    },
    provide_context=True,
    dag=dag,
)

upload_no_outliers_onehr = PythonOperator(
    task_id='upload_no_outliers_onehr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': "data/cleaned/onehr_data_no_outliers.csv",
        'task_ids': ['remove_outliers_onehr']
    },
    provide_context=True,
    dag=dag,
)

# Set task dependencies
download_cleaned_eighthr >> remove_outliers_eighthr >> upload_no_outliers_eighthr
download_cleaned_onehr >> remove_outliers_onehr >> upload_no_outliers_onehr
