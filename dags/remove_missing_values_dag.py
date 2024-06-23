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

def remove_missing_values(**kwargs):
    """Remove missing values from the data."""
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=kwargs['task_ids'][0])
    if df is not None:
        df.replace('?', np.nan, inplace=True)
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        df = df.dropna(thresh=len(df.columns) * 0.5)
        numerical_cols = df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
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
    'remove_missing_values_dag',
    default_args=default_args,
    description='DAG for removing missing values from ozone level data',
    schedule_interval='@daily',
    catchup=False,
)

download_eighthr_data = PythonOperator(
    task_id='download_eighthr_data',
    python_callable=download_from_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': "data/raw/eighthr_data.csv"
    },
    provide_context=True,
    dag=dag,
)

download_onehr_data = PythonOperator(
    task_id='download_onehr_data',
    python_callable=download_from_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': "data/raw/onehr_data.csv"
    },
    provide_context=True,
    dag=dag,
)

remove_missing_values_eighthr = PythonOperator(
    task_id='remove_missing_values_eighthr',
    python_callable=remove_missing_values,
    op_kwargs={'task_ids': ['download_eighthr_data']},
    provide_context=True,
    dag=dag,
)

remove_missing_values_onehr = PythonOperator(
    task_id='remove_missing_values_onehr',
    python_callable=remove_missing_values,
    op_kwargs={'task_ids': ['download_onehr_data']},
    provide_context=True,
    dag=dag,
)

upload_cleaned_eighthr = PythonOperator(
    task_id='upload_cleaned_eighthr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': "data/cleaned/eighthr_data_no_missing_values.csv",
        'task_ids': ['remove_missing_values_eighthr']
    },
    provide_context=True,
    dag=dag,
)

upload_cleaned_onehr = PythonOperator(
    task_id='upload_cleaned_onehr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': "data/cleaned/onehr_data_no_missing_values.csv",
        'task_ids': ['remove_missing_values_onehr']
    },
    provide_context=True,
    dag=dag,
)

# Set task dependencies
download_eighthr_data >> remove_missing_values_eighthr >> upload_cleaned_eighthr
download_onehr_data >> remove_missing_values_onehr >> upload_cleaned_onehr
