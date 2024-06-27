from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import io
import logging
import traceback

def download_from_gcs(bucket_name, source_blob_name, **kwargs):
    """Download file from GCS and return as a pandas DataFrame."""
    try:
        logging.info(f"Downloading {source_blob_name} from bucket {bucket_name}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        data_bytes = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(data_bytes))
        logging.info(f"Successfully downloaded {source_blob_name}")
        return df
    except Exception as e:
        logging.error(f"Error downloading data from GCS: {e}")
        logging.error(traceback.format_exc())
        raise

def split_data(df, test_size=0.2, val_size=0.1):
    try:
        logging.info(f"Splitting data into train, val, and test sets with test_size={test_size}, val_size={val_size}")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=val_size / (1 - test_size), random_state=42)
        logging.info("Data split successfully")
        return train_df, val_df, test_df
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        logging.error(traceback.format_exc())
        raise

def upload_to_gcs(bucket_name, destination_blob_name, df, **kwargs):
    """Upload a pandas DataFrame to GCS."""
    try:
        logging.info(f"Uploading data to {destination_blob_name} in bucket {bucket_name}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        data_bytes = df.to_csv(index=False).encode()
        blob.upload_from_string(data_bytes, content_type='text/csv')
        logging.info(f"Uploaded data to {bucket_name}/{destination_blob_name}")
    except Exception as e:
        logging.error(f"Error uploading to GCS: {e}")
        logging.error(traceback.format_exc())
        raise

def preprocess_and_split_data(bucket_name, source_blob_name, **kwargs):
    """Download, preprocess, and split data."""
    try:
        logging.info(f"Preprocessing and splitting data from {source_blob_name}")
        df = download_from_gcs(bucket_name, source_blob_name)
        train_df, val_df, test_df = split_data(df)
        logging.info(f"Data split into train, val, and test sets")
        return {'train': train_df, 'val': val_df, 'test': test_df}
    except Exception as e:
        logging.error(f"Error in preprocess_and_split_data: {e}")
        logging.error(traceback.format_exc())
        raise

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'split_and_data_dag',
    default_args=default_args,
    description='DAG for splitting and uploading scaled ozone level data',
    schedule_interval='@daily',
    catchup=False,
)

# Eighthr Data
preprocess_and_split_eighthr_data = PythonOperator(
    task_id='preprocess_and_split_eighthr_data',
    python_callable=preprocess_and_split_data,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': "data/cleaned/eighthr_data_scaled.csv"
    },
    provide_context=True,
    dag=dag,
)

upload_train_eighthr_data = PythonOperator(
    task_id='upload_train_eighthr_data',
    python_callable=lambda **kwargs: upload_to_gcs('ozone_level_detection', 'data/cleaned/eighthr_train_data.csv', kwargs['ti'].xcom_pull(task_ids='preprocess_and_split_eighthr_data')['train']),
    provide_context=True,
    dag=dag,
)

upload_val_eighthr_data = PythonOperator(
    task_id='upload_val_eighthr_data',
    python_callable=lambda **kwargs: upload_to_gcs('ozone_level_detection', 'data/cleaned/eighthr_val_data.csv', kwargs['ti'].xcom_pull(task_ids='preprocess_and_split_eighthr_data')['val']),
    provide_context=True,
    dag=dag,
)

upload_test_eighthr_data = PythonOperator(
    task_id='upload_test_eighthr_data',
    python_callable=lambda **kwargs: upload_to_gcs('ozone_level_detection', 'data/cleaned/eighthr_test_data.csv', kwargs['ti'].xcom_pull(task_ids='preprocess_and_split_eighthr_data')['test']),
    provide_context=True,
    dag=dag,
)

# Onehr Data
preprocess_and_split_onehr_data = PythonOperator(
    task_id='preprocess_and_split_onehr_data',
    python_callable=preprocess_and_split_data,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': "data/cleaned/onehr_data_scaled.csv"
    },
    provide_context=True,
    dag=dag,
)

upload_train_onehr_data = PythonOperator(
    task_id='upload_train_onehr_data',
    python_callable=lambda **kwargs: upload_to_gcs('ozone_level_detection', 'data/cleaned/onehr_train_data.csv', kwargs['ti'].xcom_pull(task_ids='preprocess_and_split_onehr_data')['train']),
    provide_context=True,
    dag=dag,
)

upload_val_onehr_data = PythonOperator(
    task_id='upload_val_onehr_data',
    python_callable=lambda **kwargs: upload_to_gcs('ozone_level_detection', 'data/cleaned/onehr_val_data.csv', kwargs['ti'].xcom_pull(task_ids='preprocess_and_split_onehr_data')['val']),
    provide_context=True,
    dag=dag,
)

upload_test_onehr_data = PythonOperator(
    task_id='upload_test_onehr_data',
    python_callable=lambda **kwargs: upload_to_gcs('ozone_level_detection', 'data/cleaned/onehr_test_data.csv', kwargs['ti'].xcom_pull(task_ids='preprocess_and_split_onehr_data')['test']),
    provide_context=True,
    dag=dag,
)

# Set task dependencies
preprocess_and_split_eighthr_data >> [upload_train_eighthr_data, upload_val_eighthr_data, upload_test_eighthr_data]
preprocess_and_split_onehr_data >> [upload_train_onehr_data, upload_val_onehr_data, upload_test_onehr_data]
