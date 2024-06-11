from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.hooks.gcs_hook import GoogleCloudStorageHook
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import io

def download_from_gcs(bucket_name, source_blob_name, **kwargs):
    """Download file from GCS and return as a pandas DataFrame."""
    hook = GoogleCloudStorageHook()
    file_bytes = hook.download(bucket_name=bucket_name, object_name=source_blob_name)
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df

def upload_to_gcs(bucket_name, destination_blob_name, df, **kwargs):
    """Upload a pandas DataFrame to GCS."""
    hook = GoogleCloudStorageHook()
    data_bytes = df.to_csv(index=False).encode()
    hook.upload(bucket_name=bucket_name, object_name=destination_blob_name, data=data_bytes, mime_type='text/csv')

def preprocess_data(**kwargs):
    """Preprocess the data."""
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=kwargs['task_ids'])
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
    else:
        raise ValueError("No data returned from XCom.")
    return df

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 10, 21, 35),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'preprocess_data',
    default_args=default_args,
    description='Preprocess ozone level data',
    schedule_interval='@daily',
)

download_and_preprocess_eighthr_data = PythonOperator(
    task_id='download_and_preprocess_eighthr_data',
    python_callable=download_from_gcs,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'source_blob_name': "data/raw/eighthr_data.csv"},
    provide_context=True,
    dag=dag,
)

clean_and_upload_eighthr_data = PythonOperator(
    task_id='clean_and_upload_eighthr_data',
    python_callable=preprocess_data,
    op_kwargs={'task_ids': 'download_and_preprocess_eighthr_data'},
    provide_context=True,
    dag=dag,
)

upload_eighthr_data_to_gcs = PythonOperator(
    task_id='upload_eighthr_data_to_gcs',
    python_callable=upload_to_gcs,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': "data/cleaned/eighthr_data_cleaned.csv"},
    provide_context=True,
    dag=dag,
)

download_and_preprocess_onehr_data = PythonOperator(
    task_id='download_and_preprocess_onehr_data',
    python_callable=download_from_gcs,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'source_blob_name': "data/raw/onehr_data.csv"},
    provide_context=True,
    dag=dag,
)

clean_and_upload_onehr_data = PythonOperator(
    task_id='clean_and_upload_onehr_data',
    python_callable=preprocess_data,
    op_kwargs={'task_ids': 'download_and_preprocess_onehr_data'},
    provide_context=True,
    dag=dag,
)

upload_onehr_data_to_gcs = PythonOperator(
    task_id='upload_onehr_data_to_gcs',
    python_callable=upload_to_gcs,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': "data/cleaned/onehr_data_cleaned.csv"},
    provide_context=True,
    dag=dag,
)

# Set task dependencies
download_and_preprocess_eighthr_data >> clean_and_upload_eighthr_data >> upload_eighthr_data_to_gcs
download_and_preprocess_onehr_data >> clean_and_upload_onehr_data >> upload_onehr_data_to_gcs

