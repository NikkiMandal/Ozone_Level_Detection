from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
import pandas as pd
from sklearn.preprocessing import StandardScaler
import io
import logging
import numpy as np

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

def scale_features(**kwargs):
    """Scale numerical features in the data."""
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=kwargs['task_ids'][0])
    if df is not None:
        numerical_cols = df.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
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
    'feature_scaling_dag',
    default_args=default_args,
    description='DAG for scaling features in ozone level data',
    schedule_interval='@daily',
    catchup=False,
)

download_no_outliers_eighthr = PythonOperator(
    task_id='download_no_outliers_eighthr',
    python_callable=download_from_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': "data/cleaned/eighthr_data_no_outliers.csv"
    },
    provide_context=True,
    dag=dag,
)

download_no_outliers_onehr = PythonOperator(
    task_id='download_no_outliers_onehr',
    python_callable=download_from_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': "data/cleaned/onehr_data_no_outliers.csv"
    },
    provide_context=True,
    dag=dag,
)

scale_features_eighthr = PythonOperator(
    task_id='scale_features_eighthr',
    python_callable=scale_features,
    op_kwargs={'task_ids': ['download_no_outliers_eighthr']},
    provide_context=True,
    dag=dag,
)

scale_features_onehr = PythonOperator(
    task_id='scale_features_onehr',
    python_callable=scale_features,
    op_kwargs={'task_ids': ['download_no_outliers_onehr']},
    provide_context=True,
    dag=dag,
)

upload_scaled_eighthr = PythonOperator(
    task_id='upload_scaled_eighthr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': "data/cleaned/eighthr_data_scaled.csv",
        'task_ids': ['scale_features_eighthr']
    },
    provide_context=True,
    dag=dag,
)

upload_scaled_onehr = PythonOperator(
    task_id='upload_scaled_onehr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': "data/cleaned/onehr_data_scaled.csv",
        'task_ids': ['scale_features_onehr']
    },
    provide_context=True,
    dag=dag,
)

# Set task dependencies
download_no_outliers_eighthr >> scale_features_eighthr >> upload_scaled_eighthr
download_no_outliers_onehr >> scale_features_onehr >> upload_scaled_onehr
