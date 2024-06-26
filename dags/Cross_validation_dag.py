from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from google.cloud import storage
import pandas as pd
import numpy as np
import io
import logging
from sklearn.model_selection import StratifiedKFold

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

def perform_cross_validation(**kwargs):
    """Perform stratified cross-validation."""
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=kwargs['task_ids'][0])
    if df is not None:
        try:
            logging.info("Performing cross-validation")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (train_index, val_index) in enumerate(skf.split(df, df['Ozone_Level'])):
                X_train, X_val = df.iloc[train_index], df.iloc[val_index]
                y_train, y_val = df['Ozone_Level'].iloc[train_index], df['Ozone_Level'].iloc[val_index]
                # Perform further processing or model training/validation here
                logging.info(f"Fold {fold}: Train size = {len(train_index)}, Validation size = {len(val_index)}")
            return df
        except Exception as e:
            logging.error(f"Error during cross-validation: {e}")
            raise
    else:
        raise ValueError("No data returned from XCom.")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'cross_validation_dag',
    default_args=default_args,
    description='DAG for performing cross-validation on data from GCS and re-uploading',
    schedule_interval='@daily',
    catchup=False,
)

# Define tasks to download data
download_eighthr_data = PythonOperator(
    task_id='download_eighthr_data',
    python_callable=download_from_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': 'data/new_cleaned/eighthr_data_KNN.csv',
    },
    provide_context=True,
    dag=dag,
)

download_onehr_data = PythonOperator(
    task_id='download_onehr_data',
    python_callable=download_from_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': 'data/new_cleaned/onehr_data_KNN.csv',
    },
    provide_context=True,
    dag=dag,
)

# Define task to perform cross-validation
cross_validate_eighthr_data = PythonOperator(
    task_id='cross_validate_eighthr_data',
    python_callable=perform_cross_validation,
    op_kwargs={'task_ids': ['download_eighthr_data']},
    provide_context=True,
    dag=dag,
)

cross_validate_onehr_data = PythonOperator(
    task_id='cross_validate_onehr_data',
    python_callable=perform_cross_validation,
    op_kwargs={'task_ids': ['download_onehr_data']},
    provide_context=True,
    dag=dag,
)

# Define task to upload cross-validated data
upload_cross_validated_eighthr = PythonOperator(
    task_id='upload_cross_validated_eighthr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': 'data/cross_validated/eighthr_data_cross_validated.csv',
        'task_ids': ['cross_validate_eighthr_data']
    },
    provide_context=True,
    dag=dag,
)

upload_cross_validated_onehr = PythonOperator(
    task_id='upload_cross_validated_onehr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': 'data/cross_validated/onehr_data_cross_validated.csv',
        'task_ids': ['cross_validate_onehr_data']
    },
    provide_context=True,
    dag=dag,
)

# Set task dependencies
download_eighthr_data >> cross_validate_eighthr_data >> upload_cross_validated_eighthr
download_onehr_data >> cross_validate_onehr_data >> upload_cross_validated_onehr
[upload_cross_validated_eighthr, upload_cross_validated_onehr]
