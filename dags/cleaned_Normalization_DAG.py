import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from airflow.hooks.base import BaseHook
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from google.cloud import storage
import pandas as pd
import numpy as np
import io
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def send_email_gmail(recipient, subject, body, **kwargs):
    # Fetch credentials from Airflow connection
    smtp_conn = BaseHook.get_connection('smtp_default')
    sender = smtp_conn.login
    password = smtp_conn.password

    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Connect to Gmail's SMTP server
    server = smtplib.SMTP(host=smtp_conn.host, port=smtp_conn.port)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()
    print(f"Email sent successfully to {recipient} with subject '{subject}'")

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

def normalize_data(**kwargs):
    """Normalize the data."""
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=kwargs['task_ids'][0])
    if df is not None:
        scaler = StandardScaler()
        numerical_cols = df.select_dtypes(include=np.number).columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logging.info(f"Data normalized for {kwargs['task_ids'][0]}")
        return df
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
    'normalize_new_data_dag',
    default_args=default_args,
    description='DAG for normalizing data from GCS and re-uploading',
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

# Define task to normalize data
normalize_eighthr_data = PythonOperator(
    task_id='normalize_eighthr_data',
    python_callable=normalize_data,
    op_kwargs={'task_ids': ['download_eighthr_data']},
    provide_context=True,
    dag=dag,
)

normalize_onehr_data = PythonOperator(
    task_id='normalize_onehr_data',
    python_callable=normalize_data,
    op_kwargs={'task_ids': ['download_onehr_data']},
    provide_context=True,
    dag=dag,
)

# Define task to upload normalized data
upload_normalized_eighthr = PythonOperator(
    task_id='upload_normalized_eighthr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': 'data/normalized/eighthr_data_normalized.csv',
        'task_ids': ['normalize_eighthr_data']
    },
    provide_context=True,
    dag=dag,
)

upload_normalized_onehr = PythonOperator(
    task_id='upload_normalized_onehr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': 'data/normalized/onehr_data_normalized.csv',
        'task_ids': ['normalize_onehr_data']
    },
    provide_context=True,
    dag=dag,
)

email_task_1 = PythonOperator(
    task_id='send_email_1',
    python_callable=send_email_gmail,
    op_kwargs={
        'recipient': 'nikitamandal03@gmail.com',  
        'subject': 'Success',
        'body': 'Normalized Eight hour data is uploaded to GCS Bucket.',
    },
    dag=dag,
)

email_task_2 = PythonOperator(
    task_id='send_email_2',
    python_callable=send_email_gmail,
    op_kwargs={
        'recipient': 'nikitamandal03@gmail.com',  
        'subject': 'Success',
        'body': 'Normalized One hour data is uploaded to GCS Bucket.',
    },
    dag=dag,
)

# Set task dependencies
download_eighthr_data >> normalize_eighthr_data >> upload_normalized_eighthr >> email_task_1
download_onehr_data >> normalize_onehr_data >> upload_normalized_onehr >> email_task_2
[upload_normalized_eighthr, upload_normalized_onehr] 
