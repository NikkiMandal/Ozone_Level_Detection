# Updated cross-validation logic
from datetime import datetime, timedelta
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
import pandas as pd
import io
from sklearn.model_selection import StratifiedKFold
from airflow.hooks.base import BaseHook
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_gmail(recipient, subject, body, **kwargs):
    smtp_conn = BaseHook.get_connection('smtp_default')
    sender = smtp_conn.login
    password = smtp_conn.password

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP(host=smtp_conn.host, port=smtp_conn.port)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()
    logging.info(f"Email sent successfully to {recipient} with subject '{subject}'")

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

# Updated cross-validation logic
def perform_cross_validation(bucket_name, object_name, dataset_name, label_column, **kwargs):
    df = load_csv_from_gcs(bucket_name, object_name)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    fold_paths = {'train': [], 'val': []}
    for train_index, val_index in skf.split(df, df[label_column]):
        X_train, X_val = df.iloc[train_index], df.iloc[val_index]
        y_train, y_val = df[label_column].iloc[train_index], df[label_column].iloc[val_index]

        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)

        train_path = f'cross_validation/{dataset_name}_train_fold{fold}.csv'
        val_path = f'cross_validation/{dataset_name}_val_fold{fold}.csv'

        upload_to_gcs(bucket_name, train_path, train_data)
        upload_to_gcs(bucket_name, val_path, val_data)

        fold_paths['train'].append(train_path)
        fold_paths['val'].append(val_path)

        fold += 1
    return fold_paths

# Define your DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 24),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'cross_validation_dag_with_split',
    default_args=default_args,
    description='Pipeline for ozone level detection using Airflow',
    schedule_interval=None,
)

cross_validation_eighthr_task = PythonOperator(
    task_id='cross_validation_eighthr',
    python_callable=perform_cross_validation,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'object_name': 'SMOTE_analysis/eighthr_train_resampled.csv', 'dataset_name': 'eighthr', 'label_column': 'Ozone_Level'},
    provide_context=True,
    dag=dag,
)

cross_validation_onehr_task = PythonOperator(
    task_id='cross_validation_onehr',
    python_callable=perform_cross_validation,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'object_name': 'SMOTE_analysis/onehr_train_resampled.csv', 'dataset_name': 'onehr', 'label_column': 'Ozone_Level'},
    provide_context=True,
    dag=dag,
)

# Email tasks
email_task_1 = PythonOperator(
    task_id='send_email_1',
    python_callable=send_email_gmail,
    op_kwargs={
        'recipient': 'nikitamandal03@gmail.com',
        'subject': 'Success',
        'body': 'Uploaded Eight hour train and validation data to GCS Bucket.',
    },
    dag=dag,
)

email_task_2 = PythonOperator(
    task_id='send_email_2',
    python_callable=send_email_gmail,
    op_kwargs={
        'recipient': 'nikitamandal03@gmail.com',
        'subject': 'Success',
        'body': 'Uploaded One hour train and validation data to GCS Bucket.',
    },
    dag=dag,
)

# Dependencies
cross_validation_eighthr_task >> email_task_1
cross_validation_onehr_task >> email_task_2
