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
    logging.info(f"Loading data from {bucket_name}/{object_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    data = blob.download_as_string()
    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
    return df

def upload_to_gcs(bucket_name, destination_blob_name, df):
    try:
        if df is not None:
            logging.info(f"Uploading data to {bucket_name}/{destination_blob_name}")
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

def perform_cross_validation(**kwargs):
    ti = kwargs['ti']
    df_dict = ti.xcom_pull(task_ids=kwargs['task_id'])
    df = pd.DataFrame(df_dict)
    
    logging.info("Performing cross-validation")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(skf.split(df, df['Ozone_Level'])):
        X_train, X_val = df.iloc[train_index], df.iloc[val_index]
        y_train, y_val = df['Ozone_Level'].iloc[train_index], df['Ozone_Level'].iloc[val_index]
        logging.info(f"Fold {fold}: Train size = {len(train_index)}, Validation size = {len(val_index)}")
    
    return df.to_dict()

def load_data_task(bucket_name, object_name, **kwargs):
    df = load_csv_from_gcs(bucket_name, object_name)
    return df.to_dict()

def upload_data_task(bucket_name, destination_blob_name, task_id, **kwargs):
    ti = kwargs['ti']
    df_dict = ti.xcom_pull(task_ids=task_id)
    df = pd.DataFrame(df_dict)
    upload_to_gcs(bucket_name, destination_blob_name, df)

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

cross_validation_eighthr_task = PythonOperator(
    task_id='cross_validation_eighthr',
    python_callable=perform_cross_validation,
    op_kwargs={'task_id': 'load_eighthr_data'},
    provide_context=True,
    dag=dag,
)

cross_validation_onehr_task = PythonOperator(
    task_id='cross_validation_onehr',
    python_callable=perform_cross_validation,
    op_kwargs={'task_id': 'load_onehr_data'},
    provide_context=True,
    dag=dag,
)

upload_eighthr_train_task = PythonOperator(
    task_id='upload_eighthr_train',
    python_callable=upload_data_task,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': 'data/cross_validated/eighthr_afterCV_train.csv', 'task_id': 'cross_validation_eighthr'},
    provide_context=True,
    dag=dag,
)

upload_onehr_train_task = PythonOperator(
    task_id='upload_onehr_train',
    python_callable=upload_data_task,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': 'data/cross_validated/onehr_afterCV_train.csv', 'task_id': 'cross_validation_onehr'},
    provide_context=True,
    dag=dag,
)

upload_eighthr_val_task = PythonOperator(
    task_id='upload_eighthr_val',
    python_callable=upload_data_task,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': 'data/cross_validated/eighthr_afterCV_val.csv', 'task_id': 'cross_validation_eighthr'},
    provide_context=True,
    dag=dag,
)

upload_onehr_val_task = PythonOperator(
    task_id='upload_onehr_val',
    python_callable=upload_data_task,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': 'data/cross_validated/onehr_afterCV_val.csv', 'task_id': 'cross_validation_onehr'},
    provide_context=True,
    dag=dag,
)

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

load_eighthr_task >> cross_validation_eighthr_task >> [upload_eighthr_train_task, upload_eighthr_val_task] >> email_task_1
load_onehr_task >> cross_validation_onehr_task >> [upload_onehr_train_task, upload_onehr_val_task] >> email_task_2
