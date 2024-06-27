from datetime import datetime, timedelta
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from airflow.hooks.base import BaseHook

# Remove the environment variable for authentication
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/sameeramandalika/Downloads/ozone-level-detection-5d4e101f1b32.json"

def send_email_gmail(**kwargs):
    # Fetch credentials from Airflow connection
    smtp_conn = BaseHook.get_connection('smtp_default')
    
    sender = smtp_conn.login
    password = smtp_conn.password
    recipient = 'nikitamandal03@gmail.com'  # Replace with the recipient's email
    subject = 'SMOTE Analysis Success'
    body = 'SMOTE analysis is successfully executed.'

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

# Function to load a CSV file from Google Cloud Storage into a Pandas DataFrame
def load_csv_from_gcs(bucket_name, object_name):
    logging.info(f"Loading data from {bucket_name}/{object_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    data = blob.download_as_string()
    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
    return df

# Function to upload a pandas DataFrame to GCS
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

# Function to apply SMOTE to balance the dataset
def apply_smote(df, target_column):
    """Apply SMOTE to balance the dataset."""
    X = df.drop(columns=[target_column, 'Date'])  # Drop Date column
    y = df[target_column]
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    df_resampled = pd.concat([pd.DataFrame(X_smote), pd.DataFrame(y_smote, columns=[target_column])], axis=1)
    return df_resampled

# Function to load data from GCS, apply SMOTE, split into train, val, test, and upload resampled data
def load_process_split_and_upload_data(bucket_name, object_name, train_destination_blob_name, val_destination_blob_name, test_destination_blob_name):
    try:
        # Load data from GCS
        df = load_csv_from_gcs(bucket_name, object_name)
        
        # Apply SMOTE
        df_resampled = apply_smote(df, 'Ozone_Level')
        
        # Split into train, val, test
        train, test = train_test_split(df_resampled, test_size=0.2, random_state=42)
        train, val = train_test_split(train, test_size=0.2, random_state=42)
        
        # Upload train, val, test data to GCS
        upload_to_gcs(bucket_name, train_destination_blob_name, train)
        upload_to_gcs(bucket_name, val_destination_blob_name, val)
        upload_to_gcs(bucket_name, test_destination_blob_name, test)
        
        # Upload resampled data after SMOTE analysis to SMOTE_analysis folder
        upload_to_gcs(bucket_name, 'SMOTE_analysis/' + train_destination_blob_name.split('/')[-1], train)
        upload_to_gcs(bucket_name, 'SMOTE_analysis/' + test_destination_blob_name.split('/')[-1], test)
        
    except Exception as e:
        logging.error(f"Error loading, processing, splitting, and uploading data: {e}")
        raise

# DAG definition
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
    'SMOTE_Analysis_dag',
    default_args=default_args,
    description='Airflow DAG mimicking the functionality of the main() function',
    schedule_interval=None,
)

# Tasks
load_and_process_eighthr_task = PythonOperator(
    task_id='load_and_process_eighthr_data',
    python_callable=load_process_split_and_upload_data,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'object_name': 'data/new_cleaned/eighthr_data_KNN.csv',
        'train_destination_blob_name': 'SplitData/eighthr_train.csv',
        'val_destination_blob_name': 'SplitData/eighthr_val.csv',
        'test_destination_blob_name': 'SplitData/eighthr_test.csv'
    },
    dag=dag,
)

load_and_process_onehr_task = PythonOperator(
    task_id='load_and_process_onehr_data',
    python_callable=load_process_split_and_upload_data,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'object_name': 'data/new_cleaned/onehr_data_KNN.csv',
        'train_destination_blob_name': 'SplitData/onehr_train.csv',
        'val_destination_blob_name': 'SplitData/onehr_val.csv',
        'test_destination_blob_name': 'SplitData/onehr_test.csv'
    },
    dag=dag,
)

email_task = PythonOperator(
    task_id='send_email',
    python_callable=send_email_gmail,
    dag=dag,
)

email_task

# Task dependencies
load_and_process_eighthr_task >> load_and_process_onehr_task >> email_task
