from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import pandas as pd
from google.cloud import storage
from io import StringIO
import logging

# Ensure you have Google Cloud SDK set up and authentication done

def download_data(url):
    """
    Download data from a specified URL and return the content.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.info(f"Successfully downloaded data from {url}")
        return response.content
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading data from {url}: {e}")
        raise

def upload_to_gcs(bucket_name, destination_blob_name, data):
    """
    Upload data to a specified Google Cloud Storage bucket.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(data.encode('utf-8'), content_type='text/csv')
        logging.info(f"Uploaded data to {bucket_name}/{destination_blob_name}")
    except Exception as e:
        logging.error(f"Error uploading to GCS: {e}")
        raise

def process_files(data_file_content, names_file_content, target_column_name):
    """
    Process the data and names files to prepare a DataFrame with appropriate column names.
    """
    try:
        column_names = []
        metadata = {}
        lines = names_file_content.decode('utf-8').splitlines()

        metadata_end_index = 0
        for i, line in enumerate(lines):
            if 'Date:' in line:
                metadata_end_index = i
                break
            metadata[f"Metadata_{i}"] = line.strip()

        for line in lines[metadata_end_index:]:
            if line.strip():
                parts = line.split(':')
                column_name = parts[0].strip()
                if len(parts) > 1:
                    metadata[column_name] = parts[1].strip()
                column_names.append(column_name)

        column_names.append(target_column_name)
        logging.info(f"Column Names: {column_names}")
        logging.info(f"Metadata: {metadata}")

        data_content = data_file_content.decode('utf-8')
        data = pd.read_csv(StringIO(data_content), header=None)

        data_columns_count = len(data.columns)
        names_columns_count = len(column_names)

        if data_columns_count != names_columns_count:
            if data_columns_count > names_columns_count:
                for i in range(data_columns_count - names_columns_count):
                    column_names.insert(-1, f'Unnamed_{i+1}')
            elif data_columns_count < names_columns_count:
                column_names = column_names[:data_columns_count]

        data.columns = column_names
        logging.info(f"Processed Data Head: {data.head()}")
        logging.info(f"Data Columns: {data.columns}")
        logging.info(f"Shape of Data: {data.shape}")

        return data
    except Exception as e:
        logging.error(f"Error processing files: {e}")
        raise

def download_process_upload_data(url_key, bucket_name, destination_blob_name, target_column_name, **kwargs):
    """
    Download, process, and upload data for a specified URL key.
    """
    try:
        urls = {
            "eighthr_data": "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/eighthr.data",
            "onehr_data": "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.data",
            "eighthr_names": "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/eighthr.names",
            "onehr_names": "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.names"
        }

        data_file_content = download_data(urls[f'{url_key}_data'])
        names_file_content = download_data(urls[f'{url_key}_names'])
        data = process_files(data_file_content, names_file_content, target_column_name)
        data_csv = data.to_csv(index=False)
        upload_to_gcs(bucket_name, destination_blob_name, data_csv)
    except Exception as e:
        logging.error(f"Error in download_process_upload_data for {url_key}: {e}")
        raise

# Define DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 10),  # Ensure this date is in the past
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Define DAG
dag = DAG(
    'ozone_data_dag',
    default_args=default_args,
    description='DAG for downloading, processing, and uploading ozone data',
    schedule_interval='@daily',  # Run daily
)

# Define tasks
tasks = []
for key in ['eighthr', 'onehr']:
    task = PythonOperator(
        task_id=f'download_process_upload_{key}_data',
        python_callable=download_process_upload_data,
        op_kwargs={
            'url_key': key,
            'bucket_name': 'ozone_level_detection',
            'destination_blob_name': f"data/raw/{key}_data.csv",
            'target_column_name': 'Ozone_Level'
        },
        dag=dag,
    )
    tasks.append(task)

# Set task dependencies
tasks[0] >> tasks[1]
