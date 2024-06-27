from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import io
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

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

def preprocess_data(ti, bucket_name, source_blob_name):
    """Download and preprocess data from GCS."""
    try:
        df = download_from_gcs(bucket_name, source_blob_name)
        numerical_cols = df.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logging.info(f"Successfully preprocessed data with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error in preprocess_data: {e}")
        raise

def train_model(ti, model_filename, **kwargs):
    """Train and save a machine learning model."""
    try:
        df_json = ti.xcom_pull(key='eighthr_preprocessed_data', task_ids='preprocess_eighthr_data')
        df = pd.read_json(df_json)
        target_column = 'Ozone_Level'
        date_column = 'Date'

        X = df.drop(columns=[target_column, date_column])
        y = df[target_column]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')  # Assuming binary classification; adjust as needed
        ])

        model.compile(optimizer= adam_v2(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

        loss, accuracy = model.evaluate(X_test, y_test)
        logging.info(f"Model accuracy: {accuracy}")

        model.save(model_filename)
        logging.info(f"Model saved as {model_filename}")
        ti.xcom_push(key='model_filename', value=model_filename)
    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        raise

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
    'ml_pipeline_dag',
    default_args=default_args,
    description='DAG for machine learning pipeline',
    schedule_interval='@daily',
    catchup=False,
)

preprocess_eighthr_task = PythonOperator(
    task_id='preprocess_eighthr_data',
    python_callable=preprocess_data,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'source_blob_name': 'data/cleaned/eighthr_data_scaled.csv'},
    provide_context=True,
    dag=dag,
)

train_eighthr_task = PythonOperator(
    task_id='train_eighthr_model',
    python_callable=train_model,
    op_kwargs={'model_filename': 'eighthr_tensorflow_model.h5'},
    provide_context=True,
    dag=dag,
)

upload_eighthr_task = PythonOperator(
    task_id='upload_eighthr_model_to_gcs',
    python_callable=upload_to_gcs,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': 'model_output/eighthr_tensorflow_model.h5', 'task_ids': ['train_eighthr_model']},
    provide_context=True,
    dag=dag,
)

preprocess_onehr_task = PythonOperator(
    task_id='preprocess_onehr_data',
    python_callable=preprocess_data,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'source_blob_name': 'data/cleaned/onehr_data_scaled.csv'},
    provide_context=True,
    dag=dag,
)

train_onehr_task = PythonOperator(
    task_id='train_onehr_model',
    python_callable=train_model,
    op_kwargs={'model_filename': 'onehr_tensorflow_model.h5'},
    provide_context=True,
    dag=dag,
)

upload_onehr_task = PythonOperator(
    task_id='upload_onehr_model_to_gcs',
    python_callable=upload_to_gcs,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'destination_blob_name': 'model_output/onehr_tensorflow_model.h5', 'task_ids': ['train_onehr_model']},
    provide_context=True,
    dag=dag,
)

# Set task dependencies
preprocess_eighthr_task >> train_eighthr_task >> upload_eighthr_task
preprocess_onehr_task >> train_onehr_task >> upload_onehr_task
