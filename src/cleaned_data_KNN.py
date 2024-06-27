from google.cloud import storage
import pandas as pd
import io
import logging

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

from google.cloud import storage
import logging

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

from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import logging

def remove_missing_values_knn(**kwargs):
    """Replace missing values in the data using K-Nearest Neighbors."""
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids=kwargs['task_ids'][0])
    if df is not None:
        df.replace('?', np.nan, inplace=True)
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        
        imputer = KNNImputer(n_neighbors=5)
        numerical_cols = df.select_dtypes(include=np.number).columns
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        logging.info(f"Missing values imputed using KNN for {kwargs['task_ids'][0]}")
        return df
    else:
        raise ValueError("No data returned from XCom.")