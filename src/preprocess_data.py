# preprocess_data.py
import pandas as pd
from google.cloud import storage
import io

def download_from_gcs(bucket_name, source_blob_name):
    """Download file from GCS and return as a pandas DataFrame."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_string()
    return pd.read_csv(io.BytesIO(data))

def upload_to_gcs(bucket_name, destination_blob_name, data):
    """Upload a pandas DataFrame to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(data.to_csv(index=False), content_type='text/csv')
    print(f"Uploaded to {bucket_name}/{destination_blob_name}")

def preprocess_data(df):
    """Clean and preprocess the data."""
    # Example preprocessing steps: remove missing values, normalize data, etc.
    df.dropna(inplace=True)  # Remove rows with missing values
    # Additional preprocessing steps can be added here
    return df

if __name__ == "__main__":
    bucket_name = "your-gcs-bucket-name"
    
    # Load data from GCS
    eighthr_data = download_from_gcs(bucket_name, "processed/eighthr_data.csv")
    onehr_data = download_from_gcs(bucket_name, "processed/onehr_data.csv")
    
    # Preprocess the data
    cleaned_eighthr_data = preprocess_data(eighthr_data)
    cleaned_onehr_data = preprocess_data(onehr_data)
    
    # Upload cleaned data to GCS
    upload_to_gcs(bucket_name, "processed/cleaned_eighthr_data.csv", cleaned_eighthr_data)
    upload_to_gcs(bucket_name, "processed/cleaned_onehr_data.csv", cleaned_onehr_data)
