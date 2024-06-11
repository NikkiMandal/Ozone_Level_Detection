# preprocess_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
import io
import os

# Set the environment variable for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/Bhuvan Karthik/Downloads/ozone-level-detection-0160dba47662.json"

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
    """
    Clean and preprocess the data.
    
    Parameters:
    df (pd.DataFrame): Raw data DataFrame.
    
    Returns:
    pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    # Display initial dataset information
    print("Initial Data Info:")
    print(df.info())
    print("Initial Data Description:")
    print(df.describe(include='all'))
    
    # Step 6: Feature Scaling
    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Step 7: Handle Duplicates
    df.drop_duplicates(inplace=True)
    
    return df

if __name__ == "__main__":
    bucket_name = "ozone_level_detection"
    
    # Load data from GCS
    eighthr_data = download_from_gcs(bucket_name, "data/processed/1_missing_values_eighthr.csv")
    onehr_data = download_from_gcs(bucket_name, "data/processed/1_missing_values_onehr.csv")
    
    # Preprocess the data
    cleaned_eighthr_data = preprocess_data(eighthr_data)
    cleaned_onehr_data = preprocess_data(onehr_data)
    
    # Upload cleaned data to GCS
    upload_to_gcs(bucket_name, "data/processed/normalized_eighthr.csv", cleaned_eighthr_data)
    upload_to_gcs(bucket_name, "data/processed/normalized_onehr.csv", cleaned_onehr_data)
