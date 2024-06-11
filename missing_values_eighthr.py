# preprocess_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from google.cloud import storage
import io
import os

# Set the environment variable for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  "/content/ozone-level-detection-0160dba47662.json"

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

    # Step 1: Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Step 2: Convert columns to appropriate types
    # Convert numerical columns to float, skipping Date
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Step 3: Handle the Date column
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

    # Step 4: Handle Missing Values
    # Removing rows where more than 50% of the values are missing
    df = df.dropna(thresh=len(df.columns) * 0.5)

    # Fill missing values for numerical columns with median
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

if __name__ == "__main__":
    bucket_name = "ozone_level_detection"
    
    # Load data from GCS
    eighthr_data = download_from_gcs(bucket_name, "data/raw/eighthr_data.csv")
    onehr_data = download_from_gcs(bucket_name, "data/raw/onehr_data.csv")
    
    # Preprocess the data
    cleaned_eighthr_data = preprocess_data(eighthr_data)
    cleaned_onehr_data = preprocess_data(onehr_data)
    
