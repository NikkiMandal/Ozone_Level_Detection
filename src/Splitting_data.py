# preprocess_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
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

    # Your preprocessing steps here
    # For example, if you want to scale numerical features:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.iloc[:, 1:])  # Assuming first column is target, rest are features
    df.iloc[:, 1:] = scaled_features  # Update the dataframe with scaled features
    
    return df

if __name__ == "__main__":
    bucket_name = "ozone_level_detection"
    
    # Load data from GCS
    eighthr_data = download_from_gcs(bucket_name, "data/processed/normalized_eighthr.csv")
    onehr_data = download_from_gcs(bucket_name, "data/processed/normalized_onehr.csv")
    
    # Preprocess the data
    cleaned_eighthr_data = preprocess_data(eighthr_data)
    cleaned_onehr_data = preprocess_data(onehr_data)
    
    # Split data into train and test sets
    eighthr_train, eighthr_test = train_test_split(cleaned_eighthr_data, test_size=0.2, random_state=42)
    onehr_train, onehr_test = train_test_split(cleaned_onehr_data, test_size=0.2, random_state=42)
    
    # Upload cleaned and split data to GCS
    upload_to_gcs(bucket_name, "data/processed/train_eighthr.csv", eighthr_train)
    upload_to_gcs(bucket_name, "data/processed/test_eighthr.csv", eighthr_test)
    upload_to_gcs(bucket_name, "data/processed/train_onehr.csv", onehr_train)
    upload_to_gcs(bucket_name, "data/processed/test_onehr.csv", onehr_test)
