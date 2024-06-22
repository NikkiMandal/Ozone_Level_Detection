# preprocess_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
import io
import logging

# Configure logging
logging.basicConfig(filename='logs/preprocess_data.log', level=logging.INFO)

def download_from_gcs(bucket_name, source_blob_name):
    """Download file from GCS and return as a pandas DataFrame."""
    logging.info(f"Downloading {source_blob_name} from GCS bucket {bucket_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_string()
    logging.info(f"Downloaded {source_blob_name} successfully")
    return pd.read_csv(io.BytesIO(data))

def upload_to_gcs(bucket_name, destination_blob_name, data):
    """Upload a pandas DataFrame to GCS."""
    logging.info(f"Uploading to {destination_blob_name} in GCS bucket {bucket_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(data.to_csv(index=False), content_type='text/csv')
    logging.info(f"Uploaded to {destination_blob_name} successfully")

def preprocess_data(df):
    """
    Clean and preprocess the data.
    
    Parameters:
    df (pd.DataFrame): Raw data DataFrame.
    
    Returns:
    pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    logging.info("Starting preprocessing")
    logging.info("Initial Data Info:")
    logging.info(df.info())
    logging.info("Initial Data Description:")
    logging.info(df.describe(include='all'))

    # Step 1: Replace '?' with NaN
    logging.info("Replacing '?' with NaN")
    df.replace('?', np.nan, inplace=True)

    # Step 2: Convert columns to appropriate types
    logging.info("Converting numerical columns to float")
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Step 3: Handle the Date column
    logging.info("Converting Date column to datetime")
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

    # Step 4: Handle Missing Values
    logging.info("Handling missing values")
    df = df.dropna(thresh=len(df.columns) * 0.5)
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    # Step 5: Handle Outliers
    logging.info("Handling outliers")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Step 6: Feature Scaling
    logging.info("Scaling numerical features")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    df[numerical_cols] = scaled_data
    logging.info(f"Scaled data mean: {scaled_data.mean(axis=0)}")
    logging.info(f"Scaled data std: {scaled_data.std(axis=0)}")

    # Step 7: Handle Duplicates
    logging.info("Dropping duplicate rows")
    df.drop_duplicates(inplace=True)

    logging.info("Final Data Info:")
    logging.info(df.info())
    logging.info("Final Data Description:")
    logging.info(df.describe(include='all'))

    logging.info("Preprocessing complete")
    return df

if __name__ == "__main__":
    bucket_name = "ozone_level_detection"

    # Load data from GCS
    logging.info("Loading data from GCS")
    eighthr_data = download_from_gcs(bucket_name, "data/raw/eighthr_data.csv")
    onehr_data = download_from_gcs(bucket_name, "data/raw/onehr_data.csv")

    # Preprocess the data
    logging.info("Preprocessing eighthr data")
    cleaned_eighthr_data = preprocess_data(eighthr_data)
    logging.info("Preprocessing onehr data")
    cleaned_onehr_data = preprocess_data(onehr_data)

    # Upload cleaned data to GCS
    logging.info("Uploading cleaned data to GCS")
    upload_to_gcs(bucket_name, "data/cleaned/eighthr_data_cleaned.csv", cleaned_eighthr_data)
    upload_to_gcs(bucket_name, "data/cleaned/onehr_data_cleaned.csv", cleaned_onehr_data)
    logging.info("Cleaned data uploaded to GCS")
