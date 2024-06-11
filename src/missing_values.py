# preprocess_data.py
import pandas as pd
import numpy as np
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
    # Convert Date column to datetime and then to date
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce').dt.date

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
    
    # Save cleaned data locally to a new Excel workbook with specified date format
    with pd.ExcelWriter('missing_values_wala_excel.xlsx', engine='xlsxwriter') as writer:
        cleaned_eighthr_data.to_excel(writer, sheet_name='Eighthr Data', index=False)
        cleaned_onehr_data.to_excel(writer, sheet_name='Onehr Data', index=False)

        # Format the Date column in both sheets
        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'mm/dd/yyyy'})
        
        # Format the date columns in the first sheet
        worksheet = writer.sheets['Eighthr Data']
        date_col_idx = cleaned_eighthr_data.columns.get_loc('Date')
        worksheet.set_column(date_col_idx, date_col_idx, None, date_format)
        
        # Format the date columns in the second sheet
        worksheet = writer.sheets['Onehr Data']
        date_col_idx = cleaned_onehr_data.columns.get_loc('Date')
        worksheet.set_column(date_col_idx, date_col_idx, None, date_format)

    print("Data has been processed and saved to 'processed_data.xlsx'.")
