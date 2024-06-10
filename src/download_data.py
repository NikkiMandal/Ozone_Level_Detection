import requests  # For making HTTP requests to download data
import pandas as pd  # For data manipulation and analysis
from google.cloud import storage  # GCP library to interact with Google Cloud Storage
from io import StringIO

def download_data(url):
    """
    Download data from a specified URL and return the content.
    """
    response = requests.get(url)  # Make a GET request to the URL
    response.raise_for_status()  # Raise an error if the request was unsuccessful
    return response.content  # Return the content of the response

def upload_to_gcs(bucket_name, destination_blob_name, data):
    """
    Upload data to a specified Google Cloud Storage bucket.
    """
    storage_client = storage.Client()  # Initialize a client to interact with Google Cloud Storage
    bucket = storage_client.bucket(bucket_name)  # Get the bucket object by name
    blob = bucket.blob(destination_blob_name)  # Create a blob object with the specified name in the bucket
    
    blob.upload_from_string(data)  # Upload the data to the blob
    print(f"Uploaded to {bucket_name}/{destination_blob_name}")  # Confirm the upload

def process_files(data_file_content, names_file_content, target_column_name):
    """
    Process the data and names files to prepare a DataFrame with appropriate column names.
    """
    column_names = []  # List to hold the column names
    metadata = {}  # Dictionary to hold metadata
    lines = names_file_content.decode('utf-8').splitlines()  # Decode and split the names file into lines

    metadata_end_index = 0
    for i, line in enumerate(lines):
        if 'Date:' in line:  # Assuming 'Date:' indicates the start of column definitions
            metadata_end_index = i
            break
        metadata[f"Metadata_{i}"] = line.strip()  # Store each line as metadata

    for line in lines[metadata_end_index:]:
        if line.strip():  # Skip empty lines
            parts = line.split(':')
            column_name = parts[0].strip()  # Extract the column name
            if len(parts) > 1:
                metadata[column_name] = parts[1].strip()  # Store additional metadata if present
            column_names.append(column_name)  # Add the column name to the list

    # Add the target class name
    column_names.append(target_column_name)

    print(f"Column Names:", column_names)
    print(f"Metadata:", metadata)

    # Load the data file into a DataFrame without headers
    data_content = data_file_content.decode('utf-8')
    data = pd.read_csv(StringIO(data_content), header=None)

    # Adjust column names to match data
    data_columns_count = len(data.columns)
    names_columns_count = len(column_names)

    if data_columns_count != names_columns_count:
        if data_columns_count > names_columns_count:
            for i in range(data_columns_count - names_columns_count):
                column_names.insert(-1, f'Unnamed_{i+1}')  # Add placeholder column names
        elif data_columns_count < names_columns_count:
            column_names = column_names[:data_columns_count]  # Truncate column names

    data.columns = column_names  # Assign the column names to the DataFrame

    print(f"Processed Data:")
    print(data.head())
    print(f"Data Columns:", data.columns)
    print(f"Shape of Data:", data.shape)

    return data  # Return the processed DataFrame

if __name__ == "__main__":
    # Define URLs for the data and names files
    urls = {
        "eighthr_data": "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/eighthr.data",
        "onehr_data": "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.data",
        "eighthr_names": "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/eighthr.names",
        "onehr_names": "https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.names"
    }

    bucket_name = "ozone_level_detection"  # Replace with your GCS bucket name

    # Process and upload the eighthr files
    eighthr_data_file_content = download_data(urls['eighthr_data'])
    eighthr_names_file_content = download_data(urls['eighthr_names'])
    eighthr_data = process_files(eighthr_data_file_content, eighthr_names_file_content, 'Ozone_Level')
    
    # Convert DataFrame to CSV format and upload to GCS
    eighthr_data_csv = eighthr_data.to_csv(index=False)
    upload_to_gcs(bucket_name, "processed/eighthr_data.csv", eighthr_data_csv)

    # Process and upload the onehr files
    onehr_data_file_content = download_data(urls['onehr_data'])
    onehr_names_file_content = download_data(urls['onehr_names'])
    onehr_data = process_files(onehr_data_file_content, onehr_names_file_content, 'Ozone_Level')
    
    # Convert DataFrame to CSV format and upload to GCS
    onehr_data_csv = onehr_data.to_csv(index=False)
    upload_to_gcs(bucket_name, "processed/onehr_data.csv", onehr_data_csv)
