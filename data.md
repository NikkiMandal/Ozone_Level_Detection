# Ozone Level Detection Dataset

## Overview
Our dataset focuses on detecting ground-level ozone concentrations and is sourced from the UCI Machine Learning Repository. It encompasses measurements of various atmospheric conditions that influence ozone levels. This dataset is instrumental in developing predictive models for ozone levels based on environmental variables.

## Dataset Introduction
The dataset includes measurements collected from 1998 to 2004 in the Houston, Galveston, and Brazoria areas. It contains two distinct datasets: the eight-hour peak dataset (`eighthr.data`) and the one-hour peak dataset (`onehr.data`). These datasets enhance model robustness by providing detailed atmospheric data across multiple years and locations.

## Dataset Source
- **Repository**: UCI Machine Learning Repository
- **URL**: [UCI Ozone Level Detection Data Set](https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection)

## Data Card

### Dataset Description
The dataset features 72 attributes representing meteorological measurements and indicators, used to classify ozone levels into two categories: high or low. It supports multivariate, sequential, and time-series analyses.

- **# Instances**: 2536
- **# Features**: 72
- **Feature Type**: Real
- **Subject Area**: Climate and Environment
- **Associated Tasks**: Classification

### Dataset Information
- **Time Period**: 1998 - 2004
- **Locations**: Houston, Galveston, and Brazoria areas

**Additional Variable Information**:
- **O3**: Local ozone peak prediction.
- **Upwind**: Upwind ozone background level.
- **EmFactor**: Precursor emissions-related factor.
- **Tmax**: Maximum temperature in degrees Fahrenheit.
- **Tb**: Base temperature where net ozone production begins (50°F).
- **SRd**: Solar radiation total for the day.
- **WSa**: Wind speed near sunrise (using 09-12 UTC forecast mode).
- **WSp**: Wind speed mid-day (using 15-21 UTC forecast mode).

### File Naming Convention
The dataset files are named based on their type and processing stage. Here’s a brief explanation:

#### Raw Data Files
- `eighthr.data`: Eight-hour peak ozone dataset.
- `onehr.data`: One-hour peak ozone dataset.
- `feature_description.txt`: Descriptions for each feature in the dataset.
- `data_dictionary.pdf`: Detailed data dictionary explaining the dataset's metadata and feature descriptions.

#### Processed Data Files
- `eighthr_data_KNN.csv` and 'onehr_data_KNN.csv': Data after cleaning using K-Nearest Neighbors (KNN) for missing value imputation.
- `eighthr_data_normalized.csv` and `onehr_data_normalized.csv`: Data after normalization for consistent scaling.
- `eighthr_train_resampled.csv` and `onehr_train_resampled.csv`: Data after Synthetic Minority Over-sampling Technique (SMOTE) analysis to handle class imbalance.
- `eighthr_train_foldx.csv` and 'eighthr_val_foldx.csv': Eight-hour data after cross-validation, ready for model evaluation. (x= 1, 2, 3, 4, 5)
- `onehr_train_foldx.csv` and 'onehr_val_foldx.csv': One-hour data after cross-validation, ready for model evaluation. (x= 1, 2, 3, 4, 5)

#### Scripts
- `cleaned_KNN_dag.py`: Script for cleaning the raw data using KNN for missing value imputation.
- `cleaned_normalization_dag.py`: Script for normalizing the data.
- `SMOTE_analysis_DAG.py`: Script for applying SMOTE to handle class imbalance.
- `cross_validation_dag.py`: Script for performing cross-validation on the dataset.

### Feature Details
All attributes start with the letter 'T' indicating temperature at different times or 'WS' indicating wind speed at various times. The features include:

- **WSR_PK**: Continuous. Peak wind speed (resultant).
- **WSR_AV**: Continuous. Average wind speed.
- **T_PK**: Continuous. Peak temperature.
- **T_AV**: Continuous. Average temperature.
- **T85**: Continuous. Temperature at 850 hPa level (approx. 1500 m height).
- **RH85**: Continuous. Relative Humidity at 850 hPa.
- **U85**: Continuous. East-West direction wind at 850 hPa.
- **V85**: Continuous. North-South direction wind at 850 hPa.
- **HT85**: Continuous. Geopotential height at 850 hPa.
- **T70**: Continuous. Temperature at 700 hPa level (approx. 3100 m height).
- **RH70**: Continuous. Relative Humidity at 700 hPa.
- **U70**: Continuous. East-West direction wind at 700 hPa.
- **V70**: Continuous. North-South direction wind at 700 hPa.
- **HT70**: Continuous. Geopotential height at 700 hPa.
- **T50**: Continuous. Temperature at 500 hPa level (approx. 5500 m height).
- **RH50**: Continuous. Relative Humidity at 500 hPa.
- **U50**: Continuous. East-West direction wind at 500 hPa.
- **V50**: Continuous. North-South direction wind at 500 hPa.
- **HT50**: Continuous. Geopotential height at 500 hPa.
- **KI**: Continuous. K-Index.
- **TT**: Continuous. T-Totals.
- **SLP**: Continuous. Sea level pressure.
- **SLP_**: Continuous. Change in sea level pressure from previous day.
- **Precp**: Continuous. Precipitation.

**Naming Convention**:
The names of the files reflect the type of dataset (eight-hour peak or one-hour peak) and their processing stages.

## Data Rights and Privacy
- **Data Compliance**: The dataset adheres to General Data Protection Regulation (GDPR) standards, ensuring the highest levels of data protection and privacy.
- **Privacy Considerations**: All personally identifiable information (PII) has been meticulously anonymized to safeguard privacy.

## Data Pipeline Components


### Data Splitting and Preparation
The dataset undergoes several critical stages of preparation to ensure it is ready for machine learning tasks:
1. **Data Cleaning and Transformation**:
   - **Data Cleaning**:
     - Utilizing **K-Nearest Neighbors (KNN)** for imputing missing values, ensuring a complete dataset without gaps.
   - **Normalization**:
     - Standardizing the dataset to ensure consistent feature scaling across all variables.
   - **SMOTE Analysis**:
     - Applying **Synthetic Minority Over-sampling Technique (SMOTE)** to address class imbalance, generating synthetic samples for minority classes.
   - **Cross-Validation**:
     - Performing **cross-validation** to evaluate the model's performance robustly, leveraging data splitting into multiple folds for training and testing.
   
2. **Data Splitting**:
   - Finally, the dataset is divided into training, testing, and validation sets:
     - **Training Set (`train_df`)**: Comprising 70% of the total dataset, used for model training.
     - **Validation Set (`val_df`)**: Consisting of 10% of the total dataset, used for parameter tuning and validation.
     - **Testing Set (`test_df`)**: Encompassing 20% of the total dataset, used to evaluate the model's performance on unseen data.
This comprehensive approach ensures that the dataset is meticulously prepared and appropriately split for effective machine learning model development and evaluation.

### Loading Data

The data pipeline involves loading data from and uploading data to Google Cloud Storage (GCS). We use Python scripts with Google Cloud's `storage` library to handle these operations. The scripts for loading and uploading data are as follows:

#### Loading Data from GCS
We load CSV files from GCS into a Pandas DataFrame using the `load_csv_from_gcs` function. This function reads the data from GCS and converts it into a DataFrame for further processing.

```python
import io
import pandas as pd
from google.cloud import storage
import logging

def load_csv_from_gcs(bucket_name, object_name):
    """Load a CSV file from Google Cloud Storage into a Pandas DataFrame."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    data = blob.download_as_string()
    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
    return df

def upload_to_gcs(bucket_name, destination_blob_name, df):
    """Upload a pandas DataFrame to GCS."""
    try:
        if df is not None:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            data_bytes = df.to_csv(index=False).encode()
            blob.upload_from_string(data_bytes, content_type='text/csv')
            logging.info(f"Uploaded data to {bucket_name}/{destination_blob_name}")
        else:
            raise ValueError("No data provided to upload.")
    except Exception as e:
        logging.error(f"Error uploading to GCS: {e}")
        raise

# Load data from GCS
data_df = load_csv_from_gcs('my_bucket', 'data/raw_data.csv')

# Process data (example processing step)
processed_df = process_data(data_df)

# Upload processed data back to GCS
upload_to_gcs('my_bucket', 'data/processed_data.csv', processed_df)
