# Ozone Level Detection Using Machine Learning Pipelines

This project aims to analyze and predict ozone levels to improve air quality management. Using historical data from the Houston, Galveston, and Brazoria areas, we will develop a model to detect peak ozone levels. The model will help in timely intervention and reducing health risks associated with high ozone levels.

The dataset can be accessed via this link:
[UCI Dataset Link](https://archive.ics.uci.edu/dataset/172/ozone+level+detection)

## Publisher of the Dataset
The UCI Machine Learning Repository is a collection of datasets widely used in the machine learning community. Hosted by the University of California, Irvine, it offers datasets from diverse domains, such as biology, medicine, and social sciences. These datasets are curated and documented, providing researchers with valuable resources for benchmarking algorithms and developing new techniques in machine learning and data mining.

## Dataset Introduction
The dataset includes records from 1998 to 2004 for eight-hour and one-hour peak ozone levels. It contains 72 features and 2536 instances, focusing on atmospheric conditions like temperature, wind speed, and humidity.

## Coverage
Those data were collected from 1998 to 2004 at the Houston, Galveston, and Brazoria area.

## Data Rights and Privacy
The dataset is publicly available for research purposes with no specific data privacy concerns, but general data protection principles should be followed.

## Columns and Tables
We have two different subsets of data where one of them records 8 hours of ozone activity whereas the other records ozone activity for 1 hour.

Below is a detailed explanation of each column in the dataset:

- **WSR0 to WSR7 (Wind Speed Readings)**:
  - **Type**: Continuous
  - **Description**: These columns represent wind speed readings taken at various times throughout the day. Each column corresponds to a specific time period.

- **WSR_PK (Peak Wind Speed)**:
  - **Type**: Continuous
  - **Description**: The peak wind speed during the day, calculated as the resultant or average of the wind vector.

- **WSR_AV (Average Wind Speed)**:
  - **Type**: Continuous
  - **Description**: The average wind speed throughout the day.

- **T_PK (Peak Temperature)**:
  - **Type**: Continuous
  - **Description**: The peak temperature recorded during the day.

- **T_AV (Average Temperature)**:
  - **Type**: Continuous
  - **Description**: The average temperature recorded throughout the day.

- **T85, RH85, U85, V85, HT85 (850 hPa Level Readings)**:
  - **Type**: Continuous
  - **Description**: These columns represent various atmospheric readings at the 850 hPa level, roughly 1500 meters above sea level. They include temperature (T85), relative humidity (RH85), east-west wind component (U85), north-south wind component (V85), and geopotential height (HT85).

- **T70, RH70, U70, V70, HT70 (700 hPa Level Readings)**:
  - **Type**: Continuous
  - **Description**: Similar to the 850 hPa readings but taken at the 700 hPa level, roughly 3100 meters above sea level.

- **T50, RH50, U50, V50, HT50 (500 hPa Level Readings)**:
  - **Type**: Continuous
  - **Description**: These readings are taken at the 500 hPa level, roughly 5500 meters above sea level.

- **KI (K-Index)**:
  - **Type**: Continuous
  - **Description**: A measure used in meteorology to estimate the likelihood of thunderstorms.

- **TT (Total Totals Index)**:
  - **Type**: Continuous
  - **Description**: An index used to predict severe weather conditions, including thunderstorms.

- **SLP (Sea Level Pressure)**:
  - **Type**: Continuous
  - **Description**: The atmospheric pressure at sea level.

- **SLP_ (Change in Sea Level Pressure)**:
  - **Type**: Continuous
  - **Description**: The change in sea level pressure from the previous day.

- **Precp (Precipitation)**:
  - **Type**: Continuous
  - **Description**: The amount of precipitation measured.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Google Cloud SDK
- Apache Airflow
- Pandas
- Numpy
- Scikit-learn
- Pandera (for schema validation)
- Google Cloud Storage (GCS) buckets for raw and processed data

### Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/SiddanthEmani/Ozone_Level_Detection.git
    cd Ozone_Level_Detection
    ```

2. **Create a virtual environment and activate it**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up Google Cloud credentials**:
    - Ensure your GCP credentials are set up correctly and accessible by the environment.

5. **Initialize DVC**:
    ```bash
    dvc init
    ```

## Data Preprocessing Pipeline

### Remove Missing Values
#### DAG: `remove_missing_values_dag.py`
- **Tasks**:
  - Download raw data from GCS.
  - Remove missing values.
  - Upload cleaned data to GCS.

### Handle Outliers
#### DAG: `handle_outliers_dag.py`
- **Tasks**:
  - Download cleaned data from previous step.
  - Remove outliers.
  - Upload data without outliers to GCS.

### Feature Scaling
#### DAG: `feature_scaling_dag.py`
- **Tasks**:
  - Download data without outliers from previous step.
  - Scale numerical features.
  - Upload scaled data to GCS.

### Final Cleaning
#### DAG: `final_clean_data_dag.py`
- **Tasks**:
  - Download scaled data from previous step.
  - Perform final cleaning (e.g., remove duplicates).
  - Upload fully cleaned data to GCS.

## Modular Syntax/Code
- Each preprocessing step is encapsulated in its own DAG and Python functions, ensuring modularity and reusability.

## Pipeline Orchestration
- Apache Airflow is used to orchestrate the preprocessing pipeline, with dependencies defined between tasks and DAGs.

## Proper Tracking, Logging
- Logging is implemented in each function to track the progress and capture errors.

## Data Version Control
- DVC is integrated to version control raw and processed datasets.

## Pipeline Flow Optimization
- Use Airflow's Gantt chart to identify and optimize bottlenecks in the pipeline.

## Schema and Statistics Generation
- Integrate Pandera for schema validation and data profiling.

## Anomalies Detection and Alert Generation
- Implement anomaly detection and alerting mechanisms within the Airflow pipeline.

## Example Commands

### Run DVC Commands
1. **Add data to DVC**:
    ```bash
    dvc add data/raw/eighthr_data.csv
    dvc add data/raw/onehr_data.csv
    dvc add data/cleaned/eighthr_data_cleaned.csv
    dvc add data/cleaned/onehr_data_cleaned.csv
    ```

2. **Create a DVC pipeline**:
    ```bash
    dvc run -n preprocess -d data/raw/eighthr_data.csv -o data/cleaned/eighthr_data_cleaned.csv python preprocess_data.py
    ```

### Run Airflow DAGs
1. **Start Airflow**:
    ```bash
    airflow db init
    airflow webserver --port 8080
    airflow scheduler
    ```

2. **Enable DAGs**:
    - Access the Airflow web UI and enable the DAGs for each preprocessing step.

3. **Trigger DAGs**:
    - Trigger the DAGs manually from the Airflow UI to start the preprocessing pipeline.

## Troubleshooting
- **Airflow Issues**: Ensure Airflow is running and all DAGs are correctly configured.
- **GCS Permissions**: Verify that the GCP credentials have the necessary permissions to read from and write to the GCS buckets.
- **Data Issues**: Check the logs for any data-related errors during preprocessing.

## Future Steps
- Implement model training and evaluation DAGs.
- Integrate anomaly detection and alerting mechanisms.
- Enhance documentation with additional details on model implementation.

---
