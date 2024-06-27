# Ozone Level Detection Dataset

## Overview
Our dataset focuses on detecting ozone levels and is sourced from the UCI Machine Learning Repository. It encompasses measurements of various atmospheric conditions that affect ozone concentration. This dataset plays a crucial role in the development of models for predicting ozone levels based on environmental variables.

## Dataset Introduction
The dataset comprises a diverse collection of measurements, including temperature, humidity, wind speed, and other atmospheric conditions, enhancing the model's robustness across various contexts and scenarios. This structured data provides a comprehensive foundation for analyzing and predicting ozone levels.

## Dataset Source
- **Repository**: UCI Machine Learning Repository
- **URL**: [UCI Ozone Level Detection Data Set](https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection)

## Data Card

### Dataset Description
The dataset includes 72 features representing meteorological measurements and indicators. These features are used to classify the ozone levels into two categories: high or low.

- **Features**: `V1` to `V72`
- **Target Variable**: `Class` (0 for low ozone levels, 1 for high ozone levels)
- **Total Samples**: 2536

### File Naming Convention
The dataset files are named based on their type and processing stage. Here’s a brief explanation:

#### Raw Data Files
- `ozone_data.csv`: Original dataset from UCI.
- `feature_description.txt`: Descriptions for each feature in the dataset.
- `data_dictionary.pdf`: Detailed data dictionary explaining the dataset's metadata and feature descriptions.

#### Processed Data Files
- `cleaned_data.csv`: Data after cleaning using K-Nearest Neighbors (KNN) for missing value imputation.
- `normalized_data.csv`: Data after normalization for consistent scaling.
- `smote_data.csv`: Data after Synthetic Minority Over-sampling Technique (SMOTE) analysis to handle class imbalance.
- `cross_validated_data.pkl`: Data after cross-validation, ready for model evaluation.
- `train_data.csv`: Training set.
- `test_data.csv`: Testing set.
- `transformed_data.pkl`: Data after transformations (e.g., encoding).

#### Scripts
- `data_cleaning.py`: Script for cleaning the raw data using KNN for missing value imputation.
- `data_normalization.py`: Script for normalizing the data.
- `smote_analysis.py`: Script for applying SMOTE to handle class imbalance.
- `cross_validation.py`: Script for performing cross-validation on the dataset.
- `data_split.py`: Script for splitting data into training and testing sets.

### Sample File Naming Explanation
For instance, `smote_data.csv` indicates data that has undergone SMOTE analysis to balance the classes.

## Data Rights and Privacy
- **Data Compliance**: The dataset adheres to General Data Protection Regulation (GDPR) standards, ensuring the highest levels of data protection and privacy.
- **Privacy Considerations**: All personally identifiable information (PII) has been meticulously anonymized to safeguard privacy.

## Data Pipeline Components

### Data Generation
The initial stage involves generating new data periodically. Given our total of 2536 samples, a pipeline task is created to simulate the availability of new data at regular intervals. This ensures a consistent flow of fresh data for model training and evaluation.

### Data Splitting and Transformation
For each new dataset generated, we perform a split into training, testing, and validation sets. We also transform the data by scaling, encoding, and feature extraction, ensuring the datasets are ready for efficient model training and evaluation. These transformed datasets are stored in a structured format for easy access and scalability.

## Usage

### Loading Data
To load the raw dataset:
```python
import pandas as pd

data = pd.read_csv('data/raw/ozone_data.csv')
