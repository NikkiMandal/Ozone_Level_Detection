import io
import pandas as pd
from google.cloud import storage
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to load data from GCS
def load_data_from_gcs(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_string()
    return pd.read_csv(io.BytesIO(data))

def main():
    # Load the model from GCS
    bucket_name = "ozone_level_detection"
    model_filename = "random_forest_model.pkl"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"models/{model_filename}")
    blob.download_to_filename(model_filename)
    
    with open(model_filename, "rb") as model_file:
        model = pickle.load(model_file)
    
    # Load preprocessed data
    eighthr_data = load_data_from_gcs(bucket_name, "data/cleaned/eighthr_data_scaled.csv")
    onehr_data = load_data_from_gcs(bucket_name, "data/cleaned/onehr_data_scaled.csv")
    
    # Combine datasets
    df = pd.concat([eighthr_data, onehr_data])
    
    # Split the data into features and target
    X = df.drop(columns=["target_column"])
    y = df["target_column"]

    # Predict and evaluate
    predictions = model.predict(X)
    report = classification_report(y, predictions)
    cm = confusion_matrix(y, predictions)
    
    logging.info("Classification Report:\n", report)
    logging.info("Confusion Matrix:\n", cm)
    
    with open("classification_report.txt", "w") as f:
        f.write(report)
        
    with open("confusion_matrix.txt", "w") as f:
        f.write(str(cm))
        
    # Upload reports to GCS
    report_blob = bucket.blob("analysis/classification_report.txt")
    report_blob.upload_from_filename("classification_report.txt")

    cm_blob = bucket.blob("analysis/confusion_matrix.txt")
    cm_blob.upload_from_filename("confusion_matrix.txt")
    
    logging.info("Model analysis reports uploaded to GCS successfully.")

if __name__ == "__main__":
    main()
