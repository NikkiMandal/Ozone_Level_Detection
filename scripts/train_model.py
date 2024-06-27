import pandas as pd
from google.cloud import storage
from google.cloud import aiplatform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
import os
import pickle
import io
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Function to load data from GCS
def load_data_from_gcs(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_string()
    return pd.read_csv(io.BytesIO(data))

def save_model_to_gcs(model, bucket_name, output_directory, model_filename):
    # Save the model locally
    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)
    
    # Upload the model to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{output_directory}/{model_filename}")
    blob.upload_from_filename(model_filename)
    logging.info("Model uploaded to GCS successfully.")

def main():
    # Load preprocessed data
    bucket_name = "ozone_level_detection"
    eighthr_data = load_data_from_gcs(bucket_name, "data/cleaned/eighthr_data_scaled.csv")
    onehr_data = load_data_from_gcs(bucket_name, "data/cleaned/onehr_data_scaled.csv")

    # Combine datasets
    df = pd.concat([eighthr_data, onehr_data])

    # Replace 'target_column' with the actual name of your target column
    target_column = 'Ozone_Level'  # Example column name; update it to match your dataset

    # Split the data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train the model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Evaluate the model
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    logging.info(f"Model accuracy: {accuracy}")

    # Save the model to the specified directory in GCS
    model_filename = "random_forest_model.pkl"
    output_directory = "model_output"  # This can be dynamically set based on input arguments or environment variables
    save_model_to_gcs(model, bucket_name, output_directory, model_filename)

    # Initialize Vertex AI
    aiplatform.init(project='ozone-level-detection', location='us-central1')

    # Upload the model to Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name="ozone_level_detection_random_forest",
        artifact_uri=f"gs://{bucket_name}/{output_directory}/{model_filename}",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
    )
    logging.info("Model registered in Vertex AI successfully.")

if __name__ == "__main__":
    main()
