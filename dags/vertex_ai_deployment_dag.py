from datetime import datetime, timedelta
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
import pandas as pd
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import ModelServiceClient
from airflow.hooks.base import BaseHook
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_gmail(recipient, subject, body, **kwargs):
    smtp_conn = BaseHook.get_connection('smtp_default')
    sender = smtp_conn.login
    password = smtp_conn.password

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP(host=smtp_conn.host, port=smtp_conn.port)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()
    logging.info(f"Email sent successfully to {recipient} with subject '{subject}'")

def load_csv_from_gcs(bucket_name, object_name):
    logging.info(f"Loading data from {bucket_name}/{object_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    data = blob.download_as_string()
    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
    return df

def save_metrics_to_gcs(bucket_name, file_path, metrics):
    logging.info(f"Saving metrics to {bucket_name}/{file_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.upload_from_string(metrics)

def collect_metrics(y_true, y_pred, dataset_name):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    metrics = [
        f"{dataset_name} Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n",
        f"{dataset_name} Confusion Matrix:\n{conf_matrix}\n",
        f"{dataset_name} Classification Report:\n{report}\n"
    ]
    return metrics

def train_evaluate_model(**kwargs):
    bucket_name = 'ozone_level_detection'
    
    # Load data
    train_eighthr = load_csv_from_gcs(bucket_name, 'SMOTE_analysis/eighthr_train_resampled.csv')
    val_eighthr = load_csv_from_gcs(bucket_name, 'SMOTE_analysis/eighthr_train.csv')
    train_onehr = load_csv_from_gcs(bucket_name, 'SMOTE_analysis/onehr_train_resampled.csv')
    val_onehr = load_csv_from_gcs(bucket_name, 'SMOTE_analysis/onehr_train.csv')
    test_eighthr = load_csv_from_gcs(bucket_name, 'SMOTE_analysis/eighthr_test.csv')
    test_onehr = load_csv_from_gcs(bucket_name, 'SMOTE_analysis/onehr_test.csv')
    
    # Initialize classifier
    clf = RandomForestClassifier()
    metrics = []
    
    # Training and validation for Eighthr data
    target_column = 'Ozone_Level'
    X_train_eighthr = train_eighthr.drop(target_column, axis=1)
    y_train_eighthr = train_eighthr[target_column]
    X_val_eighthr = val_eighthr.drop(target_column, axis=1)
    y_val_eighthr = val_eighthr[target_column]
    
    clf.fit(X_train_eighthr, y_train_eighthr)
    
    y_train_pred_eighthr = clf.predict(X_train_eighthr)
    y_val_pred_eighthr = clf.predict(X_val_eighthr)
    
    metrics += collect_metrics(y_train_eighthr, y_train_pred_eighthr, 'Eighthr Train')
    metrics += collect_metrics(y_val_eighthr, y_val_pred_eighthr, 'Eighthr Validation')
    
    # Evaluate on test set for Eighthr data
    X_test_eighthr = test_eighthr.drop(target_column, axis=1)
    y_test_eighthr = test_eighthr[target_column]
    
    y_test_eighthr_pred = clf.predict(X_test_eighthr)
    metrics += collect_metrics(y_test_eighthr, y_test_eighthr_pred, 'Eighthr Test')
    
    # Training and validation for Onehr data
    X_train_onehr = train_onehr.drop(target_column, axis=1)
    y_train_onehr = train_onehr[target_column]
    X_val_onehr = val_onehr.drop(target_column, axis=1)
    y_val_onehr = val_onehr[target_column]
    
    clf.fit(X_train_onehr, y_train_onehr)
    
    y_train_pred_onehr = clf.predict(X_train_onehr)
    y_val_pred_onehr = clf.predict(X_val_onehr)
    
    metrics += collect_metrics(y_train_onehr, y_train_pred_onehr, 'Onehr Train')
    metrics += collect_metrics(y_val_onehr, y_val_pred_onehr, 'Onehr Validation')
    
    # Evaluate on test set for Onehr data
    X_test_onehr = test_onehr.drop(target_column, axis=1)
    y_test_onehr = test_onehr[target_column]
    
    y_test_onehr_pred = clf.predict(X_test_onehr)
    metrics += collect_metrics(y_test_onehr, y_test_onehr_pred, 'Onehr Test')
    
    # Combine all metrics into a single string
    metrics_str = "\n".join(metrics)
    
    # Save metrics to GCS
    save_metrics_to_gcs(bucket_name, 'metrics/metrics.txt', metrics_str)

    # Save the model
    model_local_path = '/tmp/model.pkl'  # Change the filename to model.pkl
    joblib.dump(clf, model_local_path)

    # Upload the model to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('models/model.pkl')  # Change the filename to model.pkl
    blob.upload_from_filename(model_local_path)
    logging.info(f'Model uploaded to GCS at {bucket_name}/models/model.pkl.')

def validate_gcs_file(bucket_name, file_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    if not blob.exists():
        raise FileNotFoundError(f"The file gs://{bucket_name}/{file_path} does not exist.")
    logging.info(f"The file gs://{bucket_name}/{file_path} is verified to exist.")

def deploy_model_to_vertex_ai(**kwargs):
    project_id = 'ozone-level-detection'
    location = 'us-central1'  # Use a supported region
    model_display_name = 'random_forest_model'
    bucket_name = 'ozone_level_detection'
    model_gcs_path = f'gs://{bucket_name}/models/'  # Directory containing the model artifacts
    api_endpoint = "us-central1-aiplatform.googleapis.com"

    logging.info(f'Attempting to upload model from {model_gcs_path} to Vertex AI.')

    client_options = {"api_endpoint": api_endpoint}
    client = ModelServiceClient(client_options=client_options)
    
    model = {
        "display_name": model_display_name,
        "artifact_uri": model_gcs_path,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
            "command": [],
            "args": [],
            "env": [],
            "ports": [],
            "predict_route": "",
            "health_route": "",
        },
    }
    
    parent = f"projects/{project_id}/locations/{location}"
    response = client.upload_model(parent=parent, model=model)
    logging.info(f"Long running operation: {response.operation.name}")
    
    upload_model_response = response.result(timeout=1800)
    logging.info(f"upload_model_response: {upload_model_response}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 24),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'vertex_ai_deployment_dag',
    default_args=default_args,
    description='Pipeline for ozone level detection using Airflow',
    schedule_interval=None,
)

train_evaluate_task = PythonOperator(
    task_id='train_evaluate_model',
    python_callable=train_evaluate_model,
    provide_context=True,
    dag=dag,
)

validate_model_task = PythonOperator(
    task_id='validate_gcs_file',
    python_callable=validate_gcs_file,
    op_kwargs={'bucket_name': 'ozone_level_detection', 'file_path': 'models/model.pkl'},
    provide_context=True,
    dag=dag,
)

deploy_model_task = PythonOperator(
    task_id='deploy_model_to_vertex_ai',
    python_callable=deploy_model_to_vertex_ai,
    provide_context=True,
    dag=dag,
)

email_task = PythonOperator(
    task_id='send_email',
    python_callable=send_email_gmail,
    op_kwargs={
        'recipient': 'your_email@example.com',
        'subject': 'Model Training and Deployment Complete',
        'body': 'The Random Forest model has been trained, evaluated, and deployed to Vertex AI. Metrics have been uploaded to GCS.',
    },
    dag=dag,
)

train_evaluate_task >> validate_model_task >> deploy_model_task >> email_task
