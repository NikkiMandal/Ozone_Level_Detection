import numpy as np
import pandas as pd
import io
from google.cloud import storage
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.base import BaseHook
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load data from Google Cloud Storage
def load_csv_from_gcs(bucket_name, object_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    data = blob.download_as_string()
    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
    return df

def load_datasets():
    global train_eighthr, val_eighthr, test_eighthr, train_onehr, val_onehr, test_onehr
    train_eighthr = load_csv_from_gcs('ozone_level_detection', 'data/cleaned/eighthr_train_data.csv')
    val_eighthr = load_csv_from_gcs('ozone_level_detection', 'data/cleaned/eighthr_val_data.csv')
    test_eighthr = load_csv_from_gcs('ozone_level_detection', 'data/cleaned/eighthr_test_data.csv')
    train_onehr = load_csv_from_gcs('ozone_level_detection', 'data/cleaned/onehr_train_data.csv')
    val_onehr = load_csv_from_gcs('ozone_level_detection', 'data/cleaned/onehr_val_data.csv')
    test_onehr = load_csv_from_gcs('ozone_level_detection', 'data/cleaned/onehr_test_data.csv')

def preprocess_datasets():
    global X_train_eighthr, y_train_eighthr, X_val_eighthr, y_val_eighthr, X_test_eighthr, y_test_eighthr
    global X_train_onehr, y_train_onehr, X_val_onehr, y_val_onehr, X_test_onehr, y_test_onehr
    
    train_eighthr.drop(columns=['Date'], inplace=True)
    val_eighthr.drop(columns=['Date'], inplace=True)
    test_eighthr.drop(columns=['Date'], inplace=True)
    train_onehr.drop(columns=['Date'], inplace=True)
    val_onehr.drop(columns=['Date'], inplace=True)
    test_onehr.drop(columns=['Date'], inplace=True)

    X_train_eighthr = train_eighthr.iloc[:, :-1].values
    y_train_eighthr = train_eighthr.iloc[:, -1].values
    X_val_eighthr = val_eighthr.iloc[:, :-1].values
    y_val_eighthr = val_eighthr.iloc[:, -1].values
    X_test_eighthr = test_eighthr.iloc[:, :-1].values
    y_test_eighthr = test_eighthr.iloc[:, -1].values

    X_train_onehr = train_onehr.iloc[:, :-1].values
    y_train_onehr = train_onehr.iloc[:, -1].values
    X_val_onehr = val_onehr.iloc[:, :-1].values
    y_val_onehr = val_onehr.iloc[:, -1].values
    X_test_onehr = test_onehr.iloc[:, :-1].values
    y_test_onehr = test_onehr.iloc[:, -1].values

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization_strength=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization_strength = regularization_strength
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + self.regularization_strength * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_class)

    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'regularization_strength': self.regularization_strength
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

def train_and_evaluate_model():
    global best_model_eighthr, best_model_onehr
    
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'n_iterations': [100, 500, 1000],
        'regularization_strength': [0.01, 0.1, 1.0]
    }

    model = CustomLogisticRegression()

    grid_search_eighthr = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search_onehr = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

    grid_search_eighthr.fit(X_train_eighthr, y_train_eighthr)
    grid_search_onehr.fit(X_train_onehr, y_train_onehr)

    best_model_eighthr = grid_search_eighthr.best_estimator_
    best_model_onehr = grid_search_onehr.best_estimator_

    metrics = ""
    
    for dataset_name, (X, y, model) in {
        'Eighthr Train': (X_train_eighthr, y_train_eighthr, best_model_eighthr),
        'Eighthr Validation': (X_val_eighthr, y_val_eighthr, best_model_eighthr),
        'Eighthr Test': (X_test_eighthr, y_test_eighthr, best_model_eighthr),
        'Onehr Train': (X_train_onehr, y_train_onehr, best_model_onehr),
        'Onehr Validation': (X_val_onehr, y_val_onehr, best_model_onehr),
        'Onehr Test': (X_test_onehr, y_test_onehr, best_model_onehr)
    }.items():
        y_pred = model.predict(X)
        metrics += f"{dataset_name} Accuracy: {accuracy_score(y, y_pred)}\n"
        metrics += f"{dataset_name} Precision: {precision_score(y, y_pred, average='weighted')}\n"
        metrics += f"{dataset_name} Recall: {recall_score(y, y_pred, average='weighted')}\n"
        metrics += f"{dataset_name} F1 Score: {f1_score(y, y_pred, average='weighted')}\n"
        metrics += f"{dataset_name} Confusion Matrix:\n{confusion_matrix(y, y_pred)}\n"
        metrics += f"{dataset_name} Classification Report:\n{classification_report(y, y_pred)}\n\n"
    
    with open('/tmp/metrics.txt', 'w') as f:
        f.write(metrics)

def detect_anomalies_and_alert():
    with open('/tmp/metrics.txt', 'r') as f:
        metrics = f.read()

    anomaly_thresholds = {
        'accuracy': 0.7,
        'precision': 0.7,
        'recall': 0.7,
        'f1_score': 0.7
    }

    anomalies = []
    for line in metrics.split('\n'):
        if 'Accuracy' in line:
            score = float(line.split(':')[-1].strip())
            if score == 1.0 or score < anomaly_thresholds['accuracy']:
                anomalies.append(line)
        elif 'Precision' in line:
            score = float(line.split(':')[-1].strip())
            if score < anomaly_thresholds['precision']:
                anomalies.append(line)
        elif 'Recall' in line:
            score = float(line.split(':')[-1].strip())
            if score < anomaly_thresholds['recall']:
                anomalies.append(line)
        elif 'F1 Score' in line:
            score = float(line.split(':')[-1].strip())
            if score < anomaly_thresholds['f1_score']:
                anomalies.append(line)

    if anomalies:
        send_email_gmail(
            content="\n".join(anomalies)
        )

def send_email_gmail(content):
    # Fetch credentials from Airflow connection
    smtp_conn = BaseHook.get_connection('smtp_default')
    sender = smtp_conn.login
    password = smtp_conn.password
    recipient = 'nikitamandal03@gmail.com'  # Replace with the recipient's email
    subject = 'Airflow Alert: Anomalies Detected!'
    body = f"This is a test email from Airflow.\n\n{content}"

    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Connect to Gmail's SMTP server
    server = smtplib.SMTP(host=smtp_conn.host, port=smtp_conn.port)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'anomaly_detection_dag',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_datasets,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_datasets,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_and_evaluate_model,
    dag=dag,
)

alert_task = PythonOperator(
    task_id='detect_and_alert',
    python_callable=detect_anomalies_and_alert,
    dag=dag,
)

# Define task dependencies
load_task >> preprocess_task >> train_task >> alert_task
