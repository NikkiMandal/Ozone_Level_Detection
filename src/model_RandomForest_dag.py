import numpy as np
import pandas as pd
import os
from google.cloud import storage
import io
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Set the environment variable for authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\Bhuvan Karthik\Downloads\ozone-level-detection-0160dba47662.json"

def load_csv_from_gcs(bucket_name, object_name):
    """Load a CSV file from Google Cloud Storage into a Pandas DataFrame."""
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
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'n_iterations': [100, 500, 1000],
        'regularization_strength': [0.01, 0.1, 1.0]
    }

    # Create an instance of the model
    model = CustomLogisticRegression()

    # Create the GridSearchCV object
    grid_search_eighthr = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search_onehr = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the data
    grid_search_eighthr.fit(X_train_eighthr, y_train_eighthr)
    grid_search_onehr.fit(X_train_onehr, y_train_onehr)

    # Get the best models
    best_model_eighthr = grid_search_eighthr.best_estimator_
    best_model_onehr = grid_search_onehr.best_estimator_

    # Evaluate the best model on the train, validation, and test sets for Eighthr data
    metrics = ""

    # Train metrics for Eighthr data
    y_train_pred_eighthr = best_model_eighthr.predict(X_train_eighthr)
    metrics += "Eighthr Train Accuracy: {}\n".format(accuracy_score(y_train_eighthr, y_train_pred_eighthr))
    metrics += "Eighthr Train Precision: {}\n".format(precision_score(y_train_eighthr, y_train_pred_eighthr, average='weighted'))
    metrics += "Eighthr Train Recall: {}\n".format(recall_score(y_train_eighthr, y_train_pred_eighthr, average='weighted'))
    metrics += "Eighthr Train F1 Score: {}\n".format(f1_score(y_train_eighthr, y_train_pred_eighthr, average='weighted'))
    metrics += "Eighthr Train Confusion Matrix:\n {}\n".format(confusion_matrix(y_train_eighthr, y_train_pred_eighthr))
    metrics += "Eighthr Train Classification Report:\n {}\n".format(classification_report(y_train_eighthr, y_train_pred_eighthr))

    # Validation metrics for Eighthr data
    y_val_pred_eighthr = best_model_eighthr.predict(X_val_eighthr)
    metrics += "Eighthr Validation Accuracy: {}\n".format(accuracy_score(y_val_eighthr, y_val_pred_eighthr))
    metrics += "Eighthr Validation Precision: {}\n".format(precision_score(y_val_eighthr, y_val_pred_eighthr, average='weighted'))
    metrics += "Eighthr Validation Recall: {}\n".format(recall_score(y_val_eighthr, y_val_pred_eighthr, average='weighted'))
    metrics += "Eighthr Validation F1 Score: {}\n".format(f1_score(y_val_eighthr, y_val_pred_eighthr, average='weighted'))
    metrics += "Eighthr Validation Confusion Matrix:\n {}\n".format(confusion_matrix(y_val_eighthr, y_val_pred_eighthr))
    metrics += "Eighthr Validation Classification Report:\n {}\n".format(classification_report(y_val_eighthr, y_val_pred_eighthr))

    # Test metrics for Eighthr data
    y_test_pred_eighthr = best_model_eighthr.predict(X_test_eighthr)
    metrics += "Eighthr Test Accuracy: {}\n".format(accuracy_score(y_test_eighthr, y_test_pred_eighthr))
    metrics += "Eighthr Test Precision: {}\n".format(precision_score(y_test_eighthr, y_test_pred_eighthr, average='weighted'))
    metrics += "Eighthr Test Recall: {}\n".format(recall_score(y_test_eighthr, y_test_pred_eighthr, average='weighted'))
    metrics += "Eighthr Test F1 Score: {}\n".format(f1_score(y_test_eighthr, y_test_pred_eighthr, average='weighted'))
    metrics += "Eighthr Test Confusion Matrix:\n {}\n".format(confusion_matrix(y_test_eighthr, y_test_pred_eighthr))
    metrics += "Eighthr Test Classification Report:\n {}\n".format(classification_report(y_test_eighthr, y_test_pred_eighthr))

    # Evaluate the best model on the train, validation, and test sets for Onehr data
    # Train metrics for Onehr data
    y_train_pred_onehr = best_model_onehr.predict(X_train_onehr)
    metrics += "Onehr Train Accuracy: {}\n".format(accuracy_score(y_train_onehr, y_train_pred_onehr))
    metrics += "Onehr Train Precision: {}\n".format(precision_score(y_train_onehr, y_train_pred_onehr, average='weighted'))
    metrics += "Onehr Train Recall: {}\n".format(recall_score(y_train_onehr, y_train_pred_onehr, average='weighted'))
    metrics += "Onehr Train F1 Score: {}\n".format(f1_score(y_train_onehr, y_train_pred_onehr, average='weighted'))
    metrics += "Onehr Train Confusion Matrix:\n {}\n".format(confusion_matrix(y_train_onehr, y_train_pred_onehr))
    metrics += "Onehr Train Classification Report:\n {}\n".format(classification_report(y_train_onehr, y_train_pred_onehr))

    # Validation metrics for Onehr data
    y_val_pred_onehr = best_model_onehr.predict(X_val_onehr)
    metrics += "Onehr Validation Accuracy: {}\n".format(accuracy_score(y_val_onehr, y_val_pred_onehr))
    metrics += "Onehr Validation Precision: {}\n".format(precision_score(y_val_onehr, y_val_pred_onehr, average='weighted'))
    metrics += "Onehr Validation Recall: {}\n".format(recall_score(y_val_onehr, y_val_pred_onehr, average='weighted'))
    metrics += "Onehr Validation F1 Score: {}\n".format(f1_score(y_val_onehr, y_val_pred_onehr, average='weighted'))
    metrics += "Onehr Validation Confusion Matrix:\n {}\n".format(confusion_matrix(y_val_onehr, y_val_pred_onehr))
    metrics += "Onehr Validation Classification Report:\n {}\n".format(classification_report(y_val_onehr, y_val_pred_onehr))

    # Test metrics for Onehr data
    y_test_pred_onehr = best_model_onehr.predict(X_test_onehr)
    metrics += "Onehr Test Accuracy: {}\n".format(accuracy_score(y_test_onehr, y_test_pred_onehr))
    metrics += "Onehr Test Precision: {}\n".format(precision_score(y_test_onehr, y_test_pred_onehr, average='weighted'))
    metrics += "Onehr Test Recall: {}\n".format(recall_score(y_test_onehr, y_test_pred_onehr, average='weighted'))
    metrics += "Onehr Test F1 Score: {}\n".format(f1_score(y_test_onehr, y_test_pred_onehr, average='weighted'))
    metrics += "Onehr Test Confusion Matrix:\n {}\n".format(confusion_matrix(y_test_onehr, y_test_pred_onehr))
    metrics += "Onehr Test Classification Report:\n {}\n".format(classification_report(y_test_onehr, y_test_pred_onehr))

    # Upload the metrics string to GCS
    client = storage.Client()
    bucket = client.bucket('ozone_level_detection')
    blob = bucket.blob('metrics/metrics.txt')
    blob.upload_from_string(metrics)

if __name__ == "__main__":
    load_datasets()
    preprocess_datasets()
    train_and_evaluate_model()
