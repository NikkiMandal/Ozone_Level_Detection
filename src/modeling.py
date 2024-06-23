import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Define the GCS paths
train_eighthr_path = 'gs://ozone_level_detection/data/processed/train_eighthr.csv'
train_onehr_path = 'gs://ozone_level_detection/data/processed/train_onehr.csv'
test_eighthr_path = 'gs://ozone_level_detection/data/processed/test_eighthr.csv'
test_onehr_path = 'gs://ozone_level_detection/data/processed/test_onehr.csv'

# Load the datasets
train_eighthr = pd.read_csv(train_eighthr_path)
train_onehr = pd.read_csv(train_onehr_path)
test_eighthr = pd.read_csv(test_eighthr_path)
test_onehr = pd.read_csv(test_onehr_path)

# Split the datasets into features and target variable
X_train_eighthr = train_eighthr.iloc[:, :-1]
y_train_eighthr = train_eighthr.iloc[:, -1]
X_test_eighthr = test_eighthr.iloc[:, :-1]
y_test_eighthr = test_eighthr.iloc[:, -1]

X_train_onehr = train_onehr.iloc[:, :-1]
y_train_onehr = train_onehr.iloc[:, -1]
X_test_onehr = test_onehr.iloc[:, :-1]
y_test_onehr = test_onehr.iloc[:, -1]

# Train logistic regression model on eighthr data
model_eighthr = LogisticRegression()
model_eighthr.fit(X_train_eighthr, y_train_eighthr)

# Train logistic regression model on onehr data
model_onehr = LogisticRegression()
model_onehr.fit(X_train_onehr, y_train_onehr)

# Evaluate on eighthr test data
y_pred_eighthr = model_eighthr.predict(X_test_eighthr)

# Print evaluation metrics for eighthr
print("Eighthr Test Accuracy:", accuracy_score(y_test_eighthr, y_pred_eighthr))
print("Eighthr Precision:", precision_score(y_test_eighthr, y_pred_eighthr, average='weighted'))
print("Eighthr Recall:", recall_score(y_test_eighthr, y_pred_eighthr, average='weighted'))
print("Eighthr F1 Score:", f1_score(y_test_eighthr, y_pred_eighthr, average='weighted'))
print("Eighthr Confusion Matrix:\n", confusion_matrix(y_test_eighthr, y_pred_eighthr))
print("Eighthr Classification Report:\n", classification_report(y_test_eighthr, y_pred_eighthr))

# Evaluate on onehr test data
y_pred_onehr = model_onehr.predict(X_test_onehr)

# Print evaluation metrics for onehr
print("Onehr Test Accuracy:", accuracy_score(y_test_onehr, y_pred_onehr))
print("Onehr Precision:", precision_score(y_test_onehr, y_pred_onehr, average='weighted'))
print("Onehr Recall:", recall_score(y_test_onehr, y_pred_onehr, average='weighted'))
print("Onehr F1 Score:", f1_score(y_test_onehr, y_pred_onehr, average='weighted'))
print("Onehr Confusion Matrix:\n", confusion_matrix(y_test_onehr, y_pred_onehr))
print("Onehr Classification Report:\n", classification_report(y_test_onehr, y_pred_onehr))