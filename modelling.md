# Random Forest Classifier Model Overview

## Introduction
Random Forest is an ensemble learning method primarily used for classification and regression tasks. It operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

## How It Works

### Bootstrap Sampling
Random Forest starts with creating several bootstrap samples from the original dataset. Bootstrap sampling involves random sampling with replacement, meaning some data points may appear multiple times in one sample and not at all in another.

### Decision Tree Construction
For each bootstrap sample, a decision tree is grown. However, unlike standard decision trees, Random Forest introduces more randomness:
- At each node, a random subset of features is considered for splitting.
- The best split is chosen from this subset rather than from all available features, which introduces variability and reduces correlation between trees.

### Aggregation
Once all trees are constructed, the forest makes a prediction by aggregating the predictions of individual trees. For classification tasks, this is usually done by majority voting; for regression tasks, by averaging the predictions.

## Model Architecture
The Random Forest model architecture consists of the following components:
- **Input Layer:** The dataset features are input into the Random Forest model. Each feature vector represents a data point.
- **Bootstrap Sampling:** Multiple bootstrap samples are created from the input dataset. Each sample is used to train a different decision tree.
- **Decision Trees:** Each bootstrap sample is used to train a separate decision tree. Each tree is grown to the full depth or until a stopping criterion (e.g., minimum number of samples in a node) is met. During the training of each tree, a random subset of features is considered for splitting at each node.
- **Voting/Averaging:** For classification tasks, each tree in the forest votes for a class label, and the class label with the majority vote is chosen as the final prediction. For regression tasks, the predictions of all trees are averaged to produce the final prediction.
- **Output Layer:** The final prediction is outputted as the result of the Random Forest model.

## Optimizers and Training Details
While Random Forest does not use traditional optimization algorithms like gradient descent (commonly used in neural networks), it optimizes the decision trees through different mechanisms:
- **Random Feature Selection:** At each split in the decision tree, a random subset of features is considered. This process helps to reduce overfitting and ensures diverse trees.
- **Gini Impurity or Entropy:** The decision tree algorithm optimizes the splits based on criteria like Gini impurity or entropy, which measure the homogeneity of the nodes.
- **Tree Pruning:** While Random Forests typically grow trees to the maximum possible depth, tree pruning can be applied to avoid overfitting by removing branches that have little importance.
- **Hyperparameter Tuning:** GridSearchCV or RandomizedSearchCV can be used to find the optimal hyperparameters. Common hyperparameters include:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of each tree.
  - `min_samples_split`: Minimum number of samples required to split an internal node.
  - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
  - `max_features`: Number of features to consider when looking for the best split.

## Implementation Example
Here is an example implementation of a custom Random Forest classifier with hyperparameter tuning using GridSearchCV:

class CustomRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto'):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def load_csv_from_gcs(bucket_name, object_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    data = blob.download_as_string()
    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
    return df

def load_and_preprocess_data():
    train_eighthr = load_csv_from_gcs('ozone_level_detection', 'data/cleaned/eighthr_train_data.csv')
    val_eighthr = load_csv_from_gcs('ozone_level_detection', 'data/cleaned/eighthr_val_data.csv')
    test_eighthr = load_csv_from_gcs('ozone_level_detection', 'data/cleaned/eighthr_test_data.csv')

    train_eighthr.drop(columns=['Date'], inplace=True)
    val_eighthr.drop(columns=['Date'], inplace=True)
    test_eighthr.drop(columns=['Date'], inplace=True)

    X_train_eighthr = train_eighthr.iloc[:, :-1].values
    y_train_eighthr = train_eighthr.iloc[:, -1].values
    X_val_eighthr = val_eighthr.iloc[:, :-1].values
    y_val_eighthr = val_eighthr.iloc[:, -1].values
    X_test_eighthr = test_eighthr.iloc[:, :-1].values
    y_test_eighthr = test_eighthr.iloc[:, -1].values

    return X_train_eighthr, y_train_eighthr, X_val_eighthr, y_val_eighthr, X_test_eighthr, y_test_eighthr

def train_and_evaluate():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt']
    }

    model = CustomRandomForestClassifier()
    grid_search = GridSearchCV(estimator=model.model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    def evaluate(model, X, y, dataset_name):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        matrix = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred)
        print(f"{dataset_name} - Accuracy: {accuracy}")
        print(f"{dataset_name} - Precision: {precision}")
        print(f"{dataset_name} - Recall: {recall}")
        print(f"{dataset_name} - F1 Score: {f1}")
        print(f"{dataset_name} - Confusion Matrix:\n{matrix}")
        print(f"{dataset_name} - Classification Report:\n{report}")

    evaluate(best_model, X_train, y_train, 'Train')
    evaluate(best_model, X_val, y_val, 'Validation')
    evaluate(best_model, X_test, y_test, 'Test')

train_and_evaluate()

# Orchestrating Workflows using Airflow

In this project, orchestration of workflows is managed through Apache Airflow, a powerful tool that schedules and manages complex workflows.
In production, these workflows are scaled and managed through Google Cloud Composer, a managed Airflow service that integrates seamlessly with other Google Cloud services,
enhancing our deployment's efficiency and reliability. The model deployment is managed through Vertex AI.

## Setup for Docker
Docker setup involves creating a Dockerfile that specifies the Airflow version and configuration settings appropriate for the
 project’s needs. Docker-compose files are used to define services, volumes, and networks that represent the Airflow components
such as the web server, scheduler, and worker.

## Initial Setup for AirFlow
Setting up Airflow involves defining Directed Acyclic Graphs (DAGs), which outline the sequence of
tasks and their dependencies. Each task is represented by operators, such as PythonOperator or BashOperator, which
execute specific pieces of code necessary for the workflow.

## General Runs of Airflow inside Google Composer
When a DAG is triggered (either manually or as per its schedule), it undergoes several stages:

1. **Queued:** The task is queued by the scheduler.
2. **Running:** A worker picks up the task and begins execution.
3. **Success/Failure:** Once execution completes, the task is marked as successful or failed.

## Operating through UI inside Airflow
Airflow's web-based UI allows users to monitor and manage their workflows. It provides detailed visualizations of pipeline
 dependencies, logging, and the status of various tasks. The UI is crucial for troubleshooting and understanding the behavior
of different tasks within the DAGs.

## Conclusion
Random Forest is a powerful and versatile model that can improve prediction accuracy and robustness. By tuning its
hyperparameters using GridSearchCV, you can optimize its performance for your specific dataset and task. The model architecture,
 which involves bootstrap sampling, decision tree construction, and aggregation, helps to enhance the model's ability to generalize and provide
reliable predictions. Incorporating optimizers and detailed training procedures ensures the model is fine-tuned and performs well across different
data subsets.
