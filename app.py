from flask import Flask, request, jsonify, render_template
from google.cloud import storage
import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model
model_path = 'model.pkl'  # Ensure this path is correct
model = joblib.load(model_path)

# Function to get predictions
def get_prediction(instances):
    try:
        df = pd.DataFrame(instances)
        predictions = model.predict(df)
        return predictions.tolist()
    except Exception as e:
        logging.error(f"Error getting prediction: {e}")
        return str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        instances = request.get_json(force=True)
        predictions = get_prediction(instances)
        if isinstance(predictions, str):
            return jsonify({"error": predictions}), 500
        return jsonify(predictions)
    except Exception as e:
        logging.error(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    try:
        client = storage.Client()
        bucket_name = 'ozone_level_detection'
        file_path = 'metrics/metrics.txt'
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        metrics_data = blob.download_as_text()
        return render_template('metrics.html', metrics=metrics_data)
    except Exception as e:
        logging.error(f"Error loading metrics: {e}")
        return render_template('metrics.html', metrics="Error loading metrics")

@app.route('/feature_importance')
def feature_importance():
    try:
        importances = model.feature_importances_
        features = pd.DataFrame({'Feature': model.feature_names_in_, 'Importance': importances})
        features = features.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=features)
        plt.title('Feature Importance')
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('feature_importance.html', plot_url=plot_url)
    except Exception as e:
        logging.error(f"Error generating feature importance: {e}")
        return render_template('feature_importance.html', plot_url=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
