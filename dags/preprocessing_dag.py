from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess

# Define the GitHub repository URL and the path to the script
GITHUB_REPO_URL = 'https://github.com/SiddanthEmani/Ozone_Level_Detection.git'
SCRIPT_PATH = 'https://github.com/SiddanthEmani/Ozone_Level_Detection/src/preprocess_data.py'

def clone_and_run_script():
    """Clone the GitHub repo and run the preprocessing script."""
    try:
        # Clone the repository
        subprocess.run(['git', 'clone', GITHUB_REPO_URL], check=True)

        # Run the script
        subprocess.run(['python', SCRIPT_PATH], check=True, cwd='/home/airflow/gcs/data')  # Adjust cwd if necessary
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        raise

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'data_preprocessing_dag',
    default_args=default_args,
    description='DAG for data preprocessing',
    schedule_interval='@daily',  # Runs daily
    start_date=datetime(2024, 6, 10),
    catchup=False,
) as dag:
    run_preprocessing = PythonOperator(
        task_id='preprocess_data',
        python_callable=clone_and_run_script,
    )

    run_preprocessing
