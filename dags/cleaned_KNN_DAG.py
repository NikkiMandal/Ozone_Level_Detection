from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 10),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'remove_missing_values_dag_knn',
    default_args=default_args,
    description='DAG for removing missing values using KNN from ozone level data',
    schedule_interval='@daily',
    catchup=False,
)

download_eighthr_data = PythonOperator(
    task_id='download_eighthr_data',
    python_callable=download_from_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': "data/raw/eighthr_data.csv"
    },
    provide_context=True,
    dag=dag,
)

download_onehr_data = PythonOperator(
    task_id='download_onehr_data',
    python_callable=download_from_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'source_blob_name': "data/raw/onehr_data.csv"
    },
    provide_context=True,
    dag=dag,
)

remove_missing_values_eighthr = PythonOperator(
    task_id='remove_missing_values_eighthr',
    python_callable=remove_missing_values_knn,
    op_kwargs={'task_ids': ['download_eighthr_data']},
    provide_context=True,
    dag=dag,
)

remove_missing_values_onehr = PythonOperator(
    task_id='remove_missing_values_onehr',
    python_callable=remove_missing_values_knn,
    op_kwargs={'task_ids': ['download_onehr_data']},
    provide_context=True,
    dag=dag,
)

upload_cleaned_eighthr = PythonOperator(
    task_id='upload_cleaned_eighthr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': "data/new_cleaned/eighthr_data_KNN.csv",
        'task_ids': ['remove_missing_values_eighthr']
    },
    provide_context=True,
    dag=dag,
)

upload_cleaned_onehr = PythonOperator(
    task_id='upload_cleaned_onehr',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': 'ozone_level_detection',
        'destination_blob_name': "data/new_cleaned/onehr_data_KNN.csv",
        'task_ids': ['remove_missing_values_onehr']
    },
    provide_context=True,
    dag=dag,
)

# Set task dependencies
download_eighthr_data >> remove_missing_values_eighthr >> upload_cleaned_eighthr
download_onehr_data >> remove_missing_values_onehr >> upload_cleaned_onehr
