from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

def main():
    aiplatform.init(project='ozone-level-detection', location='us-central1')

    # Submit training job
    training_job = aiplatform.CustomJob.from_local_script(
        display_name="ozone_level_detection_training",
        script_path="gs://your-bucket/scripts/train_model.py",
        requirements=["google-cloud-aiplatform", "scikit-learn"],
        machine_type="n1-standard-4",
    )
    training_job.run()

    # Submit hyperparameter tuning job
    hyperparameter_tuning_job = aiplatform.HyperparameterTuningJob(
        display_name="ozone_level_detection_hp_tuning",
        custom_job=training_job,
        metric_spec={'accuracy': 'maximize'},
        parameter_spec={
            'max_depth': hpt.IntParameterSpec(min=2, max=10, scale='unit-linear'),
            'n_estimators': hpt.IntParameterSpec(min=10, max=100, scale='unit-linear'),
        },
        max_trial_count=20,
        parallel_trial_count=5,
    )
    hyperparameter_tuning_job.run()

    # Submit model analysis job
    analysis_job = aiplatform.CustomJob.from_local_script(
        display_name="ozone_level_detection_model_analysis",
        script_path="gs://your-bucket/scripts/model_analysis.py",
        requirements=["google-cloud-aiplatform", "scikit-learn"],
        machine_type="n1-standard-4",
    )
    analysis_job.run()

if __name__ == "__main__":
    main()
