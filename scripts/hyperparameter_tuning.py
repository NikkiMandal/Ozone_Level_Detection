from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

def main():
    aiplatform.init(project='ozone-level-detection', location='us-central1')

    # Define a custom training job with hyperparameter tuning
    job = aiplatform.CustomJob.from_local_script(
        display_name="ozone_level_detection_hp_tuning",
        script_path="scripts/train_model.py",
        requirements=["google-cloud-aiplatform", "scikit-learn"],
        machine_type="n1-standard-4",
    )

    hp_job = aiplatform.HyperparameterTuningJob(
        display_name="ozone_level_detection_hp_tuning",
        custom_job=job,
        metric_spec={'accuracy': 'maximize'},
        parameter_spec={
            'max_depth': hpt.IntParameterSpec(min=2, max=10, scale='unit-linear'),
            'n_estimators': hpt.IntParameterSpec(min=10, max=100, scale='unit-linear'),
        },
        max_trial_count=20,
        parallel_trial_count=5,
    )

    hp_job.run()

if __name__ == "__main__":
    main()
