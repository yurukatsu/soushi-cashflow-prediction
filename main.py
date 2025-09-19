import logging

import mlflow

from src.experiment import Experiment

logging.basicConfig(level=logging.INFO)

mlflow.set_tracking_uri("http://localhost:15000")
experiment_name = "soushi-cashflow-prediction-test"
mlflow.set_experiment(experiment_name)

experiment = Experiment(
    experiment_name=experiment_name,
    dataset_config_path="config/dataset/config.yml",
    preprocessing_config_path="config/preprocessing/config.yml",
    cv_config_path="config/cv/config.yml",
    preprocessing_config_after_data_split_path="config/preprocessing/config_after_data_split.yml",
    model_config_path="config/model/catboost.yml",
    metrics_config_path="config/metrics/config.yml",
)
experiment.run()
