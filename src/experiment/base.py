import datetime
import logging
from contextlib import contextmanager

import colorlog
import mlflow
import polars as pl

from src.config import (
    BaseConfig,
    CVCrossValidationConfig,
    DatasetConfig,
    MetricsConfig,
    ModelConfig,
    PreprocessingConfig,
)
from src.cv import SlidingWindowCV
from src.ingestion import DataLoader
from src.model import ModelMap
from src.model.metrics import METRIC_MAP
from src.preprocessing import PreprocessingPipeline


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        *,
        base_run_name: str | None = None,
        dataset_config: DatasetConfig | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        cv_config: CVCrossValidationConfig | None = None,
        preprocessing_config_after_data_split: PreprocessingConfig | None = None,
        model_config: ModelConfig | None = None,
        metrics_config: MetricsConfig | None = None,
        dataset_config_path: str | None = None,
        preprocessing_config_path: str | None = None,
        cv_config_path: str | None = None,
        preprocessing_config_after_data_split_path: str | None = None,
        model_config_path: str | None = None,
        metrics_config_path: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self.experiment_name = experiment_name
        self.base_run_name = base_run_name
        self.create_at = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logger: logging.Logger = logger or colorlog.getLogger(__name__)

        with self._task_status_message("load dataset config"):
            self.dataset_config: DatasetConfig = self._get_config(
                config=dataset_config,
                config_path=dataset_config_path,
                base_cls=DatasetConfig,
            )

        with self._task_status_message("load preprocessing config"):
            self.preprocessing_config: PreprocessingConfig = self._get_config(
                config=preprocessing_config,
                config_path=preprocessing_config_path,
                base_cls=PreprocessingConfig,
            )

        with self._task_status_message("load cv config"):
            self.cv_config: CVCrossValidationConfig = self._get_config(
                config=cv_config,
                config_path=cv_config_path,
                base_cls=CVCrossValidationConfig,
            )

        with self._task_status_message("load preprocessing config after data split"):
            self.preprocessing_config_after_data_split: PreprocessingConfig = (
                self._get_config(
                    config=preprocessing_config_after_data_split,
                    config_path=preprocessing_config_after_data_split_path,
                    base_cls=PreprocessingConfig,
                )
            )

        with self._task_status_message("load model config"):
            self.model_config: ModelConfig = self._get_config(
                config=model_config,
                config_path=model_config_path,
                base_cls=ModelConfig,
            )

        with self._task_status_message("load metrics config"):
            self.metrics_config: MetricsConfig = self._get_config(
                config=metrics_config,
                config_path=metrics_config_path,
                base_cls=MetricsConfig,
            )
            self.model_cls = ModelMap.get(self.model_config.model_name)

    def _get_config(
        self,
        *,
        config: BaseConfig | None = None,
        config_path: str | None = None,
        base_cls: BaseConfig = BaseConfig,
    ) -> BaseConfig:
        if config is not None:
            self.logger.info(f"Using provided config: {config}")
            return config
        if config_path is not None:
            self.logger.info(f"Loading config from: {config_path}")
            return base_cls.from_yaml(config_path)
        raise ValueError("Either config or config_path must be provided.")

    def _create_run_name(self, suffix: str | None = None) -> str:
        base_run_name = (
            self.base_run_name or f"{self.model_config.model_name}_{self.create_at}"
        )
        suffix = f"_{suffix}" if suffix is not None else ""
        return f"{base_run_name}{suffix}"

    @contextmanager
    def _task_status_message(self, task: str):
        self.logger.info(f"Starting task: {task}")
        try:
            yield
            self.logger.info(f"Completed task: {task}")
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            raise

    def _calculate_metrics(
        self, y_true: pl.DataFrame, y_pred: pl.DataFrame, prefix: str = ""
    ) -> dict[str, float]:
        array_true = y_true.to_numpy().squeeze()
        array_pred = y_pred.to_numpy().squeeze()

        results = {}
        for metric in self.metrics_config.metrics:
            metric_func = METRIC_MAP.get(metric.key)
            if metric_func is None:
                self.logger.warning(f"Metric {metric.key} not found in METRIC_MAP.")
                continue
            result = metric_func(array_true, array_pred, **(metric.params or {}))
            results[f"{prefix}{metric.name}"] = result
        return results

    def _log_config(self):
        mlflow.log_dict(self.dataset_config.model_dump(), "config/dataset_config.json")
        mlflow.log_dict(
            self.preprocessing_config.model_dump(), "config/preprocessing_config.json"
        )
        mlflow.log_dict(self.cv_config.model_dump(), "config/cv_config.json")
        mlflow.log_dict(
            self.preprocessing_config_after_data_split.model_dump(),
            "config/preprocessing_config_after_data_split.json",
        )
        mlflow.log_dict(self.model_config.model_dump(), "config/model_config.json")
        mlflow.log_dict(self.metrics_config.model_dump(), "config/metrics_config.json")

    def run(self):
        with self._task_status_message("create dataloader"):
            dataloader = DataLoader(config=self.dataset_config)

        with self._task_status_message("load training data"):
            train_df = dataloader.load_training_data()

        with self._task_status_message("load test data"):
            test_df = dataloader.load_test_data()

        with self._task_status_message("create preprocessing pipeline"):
            preprocessor = PreprocessingPipeline.from_config(self.preprocessing_config)

        with self._task_status_message("run preprocessing pipeline (train and test)"):
            train_df = preprocessor.run(train_df)
            test_df = preprocessor.run(test_df)
            X_test = test_df.drop(self.dataset_config.column_name.target)
            y_test = test_df.select(self.dataset_config.column_name.target)

        with self._task_status_message("create cross-validation splits"):
            cv = SlidingWindowCV(
                df=train_df,
                date_col=self.dataset_config.column_name.date,
                n_splits=self.cv_config.n_splits,
                start_date=self.cv_config.start_date,
                end_date=self.cv_config.end_date,
                train_duration=self.cv_config.train_duration,
                gap_duration=self.cv_config.gap_duration,
                validation_duration=self.cv_config.validation_duration,
                step_duration=self.cv_config.step_duration,
            )

        with self._task_status_message("data create preprocessor after data split"):
            preprocessor_after_data_split = PreprocessingPipeline.from_config(
                self.preprocessing_config_after_data_split
            )

        fold = 0
        for train, val in cv.split_dataframe(train_df):
            X_train = train.drop(self.dataset_config.column_name.target)
            y_train = train.select(self.dataset_config.column_name.target)
            X_val = val.drop(self.dataset_config.column_name.target)
            y_val = val.select(self.dataset_config.column_name.target)

            with self._task_status_message("Instantiate model"):
                model = self.model_cls(
                    preprocessor=preprocessor_after_data_split,
                    model_params=self.model_config.model_params,
                )

            with self._task_status_message("fit model"):
                model.fit(X_train, y_train, **self.model_config.fit_params)

            with self._task_status_message("predict train"):
                y_train_pred = model.predict(
                    X_train, **self.model_config.predict_params
                )

            with self._task_status_message("evaluate train metrics"):
                train_metrics = self._calculate_metrics(
                    y_train, y_train_pred, prefix="train_"
                )
                self.logger.info(f"Train metrics: {train_metrics}")

            with self._task_status_message("predict validation"):
                y_val_pred = model.predict(X_val, **self.model_config.predict_params)

            with self._task_status_message("evaluate metrics"):
                val_metrics = self._calculate_metrics(y_val, y_val_pred, prefix="val_")
                self.logger.info(f"Validation metrics: {val_metrics}")

            with self._task_status_message("predict test"):
                y_test_pred = model.predict(
                    X_test,
                    **self.model_config.predict_params,
                )

            with self._task_status_message("evaluate test metrics"):
                test_metrics = self._calculate_metrics(
                    y_test, y_test_pred, prefix="test_"
                )
                self.logger.info(f"Test metrics: {test_metrics}")

            with mlflow.start_run(
                run_name=self._create_run_name(suffix=f"fold{fold}"),
                tags={
                    "dataset": self.dataset_config.name,
                    "preprocessing": self.preprocessing_config.name,
                    "cv": self.cv_config.name,
                    "preprocessing_after_data_split": self.preprocessing_config_after_data_split.name,
                    "model": self.model_config.name,
                    "model_name": self.model_config.model_name,
                    "fold": str(fold),
                },
            ):
                self._log_config()

                mlflow.log_params(model.get_params() or {})
                mlflow.log_metrics(train_metrics)
                mlflow.log_metrics(val_metrics)
                mlflow.log_metrics(test_metrics)
                self.logger.info(f"{type(train_df)}")
                mlflow.log_input(
                    mlflow.data.polars_dataset.from_polars(
                        train,
                        targets="Flow",
                        name=f"{self.dataset_config.name}--{self.preprocessing_config.name}--{self.cv_config.name}--fold{fold}--train",
                    ),
                )
                mlflow.log_input(
                    mlflow.data.polars_dataset.from_polars(
                        val,
                        targets="Flow",
                        name=f"{self.dataset_config.name}--{self.preprocessing_config.name}--{self.cv_config.name}--fold{fold}--val",
                    )
                )
                mlflow.log_input(
                    mlflow.data.polars_dataset.from_polars(
                        test_df,
                        targets="Flow",
                        name=f"{self.dataset_config.name}--{self.preprocessing_config.name}--test",
                    )
                )

            fold += 1
