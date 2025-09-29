import logging
import tempfile

import click
import mlflow
import polars as pl

from src.config import CrossValidationConfig, MetricsConfig
from src.cv import SlidingWindowCV
from src.model.metrics import METRIC_MAP

logger = logging.getLogger(__name__)


COL_DATE = "dtBaseDate"
COL_ID = "AccountCode"
COL_NAME = "AccountName"
COL_TARGET = "Flow"
COL_PREDICTION = "prediction"


class BaselineModelLogging:
    def __init__(
        self,
        df_train: pl.DataFrame,
        df_test: pl.DataFrame,
        cv_config: CrossValidationConfig,
        metrics_config: MetricsConfig,
        *,
        base_run_name: str = "Baseline",
        col_date: str = COL_DATE,
        col_id: str = COL_ID,
        col_target: str = COL_TARGET,
        col_prediction: str = COL_PREDICTION,
    ):
        self.base_run_name = base_run_name
        self.df_train = df_train
        self.df_test = df_test
        self.cv_config = cv_config
        self.metrics_config = metrics_config
        self.col_date = col_date
        self.col_id = col_id
        self.col_target = col_target
        self.col_prediction = col_prediction

    def calculate_metrics(
        self, y_true: pl.DataFrame, y_pred: pl.DataFrame, prefix: str = ""
    ) -> dict[str, float]:
        array_true = y_true.to_numpy().squeeze()
        array_pred = y_pred.to_numpy().squeeze()

        results = {}
        for metric in self.metrics_config.metrics:
            metric_func = METRIC_MAP.get(metric.key)
            if metric_func is None:
                logger.warning(f"Metric {metric.key} not found in METRIC_MAP.")
                continue
            result = metric_func(array_true, array_pred, **(metric.params or {}))
            results[f"{prefix}{metric.name}"] = result
        return results

    def _log_config(self):
        mlflow.log_dict(self.cv_config.model_dump(), "config/cv_config.json")
        mlflow.log_dict(self.metrics_config.model_dump(), "config/metrics_config.json")

    def _log_output_data(
        self, df: pl.DataFrame, name: str, artifact_path: str = "output"
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/{name}.csv"
            df.write_csv(path)
            mlflow.log_artifact(path, artifact_path=artifact_path)

    def run(self):
        cv = SlidingWindowCV(
            df=self.df_train,
            date_col=self.col_date,
            n_splits=self.cv_config.n_splits,
            start_date=self.cv_config.start_date,
            end_date=self.cv_config.end_date,
            train_duration=self.cv_config.train_duration,
            gap_duration=self.cv_config.gap_duration,
            validation_duration=self.cv_config.validation_duration,
            step_duration=self.cv_config.step_duration,
        )

        fold = 0
        for train, val in cv.split_dataframe(self.df_train):
            with mlflow.start_run(
                run_name=f"{self.base_run_name}_fold{fold}",
                nested=True,
            ):
                self._log_config()

                y_train_true = train.select(self.col_target)
                y_train_pred = train.select(self.col_prediction)
                y_val_true = val.select(self.col_target)
                y_val_pred = val.select(self.col_prediction)
                y_test_true = self.df_test.select(self.col_target)
                y_test_pred = self.df_test.select(self.col_prediction)

                train_metrics = self.calculate_metrics(
                    y_true=y_train_true,
                    y_pred=y_train_pred,
                    prefix="train_",
                )
                mlflow.log_metrics(train_metrics)
                val_metrics = self.calculate_metrics(
                    y_true=y_val_true,
                    y_pred=y_val_pred,
                    prefix="val_",
                )
                mlflow.log_metrics(val_metrics)
                test_metrics = self.calculate_metrics(
                    y_true=y_test_true,
                    y_pred=y_test_pred,
                    prefix="test_",
                )
                mlflow.log_metrics(test_metrics)
                fold += 1

                self._log_output_data(train, name="train", artifact_path="output")
                self._log_output_data(val, name="val", artifact_path="output")
                self._log_output_data(self.df_test, name="test", artifact_path="output")


@click.command()
@click.option("--train_path", type=str, default="./data/raw/train.csv")
@click.option("--test_path", type=str, default="./data/raw/test.csv")
@click.option("--baseline_path", type=str, default="./data/raw/baseline.csv")
@click.option("--cv_config_path", type=str, default="./config/cv/config.yml")
@click.option("--metrics_config_path", type=str, default="./config/metrics/config.yml")
def main(
    train_path: str,
    test_path: str,
    baseline_path: str,
    cv_config_path: str,
    metrics_config_path: str,
):
    # read data
    df_train = pl.read_csv(train_path, try_parse_dates=True).select(
        [COL_DATE, COL_ID, COL_NAME, COL_TARGET]
    )
    df_test = pl.read_csv(test_path, try_parse_dates=True).select(
        [COL_DATE, COL_ID, COL_NAME, COL_TARGET]
    )
    df_baseline = pl.read_csv(baseline_path, try_parse_dates=True).select(
        [COL_DATE, COL_ID, COL_PREDICTION]
    )

    # combine data
    df_train = df_train.join(
        df_baseline,
        on=["dtBaseDate", "AccountCode"],
        how="left",
    )
    df_test = df_test.join(
        df_baseline,
        on=["dtBaseDate", "AccountCode"],
        how="left",
    )

    # read config
    cv_config: CrossValidationConfig = CrossValidationConfig.from_yaml(cv_config_path)
    metrics_config = MetricsConfig.from_yaml(metrics_config_path)

    # run baseline model logging
    experiment = BaselineModelLogging(
        df_train=df_train,
        df_test=df_test,
        cv_config=cv_config,
        metrics_config=metrics_config,
    )
    experiment.run()


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:15000")
    experiment_name = "soushi-cashflow-prediction-test"
    mlflow.set_experiment(experiment_name)
    main()
