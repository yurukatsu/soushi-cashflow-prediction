import logging

from sklearn.metrics import root_mean_squared_error

from src.config import CVCrossValidationConfig, DatasetConfig, PreprocessingConfig
from src.cv import SlidingWindowCV
from src.ingestion import DataLoader
from src.model import CatBoostModel
from src.preprocessing import PreprocessingPipeline

logging.basicConfig(level=logging.INFO)

dataset_config = DatasetConfig.from_yaml("config/dataset/config.yml")
preprocessing_config = PreprocessingConfig.from_yaml("config/preprocessing/config.yml")
cv_config: CVCrossValidationConfig = CVCrossValidationConfig.from_yaml(
    "config/cv/config.yml"
)
preprocessing_config_after_data_split = PreprocessingConfig.from_yaml(
    "config/preprocessing/config_after_data_split.yml"
)

dataloader = DataLoader(config=dataset_config)
dataset = dataloader.load_training_data()

preprocessor = PreprocessingPipeline.from_config(preprocessing_config)
train_df = preprocessor.run(dataset)

cv = SlidingWindowCV(
    df=train_df,
    date_col=dataset_config.column_name.date,
    n_splits=cv_config.n_splits,
    start_date=cv_config.start_date,
    end_date=cv_config.end_date,
    train_duration=cv_config.train_duration,
    gap_duration=cv_config.gap_duration,
    validation_duration=cv_config.validation_duration,
    step_duration=cv_config.step_duration,
)

X = train_df.drop(dataset_config.column_name.target)
y = train_df.select(dataset_config.column_name.target)

preprocessor_after_data_split = PreprocessingPipeline.from_config(
    preprocessing_config_after_data_split
)
model_params = {
    "verbose": 0,
}
model = CatBoostModel(
    preprocessor=preprocessor_after_data_split, model_params=model_params
)
model.fit(X, y)
y_pred = model.predict(X)
rmse = root_mean_squared_error(y.to_numpy().squeeze(), y_pred.to_numpy().squeeze())
logging.info(f"RMSE on training data: {rmse}")


# class Objective:
#     def __init__(self, df: pl.DataFrame, cv: SlidingWindowCV) -> None:
#         self.df = df
#         self.cv = cv

#     def __call__(self, trial: optuna.Trial) -> float:
#         param_grid = {
#             "depth": trial.suggest_int("depth", 4, 10),
#             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#             "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
#             "iterations": trial.suggest_int("iterations", 100, 3000),
#         }

#         rmse_list = []

#         for df_train, df_val in self.cv.split_dataframe(self.df):
#             X_train, y_train = (
#                 df_train.drop(dataset_config.column_name.target),
#                 df_train.select(dataset_config.column_name.target),
#             )
#             X_val, y_val = (
#                 df_val.drop(dataset_config.column_name.target),
#                 df_val.select(dataset_config.column_name.target),
#             )

#             model = CustomCatBoostRegressor(
#                 preprocessor=PreprocessingPipeline.from_config(
#                     preprocessing_config_after_data_split
#                 ),
#                 model_params=param_grid,
#             )
#             model.fit(X_train, y_train)

#             y_pred = model.predict(X_val)
#             rmse = root_mean_squared_error(
#                 y_val.to_numpy().squeeze(), y_pred.to_numpy().squeeze()
#             )
#             rmse_list.append(rmse)

#         return sum(rmse_list) / len(rmse_list)


# if __name__ == "__main__":
#     logging.info("Starting hyperparameter optimization...")

#     study = optuna.create_study(direction="minimize")
#     study.optimize(Objective(df=train_df, cv=cv), n_trials=50)

#     logging.info("Best trial:")
#     logging.info(f"  Value: {study.best_trial.value}")
#     logging.info("  Params: ")
#     logging.info(pprint.pformat(study.best_trial.params))
