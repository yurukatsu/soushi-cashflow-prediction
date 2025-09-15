import pprint

from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

from src.config import CVCrossValidationConfig, DatasetConfig, PreprocessingConfig
from src.preprocessing.step import ConvertDatetimeToInt
from src.cv import SlidingWindowCV
from src.ingestion import DataLoader
from src.preprocessing import PreprocessingPipeline
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor

dataset_config = DatasetConfig.from_yaml("config/dataset/config.yml")
preprocessing_config = PreprocessingConfig.from_yaml("config/preprocessing/config.yml")
cv_config: CVCrossValidationConfig = CVCrossValidationConfig.from_yaml(
    "config/cv/config.yml"
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

pprint.pprint(cv.windows)

X = train_df.drop(dataset_config.column_name.target).to_pandas()
y = train_df.select(dataset_config.column_name.target).to_pandas()

# for i, (train_idx, val_idx) in enumerate(cv.split(X)):
#     print(f"Fold {i + 1}")

#     X_train = X.iloc[train_idx]
#     y_train = y.iloc[train_idx]
#     X_test = X.iloc[val_idx]
#     y_test = y.iloc[val_idx]

#     print(f"  Start date: {X_train[dataset_config.column_name.date].min()}")
#     print(f"  End date: {X_train[dataset_config.column_name.date].max()}")

#     print(f"  Start date: {X_test[dataset_config.column_name.date].min()}")
#     print(f"  End date: {X_test[dataset_config.column_name.date].max()}")


param_grid = {
    "depth": [6, 8, 10],
    "learning_rate": [0.1, 0.05],
    "iterations": [200, 500],
}

class WrappedCatBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, preprocess_fn=None, **catboost_params):
        """
        sklearn BaseEstimator ラッパー for CatBoostRegressor

        :param preprocess_fn: 前処理関数 (X -> X_transformed)
        :param catboost_params: CatBoostRegressor に渡すパラメータ
        """
        self.preprocess_fn = preprocess_fn
        self.catboost_params = catboost_params
        self.model_ = None

    def preprocess(self, X):
        """ユーザー指定の前処理を適用"""
        if self.preprocess_fn is None:
            return X
        return self.preprocess_fn(X)

    def fit(self, X, y, **fit_params):
        X_prep = self.preprocess(X)
        self.model_ = CatBoostRegressor(**self.catboost_params)
        self.model_.fit(X_prep, y, **fit_params)
        return self

    def predict(self, X):
        X_prep = self.preprocess(X)
        return self.model_.predict(X_prep)

    def get_params(self, deep=True):
        return {"preprocess_fn": self.preprocess_fn, **self.catboost_params}

    def set_params(self, **params):
        if "preprocess_fn" in params:
            self.preprocess_fn = params.pop("preprocess_fn")
        self.catboost_params.update(params)
        return self

def preprocess_fn(X):
    # ここに前処理のロジックを実装
    return X.drop(columns=[dataset_config.column_name.date])

cbr = WrappedCatBoostRegressor(preprocess_fn=preprocess_fn, verbose=0, random_seed=42)
grid = GridSearchCV(cbr, param_grid, cv=3, scoring="neg_mean_squared_error")
grid.fit(X, y)

print("Best params:", grid.best_params_)
