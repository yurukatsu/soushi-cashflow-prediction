from abc import ABC

import pandas as pd
import polars as pl
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing import PreprocessingPipeline
from src.typing import AnyDict


class BaseModel(ABC):
    estimator = None  # To be defined in subclasses

    def __init__(
        self,
        *,
        preprocessor: PreprocessingPipeline = None,
        model_params: AnyDict = None,
    ):
        self.preprocessor = preprocessor
        self.model_params = model_params or {}
        self.model = self.estimator(**self.model_params)

    def preprocess(
        self, X: pl.DataFrame, y: pl.DataFrame | pl.Series = None, is_train: bool = True
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        if self.preprocessor is not None:
            X = self.preprocessor.run(X)

        return X, y.to_pandas().squeeze() if y is not None else None

    def fit(self, X: pl.DataFrame, y: pl.DataFrame, **fit_params):
        X_prep, y_prep = self.preprocess(X, y, is_train=True)
        self.model.fit(X_prep, y_prep, **fit_params)

    def predict(self, X: pl.DataFrame, **predict_params) -> pl.Series:
        X_prep, _ = self.preprocess(X, is_train=False)
        predictions = self.model.predict(X_prep, **predict_params)
        return pl.DataFrame({"prediction": predictions})

    def get_params(self) -> AnyDict:
        return self.model.get_params()


class BaseModelWithOneHotEncoding(BaseModel):
    def __init__(
        self,
        *,
        preprocessor: PreprocessingPipeline = None,
        model_params: AnyDict = None,
    ):
        super().__init__(preprocessor=preprocessor, model_params=model_params)
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def preprocess(
        self, X: pl.DataFrame, y: pl.DataFrame | pl.Series = None, is_train: bool = True
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        if self.preprocessor is not None:
            X = self.preprocessor.run(X)

        col_categorical = X.select(pl.col(pl.Categorical)).columns

        X_pd = X.to_pandas()
        X_pd_categorical = X_pd[col_categorical]

        if is_train:
            self.ohe.fit(X_pd_categorical)

        encoded_array = self.ohe.transform(X_pd_categorical)
        X_pd_encoded = pd.DataFrame(
            encoded_array,
            columns=self.ohe.get_feature_names_out(col_categorical),
        )

        X = pd.concat([X_pd.drop(columns=col_categorical), X_pd_encoded], axis=1)

        return X, y.to_pandas().squeeze() if y is not None else None
