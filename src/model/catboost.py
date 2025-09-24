import polars as pl
from catboost import CatBoostRegressor

from ._output import FeatureImportance
from .base import BaseModelWithOneHotEncoding


class CatBoostModel(BaseModelWithOneHotEncoding):
    estimator = CatBoostRegressor

    def get_feature_importance(self):
        importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_
        df = pl.DataFrame({"feature": feature_names, "importance": importance}).sort(
            "importance", descending=True
        )
        return FeatureImportance(df)
