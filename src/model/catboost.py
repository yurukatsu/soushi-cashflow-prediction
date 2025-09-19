from catboost import CatBoostRegressor

from .base import BaseModelWithOneHotEncoding


class CatBoostModel(BaseModelWithOneHotEncoding):
    estimator = CatBoostRegressor
