from .base import BaseModel
from .catboost import CatBoostModel

ModelMap = {"CatBoostModel": CatBoostModel}

__all__ = ["BaseModel", "CatBoostModel", "ModelMap"]
