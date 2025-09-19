from typing import Optional

from pydantic import Field

from .base import BaseConfig


class ModelConfig(BaseConfig):
    name: str
    description: Optional[str] = None
    model_name: str
    model_params: Optional[dict] = Field(default_factory=dict)
    fit_params: Optional[dict] = Field(default_factory=dict)
    predict_params: Optional[dict] = Field(default_factory=dict)
