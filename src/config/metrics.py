from typing import Optional

from pydantic import BaseModel, Field

from ._utils import load_yaml
from .base import BaseConfig


class MetricConfig(BaseModel):
    name: str
    key: str
    params: Optional[dict] = Field(default_factory=dict)


class MetricsConfig(BaseConfig):
    name: str
    description: Optional[str] = None
    metrics: list[MetricConfig] = Field(
        default_factory=list, description="List of metric configurations"
    )

    @classmethod
    def from_yaml(cls, filepath: str) -> "MetricsConfig":
        data = load_yaml(filepath)
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            metrics=[MetricConfig(**metric) for metric in data.get("metrics", [])],
        )
