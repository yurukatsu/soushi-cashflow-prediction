from pydantic import BaseModel, Field

from src.typing import AnyDict, StringOrPath

from ._utils import load_yaml
from .base import BaseConfig


class PreprocessingStepConfig(BaseModel):
    """
    A single step in a preprocessing pipeline.

    Attributes:
        name (str): Name of the preprocessing step.
        kwargs (AnyDict): Keyword arguments for the preprocessing step.
    """

    name: str = Field(..., description="Name of the preprocessing step.")
    kwargs: AnyDict = Field(
        default_factory=dict,
        description="Keyword arguments for the preprocessing step.",
    )


class PreprocessingConfig(BaseConfig):
    """
    Configuration for data preprocessing.

    Attributes:
        name (str): Name of the preprocessing configuration.
        description (str): Description of the preprocessing configuration.
        pipeline (list[PreprocessingStep]): List of preprocessing steps to be applied.
    """

    name: str = Field(..., description="Name of the preprocessing configuration.")
    description: str = Field(
        ..., description="Description of the preprocessing configuration."
    )
    pipeline: list[PreprocessingStepConfig] = Field(
        default_factory=list,
        description="List of preprocessing steps to be applied.",
    )

    @classmethod
    def from_yaml(cls, path: StringOrPath) -> "PreprocessingConfig":
        """
        Load configuration from a YAML file.
        """
        data = load_yaml(path)
        return cls(
            name=data.get("name", "default_preprocessing"),
            description=data.get("description", ""),
            pipeline=[
                PreprocessingStepConfig(
                    name=step.get("name", ""),
                    kwargs=step.get("kwargs", {}),
                )
                for step in data.get("pipeline", [])
            ],
        )
