from abc import ABC, abstractmethod

import polars as pl

from ..config.preprocessing import PreprocessingStepConfig


class BasePreprocessingStep(ABC):
    name: str = "BasePreprocessingStep"
    description: str = "Base class for preprocessing steps"

    @abstractmethod
    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    @classmethod
    def from_config(cls, config: PreprocessingStepConfig) -> "BasePreprocessingStep":
        return cls(**config.kwargs)
