import logging

import polars as pl

from ..config.preprocessing import PreprocessingConfig, PreprocessingStepConfig
from .base import BasePreprocessingStep
from .step import PREPROCESSING_STEP_REGISTRY

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    def __init__(self, steps: list[BasePreprocessingStep]) -> None:
        self.steps = steps

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        for step in self.steps:
            logger.info(f"Applying step: {step.name} - {step.description}")
            df = step.run(df)
        return df

    @classmethod
    def from_step_configs(
        cls, configs: list[PreprocessingStepConfig]
    ) -> "PreprocessingPipeline":
        steps = []
        for config in configs:
            step_class = PREPROCESSING_STEP_REGISTRY.get(config.name)
            if step_class is None:
                raise ValueError(f"Unknown preprocessing step: {config.name}")
            step = step_class.from_config(config)
            steps.append(step)
        return cls(steps)

    @classmethod
    def from_config(cls, config: PreprocessingConfig) -> "PreprocessingPipeline":
        return cls.from_step_configs(config.pipeline)
