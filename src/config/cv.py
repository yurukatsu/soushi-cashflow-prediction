from typing import Optional

from pydantic import Field

from .base import BaseConfig


class CVCrossValidationConfig(BaseConfig):
    """
    Configuration for cross-validation.

    Attributes:
        name (str): Name of the configuration.
        description (str): Description of the configuration.
        strategy (str): Cross-validation
        n_splits (int): Number of splits for cross-validation.
        start_date (Optional[str]): Start date for time series cross-validation (e.g., '2020-01-01').
        end_date (Optional[str]): End date for time series cross-validation (e.g., '2023-12-31').
        train_duration (Optional[str]): Duration of the training period (e.g., '12mo' for 12 months).
        validation_duration (str): Duration of the validation period (e.g., '3mo' for 3 months).
        gap_duration (Optional[str]): Duration of the gap period between training and validation (e.g., '3mo' for 3 months).
        step_duration (Optional[str]): Step duration to move the window forward (e.g., '3mo' for 3 months).
    """

    name: str = "cv"
    description: str = "Configuration for cross-validation"
    strategy: str = Field(
        ...,
        description="Cross-validation strategy (e.g., 'kfold', 'stratified', 'sliding_window')",
    )
    n_splits: int = Field(..., description="Number of splits for cross-validation")
    start_date: Optional[str] = Field(
        None,
        description="Start date for time series cross-validation (e.g., '2020-01-01')",
    )
    end_date: Optional[str] = Field(
        None,
        description="End date for time series cross-validation (e.g., '2023-12-31')",
    )
    train_duration: Optional[str] = Field(
        None, description="Duration of the training period (e.g., '12mo' for 12 months)"
    )
    validation_duration: str = Field(
        "3mo",
        description="Duration of the validation period (e.g., '3mo' for 3 months)",
    )
    gap_duration: Optional[str] = Field(
        None,
        description="Duration of the gap period between training and validation (e.g., '3mo' for 3 months)",
    )
    step_duration: Optional[str] = Field(
        None,
        description="Step duration to move the window forward (e.g., '3mo' for 3 months)",
    )
