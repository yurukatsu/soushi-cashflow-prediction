import polars as pl
from pydantic import BaseModel, Field

from src.typing import AnyDict, StringOrPath

from ._utils import load_yaml
from .base import BaseConfig


class FilePath(BaseModel):
    """
    Configuration for file paths.

    Attributes:
        training (str): Path to the training dataset.
        test (str): Path to the test dataset.
    """

    training: str = Field(..., description="Path to the training dataset")
    test: str = Field(..., description="Path to the test dataset")


class ColumnName(BaseModel):
    """
    Configuration for column names in the dataset.

    Attributes:
        target (str): Name of the target column.
        date (str): Name of the date column.
        account_id (str): Name of the account ID column.
        account_name (str): Name of the account name column.
    """

    target: str = Field(..., description="Name of the target column")
    date: str = Field(..., description="Name of the date column")
    account_id: str = Field(..., description="Name of the account ID column")
    account_name: str = Field(..., description="Name of the account name column")


class DatasetConfig(BaseConfig):
    """
    Configuration for the dataset.

    Attributes:
        name (str): Name of the dataset.
        description (str): Description of the dataset.
        filepath (FilePath): File paths for training and test datasets.
        column_name (ColumnName): Column names in the dataset.
        schema (dict[str, str]): Schema of the dataset with column names and their data types
    """

    name: str = "dataset"
    description: str = "Configuration for the dataset"
    filepath: FilePath
    column_name: ColumnName
    data_schema: AnyDict

    @classmethod
    def from_yaml(cls, filepath: StringOrPath) -> "DatasetConfig":
        """
        Load configuration from a YAML file.
        """
        data = load_yaml(filepath)
        data_schema = {
            c: getattr(pl, t) for c, t in data.get("data_schema", {}).items()
        }
        return cls(
            name=data.get("name", "dataset"),
            description=data.get("description", "Configuration for the dataset"),
            filepath=FilePath(**data.get("filepath")),
            column_name=ColumnName(**data.get("column_name")),
            data_schema=data_schema,
        )
