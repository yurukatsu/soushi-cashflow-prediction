import polars as pl

from src.config import DatasetConfig
from src.typing import StringOrPath


class DataLoader:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def load_data(self, filepath: StringOrPath) -> pl.DataFrame:
        df = pl.read_csv(filepath, schema_overrides=self.config.data_schema)
        return df

    def load_training_data(self) -> pl.DataFrame:
        return self.load_data(self.config.filepath.training)

    def load_test_data(self) -> pl.DataFrame:
        return self.load_data(self.config.filepath.test)
