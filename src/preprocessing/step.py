from __future__ import annotations

import polars as pl

from .base import BasePreprocessingStep


class DropColumns(BasePreprocessingStep):
    name: str = "DropColumns"
    description: str = "Drop specified columns from the DataFrame"

    def __init__(self, *, columns: list[str]):
        self.columns = columns or []

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(self.columns)


class BaseConvertDataType(BasePreprocessingStep):
    name: str = "BaseConvertDataType"
    description: str = "Base class for converting data types"
    source_dtype: pl.DataType = None  # To be defined in subclasses
    target_dtype: pl.DataType = None  # To be defined in subclasses

    def __init__(
        self,
        *,
        parse: bool = True,
        columns: list[str] = None,
        new_column_suffix: str = "int",
        replace_columns: bool = True,
    ):
        self.columns = columns or []
        self.parse = parse
        self.new_column_suffix = new_column_suffix
        self.replace_columns = replace_columns

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        _columns = self._parse(df) if self.parse else self.columns.copy()

        existing_cols = [col for col in _columns if col in df.columns]
        if not existing_cols:
            return df

        if self.replace_columns:
            exprs = [pl.col(col).cast(self.target_dtype) for col in existing_cols]
        else:
            exprs = [
                pl.col(col)
                .cast(self.target_dtype)
                .alias(f"{col}_{self.new_column_suffix}")
                for col in existing_cols
            ]

        return df.with_columns(exprs)

    def _parse(self, df: pl.DataFrame) -> list[str]:
        columns = []
        for col in df.columns:
            if df[col].dtype == self.source_dtype:
                columns.append(col)
        return columns


class ConvertBooleanToInt(BaseConvertDataType):
    name: str = "ConvertBooleanToInt"
    description: str = "Convert boolean columns to integer (0 and 1)"
    source_dtype: pl.DataType = pl.Boolean
    target_dtype: pl.DataType = pl.Int8


class ConvertDateToInt(BaseConvertDataType):
    name: str = "ConvertDateToInt"
    description: str = "Convert date columns to integer"
    source_dtype: pl.DataType = pl.Date
    target_dtype: pl.DataType = pl.Int32


class ConvertDatetimeToInt(BaseConvertDataType):
    name: str = "ConvertDatetimeToInt"
    description: str = "Convert datetime columns to integer"
    source_dtype: pl.DataType = pl.Datetime
    target_dtype: pl.DataType = pl.Int64


PREPROCESSING_STEP_REGISTRY: dict[str, type[BasePreprocessingStep]] = {
    "DropColumns": DropColumns,
    "ConvertBooleanToInt": ConvertBooleanToInt,
    "ConvertDateToInt": ConvertDateToInt,
    "ConvertDatetimeToInt": ConvertDatetimeToInt,
}
