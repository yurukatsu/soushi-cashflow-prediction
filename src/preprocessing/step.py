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
        exclude_columns: list[str] = None,
        new_column_suffix: str = "int",
        replace_columns: bool = True,
    ):
        self.columns = columns or []
        self.exclude_columns = exclude_columns or []
        self.parse = parse
        self.new_column_suffix = new_column_suffix
        self.replace_columns = replace_columns

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        _columns = self._parse(df) if self.parse else self.columns.copy()

        existing_cols = [
            col
            for col in _columns
            if col in df.columns and col not in self.exclude_columns
        ]
        if len(existing_cols) == 0:
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


class ImputeCategoryMissingValues(BasePreprocessingStep):
    name: str = "ImputeCategoryMissingValues"
    description: str = "Impute missing values in categorical columns"

    def __init__(
        self,
        *,
        fill_value: str = "missing",
        parse: bool = True,
        columns: list[str] = None,
        exclude_columns: list[str] = None,
    ):
        self.fill_value = fill_value
        self.columns = columns or []
        self.exclude_columns = exclude_columns or []
        self.parse = parse

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.parse:
            self.columns = df.select(pl.col(pl.Categorical)).columns

        exprs = []
        for col in self.columns:
            if col not in df.columns:
                continue
            if col in self.exclude_columns:
                continue
            exprs.append(pl.col(col).fill_null(self.fill_value))

        if len(exprs) == 0:
            return df

        return df.with_columns(exprs)


PREPROCESSING_STEP_REGISTRY: dict[str, type[BasePreprocessingStep]] = {
    "DropColumns": DropColumns,
    "ConvertBooleanToInt": ConvertBooleanToInt,
    "ConvertDateToInt": ConvertDateToInt,
    "ConvertDatetimeToInt": ConvertDatetimeToInt,
    "ImputeCategoryMissingValues": ImputeCategoryMissingValues,
}
