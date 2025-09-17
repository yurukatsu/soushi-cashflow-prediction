import datetime
from typing import Any, Generator

import numpy as np
import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import BaseCrossValidator

Window = tuple[
    datetime.datetime, datetime.datetime, datetime.datetime, datetime.datetime
]
Duration = str | relativedelta


def parse_duration(duration: str | relativedelta | None) -> relativedelta:
    """
    Parse a duration string into a relativedelta object.
    """
    if isinstance(duration, relativedelta):
        return duration

    if duration is None:
        return relativedelta()
    if duration == "":
        return relativedelta()

    if duration.endswith("m"):
        unit = int(duration[:-1])
        return relativedelta(months=unit)
    if duration.endswith("mo"):
        unit = int(duration[:-2])
        return relativedelta(months=unit)
    if duration.endswith(" month"):
        unit = int(duration[:-5])
        return relativedelta(months=unit)
    if duration.endswith(" months"):
        unit = int(duration[:-6])
        return relativedelta(months=unit)

    if duration.endswith("w"):
        unit = int(duration[:-1])
        return relativedelta(weeks=unit)
    if duration.endswith(" week"):
        unit = int(duration[:-5])
        return relativedelta(weeks=unit)
    if duration.endswith(" weeks"):
        unit = int(duration[:-6])
        return relativedelta(weeks=unit)

    if duration.endswith("d"):
        return relativedelta(days=int(duration[:-1]))
    if duration.endswith(" day"):
        return relativedelta(days=int(duration[:-4]))
    if duration.endswith(" days"):
        return relativedelta(days=int(duration[:-5]))

    raise ValueError(f"Unknown duration unit: {duration}")


def parse_date_string(date_string: str) -> datetime.datetime:
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%fZ",
        "%Y-%m-%dT%H-%M-%S",
        "%Y-%m-%dT%H-%M-%S.%f",
        "%Y-%m-%dT%H-%M-%S.%fZ",
        "%Y-%m-%d %H-%M-%S",
        "%Y-%m-%d %H-%M-%S.%f",
        "%Y-%m-%d %H-%M-%S.%fZ",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
        "%Y-%m",
        "%Y/%m",
        "%Y%m",
        "%Y",
    ]
    for format in formats:
        try:
            return datetime.datetime.strptime(date_string, format)
        except Exception:
            continue
    raise ValueError(f"Unknown date format: {date_string}")


class SlidingWindowCV(BaseCrossValidator):
    def __init__(
        self,
        df: pl.DataFrame,
        date_col: str,
        n_splits: int,
        start_date: str | datetime.datetime = None,
        end_date: str | datetime.datetime = None,
        train_duration: Duration = None,
        gap_duration: Duration = "3mo",
        validation_duration: Duration = "3mo",
        step_duration: Duration = "3mo",
    ):
        if start_date is None:
            self.start_date = df[date_col].min()
        elif isinstance(start_date, str):
            self.start_date = parse_date_string(start_date)
        else:
            self.start_date = start_date

        if end_date is None:
            self.end_date = df[date_col].max()
        elif isinstance(end_date, str):
            self.end_date = parse_date_string(end_date)
        else:
            self.end_date = end_date

        self.date_col = date_col
        self.windows = self.get_sliding_windows(
            n_splits,
            self.start_date,
            self.end_date,
            train_duration=train_duration,
            gap_duration=gap_duration,
            validation_duration=validation_duration,
            step_duration=step_duration,
        )

    @staticmethod
    def get_sliding_windows(
        n_splits: int,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        train_duration: Duration = None,
        gap_duration: Duration = "3mo",
        validation_duration: Duration = "3mo",
        step_duration: Duration = "3mo",
    ) -> list[Window]:
        """
        Get sliding windows for time series cross-validation.
        """
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")

        train_duration = parse_duration(train_duration)
        gap_duration = parse_duration(gap_duration)
        validation_duration = parse_duration(validation_duration)
        step_duration = parse_duration(step_duration)
        windows = []

        for i in range(n_splits):
            offset = step_duration * i

            val_end = end_date - offset
            val_start = val_end - validation_duration + relativedelta(days=1)

            train_end = val_start - gap_duration - relativedelta(days=1)
            if train_duration:
                train_start = train_end - train_duration + relativedelta(days=1)
            else:
                train_start = start_date + (n_splits - 1 - i) * step_duration

            windows.append((train_start, train_end, val_start, val_end))

        return sorted(windows, key=lambda x: x[3])

    def split(self, X, y=None, groups=None):
        """
        Generate indices for train/test (validation) splits based on sliding windows.
        """
        if isinstance(X, pl.DataFrame):
            dates = X[self.date_col].to_numpy()
        elif isinstance(X, pd.DataFrame):
            dates = X.loc[:, self.date_col]
        else:
            raise ValueError("X must be a polars or pandas DataFrame")

        for train_start, train_end, val_start, val_end in self.windows:
            train_idx = np.where(
                (dates >= np.datetime64(train_start))
                & (dates <= np.datetime64(train_end))
            )[0]
            val_idx = np.where(
                (dates >= np.datetime64(val_start)) & (dates <= np.datetime64(val_end))
            )[0]
            yield train_idx, val_idx

    def split_dataframe(
        self, df: pl.DataFrame
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame], Any, None]:
        """
        Generate train/test (validation) splits as polars DataFrames based on sliding windows.
        """
        for train_start, train_end, val_start, val_end in self.windows:
            train_df = df.filter(
                (pl.col(self.date_col) >= pl.lit(train_start))
                & (pl.col(self.date_col) <= pl.lit(train_end))
            )
            val_df = df.filter(
                (pl.col(self.date_col) >= pl.lit(val_start))
                & (pl.col(self.date_col) <= pl.lit(val_end))
            )
            yield train_df, val_df

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.windows)
