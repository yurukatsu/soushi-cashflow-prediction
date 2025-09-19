from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from ._regression import (
    root_mean_squared_error_w_penalty,
    sign_accuracy,
    sign_f1_score,
    sign_precision,
    sign_recall,
    symmetric_mean_absolute_percentage_error,
)

METRIC_MAP: dict[str, callable] = {
    "rmse": root_mean_squared_error,
    "mse": mean_squared_error,
    "rmse_w_penalty": root_mean_squared_error_w_penalty,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "smape": symmetric_mean_absolute_percentage_error,
    "sign_accuracy": sign_accuracy,
    "sign_recall": sign_recall,
    "sign_precision": sign_precision,
    "sign_f1_score": sign_f1_score,
    "r2": r2_score,
    "pinball": mean_pinball_loss,
}


__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "r2_score",
    "mean_pinball_loss",
    "symmetric_mean_absolute_percentage_error",
    "sign_accuracy",
    "sign_recall",
    "sign_precision",
    "sign_f1_score",
    "root_mean_squared_error",
    "root_mean_squared_error_w_penalty",
    "METRIC_MAP",
]
