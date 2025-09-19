import numpy as np
from sklearn.metrics import mean_squared_error


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute the symmetric mean absolute percentage error (SMAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
    return np.mean(diff)


def sign_accuracy(y_true, y_pred):
    """
    Compute the sign accuracy.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def sign_recall(y_true, y_pred):
    """
    Compute the sign recall.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    true_positives = np.sum((y_true > 0) & (y_pred > 0))
    false_negatives = np.sum((y_true > 0) & (y_pred <= 0))
    return (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )


def sign_precision(y_true, y_pred):
    """
    Compute the sign precision.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    true_positives = np.sum((y_true > 0) & (y_pred > 0))
    false_positives = np.sum((y_true <= 0) & (y_pred > 0))
    return (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )


def sign_f1_score(y_true, y_pred):
    """
    Compute the sign F1 score.
    """
    precision = sign_precision(y_true, y_pred)
    recall = sign_recall(y_true, y_pred)
    return (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )


def mean_squared_error_w_penalty(y_true, y_pred, penalty=2):
    """
    Compute the mean squared error (MSE) with an optional penalty.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true, y_pred = (
        (y_true - y_pred) * (1 + (penalty - 1) * (np.sign(y_true) != np.sign(y_pred))),
        np.zeros_like(y_pred),
    )

    return mean_squared_error(y_true, y_pred)


def root_mean_squared_error_w_penalty(y_true, y_pred, penalty: float = 2):
    """
    Compute the mean squared error (MSE) with an optional penalty.
    """
    return np.sqrt(mean_squared_error_w_penalty(y_true, y_pred, penalty=penalty))
