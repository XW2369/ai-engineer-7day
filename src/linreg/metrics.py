"""回归任务的评价指标。"""

import numpy as np


def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true 有 {y_true.shape[0]} 个，y_pred 有 {y_pred.shape[0]} 个，对不上")
    return y_true, y_pred


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE = mean((y_true - y_pred)^2)。"""
    y_true, y_pred = _check_shapes(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE = mean(|y_true - y_pred|)。"""
    y_true, y_pred = _check_shapes(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 = 1 - SS_res / SS_tot，其中 SS_tot = sum((y_true - mean(y_true))^2)。

    注意分母是相对「均值」的离差平方和，不是 sum(y_true^2)。
    """
    y_true, y_pred = _check_shapes(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        raise ValueError("y_true 是常数，R^2 无定义")
    return 1.0 - ss_res / ss_tot
