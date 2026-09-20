"""Linear regression from scratch (numpy only)."""

import numpy as np

from linreg.logging_utils import get_logger

logger = get_logger(__name__)


def add_bias_term(X: np.ndarray) -> np.ndarray:
    """在 X 左侧拼一列 1，让截距能被当成普通系数一起解出来。

    Args:
        X: 特征矩阵，形状 (n, d)。

    Returns:
        形状 (n, d + 1) 的矩阵，首列全为 1。
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])


def fit_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """用正规方程求解 beta_hat = (X^T X)^{-1} X^T y。

    用 np.linalg.solve 而不是 np.linalg.inv：
    数值更稳定，且 X^T X 奇异时会直接报错，而不是静默给出垃圾解。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError(f"X 必须是二维，实际是 {X.ndim} 维")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X 有 {X.shape[0]} 行，y 有 {y.shape[0]} 行，对不上")

    return np.linalg.solve(X.T @ X, X.T @ y)


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """返回 X @ beta。"""
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float).ravel()

    if X.shape[1] != beta.shape[0]:
        raise ValueError(f"X 有 {X.shape[1]} 列，beta 有 {beta.shape[0]} 个元素，乘不起来")

    return X @ beta


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_raw = rng.normal(size=(200, 3))
    y = X_raw @ np.array([2.0, -1.0, 0.5]) + 3.0 + rng.normal(scale=0.1, size=200)

    X = add_bias_term(X_raw)
    beta_hat = fit_closed_form(X, y)
    logger.info("闭式解 beta = %s", np.round(beta_hat, 4))
    logger.info("前 5 个预测值 = %s", np.round(predict(X[:5], beta_hat), 3))
