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

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把每一列变成均值 0、标准差 1（不含偏置列）。

    Args:
        X: 原始特征矩阵，形状 (n, d)。

    Returns:
        (X_std, mu, sigma)：标准化后的矩阵、每列均值、每列标准差。
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    """用梯度下降最小化均方误差。

    损失：L(beta) = (1/n) * ||X @ beta - y||^2
    梯度：dL/dbeta = (2/n) * X^T @ (X @ beta - y)

    Args:
        X: 设计矩阵 (n, d)，必须已加偏置列，且建议已标准化。
        y: 目标值 (n,)。
        learning_rate: 步长。
        max_iter: 最大迭代次数。
        tol: 参数变化小于该值时提前停止。

    Returns:
        (beta, losses)：最终参数，以及每轮迭代的损失值。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n, d = X.shape
    beta = np.zeros(d)
    losses: list[float] = []

    for i in range(max_iter):
        residual = X @ beta - y
        losses.append(float(residual @ residual / n))

        gradient = (2.0 / n) * (X.T @ residual)
        beta_new = beta - learning_rate * gradient

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            residual = X @ beta - y
            losses.append(float(residual @ residual / n))
            logger.info("第 %d 轮收敛，最终损失 %.6f", i + 1, losses[-1])
            break

        beta = beta_new
    else:
        logger.warning("跑满 %d 轮仍未收敛，考虑调大 learning_rate 或 max_iter", max_iter)

    return beta, losses


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_raw = rng.normal(size=(200, 3))
    y = X_raw @ np.array([2.0, -1.0, 0.5]) + 3.0 + rng.normal(scale=0.1, size=200)

    X_std, mu, sigma = standardize(X_raw)
    X = add_bias_term(X_std)

    beta_cf = fit_closed_form(X, y)
    logger.info("闭式解   beta = %s", np.round(beta_cf, 4))

    beta_gd, losses = fit_gradient_descent(X, y, learning_rate=0.5)
    logger.info("梯度下降 beta = %s", np.round(beta_gd, 4))
    logger.info("两者最大差异   = %.8f", np.max(np.abs(beta_cf - beta_gd)))
    logger.info("损失 %.4f -> %.6f，共 %d 轮", losses[0], losses[-1], len(losses))
