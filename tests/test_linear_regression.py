"""linreg 的单元测试，其中三条与 sklearn 对拍。"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from linreg.linear_regression import (
    add_bias_term,
    fit_closed_form,
    fit_gradient_descent,
    predict,
    standardize,
)
from linreg.metrics import mean_absolute_error, mean_squared_error, r2_score


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X_raw = rng.normal(size=(300, 4))
    true_coef = np.array([1.5, -2.0, 0.7, 3.1])
    y = X_raw @ true_coef + 4.0 + rng.normal(scale=0.05, size=300)
    X_std, _, _ = standardize(X_raw)
    true_beta = np.concatenate([[4.0], true_coef])
    return X_raw, X_std, y, true_beta



def test_add_bias_term_shape():
    out = add_bias_term(np.ones((7, 3)))
    assert out.shape == (7, 4)
    assert np.allclose(out[:, 0], 1.0)


def test_closed_form_recovers_true_beta(toy_data):
    X_raw, _, y, true_beta = toy_data
    beta = fit_closed_form(add_bias_term(X_raw), y)
    assert np.allclose(beta, true_beta, atol=0.02)



def test_closed_form_matches_sklearn(toy_data):
    _, X_std, y, _ = toy_data
    X = add_bias_term(X_std)
    ours = fit_closed_form(X, y)
    sk = LinearRegression(fit_intercept=False).fit(X, y)
    assert np.allclose(ours, sk.coef_, atol=1e-8)



def test_gradient_descent_matches_closed_form(toy_data):
    _, X_std, y, _ = toy_data
    X = add_bias_term(X_std)
    beta_cf = fit_closed_form(X, y)
    beta_gd, losses = fit_gradient_descent(X, y, learning_rate=0.5)
    assert np.allclose(beta_cf, beta_gd, atol=1e-6)
    assert losses[-1] <= losses[0]



def test_metrics_on_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mean_squared_error(y, y) == pytest.approx(0.0)
    assert mean_absolute_error(y, y) == pytest.approx(0.0)
    assert r2_score(y, y) == pytest.approx(1.0)


def test_r2_matches_sklearn(toy_data):
    from sklearn.metrics import r2_score as sk_r2

    _, X_std, y, _ = toy_data
    beta = fit_closed_form(add_bias_term(X_std), y)
    y_pred = predict(add_bias_term(X_std), beta)
    assert r2_score(y, y_pred) == pytest.approx(sk_r2(y, y_pred), abs=1e-10)
