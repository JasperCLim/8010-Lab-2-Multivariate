"""Modeling helpers for the scikit-learn diabetes progression lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "diabetes.csv"
TARGET = "target"
BMI = "bmi"
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the checked-in, unmodified diabetes dataset."""
    frame = pd.read_csv(path)
    if TARGET not in frame or BMI not in frame:
        raise ValueError(f"{path} must contain the {BMI!r} and {TARGET!r} columns")
    if frame.isna().any().any():
        raise ValueError("The diabetes dataset contains missing values")
    return frame


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Create deterministic train/validation/test partitions of 75/10/15 percent."""
    X = frame.drop(columns=TARGET)
    y = frame[TARGET]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    validation_fraction_of_remainder = 0.10 / 0.85
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction_of_remainder,
        random_state=RANDOM_STATE,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def make_univariate_models() -> dict[str, Any]:
    """Return BMI polynomial regression models for degrees zero through five."""
    models: dict[str, Any] = {"BMI polynomial degree 0": DummyRegressor(strategy="mean")}
    for degree in range(1, 6):
        models[f"BMI polynomial degree {degree}"] = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        )
    return models


def make_multivariate_models() -> dict[str, Any]:
    """Return the requested regression models using all ten baseline features."""
    return {
        "Multivariate polynomial degree 2": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
        ),
        "Multivariate polynomial degree 3": make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False), LinearRegression()
        ),
        "Decision tree max_depth=2": DecisionTreeRegressor(
            max_depth=2, random_state=RANDOM_STATE
        ),
        "Decision tree max_depth=4": DecisionTreeRegressor(
            max_depth=4, random_state=RANDOM_STATE
        ),
        "kNN k=5": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "kNN k=15": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
    }


def regression_metrics(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(X)
    return {
        "R-squared": float(r2_score(y, predictions)),
        "MAE": float(mean_absolute_error(y, predictions)),
        "MAPE (%)": float(mean_absolute_percentage_error(y, predictions) * 100),
    }


def evaluate_regressors(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit each regression model on train only and report train/validation metrics."""
    rows: list[dict[str, float | str]] = []
    fitted: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        rows.append(
            {
                "Model": name,
                **{f"Train {key}": value for key, value in regression_metrics(model, X_train, y_train).items()},
                **{
                    f"Validation {key}": value
                    for key, value in regression_metrics(model, X_validation, y_validation).items()
                },
            }
        )
    return pd.DataFrame(rows).set_index("Model"), fitted


def make_logistic_models() -> dict[str, Any]:
    """Return two regularized logistic classifiers for the high-progression label."""
    return {
        f"Logistic regression C={regularization}": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=regularization, max_iter=2000, random_state=RANDOM_STATE),
        )
        for regularization in (0.1, 1.0)
    }


def classification_metrics(
    model: Any, X: pd.DataFrame, y_binary: pd.Series
) -> dict[str, float]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": float(accuracy_score(y_binary, predictions)),
        "Precision": float(precision_score(y_binary, predictions, zero_division=0)),
        "Recall": float(recall_score(y_binary, predictions, zero_division=0)),
        "F1": float(f1_score(y_binary, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_binary, probabilities)),
    }


def evaluate_logistic_models(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], float]:
    """Fit classifiers against a training-median high-progression threshold."""
    threshold = float(y_train.median())
    y_train_high = (y_train >= threshold).astype(int)
    y_validation_high = (y_validation >= threshold).astype(int)
    rows: list[dict[str, float | str]] = []
    fitted: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(X_train, y_train_high)
        fitted[name] = model
        rows.append(
            {
                "Model": name,
                **{
                    f"Train {key}": value
                    for key, value in classification_metrics(model, X_train, y_train_high).items()
                },
                **{
                    f"Validation {key}": value
                    for key, value in classification_metrics(
                        model, X_validation, y_validation_high
                    ).items()
                },
            }
        )
    return pd.DataFrame(rows).set_index("Model"), fitted, threshold


def polynomial_parameter_count(model: Any) -> int:
    """Count polynomial coefficients plus the fitted intercept."""
    if isinstance(model, DummyRegressor):
        return 1
    polynomial = model.named_steps["polynomialfeatures"]
    linear = model.named_steps["linearregression"]
    return int(polynomial.n_output_features_ * np.size(linear.coef_) + np.size(linear.intercept_))


def describe_parameterization(model: Any) -> str:
    """Return a useful count/description for heterogeneous estimators."""
    if isinstance(model, DummyRegressor):
        return "1 fitted constant (mean)"
    if isinstance(model, DecisionTreeRegressor):
        return f"{model.get_n_leaves()} leaf values; split rules are not coefficients"
    if isinstance(model, KNeighborsRegressor) or (
        hasattr(model, "named_steps") and "kneighborsregressor" in model.named_steps
    ):
        return "No fitted coefficients; stores training examples"
    if hasattr(model, "named_steps") and "polynomialfeatures" in model.named_steps:
        return f"{polynomial_parameter_count(model)} coefficients incl. intercept"
    return "See estimator; parameter count is not directly comparable"


def polynomial_equation(model: Any, feature: str = BMI, digits: int = 2) -> str:
    """Format a fitted univariate polynomial as a readable equation."""
    if isinstance(model, DummyRegressor):
        return f"progression = {float(model.constant_[0]):.{digits}f}"
    polynomial = model.named_steps["polynomialfeatures"]
    linear = model.named_steps["linearregression"]
    terms = [f"{float(linear.intercept_):.{digits}f}"]
    for power, coefficient in enumerate(np.ravel(linear.coef_), start=1):
        terms.append(f"{float(coefficient):+.{digits}f} * {feature}^{power}")
    return "progression = " + " ".join(terms)


def select_best_by_validation_mae(metrics: pd.DataFrame) -> str:
    return str(metrics["Validation MAE"].idxmin())


def test_regression_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    return regression_metrics(model, X_test, y_test)


def test_logistic_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float
) -> dict[str, float]:
    y_test_high = (y_test >= threshold).astype(int)
    return classification_metrics(model, X_test, y_test_high)


def bmi_prediction(model: Any, bmi_value: float) -> float:
    """Predict progression at one BMI value (the source feature is standardized)."""
    return float(model.predict(pd.DataFrame({BMI: [bmi_value]}))[0])
