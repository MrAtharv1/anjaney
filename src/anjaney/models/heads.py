"""Classical ML regression heads and baselines with standard scaling."""

import lightgbm as lgb
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def get_model_head(model_type: str = "ridge", seed: int = 42) -> BaseEstimator:
    """Instantiate a model pipeline wrapped with StandardScaler."""
    norm_type = model_type.lower().strip()

    if norm_type == "ridge":
        return Pipeline(
            [("scaler", StandardScaler()), ("regressor", Ridge(alpha=10.0, random_state=seed))]
        )

    if norm_type == "elasticnet":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=seed, max_iter=2000),
                ),
            ]
        )

    if norm_type == "svr":
        return Pipeline(
            [("scaler", StandardScaler()), ("regressor", SVR(kernel="rbf", C=2.0, epsilon=0.2))]
        )

    if norm_type == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=60,
            num_leaves=7,
            max_depth=3,
            min_child_samples=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            learning_rate=0.05,
            random_state=seed,
            verbose=-1,
        )

    if norm_type == "dummy":
        return DummyRegressor(strategy="mean")

    raise ValueError(
        f"Unsupported model type '{model_type}'. Choose from 'ridge', 'elasticnet', 'svr', 'lightgbm', 'dummy'."
    )
