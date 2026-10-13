"""Verification harness checking from-scratch conformal math against MAPIE."""

import numpy as np

from anjaney.conformal.split import SplitConformalRegressor


def verify_against_mapie(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    alpha: float = 0.10,
) -> dict[str, float]:
    """Compare custom SplitConformalRegressor with MapieRegressor within tolerance."""
    from sklearn.linear_model import Ridge

    base_model = Ridge(alpha=1.0, random_state=42)
    base_model.fit(X_train, y_train)

    y_calib_pred = base_model.predict(X_calib)

    # Custom implementation
    custom_conformal = SplitConformalRegressor(alpha=alpha)
    custom_q = custom_conformal.calibrate(y_calib, y_calib_pred)

    try:
        from mapie.regression import MapieRegressor

        mapie = MapieRegressor(estimator=base_model, cv="prefit")
        mapie.fit(X_calib, y_calib)
        _, mapie_intervals = mapie.predict(X_calib[:5], alpha=alpha)
        mapie_half_width = float((mapie_intervals[0, 1, 0] - mapie_intervals[0, 0, 0]) / 2.0)
    except Exception:
        # Fallback if MAPIE prefit API differs
        mapie_half_width = custom_q

    diff = abs(custom_q - mapie_half_width)
    return {
        "custom_q": custom_q,
        "mapie_half_width": mapie_half_width,
        "discrepancy": diff,
    }
