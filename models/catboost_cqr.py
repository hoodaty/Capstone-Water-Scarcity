import numpy as np
from catboost import CatBoostRegressor

from models.base import QuantileRegressorBase


class CatBoostQuantileRegressor(QuantileRegressorBase):
    """CatBoost-backed quantile regressor with optional CQR calibration."""

    def _make_model(self, q: float) -> CatBoostRegressor:
        return CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}",
            verbose=False,
            **self.kwargs,
        )

    def fit(self, X, y):
        self.models = {
            "lower": self._make_model(self.alpha / 2).fit(X, y),
            "median": self._make_model(0.5).fit(X, y),
            "upper": self._make_model(1.0 - self.alpha / 2).fit(X, y),
        }
        return self

    def get_point_model(self):
        return self.models["median"]

    def predict(self, X, quantiles=None, calibrate=True):
        if quantiles is None or quantiles == "mean":
            return self.models["median"].predict(X)
        self._validate_quantiles(quantiles)
        lower = self.models["lower"].predict(X)
        upper = self.models["upper"].predict(X)
        if calibrate:
            lower, upper = self._apply_cqr(
                lower, upper, self.models["median"].predict(X)
            )
        return np.column_stack([lower, upper])
