import warnings

import lightgbm as lgb
import numpy as np

from models.base import QuantileRegressorBase


class LightGBMQuantileRegressor(QuantileRegressorBase):
    """LightGBM-backed quantile regressor with optional CQR calibration."""

    def fit(self, X, y):
        X = np.asarray(X)

        def _train(q):
            return lgb.LGBMRegressor(
                objective="quantile", alpha=q, verbose=-1, **self.kwargs
            ).fit(X, y)

        self.models = {
            "lower": _train(self.alpha / 2),
            "median": _train(0.5),
            "upper": _train(1.0 - self.alpha / 2),
        }
        return self

    def get_point_model(self):
        return self.models["median"]

    def predict(self, X, quantiles=None, calibrate=True):
        X = np.asarray(X)
        # LightGBM 4.6+ implements feature_names_in_ as a property returning
        # auto-generated Column_N names for numpy input. sklearn then warns when
        # predict receives a plain numpy array. Suppress: predictions are correct.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            if quantiles is None or quantiles == "mean":
                return self.models["median"].predict(X)
            self._validate_quantiles(quantiles)
            lower = self.models["lower"].predict(X)
            median = self.models["median"].predict(X)
            upper = self.models["upper"].predict(X)
        if calibrate:
            lower, upper = self._apply_cqr(lower, upper, median)
        return np.column_stack([lower, upper])
