import numpy as np
from quantile_forest import RandomForestQuantileRegressor

from models.base import QuantileRegressorBase


class QuantileRandomForestRegressor(QuantileRegressorBase):
    """Quantile Random Forest wrapper with optional CQR calibration."""

    def fit(self, X, y):
        self.model = RandomForestQuantileRegressor(**self.kwargs).fit(X, y)
        return self

    def get_point_model(self):
        return self.model

    def predict(self, X, quantiles=None, calibrate=True):
        if quantiles is None or quantiles == "mean":
            return self.model.predict(X, quantiles="mean")
        self._validate_quantiles(quantiles)
        preds = self.model.predict(X, quantiles=[self.alpha / 2, 1 - self.alpha / 2])
        lower, upper = preds[:, 0], preds[:, 1]
        if calibrate:
            lower, upper = self._apply_cqr(lower, upper, self.model.predict(X, quantiles="mean"))
        return np.column_stack([lower, upper])
