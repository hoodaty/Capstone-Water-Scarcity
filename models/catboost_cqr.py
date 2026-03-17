import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class CatBoostQuantileRegressor(RegressorMixin, BaseEstimator):
    """CatBoost wrapper for quantile intervals with optional CQR calibration."""

    def __init__(self, alpha=0.1, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs
        self.models = {}
        self.q_score = 0.0

    def _new_model(self, quantile_alpha: float) -> CatBoostRegressor:
        return CatBoostRegressor(
            loss_function=f"Quantile:alpha={quantile_alpha}",
            verbose=False,
            **self.kwargs,
        )

    def fit(self, X, y):
        self.models["lower"] = self._new_model(self.alpha / 2).fit(X, y)
        self.models["median"] = self._new_model(0.5).fit(X, y)
        self.models["upper"] = self._new_model(1.0 - self.alpha / 2).fit(X, y)
        return self

    def _validate_quantiles(self, quantiles):
        if len(quantiles) != 2:
            raise ValueError(
                "quantiles must contain exactly two values: [alpha/2, 1-alpha/2]."
            )
        expected = np.array([self.alpha / 2, 1 - self.alpha / 2], dtype=float)
        provided = np.array(quantiles, dtype=float)
        if not np.allclose(provided, expected, atol=1e-8):
            raise ValueError(
                "Only quantiles [alpha/2, 1-alpha/2] are supported for calibrated intervals."
            )

    def calibrate(self, X_calib, y_calib):
        preds = self.predict(
            X_calib, quantiles=[self.alpha / 2, 1 - self.alpha / 2], calibrate=False
        )
        lower = preds[:, 0]
        upper = preds[:, 1]
        scores = np.maximum(lower - y_calib, y_calib - upper)
        self.q_score = np.quantile(scores, 1 - self.alpha)

    def get_point_model(self):
        return self.models["median"]

    def predict(self, X, quantiles=None, calibrate=True):
        if quantiles is None or quantiles == "mean":
            return self.models["median"].predict(X)
        if isinstance(quantiles, (list, tuple, np.ndarray)):
            self._validate_quantiles(quantiles)
            lower = self.models["lower"].predict(X)
            upper = self.models["upper"].predict(X)

            if calibrate and hasattr(self, "q_score"):
                lower = lower - self.q_score
                upper = upper + self.q_score

                median = self.models["median"].predict(X)
                lower = np.maximum(0, lower)
                upper = np.maximum(0, upper)
                median = np.maximum(0, median)

                crossed = lower > upper
                if np.any(crossed):
                    center = (lower[crossed] + upper[crossed]) / 2
                    lower[crossed] = center
                    upper[crossed] = center

                lower = np.minimum(lower, median)
                upper = np.maximum(upper, median)
                upper = np.maximum(upper, lower + 1e-4)

            return np.column_stack([lower, upper])
        return self.models["median"].predict(X)
