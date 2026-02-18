import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor


class QuantileRandomForestRegressor(RegressorMixin, BaseEstimator):
    """Random Forest wrapper to produce calibrated quantile intervals."""

    def __init__(self, alpha=0.1, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs
        self.model = None
        self.q_score = 0.0

    def fit(self, X, y):
        self.model = RandomForestRegressor(**self.kwargs)
        self.model.fit(X, y)
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

    def _predict_tree_matrix(self, X):
        if self.model is None or not hasattr(self.model, "estimators_"):
            raise RuntimeError("Model is not fitted.")
        tree_preds = [tree.predict(X) for tree in self.model.estimators_]
        return np.stack(tree_preds, axis=1)

    def calibrate(self, X_calib, y_calib):
        preds = self.predict(
            X_calib, quantiles=[self.alpha / 2, 1 - self.alpha / 2], calibrate=False
        )
        low_calib = preds[:, 0]
        high_calib = preds[:, 1]
        scores = np.maximum(low_calib - y_calib, y_calib - high_calib)
        self.q_score = np.quantile(scores, 1 - self.alpha)

    def get_point_model(self):
        return self.model

    def predict(self, X, quantiles=None, calibrate=True):
        if quantiles is None or quantiles == "mean":
            return self.model.predict(X)
        if isinstance(quantiles, (list, tuple, np.ndarray)):
            self._validate_quantiles(quantiles)
            tree_matrix = self._predict_tree_matrix(X)
            lower = np.quantile(tree_matrix, self.alpha / 2, axis=1)
            upper = np.quantile(tree_matrix, 1 - self.alpha / 2, axis=1)

            if calibrate and hasattr(self, "q_score"):
                lower = lower - self.q_score
                upper = upper + self.q_score

                median = self.model.predict(X)
                lower = np.maximum(0, lower)
                upper = np.maximum(0, upper)
                median = np.maximum(0, median)

                crossed = lower > upper
                if np.any(crossed):
                    mean_val = (lower[crossed] + upper[crossed]) / 2
                    lower[crossed] = mean_val
                    upper[crossed] = mean_val

                lower = np.minimum(lower, median)
                upper = np.maximum(upper, median)
                upper = np.maximum(upper, lower + 1e-4)

            return np.column_stack([lower, upper])
        return self.model.predict(X)
