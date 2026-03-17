import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class LightGBMQuantileRegressor(RegressorMixin, BaseEstimator):
    """
    Wrapper to train Lower, Median, and Upper quantile models with CQR calibration.
    """

    def __init__(self, alpha=0.1, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs
        self.models = {}
        self.q_score = 0.0

    def fit(self, X, y):
        # Train Lower Bound
        self.models["lower"] = lgb.LGBMRegressor(
            objective="quantile", alpha=self.alpha / 2, verbose=-1, **self.kwargs
        ).fit(X, y)

        # Train Median (Point Prediction)
        self.models["median"] = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5, verbose=-1, **self.kwargs
        ).fit(X, y)

        # Train Upper Bound
        self.models["upper"] = lgb.LGBMRegressor(
            objective="quantile", alpha=1.0 - self.alpha / 2, verbose=-1, **self.kwargs
        ).fit(X, y)
        return self

    def calibrate(self, X_calib, y_calib):
        """Compute CQR q_score from calibration set."""
        # Get raw quantiles (no calibration yet)
        preds = self.predict(
            X_calib, quantiles=[self.alpha / 2, 1 - self.alpha / 2], calibrate=False
        )
        low_calib = preds[:, 0]
        high_calib = preds[:, 1]

        # CQR Score: max(low - y, y - high)
        # We want to find a correction factor q such that:
        # P(low - q <= y <= high + q) >= 1 - alpha
        # This is equivalent to q >= max(low - y, y - high)
        scores = np.maximum(low_calib - y_calib, y_calib - high_calib)

        # Compute (1 - alpha) quantile of scores
        # If scores are mostly negative (good coverage), q_score might be negative (shrinking intervals).
        # If scores are positive (poor coverage), q_score will be positive (expanding intervals).
        self.q_score = np.quantile(scores, 1 - self.alpha)

    def get_point_model(self):
        """Return the point-estimate model for analysis utilities."""
        return self.models["median"]

    def predict(self, X, quantiles=None, calibrate=True):
        if quantiles == "mean":
            return self.models["median"].predict(X)
        elif isinstance(quantiles, (list, tuple, np.ndarray)):
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
            # Return shape (N, 2) for [lower, upper]
            lower = self.models["lower"].predict(X)
            upper = self.models["upper"].predict(X)

            if calibrate and hasattr(self, "q_score"):
                lower = lower - self.q_score
                upper = upper + self.q_score

                # --- CONSISTENCY CHECKS (Strict Order) ---

                # 1. Physical Constraints (Pre-processing)
                # Water flow cannot be negative. We must clamp ALL components first.
                # If we don't clamp median here, Step 3 could re-introduce negative values.
                median = self.models["median"].predict(X)
                lower = np.maximum(0, lower)
                upper = np.maximum(0, upper)
                median = np.maximum(0, median)

                # 2. Quantile Sorting (Fix Crossing)
                # If Lower > Upper, collapse to mean.
                crossed = lower > upper
                if np.any(crossed):
                    mean_val = (lower[crossed] + upper[crossed]) / 2
                    lower[crossed] = mean_val
                    upper[crossed] = mean_val

                # 3. Conservative Expansion (Fix Median Consistency)
                # Ensure Median is within [Lower, Upper] by expanding bounds.
                lower = np.minimum(lower, median)
                upper = np.maximum(upper, median)

                # 4. Final Hard Guarantee
                # Ensure Upper is strictly greater than Lower (prevent division by zero later)
                upper = np.maximum(upper, lower + 1e-4)

            return np.column_stack([lower, upper])
        return self.models["median"].predict(X)
