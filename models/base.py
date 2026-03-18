import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class QuantileRegressorBase(RegressorMixin, BaseEstimator):
    """Base class for three-quantile (lower / median / upper) regressors with CQR calibration."""

    def __init__(self, alpha: float = 0.1, **kwargs):
        self.alpha = alpha
        self.kwargs = kwargs
        self.q_score: float = 0.0

    def _validate_quantiles(self, quantiles) -> None:
        if len(quantiles) != 2:
            raise ValueError("quantiles must contain exactly two values: [alpha/2, 1-alpha/2].")
        expected = np.array([self.alpha / 2, 1 - self.alpha / 2], dtype=float)
        if not np.allclose(np.array(quantiles, dtype=float), expected, atol=1e-8):
            raise ValueError(
                f"Only quantiles {list(expected)} are supported for calibrated intervals."
            )

    def calibrate(self, X_calib, y_calib) -> None:
        """Compute the CQR conformity score from a held-out calibration set."""
        preds = self.predict(X_calib, quantiles=[self.alpha / 2, 1 - self.alpha / 2], calibrate=False)
        scores = np.maximum(preds[:, 0] - y_calib, y_calib - preds[:, 1])
        self.q_score = float(np.quantile(scores, 1 - self.alpha))

    def _apply_cqr(
        self, lower: np.ndarray, upper: np.ndarray, median: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand intervals by q_score and enforce physical + monotonicity constraints."""
        lower = lower - self.q_score
        upper = upper + self.q_score

        # Water flow is non-negative.
        lower  = np.maximum(0.0, lower)
        upper  = np.maximum(0.0, upper)
        median = np.maximum(0.0, median)

        # Fix quantile crossing: collapse to midpoint.
        crossed = lower > upper
        if np.any(crossed):
            center = (lower[crossed] + upper[crossed]) / 2
            lower[crossed] = center
            upper[crossed] = center

        # Ensure median is contained within [lower, upper].
        lower = np.minimum(lower, median)
        upper = np.maximum(upper, median)

        # Hard lower bound on interval width to prevent division-by-zero downstream.
        upper = np.maximum(upper, lower + 1e-4)

        return lower, upper

    def get_point_model(self):
        raise NotImplementedError
