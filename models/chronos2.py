import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class Chronos2QuantileRegressor(RegressorMixin, BaseEstimator):
    """
    Chronos-2 wrapper with CQR calibration.

    This wrapper is designed for run.py's split DataFrames (with station_code and
    ObsDate), where Chronos-2 performs zero-shot forecasting and CQR widens
    prediction intervals.
    """

    def __init__(self, alpha=0.1, model_name="amazon/chronos-2", device_map=None, **kwargs):
        self.alpha = alpha
        self.model_name = model_name
        self.device_map = device_map
        self.kwargs = kwargs
        self.q_score = 0.0

        self.pipeline = None
        self.context_df = None
        self.target_col = None

    def _load_pipeline(self):
        if self.pipeline is not None:
            return

        try:
            import torch
            from chronos import Chronos2Pipeline
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "Chronos-2 requires `chronos-forecasting` and torch. "
                "Install with uv and ensure dependency versions are compatible."
            ) from exc

        device = self.device_map
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name,
            device_map=device,
            **self.kwargs,
        )

    @staticmethod
    def _coerce_timestamp(df: pd.DataFrame) -> pd.Series:
        if "ObsDate" in df.columns:
            return pd.to_datetime(df["ObsDate"])
        return pd.to_datetime(df.index)

    def fit(self, X, y=None, train_df=None, target_col=None):
        # Keep sklearn compatibility while allowing explicit DataFrame usage.
        df = train_df if train_df is not None else X
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Chronos-2 fit requires a DataFrame input with station_code and ObsDate/index.")
        if "station_code" not in df.columns:
            raise ValueError("Chronos-2 fit requires `station_code` in DataFrame columns.")
        if target_col is None:
            raise ValueError("Chronos-2 fit requires target_col.")
        if target_col not in df.columns:
            raise ValueError(f"Chronos-2 fit could not find target column `{target_col}`.")

        timestamp = self._coerce_timestamp(df)
        context_df = pd.DataFrame(
            {
                "id": df["station_code"].astype(str).values,
                "timestamp": timestamp.values,
                "target": df[target_col].astype(float).values,
            }
        ).dropna(subset=["id", "timestamp", "target"])

        context_df = context_df.sort_values(["id", "timestamp"]).reset_index(drop=True)
        self.context_df = context_df
        self.target_col = target_col
        self._load_pipeline()
        return self

    @staticmethod
    def _get_quantile_column(pred_df: pd.DataFrame, q: float) -> str:
        candidates = [str(q), f"{q:.1f}", f"{q:.2f}", f"{q:.3f}"]
        for col in pred_df.columns:
            if str(col) in candidates:
                return col
        if abs(q - 0.5) < 1e-8:
            for fallback in ("prediction", "predictions", "median", "mean"):
                if fallback in pred_df.columns:
                    return fallback
        raise KeyError(f"Chronos output missing quantile column for {q}. Columns={list(pred_df.columns)}")

    def _predict_quantiles_from_df(self, df: pd.DataFrame, quantiles: list[float]) -> np.ndarray:
        if self.context_df is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if "station_code" not in df.columns:
            raise ValueError("Chronos-2 predict requires `station_code` in DataFrame columns.")

        self._load_pipeline()

        timestamp = self._coerce_timestamp(df)
        future = pd.DataFrame(
            {
                "id": df["station_code"].astype(str).values,
                "timestamp": timestamp.values,
                "__row_id": np.arange(len(df), dtype=int),
            }
        ).dropna(subset=["id", "timestamp"])

        outputs = []
        for station_id, future_station in future.groupby("id", sort=False):
            context_station = self.context_df[self.context_df["id"] == station_id]
            if context_station.empty:
                raise ValueError(
                    f"Chronos-2 found station `{station_id}` in prediction data but not in training context."
                )

            future_station = future_station.sort_values("timestamp")
            pred_df = self.pipeline.predict_df(
                context_df=context_station[["id", "timestamp", "target"]],
                future_df=future_station[["id", "timestamp"]],
                prediction_length=len(future_station),
                quantile_levels=quantiles,
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

            q_cols = [self._get_quantile_column(pred_df, q) for q in quantiles]
            merged = future_station.merge(
                pred_df[["id", "timestamp"] + q_cols],
                on=["id", "timestamp"],
                how="left",
            )
            outputs.append(merged)

        all_preds = pd.concat(outputs, ignore_index=True).sort_values("__row_id")
        pred_values = all_preds[[self._get_quantile_column(all_preds, q) for q in quantiles]].to_numpy()
        return pred_values

    def calibrate(self, X_calib, y_calib, target_col=None):
        """Compute CQR q_score from calibration set."""
        df = X_calib
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Chronos-2 calibrate requires DataFrame calibration input.")
        _ = target_col  # kept for signature parity

        preds = self.predict(df, quantiles=[self.alpha / 2, 1 - self.alpha / 2], calibrate=False)
        low_calib = preds[:, 0]
        high_calib = preds[:, 1]

        scores = np.maximum(low_calib - y_calib, y_calib - high_calib)
        self.q_score = np.quantile(scores, 1 - self.alpha)

    def get_point_model(self):
        return self

    def predict(self, X, quantiles=None, calibrate=True):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Chronos-2 predict requires DataFrame input with station_code and ObsDate/index.")

        low_q = self.alpha / 2
        high_q = 1 - self.alpha / 2
        raw_preds = self._predict_quantiles_from_df(X, [low_q, 0.5, high_q])

        lower = raw_preds[:, 0]
        median = raw_preds[:, 1]
        upper = raw_preds[:, 2]

        if quantiles == "mean":
            return median

        if isinstance(quantiles, (list, tuple, np.ndarray)):
            if len(quantiles) != 2:
                raise ValueError("quantiles must contain exactly two values: [alpha/2, 1-alpha/2].")

            expected = np.array([self.alpha / 2, 1 - self.alpha / 2], dtype=float)
            provided = np.array(quantiles, dtype=float)
            if not np.allclose(provided, expected, atol=1e-8):
                raise ValueError("Only quantiles [alpha/2, 1-alpha/2] are supported for calibrated intervals.")

            if calibrate and hasattr(self, "q_score"):
                lower = lower - self.q_score
                upper = upper + self.q_score

                median = np.maximum(0, median)
                lower = np.maximum(0, lower)
                upper = np.maximum(0, upper)

                crossed = lower > upper
                if np.any(crossed):
                    mean_val = (lower[crossed] + upper[crossed]) / 2
                    lower[crossed] = mean_val
                    upper[crossed] = mean_val

                lower = np.minimum(lower, median)
                upper = np.maximum(upper, median)
                upper = np.maximum(upper, lower + 1e-4)

            return np.column_stack([lower, upper])

        return median

    def __getstate__(self):
        state = self.__dict__.copy()
        # Avoid serializing the potentially large HF pipeline object.
        state["pipeline"] = None
        return state

