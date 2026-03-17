from dataclasses import dataclass
from typing import Dict, Type

from models.lgbm_cqr import LightGBMQuantileRegressor
from models.quantile_rf import QuantileRandomForestRegressor


@dataclass(frozen=True)
class ModelSpec:
    id: str
    cls: Type
    display_name: str
    calibrated_name: str
    init_kwargs: Dict[str, object]


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "lgbm_cqr": ModelSpec(
        id="lgbm_cqr",
        cls=LightGBMQuantileRegressor,
        display_name="LightGBM (Quantile)",
        calibrated_name="LightGBM+CQR",
        init_kwargs={
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
        },
    ),
    "quantile_rf": ModelSpec(
        id="quantile_rf",
        cls=QuantileRandomForestRegressor,
        display_name="Quantile Random Forest",
        calibrated_name="Quantile Random Forest + CQR",
        init_kwargs={
            "n_estimators": 200,
            "max_depth": 14,
            "min_samples_leaf": 10,
            "n_jobs": -1,
            "random_state": 42,
        },
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model_id '{model_id}'. Available: {available}"
        ) from exc


def list_model_ids() -> list[str]:
    return sorted(MODEL_REGISTRY)
