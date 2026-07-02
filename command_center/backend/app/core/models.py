"""Model registry: defines all model configurations for the training suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    """A single model configuration for training."""

    name: str
    model_type: str  # "lgbm", "xgb", "catboost"
    params: Dict[str, Any] = field(default_factory=dict)
    feature_set: str = "small"
    use_engineered_features: bool = False


# ---------------------------------------------------------------------------
# Competitive hyperparameters based on leaderboard analysis:
#   lr=0.006-0.014, depth=4-6, colsample=0.06-0.14
# ---------------------------------------------------------------------------

MODEL_SUITE: List[ModelConfig] = [
    # --- LightGBM variants ---
    ModelConfig(
        name="lgbm_conservative",
        model_type="lgbm",
        params={
            "n_estimators": 2000,
            "learning_rate": 0.006,
            "max_depth": 4,
            "num_leaves": 15,
            "colsample_bytree": 0.06,
            "subsample": 0.8,
            "random_state": 42,
            "verbosity": -1,
        },
        feature_set="small",
    ),
    ModelConfig(
        name="lgbm_balanced",
        model_type="lgbm",
        params={
            "n_estimators": 2000,
            "learning_rate": 0.01,
            "max_depth": 5,
            "num_leaves": 31,
            "colsample_bytree": 0.1,
            "subsample": 0.8,
            "random_state": 123,
            "verbosity": -1,
        },
        feature_set="small",
    ),
    ModelConfig(
        name="lgbm_aggressive",
        model_type="lgbm",
        params={
            "n_estimators": 2000,
            "learning_rate": 0.014,
            "max_depth": 6,
            "num_leaves": 63,
            "colsample_bytree": 0.14,
            "subsample": 0.7,
            "random_state": 777,
            "verbosity": -1,
        },
        feature_set="small",
    ),
    # --- XGBoost ---
    ModelConfig(
        name="xgb_default",
        model_type="xgb",
        params={
            "n_estimators": 2000,
            "learning_rate": 0.008,
            "max_depth": 5,
            "colsample_bytree": 0.1,
            "subsample": 0.8,
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
        },
        feature_set="small",
    ),
    # --- CatBoost ---
    ModelConfig(
        name="catboost_default",
        model_type="catboost",
        params={
            "iterations": 2000,
            "learning_rate": 0.01,
            "depth": 5,
            "colsample_bylevel": 0.1,
            "random_seed": 42,
            "verbose": 0,
        },
        feature_set="small",
    ),
    # --- LightGBM with engineered features ---
    ModelConfig(
        name="lgbm_engineered",
        model_type="lgbm",
        params={
            "n_estimators": 2000,
            "learning_rate": 0.01,
            "max_depth": 5,
            "num_leaves": 31,
            "colsample_bytree": 0.1,
            "subsample": 0.8,
            "random_state": 42,
            "verbosity": -1,
        },
        feature_set="small",
        use_engineered_features=True,
    ),
]
