import os
import pandas as pd
import lightgbm as lgb
import cloudpickle
import threading
import json
from ..core.config import settings

class ModelAgent:
    """
    Manages model training, loading, and inference.
    """
    def __init__(self):
        self.model_name = "baseline_lgbm"
        self.feature_set = "small"
        self.status = "Idle"
        self.model = None
        self.training_log = [] # List of {iter, metric}
        self._lock = threading.Lock()

    def _set_status(self, status: str):
        with self._lock:
            self.status = status
            print(f"[ModelAgent] Status: {status}")

    def get_status(self):
        with self._lock:
            return {"status": self.status, "log": self.training_log[-50:]} # Return tail of log

    def _lgb_callback(self, env):
        # LightGBM callback
        # env.iteration, env.evaluation_result_list
        # evaluation_result_list example: [('training', 'l2', 0.123, False)]
        with self._lock:
            # Only log every 10 iterations to reduce noise
            if env.iteration % 10 == 0:
                metric = env.evaluation_result_list[0][2] if env.evaluation_result_list else 0
                self.training_log.append({"iter": env.iteration, "metric": metric})

    def _xgb_callback(self, env):
        # XGBoost callback structure is different, depends on version/interface
        # Using a simple custom callback approach or parsing output might be easier
        # For simplicity in this demo, we'll skip detailed XGB live logging or implement a simple one after research.
        pass


    def train_model(self, config: dict = None):
        """
        Trains a model based on provided configuration.
        Config structure:
        {
            "model_type": "lightgbm" | "random_forest",
            "params": { ...hyperparams... },
            "feature_set": "small" | "medium" | "all"
        }
        """
        config = config or {}
        model_type = config.get("model_type", "lightgbm")
        params = config.get("params", {})
        feature_set_name = config.get("feature_set", "small")

        self.feature_set = feature_set_name

        self._set_status(f"Training Started ({model_type})")
        try:
            # Load metadata
            with open(f"{settings.DATA_DIR}/{settings.DATA_VERSION}/features.json", "r") as f:
                metadata = json.load(f)
            features = metadata["feature_sets"][self.feature_set]

            self._set_status("Loading Training Data...")
            train = pd.read_parquet(
                f"{settings.DATA_DIR}/{settings.DATA_VERSION}/train.parquet",
                columns=["era", "target"] + features
            )

            # Downsample for speed (configurable in future, keeping simple for now)
            # Use 1/4th of eras to speed up interactive training
            train = train[train["era"].isin(train["era"].unique()[::4])]

            self._set_status(f"Fitting {model_type}...")

            if model_type == "lightgbm":
                # Default LGBM params if not provided
                lgbm_params = {
                    "n_estimators": int(params.get("n_estimators", 2000)),
                    "learning_rate": float(params.get("learning_rate", 0.01)),
                    "max_depth": int(params.get("max_depth", 5)),
                    "num_leaves": int(params.get("num_leaves", 31)),
                    "colsample_bytree": 0.1,
                    "random_state": 42,
                    "verbosity": -1,
                    "metric": "l2" # Ensure we have a metric to log
                }
                model = lgb.LGBMRegressor(**lgbm_params)

                # Fit with callback
                with self._lock:
                    self.training_log = [] # Reset log

                model.fit(
                    train[features],
                    train["target"],
                    eval_set=[(train[features], train["target"])], # Eval on train for callback connectivity
                    eval_metric="l2",
                    callbacks=[self._lgb_callback]
                )
                from sklearn.ensemble import RandomForestRegressor
                rf_params = {
                    "n_estimators": int(params.get("n_estimators", 100)),
                    "max_depth": int(params.get("max_depth", 5)),
                    "n_jobs": -1,
                    "random_state": 42
                }
                model = RandomForestRegressor(**rf_params)
            elif model_type == "xgboost":
                import xgboost as xgb
                xgb_params = {
                    "n_estimators": int(params.get("n_estimators", 2000)),
                    "learning_rate": float(params.get("learning_rate", 0.01)),
                    "max_depth": int(params.get("max_depth", 5)),
                    "colsample_bytree": 0.1,
                    "n_jobs": -1,
                    "random_state": 42
                }
                model = xgb.XGBRegressor(**xgb_params)

            elif model_type == "catboost":
                from catboost import CatBoostRegressor
                cb_params = {
                    "iterations": int(params.get("n_estimators", 2000)),
                    "learning_rate": float(params.get("learning_rate", 0.01)),
                    "depth": int(params.get("max_depth", 6)),
                    "verbose": False,
                    "random_state": 42
                }
                model = CatBoostRegressor(**cb_params)

            elif model_type == "neural_network":
                # Simple MLP using PyTorch (Sklearn wrapper for simplicity in this agent)
                from sklearn.neural_network import MLPRegressor
                nn_params = {
                    "hidden_layer_sizes": (128, 64, 32),
                    "activation": "relu",
                    "solver": "adam",
                    "alpha": 0.0001,
                    "learning_rate_init": float(params.get("learning_rate", 0.001)),
                    "max_iter": int(params.get("n_estimators", 200)), # reusing n_estimators as epochs
                    "random_state": 42
                }
                model = MLPRegressor(**nn_params)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Specific fit for LGBM handles it above. For others:
            if model_type != "lightgbm":
                 with self._lock:
                    self.training_log = [] # Reset log
                 model.fit(train[features], train["target"])

            self._set_status("Saving Model...")
            # Create prediction wrapper
            def predict(live_features: pd.DataFrame, _benchmark=None) -> pd.DataFrame:
                preds = model.predict(live_features[features])
                return pd.DataFrame(preds, index=live_features.index, columns=["prediction"])

            with open(f"{settings.DATA_DIR}/{self.model_name}.pkl", "wb") as f:
                cloudpickle.dump(predict, f)

            self._set_status("Training Complete")
        except Exception as e:
            self._set_status(f"Error Training: {str(e)}")

    def run_inference(self, live_path: str):
        """Runs inference on live data."""
        self._set_status("Inference Started")
        try:
            with open(f"{settings.DATA_DIR}/{self.model_name}.pkl", "rb") as f:
                predict_fn = cloudpickle.load(f)

            # Load metadata for features
            with open(f"{settings.DATA_DIR}/{settings.DATA_VERSION}/features.json", "r") as f:
                metadata = json.load(f)
            features = metadata["feature_sets"][self.feature_set]

            self._set_status("Loading Live Data...")
            live_df = pd.read_parquet(live_path, columns=["era"] + features)

            self._set_status("Predicting...")
            preds = predict_fn(live_df, None)

            # Ensure index is handled if previously reset
            if "id" in live_df.columns:
                 preds.index = live_df["id"]

            self._set_status("Inference Complete")
            return preds
        except Exception as e:
            self._set_status(f"Error Inference: {str(e)}")
            return None

model_agent = ModelAgent()
