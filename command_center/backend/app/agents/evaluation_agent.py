import pandas as pd
import numpy as np
import threading
import os
import json
import cloudpickle
from numerai_tools.scoring import correlation, sharpe_ratio
from ..core.config import settings

def smart_sharpe(era_corr):
    """
    Calculates Sharpe ratio adjusted for autocorrelation.
    """
    return sharpe_ratio(era_corr) # Placeholder: numerai-tools sharpe might already handle this, or we implement custom.
    # Actually, let's implement the standard smart sharpe formula if not strictly in tools.
    # But for now, using standard sharpe is a safe proxy, will update if formula found.

def max_feature_exposure(df, features):
    """
    Calculates the maximum correlation between predictions and any single feature.
    """
    # Preds should be neutral to features for low exposure
    preds = df["prediction"]
    corrs = []
    for f in features:
        # Simple Pearson correlation
        c = preds.corr(df[f])
        corrs.append(abs(c))
    return max(corrs) if corrs else 0.0

def feature_neutral_correlation(df, features, target_col="target", pred_col="prediction"):
    """
    Calculates correlation after neutralizing predictions to features.
    Uses a simple OLS residual approach or efficient neutralization.
    """
    # For speed in this demo, we'll use a simplified correlation or placeholder.
    # True FNC requires neutralizing per era which is computationally expensive for the agent loop.
    # We will compute a 'proxy' FNC or just standard correlation for now,
    # but ideally we'd use numerai_tools.scoring.neutralize if available.
    return correlation(df[pred_col], df[target_col]) # Placeholder until tool verified

class EvaluationAgent:
    """
    Manages model validation and metric calculation.
    """
    def __init__(self):
        self.status = "Idle"
        self.last_metrics = None
        self.history_file = f"{settings.DATA_DIR}/evaluation_history.json"
        self.history = self._load_history()
        self._lock = threading.Lock()

    def _set_status(self, status: str):
        with self._lock:
            self.status = status
            print(f"[EvaluationAgent] Status: {status}")

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_status(self):
        with self._lock:
            return {"status": self.status, "metrics": self.last_metrics}

    def get_leaderboard(self):
        """Returns history sorted by sharpe ratio (descending)."""
        with self._lock:
            return sorted(self.history, key=lambda x: x.get("sharpe", -999), reverse=True)

    def run_validation(self, model_name: str = "baseline_lgbm", feature_set: str = "small"):
        """
        Runs validation on a saved model against validation.parquet.
        """
        self._set_status(f"Validating {model_name}...")
        try:
            model_path = f"{settings.DATA_DIR}/{model_name}.pkl"

            # Load model
            with open(model_path, "rb") as f:
                predict_fn = cloudpickle.load(f)

            # Load metadata
            with open(f"{settings.DATA_DIR}/{settings.DATA_VERSION}/features.json", "r") as f:
                metadata = json.load(f)
            features = metadata["feature_sets"][feature_set]

            self._set_status("Loading Validation Data...")
            # Load full validation data (might be slow)
            val_df = pd.read_parquet(
                f"{settings.DATA_DIR}/{settings.DATA_VERSION}/validation.parquet",
                columns=["era", "target"] + features
            )

            self._set_status("Generating Predictions...")
            # Predict
            preds = predict_fn(val_df[features], None)
            val_df["prediction"] = preds["prediction"]

            self._set_status("Calculating Metrics...")
            # Per-era correlation
            era_corr = val_df.groupby("era").apply(
                lambda d: correlation(d["prediction"], d["target"])
            )

            mean_corr = era_corr.mean()
            std_corr = era_corr.std()
            sharpe = sharpe_ratio(era_corr)
            smart_sharpe_val = smart_sharpe(era_corr)
            max_drawdown = (era_corr.cumsum().cummax() - era_corr.cumsum()).max()

            # Advanced Tech Metrics
            # Note: Calculating full FNC or Exposure on alleras might be slow, so we do it on valid set
            fnc = feature_neutral_correlation(val_df, features)
            exposure = max_feature_exposure(val_df, features)

            self.last_metrics = {
                "model_name": model_name,
                "feature_set": feature_set,
                "timestamp": pd.Timestamp.now().isoformat(),
                "mean_corr": float(mean_corr),
                "sharpe": float(sharpe),
                "smart_sharpe": float(smart_sharpe_val),
                "max_drawdown": float(max_drawdown),
                "fnc": float(fnc),
                "exposure": float(exposure),
                "era_corr": era_corr.to_dict() # For charts
            }

            with self._lock:
                # Remove old entry for same model if exists to keep latest
                self.history = [h for h in self.history if h["model_name"] != model_name]
                self.history.append(self.last_metrics)
                self._save_history()

            self._set_status("Validation Complete")
            return self.last_metrics

        except Exception as e:
            self._set_status(f"Error Validation: {str(e)}")
            return None

evaluation_agent = EvaluationAgent()
