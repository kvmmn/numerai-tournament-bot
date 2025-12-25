import pandas as pd
import numpy as np
from numerai_tools.scoring import correlation, sharpe_ratio
import cloudpickle
import os
import json

# Configuration
DATA_VERSION = "v5.2"
MODEL_PATH = "baseline_lgbm.pkl"
FEATURES_SET = "small" # Should match training

def validate(model_path, data_version, feature_set_name):
    print(f"--- Starting Validation of {model_path} ---")
    
    # Load model
    with open(model_path, "rb") as f:
        predict_fn = cloudpickle.load(f)
    
    # Load metadata for features
    with open(f"{data_version}/features.json", "r") as f:
        metadata = json.load(f)
    features = metadata["feature_sets"][feature_set_name]
    
    # Load validation data
    print("Loading validation.parquet...")
    validation_data = pd.read_parquet(
        f"{data_version}/validation.parquet",
        columns=["era", "target"] + features
    )
    
    # Generate predictions
    print("Generating predictions...")
    # The predict_fn expects (live_features, live_benchmark_models)
    # We pass None for benchmark models as we don't have them here
    preds_df = predict_fn(validation_data[features], None)
    validation_data["prediction"] = preds_df["prediction"]
    
    # Calculate per-era correlation
    print("Calculating per-era correlation...")
    era_corr = validation_data.groupby("era").apply(
        lambda d: correlation(d["prediction"], d["target"])
    )
    
    # Metrics
    mean_corr = era_corr.mean()
    std_corr = era_corr.std()
    sharpe = sharpe_ratio(era_corr)
    max_drawdown = (era_corr.cumsum().cummax() - era_corr.cumsum()).max()
    
    print("\n--- Validation Metrics ---")
    print(f"Mean Correlation: {mean_corr:.6f}")
    print(f"Sharpe Ratio:     {sharpe:.6f}")
    print(f"Max Drawdown:     {max_drawdown:.6f}")
    
    return {
        "mean_corr": mean_corr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown
    }

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(f"{DATA_VERSION}/validation.parquet"):
        validate(MODEL_PATH, DATA_VERSION, FEATURES_SET)
    else:
        print(f"Error: Required files not found. Ensure {MODEL_PATH} and {DATA_VERSION}/validation.parquet exist.")
