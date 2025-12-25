import os
import json
import pandas as pd
import lightgbm as lgb
from numerapi import NumerAPI
import cloudpickle
from dotenv import load_dotenv

# Load credentials from .env if it exists
load_dotenv()

# Configuration
DATA_VERSION = "v5.2"
MODEL_NAME = "baseline_lgbm"
FEATURES_SET = "small" # small, medium, or all

def setup_data(napi):
    print(f"--- Checking data for version {DATA_VERSION} ---")
    
    # Download metadata if missing
    metadata_path = f"{DATA_VERSION}/features.json"
    if not os.path.exists(metadata_path):
        print("Downloading features.json...")
        napi.download_dataset(f"{DATA_VERSION}/features.json", metadata_path)
    
    # Download training data if missing
    train_path = f"{DATA_VERSION}/train.parquet"
    if not os.path.exists(train_path):
        print("Downloading train.parquet (this might take a while)...")
        napi.download_dataset(f"{DATA_VERSION}/train.parquet", train_path)
    else:
        print("train.parquet already exists. Skipping download.")

def download_validation(napi):
    validation_path = f"{DATA_VERSION}/validation.parquet"
    if not os.path.exists(validation_path):
        print("Downloading validation.parquet...")
        napi.download_dataset(f"{DATA_VERSION}/validation.parquet", validation_path)
    else:
        print("validation.parquet already exists. Skipping download.")

def load_data(version, feature_set_name):
    print(f"--- Loading data (feature set: {feature_set_name}) ---")
    with open(f"{version}/features.json", "r") as f:
        metadata = json.load(f)
    
    features = metadata["feature_sets"][feature_set_name]
    
    print("Loading training data...")
    # Loading with specific columns to save memory
    train = pd.read_parquet(
        f"{version}/train.parquet",
        columns=["era", "target"] + features
    )
    
    # Optional: Downsample for faster experimentation (remove for final model)
    train = train[train["era"].isin(train["era"].unique()[::4])]
    
    return train, features

def train_model(train, features):
    print("--- Training Baseline LightGBM Model ---")
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=2**5-1,
        colsample_bytree=0.1,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(
        train[features],
        train["target"]
    )
    return model

def save_model(model, features, name):
    print(f"--- Saving model as {name}.pkl ---")
    
    def predict(live_features: pd.DataFrame, _live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
        preds = model.predict(live_features[features])
        submission = pd.Series(preds, index=live_features.index)
        return submission.to_frame("prediction")
    
    with open(f"{name}.pkl", "wb") as f:
        cloudpickle.dump(predict, f)
    print("Model saved successfully.")

if __name__ == "__main__":
    napi = NumerAPI()
    
    # Ensure data directories exist
    os.makedirs(DATA_VERSION, exist_ok=True)
    
    setup_data(napi)
    train_df, features_list = load_data(DATA_VERSION, FEATURES_SET)
    
    model = train_model(train_df, features_list)
    save_model(model, features_list, MODEL_NAME)
    
    print("--- Training Completed. Starting background validation download ---")
    download_validation(napi)
    
    print("--- Baseline Phase 1 Completed ---")
