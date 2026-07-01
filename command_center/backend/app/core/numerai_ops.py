from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cloudpickle
import numpy as np
import pandas as pd
from numerapi import NumerAPI
from numerai_tools.scoring import correlation, sharpe_ratio

from .config import settings
from .ensemble import era_wise_rank_ensemble
from .feature_engineering import compute_group_features, neutralize_predictions_by_era
from .models import ModelConfig, MODEL_SUITE
from .submission_guard import (
    SubmissionGuardError,
    validate_raw_prediction_series,
)

logger = logging.getLogger(__name__)


class NumeraiOpsError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def root_dir() -> Path:
    return Path(settings.DATA_DIR).expanduser().resolve()


def data_dir() -> Path:
    return root_dir() / settings.DATA_VERSION


def model_path(name: str | None = None) -> Path:
    return root_dir() / f"{name or settings.MODEL_NAME}.pkl"


def champion_path() -> Path:
    return root_dir() / "champion.pkl"


def champion_metrics_path() -> Path:
    return root_dir() / "champion_metrics.json"


def last_known_good_path() -> Path:
    return root_dir() / "last_known_good.pkl"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def build_napi() -> NumerAPI:
    if not settings.NUMERAI_PUBLIC_ID or not settings.NUMERAI_SECRET_KEY:
        raise NumeraiOpsError(
            "Missing credentials. Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY in .env."
        )
    return NumerAPI(
        public_id=settings.NUMERAI_PUBLIC_ID,
        secret_key=settings.NUMERAI_SECRET_KEY,
    )


def load_features(feature_set: str | None = None) -> list[str]:
    selected = feature_set or settings.FEATURE_SET
    features_path = data_dir() / "features.json"
    if not features_path.exists():
        raise NumeraiOpsError(f"Missing features metadata: {features_path}")

    with features_path.open("r") as f:
        metadata = json.load(f)

    feature_sets = metadata.get("feature_sets", {})
    if selected not in feature_sets:
        available = ", ".join(sorted(feature_sets.keys()))
        raise NumeraiOpsError(
            f"Feature set '{selected}' is not available. Available: {available}"
        )
    return feature_sets[selected]


# ---------------------------------------------------------------------------
# Data sync
# ---------------------------------------------------------------------------

def _validate_dataset_file(path: Path, name: str) -> Dict[str, Any]:
    if name == "features.json":
        payload = json.loads(path.read_text())
        if not payload.get("feature_sets"):
            raise NumeraiOpsError(f"Features metadata has no feature_sets: {path}")
        return {"file": name, "feature_sets": len(payload["feature_sets"])}

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    required = ["era"]
    if name != "live.parquet":
        required.append("target")
    missing = [column for column in required if column not in parquet.schema.names]
    if missing:
        raise NumeraiOpsError(f"{name} is missing required columns: {', '.join(missing)}")
    # Metadata alone does not detect truncated/corrupt data pages. Read the
    # small control columns from every row group to validate the whole file.
    for index in range(parquet.num_row_groups):
        parquet.read_row_group(index, columns=required, use_threads=False)
    return {
        "file": name,
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.num_row_groups),
    }


def sync_datasets(napi: NumerAPI) -> Dict[str, Any]:
    files = ["features.json", "train.parquet", "validation.parquet", "live.parquet"]
    version_dir = data_dir()
    version_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "version": settings.DATA_VERSION,
        "path": str(version_dir),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "found": [],
        "downloaded": [],
        "refreshed": [],
        "invalid": [],
        "integrity": {},
        "missing": [],
    }

    for name in files:
        local_path = version_dir / name
        remote_name = f"{settings.DATA_VERSION}/{name}"
        must_download = not local_path.exists() or name == "live.parquet"

        if local_path.exists() and name == "live.parquet":
            report["refreshed"].append(name)
        elif local_path.exists():
            try:
                report["integrity"][name] = _validate_dataset_file(local_path, name)
                report["found"].append(name)
                must_download = False
            except Exception as exc:
                report["invalid"].append({"file": name, "error": str(exc)})
                must_download = True

        if not must_download:
            continue

        partial_path = local_path.with_suffix(local_path.suffix + ".partial")
        try:
            if partial_path.exists():
                partial_path.unlink()
            napi.download_dataset(remote_name, str(partial_path))
            report["integrity"][name] = _validate_dataset_file(partial_path, name)
            os.replace(partial_path, local_path)
            report["downloaded"].append(name)
        except Exception as exc:
            if partial_path.exists():
                partial_path.unlink()
            report["missing"].append(name)
            raise NumeraiOpsError(f"Failed to download {remote_name}: {exc}") from exc

    return report


# ---------------------------------------------------------------------------
# Single model training (backward compatible)
# ---------------------------------------------------------------------------

def _era_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    match = re.search(r"(\d+)$", text)
    return (int(match.group(1)) if match else -1, text)


def _prepare_training_frame(
    train: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    original_rows = int(len(train))
    resolved = train["target"].notna() & np.isfinite(train["target"].astype(float))
    dropped_target_rows = int((~resolved).sum())
    train = train.loc[resolved].copy()
    feature_values = train[features].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(feature_values).all():
        raise NumeraiOpsError("Training features contain non-finite values.")
    for column in features:
        train[column] = train[column].astype(np.float32)
    ordered = sorted(train["era"].unique(), key=_era_sort_key)
    stride = max(1, int(settings.TRAIN_ERA_STRIDE))
    selected = ordered[::stride]
    if len(selected) < settings.MIN_TRAIN_ERAS:
        raise NumeraiOpsError(
            f"Only {len(selected)} training eras remain; minimum is {settings.MIN_TRAIN_ERAS}."
        )
    train = train[train["era"].isin(selected)].copy()
    return train, {
        "original_rows": original_rows,
        "dropped_target_rows": dropped_target_rows,
        "resolved_era_count": len(ordered),
        "selected_era_count": len(selected),
        "selected_first_era": str(selected[0]),
        "selected_last_era": str(selected[-1]),
        "era_stride": stride,
        "train_rows": int(len(train)),
    }


def _predict_wrapper(
    model: Any,
    features: list[str],
    neutralization_proportion: float = 0.0,
):
    def predict(live_features: pd.DataFrame, _live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.Series(
            model.predict(live_features[features]),
            index=live_features.index,
        )
        if neutralization_proportion > 0:
            eras = (
                live_features["era"]
                if "era" in live_features.columns
                else pd.Series("era", index=live_features.index)
            )
            predictions = neutralize_predictions_by_era(
                predictions,
                live_features[features],
                eras,
                proportion=neutralization_proportion,
            )
        return pd.DataFrame({"prediction": predictions}, index=live_features.index)
    return predict


def _build_model(config: ModelConfig):
    """Instantiate a model object from a ModelConfig."""
    if config.model_type == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**config.params)
    elif config.model_type == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(**config.params)
    elif config.model_type == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**config.params)
    else:
        raise NumeraiOpsError(f"Unknown model type: {config.model_type}")


def train_model() -> Dict[str, Any]:
    """Train a single model (backward compatible)."""
    import lightgbm as lgb

    features = load_features(settings.FEATURE_SET)
    train_path = data_dir() / "train.parquet"
    if not train_path.exists():
        raise NumeraiOpsError(f"Missing train data: {train_path}")

    train = pd.read_parquet(train_path, columns=["era", "target"] + features)
    train, training_data_report = _prepare_training_frame(train, features)

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=31,
        colsample_bytree=0.1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train[features], train["target"])

    save_path = model_path()
    with save_path.open("wb") as f:
        cloudpickle.dump(
            _predict_wrapper(
                model,
                features,
                neutralization_proportion=settings.NEUTRALIZATION_PROPORTION,
            ),
            f,
        )

    checksum = hashlib.sha256(save_path.read_bytes()).hexdigest()
    model_id = f"{settings.MODEL_NAME}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    return {
        "model_id": model_id,
        "path": str(save_path),
        "checksum": checksum,
        "feature_set": settings.FEATURE_SET,
        "neutralization_proportion": settings.NEUTRALIZATION_PROPORTION,
        "train_rows": int(train.shape[0]),
        "training_data_report": training_data_report,
    }


# ---------------------------------------------------------------------------
# Multi-model training suite (NEW)
# ---------------------------------------------------------------------------

def train_model_suite(
    configs: List[ModelConfig] | None = None,
) -> List[Dict[str, Any]]:
    """
    Train multiple models sequentially with memory optimization.

    Each model is trained, serialized, and freed from memory before
    training the next one. This keeps RAM usage under control on 16GB systems.

    Returns a list of artifact dicts (one per model).
    """
    configs = configs or MODEL_SUITE
    artifacts: List[Dict[str, Any]] = []

    # Load training data once (shared across all models)
    all_feature_sets = set()
    for cfg in configs:
        all_feature_sets.add(cfg.feature_set)

    # Collect all unique features across all configs
    all_features = set()
    for fs_name in all_feature_sets:
        all_features.update(load_features(fs_name))

    train_path = data_dir() / "train.parquet"
    if not train_path.exists():
        raise NumeraiOpsError(f"Missing train data: {train_path}")

    all_features_list = sorted(all_features)
    logger.info("Loading training data with %d features...", len(all_features_list))
    train = pd.read_parquet(train_path, columns=["era", "target"] + all_features_list)

    train, training_data_report = _prepare_training_frame(
        train,
        all_features_list,
    )

    # Compute engineered features once (shared across models that need them)
    engineered_cols: list[str] = []
    needs_engineered = any(cfg.use_engineered_features for cfg in configs)
    if needs_engineered:
        logger.info("Computing group-level engineered features...")
        engineered_cols = compute_group_features(train, data_dir())
        logger.info("Added %d engineered features.", len(engineered_cols))

    # Train each model sequentially
    for cfg in configs:
        logger.info("Training model: %s (%s)", cfg.name, cfg.model_type)
        try:
            features = load_features(cfg.feature_set)
            if cfg.use_engineered_features and engineered_cols:
                features = features + engineered_cols

            model = _build_model(cfg)
            model.fit(train[features], train["target"])

            save = model_path(cfg.name)
            with save.open("wb") as f:
                cloudpickle.dump(
                    _predict_wrapper(
                        model,
                        features,
                        neutralization_proportion=settings.NEUTRALIZATION_PROPORTION,
                    ),
                    f,
                )

            checksum = hashlib.sha256(save.read_bytes()).hexdigest()
            model_id = f"{cfg.name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

            artifacts.append({
                "model_id": model_id,
                "name": cfg.name,
                "model_type": cfg.model_type,
                "path": str(save),
                "checksum": checksum,
                "feature_set": cfg.feature_set,
                "use_engineered_features": cfg.use_engineered_features,
                "train_rows": int(train.shape[0]),
                "training_data_report": training_data_report,
            })

            # Free model from memory
            del model
            gc.collect()
            logger.info("Model %s saved to %s", cfg.name, save)

        except Exception as exc:
            logger.error("Failed to train %s: %s", cfg.name, exc)
            artifacts.append({
                "model_id": f"{cfg.name}-FAILED",
                "name": cfg.name,
                "model_type": cfg.model_type,
                "path": "",
                "checksum": "",
                "feature_set": cfg.feature_set,
                "use_engineered_features": cfg.use_engineered_features,
                "train_rows": 0,
                "error": str(exc),
            })

    # Free training data
    del train
    gc.collect()

    return artifacts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(path: str | None = None, feature_set: str | None = None) -> Dict[str, Any]:
    eval_path = Path(path) if path else model_path()
    if not eval_path.exists():
        raise NumeraiOpsError(f"Missing model artifact: {eval_path}")

    features = load_features(feature_set or settings.FEATURE_SET)
    validation_path = data_dir() / "validation.parquet"
    if not validation_path.exists():
        raise NumeraiOpsError(f"Missing validation data: {validation_path}")

    with eval_path.open("rb") as f:
        predict_fn = cloudpickle.load(f)

    val_df = pd.read_parquet(validation_path, columns=["era", "target"] + features)
    # Recent validation files include unresolved eras with null targets. They
    # are valid for live-like inference but must not enter historical scoring.
    val_df = val_df[val_df["target"].notna()].copy()
    # Memory optimization
    for col in features:
        val_df[col] = val_df[col].astype(np.float32)

    preds = predict_fn(val_df[["era"] + features], None)
    if "prediction" not in preds.columns:
        first_col = preds.columns[0]
        preds = preds.rename(columns={first_col: "prediction"})
    val_df["prediction"] = preds["prediction"].to_numpy()

    era_corr = val_df.groupby("era", sort=False).apply(
        lambda era_df: correlation(era_df["target"], era_df["prediction"])
    )
    if era_corr.empty:
        raise NumeraiOpsError("Validation correlations are empty.")

    mean_corr = float(era_corr.mean())
    std_corr = float(era_corr.std())
    sharpe = float(sharpe_ratio(era_corr))
    max_drawdown = float((era_corr.cumsum().cummax() - era_corr.cumsum()).max())
    exposure = float(val_df[features].corrwith(val_df["prediction"]).abs().max())

    return {
        "validation_correlation": mean_corr,
        "correlation_std": std_corr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "feature_exposure": exposure,
        "drift_score": std_corr,
        "era_count": int(era_corr.shape[0]),
    }


def evaluate_model_suite(
    artifacts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Evaluate all models in the suite. Returns artifacts enriched with metrics.

    Memory-optimized: loads validation data once and reuses across models.
    """
    validation_path = data_dir() / "validation.parquet"
    if not validation_path.exists():
        raise NumeraiOpsError(f"Missing validation data: {validation_path}")

    # Collect all features needed
    all_features = set()
    for art in artifacts:
        if art.get("error"):
            continue
        fs = art.get("feature_set", settings.FEATURE_SET)
        all_features.update(load_features(fs))

    all_features_list = sorted(all_features)
    logger.info("Loading validation data with %d features...", len(all_features_list))
    val_df = pd.read_parquet(validation_path, columns=["era", "target"] + all_features_list)
    val_df = val_df[val_df["target"].notna()].copy()
    for col in all_features_list:
        val_df[col] = val_df[col].astype(np.float32)

    # Compute engineered features if any model needs them
    needs_engineered = any(art.get("use_engineered_features") for art in artifacts if not art.get("error"))
    engineered_cols: list[str] = []
    if needs_engineered:
        engineered_cols = compute_group_features(val_df, data_dir())

    results = []
    predictions_for_ensemble: Dict[str, pd.Series] = {}

    for art in artifacts:
        if art.get("error"):
            results.append({**art, "metrics": None})
            continue

        art_path = Path(art["path"])
        if not art_path.exists():
            results.append({**art, "metrics": None, "error": f"Missing: {art_path}"})
            continue

        try:
            with art_path.open("rb") as f:
                predict_fn = cloudpickle.load(f)

            features = load_features(art.get("feature_set", settings.FEATURE_SET))
            if art.get("use_engineered_features") and engineered_cols:
                features = features + engineered_cols

            preds = predict_fn(val_df[["era"] + features], None)
            if "prediction" not in preds.columns:
                preds = preds.rename(columns={preds.columns[0]: "prediction"})

            pred_series = preds["prediction"].to_numpy()
            val_df[f"pred_{art['name']}"] = pred_series
            predictions_for_ensemble[art["name"]] = pd.Series(pred_series, index=val_df.index)

            era_corr = val_df.groupby("era", sort=False).apply(
                lambda era_df, name=art["name"]: correlation(
                    era_df["target"], era_df[f"pred_{name}"]
                )
            )

            metrics = {
                "validation_correlation": float(era_corr.mean()),
                "correlation_std": float(era_corr.std()),
                "sharpe": float(sharpe_ratio(era_corr)),
                "max_drawdown": float((era_corr.cumsum().cummax() - era_corr.cumsum()).max()),
                "feature_exposure": float(
                    val_df[features].corrwith(val_df[f"pred_{art['name']}"]).abs().max()
                ),
                "era_count": int(era_corr.shape[0]),
            }
            results.append({**art, "metrics": metrics})

            del predict_fn
            gc.collect()

        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", art["name"], exc)
            results.append({**art, "metrics": None, "error": str(exc)})

    # Build ensemble predictions
    ensemble_metrics = None
    if len(predictions_for_ensemble) >= 2:
        logger.info("Building era-wise ranking ensemble from %d models...", len(predictions_for_ensemble))
        ensemble_preds = era_wise_rank_ensemble(predictions_for_ensemble, val_df["era"])
        val_df["pred_ensemble"] = ensemble_preds.values

        era_corr_ens = val_df.groupby("era", sort=False).apply(
            lambda era_df: correlation(era_df["target"], era_df["pred_ensemble"])
        )
        ensemble_metrics = {
            "validation_correlation": float(era_corr_ens.mean()),
            "correlation_std": float(era_corr_ens.std()),
            "sharpe": float(sharpe_ratio(era_corr_ens)),
            "max_drawdown": float((era_corr_ens.cumsum().cummax() - era_corr_ens.cumsum()).max()),
            "era_count": int(era_corr_ens.shape[0]),
        }
        logger.info(
            "Ensemble metrics: corr=%.5f, sharpe=%.4f",
            ensemble_metrics["validation_correlation"],
            ensemble_metrics["sharpe"],
        )

    del val_df
    gc.collect()

    return {
        "models": results,
        "ensemble_metrics": ensemble_metrics,
        "model_names_in_ensemble": list(predictions_for_ensemble.keys()),
    }


# ---------------------------------------------------------------------------
# Champion-Challenger
# ---------------------------------------------------------------------------

def save_champion(metrics: Dict[str, Any], artifact_path: str | None = None) -> None:
    """Save current model as champion with its metrics."""
    metrics_dest = champion_metrics_path()
    with metrics_dest.open("w") as f:
        json.dump(metrics, f, indent=2)

    if artifact_path:
        import shutil
        shutil.copy2(artifact_path, str(champion_path()))

    logger.info("Champion saved with corr=%.5f", metrics.get("validation_correlation", 0))


def load_champion_metrics() -> Optional[Dict[str, Any]]:
    """Load champion metrics. Returns None if no champion exists."""
    path = champion_metrics_path()
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def compare_with_champion(
    challenger_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare challenger against champion.

    Challenger wins if:
      - Better validation correlation AND
      - Equal or better Sharpe ratio
    """
    champion = load_champion_metrics()

    if champion is None:
        return {
            "decision": "PROMOTE",
            "reason": "No existing champion. Promoting challenger.",
            "champion_metrics": None,
            "challenger_metrics": challenger_metrics,
        }

    champ_corr = champion.get("validation_correlation", 0)
    chall_corr = challenger_metrics.get("validation_correlation", 0)
    champ_sharpe = champion.get("sharpe", 0)
    chall_sharpe = challenger_metrics.get("sharpe", 0)

    if chall_corr > champ_corr and chall_sharpe >= champ_sharpe * 0.95:
        return {
            "decision": "PROMOTE",
            "reason": (
                f"Challenger improves correlation ({chall_corr:.5f} > {champ_corr:.5f}) "
                f"with acceptable Sharpe ({chall_sharpe:.4f} vs {champ_sharpe:.4f})."
            ),
            "champion_metrics": champion,
            "challenger_metrics": challenger_metrics,
        }

    return {
        "decision": "KEEP",
        "reason": (
            f"Champion is better or equal. "
            f"corr: {champ_corr:.5f} vs {chall_corr:.5f}, "
            f"sharpe: {champ_sharpe:.4f} vs {chall_sharpe:.4f}."
        ),
        "champion_metrics": champion,
        "challenger_metrics": challenger_metrics,
    }


# ---------------------------------------------------------------------------
# Last-known-good fallback
# ---------------------------------------------------------------------------

def save_last_known_good(artifact_path: str) -> None:
    """Save model as last-known-good after successful submission."""
    import shutil
    shutil.copy2(artifact_path, str(last_known_good_path()))
    logger.info("Saved last-known-good model from %s", artifact_path)


def has_last_known_good() -> bool:
    return last_known_good_path().exists()


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def build_submission_dataframe(path: str | None = None) -> pd.DataFrame:
    artifact_path = Path(path) if path else model_path()
    if not artifact_path.exists():
        raise NumeraiOpsError(f"Missing model artifact: {artifact_path}")

    features = load_features(settings.FEATURE_SET)
    live_path = data_dir() / "live.parquet"
    if not live_path.exists():
        raise NumeraiOpsError(f"Missing live data: {live_path}")

    with artifact_path.open("rb") as f:
        predict_fn = cloudpickle.load(f)

    live_df = pd.read_parquet(live_path, columns=["era"] + features)
    preds = predict_fn(live_df[["era"] + features], None)
    if "prediction" not in preds.columns:
        first_col = preds.columns[0]
        preds = preds.rename(columns={first_col: "prediction"})
    raw_predictions = pd.to_numeric(preds["prediction"], errors="coerce")
    try:
        validate_raw_prediction_series(
            raw_predictions,
            min_unique_predictions=settings.SUBMISSION_MIN_UNIQUE_PREDICTIONS,
        )
    except SubmissionGuardError as exc:
        raise NumeraiOpsError(str(exc)) from exc

    submission = pd.DataFrame(index=live_df.index)
    submission.index.name = live_df.index.name or "id"
    submission["prediction"] = raw_predictions.to_numpy()
    submission["prediction"] = submission["prediction"].rank(method="first", pct=True)

    submission = submission.reset_index()
    if submission.columns[0] != "id":
        submission = submission.rename(columns={submission.columns[0]: "id"})

    if "id" not in submission.columns:
        raise NumeraiOpsError("Submission dataframe must include an id column.")

    return submission[["id", "prediction"]]


def build_ensemble_submission() -> pd.DataFrame:
    """
    Build ensemble submission from all trained models in the suite.

    Loads each model, generates live predictions, and combines using
    era-wise ranking ensemble.
    """
    features_all = set()
    for cfg in MODEL_SUITE:
        features_all.update(load_features(cfg.feature_set))
    features_all_list = sorted(features_all)

    live_path = data_dir() / "live.parquet"
    if not live_path.exists():
        raise NumeraiOpsError(f"Missing live data: {live_path}")

    live_df = pd.read_parquet(live_path, columns=["era"] + features_all_list)
    for col in features_all_list:
        live_df[col] = live_df[col].astype(np.float32)

    # Compute engineered features for models that need them
    needs_engineered = any(cfg.use_engineered_features for cfg in MODEL_SUITE)
    engineered_cols: list[str] = []
    if needs_engineered:
        engineered_cols = compute_group_features(live_df, data_dir())

    predictions: Dict[str, pd.Series] = {}

    for cfg in MODEL_SUITE:
        pkl_path = model_path(cfg.name)
        if not pkl_path.exists():
            logger.warning("Skipping %s: model file not found at %s", cfg.name, pkl_path)
            continue

        try:
            with pkl_path.open("rb") as f:
                predict_fn = cloudpickle.load(f)

            features = load_features(cfg.feature_set)
            if cfg.use_engineered_features and engineered_cols:
                features = features + engineered_cols

            preds = predict_fn(live_df[["era"] + features], None)
            if "prediction" not in preds.columns:
                preds = preds.rename(columns={preds.columns[0]: "prediction"})

            predictions[cfg.name] = preds["prediction"]

            del predict_fn
            gc.collect()

        except Exception as exc:
            logger.error("Failed to predict with %s: %s", cfg.name, exc)

    if not predictions:
        raise NumeraiOpsError("No models produced predictions for ensemble.")

    valid_predictions = {
        name: values
        for name, values in predictions.items()
        if int(pd.to_numeric(values, errors="coerce").nunique())
        >= min(settings.SUBMISSION_MIN_UNIQUE_PREDICTIONS, len(values))
    }
    rejected = sorted(set(predictions) - set(valid_predictions))
    if rejected:
        logger.warning("Rejected constant/low-diversity live predictions from: %s", ", ".join(rejected))
    predictions = valid_predictions
    if not predictions:
        raise NumeraiOpsError("All ensemble members failed raw prediction diversity validation.")

    if len(predictions) == 1:
        ensemble = list(predictions.values())[0]
    else:
        ensemble = era_wise_rank_ensemble(predictions, live_df["era"])

    submission = pd.DataFrame(index=live_df.index)
    submission.index.name = live_df.index.name or "id"
    submission["prediction"] = ensemble.values
    submission["prediction"] = submission["prediction"].rank(method="first", pct=True)

    submission = submission.reset_index()
    if submission.columns[0] != "id":
        submission = submission.rename(columns={submission.columns[0]: "id"})

    return submission[["id", "prediction"]]


def resolve_target_models(napi: NumerAPI) -> Dict[str, str]:
    models = napi.get_models()
    if not models:
        raise NumeraiOpsError("No models were found in your Numerai account.")

    configured = [
        name.strip()
        for name in (settings.NUMERAI_MODEL_NAMES or "").split(",")
        if name.strip()
    ]
    if not configured:
        return dict(models)

    by_casefold = {name.casefold(): name for name in models}
    missing = [name for name in configured if name.casefold() not in by_casefold]
    if missing:
        available = ", ".join(sorted(models.keys()))
        missing_txt = ", ".join(missing)
        raise NumeraiOpsError(
            f"Configured model names not found: {missing_txt}. Available: {available}"
        )

    return {
        by_casefold[name.casefold()]: models[by_casefold[name.casefold()]]
        for name in configured
    }


def submit_to_models(
    napi: NumerAPI, submission_df: pd.DataFrame, model_map: Dict[str, str]
) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}

    for model_name, model_id in model_map.items():
        try:
            submission_id = napi.upload_predictions(
                df=submission_df,
                model_id=model_id,
                timeout=600,
            )
            results[model_name] = {
                "status": "submitted",
                "model_id": model_id,
                "submission_id": str(submission_id),
            }
        except Exception as exc:
            results[model_name] = {
                "status": "error",
                "model_id": model_id,
                "error": str(exc),
            }

    return results
