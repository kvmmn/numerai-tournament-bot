from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd

from .config import settings
from .feature_engineering import neutralize_predictions_by_era
from .numerai_ops import (
    _predict_wrapper,
    _prepare_training_frame,
    data_dir,
    load_features,
)
from .research import (
    build_robustness_packet,
    ordered_eras,
    per_era_correlation,
    promotion_recommendation,
    summarize_era_scores,
)


class OptimizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SweepConfig:
    seeds: tuple[int, ...] = (42, 123, 777)
    neutralization_proportions: tuple[float, ...] = (0.65, 0.75, 0.85)
    lockbox_eras: int = 50
    n_estimators: int = 2000
    learning_rate: float = 0.01
    max_depth: int = 5
    num_leaves: int = 31
    colsample_bytree: float = 0.1


@dataclass(frozen=True)
class TargetEnsembleConfig:
    target_groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "short_horizon",
            (
                "target",
                "target_alpha_20",
                "target_jeremy_20",
                "target_teager2b_20",
                "target_victor_20",
            ),
        ),
        (
            "mixed_horizon",
            (
                "target",
                "target_alpha_20",
                "target_alpha_60",
                "target_jeremy_20",
                "target_jeremy_60",
            ),
        ),
        (
            "broad_horizon",
            (
                "target",
                "target_alpha_20",
                "target_alpha_60",
                "target_jeremy_20",
                "target_jeremy_60",
                "target_teager2b_20",
                "target_teager2b_60",
            ),
        ),
    )
    neutralization_proportions: tuple[float, ...] = (0.65, 0.75, 0.85)
    seed: int = 123
    lockbox_eras: int = 50
    n_estimators: int = 2000


class TargetEnsemblePredictor:
    def __init__(
        self,
        models: dict[str, Any],
        features: list[str],
        neutralization_proportion: float,
    ):
        self.models = models
        self.features = features
        self.neutralization_proportion = neutralization_proportion

    def __call__(
        self,
        live_features: pd.DataFrame,
        _live_benchmark_models: pd.DataFrame | None,
    ) -> pd.DataFrame:
        eras = (
            live_features["era"]
            if "era" in live_features
            else pd.Series("era", index=live_features.index)
        )
        matrix = pd.DataFrame(
            {
                name: model.predict(live_features[self.features])
                for name, model in self.models.items()
            },
            index=live_features.index,
        )
        ranked = matrix.groupby(eras, sort=False).rank(pct=True)
        ensemble = ranked.mean(axis=1)
        ensemble = neutralize_predictions_by_era(
            ensemble,
            live_features[self.features],
            eras,
            proportion=self.neutralization_proportion,
        )
        return pd.DataFrame({"prediction": ensemble}, index=live_features.index)


@dataclass(frozen=True)
class FeatureEnsembleConfig:
    feature_groups: tuple[str, ...] = (
        "small",
        "intelligence",
        "dexterity",
        "serenity",
    )
    combinations: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("all_families", ("small", "intelligence", "dexterity", "serenity")),
        ("specialists", ("intelligence", "dexterity", "serenity")),
        ("small_serenity", ("small", "serenity")),
    )
    neutralization_proportions: tuple[float, ...] = (0.65, 0.75, 0.85)
    seed: int = 123
    lockbox_eras: int = 50
    n_estimators: int = 2000


class FeatureFamilyEnsemblePredictor:
    def __init__(
        self,
        members: dict[str, tuple[Any, list[str]]],
        neutralization_features: list[str],
        neutralization_proportion: float,
    ):
        self.members = members
        self.neutralization_features = neutralization_features
        self.neutralization_proportion = neutralization_proportion

    def __call__(
        self,
        live_features: pd.DataFrame,
        _live_benchmark_models: pd.DataFrame | None,
    ) -> pd.DataFrame:
        eras = (
            live_features["era"]
            if "era" in live_features
            else pd.Series("era", index=live_features.index)
        )
        matrix = pd.DataFrame(
            {
                name: model.predict(live_features[features])
                for name, (model, features) in self.members.items()
            },
            index=live_features.index,
        )
        ensemble = matrix.groupby(eras, sort=False).rank(pct=True).mean(axis=1)
        ensemble = neutralize_predictions_by_era(
            ensemble,
            live_features[self.neutralization_features],
            eras,
            proportion=self.neutralization_proportion,
        )
        return pd.DataFrame({"prediction": ensemble}, index=live_features.index)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _objective(packet: dict[str, Any]) -> float:
    return float(
        packet["overall"]["mean_correlation"]
        + 0.5 * packet["recent"]["50"]["mean_correlation"]
        - 0.01 * packet["feature_exposure"]
        - 0.01 * packet["overall"]["max_drawdown"]
    )


def select_best_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise OptimizationError("No optimization results are available.")
    eligible = [
        result
        for result in results
        if result["development_recommendation"]["decision"] == "PROMOTE"
    ]
    pool = eligible or results
    return max(pool, key=lambda result: result["development_objective"])


def run_seed_neutralization_sweep(
    output_dir: str | Path,
    *,
    config: SweepConfig | None = None,
) -> dict[str, Any]:
    config = config or SweepConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = load_features(settings.FEATURE_SET)

    train = pd.read_parquet(
        data_dir() / "train.parquet",
        columns=["era", "target"] + features,
    )
    train, training_data_report = _prepare_training_frame(train, features)
    validation = pd.read_parquet(
        data_dir() / "validation.parquet",
        columns=["era", "target"] + features,
    )
    validation = validation[validation["target"].notna()].copy()
    for feature in features:
        validation[feature] = validation[feature].astype(np.float32)

    eras = ordered_eras(validation["era"])
    if len(eras) <= config.lockbox_eras + 100:
        raise OptimizationError("Insufficient validation eras for development and lockbox.")
    lockbox = set(eras[-config.lockbox_eras :])
    development_mask = ~validation["era"].astype(str).isin(lockbox)
    lockbox_mask = ~development_mask
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []

    import lightgbm as lgb

    for seed in config.seeds:
        model = lgb.LGBMRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            num_leaves=config.num_leaves,
            colsample_bytree=config.colsample_bytree,
            random_state=seed,
            verbosity=-1,
        )
        model.fit(train[features], train["target"])
        raw = pd.Series(
            model.predict(validation[features]),
            index=validation.index,
        )
        fully_neutralized = neutralize_predictions_by_era(
            raw,
            validation[features],
            validation["era"],
            proportion=1.0,
        )

        for proportion in config.neutralization_proportions:
            predictions = (1.0 - proportion) * raw + proportion * fully_neutralized
            candidate_name = f"lgbm_seed{seed}_neutral{int(proportion * 100):02d}"
            artifact_path = run_dir / f"{candidate_name}.pkl"
            with artifact_path.open("wb") as handle:
                cloudpickle.dump(
                    _predict_wrapper(
                        model,
                        features,
                        neutralization_proportion=proportion,
                    ),
                    handle,
                )

            development_scores = per_era_correlation(
                validation.loc[development_mask, "target"],
                predictions.loc[development_mask],
                validation.loc[development_mask, "era"],
            )
            development_exposure = float(
                validation.loc[development_mask, features]
                .corrwith(predictions.loc[development_mask])
                .abs()
                .max()
            )
            development_packet = build_robustness_packet(
                development_scores,
                feature_exposure=development_exposure,
                artifact_path=artifact_path,
                model_name=candidate_name,
            )
            development_recommendation = promotion_recommendation(
                development_packet,
                champion=None,
            )
            lockbox_scores = per_era_correlation(
                validation.loc[lockbox_mask, "target"],
                predictions.loc[lockbox_mask],
                validation.loc[lockbox_mask, "era"],
            )
            full_scores = per_era_correlation(
                validation["target"],
                predictions,
                validation["era"],
            )
            full_exposure = float(
                validation[features].corrwith(predictions).abs().max()
            )
            full_packet = build_robustness_packet(
                full_scores,
                feature_exposure=full_exposure,
                artifact_path=artifact_path,
                model_name=candidate_name,
            )
            result = {
                "candidate_name": candidate_name,
                "seed": seed,
                "neutralization_proportion": proportion,
                "artifact_path": str(artifact_path),
                "development_packet": development_packet,
                "development_recommendation": development_recommendation,
                "development_objective": _objective(development_packet),
                "lockbox": summarize_era_scores(lockbox_scores),
                "full_packet": full_packet,
            }
            results.append(result)
            _atomic_json(run_dir / f"{candidate_name}.json", result)

        del model, raw, fully_neutralized
        gc.collect()

    best = select_best_candidate(results)
    final_recommendation = promotion_recommendation(
        best["full_packet"],
        champion=None,
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "training_data_report": training_data_report,
        "development_era_count": len(eras) - config.lockbox_eras,
        "lockbox_eras": eras[-config.lockbox_eras :],
        "candidate_count": len(results),
        "best_candidate": best,
        "final_recommendation": final_recommendation,
        "all_candidates": [
            {
                "candidate_name": result["candidate_name"],
                "development_objective": result["development_objective"],
                "development_decision": result["development_recommendation"]["decision"],
                "development_failures": result["development_recommendation"]["failures"],
                "development_mean_correlation": result["development_packet"]["overall"][
                    "mean_correlation"
                ],
                "development_recent_50": result["development_packet"]["recent"]["50"][
                    "mean_correlation"
                ],
                "lockbox_mean_correlation": result["lockbox"]["mean_correlation"],
                "full_feature_exposure": result["full_packet"]["feature_exposure"],
            }
            for result in results
        ],
    }
    _atomic_json(run_dir / "summary.json", summary)
    _atomic_json(output_dir / "latest.json", summary)
    return summary


def run_target_ensemble_sweep(
    output_dir: str | Path,
    *,
    config: TargetEnsembleConfig | None = None,
) -> dict[str, Any]:
    config = config or TargetEnsembleConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = load_features(settings.FEATURE_SET)
    all_targets = sorted(
        {
            target
            for _, targets in config.target_groups
            for target in targets
        }
    )
    train = pd.read_parquet(
        data_dir() / "train.parquet",
        columns=["era"] + all_targets + features,
    )
    primary = train[["era", "target"] + features].copy()
    primary, training_data_report = _prepare_training_frame(primary, features)
    selected_eras = set(primary["era"].astype(str))
    train = train[train["era"].astype(str).isin(selected_eras)].copy()
    for feature in features:
        train[feature] = train[feature].astype(np.float32)

    validation = pd.read_parquet(
        data_dir() / "validation.parquet",
        columns=["era", "target"] + features,
    )
    validation = validation[validation["target"].notna()].copy()
    for feature in features:
        validation[feature] = validation[feature].astype(np.float32)

    import lightgbm as lgb

    models: dict[str, Any] = {}
    validation_predictions: dict[str, pd.Series] = {}
    for target in all_targets:
        target_rows = train[target].notna() & np.isfinite(train[target].astype(float))
        if int(target_rows.sum()) == 0:
            raise OptimizationError(f"Target has no resolved training rows: {target}")
        model = lgb.LGBMRegressor(
            n_estimators=config.n_estimators,
            learning_rate=0.01,
            max_depth=5,
            num_leaves=31,
            colsample_bytree=0.1,
            random_state=config.seed,
            verbosity=-1,
        )
        model.fit(train.loc[target_rows, features], train.loc[target_rows, target])
        models[target] = model
        validation_predictions[target] = pd.Series(
            model.predict(validation[features]),
            index=validation.index,
        )

    prediction_matrix = pd.DataFrame(validation_predictions)
    ranked_matrix = prediction_matrix.groupby(
        validation["era"],
        sort=False,
    ).rank(pct=True)
    eras = ordered_eras(validation["era"])
    lockbox = set(eras[-config.lockbox_eras :])
    development_mask = ~validation["era"].astype(str).isin(lockbox)
    lockbox_mask = ~development_mask
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for group_name, targets in config.target_groups:
        base_ensemble = ranked_matrix[list(targets)].mean(axis=1)
        fully_neutralized = neutralize_predictions_by_era(
            base_ensemble,
            validation[features],
            validation["era"],
            proportion=1.0,
        )
        group_models = {target: models[target] for target in targets}
        for proportion in config.neutralization_proportions:
            predictions = (
                (1.0 - proportion) * base_ensemble
                + proportion * fully_neutralized
            )
            candidate_name = f"target_{group_name}_neutral{int(proportion * 100):02d}"
            artifact_path = run_dir / f"{candidate_name}.pkl"
            with artifact_path.open("wb") as handle:
                cloudpickle.dump(
                    TargetEnsemblePredictor(
                        group_models,
                        features,
                        proportion,
                    ),
                    handle,
                )
            development_scores = per_era_correlation(
                validation.loc[development_mask, "target"],
                predictions.loc[development_mask],
                validation.loc[development_mask, "era"],
            )
            development_exposure = float(
                validation.loc[development_mask, features]
                .corrwith(predictions.loc[development_mask])
                .abs()
                .max()
            )
            development_packet = build_robustness_packet(
                development_scores,
                feature_exposure=development_exposure,
                artifact_path=artifact_path,
                model_name=candidate_name,
            )
            development_recommendation = promotion_recommendation(
                development_packet,
                champion=None,
            )
            lockbox_scores = per_era_correlation(
                validation.loc[lockbox_mask, "target"],
                predictions.loc[lockbox_mask],
                validation.loc[lockbox_mask, "era"],
            )
            full_scores = per_era_correlation(
                validation["target"],
                predictions,
                validation["era"],
            )
            full_exposure = float(
                validation[features].corrwith(predictions).abs().max()
            )
            full_packet = build_robustness_packet(
                full_scores,
                feature_exposure=full_exposure,
                artifact_path=artifact_path,
                model_name=candidate_name,
            )
            result = {
                "candidate_name": candidate_name,
                "targets": list(targets),
                "neutralization_proportion": proportion,
                "artifact_path": str(artifact_path),
                "development_packet": development_packet,
                "development_recommendation": development_recommendation,
                "development_objective": _objective(development_packet),
                "lockbox": summarize_era_scores(lockbox_scores),
                "full_packet": full_packet,
            }
            results.append(result)
            _atomic_json(run_dir / f"{candidate_name}.json", result)
        del base_ensemble, fully_neutralized
        gc.collect()

    best = select_best_candidate(results)
    final_recommendation = promotion_recommendation(
        best["full_packet"],
        champion=None,
    )
    summary = {
        "schema_version": 1,
        "experiment": "target_ensemble",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "training_data_report": training_data_report,
        "development_era_count": len(eras) - config.lockbox_eras,
        "lockbox_eras": eras[-config.lockbox_eras :],
        "candidate_count": len(results),
        "best_candidate": best,
        "final_recommendation": final_recommendation,
        "all_candidates": [
            {
                "candidate_name": result["candidate_name"],
                "development_objective": result["development_objective"],
                "development_decision": result["development_recommendation"]["decision"],
                "development_failures": result["development_recommendation"]["failures"],
                "development_mean_correlation": result["development_packet"]["overall"][
                    "mean_correlation"
                ],
                "development_recent_50": result["development_packet"]["recent"]["50"][
                    "mean_correlation"
                ],
                "lockbox_mean_correlation": result["lockbox"]["mean_correlation"],
                "full_mean_correlation": result["full_packet"]["overall"][
                    "mean_correlation"
                ],
                "full_recent_50": result["full_packet"]["recent"]["50"][
                    "mean_correlation"
                ],
                "full_feature_exposure": result["full_packet"]["feature_exposure"],
            }
            for result in results
        ],
    }
    _atomic_json(run_dir / "summary.json", summary)
    _atomic_json(output_dir / "latest.json", summary)
    return summary


def run_feature_family_sweep(
    output_dir: str | Path,
    *,
    config: FeatureEnsembleConfig | None = None,
) -> dict[str, Any]:
    config = config or FeatureEnsembleConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_by_group = {
        group: load_features(group)
        for group in config.feature_groups
    }
    all_features = sorted(
        {
            feature
            for features in features_by_group.values()
            for feature in features
        }
    )
    neutralization_features = features_by_group["small"]
    train = pd.read_parquet(
        data_dir() / "train.parquet",
        columns=["era", "target"] + all_features,
    )
    train, training_data_report = _prepare_training_frame(train, all_features)
    validation = pd.read_parquet(
        data_dir() / "validation.parquet",
        columns=["era", "target"] + all_features,
    )
    validation = validation[validation["target"].notna()].copy()
    for feature in all_features:
        validation[feature] = validation[feature].astype(np.float32)

    import lightgbm as lgb

    models: dict[str, tuple[Any, list[str]]] = {}
    predictions: dict[str, pd.Series] = {}
    for group, features in features_by_group.items():
        model = lgb.LGBMRegressor(
            n_estimators=config.n_estimators,
            learning_rate=0.01,
            max_depth=5,
            num_leaves=31,
            colsample_bytree=0.1,
            random_state=config.seed,
            verbosity=-1,
        )
        model.fit(train[features], train["target"])
        models[group] = (model, features)
        predictions[group] = pd.Series(
            model.predict(validation[features]),
            index=validation.index,
        )

    prediction_matrix = pd.DataFrame(predictions)
    ranked_matrix = prediction_matrix.groupby(
        validation["era"],
        sort=False,
    ).rank(pct=True)
    eras = ordered_eras(validation["era"])
    lockbox = set(eras[-config.lockbox_eras :])
    development_mask = ~validation["era"].astype(str).isin(lockbox)
    lockbox_mask = ~development_mask
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for combination_name, groups in config.combinations:
        base_ensemble = ranked_matrix[list(groups)].mean(axis=1)
        fully_neutralized = neutralize_predictions_by_era(
            base_ensemble,
            validation[neutralization_features],
            validation["era"],
            proportion=1.0,
        )
        members = {group: models[group] for group in groups}
        for proportion in config.neutralization_proportions:
            candidate_predictions = (
                (1.0 - proportion) * base_ensemble
                + proportion * fully_neutralized
            )
            candidate_name = (
                f"feature_{combination_name}_neutral{int(proportion * 100):02d}"
            )
            artifact_path = run_dir / f"{candidate_name}.pkl"
            with artifact_path.open("wb") as handle:
                cloudpickle.dump(
                    FeatureFamilyEnsemblePredictor(
                        members,
                        neutralization_features,
                        proportion,
                    ),
                    handle,
                )
            development_scores = per_era_correlation(
                validation.loc[development_mask, "target"],
                candidate_predictions.loc[development_mask],
                validation.loc[development_mask, "era"],
            )
            development_exposure = float(
                validation.loc[development_mask, neutralization_features]
                .corrwith(candidate_predictions.loc[development_mask])
                .abs()
                .max()
            )
            development_packet = build_robustness_packet(
                development_scores,
                feature_exposure=development_exposure,
                artifact_path=artifact_path,
                model_name=candidate_name,
            )
            development_recommendation = promotion_recommendation(
                development_packet,
                champion=None,
            )
            lockbox_scores = per_era_correlation(
                validation.loc[lockbox_mask, "target"],
                candidate_predictions.loc[lockbox_mask],
                validation.loc[lockbox_mask, "era"],
            )
            full_scores = per_era_correlation(
                validation["target"],
                candidate_predictions,
                validation["era"],
            )
            full_exposure = float(
                validation[neutralization_features]
                .corrwith(candidate_predictions)
                .abs()
                .max()
            )
            full_packet = build_robustness_packet(
                full_scores,
                feature_exposure=full_exposure,
                artifact_path=artifact_path,
                model_name=candidate_name,
            )
            result = {
                "candidate_name": candidate_name,
                "feature_groups": list(groups),
                "neutralization_proportion": proportion,
                "artifact_path": str(artifact_path),
                "development_packet": development_packet,
                "development_recommendation": development_recommendation,
                "development_objective": _objective(development_packet),
                "lockbox": summarize_era_scores(lockbox_scores),
                "full_packet": full_packet,
            }
            results.append(result)
            _atomic_json(run_dir / f"{candidate_name}.json", result)
        del base_ensemble, fully_neutralized
        gc.collect()

    best = select_best_candidate(results)
    final_recommendation = promotion_recommendation(
        best["full_packet"],
        champion=None,
    )
    summary = {
        "schema_version": 1,
        "experiment": "feature_family_ensemble",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "training_data_report": training_data_report,
        "development_era_count": len(eras) - config.lockbox_eras,
        "lockbox_eras": eras[-config.lockbox_eras :],
        "candidate_count": len(results),
        "best_candidate": best,
        "final_recommendation": final_recommendation,
        "all_candidates": [
            {
                "candidate_name": result["candidate_name"],
                "development_objective": result["development_objective"],
                "development_decision": result["development_recommendation"]["decision"],
                "development_failures": result["development_recommendation"]["failures"],
                "development_mean_correlation": result["development_packet"]["overall"][
                    "mean_correlation"
                ],
                "development_recent_50": result["development_packet"]["recent"]["50"][
                    "mean_correlation"
                ],
                "lockbox_mean_correlation": result["lockbox"]["mean_correlation"],
                "full_mean_correlation": result["full_packet"]["overall"][
                    "mean_correlation"
                ],
                "full_recent_50": result["full_packet"]["recent"]["50"][
                    "mean_correlation"
                ],
                "full_feature_exposure": result["full_packet"]["feature_exposure"],
            }
            for result in results
        ],
    }
    _atomic_json(run_dir / "summary.json", summary)
    _atomic_json(output_dir / "latest.json", summary)
    return summary
