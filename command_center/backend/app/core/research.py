from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd

from .config import settings
from .numerai_ops import data_dir, load_features, predictor_features


class ResearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class RobustnessPolicy:
    minimum_mean_correlation: float = 0.01
    minimum_sharpe: float = 0.3
    maximum_feature_exposure: float = 0.1
    maximum_drawdown: float = 0.25
    minimum_recent_50_correlation: float = 0.0
    minimum_recent_100_correlation: float = 0.0
    minimum_bootstrap_correlation: float = 0.0


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    train_eras: list[str]
    embargo_eras: list[str]
    validation_eras: list[str]


def _era_sort_key(era: Any) -> tuple[int, str]:
    text = str(era)
    match = re.search(r"(\d+)$", text)
    return (int(match.group(1)) if match else -1, text)


def ordered_eras(eras: pd.Series | list[Any]) -> list[str]:
    values = {str(value) for value in eras if pd.notna(value)}
    return sorted(values, key=_era_sort_key)


def build_walk_forward_folds(
    eras: pd.Series | list[Any],
    *,
    minimum_train_eras: int = 200,
    validation_eras: int = 50,
    embargo_eras: int = 4,
    lockbox_eras: int = 50,
) -> dict[str, Any]:
    ordered = ordered_eras(eras)
    required = minimum_train_eras + embargo_eras + validation_eras + lockbox_eras
    if len(ordered) < required:
        raise ResearchError(
            f"Insufficient eras for walk-forward validation: {len(ordered)} < {required}."
        )
    lockbox = ordered[-lockbox_eras:]
    development = ordered[:-lockbox_eras]
    folds: list[WalkForwardFold] = []
    validation_start = minimum_train_eras + embargo_eras
    while validation_start + validation_eras <= len(development):
        train_end = validation_start - embargo_eras
        fold = WalkForwardFold(
            fold_id=f"fold_{len(folds) + 1}",
            train_eras=development[:train_end],
            embargo_eras=development[train_end:validation_start],
            validation_eras=development[
                validation_start : validation_start + validation_eras
            ],
        )
        folds.append(fold)
        validation_start += validation_eras
    if not folds:
        raise ResearchError("No complete walk-forward folds could be created.")
    return {
        "ordered_era_count": len(ordered),
        "development_eras": development,
        "lockbox_eras": lockbox,
        "folds": [asdict(fold) for fold in folds],
        "policy": {
            "minimum_train_eras": minimum_train_eras,
            "validation_eras": validation_eras,
            "embargo_eras": embargo_eras,
            "lockbox_eras": lockbox_eras,
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def per_era_correlation(
    targets: pd.Series,
    predictions: pd.Series,
    eras: pd.Series,
) -> pd.Series:
    frame = pd.DataFrame(
        {"target": targets, "prediction": predictions, "era": eras}
    ).dropna()
    if frame.empty:
        raise ResearchError("No resolved validation rows are available.")

    def score(group: pd.DataFrame) -> float:
        if group["target"].nunique() < 2 or group["prediction"].nunique() < 2:
            return float("nan")
        return float(group["target"].corr(group["prediction"]))

    scores = (
        frame.groupby("era", sort=False)[["target", "prediction"]]
        .apply(score)
        .dropna()
    )
    if scores.empty:
        raise ResearchError("No valid per-era correlations could be calculated.")
    return scores


def summarize_era_scores(scores: pd.Series) -> dict[str, Any]:
    if scores.empty:
        raise ResearchError("Cannot summarize empty era scores.")
    cumulative = scores.cumsum()
    drawdown = cumulative.cummax() - cumulative
    std = float(scores.std())
    return {
        "era_count": int(len(scores)),
        "mean_correlation": float(scores.mean()),
        "correlation_std": std,
        "sharpe": float(scores.mean() / std) if std > 0 else 0.0,
        "max_drawdown": float(drawdown.max()),
        "hit_rate": float((scores > 0).mean()),
        "fifth_percentile": float(scores.quantile(0.05)),
        "worst_era_correlation": float(scores.min()),
        "best_era_correlation": float(scores.max()),
    }


def bootstrap_mean_interval(
    scores: pd.Series,
    *,
    samples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    values = scores.to_numpy(dtype=float)
    if len(values) < 2:
        raise ResearchError("At least two eras are required for bootstrap analysis.")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "confidence": confidence,
        "lower": float(np.quantile(means, alpha)),
        "median": float(np.quantile(means, 0.5)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
    }


def build_robustness_packet(
    scores: pd.Series,
    *,
    feature_exposure: float,
    artifact_path: str | Path,
    model_name: str,
) -> dict[str, Any]:
    ordered = scores.reset_index(drop=True)
    recent_25 = ordered.iloc[-min(25, len(ordered)) :]
    recent_50 = ordered.iloc[-min(50, len(ordered)) :]
    recent_100 = ordered.iloc[-min(100, len(ordered)) :]
    quartiles = [
        pd.Series(segment)
        for segment in np.array_split(ordered.to_numpy(), 4)
    ]
    artifact_path = Path(artifact_path).resolve()
    packet = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "artifact_path": str(artifact_path),
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "feature_exposure": float(feature_exposure),
        "overall": summarize_era_scores(ordered),
        "recent": {
            "25": summarize_era_scores(recent_25),
            "50": summarize_era_scores(recent_50),
            "100": summarize_era_scores(recent_100),
        },
        "regimes": {
            f"quartile_{index + 1}": summarize_era_scores(segment)
            for index, segment in enumerate(quartiles)
            if not segment.empty
        },
        "bootstrap_mean_correlation": bootstrap_mean_interval(ordered),
        "era_scores": [float(value) for value in ordered],
    }
    canonical = json.dumps(packet, sort_keys=True, default=str)
    packet["packet_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return packet


def evaluate_artifact_robustness(
    artifact_path: str | Path,
    *,
    model_name: str,
    feature_set: str | None = None,
) -> dict[str, Any]:
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise ResearchError(f"Model artifact is missing: {artifact_path}")
    with artifact_path.open("rb") as handle:
        predict = cloudpickle.load(handle)
    features = predictor_features(
        predict,
        fallback_feature_set=feature_set or settings.FEATURE_SET,
    )
    validation_path = data_dir() / "validation.parquet"
    validation = pd.read_parquet(
        validation_path,
        columns=["era", "target"] + features,
    )
    validation = validation[validation["target"].notna()].copy()
    predictions = predict(validation[["era"] + features], None)["prediction"]
    if predictions.isna().any() or predictions.nunique() < 100:
        raise ResearchError("Model predictions are null or insufficiently diverse.")
    scores = per_era_correlation(
        validation["target"],
        predictions,
        validation["era"],
    )
    exposure = float(
        validation[features].corrwith(predictions).abs().max()
    )
    return build_robustness_packet(
        scores,
        feature_exposure=exposure,
        artifact_path=artifact_path,
        model_name=model_name,
    )


def promotion_recommendation(
    candidate: dict[str, Any],
    *,
    champion: dict[str, Any] | None,
    policy: RobustnessPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or RobustnessPolicy()
    overall = candidate["overall"]
    recent_100 = candidate["recent"]["100"]
    bootstrap = candidate["bootstrap_mean_correlation"]
    failures: list[str] = []
    checks = {
        "mean_correlation": overall["mean_correlation"] >= policy.minimum_mean_correlation,
        "sharpe": overall["sharpe"] >= policy.minimum_sharpe,
        "feature_exposure": candidate["feature_exposure"] <= policy.maximum_feature_exposure,
        "max_drawdown": overall["max_drawdown"] <= policy.maximum_drawdown,
        "recent_50": candidate["recent"]["50"]["mean_correlation"]
        >= policy.minimum_recent_50_correlation,
        "recent_100": recent_100["mean_correlation"] >= policy.minimum_recent_100_correlation,
        "bootstrap_lower": bootstrap["lower"] >= policy.minimum_bootstrap_correlation,
    }
    failures.extend(name for name, passed in checks.items() if not passed)

    paired_interval = None
    if champion is not None:
        candidate_scores = np.asarray(candidate["era_scores"], dtype=float)
        champion_scores = np.asarray(champion["era_scores"], dtype=float)
        common = min(len(candidate_scores), len(champion_scores))
        differences = pd.Series(candidate_scores[-common:] - champion_scores[-common:])
        paired_interval = bootstrap_mean_interval(differences)
        if overall["mean_correlation"] <= champion["overall"]["mean_correlation"]:
            failures.append("does_not_improve_champion_mean")
        if candidate["feature_exposure"] > champion["feature_exposure"] * 1.05:
            failures.append("feature_exposure_regression")
        if paired_interval["lower"] < -0.001:
            failures.append("paired_bootstrap_regression_risk")

    decision = "PROMOTE" if not failures else "REJECT"
    return {
        "decision": decision,
        "checks": checks,
        "failures": failures,
        "paired_difference_interval": paired_interval,
        "candidate_packet_sha256": candidate["packet_sha256"],
        "champion_packet_sha256": champion.get("packet_sha256") if champion else None,
        "policy": asdict(policy),
    }


def write_evaluation_artifacts(
    output_dir: str | Path,
    *,
    packet: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    model_name = packet["model_name"]
    packet_path = output_dir / f"{model_name}_evaluation_packet.json"
    recommendation_path = output_dir / f"{model_name}_promotion_recommendation.json"
    _atomic_json(packet_path, packet)
    _atomic_json(recommendation_path, recommendation)
    return {
        "evaluation_packet": str(packet_path),
        "promotion_recommendation": str(recommendation_path),
    }
