"""Feature engineering: group aggregations and prediction neutralization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import settings


# The 8 canonical Numerai feature groups
FEATURE_GROUPS = [
    "intelligence",
    "wisdom",
    "charisma",
    "dexterity",
    "strength",
    "constitution",
    "agility",
    "serenity",
]


def _load_feature_groups(data_dir: Path) -> Dict[str, List[str]]:
    """Load feature group memberships from features.json."""
    features_path = data_dir / "features.json"
    with features_path.open("r") as f:
        metadata = json.load(f)
    feature_sets = metadata.get("feature_sets", {})
    groups = {}
    for name in FEATURE_GROUPS:
        if name in feature_sets:
            groups[name] = feature_sets[name]
    return groups


def compute_group_features(df: pd.DataFrame, data_dir: Path | None = None) -> pd.DataFrame:
    """
    Add statistical aggregations per feature group.

    For each of the 8 groups, computes: mean, std, min, max, range.
    That gives 8 x 5 = 40 new features (very lightweight).

    Operates in-place on the dataframe for memory efficiency.
    Returns the list of new column names.
    """
    if data_dir is None:
        from .numerai_ops import data_dir as _data_dir
        data_dir = _data_dir()

    groups = _load_feature_groups(data_dir)
    new_cols: list[str] = []

    for group_name, group_features in groups.items():
        # Only use features that are actually present in the dataframe
        available = [f for f in group_features if f in df.columns]
        if not available:
            continue

        group_data = df[available]
        prefix = f"group_{group_name}"

        df[f"{prefix}_mean"] = group_data.mean(axis=1).astype(np.float32)
        df[f"{prefix}_std"] = group_data.std(axis=1).astype(np.float32)
        df[f"{prefix}_min"] = group_data.min(axis=1).astype(np.float32)
        df[f"{prefix}_max"] = group_data.max(axis=1).astype(np.float32)
        df[f"{prefix}_range"] = (df[f"{prefix}_max"] - df[f"{prefix}_min"]).astype(np.float32)

        new_cols.extend([
            f"{prefix}_mean",
            f"{prefix}_std",
            f"{prefix}_min",
            f"{prefix}_max",
            f"{prefix}_range",
        ])

    return new_cols


def get_engineered_feature_names(data_dir: Path | None = None) -> list[str]:
    """Return the list of engineered feature column names (without computing them)."""
    if data_dir is None:
        from .numerai_ops import data_dir as _data_dir
        data_dir = _data_dir()

    groups = _load_feature_groups(data_dir)
    names = []
    for group_name in groups:
        prefix = f"group_{group_name}"
        names.extend([
            f"{prefix}_mean",
            f"{prefix}_std",
            f"{prefix}_min",
            f"{prefix}_max",
            f"{prefix}_range",
        ])
    return names


def neutralize_predictions(
    predictions: pd.Series,
    features: pd.DataFrame,
    proportion: float = 1.0,
) -> pd.Series:
    """
    Neutralize predictions against a set of features using linear regression.

    This reduces feature exposure and improves MMC (Meta Model Contribution).
    FNCv3 neutralizes against the "medium" feature subset.

    Parameters
    ----------
    predictions : pd.Series
        Raw model predictions.
    features : pd.DataFrame
        Feature matrix to neutralize against.
    proportion : float
        How much to neutralize (0.0 = no change, 1.0 = full neutralization).

    Returns
    -------
    pd.Series
        Neutralized predictions.
    """
    if proportion == 0.0:
        return predictions

    preds = predictions.values.reshape(-1, 1)
    feat = features.values

    # Add intercept
    feat_with_intercept = np.column_stack([feat, np.ones(len(feat))])

    # Solve least squares: predictions = features @ beta + residual
    beta, _, _, _ = np.linalg.lstsq(feat_with_intercept, preds, rcond=None)

    # Residual is the neutralized prediction
    projection = feat_with_intercept @ beta
    neutralized = preds - proportion * projection

    result = pd.Series(neutralized.ravel(), index=predictions.index)
    return result


def neutralize_predictions_by_era(
    predictions: pd.Series,
    features: pd.DataFrame,
    eras: pd.Series,
    proportion: float,
) -> pd.Series:
    if proportion == 0.0:
        return predictions
    result = pd.Series(index=predictions.index, dtype=float)
    for _, indices in eras.groupby(eras, sort=False).groups.items():
        result.loc[indices] = neutralize_predictions(
            predictions.loc[indices],
            features.loc[indices],
            proportion=proportion,
        )
    return result
