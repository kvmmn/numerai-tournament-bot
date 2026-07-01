"""Ensemble: era-wise ranking ensemble (proven competitive technique)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def era_wise_rank_ensemble(
    predictions: Dict[str, pd.Series],
    eras: pd.Series,
) -> pd.Series:
    """
    Build an ensemble using era-wise ranking.

    For each era:
      1. Rank each model's predictions within that era
      2. Average the ranks across models
      3. MinMaxScale the result to [0, 1]

    This prevents dominant models from overwhelming the ensemble and
    respects the per-era structure of Numerai scoring.

    Parameters
    ----------
    predictions : dict[str, pd.Series]
        Mapping of model_name -> raw predictions. All must share the same index.
    eras : pd.Series
        Era labels aligned with predictions index.

    Returns
    -------
    pd.Series
        Ensemble predictions in [0, 1] range.
    """
    if not predictions:
        raise ValueError("No predictions provided for ensemble.")

    # Combine into a DataFrame for vectorized operations
    pred_df = pd.DataFrame(predictions)
    pred_df["era"] = eras.values

    # Rank within each era for each model
    ranked = pred_df.groupby("era", sort=False)[list(predictions.keys())].rank(pct=True)

    # Average ranks across models
    ensemble = ranked.mean(axis=1)

    # MinMaxScale to [0, 1] globally
    e_min = ensemble.min()
    e_max = ensemble.max()
    if e_max > e_min:
        ensemble = (ensemble - e_min) / (e_max - e_min)
    else:
        ensemble[:] = 0.5

    return pd.Series(ensemble.values, index=pred_df.index, name="prediction")
