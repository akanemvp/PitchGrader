"""
Location+ model — predicts raw xRV from plate location + count only.

Features (4):
  plate_x    horizontal plate location
  plate_z    vertical plate location
  balls      pre-pitch ball count
  strikes    pre-pitch strike count

Residual = raw delta_run_exp - location_prediction = pure stuff signal
(location- and count-neutral). NOT loaded at inference time.
"""

import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import MODEL_DIR

logger = logging.getLogger(__name__)

LOCATION_FEATURES = [
    "plate_x",
    "plate_z",
    "balls",
    "strikes",
]

_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["balls"]   = pd.to_numeric(df.get("balls",   0), errors="coerce").fillna(0).astype(int).clip(0, 3)
    df["strikes"] = pd.to_numeric(df.get("strikes", 0), errors="coerce").fillna(0).astype(int).clip(0, 2)
    return df


def train_location_plus(df: pd.DataFrame, target_col: str = "delta_run_exp") -> lgb.LGBMRegressor:
    """Train Location+ on target_col ~ location + count (all pitches)."""
    df = add_location_features(df)
    feats = [f for f in LOCATION_FEATURES if f in df.columns]
    valid = df[feats + [target_col]].notna().all(axis=1)
    sub = df[valid]
    logger.info(f"  Location+: n={len(sub):,}  features={feats}  target={target_col}")
    model = lgb.LGBMRegressor(**_PARAMS)
    model.fit(sub[feats], sub[target_col])
    importances = sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1])
    logger.info("  Feature importances:")
    for feat, imp in importances:
        logger.info(f"    {feat:25s}  {imp}")
    return model


def compute_residuals(df: pd.DataFrame, model: lgb.LGBMRegressor, target_col: str = "delta_run_exp") -> pd.DataFrame:
    """Add residual_xrv = target_col - location_prediction."""
    df = add_location_features(df)
    feats = [f for f in LOCATION_FEATURES if f in df.columns]
    valid = df[feats].notna().all(axis=1) & df[target_col].notna()
    df["location_rv"]  = np.nan
    df["residual_xrv"] = np.nan
    df.loc[valid, "location_rv"]  = model.predict(df.loc[valid, feats])
    df.loc[valid, "residual_xrv"] = (
        df.loc[valid, target_col] - df.loc[valid, "location_rv"]
    )
    logger.info(
        f"  Residuals: mean={df['residual_xrv'].dropna().mean():.5f}  "
        f"std={df['residual_xrv'].dropna().std():.5f}  "
        f"non-null={valid.sum():,}"
    )
    return df


def save_location_plus(model: lgb.LGBMRegressor) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "location_plus.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Location+ saved → {path}")
