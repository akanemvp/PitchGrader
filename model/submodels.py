"""
Shared model I/O and run-value baselines.

This module holds the pieces the production pipeline needs around the single
Driveline run-value regressor (model/prob_resid.py):

  • compute_rv_baselines      — count-adjusted per-outcome run values (whiff/ball/
                                cs/foul) and the league contact anchors, computed
                                once from training data (used by prob_resid).
  • save_ensemble/load_ensemble       — pickle the trained "all" model.
  • save_rv_baselines/load_rv_baselines — pickle the RV baselines dict.

Count-adjusted RV means ca_rv = delta_run_exp − mean(delta_run_exp | balls,
strikes): centering each count at 0 naturally yields rv_whiff≈−0.11,
rv_ball≈+0.06, rv_cs≈−0.06, with fouls held at exactly 0.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR

logger = logging.getLogger(__name__)

XWOBA_COL  = "estimated_woba_using_speedangle"
WOBA_SCALE = 1.15

# Statcast pitch-description buckets used to derive per-pitch outcome flags.
SWING_DESCS   = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
    "foul", "foul_tip", "bunt_foul_tip", "foul_bunt", "hit_into_play",
}
WHIFF_DESCS   = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOUL_DESCS    = {"foul", "foul_tip", "bunt_foul_tip", "foul_bunt"}
CONTACT_DESCS = {"hit_into_play"}
BALL_OUTCOME_DESCS = {"ball", "blocked_ball", "pitchout", "intent_ball", "hit_by_pitch"}

MIN_SAMPLES = 200

ENSEMBLE_KEY = "all"


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    desc = df["description"].fillna("")
    df = df.copy()
    df["is_swing"]   = desc.isin(SWING_DESCS).astype(int)
    df["is_whiff"]   = desc.isin(WHIFF_DESCS).astype(int)
    df["is_foul"]    = desc.isin(FOUL_DESCS).astype(int)
    df["is_contact"] = desc.isin(CONTACT_DESCS).astype(int)
    df["is_take"]    = (~desc.isin(SWING_DESCS)).astype(int)
    df["is_cs"]      = (desc == "called_strike").astype(int)
    df["is_ball"]    = desc.isin(BALL_OUTCOME_DESCS).astype(int)
    return df


def compute_rv_baselines(df: pd.DataFrame) -> dict:
    """Compute count-adjusted RV baselines (Fangraphs approach).

    Count-adjusted RV: ca_rv = delta_run_exp - mean(delta_run_exp | balls, strikes).
    This centers the average outcome at each count to 0, which naturally produces:
      rv_whiff < 0  (whiffs are much better than the average outcome)
      rv_ball  > 0  (balls are worse than the average outcome)
      rv_foul  = 0.0     (fouls are neutral — explicitly set per Fangraphs)
    plus the league contact anchors (lg_xwoba_con, mean_contact_rv) used to turn
    in-play xwOBA into runs.
    """
    df = _add_flags(df)
    baselines = {}

    # Count-adjusted RV: subtract per-count mean from delta_run_exp
    balls_col   = pd.to_numeric(df.get("balls",   0), errors="coerce").fillna(0).astype(int)
    strikes_col = pd.to_numeric(df.get("strikes", 0), errors="coerce").fillna(0).astype(int)
    dre = pd.to_numeric(df.get("delta_run_exp", np.nan), errors="coerce")
    tmp = pd.DataFrame({"balls": balls_col, "strikes": strikes_col, "dre": dre})
    count_mean = tmp.groupby(["balls", "strikes"])["dre"].transform("mean")
    ca_rv = dre - count_mean

    def _ca_mean(mask: pd.Series) -> float:
        vals = ca_rv[mask & dre.notna() & ca_rv.notna()]
        return float(vals.mean()) if len(vals) > 0 else 0.0

    baselines["rv_whiff"] = _ca_mean(df["is_whiff"] == 1)
    baselines["rv_ball"]  = _ca_mean(df["is_ball"]  == 1)
    baselines["rv_cs"]    = _ca_mean(df["is_cs"]    == 1)
    baselines["rv_foul"]  = 0.0  # Fangraphs: fouls are explicitly neutral

    logger.info(
        f"  rv_baseline rv_whiff={baselines['rv_whiff']:.5f}  "
        f"rv_ball={baselines['rv_ball']:.5f}  "
        f"rv_cs={baselines['rv_cs']:.5f}  rv_foul=0.0  (count-adjusted)"
    )

    contact_rows = df[df["is_contact"] == 1].copy()
    contact_ca   = ca_rv[df["is_contact"] == 1]
    if XWOBA_COL in contact_rows.columns and contact_rows[XWOBA_COL].notna().sum() > MIN_SAMPLES:
        lg_xwoba_con    = float(contact_rows[XWOBA_COL].dropna().mean())
        mean_contact_rv = float(contact_ca.dropna().mean()) if contact_ca.notna().any() else 0.0
    else:
        lg_xwoba_con    = 0.370
        mean_contact_rv = float(contact_ca.dropna().mean()) if len(contact_rows) else 0.0

    baselines["lg_xwoba_con"]    = lg_xwoba_con
    baselines["mean_contact_rv"] = mean_contact_rv
    baselines["woba_scale"]      = WOBA_SCALE
    baselines["rv_contact"]      = mean_contact_rv
    logger.info(
        f"  rv_baseline contact: lg_xwoba_con={lg_xwoba_con:.4f}  "
        f"mean_contact_rv={mean_contact_rv:.5f}  woba_scale={WOBA_SCALE}"
    )
    return baselines


def save_ensemble(ensemble: dict, family: str = ENSEMBLE_KEY) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"ensemble_{family}.pkl")
    with open(path, "wb") as f:
        pickle.dump(ensemble, f)


def load_ensemble(family: str = ENSEMBLE_KEY) -> "dict | None":
    path = os.path.join(MODEL_DIR, f"ensemble_{family}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_rv_baselines(baselines: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "rv_baselines.pkl")
    with open(path, "wb") as f:
        pickle.dump(baselines, f)


def load_rv_baselines() -> "dict | None":
    path = os.path.join(MODEL_DIR, "rv_baselines.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
