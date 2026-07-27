"""
Stuff+ inference — loads the trained model and turns pitches into grades.

StuffPlusPredictor loads the three family probability models (fastball / offspeed /
breaking) plus the cutter router, then:
  1. engineers shape features for each pitch (unless already engineered),
  2. routes each pitch to its family (cutters via the Mahalanobis router),
  3. predicts each pitch's expected run value (xRV) with that family's model,
  4. normalizes on that family's OWN scale: 100 = family average, 10 = one SD.
Lower xRV = better pitch = higher grade. Because each family self-normalizes, a
fastball is graded against fastballs — no cross-family shafting.

The shape features are arm-normalized and carry no handedness, so a lefty and a righty
throwing physically identical pitches grade identically — one scoring pass per pitch.

Two grade columns are produced: `stuff_plus` (the raw z-score, used for aggregation/
leaderboards) and `stuff_plus_display` (a percentile-anchored soft-cap so individual
pitches top out ~135 instead of running to extreme outliers).
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR
from features.engineering import engineer_features
from model.submodels import load_ensemble
from model.prob_resid import assign_family, predict_family_rv, add_magnus, SHAPE_FEATS

logger = logging.getLogger(__name__)

_POWER_TYPES = {"FF", "FA", "SI", "FC", "ST"}
_FAM_KEY = {"FB": "fb", "OFF": "os", "BR": "br"}

_DG_KNEE, _DG_CEIL, _DG_SOFT = 125.0, 135.0, 12.0


def display_grade(sp):
    """Percentile-anchored display grade for individual pitches."""
    sp = np.asarray(sp, dtype=float)
    out = sp.copy()
    hi = sp > _DG_KNEE
    out[hi] = _DG_KNEE + (_DG_CEIL - _DG_KNEE) * (1.0 - np.exp(-(sp[hi] - _DG_KNEE) / _DG_SOFT))
    return out


class StuffPlusPredictor:
    def __init__(self):
        self.ensembles: dict = {}
        self.router = None
        self.baselines = None
        self._load()

    def _load(self):
        bpath = os.path.join(MODEL_DIR, "movement_baselines.pkl")
        if os.path.exists(bpath):
            with open(bpath, "rb") as f:
                self.baselines = pickle.load(f)

        self.ensembles = {}
        for key in ("fb", "os", "br"):
            e = load_ensemble(key)
            if e is not None:
                self.ensembles[key] = e
        if self.ensembles:
            logger.info(f"Family models loaded: {sorted(self.ensembles)}")
        else:
            logger.warning("No family models found — run 'python main.py train' first")

        rpath = os.path.join(MODEL_DIR, "cutter_router.pkl")
        if os.path.exists(rpath):
            with open(rpath, "rb") as f:
                self.router = pickle.load(f)

    def _fam_norm(self, ens, norm_set):
        n = ens.get("norm_hist") if (norm_set == "historical" and ens.get("norm_hist")) else ens["norm"]
        return n["mean"], max(n["std"], 1e-6)

    def predict(self, df, baselines=None, already_engineered=False, norm_set="current"):
        if not already_engineered:
            bl = baselines if baselines is not None else self.baselines
            df, _ = engineer_features(df, baselines=bl)

        df = df.copy()
        df["stuff_plus"] = np.nan
        if not self.ensembles:
            logger.warning("Model not loaded — returning NaN")
            return df
        if "ind_vert" not in df.columns:
            add_magnus(df)

        fam = assign_family(df, self.router)
        sp = np.full(len(df), np.nan)
        for family, key in _FAM_KEY.items():
            ens = self.ensembles.get(key)
            if ens is None:
                continue
            rows = (fam == family).values
            if not rows.any():
                continue
            rv = predict_family_rv(df.loc[rows], ens)
            gm, gs = self._fam_norm(ens, norm_set)
            sp[rows] = 100.0 + (gm - rv) / gs * 10.0

        # Velocity floor: a "power" pitch type below 75 mph is almost always a mislabel.
        low_velo = (pd.to_numeric(df["release_speed"], errors="coerce") < 75.0).values
        power_mask = df["pitch_type"].isin(_POWER_TYPES).values
        sp[power_mask & low_velo] = np.nan

        df["stuff_plus"] = np.clip(sp, -50.0, 200.0)
        df["stuff_plus_display"] = display_grade(df["stuff_plus"].values)
        return df


_COMPOSITE_PREDICTOR = None


def _composite_pitch_grade(df, norm_set: str = "current") -> dict:
    """Grade each pitch type's AVERAGE pitch — {pitch_type: Stuff+}.

    Averages the model's shape features within a pitch type, routes each averaged pitch
    to its family, and scores it on that family's scale. `df` must already be engineered
    (from a *_scored table). Cutter routing uses this pitcher's cutters (df is per-pitcher).
    """
    global _COMPOSITE_PREDICTOR
    if _COMPOSITE_PREDICTOR is None:
        _COMPOSITE_PREDICTOR = StuffPlusPredictor()
    p = _COMPOSITE_PREDICTOR
    if not p.ensembles or df is None or df.empty or "pitch_type" not in df.columns:
        logger.warning("composite grade skipped: no model or empty frame")
        return {}

    missing = [f for f in SHAPE_FEATS if f not in df.columns]
    if missing:
        logger.warning(f"composite grade skipped — missing model features: {missing}")
        return {}

    comp = df.groupby("pitch_type")[SHAPE_FEATS].mean().dropna()
    if comp.empty:
        return {}
    comp = comp.reset_index()
    if "player_name" in df.columns and df["player_name"].nunique() == 1:
        comp["player_name"] = df["player_name"].iloc[0]

    fam = assign_family(comp, p.router)
    out: dict = {}
    for family, key in _FAM_KEY.items():
        ens = p.ensembles.get(key)
        if ens is None:
            continue
        rows = (fam == family).values
        if not rows.any():
            continue
        sub = comp.loc[rows]
        rv = predict_family_rv(sub, ens)
        gm, gs = p._fam_norm(ens, norm_set)
        sp = np.clip(100.0 + (gm - rv) / gs * 10.0, -50.0, 200.0)
        for pt, v in zip(sub["pitch_type"].values, sp):
            if np.isfinite(v):
                out[pt] = float(v)
    return out


def load_baselines():
    path = os.path.join(MODEL_DIR, "movement_baselines.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No baselines at {path}. Run train first.")
    with open(path, "rb") as f:
        return pickle.load(f)
