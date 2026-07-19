"""
Stuff+ inference — loads the trained model and turns pitches into grades.

StuffPlusPredictor loads the saved model (a single LightGBM regressor on the
Driveline run-value target) plus the frozen normalization baseline, then:
  1. engineers shape features for each pitch (unless already engineered),
  2. predicts each pitch's expected run value (xRV),
  3. normalizes to the Stuff+ scale: 100 = league average, 10 = one standard
     deviation. Lower xRV = better pitch = higher grade.

Every pitch is scored twice — once as if facing a same-hand batter and once
opposite-hand — and averaged 50/50, so a pitcher's grade is independent of the
platoon mix they happened to face.

Two grade columns are produced: `stuff_plus` (the raw z-score, used for
aggregation/leaderboards) and `stuff_plus_display` (a percentile-anchored soft-cap
so individual pitches top out ~135 instead of running to extreme outliers).
Grades are globally comparable across pitch types — no per-type re-normalization.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR
from features.engineering import engineer_features
from model.submodels import load_ensemble
from model.prob_resid import predict_prob_resid_rv

logger = logging.getLogger(__name__)

_POWER_TYPES = {"FF", "FA", "SI", "FC", "ST"}

# Percentile-anchored display grade. The raw Stuff+ z-score is a fine scale for the
# bulk and for aggregation (pitcher / pitch-type means all sit well below the knee),
# but on individual pitches the xRV distribution is fat-tailed, so a linear z-score
# sends the rare elite pitch to a fictional 6 SD (clipped 160). This remaps single
# pitches: identity below the ~99th-pct knee, then a saturating soft-cap so the tail
# compresses into a realistic ceiling (~135) instead of running away. Aggregated
# grades are unchanged because they live below the knee.
_DG_KNEE, _DG_CEIL, _DG_SOFT = 125.0, 135.0, 12.0   # knee=99th pct, ceiling, softness


def display_grade(sp):
    """Percentile-anchored display grade for individual pitches (see note above)."""
    sp = np.asarray(sp, dtype=float)
    out = sp.copy()
    hi = sp > _DG_KNEE          # NaN-safe: NaN > knee is False, so NaNs pass through
    out[hi] = _DG_KNEE + (_DG_CEIL - _DG_KNEE) * (1.0 - np.exp(-(sp[hi] - _DG_KNEE) / _DG_SOFT))
    return out


class StuffPlusPredictor:
    def __init__(self):
        self.ensembles:              dict = {}
        self.norm_global:            dict = {}
        self.norm_global_historical: dict = {}
        self.baselines                    = None
        self._load()

    def _load(self):
        bpath = os.path.join(MODEL_DIR, "movement_baselines.pkl")
        if os.path.exists(bpath):
            with open(bpath, "rb") as f:
                self.baselines = pickle.load(f)

        gpath = os.path.join(MODEL_DIR, "norm_global.pkl")
        if os.path.exists(gpath):
            with open(gpath, "rb") as f:
                self.norm_global = pickle.load(f)
            _gm = self.norm_global.get("mean", self.norm_global.get("global_mean", 0.0))
            logger.info(f"Global norm loaded: mean={_gm:.5f}")

        hg_path = os.path.join(MODEL_DIR, "norm_global_historical.pkl")
        if os.path.exists(hg_path):
            with open(hg_path, "rb") as f:
                self.norm_global_historical = pickle.load(f)
            _hgm = self.norm_global_historical.get("mean", self.norm_global_historical.get("global_mean", 0.0))
            logger.info(f"Historical global norm loaded: mean={_hgm:.5f}")

        ens = load_ensemble("all")
        if ens is not None:
            self.ensembles["all"] = ens
            logger.info("Global model loaded")
        else:
            logger.warning("No model found — run 'python main.py train' first")

    def predict(self, df, baselines=None, already_engineered=False, norm_set="current"):
        if not already_engineered:
            bl = baselines if baselines is not None else self.baselines
            df, _ = engineer_features(df, baselines=bl)

        df = df.copy()
        df["stuff_plus"] = np.nan

        if "all" not in self.ensembles:
            logger.warning("Model not loaded — returning NaN")
            return df

        _global_src  = self.norm_global_historical if (norm_set == "historical" and self.norm_global_historical) else self.norm_global
        _global_mean = _global_src.get("mean", _global_src.get("global_mean", 0.0))
        _global_std  = max(_global_src.get("std", _global_src.get("global_std", 0.007)), 1e-6)

        ens = self.ensembles["all"]

        # Platoon-neutral scoring: score each pitch once as if facing a same-hand
        # batter (same_hand=1.0) and once opposite-hand (same_hand=0.0), average
        # 50/50. The Driveline model uses no location feature, so pitches are scored
        # at their raw shape with no plate_x/plate_z substitution.
        df_same = df.copy(); df_same["same_hand"] = 1.0
        df_opp  = df.copy(); df_opp["same_hand"]  = 0.0
        expected_rv = 0.5 * predict_prob_resid_rv(df_same, ens) + 0.5 * predict_prob_resid_rv(df_opp, ens)
        stuff_plus = 100.0 + (_global_mean - expected_rv) / _global_std * 10.0

        # Velocity floor: a "power" pitch type below 75 mph is almost always a
        # mislabel, so null it rather than emit a bogus grade.
        low_velo   = (df["release_speed"] < 75.0).values
        power_mask = df["pitch_type"].isin(_POWER_TYPES).values
        stuff_plus[power_mask & low_velo] = np.nan

        df["stuff_plus"] = np.clip(stuff_plus, -50.0, 200.0)
        # Raw stuff_plus stays as-is for aggregation; display grade compresses the
        # individual-pitch tail into a realistic ceiling (~135).
        df["stuff_plus_display"] = display_grade(df["stuff_plus"].values)
        return df


_COMPOSITE_PREDICTOR = None


def _composite_pitch_grade(df, norm_set: str = "current") -> dict:
    """Grade each pitch type's AVERAGE pitch — {pitch_type: Stuff+}.

    Averages the model's shape features within a pitch type, then scores that single
    composite pitch. This is not the same as averaging the per-pitch grades: the model
    is a tree ensemble, so mean(grade(x)) != grade(mean(x)), and averaging grades
    quietly penalizes pitchers whose stuff varies more from pitch to pitch.

    `df` must already be engineered (it comes from a *_scored table). Returns {} if the
    model or the required feature columns aren't available, so callers can fall back.
    """
    global _COMPOSITE_PREDICTOR
    if _COMPOSITE_PREDICTOR is None:
        _COMPOSITE_PREDICTOR = StuffPlusPredictor()
    p = _COMPOSITE_PREDICTOR
    ens = p.ensembles.get("all")
    if ens is None or df is None or df.empty or "pitch_type" not in df.columns:
        return {}
    feats = [f for f in ens.get("feats", []) if f in df.columns]
    if len(feats) < len(ens.get("feats", [])):
        return {}

    src = p.norm_global_historical if (norm_set == "historical" and p.norm_global_historical) else p.norm_global
    gm = src.get("mean", src.get("global_mean", 0.0))
    gs = max(src.get("std", src.get("global_std", 0.007)), 1e-6)

    comp = df.groupby("pitch_type")[feats].mean().dropna()
    if comp.empty:
        return {}
    comp = comp.reset_index()
    # neutral columns the scorer touches but the model doesn't use as features
    for col, val in (("plate_x", 0.0), ("plate_z", 2.5), ("same_hand", 0.0),
                     ("vx0", 0.0), ("vy0", -130.0), ("vz0", 0.0), ("ay", 25.0)):
        if col not in comp.columns:
            comp[col] = val
    rv = predict_prob_resid_rv(comp, ens)
    sp = np.clip(100.0 + (gm - rv) / gs * 10.0, -50.0, 200.0)
    return {pt: float(v) for pt, v in zip(comp["pitch_type"], sp) if np.isfinite(v)}


def load_baselines():
    path = os.path.join(MODEL_DIR, "movement_baselines.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No baselines at {path}. Run train first.")
    with open(path, "rb") as f:
        return pickle.load(f)
