"""
Stuff+ model training.

Trains the production model: a single LightGBM regressor on a swing-outcome
run-value target (see model/prob_resid.py). One model covers every pitch type
("all"); there is no per-type or family split, no location model, no count model,
and no platoon marginalization — the shape features are arm-normalized, so the
model cannot identify pitcher handedness and one grade per pitch suffices.

train_unified() does the full run:
  1. engineer shape features (cached to model/feature_cache.parquet),
  2. save the movement/spin baselines inference needs,
  3. train the "all" model (train_prob_resid_ensemble) and save it,
  4. compute and save the global normalization (current + historical)
     that maps xRV onto the 100 = league-average, 10 = one-SD Stuff+ scale,
  5. write the model version hash.
"""

import hashlib
import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR
from features.engineering import CORE_FEATURES, engineer_features
from model.submodels import save_ensemble

logger = logging.getLogger(__name__)

MODEL_VERSION = "v252_core9_spin"


def train_unified(df: pd.DataFrame) -> dict:
    """Train the single Driveline run-value regressor and save all inference artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    FEAT_CACHE      = os.path.join(MODEL_DIR, "feature_cache.parquet")
    FEAT_CACHE_META = os.path.join(MODEL_DIR, "feature_cache_meta.pkl")
    _cache_key = hashlib.md5(str(sorted(df["game_year"].unique().tolist()) if "game_year" in df.columns else df.shape).encode()).hexdigest()

    # Raw columns that pass through engineering unchanged — needed for training
    # but not saved in older caches. We restore them from the original df after
    # loading the cache so we never need to re-engineer just because a raw column
    # was added to RAW_COLS.
    _RAW_PASSTHROUGH = ["bb_type", "launch_speed", "launch_angle", "hc_x", "hc_y", "stand"]

    _cache_loaded = False
    if os.path.exists(FEAT_CACHE) and os.path.exists(FEAT_CACHE_META):
        _meta = pickle.load(open(FEAT_CACHE_META, "rb"))
        if _meta.get("key") == _cache_key and _meta.get("stage") == "post_fastball":
            logger.info("Loading cached engineered features (skipping feature engineering) …")
            df_raw = df  # keep original for passthrough cols
            df = pd.read_parquet(FEAT_CACHE)
            logger.info(f"  Loaded {len(df):,} rows from cache.")
            # Restore any raw passthrough columns missing from the cache
            for _col in _RAW_PASSTHROUGH:
                if _col not in df.columns and _col in df_raw.columns:
                    df[_col] = df_raw[_col].values
                    logger.info(f"  Restored raw column '{_col}' from original df.")
            _cache_loaded = True

    if not _cache_loaded:
        logger.info("Engineering features …")
        df, _baselines = engineer_features(df)

        # Save movement baselines so inference uses the same regression coefficients
        # (vaa_adj / haa_adj slopes, arm_bin_edges, etc.) as training.
        if _baselines is not None:
            _bpath = os.path.join(MODEL_DIR, "movement_baselines.pkl")
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(_bpath, "wb") as _fh:
                pickle.dump(_baselines, _fh)
            logger.info(f"Movement baselines saved → {_bpath}")

        logger.info("Saving feature cache …")
        df.to_parquet(FEAT_CACHE, index=False)
        pickle.dump({"key": _cache_key, "stage": "post_fastball"}, open(FEAT_CACHE_META, "wb"))
        logger.info(f"  Feature cache saved ({len(df):,} rows)")

    # Save spin axis population means so inference can compute spin_axis_rel correctly
    # (avoids collapse to 0 when scoring a single pitcher in isolation).
    if "spin_axis_arm" in df.columns:
        _spin_axis_path = os.path.join(MODEL_DIR, "spin_axis_lookup.pkl")
        _spin_lookup: dict = {}
        _gc = [c for c in ["pitch_type", "p_throws", "game_year"] if c in df.columns]
        if _gc:
            _gm = df.groupby(_gc)["spin_axis_arm"].mean()
            for idx, val in _gm.items():
                if len(_gc) >= 3:
                    pt, throws, yr = idx
                    _spin_lookup[(str(pt), str(throws), int(yr))] = float(val)
                elif len(_gc) == 2:
                    pt, throws = idx
                    _spin_lookup[(str(pt), str(throws), 0)] = float(val)
                else:
                    _spin_lookup[(str(idx), "R", 0)] = float(val)
        # Year=0 fallback: multi-year mean per (pt, throws)
        _fb_cols = [c for c in ["pitch_type", "p_throws"] if c in df.columns]
        if _fb_cols:
            for idx, val in df.groupby(_fb_cols)["spin_axis_arm"].mean().items():
                pt, throws = (idx if isinstance(idx, tuple) else (idx, "R"))
                _spin_lookup[(str(pt), str(throws), 0)] = float(val)
        with open(_spin_axis_path, "wb") as _fh:
            pickle.dump(_spin_lookup, _fh)
        logger.info(f"Spin axis lookup saved ({len(_spin_lookup)} entries)")

    if "stand" in df.columns and "stand_r" not in df.columns:
        df["stand_r"] = (df["stand"] == "R").astype(int)

    # --- Train the single "all" regressor ---
    all_feats = [f for f in CORE_FEATURES if f in df.columns]
    logger.info(f"  [all] Driveline run-value regressor on {len(df):,} pitches, {len(all_feats)} features")
    from model.prob_resid import train_prob_resid_ensemble, predict_prob_resid_rv
    ens_all = train_prob_resid_ensemble(df, features=all_feats)
    if ens_all is None:
        raise RuntimeError("Global model training failed")
    save_ensemble(ens_all, "all")
    ensembles = {"all": ens_all}
    logger.info(f"  [all] model saved.")
    # Remove stale family models so predict.py uses the global model
    for _fam in ("fb", "br", "os", "nfb"):
        _fam_path = os.path.join(MODEL_DIR, f"ensemble_{_fam}.pkl")
        if os.path.exists(_fam_path):
            os.remove(_fam_path)
            logger.info(f"  Removed stale ensemble_{_fam}.pkl")

    def _compute_global_norm(df_norm, suffix=""):
        """Score a sample of pitches with the trained model and return {mean, std}.
        These map xRV onto the Stuff+ scale at inference. A 400k sample estimates
        mean/std tightly while keeping scoring cheap."""
        if len(df_norm) > 400_000:
            df_norm = df_norm.sample(n=400_000, random_state=42)
        e_rv = predict_prob_resid_rv(df_norm, ensembles["all"])
        valid = np.isfinite(e_rv)
        g_mean = float(e_rv[valid].mean())
        g_std  = float(e_rv[valid].std() + 1e-8)
        global_norm = {"mean": g_mean, "std": g_std}
        logger.info(f"  Global norm{suffix}: mean={g_mean:.5f}  std={g_std:.5f}  n={valid.sum():,}")
        return global_norm

    # Current norm (all training years). norm_family / norm_per_type are written
    # empty — predict.py falls back to the single global norm.
    global_norm = _compute_global_norm(df)
    with open(os.path.join(MODEL_DIR, "norm_family.pkl"), "wb") as f:
        pickle.dump({}, f)
    with open(os.path.join(MODEL_DIR, "norm_per_type.pkl"), "wb") as f:
        pickle.dump({}, f)
    with open(os.path.join(MODEL_DIR, "norm_global.pkl"), "wb") as f:
        pickle.dump(global_norm, f)
    logger.info(f"Global norm saved: {global_norm}")

    # Historical norm (2020-2024) — used to score 2023-2025 without contamination
    # from the current season's population.
    if "game_year" in df.columns:
        df_hist = df[df["game_year"] <= 2024]
        global_norm_hist = _compute_global_norm(df_hist, suffix=" [hist]")
        with open(os.path.join(MODEL_DIR, "norm_family_historical.pkl"), "wb") as f:
            pickle.dump({}, f)
        with open(os.path.join(MODEL_DIR, "norm_per_type_historical.pkl"), "wb") as f:
            pickle.dump({}, f)
        with open(os.path.join(MODEL_DIR, "norm_global_historical.pkl"), "wb") as f:
            pickle.dump(global_norm_hist, f)
        logger.info(f"Historical norms (2020-2024) saved")

    version = hashlib.md5(MODEL_VERSION.encode()).hexdigest()[:12]
    with open(os.path.join(MODEL_DIR, "model_version.txt"), "w") as f:
        f.write(version)
    logger.info(f"Model version: {version}")
    logger.info("Done.")

    return ensembles


def train_all(df: pd.DataFrame) -> dict:
    return train_unified(df)


def save_baselines(baselines):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "movement_baselines.pkl")
    with open(path, "wb") as f:
        pickle.dump(baselines, f)
    logger.info(f"Baselines saved → {path}")


def load_baselines():
    path = os.path.join(MODEL_DIR, "movement_baselines.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
