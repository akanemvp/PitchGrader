"""
Stuff+ model training — 3-family ensemble architecture.

Architecture
-----------
• 3 LightGBM ensemble models: fb / br / os
• Each ensemble: whiff (classifier) + foul (classifier) + contact_rv (regressor)
• pitch_type_code as LightGBM categorical feature within each family model
• movement_rv lookup is per pitch type — correct values fed into each family model
• Target: xRV = P(swing)×[P(whiff)×rv_whiff + (1−P(whiff))×rv_nonwhiff] + P(take)×[P(CS)×rv_cs + P(ball)×rv_ball]
• Normalization: global (single mean/std across all types) at output
• Pitcher-level GroupShuffleSplit to prevent leakage
"""

import hashlib
import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR, RV_COL
from features.engineering import CORE_FEATURES, DIFF_FEATURES, OS_FEATURES, FB_FEATURES, BR_FEATURES, FEATURES_BY_TYPE, engineer_features, compute_fastball_diffs, build_movement_rv_lookup, apply_movement_rv, apply_fb_context
from model.location_plus import train_location_plus, compute_residuals, save_location_plus
from model.submodels import (
    train_residual_model,
    train_ensemble,
    train_count_neutral_ensemble,
    train_whiff_model,
    predict_residual_rv,
    predict_ensemble_rv,
    predict_count_neutral_rv,
    predict_whiff_prob,
    compute_rv_baselines,
    compute_linear_weights_target,
    compute_swing_quality_target,
    compute_siera_rv_target,
    compute_count_rv_lookup,
    apply_count_rv_target,
    save_ensemble,
    load_ensemble,
    save_rv_baselines,
    load_rv_baselines,
    LGBM_CLASSIFIER_PARAMS,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "v150_three_family_bip_ev"

# 8 separate per-pitch-type models (v89-v94 architecture — best within-type correlations)
MODEL_KEYS = ["ff", "si", "fc", "sl", "st", "cu", "ch", "fs"]

FAMILY_GROUPS: dict = {
    "ff": ["FF", "FA"],
    "si": ["SI"],
    "fc": ["FC"],
    "sl": ["SL", "SV"],
    "st": ["ST", "SC", "GY"],
    "cu": ["CU", "KC", "CS"],
    "ch": ["CH", "EP", "KN"],
    "fs": ["FS", "FO"],
}

# 3-family grouping: fastball / breaking / offspeed
THREE_FAMILY_GROUPS: dict = {
    "fb": ["FF", "FA", "SI", "FC"],
    "br": ["SL", "SV", "ST", "SC", "GY", "CU", "KC", "CS"],
    "os": ["CH", "FO", "EP", "KN", "FS"],
}

PT_TO_FAMILY: dict = {pt: fam for fam, pts in FAMILY_GROUPS.items() for pt in pts}
FAMILY_FALLBACK: dict = {k: k for k in FAMILY_GROUPS}
PT_TO_GROUP:    dict = PT_TO_FAMILY
PITCH_TYPE_GROUPS: dict = FAMILY_GROUPS


def _compute_spd_from_fb(df: pd.DataFrame) -> pd.DataFrame:
    """Add spd_from_fb = (median FF/SI speed in this df) - release_speed."""
    fb_mask = df["pitch_type"].isin(["FF", "SI", "FA"])
    fb_median = df.loc[fb_mask, "release_speed"].median() if fb_mask.any() else 93.0
    df = df.copy()
    df["spd_from_fb"] = fb_median - df["release_speed"]
    return df


def _predict_rv(df_in, ens, rv_baselines, use_residual=False):
    """Call the right predict function depending on model type."""
    if "whiff_clf" in ens:
        return predict_whiff_prob(df_in, ens)
    if use_residual or "residual" in ens:
        return predict_residual_rv(df_in, ens)
    if "swing_quality" in ens:
        return predict_count_neutral_rv(df_in, ens, rv_baselines)
    # Ensemble: use swing_only=True so _build_norms uses the same formula as inference.
    # This was the v136 bug: norm used full-tree but inference used swing-only → inflation.
    return predict_ensemble_rv(df_in, ens, rv_baselines, swing_only=True)


def _marginalise_rv(pt_df, ens, rv_baselines, count_dist, count_rv_lookup):
    """Kept for residual/legacy model paths. For the 3-sub-model ensemble,
    norms are computed on actual training data (Decision model sees real count)."""
    return _predict_rv(pt_df, ens, rv_baselines)


PT_TO_THREE_FAMILY: dict = {pt: fam for fam, pts in THREE_FAMILY_GROUPS.items() for pt in pts}


def _score_families(df: pd.DataFrame, ensembles: dict, rv_baselines: dict) -> np.ndarray:
    """Score all pitches by routing each to its family ensemble."""
    e_rv = np.full(len(df), np.nan)
    pt_arr = df["pitch_type"].fillna("").values
    # Try THREE_FAMILY_GROUPS first (fb/br/os), fall back to FAMILY_GROUPS (8-type)
    groups = THREE_FAMILY_GROUPS if any(k in ensembles for k in ("fb","br","os")) else FAMILY_GROUPS
    for fam, pts in groups.items():
        ens = ensembles.get(fam)
        if ens is None:
            continue
        mask = np.isin(pt_arr, pts)
        if not mask.any():
            continue
        e_rv[mask] = _predict_rv(df[mask], ens, rv_baselines)
    return e_rv


def _build_norms(df: pd.DataFrame, ensembles: dict, rv_baselines: dict, suffix: str = "",
                 use_residual: bool = False, count_dist: dict = None,
                 count_rv_lookup: "pd.DataFrame | None" = None) -> tuple:
    """Compute global and per-type norms. Works with 3-family or single global ensemble.
    Returns (per_type, family={}, global).
    """
    # Route to correct scoring function
    if "all" in ensembles:
        ens = ensembles["all"]
        e_rv = _predict_rv(df, ens, rv_baselines, use_residual)
    else:
        e_rv = _score_families(df, ensembles, rv_baselines)

    valid = np.isfinite(e_rv)
    arr_all = e_rv[valid]

    # Normalize on raw pitch predictions so between-pitch-type variance is preserved.
    norm_mean = float(arr_all.mean())
    norm_std  = float(arr_all.std() + 1e-8)
    logger.info(f"  Global norm{suffix} [raw pitches]: mean={norm_mean:.5f}  std={norm_std:.5f}  n={valid.sum():,}")

    global_norm = {
        "global_mean": norm_mean,
        "global_std":  norm_std,
    }
    logger.info(
        f"  Global norm{suffix}: mean={global_norm['global_mean']:.5f}  "
        f"std={global_norm['global_std']:.5f}  n={len(arr_all):,}"
    )

    # Per-type norms: each pitch type normalized against itself using per-pitcher means.
    # Std is computed on pitcher-level means so leaderboard has std≈10 spread.
    per_type_norm = {}
    has_pitcher = "pitcher" in df.columns
    for pt in sorted(df["pitch_type"].dropna().unique()):
        pt_df = df[df["pitch_type"] == pt]
        if len(pt_df) < 50:
            continue
        pt_rv = _predict_rv(pt_df, ens, rv_baselines, use_residual)
        pt_valid = np.isfinite(pt_rv)
        if pt_valid.sum() < 20:
            continue
        pt_arr = pt_rv[pt_valid]
        if has_pitcher:
            _agg = pd.DataFrame({"pitcher": pt_df["pitcher"].values[pt_valid], "rv": pt_arr})
            _means = _agg.groupby("pitcher")["rv"].mean()
            _means = _means[np.isfinite(_means)]
            pt_mean = float(_means.mean()) if len(_means) >= 5 else float(pt_arr.mean())
            pt_std  = float(_means.std())  if len(_means) >= 5 else float(pt_arr.std())
        else:
            pt_mean = float(pt_arr.mean())
            pt_std  = float(pt_arr.std())
        per_type_norm[pt] = {"mean": pt_mean, "std": max(pt_std, 1e-6), "n": len(pt_arr)}
        logger.info(f"  [{pt}] norm{suffix}: mean={pt_mean:.5f}  std={pt_std:.5f}  n_pitchers={len(_means) if has_pitcher else len(pt_arr):,}")

    return per_type_norm, {}, global_norm


def _score_all_families(df: pd.DataFrame, ensembles: dict, rv_baselines: dict, features: list, use_residual: bool = False) -> np.ndarray:
    """Score all pitches with family models, return raw e_rv array."""
    e_rv = np.full(len(df), np.nan)
    pt_arr = df["pitch_type"].values
    for fam, pts in FAMILY_GROUPS.items():
        ens = ensembles.get(fam)
        if ens is None:
            continue
        mask = np.isin(pt_arr, pts)
        if mask.any():
            e_rv[mask] = _predict_rv(df[mask], ens, rv_baselines, use_residual)
    return e_rv


def train_unified(df: pd.DataFrame, ensemble_movement_rv: bool = False, residual_model: bool = False, linear_weights: bool = False, count_rv: bool = False, count_neutral: bool = False, swing_quality: bool = False, whiff_model: bool = False, residual_location: bool = False, siera: bool = False) -> dict:
    """Train 3 family models (fb / br / os).

    ensemble_movement_rv: if True, use a 2-pass approach where movement_rv surface is built
    from bootstrap ensemble predictions instead of residual_xrv.
    residual_model: if True, train a single LightGBM regressor on residual_xrv per family
    instead of the whiff/foul/contact_rv ensemble. movement_rv uses residual lookup.
    linear_weights: if True (implies residual_model=True), use context-neutral per-pitch RV
    as training target instead of residual_xrv. Removes count, location, AND base-out contamination.
    count_rv: if True (implies residual_model=True), use TJ Stats approach — count-stratified
    average RV per (event, balls, strikes) as training target.
    """
    import hashlib as _hashlib, pickle as _pkl
    os.makedirs(MODEL_DIR, exist_ok=True)

    FEAT_CACHE      = os.path.join(MODEL_DIR, "feature_cache.parquet")
    FEAT_CACHE_META = os.path.join(MODEL_DIR, "feature_cache_meta.pkl")
    _cache_key = _hashlib.md5(str(sorted(df["game_year"].unique().tolist()) if "game_year" in df.columns else df.shape).encode()).hexdigest()

    # Raw columns that pass through engineering unchanged — needed for training
    # but not saved in older caches. We restore them from the original df after
    # loading the cache so we never need to re-engineer just because a raw column
    # was added to RAW_COLS.
    _RAW_PASSTHROUGH = ["bb_type", "launch_speed"]

    _cache_loaded = False
    if os.path.exists(FEAT_CACHE) and os.path.exists(FEAT_CACHE_META):
        _meta = _pkl.load(open(FEAT_CACHE_META, "rb"))
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
        df, _ = engineer_features(df)

        logger.info("Saving feature cache …")
        df.to_parquet(FEAT_CACHE, index=False)
        _pkl.dump({"key": _cache_key, "stage": "post_fastball"}, open(FEAT_CACHE_META, "wb"))
        logger.info(f"  Feature cache saved ({len(df):,} rows)")

    logger.info("Computing RV baselines …")
    rv_baselines = compute_rv_baselines(df)
    save_rv_baselines(rv_baselines)

    # Save count-specific V_s lookup (RE24 transition matrix) for use at inference
    _count_rv_lookup = compute_count_rv_lookup(df)
    _count_rv_lookup.to_parquet(os.path.join(MODEL_DIR, "count_rv_lookup.parquet"), index=False)
    logger.info(f"Count-specific V_s lookup saved ({len(_count_rv_lookup)} cells)")

    features = list(CORE_FEATURES)
    # No pitch_type_code — each model is per-pitch-type so the feature is redundant
    FAMILY_EXCLUDE = {}

    _needs_movement_rv = "movement_rv" in features
    if residual_location or siera:
        # Residual pipeline: Location+ → residual → Stuff+
        logger.info("Building FB context lookup …")
        df, _ = apply_fb_context(df, lookup=None)
        # Step 2: (siera) compute SIERA target before location regression
        if siera:
            logger.info("Computing SIERA-calibrated target …")
            df["siera_rv"] = compute_siera_rv_target(df)
            target_col = "siera_rv"
        # Step 3: train Location+ model on the chosen target and compute residuals
        logger.info("Training Location+ model …")
        location_model = train_location_plus(df, target_col=target_col if siera else "delta_run_exp")
        save_location_plus(location_model)
        df = compute_residuals(df, location_model, target_col=target_col if siera else "delta_run_exp")
        # residual is now the target for the stuff model
        target_col = "residual_xrv"
        residual_model = True  # use single-regressor path below
    elif not _needs_movement_rv or count_rv or linear_weights or whiff_model or count_neutral:
        # Fast path: skip Location+, residuals, and movement_rv surface
        logger.info("Building FB context lookup …")
        df, _ = apply_fb_context(df, lookup=None)
        pass  # spd_from_fb removed — not in CORE_FEATURES
        # stand_r needed for count-neutral baseline swing model
        if "stand" in df.columns and "stand_r" not in df.columns:
            df["stand_r"] = (df["stand"] == "R").astype(int)
    elif ensemble_movement_rv:
        logger.info("Training Location+ model …")
        location_model = train_location_plus(df)
        save_location_plus(location_model)
        df = compute_residuals(df, location_model)

        logger.info("Building residual movement_rv lookup …")
        mv_lookup_residual = build_movement_rv_lookup(df, rv_col="residual_xrv")
        residual_path = os.path.join(MODEL_DIR, "movement_rv_lookup_residual.pkl")
        with open(residual_path, "wb") as f:
            pickle.dump(mv_lookup_residual, f)
        logger.info(f"Residual movement_rv lookup saved ({len(mv_lookup_residual)} pitch types)")

        # --- Pass 1: bootstrap ensemble with movement_rv=0 ---
        logger.info("Pass 1: Bootstrap ensemble (movement_rv=0) for surface building …")
        df["movement_rv"] = 0.0
        logger.info("Building FB context lookup …")
        df, _ = apply_fb_context(df, lookup=None)
        pass  # spd_from_fb removed — not in CORE_FEATURES

        boot_feats = [f for f in features if f != "movement_rv"] + ["movement_rv"]
        bootstrap_ensembles = {}
        for fam, pts in FAMILY_GROUPS.items():
            fam_df = df[df["pitch_type"].isin(pts)]
            if len(fam_df) < 500:
                continue
            fam_boot_feats = [f for f in boot_feats if f not in FAMILY_EXCLUDE.get(fam, set())]
            logger.info(f"  [boot/{fam}] {len(fam_df):,} pitches …")
            ens = train_ensemble(fam_df, rv_baselines, features=fam_boot_feats)
            if ens is not None:
                bootstrap_ensembles[fam] = ens

        logger.info("Scoring training data with bootstrap ensemble …")
        df["bootstrap_erv"] = _score_all_families(df, bootstrap_ensembles, rv_baselines, boot_feats)
        valid = df["bootstrap_erv"].notna().sum()
        logger.info(f"Bootstrap e_rv: {valid:,} valid, mean={df['bootstrap_erv'].mean():.5f}")

        logger.info("Building ensemble-based movement_rv lookup …")
        mv_lookup = build_movement_rv_lookup(df, rv_col="bootstrap_erv")
        mv_lookup_path = os.path.join(MODEL_DIR, "movement_rv_lookup.pkl")
        with open(mv_lookup_path, "wb") as f:
            pickle.dump(mv_lookup, f)
        logger.info(f"Ensemble movement_rv lookup saved ({len(mv_lookup)} pitch types)")

        df = apply_movement_rv(df, mv_lookup)
        logger.info("Pass 1 complete — movement_rv rebuilt from ensemble predictions.")
    else:
        logger.info("Training Location+ model …")
        location_model = train_location_plus(df)
        save_location_plus(location_model)
        df = compute_residuals(df, location_model)

        logger.info("Building residual movement_rv lookup …")
        mv_lookup_residual = build_movement_rv_lookup(df, rv_col="residual_xrv")
        residual_path = os.path.join(MODEL_DIR, "movement_rv_lookup_residual.pkl")
        with open(residual_path, "wb") as f:
            pickle.dump(mv_lookup_residual, f)
        logger.info(f"Residual movement_rv lookup saved ({len(mv_lookup_residual)} pitch types)")

        mv_lookup = mv_lookup_residual
        mv_lookup_path = os.path.join(MODEL_DIR, "movement_rv_lookup.pkl")
        with open(mv_lookup_path, "wb") as f:
            pickle.dump(mv_lookup, f)
        df = apply_movement_rv(df, mv_lookup)

        logger.info("Building FB context lookup …")
        df, _ = apply_fb_context(df, lookup=None)
        pass  # spd_from_fb removed — not in CORE_FEATURES

    # --- Compute target if requested ---
    if not (residual_location or siera):
        target_col = "residual_xrv"
    if swing_quality:
        logger.info("Computing swing quality target (whiff + xwOBAcon contact, swings only) …")
        df["swing_quality_rv"] = compute_swing_quality_target(df, rv_baselines)
        n_valid = df["swing_quality_rv"].notna().sum()
        logger.info(f"  swing_quality_rv: {n_valid:,} swing pitches, mean={df['swing_quality_rv'].mean():.5f}")
        target_col = "swing_quality_rv"
    elif count_rv:
        logger.info("Computing count-stratified RV lookup (TJ Stats approach) …")
        count_rv_lookup = compute_count_rv_lookup(df)
        df["count_rv"] = apply_count_rv_target(df, count_rv_lookup)
        n_valid = df["count_rv"].notna().sum()
        logger.info(f"  count_rv: {n_valid:,} valid pitches, mean={df['count_rv'].mean():.5f}")
        target_col = "count_rv"
    elif linear_weights:
        logger.info("Computing linear weights target (context-neutral per-pitch RV) …")
        df["linear_weight_rv"] = compute_linear_weights_target(df, rv_baselines)
        n_valid = df["linear_weight_rv"].notna().sum()
        logger.info(f"  linear_weight_rv: {n_valid:,} valid pitches, mean={df['linear_weight_rv'].mean():.5f}")
        target_col = "linear_weight_rv"

    # --- Train single global model across all pitch types ---
    all_feats = [f for f in CORE_FEATURES if f in df.columns]
    logger.info(f"  [all] Training on {len(df):,} pitches, {len(all_feats)} features …")
    ensembles = {}
    ens = train_ensemble(df.copy(), rv_baselines, features=all_feats)
    if ens is None:
        raise RuntimeError("Global model training failed")
    save_ensemble(ens, "all")
    ensembles["all"] = ens
    logger.info(f"  [all] model saved.")

    def _save_family_norms(df_norm, suffix=""):
        """Compute global norm from single model using raw pitch predictions."""
        ens = ensembles.get("all")
        all_e_rv = _predict_rv(df_norm, ens, rv_baselines) if ens else np.full(len(df_norm), np.nan)
        valid_all = np.isfinite(all_e_rv)
        g_mean = float(all_e_rv[valid_all].mean())
        g_std  = float(all_e_rv[valid_all].std() + 1e-8)
        global_norm = {"mean": g_mean, "std": g_std}
        logger.info(f"  Global norm{suffix}: mean={g_mean:.5f}  std={g_std:.5f}  n={valid_all.sum():,}")
        return {}, {}, global_norm

    fam_norms, type_norms, global_norm = _save_family_norms(df)
    with open(os.path.join(MODEL_DIR, "norm_family.pkl"), "wb") as f:
        pickle.dump(fam_norms, f)
    with open(os.path.join(MODEL_DIR, "norm_per_type.pkl"), "wb") as f:
        pickle.dump(type_norms, f)
    with open(os.path.join(MODEL_DIR, "norm_global.pkl"), "wb") as f:
        pickle.dump(global_norm, f)
    logger.info(f"Family norms saved: {fam_norms}")
    logger.info(f"Global norm saved: {global_norm}")

    # Historical norms (2020-2024)
    if "game_year" in df.columns:
        df_hist = df[df["game_year"] <= 2024]
        fam_norms_hist, type_norms_hist, global_norm_hist = _save_family_norms(df_hist, suffix=" [hist]")
        with open(os.path.join(MODEL_DIR, "norm_family_historical.pkl"), "wb") as f:
            pickle.dump(fam_norms_hist, f)
        with open(os.path.join(MODEL_DIR, "norm_per_type_historical.pkl"), "wb") as f:
            pickle.dump(type_norms_hist, f)
        with open(os.path.join(MODEL_DIR, "norm_global_historical.pkl"), "wb") as f:
            pickle.dump(global_norm_hist, f)
        logger.info(f"Historical norms (2020-2024) saved")

    version = hashlib.md5(MODEL_VERSION.encode()).hexdigest()[:12]
    with open(os.path.join(MODEL_DIR, "model_version.txt"), "w") as f:
        f.write(version)
    logger.info(f"Model version: {version}")
    logger.info("Done.")

    return ensembles


def train_per_type(df: pd.DataFrame) -> dict:
    return train_unified(df)

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


def recalibrate(df: pd.DataFrame, target_mean: float = 100.0, target_std: float = 10.0) -> pd.DataFrame:
    df = df.copy()
    valid = df["stuff_plus"].notna() & np.isfinite(df["stuff_plus"])
    for pt in df.loc[valid, "pitch_type"].unique():
        mask   = valid & (df["pitch_type"] == pt)
        scores = df.loc[mask, "stuff_plus"]
        if len(scores) < 10:
            continue
        pt_mean = scores.mean()
        pt_std  = scores.std()
        if pt_std < 1e-6:
            df.loc[mask, "stuff_plus"] = target_mean
            continue
        df.loc[mask, "stuff_plus"] = target_mean + (scores - pt_mean) / pt_std * target_std
    return df
