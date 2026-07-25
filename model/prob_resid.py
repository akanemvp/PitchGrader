"""Stuff+ — direct run-value regressor (mean-outcome target).

A pitch's grade is its expected run value given shape alone. Rather than composing
a tree of conditional-probability heads, we assign every pitch the *mean* count-
adjusted run value of its outcome bucket and fit a single LightGBM regressor to
predict that value from shape:

    target(pitch) = mean count-adjusted RV of its outcome bucket
    xRV(shape)    = regressor(shape)          # E[RV | shape]

Outcome buckets (each pitch falls in exactly one):
    whiff · foul · in-play HR · in-play non-HR · called strike · ball
Called strike vs ball is the only place location enters — an in-zone take is a
called strike, an out-of-zone take is a ball — labeled at TRAIN time against the
individual hitter's strike zone. Whiff/foul/in-play values are global (not split
by zone). Everything is trained on the 8 arm-normalized SHAPE_FEATS; no location,
count, or handedness as *features*.

Why the mean-outcome target: raw per-pitch delta_run_exp is dominated by sequence
luck (the same whiff scores differently by count/base-out state). Collapsing each
outcome to its league-mean RV strips that noise, and one regressor fit directly to
E[RV | shape] is better-calibrated on cross-year predictiveness than a cascade of
independently-fit heads — and simpler. A finer contact grid (GB/AIR/HR/PU) and a
7-head structured tree were both tried and lost: shape cannot predict exit velocity,
and the extra structure added noise, not signal.

Lower xRV = better; normalized to 100 = league average, 10 = one standard deviation.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)

BALL_DESCS   = {"ball", "blocked_ball", "hit_by_pitch", "pitchout"}
CALLED_DESCS = {"called_strike"}
WHIFF_DESCS  = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOUL_DESCS   = {"foul", "foul_tip", "bunt_foul_tip", "foul_bunt"}
INPLAY_DESC  = "hit_into_play"

SHAPE_FEATS = [
    "release_speed", "release_extension", "az", "ax_arm",
    "release_pos_x_arm", "release_pos_z", "arm_angle", "release_spin_rate",
]
# half plate-width + ball radius (ft): the horizontal edge of the rulebook zone
ZONE_HALF = 0.83

# LightGBM: light smoothing (path_smooth 0.5, min_child 400) so the velocity/extension
# response stays smooth without over-regularizing the way the old structured model did,
# plus per-fit early stopping so the regressor stops when held-out loss stalls rather
# than overlearning. deterministic + force_col_wise give reproducible grades.
_LGBM = dict(
    n_estimators=6000, max_depth=6, learning_rate=0.04, subsample=0.8,
    colsample_bytree=0.8, min_child_samples=400, reg_lambda=2.0,
    path_smooth=0.5, min_split_gain=1e-5, extra_trees=False,
    random_state=42, n_jobs=-1, deterministic=True, force_col_wise=True, verbose=-1,
)
_ES_PATIENCE = 50   # early-stopping rounds on the 15% validation split


def _batter_zone(df: pd.DataFrame) -> pd.Series:
    """Boolean in-zone mask using each batter's own strike zone (mean sz_top/sz_bot)."""
    px = pd.to_numeric(df.get("plate_x"), errors="coerce")
    pz = pd.to_numeric(df.get("plate_z"), errors="coerce")
    szt = pd.to_numeric(df.get("sz_top"), errors="coerce")
    szb = pd.to_numeric(df.get("sz_bot"), errors="coerce")
    if "batter" in df.columns:
        bz = pd.DataFrame({"batter": df["batter"], "_t": szt, "_b": szb}).groupby("batter").agg(
            zt=("_t", "mean"), zb=("_b", "mean"))
        z = df[["batter"]].merge(bz, on="batter", how="left")
        zt_b, zb_b = z["zt"].values, z["zb"].values
    else:
        zt_b, zb_b = szt.values, szb.values
    inz = (px.abs().values <= ZONE_HALF) & (pz.values >= zb_b) & (pz.values <= zt_b)
    return pd.Series(np.where(np.isfinite(zt_b) & pz.notna().values, inz, False),
                     index=df.index).fillna(False)


def train_prob_resid_ensemble(df: pd.DataFrame, features=None) -> dict:
    df = df.copy()
    dd = df["description"].fillna("").astype(str)
    ev = df["events"].fillna("").astype(str)
    dre = pd.to_numeric(df["delta_run_exp"], errors="coerce")
    b = pd.to_numeric(df["balls"], errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(df["strikes"], errors="coerce").fillna(0).astype(int)
    # count-adjusted run value: subtract the per-count mean so values are count-neutral
    ca = dre - pd.DataFrame({"b": b, "s": s, "d": dre}).groupby(["b", "s"]).d.transform("mean")

    inz = _batter_zone(df); ooz = ~inz
    isw = dd.isin(WHIFF_DESCS); isf = dd.isin(FOUL_DESCS); isip = dd.eq(INPLAY_DESC)
    ishr = ev.eq("home_run")
    isswing = isw | isf | isip; istake = ~isswing
    sf = df[SHAPE_FEATS].notna().all(axis=1)

    def cav(mask):   # mean count-adjusted RV under a mask
        v = ca[mask & ca.notna()]
        return float(v.mean()) if len(v) else 0.0

    # league-mean run value per outcome bucket
    W = {
        "whiff": cav(isw), "foul": cav(isf),
        "hr":    cav(isip & ishr), "nonhr": cav(isip & ~ishr),
        "cs":    cav(inz & istake),   # in-zone take = called strike
        "ball":  cav(ooz & istake),   # out-of-zone take = ball
    }
    logger.info(f"  bucket RVs: whiff={W['whiff']:+.4f} foul={W['foul']:+.4f} "
                f"hr={W['hr']:+.3f} nonhr={W['nonhr']:+.4f} cs={W['cs']:+.4f} ball={W['ball']:+.4f}")

    # assign each pitch its bucket's mean RV (the denoised regression target)
    tgt = pd.Series(np.nan, index=df.index)
    tgt[isw] = W["whiff"]; tgt[isf] = W["foul"]
    tgt[isip & ~ishr] = W["nonhr"]; tgt[isip & ishr] = W["hr"]
    tgt[inz & istake] = W["cs"]; tgt[ooz & istake] = W["ball"]

    mask = sf & tgt.notna()
    X = df.loc[mask, SHAPE_FEATS].values
    y = tgt[mask].values
    logger.info(f"  direct RV regressor on {int(mask.sum()):,} shaped pitches")

    # 85/15 split (fixed seed) with early stopping so the fit stops at convergence
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X)); cut = int(len(X) * 0.85)
    reg = lgb.LGBMRegressor(**_LGBM)
    reg.fit(X[idx[:cut]], y[idx[:cut]],
            eval_set=[(X[idx[cut:]], y[idx[cut:]])],
            callbacks=[lgb.early_stopping(_ES_PATIENCE, verbose=False)])
    logger.info(f"  early-stopped at {reg.best_iteration_} trees")

    ens = {"method": "direct", "feats": SHAPE_FEATS, "shape_feats": SHAPE_FEATS,
           "regressor": reg, "weights": W,
           # legacy keys some callers probe:
           "groups": {"all": None}, "count_dist": [((0, 0), 1.0)]}

    samp = df.loc[mask, SHAPE_FEATS].sample(n=min(150000, int(mask.sum())), random_state=42)
    preds = predict_prob_resid_rv(samp, ens)
    ens["norm"] = {"raw_mean": float(np.nanmean(preds)), "raw_std": float(np.nanstd(preds)),
                   "target_mean": 100.0, "target_std": 10.0}
    logger.info(f"  Norm (n={len(samp):,}): xRV mean={ens['norm']['raw_mean']:+.5f} "
                f"std={ens['norm']['raw_std']:.5f}")
    return ens


def predict_prob_resid_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    """Expected run value from shape alone — the direct regressor's prediction."""
    if len(df) == 0:
        return np.full(0, np.nan)
    reg = ens["regressor"]
    X = df[SHAPE_FEATS].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).values
    return np.asarray(reg.predict(X), dtype=float)
