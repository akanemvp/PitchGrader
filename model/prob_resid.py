"""Driveline-style Stuff+ — one LightGBM regressor on a swing-outcome run-value target.

For each pitch, xRV is predicted from 8 shape features: velocity, extension, vertical
acceleration, arm-normalized horizontal acceleration, arm-normalized release side,
release height, arm angle, and spin rate. No location, no count, no batter handedness.

The horizontal features are MIRRORED (arm-side positive for both hands) so a lefty and
a righty throwing physically identical pitches grade identically — verified by a mirror
test (asymmetry 0.00; it was +4.7 with raw signed features).

Target — swings only; balls and called strikes are excluded, since whether a taken
pitch is a ball or a strike is mostly location/command, which this model cannot see:
  • whiff / foul  → actual delta_run_exp
  • in play       → spray-aware xRV from (exit velo, launch angle, spray angle),
                    which xwOBAcon cannot capture because it is spray-blind

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
# ax_arm / release_pos_x_arm are the MIRRORED (arm-normalized) versions of ax and
# release_pos_x — arm-side is positive for both hands. Using the raw signed values
# forced the model to learn identical physics separately in each sign region from
# ~73/27 lopsided data, handing lefties a ~4-5 point phantom bonus. Verified via a
# mirror test: flipping a pitch's handedness moved its grade +4.7 with raw features
# and exactly 0.00 with these.
#
# `same_hand` is deliberately absent: with mirrored features the model cannot
# identify pitcher handedness, so platoon marginalization became a no-op (+0.08
# points league-wide). One grade per pitch, single scoring pass.
LOC_FEATS = ["plate_x", "plate_z", "sz_top", "sz_bot"]   # location + batter strike-zone size
COUNT_FEATS = []                                   # no count (faithful Driveline Stuff+)
XWOBA_COL = "estimated_woba_using_speedangle"
LGBM_REG = dict(n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=200, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbose=-1)

FASTBALL_T = {"FF", "FA", "SI", "FT"}
OFFSPEED_T = {"CH", "FS", "FO"}
GROUPED = False                  # False = one model for all pitches (no family split)


def _approach_angles(df):
    """Vertical & horizontal approach angle (deg) at plate front (y=17/12) from velocity/accel."""
    y0, yf = 50.0, 17.0 / 12.0
    g = lambda c: pd.to_numeric(df.get(c), errors="coerce")
    vx0, vy0, vz0, ax, ay, az = g("vx0"), g("vy0"), g("vz0"), g("ax"), g("ay"), g("az")
    disc = (vy0 ** 2 - 2 * ay * (y0 - yf)).clip(lower=0)
    t = (-vy0 - np.sqrt(disc)) / ay
    vy_f = vy0 + ay * t; vz_f = vz0 + az * t; vx_f = vx0 + ax * t
    vaa = np.degrees(np.arctan2(vz_f, -vy_f))
    haa = np.degrees(np.arctan2(vx_f, -vy_f))
    return vaa, haa


def _fit_approach(df):
    """Fit the location (plate_z/plate_x) component of VAA/HAA so it can be removed (deg-2 poly)."""
    vaa, haa = _approach_angles(df)
    pz = pd.to_numeric(df.get("plate_z"), errors="coerce")
    px = pd.to_numeric(df.get("plate_x"), errors="coerce")
    mv = vaa.notna() & pz.notna(); mh = haa.notna() & px.notna()
    cz = np.polyfit(pz[mv].values, vaa[mv].values, 2)
    cx = np.polyfit(px[mh].values, haa[mh].values, 2)
    return cz.tolist(), cx.tolist()


def _apply_approach(df, cz, cx):
    """Add location-adjusted VAA/HAA columns: raw approach angle minus its location-expected value."""
    vaa, haa = _approach_angles(df)
    pz = pd.to_numeric(df.get("plate_z"), errors="coerce"); pz = pz.fillna(pz.median())
    px = pd.to_numeric(df.get("plate_x"), errors="coerce"); px = px.fillna(px.median())
    df["vaa_adj"] = vaa.values - np.polyval(cz, pz.values)
    df["haa_adj"] = haa.values - np.polyval(cx, px.values)
    return df


def _pitch_group(df) -> pd.Series:
    """fastball / breaking / offspeed. Cutters split by shape via Stage-0 classifier.
    When GROUPED is False, one model for everything."""
    if not GROUPED:
        return pd.Series("all", index=df.index)
    pt = df.get("pitch_type", pd.Series("", index=df.index)).fillna("").astype(str)
    grp = pd.Series("breaking", index=df.index, dtype=object)
    grp[pt.isin(FASTBALL_T)] = "fastball"
    grp[pt.isin(OFFSPEED_T)] = "offspeed"
    fc = pt == "FC"
    if fc.any():
        # cutter Stage-0 method classifies by shape; route to breaking or offspeed (never fastball).
        # Cutters are hard pitches (breaking-family), never changeup/splitter-like, so they land in breaking.
        try:
            from model.cutter_stage0 import classify_cutters
            classify_cutters(df)                   # method runs; breaking-vs-offspeed → all breaking
        except Exception:
            pass
        grp[fc] = "breaking"
    return grp


def _fit_group(g_base, mf, off):
    """Driveline single RV regressor on the all-outcome target '_swrv' (no location)."""
    tgt = g_base["_swrv"].values
    reg = lgb.LGBMRegressor(**LGBM_REG).fit(g_base[mf].values, tgt)
    return {"reg": reg, "base": 0.0}, len(g_base)


def train_prob_resid_ensemble(df: pd.DataFrame, rv_baselines: dict, features=None) -> dict:
    df = df[df["delta_run_exp"].notna() & np.isfinite(df["delta_run_exp"])].copy()
    desc = df["description"].fillna("").astype(str)
    dre = pd.to_numeric(df["delta_run_exp"], errors="coerce")
    is_w = desc.isin(WHIFF_DESCS); is_f = desc.isin(FOUL_DESCS); is_ip = desc == INPLAY_DESC
    # SWING-ONLY: balls and called strikes are excluded. Whether a taken pitch is a
    # ball or a strike is mostly location/command, which this model can't see (it has
    # no location features), so takes inject noise rather than shape signal.
    _n_takes = int((desc.isin(BALL_DESCS) | desc.isin(CALLED_DESCS)).sum())
    df = df[is_w | is_f | is_ip].copy()
    desc = desc[df.index]; dre = dre[df.index]
    is_w = is_w[df.index]; is_f = is_f[df.index]; is_ip = is_ip[df.index]
    logger.info(f"  Swing-only target: dropped {_n_takes:,} takes; training on {len(df):,} swings")
    cz, cx = _fit_approach(df); _apply_approach(df, cz, cx)        # location-adjusted VAA/HAA
    sh = [f for f in SHAPE_FEATS if f in df.columns]
    off = [f for f in LOC_FEATS if f in df.columns]
    for zc in ("sz_top", "sz_bot"):                            # fill sparse zone nulls, don't drop rows
        if zc in df.columns:
            df[zc] = pd.to_numeric(df[zc], errors="coerce").fillna(df[zc].median())

    cnt = [f for f in COUNT_FEATS if f in df.columns]
    mf = sh + cnt

    # ---- Target: whiff/foul = actual delta_run_exp; in-play = SPRAY-AWARE xRV ----
    # xwOBAcon knows only exit velo + launch angle — it is blind to spray direction,
    # so it systematically flatters breaking-ball contact and penalizes grounders
    # (measured: sinkers beat their xwOBAcon by ~23 wOBA pts, sweepers trailed by ~41).
    # A model on (EV, LA, spray) predicts in-play run value far better out-of-sample
    # (r 0.641 -> 0.701) and cuts the pitch-type bias ~18%.
    mean_c = float(dre[is_ip].mean())
    df["_swrv"] = np.nan
    df.loc[is_w | is_f, "_swrv"] = dre[is_w | is_f]
    _spray_cols = ["launch_speed", "launch_angle", "hc_x", "hc_y", "stand"]
    if is_ip.any() and all(c in df.columns for c in _spray_cols):
        _sp = np.degrees(np.arctan2(pd.to_numeric(df["hc_x"], errors="coerce") - 125.42,
                                    198.27 - pd.to_numeric(df["hc_y"], errors="coerce")))
        df["_spray"] = _sp * np.where(df["stand"].astype(str) == "R", 1.0, -1.0)
        _F = ["launch_speed", "launch_angle", "_spray"]
        _fit = df[is_ip].dropna(subset=_F + ["delta_run_exp"])
        _cm = lgb.LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbose=-1).fit(_fit[_F], _fit["delta_run_exp"])
        _ok = df[_F].notna().all(axis=1) & is_ip
        df.loc[_ok, "_swrv"] = _cm.predict(df.loc[_ok, _F])
        df.loc[is_ip & ~_ok, "_swrv"] = mean_c
        logger.info(f"  In-play target: spray-aware xRV on (EV, LA, spray), fit on {len(_fit):,} BIP")
    else:
        lg = rv_baselines.get("lg_xwoba_con", 0.370); scl = rv_baselines.get("woba_scale", 1.15)
        xw = pd.to_numeric(df.get(XWOBA_COL), errors="coerce")
        df.loc[is_ip, "_swrv"] = ((xw - lg) / scl + mean_c)[is_ip].fillna(mean_c)
        logger.warning("  spray columns missing — falling back to xwOBAcon in-play target")

    base = df.dropna(subset=mf + off + ["_swrv"])
    grp = _pitch_group(base)
    groups = {}
    for g in sorted(grp.unique()):
        gi = base.index[grp == g]
        groups[g], n = _fit_group(base.loc[gi], mf, off)
        logger.info(f"  [{g:8s}] n={n:,}  (grouped={GROUPED}, no location)")

    ens = {"prob_resid": True, "grouped": True, "feats": mf, "shape_feats": sh, "count_feats": cnt,
           "loc_feats": off, "groups": groups, "vaa_coef": cz, "haa_coef": cx}

    # ---- count: score at the MEAN count (1 eval ≈ full 12-count marginal, ~12× faster) ----
    if cnt:
        mb = float(base["balls"].mean()); ms = float(base["strikes"].mean())
        ens["count_dist"] = [((mb, ms), 1.0)]
        logger.info(f"  count: scored at mean count (balls={mb:.2f}, strikes={ms:.2f}) — fast marginal approx")
    else:
        ens["count_dist"] = [((0, 0), 1.0)]

    samp = base.sample(n=min(120_000, len(base)), random_state=42)
    preds = predict_prob_resid_rv(samp, ens)
    ens["norm"] = {"raw_mean": float(np.nanmean(preds)), "raw_std": float(np.nanstd(preds)),
                   "target_mean": 100.0, "target_std": 10.0}
    logger.info(f"  Norm (n={len(samp):,}): xRV mean={ens['norm']['raw_mean']:+.5f} "
                f"std={ens['norm']['raw_std']:.5f}")
    return ens


def _group_xrv(sub, node, mf, count_dist):
    """Driveline RV regressor xRV, marginalized over platoon (and count)."""
    s = sub[[f for f in mf if f in sub.columns]].copy()
    for f in s.columns:
        if f in ("same_hand", "balls", "strikes"):
            continue
        if s[f].isna().any():
            s[f] = s[f].fillna(s[f].median())
    out = np.zeros(len(sub), dtype=float)
    hand_states = [1.0, 0.0] if "same_hand" in mf else [None]
    for h in hand_states:
        if h is not None:
            s["same_hand"] = h
        for (b, st), w in count_dist:
            if "balls" in mf:
                s["balls"] = b; s["strikes"] = st
            out += w * (node["base"] + node["reg"].predict(s[mf].values))
    return out / len(hand_states)


def predict_prob_resid_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    if len(df) == 0:
        return np.full(0, np.nan)
    if "vaa_coef" in ens:
        df = _apply_approach(df.copy(), ens["vaa_coef"], ens["haa_coef"])
    mf = ens["feats"]
    cd = ens.get("count_dist") or [((0, 0), 1.0)]
    grp = _pitch_group(df)
    out = np.full(len(df), np.nan)
    pos = {idx: i for i, idx in enumerate(df.index)}
    for g in ens["groups"]:
        m = (grp == g)
        if not m.any():
            continue
        sub = df[m]
        vals = _group_xrv(sub, ens["groups"][g], mf, cd)
        for idx, v in zip(sub.index, vals):
            out[pos[idx]] = v
    return out
