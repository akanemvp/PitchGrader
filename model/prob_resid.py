"""Stuff+ — 3-family probability model (whiff / in-play HR).

Three separate models — fastball, offspeed, breaking — each trained and normalized on
its OWN scale, so a fastball is graded against fastballs (no cross-family shafting where
splitters out-grade four-seams per pitch). Each family model composes:

    xRV = P(whiff)·V_whiff + P(in-play)·P(HR | in-play)·V_HR

from a multiclass {other, whiff, foul, in-play} head plus a binary P(HR | in-play) head,
both on the 8 arm-normalized induced-Magnus SHAPE_FEATS. Foul is a class (so the
probabilities are proper) but is not valued — only whiff and in-play home-run damage
drive the grade, which keeps power fastballs (low slot, high ride) on top.

Cutters (Statcast "FC") behave like fastballs for some pitchers and like sliders for
others. A Mahalanobis router places each pitcher's cutters in the fastball or breaking
family (per-pitcher majority vote) so slider-cutters aren't scored on the fastball scale.

Lower xRV = better; each family is normalized to 100 = family average, 10 = one SD.
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

# 8 arm-normalized induced-Magnus shape features (ind_vert/ind_horiz_arm added by
# features.engineering.add_magnus; the rest are standard engineered columns).
SHAPE_FEATS = [
    "release_speed", "release_extension", "release_pos_x_arm", "release_pos_z",
    "arm_angle", "release_spin_rate", "ind_vert", "ind_horiz_arm",
]
# Cutter-router space: velocity + arm-relative movement + spin + slot.
ROUTER_FEATS = ["release_speed", "ind_vert", "ind_horiz_arm", "release_spin_rate", "arm_angle"]

# Family membership by Statcast pitch_type. Cutters (FC) are routed at train/score time.
FB_TYPES  = {"FF", "FA", "SI"}
BR_TYPES  = {"SL", "ST", "SV", "SC", "GY", "CU", "KC", "CS", "SLV"}
OFF_TYPES = {"CH", "FO", "FS", "EP", "KN"}
FAMILIES  = ("FB", "OFF", "BR")

ZONE_HALF = 0.83   # half plate-width + ball radius (ft)
_G = 32.174

_LGBM = dict(
    max_depth=8, num_leaves=20, learning_rate=0.05, min_child_samples=30,
    reg_alpha=0.5, reg_lambda=2.0, n_jobs=-1, verbose=-1, random_state=42,
)
_N_MULTI, _N_HR = 600, 500
_SAMPLE_MULTI, _SAMPLE_HR, _SAMPLE_NORM = 1_200_000, 800_000, 150_000


def add_magnus(df: pd.DataFrame) -> pd.DataFrame:
    """Add ind_vert / ind_horiz_arm (induced-Magnus acceleration) from raw kinematics.

    Idempotent — safe to call on an already-engineered frame or a stale feature cache.
    ind_vert  = Magnus (spin-induced) vertical accel = (az+g) minus the drag-parallel part.
    ind_horiz_arm = Magnus horizontal accel, arm-side signed (positive = arm-side run).
    """
    hs = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0).values
    vx = pd.to_numeric(df.get("vx0"), errors="coerce").values
    vy = pd.to_numeric(df.get("vy0"), errors="coerce").values
    vz = pd.to_numeric(df.get("vz0"), errors="coerce").values
    ax = pd.to_numeric(df.get("ax"), errors="coerce").values
    ay = pd.to_numeric(df.get("ay"), errors="coerce").values
    az = pd.to_numeric(df.get("az"), errors="coerce").values
    vm = np.sqrt(vx * vx + vy * vy + vz * vz)
    aax, aaz = ax, az + _G
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = (aax * vx + ay * vy + aaz * vz) / vm
        pz = dot * vz / vm
        px = dot * vx / vm
    df["ind_vert"] = aaz - pz
    df["ind_horiz_arm"] = (aax - px) * hs
    return df


def _basefam(pt: str):
    if pt in FB_TYPES:  return "FB"
    if pt in BR_TYPES:  return "BR"
    if pt in OFF_TYPES: return "OFF"
    return None   # FC handled by the router; unknown types unscored


def _maha(A, m, ci):
    d = A - m
    return np.einsum("ij,jk,ik->i", d, ci, d)


def fit_cutter_router(df: pd.DataFrame) -> dict:
    """Fit Mahalanobis centroids for the fastball vs breaking families in ROUTER_FEATS."""
    Z = df[ROUTER_FEATS].apply(pd.to_numeric, errors="coerce")
    mu, sd = Z.mean(), Z.std().replace(0, 1.0)
    Zs = (Z - mu) / sd
    ok = Zs.notna().all(axis=1)
    pt = df["pitch_type"].astype(str)

    def params(mask):
        A = Zs[ok & mask].values
        return A.mean(0), np.linalg.inv(np.cov(A.T) + 1e-6 * np.eye(A.shape[1]))

    m_fb, ci_fb = params(pt.isin(["FF", "SI"]))
    m_br, ci_br = params(pt.isin(["SL", "ST", "CU", "KC"]))
    return {"mu": mu.values, "sd": sd.values, "m_fb": m_fb, "ci_fb": ci_fb,
            "m_br": m_br, "ci_br": ci_br}


def assign_family(df: pd.DataFrame, router: dict | None) -> pd.Series:
    """Family (FB/OFF/BR) per pitch. Cutters routed by Mahalanobis; per-pitcher majority
    when a pitcher/name column is present, else per-pitch."""
    pt = df["pitch_type"].astype(str)
    fam = pt.map(_basefam)
    fc = pt.eq("FC")
    if fc.any() and router is not None:
        Z = (df.loc[fc, ROUTER_FEATS].apply(pd.to_numeric, errors="coerce").values
             - router["mu"]) / router["sd"]
        ok = np.isfinite(Z).all(axis=1)
        r = np.full(int(fc.sum()), None, dtype=object)
        if ok.any():
            d_fb = _maha(Z[ok], router["m_fb"], router["ci_fb"])
            d_br = _maha(Z[ok], router["m_br"], router["ci_br"])
            r[ok] = np.where(d_fb < d_br, "FB", "BR")
        rr = pd.Series(r, index=df.index[fc])
        pcol = "player_name" if "player_name" in df.columns else ("pitcher" if "pitcher" in df.columns else None)
        if pcol is not None:
            tmp = pd.DataFrame({"p": df.loc[fc, pcol].values, "r": rr.values})
            tmp = tmp[tmp["r"].notna()]
            if len(tmp):
                maj = tmp.groupby("p")["r"].agg(lambda x: x.value_counts().index[0])
                fam.loc[df.index[fc]] = df.loc[fc, pcol].map(maj).values
        else:
            fam.loc[rr.index] = rr.values
    return fam


def _cav(ca, mask):
    v = ca[mask & ca.notna()]
    return float(v.mean()) if len(v) else 0.0


def bucket_values(df: pd.DataFrame) -> dict:
    """Global count-adjusted run values for the valued outcomes (whiff, in-play HR)."""
    dd = df["description"].fillna("").astype(str)
    ev = df["events"].fillna("").astype(str)
    dre = pd.to_numeric(df["delta_run_exp"], errors="coerce")
    b = pd.to_numeric(df["balls"], errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(df["strikes"], errors="coerce").fillna(0).astype(int)
    ca = dre - pd.DataFrame({"b": b, "s": s, "d": dre}).groupby(["b", "s"]).d.transform("mean")
    isw = dd.isin(WHIFF_DESCS); isip = dd.eq(INPLAY_DESC); ishr = ev.eq("home_run")
    return {"whiff": _cav(ca, isw), "hr": _cav(ca, isip & ishr)}


def train_family_prob(df: pd.DataFrame, fam_mask: pd.Series, family: str, V: dict) -> dict:
    """Train one family's probability model: multiclass {other,whiff,foul,inplay} + P(HR|inplay)."""
    dd = df["description"].fillna("").astype(str)
    ev = df["events"].fillna("").astype(str)
    isw = dd.isin(WHIFF_DESCS); isf = dd.isin(FOUL_DESCS)
    isip = dd.eq(INPLAY_DESC); ishr = ev.eq("home_run")
    sf = df[SHAPE_FEATS].notna().all(axis=1)
    base = fam_mask & sf
    rng = np.random.RandomState(42)

    lab = pd.Series(0, index=df.index)
    lab[isw] = 1; lab[isf] = 2; lab[isip] = 3
    idx = df.index[base]
    if len(idx) > _SAMPLE_MULTI:
        idx = pd.Index(rng.choice(idx, _SAMPLE_MULTI, replace=False))
    mc = lgb.LGBMClassifier(objective="multiclass", num_class=4, n_estimators=_N_MULTI, **_LGBM)
    mc.fit(df.loc[idx, SHAPE_FEATS].values, lab.loc[idx].values)

    hidx = df.index[base & isip]
    if len(hidx) > _SAMPLE_HR:
        hidx = pd.Index(rng.choice(hidx, _SAMPLE_HR, replace=False))
    hh = lgb.LGBMClassifier(objective="binary", n_estimators=_N_HR, **_LGBM)
    hh.fit(df.loc[hidx, SHAPE_FEATS].values, ishr.loc[hidx].astype(int).values)

    ens = {"method": "prob_family", "family": family, "feats": SHAPE_FEATS,
           "shape_feats": SHAPE_FEATS, "multiclass": mc, "hr": hh,
           "V_whiff": V["whiff"], "V_hr": V["hr"], "weights": V,
           "groups": {family: None}, "count_dist": [((0, 0), 1.0)]}

    samp = df.loc[base, SHAPE_FEATS].sample(n=min(_SAMPLE_NORM, int(base.sum())), random_state=42)
    e = predict_family_rv(samp, ens)
    ens["norm"] = {"mean": float(np.nanmean(e)), "std": float(np.nanstd(e) + 1e-8)}
    logger.info(f"  [{family}] whiff={V['whiff']:+.4f} hr={V['hr']:+.3f} "
                f"norm mean={ens['norm']['mean']:+.5f} std={ens['norm']['std']:.5f} "
                f"(multi n={len(idx):,}, hr n={len(hidx):,})")
    return ens


def predict_family_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    """Expected run value from one family's probability model: P(wh)·Vwh + P(ip)·P(HR)·Vhr."""
    if len(df) == 0:
        return np.full(0, np.nan)
    X = df[SHAPE_FEATS].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).values
    P = ens["multiclass"].predict_proba(X)
    phr = ens["hr"].predict_proba(X)[:, 1]
    return P[:, 1] * ens["V_whiff"] + P[:, 3] * phr * ens["V_hr"]


# --- back-compat: some callers still import these names ---
def _batter_zone(df: pd.DataFrame) -> pd.Series:
    px = pd.to_numeric(df.get("plate_x"), errors="coerce")
    pz = pd.to_numeric(df.get("plate_z"), errors="coerce")
    szt = pd.to_numeric(df.get("sz_top"), errors="coerce")
    szb = pd.to_numeric(df.get("sz_bot"), errors="coerce")
    inz = (px.abs().values <= ZONE_HALF) & (pz.values >= szb.values) & (pz.values <= szt.values)
    return pd.Series(np.where(pz.notna().values, inz, False), index=df.index).fillna(False)


def predict_prob_resid_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    """Back-compat shim: score with a single family ensemble."""
    return predict_family_rv(df, ens)
