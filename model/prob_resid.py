"""Stuff+ — one global swing-outcome model (Pitch Profiler / proStuff+ style).

A single model, trained on every pitch, grades pitch shape by its expected run
value from three swing outcomes:

    xRV = P(whiff)·V_whiff + P(foul)·V_foul + P(in-play)·E[contact RV | in-play]

  • A swing softmax head predicts {whiff, foul, in-play} from pitch shape.
  • A 5-cell contact grid predicts where an in-play ball lands — ground ball / air,
    each split at 95 mph exit velocity, plus pop-ups — and each cell carries its
    empirical run value, so E[contact RV | in-play] is the grid's expectation.

Shape is described by 8 arm-normalized features: velocity, spin rate, raw vertical
and (arm-side) horizontal acceleration, arm angle, release side and height, and
extension. No location, count, or game-state — a pitch grades on its shape alone,
and a lefty and righty throwing physically identical pitches grade identically.

Lower xRV = better. Grades are one global z-score: 100 = league-average pitch,
10 = one standard deviation. Because the scale is shared across pitch types,
breaking balls (higher whiff, softer contact) sit above fastballs on average.

The cutter router (Mahalanobis, fastball vs breaking) and family helpers are kept
for back-compat with callers that tag pitch families; scoring itself is global.
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

# 8 arm-normalized shape features. ax_arm / az are RAW accelerations (arm-side and
# vertical); release_pos_x_arm is the arm-side release point. add_shape_features
# derives the three arm-signed columns from raw kinematics.
SHAPE_FEATS = [
    "release_speed", "release_spin_rate", "ax_arm", "az",
    "arm_angle", "release_pos_x_arm", "release_pos_z", "release_extension",
]
# Cutter-router space: velocity + arm-relative movement + spin + slot.
ROUTER_FEATS = ["release_speed", "ind_vert", "ind_horiz_arm", "release_spin_rate", "arm_angle"]

# Family membership by Statcast pitch_type. Cutters (FC) are routed by the router.
FB_TYPES  = {"FF", "FA", "SI"}
BR_TYPES  = {"SL", "ST", "SV", "SC", "GY", "CU", "KC", "CS", "SLV"}
OFF_TYPES = {"CH", "FO", "FS", "EP", "KN"}
FAMILIES  = ("FB", "OFF", "BR")

ZONE_HALF = 0.83   # half plate-width + ball radius (ft)
_G = 32.174

# Swing softmax {whiff, foul, in-play}: 31 leaves, slow lr (tjStuff-style).
_LGBM_SWING = dict(
    num_leaves=31, learning_rate=0.01, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.2, n_jobs=-1, verbose=-1, random_state=42,
)
# 5-cell contact grid — heavily smoothed. The per-cell run values are steady but the
# joint shape->cell map is noisy per pitch, so big min_child + strong reg keep the
# contact expectation gently varying with shape instead of over-fitting the tail.
_LGBM_GRID = dict(
    num_leaves=15, learning_rate=0.03, min_child_samples=3000,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.2, reg_lambda=0.5, n_jobs=-1, verbose=-1, random_state=42,
)
_N_SWING, _N_GRID = 1000, 500
_SAMPLE_SWING, _SAMPLE_GRID, _SAMPLE_NORM = 2_500_000, 1_500_000, 300_000

_CELL_NAMES = ["GB<95", "GB95+", "air<95", "air95+", "pop"]


def add_magnus(df: pd.DataFrame) -> pd.DataFrame:
    """Add induced-Magnus accel components from raw kinematics. Idempotent.

    ind_vert / ind_horiz(_arm) feed the cutter router (ROUTER_FEATS). The scoring
    model uses raw accelerations (add_shape_features), not these.
    """
    hs = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0).values
    vx = pd.to_numeric(df.get("vx0"), errors="coerce").values
    vy = pd.to_numeric(df.get("vy0"), errors="coerce").values
    vz = pd.to_numeric(df.get("vz0"), errors="coerce").values
    ax = pd.to_numeric(df.get("ax"), errors="coerce").values
    az = pd.to_numeric(df.get("az"), errors="coerce").values
    ay = pd.to_numeric(df.get("ay"), errors="coerce").values
    vm = np.sqrt(vx * vx + vy * vy + vz * vz)
    aax, aaz = ax, az + _G
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = (aax * vx + ay * vy + aaz * vz) / vm
        pz = dot * vz / vm
        px = dot * vx / vm
    df["ind_vert"] = aaz - pz
    df["ind_horiz"] = aax - px
    df["ind_horiz_arm"] = (aax - px) * hs
    return df


def add_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the raw arm-signed shape columns the scoring model needs. Idempotent.

    ax_arm            = raw horizontal accel, arm-side signed.
    az                = raw vertical accel (unsigned; downward is negative).
    release_pos_x_arm = release side, arm-side signed.
    """
    hs = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0).values
    df["ax_arm"] = pd.to_numeric(df.get("ax"), errors="coerce").values * hs
    df["az"] = pd.to_numeric(df.get("az"), errors="coerce").values
    df["release_pos_x_arm"] = pd.to_numeric(df.get("release_pos_x"), errors="coerce").values * hs
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
    when a pitcher/name column is present, else per-pitch. (Tagging only — scoring is global.)"""
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


def _count_adjusted_rv(df: pd.DataFrame) -> pd.Series:
    """delta_run_exp with the mean removed per (balls, strikes) count — a shape-neutral,
    count-adjusted run value. Base-out state is added when those columns are present."""
    dre = pd.to_numeric(df["delta_run_exp"], errors="coerce")
    b = pd.to_numeric(df.get("balls"), errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(df.get("strikes"), errors="coerce").fillna(0).astype(int)
    grp = pd.DataFrame({"b": b, "s": s, "d": dre})
    keys = ["b", "s"]
    have_situ = all(c in df.columns for c in ("on_1b", "on_2b", "on_3b", "outs_when_up"))
    if have_situ:
        b1 = df["on_1b"].notna().astype(int); b2 = df["on_2b"].notna().astype(int)
        b3 = df["on_3b"].notna().astype(int)
        outs = pd.to_numeric(df["outs_when_up"], errors="coerce").fillna(0).astype(int)
        grp["situ"] = b1 * 4 + b2 * 2 + b3 + outs * 8
        keys.append("situ")
    return dre - grp.groupby(keys)["d"].transform("mean")


def _grid_cell(df: pd.DataFrame) -> pd.Series:
    """5-cell contact grid label: 0 GB<95, 1 GB95+, 2 air<95, 3 air95+, 4 pop-up (grouped)."""
    la = pd.to_numeric(df.get("launch_angle"), errors="coerce")
    ev = pd.to_numeric(df.get("launch_speed"), errors="coerce")
    bt = pd.Series(np.nan, index=df.index)
    bt[la < 10] = 0                       # ground ball
    bt[(la >= 10) & (la < 50)] = 1        # air ball
    bt[la >= 50] = 2                      # pop-up
    eb = (ev >= 95).astype(float); eb[ev.isna()] = np.nan
    cell = pd.Series(np.nan, index=df.index)
    gbair = bt.isin([0, 1])
    cell[gbair] = bt[gbair] * 2 + eb[gbair]
    cell[bt.eq(2)] = 4.0                   # pop-up, no EV split
    return cell


def bucket_values(df: pd.DataFrame) -> dict:
    """Count-adjusted run values for the valued swing outcomes (whiff, foul)."""
    dd = df["description"].fillna("").astype(str)
    ca = _count_adjusted_rv(df)
    isw = dd.isin(WHIFF_DESCS); isf = dd.isin(FOUL_DESCS)

    def cav(mask):
        v = ca[mask & ca.notna()]
        return float(v.mean()) if len(v) else 0.0

    return {"whiff": cav(isw), "foul": cav(isf)}


def train_global_model(df: pd.DataFrame, V: dict) -> dict:
    """Train the one global model: swing softmax {whiff,foul,in-play} + 5-cell contact grid."""
    dd = df["description"].fillna("").astype(str)
    isw = dd.isin(WHIFF_DESCS); isf = dd.isin(FOUL_DESCS); isip = dd.eq(INPLAY_DESC)
    sf = df[SHAPE_FEATS].notna().all(axis=1)
    rng = np.random.RandomState(42)

    # --- swing softmax: 0 whiff, 1 foul, 2 in-play ---
    lab = pd.Series(-1, index=df.index)
    lab[isw] = 0; lab[isf] = 1; lab[isip] = 2
    swi = df.index[(isw | isf | isip) & sf]
    if len(swi) > _SAMPLE_SWING:
        swi = pd.Index(rng.choice(swi, _SAMPLE_SWING, replace=False))
    swing = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                               n_estimators=_N_SWING, **_LGBM_SWING)
    swing.fit(df.loc[swi, SHAPE_FEATS].values, lab.loc[swi].values)

    # --- 5-cell contact grid + per-cell run values ---
    ca = _count_adjusted_rv(df)
    cell = _grid_cell(df)
    ipall = isip & sf & cell.notna() & ca.notna()
    Vcell = np.array([ca[ipall & (cell == c)].mean() if int((ipall & (cell == c)).sum()) > 0
                      else np.nan for c in range(5)])
    Vcell = np.where(np.isfinite(Vcell), Vcell, float(ca[ipall].mean()))
    gi = df.index[ipall]
    if len(gi) > _SAMPLE_GRID:
        gi = pd.Index(rng.choice(gi, _SAMPLE_GRID, replace=False))
    grid = lgb.LGBMClassifier(objective="multiclass", n_estimators=_N_GRID, **_LGBM_GRID)
    grid.fit(df.loc[gi, SHAPE_FEATS].values, cell.loc[gi].astype(int).values)

    ens = {"method": "global_swing_grid", "feats": SHAPE_FEATS, "shape_feats": SHAPE_FEATS,
           "swing": swing, "grid": grid, "grid_classes": grid.classes_,
           "Vcell": Vcell, "V_whiff": V["whiff"], "V_foul": V["foul"], "weights": V}

    samp = df.loc[sf, SHAPE_FEATS].sample(n=min(_SAMPLE_NORM, int(sf.sum())), random_state=42)
    e = predict_global_rv(samp, ens)
    ens["norm"] = {"mean": float(np.nanmean(e)), "std": float(np.nanstd(e) + 1e-8)}
    logger.info(
        f"  global: whiff={V['whiff']:+.4f} foul={V['foul']:+.4f}  "
        f"grid RV=[{', '.join(f'{n}={v:+.3f}' for n, v in zip(_CELL_NAMES, Vcell))}]  "
        f"norm mean={ens['norm']['mean']:+.5f} std={ens['norm']['std']:.5f}  "
        f"(swings n={len(swi):,}, in-play n={len(gi):,})")
    return ens


def predict_global_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    """Expected run value from the global model: P(wh)·Vwh + P(foul)·Vf + P(ip)·E[contact RV]."""
    if len(df) == 0:
        return np.full(0, np.nan)
    X = df[SHAPE_FEATS].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).values
    P = ens["swing"].predict_proba(X)          # cols: 0 whiff, 1 foul, 2 in-play
    Pg = ens["grid"].predict_proba(X)
    cdmg = Pg @ ens["Vcell"][ens["grid_classes"]]   # E[contact RV | in-play]
    return P[:, 0] * ens["V_whiff"] + P[:, 1] * ens["V_foul"] + P[:, 2] * cdmg


# --- back-compat aliases: callers still import these names ---
def predict_family_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    """Back-compat shim — the model is now one global model, so this scores globally."""
    return predict_global_rv(df, ens)


def predict_prob_resid_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    return predict_global_rv(df, ens)


def _batter_zone(df: pd.DataFrame) -> pd.Series:
    px = pd.to_numeric(df.get("plate_x"), errors="coerce")
    pz = pd.to_numeric(df.get("plate_z"), errors="coerce")
    szt = pd.to_numeric(df.get("sz_top"), errors="coerce")
    szb = pd.to_numeric(df.get("sz_bot"), errors="coerce")
    inz = (px.abs().values <= ZONE_HALF) & (pz.values >= szb.values) & (pz.values <= szt.values)
    return pd.Series(np.where(pz.notna().values, inz, False), index=df.index).fillna(False)
