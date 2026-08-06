"""Stuff+ — fastball / non-fastball split model with a SIERA-style contact head.

Two shape models — one for fastballs, one for everything else (cutters routed by a
Mahalanobis classifier) — each grade pitch shape by expected run value:

    xRV = P(whiff)·V_whiff + P(foul)·V_foul + P(in-play)·E[contact RV | in-play]

  • A swing softmax head predicts {whiff, foul, in-play} from pitch shape.
  • A SIERA-style contact head predicts the batted-ball type — ground ball, fly ball,
    pop-up — and values in-play contact as a·(P(GB) − P(FB) − P(PU)) + b. Line drives
    are excluded from training entirely: line-drive rate is a batter/luck outcome, not a
    repeatable pitcher skill. The contact score is calibrated to run value per group.

Each group is normalized on its OWN scale, so a fastball is graded against other
fastballs and a breaking ball against other breaking balls (100 = group-average pitch,
10 = one SD). This avoids burying fastballs under breaking balls on a single scale.

Shape is 10 arm-normalized features: velocity, spin rate, the Magnus and non-Magnus
(seam-shifted) vertical & arm-side components of the pitch's transverse acceleration,
arm angle, release side and height, and extension. The Magnus / non-Magnus split is
derived from the measured 3D spin axis (Alan Nathan's method) — so a pitch cannot be
scored until Statcast has filled in its per-pitch spin_axis. No location, count, or
game-state; a lefty and righty throwing physically identical pitches grade identically.

Lower xRV = better. Features are RobustScaler-standardized before the trees.
"""
from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

from config import MODEL_DIR

logger = logging.getLogger(__name__)

BALL_DESCS   = {"ball", "blocked_ball", "hit_by_pitch", "pitchout"}
CALLED_DESCS = {"called_strike"}
WHIFF_DESCS  = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOUL_DESCS   = {"foul", "foul_tip", "bunt_foul_tip", "foul_bunt"}
INPLAY_DESC  = "hit_into_play"

# 10 arm-normalized shape features. The four mag_/nonmag_ columns are the Magnus and
# non-Magnus (seam-shifted-wake) vertical & arm-side parts of the transverse accel,
# derived from the measured spin axis by add_shape_features. The rest are direct.
SHAPE_FEATS = [
    "release_speed", "release_spin_rate",
    "mag_vert", "mag_horiz_arm", "nonmag_vert", "nonmag_horiz_arm",
    "arm_angle", "release_pos_x_arm", "release_pos_z", "release_extension",
]
# Raw kinematic columns add_shape_features consumes to build the Magnus split.
KINEMATIC_COLS = ["vx0", "vy0", "vz0", "ax", "ay", "az",
                  "release_extension", "spin_axis", "release_pos_x", "p_throws"]
# Cutter-router space: velocity + arm-relative movement + spin + slot.
ROUTER_FEATS = ["release_speed", "ind_vert", "ind_horiz_arm", "release_spin_rate", "arm_angle"]

# Family membership by Statcast pitch_type. Cutters (FC) are routed by the router.
FB_TYPES  = {"FF", "FA", "SI"}
BR_TYPES  = {"SL", "ST", "SV", "SC", "GY", "CU", "KC", "CS", "SLV"}
OFF_TYPES = {"CH", "FO", "FS", "EP", "KN"}
FAMILIES  = ("FB", "OFF", "BR")
GROUPS    = ("FB", "NONFB")   # two scoring groups; cutters routed by the classifier

ZONE_HALF = 0.83   # half plate-width + ball radius (ft)
_G = 32.174

_SPIN_OFFSET_PATH = os.path.join(MODEL_DIR, "spin_offset.pkl")   # per-hand spin-axis convention offset

# Both heads: base LightGBM defaults + linear_tree (leaf linear models extrapolate the
# velocity tail cleanly). RobustScaler is applied to the features beforehand.
_LGBM = dict(learning_rate=0.1, linear_tree=True, n_jobs=-1, verbose=-1, random_state=42)
_N_SWING, _N_GRID = 500, 300
_SAMPLE_SWING, _SAMPLE_GRID, _SAMPLE_NORM = 2_500_000, 1_500_000, 300_000

_CELL_NAMES = ["GB", "FB", "PU"]


def add_magnus(df: pd.DataFrame) -> pd.DataFrame:
    """Add induced-Magnus accel components from raw kinematics (feeds the cutter router).
    Idempotent. ind_vert / ind_horiz(_arm) are ROUTER_FEATS only — scoring uses SHAPE_FEATS.
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


def _transverse_accel(df: pd.DataFrame):
    """Alan Nathan transverse (Magnus-frame) acceleration components aTx, aTz, and the
    arm sign, from raw 9P kinematics. Drag is projected out along the ball's velocity."""
    hs = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0).values
    vx0 = pd.to_numeric(df.get("vx0"), errors="coerce").values
    vy0 = pd.to_numeric(df.get("vy0"), errors="coerce").values
    vz0 = pd.to_numeric(df.get("vz0"), errors="coerce").values
    ax  = pd.to_numeric(df.get("ax"), errors="coerce").values
    ay  = pd.to_numeric(df.get("ay"), errors="coerce").values
    az  = pd.to_numeric(df.get("az"), errors="coerce").values
    ext = pd.to_numeric(df.get("release_extension"), errors="coerce").clip(4, 8).values
    with np.errstate(invalid="ignore", divide="ignore"):
        yR = 60.5 - ext
        tR = (-vy0 - np.sqrt(np.clip(vy0 ** 2 - 2 * ay * (50 - yR), 0, None))) / ay
        vxr = vx0 + ax * tR; vyr = vy0 + ay * tR; vzr = vz0 + az * tR
        tc = (-vyr - np.sqrt(np.clip(vyr ** 2 - 2 * ay * (yR - 17 / 12), 0, None))) / ay
        vxb = vxr + 0.5 * ax * tc; vyb = vyr + 0.5 * ay * tc; vzb = vzr + 0.5 * az * tc
        vb = np.sqrt(vxb ** 2 + vyb ** 2 + vzb ** 2)
        adrag = -(ax * vxb + ay * vyb + (az + _G) * vzb) / vb
        aTx = ax + adrag * vxb / vb
        aTz = az + adrag * vzb / vb + _G
    return aTx, aTz, hs


def fit_spin_offset(df: pd.DataFrame) -> dict:
    """Per-hand spin-axis convention offset = circular mean(spin_axis − movement_dir) over
    four-seamers, whose movement is pure Magnus. Saved so inference can align a single
    pitcher's spin axis to the movement frame. Returns {'R': deg, 'L': deg}."""
    aTx, aTz, _ = _transverse_accel(df)
    sa = pd.to_numeric(df.get("spin_axis"), errors="coerce").values
    mdir = np.degrees(np.arctan2(aTz, aTx)) % 360
    ff = (df["pitch_type"].astype(str) == "FF").values
    throws = df["p_throws"].astype(str).values
    off = {}
    for h in ("R", "L"):
        mk = ff & (throws == h) & np.isfinite(sa) & np.isfinite(mdir)
        if mk.sum() < 100:
            off[h] = 97.0 if h == "R" else 83.0        # documented fallback
            continue
        dev = np.radians(sa[mk] - mdir[mk])
        off[h] = float(np.degrees(np.arctan2(np.nanmean(np.sin(dev)), np.nanmean(np.cos(dev)))))
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(_SPIN_OFFSET_PATH, "wb") as fh:
        pickle.dump(off, fh)
    global _SPIN_OFFSET
    _SPIN_OFFSET = off
    return off


_SPIN_OFFSET: dict | None = None


def _spin_offset() -> dict:
    global _SPIN_OFFSET
    if _SPIN_OFFSET is None:
        if os.path.exists(_SPIN_OFFSET_PATH):
            with open(_SPIN_OFFSET_PATH, "rb") as fh:
                _SPIN_OFFSET = pickle.load(fh)
        else:
            _SPIN_OFFSET = {"R": 97.0, "L": 83.0}       # documented fallback
    return _SPIN_OFFSET


def add_shape_features(df: pd.DataFrame, spin_off: dict | None = None) -> pd.DataFrame:
    """Add the 10-feature shape columns the scoring model needs. Idempotent.

    Splits the transverse acceleration into Magnus (spin) and non-Magnus (seam-shifted)
    vertical & arm-side components using the measured spin_axis and the saved per-hand
    convention offset. release_pos_x_arm is the arm-side release point. Pitches missing
    spin_axis get NaN Magnus features and stay unscorable (no card until Statcast fills
    the 3D spin axis in).
    """
    off = spin_off or _spin_offset()
    aTx, aTz, hs = _transverse_accel(df)
    sa = pd.to_numeric(df.get("spin_axis"), errors="coerce").values
    throws = df["p_throws"].astype(str).values
    off_row = np.where(throws == "L", off.get("L", 83.0), off.get("R", 97.0))
    phiM = np.radians(sa - off_row)
    ux, uz = np.cos(phiM), np.sin(phiM)
    proj = aTx * ux + aTz * uz
    magx, magz = proj * ux, proj * uz
    nmx, nmz = aTx - magx, aTz - magz
    df["mag_vert"] = magz
    df["mag_horiz_arm"] = magx * hs
    df["nonmag_vert"] = nmz
    df["nonmag_horiz_arm"] = nmx * hs
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
    when a pitcher/name column is present, else per-pitch."""
    pt = df["pitch_type"].astype(str)
    fam = pt.map(_basefam)
    fc = pt.eq("FC")
    if fc.any() and router is not None and all(c in df.columns for c in ROUTER_FEATS):
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


def _assign_group(df: pd.DataFrame, router: dict | None) -> pd.Series:
    """Collapse the FB/OFF/BR family to two scoring groups — FB and NONFB — with cutters
    routed to whichever they resemble. Anything unclassified defaults to NONFB."""
    fam = assign_family(df, router)
    return fam.map({"FB": "FB", "OFF": "NONFB", "BR": "NONFB"}).fillna("NONFB")


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
    """SIERA-style batted-ball label by launch angle — 0 ground ball (<10°),
    1 fly ball (25–50°), 2 pop-up (≥50°). Line drives (10–25°) are left NaN and excluded
    from training: line-drive rate is a batter/luck outcome, not a pitcher skill."""
    la = pd.to_numeric(df.get("launch_angle"), errors="coerce")
    cell = pd.Series(np.nan, index=df.index)
    cell[la < 10] = 0                     # ground ball
    cell[(la >= 25) & (la < 50)] = 1      # fly ball
    cell[la >= 50] = 2                    # pop-up
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


def _siera_score(grid, Xg: np.ndarray) -> np.ndarray:
    """SIERA in-play skill score = P(GB) − P(FB) − P(PU) from a fitted 3-class grid head."""
    Pg = grid.predict_proba(Xg)
    cls = list(grid.classes_)
    return Pg[:, cls.index(0)] - Pg[:, cls.index(1)] - Pg[:, cls.index(2)]


def _fit_group(sub: pd.DataFrame, scaler: RobustScaler, isw, isf, isip, cell, ca, rng) -> dict:
    """Train one group's swing softmax {whiff,foul,in-play} + SIERA GB/FB/PU contact head,
    and calibrate the contact score to run value (RV = a·(GB−FB−PU) + b)."""
    sf = sub[SHAPE_FEATS].notna().all(axis=1)
    Xs = scaler.transform(sub[SHAPE_FEATS].values)

    lab = pd.Series(-1, index=sub.index)
    lab[isw] = 0; lab[isf] = 1; lab[isip] = 2
    swi = np.where(((isw | isf | isip) & sf).values)[0]
    if len(swi) > _SAMPLE_SWING:
        swi = rng.choice(swi, _SAMPLE_SWING, replace=False)
    swing = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=_N_SWING, **_LGBM)
    swing.fit(Xs[swi], lab.values[swi])

    ipall = (isip & sf & cell.notna() & ca.notna()).values
    gi = np.where(ipall)[0]
    if len(gi) > _SAMPLE_GRID:
        gi = rng.choice(gi, _SAMPLE_GRID, replace=False)
    grid = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=_N_GRID, **_LGBM)
    grid.fit(Xs[gi], cell.values[gi].astype(int))

    score = _siera_score(grid, Xs[gi])
    a, b = np.polyfit(score, ca.values[gi], 1)      # SIERA score -> count-adjusted run value
    return {"swing": swing, "grid": grid, "a": float(a), "b": float(b),
            "n_swings": int(len(swi)), "n_inplay": int(len(gi))}


def train_split_model(df: pd.DataFrame, V: dict, router: dict) -> dict:
    """Two group models (fastball / non-fastball, cutters routed) — each a swing softmax
    + SIERA GB/FB/PU contact head on RobustScaler features — normalized on SEPARATE
    per-group scales so fastballs grade vs fastballs and non-fastballs vs non-fastballs."""
    dd = df["description"].fillna("").astype(str)
    isw = dd.isin(WHIFF_DESCS); isf = dd.isin(FOUL_DESCS); isip = dd.eq(INPLAY_DESC)
    sf = df[SHAPE_FEATS].notna().all(axis=1)
    grp = _assign_group(df, router)
    cell = _grid_cell(df)
    ca = _count_adjusted_rv(df)
    rng = np.random.RandomState(42)

    scaler = RobustScaler().fit(df.loc[sf, SHAPE_FEATS].values)

    ens = {"method": "two_group_split_siera", "feats": SHAPE_FEATS, "shape_feats": SHAPE_FEATS,
           "scaler": scaler, "router": router, "spin_offset": _spin_offset(),
           "V_whiff": V["whiff"], "V_foul": V["foul"], "weights": V, "groups": {}}
    for gname in GROUPS:
        gm = (grp == gname)
        sub = df[gm]
        ens["groups"][gname] = _fit_group(
            sub, scaler, isw[gm], isf[gm], isip[gm], cell[gm], ca[gm], rng)

    # --- SEPARATE per-group scales (100 = group-average, 10 = one group SD) ---
    for gname in GROUPS:
        idx = df.index[(grp == gname) & sf]
        if len(idx) > _SAMPLE_NORM:
            idx = pd.Index(np.random.RandomState(42).choice(idx.values, _SAMPLE_NORM, replace=False))
        e = predict_group_rv(df.loc[idx], ens)          # full rows (needs pitch_type + router feats)
        ens["groups"][gname]["norm"] = {"mean": float(np.nanmean(e)), "std": float(np.nanstd(e) + 1e-8)}

    logger.info(f"  two-group split (separate scales): whiff={V['whiff']:+.4f} foul={V['foul']:+.4f}")
    for gname, g in ens["groups"].items():
        logger.info(f"    [{gname}] RV={g['a']:+.4f}*(GB-FB-PU){g['b']:+.4f}  "
                    f"norm mean={g['norm']['mean']:+.5f} std={g['norm']['std']:.5f}  "
                    f"(swings n={g['n_swings']:,}, in-play n={g['n_inplay']:,})")
    return ens


def predict_group_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    """Expected run value, routing each pitch to its group model:
    P(wh)*Vwh + P(foul)*Vf + P(in-play)*(a*(GB-FB-PU) + b). Raw xRV (un-normalized).

    Only pitches with all 10 shape features present are scored — the four Magnus/non-Magnus
    features need spin_axis, so a pitch with no 3D spin axis returns NaN (no card)."""
    out = np.full(len(df), np.nan)
    if len(df) == 0:
        return out
    X = df[SHAPE_FEATS].apply(pd.to_numeric, errors="coerce").values
    ok = np.isfinite(X).all(axis=1)
    if not ok.any():
        return out
    Xs = ens["scaler"].transform(X[ok])
    grp = _assign_group(df, ens.get("router")).values[ok]
    Vw, Vf = ens["V_whiff"], ens["V_foul"]
    scored = np.full(int(ok.sum()), np.nan)
    for gname, g in ens["groups"].items():
        rows = (grp == gname)
        if not rows.any():
            continue
        Xg = Xs[rows]
        P = g["swing"].predict_proba(Xg)                     # 0 whiff, 1 foul, 2 in-play
        rv_contact = g["a"] * _siera_score(g["grid"], Xg) + g["b"]
        scored[rows] = P[:, 0] * Vw + P[:, 1] * Vf + P[:, 2] * rv_contact
    out[ok] = scored
    return out


def grade_pitches(df: pd.DataFrame, ens: dict, norm_set: str = "current") -> np.ndarray:
    """xRV routed to each group, then z-scored on that group's OWN scale:
    100 = group-average pitch, 10 = one group SD (lower xRV -> higher grade)."""
    if len(df) == 0:
        return np.full(0, np.nan)
    rv = predict_group_rv(df, ens)
    grp = _assign_group(df, ens.get("router")).values
    out = np.full(len(df), np.nan)
    for gname, g in ens["groups"].items():
        rows = (grp == gname) & np.isfinite(rv)
        if not rows.any():
            continue
        n = g.get("norm_hist") if (norm_set == "historical" and g.get("norm_hist")) else g["norm"]
        out[rows] = 100.0 + (n["mean"] - rv[rows]) / max(n["std"], 1e-6) * 10.0
    return out


# --- back-compat aliases: external callers still import these names ---
def predict_global_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    return predict_group_rv(df, ens)


def predict_family_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    return predict_group_rv(df, ens)


def _batter_zone(df: pd.DataFrame) -> pd.Series:
    px = pd.to_numeric(df.get("plate_x"), errors="coerce")
    pz = pd.to_numeric(df.get("plate_z"), errors="coerce")
    szt = pd.to_numeric(df.get("sz_top"), errors="coerce")
    szb = pd.to_numeric(df.get("sz_bot"), errors="coerce")
    inz = (px.abs().values <= ZONE_HALF) & (pz.values >= szb.values) & (pz.values <= szt.values)
    return pd.Series(np.where(pz.notna().values, inz, False), index=df.index).fillna(False)
