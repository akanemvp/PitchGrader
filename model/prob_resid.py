"""Stuff+ — Method A: a two-zone conditional-outcome model.

A pitch's expected run value is split by where the shape tends to go, then each
zone is scored by its full swing/take outcome tree. The final value is the
shape-predicted-zone-rate blend of the two:

    xRV(shape) = P(in-zone | shape) · IN-ZONE value
               + P(out-zone | shape) · OUT-ZONE value

  IN-ZONE value  = P(swing | in-zone) · [whiff / foul / GB·AIR·PU·HR contact]
                 + P(take | in-zone)  · V_called_strike        (an in-zone take is a strike)

  OUT-ZONE value = P(chase | out-zone) · [whiff / foul / GB·AIR·PU·HR contact]
                 + P(no-chase | out-zone) · V_ball             (an out-zone take is a ball)

Everything is trained on the 8 arm-normalized SHAPE_FEATS — no location, count, or
handedness as *features*. Location is used only at TRAIN time, to label each pitch
in-zone vs out-of-zone against the individual hitter's strike zone; at inference the
seven heads are applied to shape alone. The zone weight P(in-zone | shape) is the
shape's population-average zone tendency (validated calibrated, AUC 0.58) — it is
command-neutral (trained across all pitchers) yet accurate (fastball ~0.51, slider
~0.42), which is why it credits called strikes and balls fairly without importing an
individual pitcher's command.

Contact is a 4-cell trajectory model — ground ball (LA<10), air/line-drive (10-50),
pop-up (>50), home run — valued per cell by count-adjusted run value, with chase
contact valued as weaker than in-zone contact.

Best-validated configuration of the session: reality corr -0.20 vs actual run value,
cross-type ordering -0.36, un-compressed fastballs and sliders, and it attaches
shape credit to takes (in-zone called strikes, out-of-zone balls) the fair way.

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
XWOBA_COL = "estimated_woba_using_speedangle"   # retained for import compatibility

# half plate-width + ball radius (ft): the horizontal edge of the rulebook zone
ZONE_HALF = 0.83

# LightGBM: strongly regularized so the velocity/extension response stays smooth into
# the sparse tail. path_smooth + large min_child_samples + extra_trees, no monotone
# constraints. deterministic + force_col_wise give reproducible grades at full speed.
_LGBM = dict(
    n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
    colsample_bytree=0.8, min_child_samples=2500, reg_lambda=5.0,
    path_smooth=5.0, min_split_gain=1e-5, extra_trees=True,
    random_state=42, n_jobs=-1, deterministic=True, force_col_wise=True, verbose=-1,
)

# in-play trajectory cells, in fixed order
_IP4 = ["GB", "AIR", "PU", "HR"]
_SW3 = ["foul", "whiff", "inplay"]


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


def _ip_cell(la: pd.Series, ev: pd.Series, mask: pd.Series) -> pd.Series:
    """GB / AIR / PU / HR label for in-play pitches under `mask`."""
    cell = pd.Series(index=la.index, dtype=object)
    m = mask & la.notna()
    hr = m & (ev == "home_run")
    rest = m & (ev != "home_run")
    cell[hr] = "HR"
    cell[rest & (la < 10)] = "GB"
    cell[rest & (la > 50)] = "PU"
    cell[rest & (la >= 10) & (la <= 50)] = "AIR"
    return cell


def train_prob_resid_ensemble(df: pd.DataFrame, rv_baselines: dict, features=None) -> dict:
    df = df.copy()
    dd = df["description"].fillna("").astype(str)
    ev = df["events"].fillna("").astype(str)
    la = pd.to_numeric(df.get("launch_angle"), errors="coerce")
    dre = pd.to_numeric(df["delta_run_exp"], errors="coerce")
    b = pd.to_numeric(df["balls"], errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(df["strikes"], errors="coerce").fillna(0).astype(int)
    # count-adjusted run value: subtract the per-count mean so values are count-neutral
    ca = dre - pd.DataFrame({"b": b, "s": s, "d": dre}).groupby(["b", "s"]).d.transform("mean")

    inz = _batter_zone(df); ooz = ~inz
    isw = dd.isin(WHIFF_DESCS); isf = dd.isin(FOUL_DESCS)
    iscs = dd.isin(CALLED_DESCS); isip = dd.eq(INPLAY_DESC) & la.notna()
    isswing = isw | isf | isip
    sf = df[SHAPE_FEATS].notna().all(axis=1)
    logger.info(f"  Method A: {int(inz.sum()):,} in-zone / {int(ooz.sum()):,} out-of-zone")

    def cav(mask):   # mean count-adjusted RV under a mask
        v = ca[mask & ca.notna()]
        return float(v.mean()) if len(v) else 0.0

    ip_iz = _ip_cell(la, ev, inz); ip_oz = _ip_cell(la, ev, ooz)
    Vi = {"foul": cav(inz & isf), "whiff": cav(inz & isw),
          **{c: cav(inz & (ip_iz == c)) for c in _IP4}}
    Vo = {"foul": cav(ooz & isf), "whiff": cav(ooz & isw),
          **{c: cav(ooz & (ip_oz == c)) for c in _IP4}}
    v_cs = cav(inz & iscs)              # in-zone take = called strike
    v_ball = cav(ooz & ~isswing)        # out-zone take = ball
    logger.info(f"  values: v_cs={v_cs:+.4f} v_ball={v_ball:+.4f}  "
                f"Vi.HR={Vi['HR']:+.3f} Vo.GB={Vo['GB']:+.3f}")

    def fit(mask, y, obj=None, k=None):
        kw = dict(_LGBM)
        if obj:
            kw.update(objective=obj, num_class=k)
        return lgb.LGBMClassifier(**kw).fit(df.loc[mask, SHAPE_FEATS].values, y[mask].values)

    idx3 = {c: i for i, c in enumerate(_SW3)}; idx4 = {c: i for i, c in enumerate(_IP4)}
    sw3 = pd.Series(index=df.index, dtype=object); sw3[isf] = "foul"; sw3[isw] = "whiff"; sw3[isip] = "inplay"

    H = {}
    H["zone"]     = fit(sf, inz.astype(int))                                              # P(in-zone | shape)
    H["izgate"]   = fit(inz & (isswing | iscs) & sf, isswing.astype(int))                 # P(swing | in-zone)
    H["iz_swing"] = fit(inz & isswing & sw3.notna() & sf, sw3.map(idx3), "multiclass", 3) # foul/whiff/inplay | iz swing
    H["iz_ip"]    = fit(inz & ip_iz.isin(_IP4) & sf, ip_iz.map(idx4), "multiclass", 4)    # GB/AIR/PU/HR | iz inplay
    H["chase"]    = fit(ooz & sf, isswing.astype(int))                                    # P(chase | out-zone)
    H["oz_swing"] = fit(ooz & isswing & sw3.notna() & sf, sw3.map(idx3), "multiclass", 3) # foul/whiff/inplay | chase
    H["oz_ip"]    = fit(ooz & ip_oz.isin(_IP4) & sf, ip_oz.map(idx4), "multiclass", 4)    # GB/AIR/PU/HR | oz inplay
    logger.info(f"  trained 7 heads on {int(sf.sum()):,} shaped pitches")

    ens = {"method": "A", "feats": SHAPE_FEATS, "shape_feats": SHAPE_FEATS,
           "heads": H, "Vi": Vi, "Vo": Vo, "v_cs": v_cs, "v_ball": v_ball,
           # legacy keys some callers probe:
           "groups": {"all": None}, "count_dist": [((0, 0), 1.0)]}

    samp = df.loc[sf, SHAPE_FEATS].sample(n=min(150000, int(sf.sum())), random_state=42)
    preds = predict_prob_resid_rv(samp, ens)
    ens["norm"] = {"raw_mean": float(np.nanmean(preds)), "raw_std": float(np.nanstd(preds)),
                   "target_mean": 100.0, "target_std": 10.0}
    logger.info(f"  Norm (n={len(samp):,}): xRV mean={ens['norm']['raw_mean']:+.5f} "
                f"std={ens['norm']['raw_std']:.5f}")
    return ens


def predict_prob_resid_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    """Method A expected run value from shape alone (7 heads composed)."""
    if len(df) == 0:
        return np.full(0, np.nan)
    H = ens["heads"]; Vi = ens["Vi"]; Vo = ens["Vo"]; v_cs = ens["v_cs"]; v_ball = ens["v_ball"]
    X = df[SHAPE_FEATS].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).values
    vi = np.array([Vi[c] for c in _IP4]); vo = np.array([Vo[c] for c in _IP4])

    psw = H["izgate"].predict_proba(X)[:, 1]
    s3i = H["iz_swing"].predict_proba(X); ipi = H["iz_ip"].predict_proba(X)
    in_swing = s3i[:, 0] * Vi["foul"] + s3i[:, 1] * Vi["whiff"] + s3i[:, 2] * (ipi @ vi)
    in_xrv = psw * in_swing + (1 - psw) * v_cs

    pch = H["chase"].predict_proba(X)[:, 1]
    s3o = H["oz_swing"].predict_proba(X); ipo = H["oz_ip"].predict_proba(X)
    oz_swing = s3o[:, 0] * Vo["foul"] + s3o[:, 1] * Vo["whiff"] + s3o[:, 2] * (ipo @ vo)
    oz_xrv = pch * oz_swing + (1 - pch) * v_ball

    w = H["zone"].predict_proba(X)[:, 1]
    return w * in_xrv + (1 - w) * oz_xrv
