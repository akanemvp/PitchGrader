"""
LightGBM ensemble for Stuff+ — swing-conditional outcome tree (Fangraphs/PitchingBot architecture).

Sub-models:
  whiff      LGBMClassifier  — P(whiff | swing)         swing pitches, stuff only
  foul       LGBMClassifier  — P(foul  | contact)       contact pitches (swings - whiffs), stuff only
  joint      LGBMClassifier  — P(bb_type, ev_bucket)    15 classes (3 BIP × 5 EV), stuff only
  decision   LGBMClassifier  — P(swing)                 all pitches, stuff+loc+count (not used for Stuff+)
  cs         LGBMClassifier  — P(called_strike|take)    take pitches, stuff+loc (not used for Stuff+)

Stuff+ xRV (swing-conditional):
  xRV = P(whiff|sw)×rv_whiff + P(bip|sw)×rv_joint_contact
  where P(bip|sw) = (1 - P(whiff|sw)) × (1 - P(foul|contact))
  rv_foul = 0.0 always (Fangraphs: fouls are neutral)

rv_table uses count-adjusted RV: ca_rv = delta_run_exp - mean(delta_run_exp | count)
This centers each count at 0, naturally producing rv_whiff≈-0.235, rv_ball≈+0.127, rv_bip≈0.

Joint BIP model: 3 types (GB=0, LD=1, FB=2 — popup merged into FB) × 5 EV buckets = 15 classes.
"""

import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from config import MODEL_DIR, RANDOM_STATE, TEST_SIZE
from features.engineering import CORE_FEATURES

logger = logging.getLogger(__name__)

XWOBA_COL  = "estimated_woba_using_speedangle"
WOBA_SCALE = 1.15

SWING_DESCS   = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
    "foul", "foul_tip", "bunt_foul_tip", "foul_bunt", "hit_into_play",
}
WHIFF_DESCS   = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOUL_DESCS    = {"foul", "foul_tip", "bunt_foul_tip", "foul_bunt"}
CONTACT_DESCS = {"hit_into_play"}

MODEL_FEATURES = CORE_FEATURES

LGBM_CLASSIFIER_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=150,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

LGBM_REGRESSOR_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=150,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

MIN_SAMPLES = 200
CATEGORICAL_FEATURES = ["pitch_type_code"]


BALL_OUTCOME_DESCS = {"ball", "blocked_ball", "pitchout", "intent_ball", "hit_by_pitch"}

def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    desc = df["description"].fillna("")
    df = df.copy()
    df["is_swing"]   = desc.isin(SWING_DESCS).astype(int)
    df["is_whiff"]   = desc.isin(WHIFF_DESCS).astype(int)
    df["is_foul"]    = desc.isin(FOUL_DESCS).astype(int)
    df["is_contact"] = desc.isin(CONTACT_DESCS).astype(int)
    df["is_take"]    = (~desc.isin(SWING_DESCS)).astype(int)
    df["is_cs"]      = (desc == "called_strike").astype(int)
    df["is_ball"]    = desc.isin(BALL_OUTCOME_DESCS).astype(int)
    return df


def _group_split(df: pd.DataFrame):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    groups = df["pitcher"].fillna(-1) if "pitcher" in df.columns else pd.Series(range(len(df)))
    idx_tr, idx_te = next(gss.split(df, groups=groups))
    return df.iloc[idx_tr], df.iloc[idx_te]


def _avail(features, df):
    return [f for f in features if f in df.columns]


def _cat_feats(features):
    return [f for f in CATEGORICAL_FEATURES if f in features]


def _xwoba_to_rv(xwoba_series: pd.Series, rv_baselines: dict) -> pd.Series:
    lg  = rv_baselines.get("lg_xwoba_con", 0.370)
    scl = rv_baselines.get("woba_scale",   WOBA_SCALE)
    mcr = rv_baselines.get("mean_contact_rv", 0.05)
    return (xwoba_series - lg) / scl + mcr


def compute_rv_baselines(df: pd.DataFrame) -> dict:
    """Compute count-adjusted RV baselines (Fangraphs approach).

    Count-adjusted RV: ca_rv = delta_run_exp - mean(delta_run_exp | balls, strikes).
    This centers the average outcome at each count to 0, which naturally produces:
      rv_whiff ≈ -0.235  (whiffs are much better than average outcome)
      rv_ball  ≈ +0.127  (balls are much worse than average outcome)
      rv_foul  = 0.0     (fouls are neutral — explicitly set per Fangraphs)
      rv_bip   ≈ 0.0     (average contact is roughly neutral after centering)
    """
    df = _add_flags(df)
    baselines = {}

    # Count-adjusted RV: subtract per-count mean from delta_run_exp
    balls_col   = pd.to_numeric(df.get("balls",   0), errors="coerce").fillna(0).astype(int)
    strikes_col = pd.to_numeric(df.get("strikes", 0), errors="coerce").fillna(0).astype(int)
    dre = pd.to_numeric(df.get("delta_run_exp", np.nan), errors="coerce")
    tmp = pd.DataFrame({"balls": balls_col, "strikes": strikes_col, "dre": dre})
    count_mean = tmp.groupby(["balls", "strikes"])["dre"].transform("mean")
    ca_rv = dre - count_mean

    def _ca_mean(mask: pd.Series) -> float:
        vals = ca_rv[mask & dre.notna() & ca_rv.notna()]
        return float(vals.mean()) if len(vals) > 0 else 0.0

    baselines["rv_whiff"] = _ca_mean(df["is_whiff"] == 1)
    baselines["rv_ball"]  = _ca_mean(df["is_ball"]  == 1)
    baselines["rv_cs"]    = _ca_mean(df["is_cs"]    == 1)
    baselines["rv_foul"]  = 0.0  # Fangraphs: fouls are explicitly neutral

    logger.info(
        f"  rv_baseline rv_whiff={baselines['rv_whiff']:.5f}  "
        f"rv_ball={baselines['rv_ball']:.5f}  "
        f"rv_cs={baselines['rv_cs']:.5f}  rv_foul=0.0  (count-adjusted)"
    )

    contact_rows = df[df["is_contact"] == 1].copy()
    contact_ca   = ca_rv[df["is_contact"] == 1]
    if XWOBA_COL in contact_rows.columns and contact_rows[XWOBA_COL].notna().sum() > MIN_SAMPLES:
        lg_xwoba_con    = float(contact_rows[XWOBA_COL].dropna().mean())
        mean_contact_rv = float(contact_ca.dropna().mean()) if contact_ca.notna().any() else 0.0
    else:
        lg_xwoba_con    = 0.370
        mean_contact_rv = float(contact_ca.dropna().mean()) if len(contact_rows) else 0.0

    baselines["lg_xwoba_con"]    = lg_xwoba_con
    baselines["mean_contact_rv"] = mean_contact_rv
    baselines["woba_scale"]      = WOBA_SCALE
    baselines["rv_contact"]      = mean_contact_rv
    logger.info(
        f"  rv_baseline contact: lg_xwoba_con={lg_xwoba_con:.4f}  "
        f"mean_contact_rv={mean_contact_rv:.5f}  woba_scale={WOBA_SCALE}"
    )
    return baselines


def _train_classifier(df_tr, df_te, features, target, label):
    X_tr = df_tr[features]; y_tr = df_tr[target]
    X_te = df_te[features]; y_te = df_te[target]
    cats = _cat_feats(features)
    m = lgb.LGBMClassifier(**LGBM_CLASSIFIER_PARAMS)
    m.fit(
        X_tr, y_tr,
        categorical_feature=cats or "auto",
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    ll = log_loss(y_te, m.predict_proba(X_te)[:, 1])
    logger.info(f"    [{label}] log_loss={ll:.4f}  n_tr={len(df_tr):,}")
    return m


def _train_multiclass(df_tr, df_te, features, target, label):
    """Train a multiclass LightGBM classifier. Returns model and n_classes."""
    X_tr = df_tr[features]; y_tr = df_tr[target]
    X_te = df_te[features]; y_te = df_te[target]
    cats = _cat_feats(features)
    params = {**LGBM_CLASSIFIER_PARAMS, "objective": "multiclass", "num_class": int(y_tr.nunique())}
    m = lgb.LGBMClassifier(**params)
    m.fit(
        X_tr, y_tr,
        categorical_feature=cats or "auto",
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    ll = log_loss(y_te, m.predict_proba(X_te))
    logger.info(f"    [{label}] log_loss={ll:.4f}  n_tr={len(df_tr):,}  classes={list(sorted(y_tr.unique()))}")
    return m


def _train_regressor(df_tr, df_te, features, target, label):
    X_tr = df_tr[features]; y_tr = df_tr[target]
    X_te = df_te[features]; y_te = df_te[target]
    cats = _cat_feats(features)
    m = lgb.LGBMRegressor(**LGBM_REGRESSOR_PARAMS)
    m.fit(
        X_tr, y_tr,
        categorical_feature=cats or "auto",
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    rmse = np.sqrt(mean_squared_error(y_te, m.predict(X_te)))
    logger.info(f"    [{label}] RMSE={rmse:.5f}  n_tr={len(df_tr):,}")
    return m


# Location/count features used by swing and CS models during training.
# At inference these are held at neutral so scores reflect pure stuff quality.
SWING_LOCATION_FEATURES = ["plate_x", "plate_z", "balls", "strikes"]
CS_LOCATION_FEATURES    = ["plate_x", "plate_z"]
NEUTRAL_BALLS   = 1
NEUTRAL_STRIKES = 1
NEUTRAL_PLATE_X = 0.0
NEUTRAL_PLATE_Z = 2.3


# BIP type classes (index must be stable — used to index rv_bip array at inference)
# Fangraphs/PitchingBot: 3 types only (GB, LD, FB) — popup is merged into fly_ball
BIP_CLASSES   = {"ground_ball": 0, "line_drive": 1, "fly_ball": 2}
BIP_CLASS_INV = {v: k for k, v in BIP_CLASSES.items()}

# Exit-velocity buckets for joint BIP×EV run-value lookup (Fangraphs/PitchingBot approach)
EV_BUCKETS   = [0.0, 90.0, 95.0, 100.0, 105.0, 999.0]   # 5 bins
N_EV_BUCKETS = len(EV_BUCKETS) - 1                        # 5


def _ev_bucket(launch_speed: pd.Series) -> pd.Series:
    """Bin exit velocity into 5 buckets: 0=<90, 1=90-95, 2=95-100, 3=100-105, 4=≥105."""
    return pd.cut(
        launch_speed.clip(0, 250),
        bins=EV_BUCKETS,
        labels=False,
        right=False,
        include_lowest=True,
    ).fillna(0).astype(int)


N_BIP_CLASSES  = len(BIP_CLASSES)              # 3
N_JOINT        = N_BIP_CLASSES * N_EV_BUCKETS  # 15  (3 BIP × 5 EV)

def _joint_class(bb_idx: int, ev_idx: int) -> int:
    return bb_idx * N_EV_BUCKETS + ev_idx

def _align_joint_proba(proba: np.ndarray, model) -> np.ndarray:
    """Expand predict_proba to full N_JOINT=15 columns, padding absent classes with 0."""
    classes = list(model.classes_) if hasattr(model, "classes_") else list(range(proba.shape[1]))
    if len(classes) == N_JOINT:
        return proba
    out = np.zeros((len(proba), N_JOINT))
    for col_i, cls in enumerate(classes):
        if 0 <= cls < N_JOINT:
            out[:, cls] = proba[:, col_i]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return out / row_sums


def train_ensemble(df: pd.DataFrame, rv_baselines: dict, features: "list | None" = None) -> "dict | None":
    """Train Stuff+ model.

    Target = raw delta_run_exp with location + count regressed out (residuals).
    All pitches included (swings, takes, fouls, balls, called strikes, in-play).
    No sub-models, no swing conditioning, no xwOBA component.
    """
    df = _add_flags(df)
    df = df[df["delta_run_exp"].notna() & np.isfinite(df["delta_run_exp"])].copy()

    if len(df) < MIN_SAMPLES:
        logger.warning(f"Too few rows ({len(df)}) — skipping")
        return None

    # Regress out location + count from delta_run_exp; residuals = pure stuff RV
    df["target_rv"] = compute_location_regressed_rv(df)

    # Drop rows with no target assignment
    df = df[df["target_rv"].notna()].copy()
    if len(df) < MIN_SAMPLES:
        logger.warning(f"Too few rows with target_rv ({len(df)}) — skipping")
        return None

    feats = _avail(features if features is not None else MODEL_FEATURES, df)

    tr, te = _group_split(df)
    logger.info(f"  Training direct RV regressor (n={len(tr):,}) [location+count-regressed RV]")
    rv_m = _train_regressor(tr, te, feats, "target_rv", "rv")

    # Compute norm on training data
    train_pred = rv_m.predict(tr[feats])
    valid = np.isfinite(train_pred)
    norm = {
        "raw_mean": float(train_pred[valid].mean()),
        "raw_std":  float(train_pred[valid].std()),
        "target_mean": 100.0,
        "target_std":  10.0,
    }
    logger.info(f"  Norm: raw_mean={norm['raw_mean']:.5f}  raw_std={norm['raw_std']:.5f}")

    return {
        "rv_model":    rv_m,
        "whiff_feats": feats,   # kept for compat — same features used for prediction
        "norm":        norm,
    }


def _align_ev_proba(proba: np.ndarray, model) -> np.ndarray:
    """Expand predict_proba output to full N_EV_BUCKETS columns.

    If some EV bucket classes were absent from training data, LightGBM's
    multiclass model will have fewer than N_EV_BUCKETS columns.  This
    function pads missing columns with zeros so the output is always (n, 5).
    """
    classes = list(model.classes_) if hasattr(model, "classes_") else list(range(proba.shape[1]))
    if len(classes) == N_EV_BUCKETS:
        return proba
    out = np.zeros((len(proba), N_EV_BUCKETS))
    for col_i, cls in enumerate(classes):
        if 0 <= cls < N_EV_BUCKETS:
            out[:, cls] = proba[:, col_i]
    # Re-normalise rows that have probability mass (don't touch all-zero rows)
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return out / row_sums


def _align_bip_proba(proba: np.ndarray, model) -> np.ndarray:
    """Expand predict_proba output to full len(BIP_CLASSES) columns."""
    n_bip = len(BIP_CLASSES)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(range(proba.shape[1]))
    if len(classes) == n_bip:
        return proba
    out = np.zeros((len(proba), n_bip))
    for col_i, cls in enumerate(classes):
        if 0 <= cls < n_bip:
            out[:, col_i] = proba[:, col_i]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return out / row_sums


def predict_ensemble_rv(
    df: pd.DataFrame,
    ensemble: dict,
    rv_baselines: dict,
    v_whiff: "np.ndarray | float | None" = None,
    v_cs:    "np.ndarray | float | None" = None,
    v_ball:  "np.ndarray | float | None" = None,
    swing_only: bool = False,
) -> np.ndarray:
    """Predict Stuff+ xRV — TJ Stats architecture.

    Single LGBMRegressor predicts E[rv | stuff features] directly.
    Trained on count-adjusted outcome RV for all pitch outcomes.
    """
    rv_m = ensemble.get("rv_model")

    if rv_m is None or len(df) == 0:
        return np.full(len(df), np.nan)

    feats  = ensemble.get("whiff_feats", [])
    avail  = [f for f in feats if f in df.columns]
    if not avail:
        return np.full(len(df), np.nan)

    return rv_m.predict(df[avail])


# ---------------------------------------------------------------------------
# Count-stratified RV target (TJ Stats approach)
# ---------------------------------------------------------------------------

# Map Statcast description → canonical event name for lookup
_DESC_TO_EVENT = {
    "ball": "ball", "blocked_ball": "ball", "pitchout": "ball", "intent_ball": "ball",
    "called_strike": "called_strike",
    "swinging_strike": "swinging_strike", "swinging_strike_blocked": "swinging_strike",
    "missed_bunt": "swinging_strike",
    "foul": "foul", "foul_tip": "foul", "foul_bunt": "foul", "bunt_foul_tip": "foul",
    "hit_by_pitch": "hit_by_pitch",
    "hit_into_play": None,  # use events column instead
}
_EVENTS_TO_EVENT = {
    "strikeout": "strikeout", "strikeout_double_play": "strikeout",
    "walk": "walk", "intent_walk": "walk",
    "hit_by_pitch": "hit_by_pitch",
    "single": "single", "double": "double", "triple": "triple", "home_run": "home_run",
    "field_out": "field_out", "force_out": "field_out", "grounded_into_double_play": "field_out",
    "double_play": "field_out", "triple_play": "field_out", "sac_fly": "field_out",
    "sac_bunt": "field_out", "fielders_choice": "field_out", "fielders_choice_out": "field_out",
    "sac_fly_double_play": "field_out", "other_out": "field_out",
}


def compute_count_rv_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Build (event, balls, strikes) → mean delta_run_exp lookup from training data."""
    rows = []
    desc = df.get("description", pd.Series("", index=df.index)).fillna("")
    events = df.get("events", pd.Series("", index=df.index)).fillna("")
    balls   = pd.to_numeric(df.get("balls",   0), errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(df.get("strikes", 0), errors="coerce").fillna(0).astype(int)
    rv = pd.to_numeric(df.get("delta_run_exp", np.nan), errors="coerce")

    event_col = pd.Series("", index=df.index)
    for d, e in _DESC_TO_EVENT.items():
        mask = desc == d
        if e is not None:
            event_col[mask] = e
        else:
            # hit_into_play: use events column
            event_col[mask] = events[mask].map(_EVENTS_TO_EVENT).fillna("field_out")
    # fallback: use events column for rows not matched by description
    unmatched = event_col == ""
    event_col[unmatched] = events[unmatched].map(_EVENTS_TO_EVENT).fillna("")
    event_col[event_col == ""] = np.nan

    tmp = pd.DataFrame({"event": event_col, "balls": balls, "strikes": strikes, "rv": rv})
    tmp = tmp[tmp["event"].notna() & tmp["rv"].notna()]
    lookup = tmp.groupby(["event", "balls", "strikes"])["rv"].mean().reset_index()
    lookup.columns = ["event", "balls", "strikes", "count_rv"]
    logger.info(f"  Count-stratified RV lookup: {len(lookup)} (event, balls, strikes) cells")
    return lookup


def compute_location_regressed_rv(df: pd.DataFrame) -> pd.Series:
    """Regress delta_run_exp on location + count variables using XGBoost, return residuals.

    Location variables: plate_x, plate_z (sz-normalized), balls, strikes,
    plus interaction/context features matching the Pitch Profiler architecture.
    Residuals = location/count-neutral RV — pure pitch quality contribution.
    """
    import xgboost as xgb

    rv = pd.to_numeric(df.get("delta_run_exp", np.nan), errors="coerce")
    plate_x  = pd.to_numeric(df.get("plate_x",  np.nan), errors="coerce")
    plate_z  = pd.to_numeric(df.get("plate_z",  np.nan), errors="coerce")
    sz_top   = pd.to_numeric(df.get("sz_top",   3.5),    errors="coerce").fillna(3.5)
    sz_bot   = pd.to_numeric(df.get("sz_bot",   1.5),    errors="coerce").fillna(1.5)
    balls    = pd.to_numeric(df.get("balls",    0),      errors="coerce").fillna(0)
    strikes  = pd.to_numeric(df.get("strikes",  0),      errors="coerce").fillna(0)

    sz_height    = (sz_top - sz_bot).clip(lower=0.5)
    sz_mid       = (sz_top + sz_bot) / 2
    plate_z_norm = (plate_z - sz_mid) / sz_height

    # Distance from center of strike zone
    dist_from_center = np.sqrt(plate_x ** 2 + plate_z_norm ** 2)
    # Distance from strike zone edges (negative = in zone)
    edge_x = (plate_x.abs() - 0.83).clip(lower=0)
    edge_z = (plate_z_norm.abs() - 0.5).clip(lower=0)

    loc = pd.DataFrame({
        "plate_x":          plate_x,
        "plate_z_norm":     plate_z_norm,
        "balls":            balls,
        "strikes":          strikes,
        "dist_from_center": dist_from_center,
        "edge_x":           edge_x,
        "edge_z":           edge_z,
        "plate_x_sq":       plate_x ** 2,
        "plate_z_sq":       plate_z_norm ** 2,
        "xz_interact":      plate_x * plate_z_norm,
        "count_interact":   balls * strikes,
    })

    valid = rv.notna() & np.isfinite(rv) & loc.notna().all(axis=1)
    result = pd.Series(np.nan, index=df.index)

    if valid.sum() < 1000:
        logger.warning("Not enough valid rows for location regression — falling back to raw RV")
        result[valid] = rv[valid] - rv[valid].mean()
        return result

    X = loc[valid].values
    y = rv[valid].values

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    preds = model.predict(X)
    residuals = y - preds
    result[valid] = residuals

    r2 = 1 - np.var(residuals) / np.var(y)
    logger.info(f"  Location+ XGBoost R²={r2:.4f}  n={valid.sum():,}  residual_std={residuals.std():.5f}")
    return result


def apply_count_rv_target(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.Series:
    """Assign each pitch its count-stratified average RV from the lookup table."""
    desc   = df.get("description", pd.Series("", index=df.index)).fillna("")
    events = df.get("events",      pd.Series("", index=df.index)).fillna("")
    balls   = pd.to_numeric(df.get("balls",   0), errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(df.get("strikes", 0), errors="coerce").fillna(0).astype(int)

    event_col = pd.Series("", index=df.index)
    for d, e in _DESC_TO_EVENT.items():
        mask = desc == d
        if e is not None:
            event_col[mask] = e
        else:
            event_col[mask] = events[mask].map(_EVENTS_TO_EVENT).fillna("field_out")
    unmatched = event_col == ""
    event_col[unmatched] = events[unmatched].map(_EVENTS_TO_EVENT).fillna("")
    event_col[event_col == ""] = np.nan

    tmp = pd.DataFrame({"event": event_col, "balls": balls, "strikes": strikes}, index=df.index)
    tmp = tmp.merge(lookup, on=["event", "balls", "strikes"], how="left")
    return tmp["count_rv"].values


# ---------------------------------------------------------------------------
# Linear weights target (context-neutral per-pitch RV)
# ---------------------------------------------------------------------------

BALL_DESCS = {"ball", "blocked_ball", "pitchout", "intent_ball"}
HBP_DESCS  = {"hit_by_pitch"}

# ---------------------------------------------------------------------------
# SIERA-calibrated target
# ---------------------------------------------------------------------------

# Partial derivatives of SIERA at league-average rates (k=0.225, bb=0.085, ng=0.031)
# Per-pitch run value = -∂SIERA/∂x / 38  (38 PA per 9 innings)
# K:  -(-11.585)/38 = +0.305   BB: -(+12.514)/38 = -0.329
# GB vs avg BIP: +1.057/38 = +0.028   FB vs avg BIP: -0.028
# Base BIP: solve 0 = P(K)*K + P(BB)*BB + P(BIP)*BIP → BIP = -0.059
# Sign convention matches delta_run_exp: negative = good for pitcher, positive = bad.
_SIERA_K    = -0.305   # strikeout    (very good for pitcher)
_SIERA_BB   =  0.329   # walk / HBP   (very bad for pitcher)
_SIERA_GB   =  0.031   # ground ball  (slightly bad vs avg PA, but best BIP type)
_SIERA_FB   =  0.087   # fly ball     (worst BIP — HR risk)
_SIERA_LD   =  0.059   # line drive   (average BIP, no SIERA skill credit)
_SIERA_PU   =  0.031   # popup        (same as GB — near-automatic out, no HR)

_K_EVENTS   = {"strikeout", "strikeout_double_play"}
_BB_EVENTS  = {"walk", "hit_by_pitch", "intent_walk"}
_BB_DESCS   = {"ball", "blocked_ball", "pitchout", "intent_ball", "hit_by_pitch"}


def compute_siera_rv_target(df: pd.DataFrame) -> pd.Series:
    """SIERA-calibrated per-pitch run value target.

    Terminal outcomes get fixed SIERA-derived weights (strips BABIP luck and
    situational context). Non-terminal pitches keep delta_run_exp so count-
    transition value is preserved for the location model to absorb.
    """
    desc   = df["description"].fillna("") if "description" in df.columns else pd.Series("", index=df.index)
    events = df["events"].fillna("")      if "events"      in df.columns else pd.Series("", index=df.index)
    bbt    = df["bb_type"].fillna("")     if "bb_type"     in df.columns else pd.Series("", index=df.index)

    rv = df["delta_run_exp"].copy() if "delta_run_exp" in df.columns else pd.Series(np.nan, index=df.index)

    # Terminal: strikeout
    rv[events.isin(_K_EVENTS)] = _SIERA_K

    # Terminal: walk / HBP
    rv[events.isin(_BB_EVENTS)] = _SIERA_BB

    # Terminal: batted ball — assign by bb_type
    bip_mask = desc == "hit_into_play"
    rv[bip_mask & (bbt == "ground_ball")] = _SIERA_GB
    rv[bip_mask & (bbt == "fly_ball")]    = _SIERA_FB
    rv[bip_mask & (bbt == "line_drive")]  = _SIERA_LD
    rv[bip_mask & (bbt == "popup")]       = _SIERA_PU
    # BIP with missing bb_type → use average BIP
    rv[bip_mask & ~bbt.isin({"ground_ball", "fly_ball", "line_drive", "popup"})] = -0.059

    n_terminal = (events.isin(_K_EVENTS) | events.isin(_BB_EVENTS) | bip_mask).sum()
    n_nonterminal = rv.notna().sum() - n_terminal
    logger.info(
        f"  siera_rv: {n_terminal:,} terminal pitches / {n_nonterminal:,} non-terminal  "
        f"mean={rv.mean():.5f}  std={rv.std():.5f}"
    )
    return rv

def compute_linear_weights_target(df: pd.DataFrame, rv_baselines: dict) -> pd.Series:
    """Assign each pitch a context-neutral run value based purely on its outcome type.

    Non-contact outcomes get fixed global baselines (count/base-out neutral).
    Contact outcomes use xwOBA → rv so quality of contact is captured.
    This removes ALL game-context contamination from the training signal.
    """
    desc = df["description"].fillna("") if "description" in df.columns else pd.Series("", index=df.index)
    rv = pd.Series(np.nan, index=df.index)

    rv_whiff   = rv_baselines.get("rv_whiff",   -0.114)
    rv_foul    = rv_baselines.get("rv_foul",    -0.039)
    rv_cs      = rv_baselines.get("rv_cs",      -0.065)
    rv_ball    = rv_baselines.get("rv_ball",     0.061)
    rv_contact = rv_baselines.get("rv_contact",  0.052)

    rv[desc.isin(WHIFF_DESCS)]   = rv_whiff
    rv[desc.isin(FOUL_DESCS)]    = rv_foul
    rv[desc == "called_strike"]  = rv_cs
    rv[desc.isin(BALL_DESCS)]    = rv_ball
    rv[desc.isin(HBP_DESCS)]     = rv_ball  # treat HBP same as ball for pitcher

    # Contact: use xwOBA-based rv if available, else fallback to rv_contact baseline
    contact_mask = desc.isin(CONTACT_DESCS)
    if XWOBA_COL in df.columns:
        xwoba_rv = _xwoba_to_rv(df.loc[contact_mask, XWOBA_COL], rv_baselines)
        rv[contact_mask] = xwoba_rv.values
    else:
        rv[contact_mask] = rv_contact

    return rv


# ---------------------------------------------------------------------------
# Swing quality target (whiff + weak contact only — count/take neutral)
# ---------------------------------------------------------------------------

def compute_swing_quality_target(df: pd.DataFrame, rv_baselines: dict) -> pd.Series:
    """Assign each SWING pitch its swing-outcome run value.

    Target = is_whiff × rv_whiff + is_foul × rv_foul + is_contact × xwoba_rv
    Take pitches (balls, called strikes) get NaN — they are excluded from training.

    This removes count contamination: P(swing) is count-driven, but given a swing
    happened, whiff/foul/contact outcomes are far less count-dependent.
    Using xwOBAcon for contact removes base-out state luck from batted ball outcomes.
    """
    desc = df["description"].fillna("") if "description" in df.columns else pd.Series("", index=df.index)
    rv = pd.Series(np.nan, index=df.index)

    rv_whiff   = rv_baselines.get("rv_whiff",  -0.114)
    rv_foul    = rv_baselines.get("rv_foul",   -0.039)
    rv_contact = rv_baselines.get("rv_contact",  0.052)

    rv[desc.isin(WHIFF_DESCS)] = rv_whiff
    rv[desc.isin(FOUL_DESCS)]  = rv_foul

    contact_mask = desc.isin(CONTACT_DESCS)
    if XWOBA_COL in df.columns and df.loc[contact_mask, XWOBA_COL].notna().sum() > 100:
        xwoba_rv = _xwoba_to_rv(df.loc[contact_mask, XWOBA_COL], rv_baselines)
        rv[contact_mask] = xwoba_rv.values
    else:
        rv[contact_mask] = rv_contact

    n_swings = rv.notna().sum()
    n_total  = len(df)
    logger.info(f"  Swing quality target: {n_swings:,} swing pitches / {n_total:,} total ({n_swings/n_total*100:.1f}%)")
    return rv


# ---------------------------------------------------------------------------
# Residual xRV model (Location+ approach)
# ---------------------------------------------------------------------------

def train_residual_model(df: pd.DataFrame, features: list, target_col: str = "residual_xrv") -> "dict | None":
    """Single LightGBM regressor trained on residual_xrv (delta_run_exp minus location effect)."""
    avail_feats = _avail(features, df)
    valid = df[target_col].notna() & np.isfinite(df[target_col])
    df_valid = df[valid].copy()

    if len(df_valid) < MIN_SAMPLES:
        logger.warning(f"  Too few residual samples: {len(df_valid)}")
        return None

    train_df, test_df = _group_split(df_valid)
    X_tr = train_df[avail_feats].fillna(0)
    y_tr = train_df[target_col]
    X_te = test_df[avail_feats].fillna(0)
    y_te = test_df[target_col]

    cat_feats = [f for f in ["pitch_type_code"] if f in avail_feats]

    model = lgb.LGBMRegressor(**LGBM_REGRESSOR_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        categorical_feature=cat_feats if cat_feats else "auto",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    rmse = float(np.sqrt(mean_squared_error(y_te, model.predict(X_te))))
    logger.info(f"  [residual] RMSE={rmse:.5f}  n_tr={len(train_df):,}")
    return {"residual": model, "features": avail_feats}


def predict_residual_rv(df: pd.DataFrame, ensemble: dict) -> np.ndarray:
    """Predict raw residual xRV from physics features."""
    model = ensemble.get("residual")
    if model is None or len(df) == 0:
        return np.full(len(df), np.nan)
    feats = _avail(ensemble.get("features", []), df)
    if not feats:
        return np.full(len(df), np.nan)
    return model.predict(df[feats].fillna(0))


# ---------------------------------------------------------------------------
# Whiff classifier — P(whiff) direct binary classification
# ---------------------------------------------------------------------------

LGBM_WHIFF_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=100,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


def train_whiff_model(df: pd.DataFrame, features: list) -> "dict | None":
    """Train a single LGBMClassifier directly on is_whiff (all pitches).

    Returns {"whiff_clf": model, "features": feats, "norm": {mean, std}}
    where norm is computed on the training predictions so scoring can
    normalize P(whiff) to the 100/10 Stuff+ scale.
    """
    df = df.copy()
    df["is_whiff"] = df["description"].fillna("").isin(WHIFF_DESCS).astype(int)

    avail_feats = _avail(features, df)
    valid = df[avail_feats].notna().all(axis=1) | True  # keep all rows, fillna below
    df_valid = df.copy()

    if len(df_valid) < MIN_SAMPLES:
        logger.warning(f"  Too few samples: {len(df_valid)}")
        return None

    train_df, test_df = _group_split(df_valid)
    X_tr = train_df[avail_feats].fillna(0)
    y_tr = train_df["is_whiff"]
    X_te = test_df[avail_feats].fillna(0)
    y_te = test_df["is_whiff"]

    model = lgb.LGBMClassifier(**LGBM_WHIFF_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    ll = log_loss(y_te, model.predict_proba(X_te))
    logger.info(f"  [whiff_clf] log_loss={ll:.4f}  n_tr={len(train_df):,}")

    # Compute norm on full training set P(whiff) predictions
    p_whiff = model.predict_proba(X_tr)[:, 1]
    norm = {
        "raw_mean": float(p_whiff.mean()),
        "raw_std":  float(p_whiff.std()),
    }
    logger.info(f"  [whiff_clf] norm: mean={norm['raw_mean']:.4f}  std={norm['raw_std']:.4f}")

    return {"whiff_clf": model, "features": avail_feats, "norm": norm}


def predict_whiff_prob(df: pd.DataFrame, ensemble: dict) -> np.ndarray:
    """Predict P(whiff) from the whiff classifier. Returns raw probability."""
    model = ensemble.get("whiff_clf")
    if model is None or len(df) == 0:
        return np.full(len(df), np.nan)
    feats = _avail(ensemble.get("features", []), df)
    if not feats:
        return np.full(len(df), np.nan)
    return model.predict_proba(df[feats].fillna(0))[:, 1]


# ---------------------------------------------------------------------------
# Count-neutral ensemble (swing residual approach)
# ---------------------------------------------------------------------------

SWING_BASELINE_FEATURES = ["balls", "strikes", "stand_r"]
SWING_QUALITY_EXTRA     = ["plate_x", "plate_z"]   # added on top of CORE_FEATURES for swing submodel


def train_count_neutral_ensemble(
    df: pd.DataFrame,
    rv_baselines: dict,
    features: "list | None" = None,
) -> "dict | None":
    """Multi-target ensemble with count-neutral swing probability.

    Step 1: Baseline P(swing | balls, strikes, stand_r) — captures count/platoon context.
    Step 2: swing_residual = is_swing - expected_swing — pitch quality component of swing decisions.
    Step 3: Quality swing regressor on CORE_FEATURES + plate_x + plate_z → predicts swing_residual.
    Step 4: whiff / foul / contact_rv / cs trained same as standard ensemble (already more neutral,
            since they condition on an event already occurring).

    At prediction time:
        p_swing = clip(avg_swing_rate + swing_quality_model.predict(features), 0, 1)
    """
    df = _add_flags(df)
    df = df[df["delta_run_exp"].notna() & np.isfinite(df["delta_run_exp"])].copy()

    feats = _avail(features if features is not None else MODEL_FEATURES, df)

    # Swing features = quality features + plate location
    swing_extra = _avail(SWING_QUALITY_EXTRA, df)
    swing_feats = feats + [f for f in swing_extra if f not in feats]

    # --- Step 1: Baseline swing model (count + stand only) ---
    baseline_feats = _avail(SWING_BASELINE_FEATURES, df)
    if not baseline_feats:
        logger.warning("Baseline features missing — falling back to standard ensemble")
        return train_ensemble(df, rv_baselines, features=features)

    tr_all, te_all = _group_split(df)
    logger.info(f"  Training swing baseline  (n={len(df):,})")
    baseline_m = _train_classifier(tr_all, te_all, baseline_feats, "is_swing", "baseline/swing")

    # --- Step 2: Compute swing residual on full training set ---
    df["expected_swing"]  = baseline_m.predict_proba(df[baseline_feats])[:, 1]
    df["swing_residual"]  = df["is_swing"] - df["expected_swing"]
    avg_swing_rate        = float(df["expected_swing"].mean())
    logger.info(f"  avg_swing_rate={avg_swing_rate:.4f}  swing_residual mean={df['swing_residual'].mean():.5f}")

    # Re-split after residual computation (same random state → same split)
    tr_all, te_all = _group_split(df)

    # --- Step 3: Quality swing regressor ---
    avail_swing = _avail(swing_feats, df)
    logger.info(f"  Training swing quality   (n={len(df):,})  feats={len(avail_swing)}")
    swing_quality_m = _train_regressor(tr_all, te_all, avail_swing, "swing_residual", "quality/swing")

    # --- Step 4: whiff / foul / contact_rv / cs (same as standard ensemble) ---
    swing_df   = df[df["is_swing"]   == 1].copy()
    take_df    = df[df["is_take"]    == 1].copy()
    contact_df = df[df["is_contact"] == 1].copy()

    tr_sw, te_sw = _group_split(swing_df)
    logger.info(f"  Training whiff model     (n={len(swing_df):,})")
    whiff_m = _train_classifier(tr_sw, te_sw, feats, "is_whiff", "quality/whiff")

    logger.info(f"  Training foul model      (n={len(swing_df):,})")
    foul_m  = _train_classifier(tr_sw, te_sw, feats, "is_foul",  "quality/foul")

    contact_tr = contact_df.loc[contact_df.index.isin(tr_sw.index)]
    contact_te = contact_df.loc[contact_df.index.isin(te_sw.index)]
    if len(contact_tr) < MIN_SAMPLES:
        contact_tr = contact_df.sample(frac=0.8, random_state=RANDOM_STATE)
        contact_te = contact_df.drop(contact_tr.index)

    if XWOBA_COL in contact_tr.columns and contact_tr[XWOBA_COL].notna().sum() > MIN_SAMPLES:
        contact_tr = contact_tr.copy()
        contact_te = contact_te.copy()
        contact_tr["contact_rv_target"] = _xwoba_to_rv(contact_tr[XWOBA_COL], rv_baselines)
        contact_te["contact_rv_target"] = _xwoba_to_rv(contact_te[XWOBA_COL], rv_baselines)
        contact_target = "contact_rv_target"
    else:
        contact_tr = contact_tr.copy()
        contact_te = contact_te.copy()
        contact_tr["contact_rv_target"] = contact_tr["delta_run_exp"]
        contact_te["contact_rv_target"] = contact_te["delta_run_exp"]
        contact_target = "contact_rv_target"

    logger.info(f"  Training contact model   (n={len(contact_tr):,})")
    contact_m = _train_regressor(
        contact_tr[feats + [contact_target]].dropna(),
        contact_te[feats + [contact_target]].dropna(),
        feats, contact_target, "quality/contact_rv",
    )

    tr_take, te_take = _group_split(take_df)
    logger.info(f"  Training cs model        (n={len(take_df):,})")
    cs_m = _train_classifier(tr_take, te_take, feats, "is_cs", "quality/cs")

    # Compute raw xRV on training set for norm
    rv_whiff_bl = rv_baselines.get("rv_whiff", -0.114)
    rv_foul_bl  = rv_baselines.get("rv_foul",  -0.039)
    rv_cs_bl    = rv_baselines.get("rv_cs",    -0.065)
    rv_ball_bl  = rv_baselines.get("rv_ball",   0.059)

    avail_f = _avail(feats, tr_all)
    avail_s = _avail(avail_swing, tr_all)

    p_swing_adj = swing_quality_m.predict(tr_all[avail_s])
    p_swing     = np.clip(avg_swing_rate + p_swing_adj, 0.0, 1.0)
    p_take      = 1.0 - p_swing
    p_whiff     = whiff_m.predict_proba(tr_all[avail_f])[:, 1]
    p_foul      = foul_m.predict_proba(tr_all[avail_f])[:, 1]
    p_contact   = (1.0 - p_whiff - p_foul).clip(0.0)
    rv_con      = contact_m.predict(tr_all[avail_f])
    p_cs        = cs_m.predict_proba(tr_all[avail_f])[:, 1]
    p_ball      = 1.0 - p_cs

    train_rv = (
        p_swing * (p_whiff * rv_whiff_bl + p_foul * rv_foul_bl + p_contact * rv_con)
        + p_take * (p_cs * rv_cs_bl + p_ball * rv_ball_bl)
    )
    valid = np.isfinite(train_rv)
    norm = {
        "raw_mean": float(train_rv[valid].mean()),
        "raw_std":  float(train_rv[valid].std()),
        "target_mean": 100.0,
        "target_std":  10.0,
    }
    logger.info(f"  Norm: raw_mean={norm['raw_mean']:.5f}  raw_std={norm['raw_std']:.5f}")

    return {
        "swing_baseline":    baseline_m,
        "swing_baseline_feats": baseline_feats,
        "swing_quality":     swing_quality_m,
        "swing_quality_feats": avail_swing,
        "avg_swing_rate":    avg_swing_rate,
        "whiff":             whiff_m,
        "foul":              foul_m,
        "contact_rv":        contact_m,
        "cs":                cs_m,
        "whiff_feats":       feats,
        "foul_feats":        feats,
        "contact_feats":     feats,
        "norm":              norm,
    }


def predict_count_neutral_rv(df: pd.DataFrame, ensemble: dict, rv_baselines: dict) -> np.ndarray:
    """Predict count-neutral xRV from count-neutral ensemble."""
    swing_q_m  = ensemble.get("swing_quality")
    whiff_m    = ensemble.get("whiff")
    foul_m     = ensemble.get("foul")
    contact_m  = ensemble.get("contact_rv")
    cs_m       = ensemble.get("cs")
    avg_swing  = ensemble.get("avg_swing_rate", 0.47)

    if whiff_m is None or len(df) == 0:
        return np.full(len(df), np.nan)

    feats       = _avail(ensemble.get("whiff_feats", []), df)
    swing_feats = _avail(ensemble.get("swing_quality_feats", feats), df)

    if not feats:
        return np.full(len(df), np.nan)

    rv_whiff_bl = rv_baselines.get("rv_whiff", -0.114)
    rv_foul_bl  = rv_baselines.get("rv_foul",  -0.039)
    rv_cs_bl    = rv_baselines.get("rv_cs",    -0.065)
    rv_ball_bl  = rv_baselines.get("rv_ball",   0.059)

    # Count-neutral swing probability
    if swing_q_m is not None:
        p_swing_adj = swing_q_m.predict(df[swing_feats])
        p_swing     = np.clip(avg_swing + p_swing_adj, 0.0, 1.0)
    else:
        p_swing = np.full(len(df), avg_swing)
    p_take = 1.0 - p_swing

    p_whiff   = whiff_m.predict_proba(df[feats])[:, 1]
    p_foul    = foul_m.predict_proba(df[feats])[:, 1] if foul_m else np.full(len(df), 0.20)
    p_contact = (1.0 - p_whiff - p_foul).clip(0.0)
    rv_con    = contact_m.predict(df[feats]) if contact_m else np.full(len(df), rv_baselines.get("mean_contact_rv", 0.05))
    p_cs      = cs_m.predict_proba(df[feats])[:, 1] if cs_m else np.full(len(df), 0.35)
    p_ball    = 1.0 - p_cs

    return (
        p_swing * (p_whiff * rv_whiff_bl + p_foul * rv_foul_bl + p_contact * rv_con)
        + p_take * (p_cs * rv_cs_bl + p_ball * rv_ball_bl)
    )


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

ENSEMBLE_KEY = "all"


def save_ensemble(ensemble: dict, family: str = ENSEMBLE_KEY) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"ensemble_{family}.pkl")
    with open(path, "wb") as f:
        pickle.dump(ensemble, f)


def load_ensemble(family: str = ENSEMBLE_KEY) -> "dict | None":
    path = os.path.join(MODEL_DIR, f"ensemble_{family}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_rv_baselines(baselines: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "rv_baselines.pkl")
    with open(path, "wb") as f:
        pickle.dump(baselines, f)


def load_rv_baselines() -> "dict | None":
    path = os.path.join(MODEL_DIR, "rv_baselines.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
