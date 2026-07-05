"""
Stuff+ inference — single global model, global normalization.

One LightGBM ensemble (whiff / decision / called-strike sub-models) trained on all
pitch types. Grades are globally comparable: 100 = average across all pitch types,
std=10. No per-type or per-family re-normalization.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR
from features.engineering import engineer_features, apply_movement_rv, OS_FEATURES
from model.submodels import predict_residual_rv, predict_ensemble_rv, predict_count_neutral_rv, predict_prostuff_rv, predict_prostuff_paper_rv, predict_prostuff_paper_contact_rv, predict_paper_hbbe_rv, predict_residual_stuff_rv, predict_swing_outcome_rv, load_ensemble, load_rv_baselines
from model.bam_shape_v2 import predict_bam_shape_v2
from model.nn_shape import predict_nn_shape
from model.swing_tree import predict_swing_tree_rv
from model.tj_locresid import predict_tj_locresid_rv
from model.prob_resid import predict_prob_resid_rv
from model.rv_locresid import predict_rv_locresid_rv
from model.grl_model import predict_grl_rv, predict_grl_tree_rv

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
        self.ensembles:               dict = {}
        self.rv_baselines:            dict = {}
        self.norm_global:             dict = {}
        self.norm_global_historical:  dict = {}
        self.norm_family:             dict = {}
        self.norm_family_historical:  dict = {}
        self.norm_per_type:           dict = {}
        self.norm_per_type_historical: dict = {}
        self.baselines                     = None
        self.location_grid:           list = []  # list of (plate_x, plate_z) cells
        self.location_weights:        dict = {}  # pitch_type → np.ndarray of weights summing to 1
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
            _gm = self.norm_global.get('mean', self.norm_global.get('global_mean', 0.0))
            logger.info(f"Global norm loaded: mean={_gm:.5f}")

        hg_path = os.path.join(MODEL_DIR, "norm_global_historical.pkl")
        if os.path.exists(hg_path):
            with open(hg_path, "rb") as f:
                self.norm_global_historical = pickle.load(f)
            _hgm = self.norm_global_historical.get('mean', self.norm_global_historical.get('global_mean', 0.0))
            logger.info(f"Historical global norm loaded: mean={_hgm:.5f}")

        fn_path = os.path.join(MODEL_DIR, "norm_family.pkl")
        if os.path.exists(fn_path):
            with open(fn_path, "rb") as f:
                self.norm_family = pickle.load(f)
            logger.info(f"Family norms loaded: {list(self.norm_family.keys())}")

        fnh_path = os.path.join(MODEL_DIR, "norm_family_historical.pkl")
        if os.path.exists(fnh_path):
            with open(fnh_path, "rb") as f:
                self.norm_family_historical = pickle.load(f)
            logger.info(f"Historical family norms loaded")

        npt_path = os.path.join(MODEL_DIR, "norm_per_type.pkl")
        if os.path.exists(npt_path):
            with open(npt_path, "rb") as f:
                self.norm_per_type = pickle.load(f)
            logger.info(f"Per-type norms loaded: {list(self.norm_per_type.keys())}")

        npth_path = os.path.join(MODEL_DIR, "norm_per_type_historical.pkl")
        if os.path.exists(npth_path):
            with open(npth_path, "rb") as f:
                self.norm_per_type_historical = pickle.load(f)
            logger.info(f"Historical per-type norms loaded")


        rv_bl = load_rv_baselines()
        if rv_bl is not None:
            self.rv_baselines = rv_bl

        # Load ensemble FIRST, then check if it's GRL — if so, skip loading
        # location_regressors (LightGBM) which conflicts with torch on this stack.
        ens = load_ensemble("all")
        if ens is not None:
            self.ensembles["all"] = ens

        if ens is None:
            # No global model — load the fb/nfb family split (current production path).
            _loaded = 0
            for _fam in ("fb", "nfb"):
                _e = load_ensemble(_fam)
                if _e is not None:
                    self.ensembles[_fam] = _e
                    _loaded += 1
            if _loaded:
                logger.info(f"Family models loaded: {_loaded} (fb/nfb)")
            else:
                logger.warning("No model found — run 'python main.py train' first")

        _any_grl = any(e.get("grl") for e in self.ensembles.values())
        if _any_grl:
            # GRL models are NN-only at inference (use_tree is never set); no tree to load,
            # and LightGBM location_regressors conflict with torch on this stack.
            self.loc_regressors = None
        else:
            lr_path = os.path.join(MODEL_DIR, "location_regressors.pkl")
            if os.path.exists(lr_path):
                with open(lr_path, "rb") as f:
                    self.loc_regressors = pickle.load(f)
                logger.info(f"Location regressors loaded: feats={len(self.loc_regressors['feats'])}")
            else:
                self.loc_regressors = None

        if "all" in self.ensembles:
            logger.info("Global model loaded")



    def predict(self, df, baselines=None, already_engineered=False, norm_set="current"):
        if not already_engineered:
            bl = baselines if baselines is not None else self.baselines
            df, _ = engineer_features(df, baselines=bl)

        df = df.copy()
        df["stuff_plus"] = np.nan

        _TYPE_KEYS = ("fb", "nfb")

        # Determine model mode: fb/nfb family split or single global fallback
        _use_types = any(k in self.ensembles for k in _TYPE_KEYS)
        if not _use_types and "all" not in self.ensembles:
            logger.warning("Model not loaded — returning NaN")
            return df

        # Per-pitch family routing (FF/FA/SI -> fb, breaking/offspeed -> nfb, FC via Stage 0).
        # Computed once on df; routing is independent of same_hand so it's reused for both passes.
        if _use_types:
            from model.cutter_stage0 import route_to_model
            _fam_row = route_to_model(df).to_numpy()
        else:
            _fam_row = None

        _fam_norm_src = self.norm_family_historical if (norm_set == "historical" and self.norm_family_historical) else self.norm_family
        _global_src  = self.norm_global_historical if (norm_set == "historical" and self.norm_global_historical) else self.norm_global
        _global_mean = _global_src.get("mean", _global_src.get("global_mean", 0.0))
        _global_std  = max(_global_src.get("std",  _global_src.get("global_std",  0.007)), 1e-6)

        def _score_df(df_in):
            if _use_types:
                e_rv = np.full(len(df_in), np.nan)
                for fam in _TYPE_KEYS:
                    ens = self.ensembles.get(fam)
                    if ens is None:
                        continue
                    mask = _fam_row == fam
                    if not mask.any():
                        continue
                    sub = df_in[mask]
                    if ens.get("grl"):
                        e_rv[mask] = predict_grl_tree_rv(sub, ens) if ens.get("use_tree") else predict_grl_rv(sub, ens)
                    elif ens.get("prostuff_paper"):
                        e_rv[mask] = predict_prostuff_paper_rv(sub, ens)
                    elif ens.get("prostuff"):
                        e_rv[mask] = predict_prostuff_rv(sub, ens)
                    elif "residual" in ens:
                        e_rv[mask] = predict_residual_rv(sub, ens)
                    elif "swing_quality" in ens:
                        e_rv[mask] = predict_count_neutral_rv(sub, ens, self.rv_baselines)
                    else:
                        e_rv[mask] = predict_ensemble_rv(sub, ens, self.rv_baselines)
                return e_rv
            else:
                ens = self.ensembles["all"]
                if ens.get("grl"):
                    if ens.get("use_tree"):
                        return predict_grl_tree_rv(df_in, ens)
                    return predict_grl_rv(df_in, ens)
                if ens.get("swing_tree"):
                    return predict_swing_tree_rv(df_in, ens)
                if ens.get("tj_locresid"):
                    return predict_tj_locresid_rv(df_in, ens)
                if ens.get("prob_resid"):
                    return predict_prob_resid_rv(df_in, ens)
                if ens.get("rv_locresid"):
                    return predict_rv_locresid_rv(df_in, ens)
                if ens.get("nn_shape"):
                    return predict_nn_shape(df_in, ens)
                if ens.get("bam_shape_v2"):
                    return predict_bam_shape_v2(df_in, ens)
                if ens.get("swing_outcome"):
                    return predict_swing_outcome_rv(df_in, ens)
                if ens.get("residual_stuff"):
                    return predict_residual_stuff_rv(df_in, ens)
                if ens.get("hbbe"):
                    return predict_paper_hbbe_rv(df_in, ens)
                if ens.get("prostuff_paper_contact"):
                    return predict_prostuff_paper_contact_rv(df_in, ens)
                if ens.get("prostuff_paper"):
                    return predict_prostuff_paper_rv(df_in, ens)
                if ens.get("prostuff"):
                    return predict_prostuff_rv(df_in, ens)
                if "residual" in ens:
                    return predict_residual_rv(df_in, ens)
                if "swing_quality" in ens:
                    return predict_count_neutral_rv(df_in, ens, self.rv_baselines)
                return predict_ensemble_rv(df_in, ens, self.rv_baselines)

        def _normalize(e_rv, pt_series):
            """Normalize using one global norm (single scale across all pitch types & models)."""
            return 100.0 + (_global_mean - e_rv) / _global_std * 10.0

        # Inference path depends on model type:
        # - GRL model: location is adversarially decoupled in training; just one forward pass
        # - Otherwise (LightGBM): use predicted-location aggregation with location regressors
        ens = self.ensembles.get("all", {})
        _any_grl = any(e.get("grl") for e in self.ensembles.values())
        # Platoon-neutral scoring: every pitch is scored TWICE — once as if
        # facing a same-hand batter (same_hand=1.0) and once as if facing an
        # opposite-hand batter (same_hand=0.0) — and the two predictions are
        # averaged 50/50. This makes a pitcher's grade independent of the
        # platoon mix they actually happened to face.
        if _any_grl or ens.get("loc_grid") or ens.get("hbbe") or ens.get("residual_stuff") or ens.get("swing_outcome") or ens.get("bam_shape_v2") or ens.get("nn_shape") or ens.get("swing_tree") or ens.get("tj_locresid") or ens.get("prob_resid") or ens.get("rv_locresid"):
            df_same = df.copy(); df_same["same_hand"] = 1.0
            df_opp  = df.copy(); df_opp["same_hand"]  = 0.0
            rv_same = _score_df(df_same)
            rv_opp  = _score_df(df_opp)
            expected_rv = 0.5 * rv_same + 0.5 * rv_opp
            stuff_plus = _normalize(expected_rv, df["pitch_type"])
        elif self.loc_regressors is not None:
            lr = self.loc_regressors
            loc_feats = lr["feats"]
            df_same = df.copy(); df_same["same_hand"] = 1.0
            df_opp  = df.copy(); df_opp["same_hand"]  = 0.0
            px_same = lr["plate_x"].predict(df_same[loc_feats])
            pz_same = lr["plate_z"].predict(df_same[loc_feats])
            px_opp  = lr["plate_x"].predict(df_opp[loc_feats])
            pz_opp  = lr["plate_z"].predict(df_opp[loc_feats])
            df_same_loc = df.copy(); df_same_loc["plate_x"] = px_same; df_same_loc["plate_z"] = pz_same
            df_opp_loc  = df.copy(); df_opp_loc["plate_x"]  = px_opp;  df_opp_loc["plate_z"]  = pz_opp
            rv_same = _score_df(df_same_loc)
            rv_opp  = _score_df(df_opp_loc)
            expected_rv = 0.5 * rv_same + 0.5 * rv_opp
            stuff_plus = _normalize(expected_rv, df["pitch_type"])
        else:
            df_neutral = df.copy()
            df_neutral["plate_x"] = 0.0
            df_neutral["plate_z"] = 2.5
            df_same = df_neutral.copy(); df_same["same_hand"] = 1.0
            df_opp  = df_neutral.copy(); df_opp["same_hand"]  = 0.0
            expected_rv = 0.5 * _score_df(df_same) + 0.5 * _score_df(df_opp)
            stuff_plus = _normalize(expected_rv, df["pitch_type"])

        low_velo   = (df["release_speed"] < 75.0).values
        power_mask = df["pitch_type"].isin(_POWER_TYPES).values
        null_mask  = power_mask & low_velo
        stuff_plus[null_mask] = np.nan

        df["stuff_plus"] = np.clip(stuff_plus, -50.0, 200.0)
        # Raw stuff_plus stays as-is for aggregation; display grade compresses the
        # individual-pitch tail into a realistic ceiling (~135).
        df["stuff_plus_display"] = display_grade(df["stuff_plus"].values)
        return df


def load_baselines():
    path = os.path.join(MODEL_DIR, "movement_baselines.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No baselines at {path}. Run train first.")
    with open(path, "rb") as f:
        return pickle.load(f)
