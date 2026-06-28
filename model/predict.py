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
from features.engineering import engineer_features, apply_movement_rv, apply_fb_context, OS_FEATURES
from model.submodels import predict_residual_rv, predict_ensemble_rv, predict_count_neutral_rv, load_ensemble, load_rv_baselines

logger = logging.getLogger(__name__)

_POWER_TYPES = {"FF", "FA", "SI", "FC", "ST"}


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
        self.fb_context_lookup:       dict = {}
        self.baselines                     = None
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

        ens = load_ensemble("all")
        if ens is not None:
            self.ensembles["all"] = ens
            logger.info("Global model loaded")
        else:
            _loaded = 0
            for _fam in ("fb", "br", "os"):
                _ens = load_ensemble(_fam)
                if _ens is not None:
                    self.ensembles[_fam] = _ens
                    _loaded += 1
            if _loaded:
                logger.info(f"Family models loaded: {_loaded}/3 (fb/br/os)")
            else:
                logger.warning("No model found — run 'python main.py train' first")

        fb_ctx_path = os.path.join(MODEL_DIR, "fb_context_lookup.pkl")
        if os.path.exists(fb_ctx_path):
            with open(fb_ctx_path, "rb") as f:
                self.fb_context_lookup = pickle.load(f)
            logger.info(f"FB context lookup loaded ({len(self.fb_context_lookup)} pitchers)")


    def predict(self, df, baselines=None, already_engineered=False, norm_set="current"):
        if not already_engineered:
            bl = baselines if baselines is not None else self.baselines
            df, _ = engineer_features(df, baselines=bl)

        df = df.copy()
        df["stuff_plus"]        = np.nan
        df["stuff_plus_vs_rhb"] = np.nan
        df["stuff_plus_vs_lhb"] = np.nan

        _TYPE_GROUPS = {"fb": ["FF","FA","SI","FC"], "br": ["SL","SV","ST","SC","GY","CU","KC","CS"], "os": ["CH","FO","EP","KN","FS"]}
        _TYPE_KEYS = ("fb", "br", "os")

        # Determine model mode: 3-family or single global fallback
        _use_types = any(k in self.ensembles for k in _TYPE_KEYS)
        if not _use_types and "all" not in self.ensembles:
            logger.warning("Model not loaded — returning NaN")
            return df

        _fam_norm_src = self.norm_family_historical if (norm_set == "historical" and self.norm_family_historical) else self.norm_family
        _global_src  = self.norm_global_historical if (norm_set == "historical" and self.norm_global_historical) else self.norm_global
        _global_mean = _global_src.get("mean", _global_src.get("global_mean", 0.0))
        _global_std  = max(_global_src.get("std",  _global_src.get("global_std",  0.007)), 1e-6)

        # Apply FB context features (sets fb_velo per pitcher)
        if self.fb_context_lookup:
            df, _ = apply_fb_context(df, lookup=self.fb_context_lookup)
        elif "fb_ivb_adj" not in df.columns:
            df["fb_ivb_adj"] = 0.0
            df["fb_hb_adj"]  = 0.0
            df["fb_velo"]    = df.get("release_speed", pd.Series(93.0, index=df.index))

        def _score_df(df_in):
            if _use_types:
                e_rv = np.full(len(df_in), np.nan)
                pt_arr = df_in["pitch_type"].fillna("").values
                for fam, pts in _TYPE_GROUPS.items():
                    ens = self.ensembles.get(fam)
                    if ens is None:
                        continue
                    mask = np.isin(pt_arr, pts)
                    if mask.any():
                        e_rv[mask] = predict_ensemble_rv(df_in[mask], ens, self.rv_baselines)
                return e_rv
            else:
                ens = self.ensembles["all"]
                if "residual" in ens:
                    return predict_residual_rv(df_in, ens)
                return predict_ensemble_rv(df_in, ens, self.rv_baselines)

        def _normalize(e_rv, pt_series):
            """Normalize using global norm (single scale across all pitch types)."""
            if _fam_norm_src:
                grades = np.full(len(e_rv), np.nan)
                pt_arr = pt_series.values if hasattr(pt_series, "values") else np.asarray(pt_series)
                for fam, pts in _TYPE_GROUPS.items():
                    mask = np.isin(pt_arr, pts)
                    if not mask.any():
                        continue
                    fn = _fam_norm_src.get(fam, {})
                    fmean = fn.get("mean", _global_mean)
                    fstd  = max(fn.get("std", _global_std), 1e-6)
                    grades[mask] = 100.0 + (fmean - e_rv[mask]) / fstd * 10.0
                unknown = np.isnan(grades) & np.isfinite(e_rv)
                if unknown.any():
                    grades[unknown] = 100.0 + (_global_mean - e_rv[unknown]) / _global_std * 10.0
                return grades
            else:
                return 100.0 + (_global_mean - e_rv) / _global_std * 10.0

        # count not used as feature (regressed out of target during training)

        # Score vs same-hand and opposite-hand batters independently
        df_sh = df.copy(); df_sh["same_hand"] = 1
        df_op = df.copy(); df_op["same_hand"] = 0

        sp_same = _normalize(_score_df(df_sh), df["pitch_type"])
        sp_opp  = _normalize(_score_df(df_op), df["pitch_type"])

        # Map same/opposite → vs_rhb/vs_lhb based on pitcher handedness
        p_throws_r = df["p_throws_r"].values if "p_throws_r" in df.columns else np.ones(len(df))
        vs_rhb     = np.where(p_throws_r == 1, sp_same, sp_opp)
        vs_lhb     = np.where(p_throws_r == 1, sp_opp,  sp_same)
        stuff_plus = (vs_rhb + vs_lhb) / 2.0

        low_velo   = (df["release_speed"] < 75.0).values
        power_mask = df["pitch_type"].isin(_POWER_TYPES).values
        null_mask  = power_mask & low_velo
        stuff_plus[null_mask] = np.nan
        vs_rhb[null_mask]     = np.nan
        vs_lhb[null_mask]     = np.nan

        df["stuff_plus"]        = np.clip(stuff_plus, -50.0, 200.0)
        df["stuff_plus_vs_rhb"] = np.clip(vs_rhb,     -50.0, 200.0)
        df["stuff_plus_vs_lhb"] = np.clip(vs_lhb,     -50.0, 200.0)
        return df


def load_baselines():
    path = os.path.join(MODEL_DIR, "movement_baselines.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No baselines at {path}. Run train first.")
    with open(path, "rb") as f:
        return pickle.load(f)
