"""Unit tests for the model core (model/prob_resid.py).

These cover the pure, deterministic pieces of the three-group (FB / BR / OFF) model — the
induced-movement shape derivation, arm-side normalization, the GB/air contact labelling,
the outcome run values, and family routing — so they run in a second without the trained
artifacts or feature cache.
"""
import numpy as np
import pandas as pd
import pytest

import model.prob_resid as pr


def _raw_ff(**over):
    """A single physically-valid raw four-seam pitch (RHP), pre-shape-features."""
    row = dict(p_throws="R", pitch_type="FF", release_speed=95.0, release_spin_rate=2300.0,
               arm_angle=48.0, release_extension=6.5, release_pos_x=-1.8, release_pos_z=5.9,
               vx0=6.0, vy0=-138.0, vz0=-4.5, ax=-11.0, ay=27.0, az=-13.0, spin_axis=205.0)
    row.update(over)
    return pd.DataFrame([row])


def test_add_shape_features_all_finite():
    """A valid pitch gets all 8 shape features, and the induced split lands a riding
    four-seam's lift as positive ind_vert."""
    df = _raw_ff()
    pr.add_shape_features(df)
    assert len(pr.SHAPE_FEATS) == 8
    assert np.isfinite(df[pr.SHAPE_FEATS].values).all()
    assert df["ind_vert"].iloc[0] > 0        # a riding four-seam has positive induced vertical


def test_scores_from_kinematics_without_spin_axis():
    """The induced features come from the pitch's trajectory (vx0/ax/…), not spin_axis, so a
    pitch with no spin_axis still gets finite shape features (minor-league feeds score)."""
    df = _raw_ff(spin_axis=np.nan)
    pr.add_shape_features(df)
    assert np.isfinite(df[pr.SHAPE_FEATS].values).all()


def test_add_shape_features_idempotent():
    df = _raw_ff()
    pr.add_shape_features(df); once = df["ind_vert"].iloc[0]
    pr.add_shape_features(df)
    assert df["ind_vert"].iloc[0] == once


def test_release_side_is_arm_normalized():
    """release_pos_x_arm negates for a RHP and keeps sign for a LHP, so a mirror-image
    lefty and righty land on the same arm side."""
    rhp = _raw_ff(p_throws="R", release_pos_x=1.8)
    lhp = _raw_ff(p_throws="L", release_pos_x=-1.8)
    pr.add_shape_features(rhp); pr.add_shape_features(lhp)
    assert rhp["release_pos_x_arm"].iloc[0] == pytest.approx(lhp["release_pos_x_arm"].iloc[0])


def test_induced_horizontal_is_arm_normalized():
    """ind_horiz_arm is arm-signed: a full mirror-image lefty and righty (all x-components
    reflected) land on the same arm-side run."""
    rhp = _raw_ff(p_throws="R", vx0=6.0, ax=-11.0, release_pos_x=-1.8)
    lhp = _raw_ff(p_throws="L", vx0=-6.0, ax=11.0, release_pos_x=1.8)   # reflect vx0, ax, release side
    pr.add_shape_features(rhp); pr.add_shape_features(lhp)
    assert rhp["ind_horiz_arm"].iloc[0] == pytest.approx(lhp["ind_horiz_arm"].iloc[0])


def test_grid_cell_gb_vs_air():
    """Binary contact cells: 0 ground ball (<10 deg), 1 air (>=10 deg, any non-GB contact)."""
    df = pd.DataFrame({"launch_angle": [-5.0, 5.0, 15.0, 30.0, 60.0]})
    assert pr._grid_cell(df).tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_grid_cell_nan_without_launch_angle():
    df = pd.DataFrame({"launch_angle": [np.nan]})
    assert pr._grid_cell(df).isna().all()


def test_dre_values_signs():
    """Whiffs, fouls, and grounders cost the batter run value; air balls add it. A whiff
    costs more than a foul, and an air ball is worse (for the pitcher) than a grounder."""
    n = 3000
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "description": (["swinging_strike"] * n + ["foul"] * n + ["hit_into_play"] * n
                        + ["hit_into_play"] * n),
        "delta_run_exp": np.r_[
            rng.normal(-0.11, 0.01, n), rng.normal(-0.04, 0.01, n),
            rng.normal(-0.05, 0.01, n), rng.normal(0.13, 0.01, n)],
        "launch_angle": np.r_[
            np.full(n, np.nan), np.full(n, np.nan), np.full(n, 3.0), np.full(n, 30.0)],
    })
    lab = pr._outcome_label(df)
    V = pr._dre_values(df, lab)
    assert V["whiff"] < V["foul"] < 0.0
    assert V["gb"] < 0.0 < V["air"]


def test_bucket_values_signs():
    """Count-adjusted swing values: a whiff costs the batter more than a foul, both < 0."""
    n = 4000
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "description": ["swinging_strike"] * n + ["foul"] * n + ["ball"] * n,
        "delta_run_exp": np.r_[
            rng.normal(-0.10, 0.01, n), rng.normal(-0.04, 0.01, n), rng.normal(0.03, 0.01, n)],
        "balls": rng.randint(0, 4, 3 * n),
        "strikes": rng.randint(0, 3, 3 * n),
    })
    V = pr.bucket_values(df)
    assert V["whiff"] < V["foul"] < 0.0


def test_basefam_mapping():
    assert pr._basefam("FF") == "FB"
    assert pr._basefam("SI") == "FB"
    assert pr._basefam("SL") == "BR"
    assert pr._basefam("CH") == "OFF"
    assert pr._basefam("FC") is None    # cutters routed separately


def test_assign_group_three_families():
    """Fastballs -> FB; breaking -> BR; offspeed -> OFF; unrouted cutters default to BR."""
    df = pd.DataFrame({"pitch_type": ["FF", "SI", "SL", "CU", "CH", "FS", "FC"]})
    grp = pr._assign_group(df, router=None)
    assert grp.tolist() == ["FB", "FB", "BR", "BR", "OFF", "OFF", "BR"]


def test_fit_spin_offset_structure(tmp_path, monkeypatch):
    """fit_spin_offset returns a per-hand offset dict and never crashes on thin data."""
    monkeypatch.setattr(pr, "_SPIN_OFFSET_PATH", str(tmp_path / "off.pkl"))
    monkeypatch.setattr(pr, "_SPIN_OFFSET", None)
    df = pd.concat([_raw_ff()] * 150, ignore_index=True)
    off = pr.fit_spin_offset(df)
    assert set(off) == {"R", "L"}
    assert np.isfinite(off["R"]) and np.isfinite(off["L"])


def test_predict_group_rv_empty_frame():
    empty = pd.DataFrame({c: [] for c in pr.SHAPE_FEATS + ["pitch_type"]})
    assert len(pr.predict_group_rv(empty, {"groups": {}, "values": {}})) == 0
