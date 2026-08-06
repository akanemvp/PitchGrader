"""Unit tests for the model core (model/prob_resid.py).

These cover the pure, deterministic pieces of the fastball / non-fastball split model —
the Magnus/non-Magnus shape derivation (and its spin_axis dependency), the SIERA
GB/FB/PU labelling (with line drives excluded), the outcome run values, and FB/NONFB
group routing — so they run in a second without the trained artifacts or feature cache.
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


def test_add_shape_features_magnus_finite():
    """A valid pitch with a spin axis gets all 10 shape features, and the Magnus split
    lands the four-seam's lift in mag_vert (positive ride)."""
    df = _raw_ff()
    pr.add_shape_features(df)
    assert np.isfinite(df[pr.SHAPE_FEATS].values).all()
    assert df["mag_vert"].iloc[0] > 0        # a riding four-seam has positive Magnus vertical


def test_add_shape_features_requires_spin_axis():
    """No 3D spin axis -> the Magnus/non-Magnus features are NaN, so the pitch is
    unscorable (a card can't be made until Statcast fills spin_axis in)."""
    df = _raw_ff(spin_axis=np.nan)
    pr.add_shape_features(df)
    for c in ("mag_vert", "mag_horiz_arm", "nonmag_vert", "nonmag_horiz_arm"):
        assert pd.isna(df[c].iloc[0])


def test_add_shape_features_idempotent():
    df = _raw_ff()
    pr.add_shape_features(df); once = df["mag_vert"].iloc[0]
    pr.add_shape_features(df)
    assert df["mag_vert"].iloc[0] == once


def test_release_side_is_arm_normalized():
    """release_pos_x_arm negates for a RHP and keeps sign for a LHP, so a mirror-image
    lefty and righty land on the same arm side."""
    rhp = _raw_ff(p_throws="R", release_pos_x=1.8)
    lhp = _raw_ff(p_throws="L", release_pos_x=-1.8)
    pr.add_shape_features(rhp); pr.add_shape_features(lhp)
    assert rhp["release_pos_x_arm"].iloc[0] == pytest.approx(lhp["release_pos_x_arm"].iloc[0])


def test_grid_cell_siera_labels():
    """SIERA contact cells: 0 ground ball (<10 deg), 1 fly ball (25-50 deg), 2 pop-up (>=50 deg)."""
    df = pd.DataFrame({"launch_angle": [-5.0, 5.0, 30.0, 45.0, 60.0]})
    assert pr._grid_cell(df).tolist() == [0.0, 0.0, 1.0, 1.0, 2.0]


def test_grid_cell_excludes_line_drives():
    """Line drives (10-25 deg) are left NaN — line-drive rate is batter/luck, not a pitcher
    skill, so they are excluded from the contact head entirely."""
    df = pd.DataFrame({"launch_angle": [10.0, 15.0, 24.0]})
    assert pr._grid_cell(df).isna().all()


def test_bucket_values_signs():
    """Whiffs and fouls both cost the batter run value; a whiff costs more than a foul."""
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


def test_assign_group_fb_vs_nonfb():
    """Fastballs -> FB; breaking/offspeed -> NONFB; unrouted cutters default to NONFB."""
    df = pd.DataFrame({"pitch_type": ["FF", "SI", "SL", "CH", "CU", "FC"]})
    grp = pr._assign_group(df, router=None)
    assert grp.tolist() == ["FB", "FB", "NONFB", "NONFB", "NONFB", "NONFB"]


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
    assert len(pr.predict_group_rv(empty, {"groups": {}})) == 0
